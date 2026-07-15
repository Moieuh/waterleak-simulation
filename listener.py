# listener.py
"""
Boucle d'inference temps reel : consomme les paquets recus via MQTT
(mqtt_ingest.py), les accumule en fenetres completes comparables aux mesures
d'entrainement, nettoie + extrait les features (features.py), predit avec le
modele Isolation Forest, et POST le resultat vers l'API locale.

L'ESP32 n'envoie pas une mesure complete par message MQTT : chaque message ne
contient que ~200 echantillons (~10ms), alors que le modele a ete entraine sur
des captures d'1 seconde (20000 echantillons, cf. pipeline.ipynb). On accumule
donc plusieurs paquets consecutifs (via chunk_index) jusqu'a obtenir une fenetre
de la meme taille que l'entrainement avant de lancer l'inference — sinon les
features (nettoyage, bandes de frequence) sont calculees sur un signal bien
trop court et ne ressemblent a rien de ce que le modele a appris.

Remplace l'ancien simulation.py : plus de scenarios, plus de replay de
fichiers parquet telecharges — les donnees viennent en direct de l'ESP32.
"""
import os
import queue
import threading
import time
from collections import deque

import joblib
import numpy as np
import requests

import features
from mqtt_ingest import MqttListener

API_URL = os.environ.get("WATERLEAK_API_URL", "http://localhost:8000")

# Taille de la fenetre d'analyse : par defaut features.FS (20000 echantillons,
# soit 1s a 20000Hz), pour matcher exactement la duree des captures utilisees
# a l'entrainement (cf. pipeline.ipynb : capture_seconds=1, n_samples=20000).
WINDOW_SAMPLES = int(os.environ.get("WATERLEAK_WINDOW_SAMPLES", features.FS))

# Seuil de decision + fenetre de lissage, remontes pour reduire les fausses
# alertes le temps que le modele soit ameliore (cf. matrice de confusion :
# 50% de faux positifs sur les mesures "normal" en test chez Nathan). Ce
# reglage attenue le symptome, il ne corrige pas le modele lui-meme. Chaque
# "mesure" lissee ici correspond maintenant a une vraie fenetre de ~1s (et non
# plus a un paquet MQTT de 10ms), donc ANOMALY_WINDOW=15 lisse sur ~15s reelles.
#
# ANOMALY_WINDOW=15 a ete choisi apres test empirique (rejeu des 60 captures
# labellisees avec_fuite/sans_fuite, cf. simulation.zip) compare a 20 et 10 :
# 15 reduit le retard de detection apres un changement d'etat (7-4 fichiers
# contre 9-8 a 20) SANS augmenter le nombre d'erreurs de statut en regime
# stable (5-3 contre 7-4 a 20) ; a 10, le retard baisse encore mais le
# clignotement remonte au niveau de 20 (fenetre trop courte face au bruit
# du modele, ~43% de faux positifs sur "normal").
LEAK_THRESHOLD = float(os.environ.get("WATERLEAK_LEAK_THRESHOLD", "0.70"))
ANOMALY_WINDOW = int(os.environ.get("WATERLEAK_ANOMALY_WINDOW", "15"))
SOURCE_LABEL = "esp32_live"


def _check_model_compatibility(scaler, model):
    """Verifie que le scaler/modele charges attendent bien les features produites
    par features.py (memes noms, meme ordre). Si le modele a ete reentraine avec
    un jeu de features different, on prefere planter tout de suite avec un message
    clair plutot que de tourner en silence avec des predictions fausses."""
    expected = features.FEATURE_COLS
    scaler_cols = getattr(scaler, "feature_names_in_", None)
    if scaler_cols is not None and list(scaler_cols) != expected:
        raise RuntimeError(
            "Incompatibilite entre models/scaler.pkl et features.py.\n"
            f"  features.py produit : {expected}\n"
            f"  scaler.pkl attend    : {list(scaler_cols)}\n"
            "Si le jeu de features a change cote entrainement (nouvelles bandes, "
            "fs different, etc.), il faut mettre a jour features.py en consequence."
        )

    n_expected = len(expected)
    for name, obj in (("scaler.pkl", scaler), ("isolation_forest.pkl", model)):
        n_features = getattr(obj, "n_features_in_", None)
        if n_features is not None and n_features != n_expected:
            raise RuntimeError(
                f"Incompatibilite entre models/{name} et features.py : "
                f"{name} attend {n_features} features, features.py en produit {n_expected}."
            )

    print(f"Modele compatible : {n_expected} features ({', '.join(expected)})")


class Listener:
    def __init__(self, run_id=None):
        self.run_id = run_id
        self.running = False
        self.mqtt = MqttListener()
        self.model = joblib.load("models/isolation_forest.pkl")
        self.scaler = joblib.load("models/scaler.pkl")
        _check_model_compatibility(self.scaler, self.model)
        self.recent_preds = deque(maxlen=ANOMALY_WINDOW)
        self.start_time = None
        self._thread = None

        # Buffer d'accumulation des paquets MQTT en une fenetre complete.
        self._buffer_chunks = []
        self._buffer_samples = 0
        self._expected_chunk_index = None

        print(f"Fenetre d'analyse : {WINDOW_SAMPLES} echantillons "
              f"(~{WINDOW_SAMPLES/features.FS:.1f}s a {features.FS}Hz), "
              f"accumulee a partir des paquets MQTT")
        print(f"Seuil de fuite: {LEAK_THRESHOLD:.0%} sur les {ANOMALY_WINDOW} dernieres fenetres")

    def run(self):
        self.running = True
        self.start_time = time.time()
        self.mqtt.start()
        print(f"Ecoute MQTT demarree (run_id={self.run_id})")
        try:
            while self.running:
                try:
                    item = self.mqtt.get(timeout=1.0)
                except queue.Empty:
                    continue
                window = self._accumulate(item)
                if window is not None:
                    self._process(window)
        finally:
            self.mqtt.stop()
            print("Ecoute MQTT arretee")

    def _accumulate(self, item: dict):
        """Empile les paquets MQTT recus jusqu'a obtenir une fenetre de
        WINDOW_SAMPLES echantillons contigus. Retourne la fenetre (np.ndarray)
        des qu'elle est prete, sinon None. Si un trou est detecte dans
        chunk_index (paquet MQTT perdu), la fenetre en cours est abandonnee
        pour eviter de recoller des morceaux de signal non contigus."""
        samples = item.get("samples")
        chunk_index = item.get("chunk_index")

        if (self._expected_chunk_index is not None and chunk_index is not None
                and chunk_index != self._expected_chunk_index):
            print(f"MQTT: trou detecte (attendu chunk {self._expected_chunk_index}, "
                  f"recu {chunk_index}) -> fenetre en cours abandonnee")
            self._buffer_chunks = []
            self._buffer_samples = 0

        if samples is None or len(samples) == 0:
            return None

        self._buffer_chunks.append(samples)
        self._buffer_samples += len(samples)
        self._expected_chunk_index = (chunk_index + 1) if chunk_index is not None else None

        if self._buffer_samples < WINDOW_SAMPLES:
            return None

        full = np.concatenate(self._buffer_chunks)
        window = full[:WINDOW_SAMPLES]
        leftover = full[WINDOW_SAMPLES:]
        self._buffer_chunks = [leftover] if len(leftover) else []
        self._buffer_samples = len(leftover)
        return window

    def _process(self, window: np.ndarray):
        try:
            # Centrage sur la fenetre complete (et non paquet par paquet),
            # comme le fait charger_signaux() dans le notebook d'entrainement.
            signal = window - np.mean(window)

            clean = features.clean_signal(signal)
            if len(clean) < 10:
                return

            x = features.extract_feature_vector(clean).reshape(1, -1)
            xs = self.scaler.transform(x)
            pred = self.model.predict(xs)[0]  # -1 = anomalie, 1 = normal
            # Score continu (decision_function) en plus du predict binaire :
            # positif = plutot normal, negatif = plutot anomalie. Contrairement
            # a anomaly_ratio (lisse sur ANOMALY_WINDOW mesures), ce score
            # reagit immediatement fenetre par fenetre, sans le retard du
            # lissage. Expose en plus du badge binaire, pas a sa place.
            score = float(self.model.decision_function(xs)[0])

            self.recent_preds.append(1 if pred == -1 else 0)
            anomaly_ratio = float(np.mean(self.recent_preds))
            leak = anomaly_ratio >= LEAK_THRESHOLD

            t = time.time() - self.start_time
            status = "FUITE DETECTEE" if leak else "normal"
            print(f"[t={t:6.1f}s] anomalies={anomaly_ratio:4.0%} (seuil {LEAK_THRESHOLD:.0%}, "
                  f"fenetre {len(self.recent_preds)}/{ANOMALY_WINDOW}) | score={score:+.4f} | {status}")

            requests.post(f"{API_URL}/results", json={
                "t": t,
                "anomaly_ratio": anomaly_ratio,
                "anomaly_score": score,
                "leak_detected": bool(leak),
                "scenario": SOURCE_LABEL,
                "run_id": self.run_id,
            }, timeout=5)
        except Exception as e:
            print(f"Erreur de traitement d'une fenetre: {e}")

    def stop(self):
        self.running = False

    # --- pilotage depuis un thread (utilise par api.py) ---
    def start_in_thread(self):
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        return self._thread

    def is_alive(self):
        return bool(self._thread and self._thread.is_alive())


def main(run_id=None):
    Listener(run_id=run_id).run()


if __name__ == "__main__":
    main()
