# listener.py
"""
Boucle d'inference temps reel : consomme les mesures recues via MQTT
(mqtt_ingest.py), nettoie + extrait les features (features.py), predit
avec le modele Isolation Forest, et POST le resultat vers l'API locale.

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
LEAK_THRESHOLD = 0.70
ANOMALY_WINDOW = 10  # nb de dernieres mesures utilisees pour lisser anomaly_ratio
SOURCE_LABEL = "esp32_live"


class Listener:
    def __init__(self, run_id=None):
        self.run_id = run_id
        self.running = False
        self.mqtt = MqttListener()
        self.model = joblib.load("models/isolation_forest.pkl")
        self.scaler = joblib.load("models/scaler.pkl")
        self.recent_preds = deque(maxlen=ANOMALY_WINDOW)
        self.start_time = None
        self._thread = None

    def run(self):
        self.running = True
        self.start_time = time.time()
        self.mqtt.start()
        print(f"Ecoute MQTT demarree (run_id={self.run_id})")
        try:
            while self.running:
                try:
                    signal = self.mqtt.get(timeout=1.0)
                except queue.Empty:
                    continue
                self._process(signal)
        finally:
            self.mqtt.stop()
            print("Ecoute MQTT arretee")

    def _process(self, signal: np.ndarray):
        try:
            clean = features.clean_signal(signal)
            if len(clean) < 10:
                return

            x = features.extract_feature_vector(clean).reshape(1, -1)
            xs = self.scaler.transform(x)
            pred = self.model.predict(xs)[0]  # -1 = anomalie, 1 = normal

            self.recent_preds.append(1 if pred == -1 else 0)
            anomaly_ratio = float(np.mean(self.recent_preds))
            leak = anomaly_ratio >= LEAK_THRESHOLD

            t = time.time() - self.start_time
            status = "FUITE DETECTEE" if leak else "normal"
            print(f"[t={t:6.1f}s] anomalies={anomaly_ratio:4.0%} | {status}")

            requests.post(f"{API_URL}/results", json={
                "t": t,
                "anomaly_ratio": anomaly_ratio,
                "leak_detected": bool(leak),
                "scenario": SOURCE_LABEL,
                "run_id": self.run_id,
            }, timeout=5)
        except Exception as e:
            print(f"Erreur de traitement d'une mesure: {e}")

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
