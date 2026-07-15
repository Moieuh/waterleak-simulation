# mqtt_ingest.py
"""
Ecoute du topic MQTT sur lequel l'ESP32 publie ses mesures, en remplacement
du telechargement de dataset (ancien ingest.py).

Format observe en live (topic aquaoptim/esp32_001/data_json, via MQTT Explorer) :
{
    "project": "AquaOptim",
    "type": "signal_chunk",
    "chunk_index": 2393,
    "sample_rate_hz": 20000,
    "adc_pin": "GPIO34",
    "stats": {...},
    "samples": [1847, 1844, ...]   <- ~200 echantillons par message, pas 20000 !
}

Chaque message ne contient qu'un petit paquet (~200 echantillons, ~10ms) d'un flux
continu, pas une mesure complete comme les captures d'1s/20000 echantillons utilisees
a l'entrainement. L'accumulation en fenetres de taille comparable a l'entrainement se
fait dans listener.py, pas ici : ce module se contente de livrer les paquets bruts,
non centres, dans l'ordre (chunk_index sert a listener.py pour detecter les trous).
"""
import json
import os
import queue
import threading

import numpy as np
import paho.mqtt.client as mqtt

MQTT_HOST = os.environ.get("WATERLEAK_MQTT_HOST", "test.mosquitto.org")
MQTT_PORT = int(os.environ.get("WATERLEAK_MQTT_PORT", "1883"))
MQTT_TOPIC = os.environ.get("WATERLEAK_MQTT_TOPIC", "aquaoptim/esp32_001/data_json")
MQTT_USER = os.environ.get("WATERLEAK_MQTT_USER")
MQTT_PASSWORD = os.environ.get("WATERLEAK_MQTT_PASSWORD")


def _decode(payload: bytes):
    """Parse un message MQTT brut. Retourne (echantillons_bruts, data_brut).
    Pas de centrage ici : un paquet de ~200 echantillons est trop court pour
    calculer une moyenne representative. Le centrage se fait dans listener.py,
    une fois plusieurs paquets accumules en une fenetre complete."""
    data = json.loads(payload)
    samples = np.array(data["samples"], dtype=np.float64)
    return samples, data


def parse_payload(payload: bytes) -> np.ndarray:
    """Utilitaire de test/inspection : parse UN SEUL message MQTT et retourne
    son signal centre (signal - moyenne de ce paquet uniquement). Le pipeline
    live n'utilise pas cette fonction pour l'inference (voir listener.py, qui
    accumule plusieurs paquets avant de centrer)."""
    samples, _ = _decode(payload)
    return samples - np.mean(samples)


class MqttListener:
    """
    Se connecte au broker, s'abonne a MQTT_TOPIC, et pousse chaque paquet recu
    dans une queue consommee par listener.py.

    Chaque element pousse dans la queue est un dict :
    {"samples": <np.ndarray brut, non centre>, "chunk_index": ..., "sample_rate_hz": ...}

    La connexion est geree en arriere-plan (connect_async + loop_start) avec
    retries automatiques : si le broker est injoignable au demarrage ou se
    deconnecte, le client reessaie tout seul au lieu de planter.

    Les messages "retained" (rejoues automatiquement par le broker a chaque
    (re)connexion/abonnement, meme si l'ESP32 n'envoie plus rien) sont ignores,
    tout comme les doublons de chunk_index : sans ca, une coupure d'envoi cote
    ESP32 faisait rejouer en boucle la derniere mesure (donc la derniere fuite
    detectee) a chaque tentative de reconnexion.
    """

    def __init__(self, host=MQTT_HOST, port=MQTT_PORT, topic=MQTT_TOPIC,
                 username=MQTT_USER, password=MQTT_PASSWORD, verbose=True):
        self.host = host
        self.port = port
        self.topic = topic
        self.verbose = verbose
        self.data_queue = queue.Queue()
        self.connected = threading.Event()
        self._last_chunk_index = None
        self.n_received = 0

        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:
            # compat avec les anciennes versions de paho-mqtt (<2.0)
            self.client = mqtt.Client()

        if username:
            self.client.username_pw_set(username, password)

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        if hasattr(self.client, "on_connect_fail"):
            self.client.on_connect_fail = self._on_connect_fail

        # backoff automatique entre les tentatives de (re)connexion
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            client.subscribe(self.topic)
            self.connected.set()
            print(f"MQTT: connecte a {self.host}:{self.port}, abonne a '{self.topic}'")
        else:
            print(f"MQTT: echec de connexion (code {reason_code})")

    def _on_disconnect(self, client, userdata, *args):
        self.connected.clear()
        print("MQTT: deconnecte, nouvelle tentative en arriere-plan...")

    def _on_connect_fail(self, client, userdata):
        print(f"MQTT: impossible de joindre {self.host}:{self.port}, nouvelle tentative...")

    def _on_message(self, client, userdata, msg):
        if getattr(msg, "retain", False):
            # message rejoue par le broker (dernier connu sur le topic), pas une
            # nouvelle mesure en direct -> on l'ignore
            return
        try:
            samples, data = _decode(msg.payload)
            chunk_index = data.get("chunk_index")
            if chunk_index is not None and chunk_index == self._last_chunk_index:
                return  # meme paquet deja traite, evite les doublons
            self._last_chunk_index = chunk_index
            self.n_received += 1

            if self.verbose:
                stats = data.get("stats", {})
                print(
                    f"MQTT #{self.n_received} recu sur '{msg.topic}' | "
                    f"chunk_index={chunk_index} | "
                    f"sample_rate_hz={data.get('sample_rate_hz')} | "
                    f"{len(samples)} echantillons | "
                    f"premiers bruts={data.get('samples', [])[:5]} | "
                    f"raw_mean={stats.get('raw_mean')} raw_std={stats.get('raw_std')} "
                    f"raw_peak_to_peak={stats.get('raw_peak_to_peak')}"
                )

            self.data_queue.put({
                "samples": samples,
                "chunk_index": chunk_index,
                "sample_rate_hz": data.get("sample_rate_hz"),
            })
        except Exception as e:
            print(f"MQTT: message ignore ({e})")

    def start(self):
        # connect_async + loop_start : la (re)connexion est geree en arriere-plan
        # avec retries automatiques, meme si le broker est injoignable au demarrage.
        self.client.connect_async(self.host, self.port, keepalive=30)
        self.client.loop_start()

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()

    def get(self, timeout=None):
        """Recupere le prochain paquet (bloquant, leve queue.Empty si timeout).
        Retourne un dict {samples, chunk_index, sample_rate_hz}."""
        return self.data_queue.get(timeout=timeout)
