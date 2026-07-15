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
    "samples": [1847, 1844, ...]
}
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
    """Parse un message MQTT brut. Retourne (signal_centre, data_brut)."""
    data = json.loads(payload)
    samples = np.array(data["samples"], dtype=np.float64)
    signal = samples - np.mean(samples)
    return signal, data


def parse_payload(payload: bytes) -> np.ndarray:
    """Parse un message MQTT et retourne le signal centre (signal - moyenne),
    comme le fait charger_signaux() dans le notebook."""
    signal, _ = _decode(payload)
    return signal


class MqttListener:
    """
    Se connecte au broker, s'abonne a MQTT_TOPIC, et pousse chaque mesure
    recue dans une queue consommee par listener.py.

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
                 username=MQTT_USER, password=MQTT_PASSWORD):
        self.host = host
        self.port = port
        self.topic = topic
        self.data_queue = queue.Queue()
        self.connected = threading.Event()
        self._last_chunk_index = None

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
            signal, data = _decode(msg.payload)
            chunk_index = data.get("chunk_index")
            if chunk_index is not None and chunk_index == self._last_chunk_index:
                return  # meme mesure deja traitee, evite les doublons
            self._last_chunk_index = chunk_index
            self.data_queue.put(signal)
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
        """Recupere la prochaine mesure (bloquant, leve queue.Empty si timeout)."""
        return self.data_queue.get(timeout=timeout)
