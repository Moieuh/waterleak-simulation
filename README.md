# Waterleak Simulation

Detection de fuites en temps reel sur un reseau de canalisations via Isolation Forest.

Les mesures viennent en direct d'un ESP32 qui publie sur un broker MQTT (plus de dataset telecharge / rejoue).

## Source des donnees

- Broker : `test.mosquitto.org` (public, port 1883)
- Topic : `aquaoptim/esp32_001/data_json`
- Format des messages (observe via MQTT Explorer sur le topic ci-dessus) :

```json
{
  "project": "AquaOptim",
  "type": "signal_chunk",
  "chunk_index": 2393,
  "sample_rate_hz": 20000,
  "adc_pin": "GPIO34",
  "adc_min": 0,
  "adc_max": 4095,
  "stats": {"raw_mean": 1844.43, "raw_std": 3.16, "...": "..."},
  "samples": [1847, 1844, 1846, "..."]
}
```

Seule la cle `samples` est utilisee (`mqtt_ingest.parse_payload`). `stats` est ignore : les
features sont recalculees a partir de `samples` pour rester identiques au notebook
d'entrainement.

## Modele

`models/isolation_forest.pkl` et `models/scaler.pkl` sont le modele entraine par l'equipe
(cf. `pipeline.ipynb`) sur des mesures pompe_allumee / pompe_avec_fuite : nettoyage des zones
parasites, puis 11 features (RMS, variance, crest factor, kurtosis, peak-to-peak + 6 bandes
d'energie 0-10000 Hz, fs=20000 Hz). `features.py` reprend ces memes fonctions pour l'inference
en direct.

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
python main.py
```

Demarre l'API sur `http://localhost:8000` et l'ecoute MQTT en continu.

Variables d'environnement disponibles :

| Variable | Defaut | Description |
|---|---|---|
| `WATERLEAK_MQTT_HOST` | `test.mosquitto.org` | Broker MQTT |
| `WATERLEAK_MQTT_PORT` | `1883` | Port du broker |
| `WATERLEAK_MQTT_TOPIC` | `aquaoptim/esp32_001/data_json` | Topic ecoute |
| `WATERLEAK_MQTT_USER` / `WATERLEAK_MQTT_PASSWORD` | - | Credentials si besoin |
| `WATERLEAK_API_URL` | `http://localhost:8000` | URL de l'API pour les POST /results |
| `WATERLEAK_DB_PATH` | `data/results.db` | Base sqlite des resultats |

### Piloter l'ecoute depuis le dashboard

Demarrer l'API, puis utiliser `POST /simulation/start` / `POST /simulation/stop` (proxy `/waterleak-api`
cote dashboard). Le POST des resultats, l'historique et le format des reponses ne changent pas.

## Endpoints API

| Endpoint | Description |
|----------|-------------|
| `GET /results` | Resultats de l'ecoute en cours |
| `GET /results/latest` | Dernier resultat |
| `GET /results/history` | Historique complet (ecoutes archivees + en cours) |
| `DELETE /results` | Reinitialise les resultats |
| `GET /status` | Etat de l'ecoute (running, leak_detected, run_id...) |
| `POST /simulation/start` | Demarre l'ecoute MQTT (archive les resultats precedents) |
| `POST /simulation/stop` | Arrete l'ecoute MQTT en cours |

## Structure

```text
models/          <- modele Isolation Forest + scaler (fournis par l'equipe, cf. pipeline.ipynb)
features.py      <- nettoyage de signal + extraction de features (identique au notebook)
mqtt_ingest.py    <- client MQTT, parsing des messages ESP32
listener.py       <- boucle d'inference temps reel (predict + POST /results)
api.py            <- FastAPI (resultats, historique, start/stop de l'ecoute)
main.py           <- point d'entree (API + ecoute MQTT)
pipeline.ipynb    <- notebook d'entrainement du modele (reference)
```
