# Waterleak Simulation

Detection de fuites en temps reel sur un reseau de canalisations via Isolation Forest.

## Dataset

Source : Aghashahi et al. (2022) - Dataset of Leak Simulations in Experimental Testbed Water Distribution System
DOI : https://doi.org/10.17632/tbrnp6vrnj.1
Licence : CC BY 4.0

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
python main.py
```

Lance le scenario `ideal` par defaut. Telecharge automatiquement les donnees depuis HuggingFace, demarre l'API sur `http://localhost:8000` et lance la simulation.

### Choisir un scenario en ligne de commande

```bash
python main.py --scenario sain
python main.py --scenario ideal
python main.py --scenario debit_faible
python main.py --scenario debit_fort
python main.py --scenario transitoire
```

### Lancer un scenario depuis le dashboard

Demarrer l'API, puis utiliser le controle Scenario du dashboard. Le dashboard appelle `POST /simulation/start` via le proxy `/waterleak-api`.

Si l'API n'est pas sur le port 8000, definir `WATERLEAK_API_URL`, par exemple :

```bash
$env:WATERLEAK_API_URL="http://127.0.0.1:8010"
```

## Endpoints API

| Endpoint | Description |
|----------|-------------|
| `GET /results` | Tous les resultats |
| `GET /results/latest` | Dernier resultat |
| `DELETE /results` | Reinitialise les resultats |
| `GET /status` | Etat de la simulation |
| `GET /scenarios` | Liste des scenarios disponibles |
| `POST /simulation/start` | Lance un scenario (`{"scenario":"ideal","reset":true}`) |

## Structure

```text
models/          <- modele Isolation Forest + scaler
features.py      <- extraction de features
simulation.py    <- simulation temps reel
api.py           <- FastAPI
ingest.py        <- telechargement des donnees
main.py          <- point d'entree
```
