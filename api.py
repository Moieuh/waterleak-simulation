# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import sqlite3
from pathlib import Path

app = FastAPI()

DB_PATH = Path("data/results.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute("""
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    t REAL,
    anomaly_ratio REAL,
    leak_detected INTEGER,
    timestamp TEXT
)
""")
conn.commit()

class Result(BaseModel):
    t: float
    anomaly_ratio: float
    leak_detected: bool

@app.post("/results")
def push_result(r: Result):
    conn.execute(
        "INSERT INTO results (t, anomaly_ratio, leak_detected, timestamp) VALUES (?, ?, ?, ?)",
        (r.t, r.anomaly_ratio, int(r.leak_detected), datetime.now().isoformat())
    )
    conn.commit()
    return {"ok": True}

@app.get("/results")
def get_results():
    rows = conn.execute("SELECT t, anomaly_ratio, leak_detected, timestamp FROM results ORDER BY id").fetchall()
    return [
        {"t": t, "anomaly_ratio": ar, "leak_detected": bool(ld), "timestamp": ts}
        for t, ar, ld, ts in rows
    ]

@app.get("/results/latest")
def get_latest():
    row = conn.execute("SELECT t, anomaly_ratio, leak_detected, timestamp FROM results ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return {"message": "Pas encore de résultats"}
    t, ar, ld, ts = row
    return {"t": t, "anomaly_ratio": ar, "leak_detected": bool(ld), "timestamp": ts}

@app.delete("/results")
def reset_results():
    conn.execute("DELETE FROM results")
    conn.commit()
    return {"ok": True}

@app.get("/status")
def get_status():
    row = conn.execute("SELECT leak_detected, timestamp FROM results ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return {"running": False, "leak_detected": False}
    ld, ts = row
    return {"running": True, "leak_detected": bool(ld), "last_update": ts}