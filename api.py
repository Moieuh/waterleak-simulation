# api.py
from datetime import datetime, timedelta
import os
from pathlib import Path
import sqlite3
import threading
import uuid
from typing import Optional

from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel

import simulation

app = FastAPI()

DB_PATH = Path(os.environ.get("WATERLEAK_DB_PATH", "data/results.db"))
RUNNING_TTL_SECONDS = 15
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute("""
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    t REAL,
    anomaly_ratio REAL,
    leak_detected INTEGER,
    scenario TEXT,
    run_id TEXT,
    timestamp TEXT
)
""")
conn.commit()
conn.execute("""
CREATE TABLE IF NOT EXISTS results_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    t REAL,
    anomaly_ratio REAL,
    leak_detected INTEGER,
    scenario TEXT,
    run_id TEXT,
    timestamp TEXT
)
""")
conn.commit()

columns = {row[1] for row in conn.execute("PRAGMA table_info(results)").fetchall()}
if "scenario" not in columns:
    conn.execute("ALTER TABLE results ADD COLUMN scenario TEXT")
    conn.commit()
if "run_id" not in columns:
    conn.execute("ALTER TABLE results ADD COLUMN run_id TEXT")
    conn.commit()

history_columns = {row[1] for row in conn.execute("PRAGMA table_info(results_history)").fetchall()}
if "scenario" not in history_columns:
    conn.execute("ALTER TABLE results_history ADD COLUMN scenario TEXT")
    conn.commit()
if "run_id" not in history_columns:
    conn.execute("ALTER TABLE results_history ADD COLUMN run_id TEXT")
    conn.commit()


class Result(BaseModel):
    t: float
    anomaly_ratio: float
    leak_detected: bool
    scenario: Optional[str] = None
    run_id: Optional[str] = None


class SimulationStart(BaseModel):
    scenario: str


simulation_lock = threading.Lock()
simulation_state = {
    "thread": None,
    "scenario": None,
    "run_id": None,
    "started_at": None,
}


def is_simulation_thread_running():
    thread = simulation_state["thread"]
    return bool(thread and thread.is_alive())


def get_live_run_id():
    if simulation_state["run_id"]:
        return simulation_state["run_id"]

    row = conn.execute(
        "SELECT run_id FROM results WHERE run_id IS NOT NULL ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None


def archive_current_results():
    conn.execute("""
        INSERT INTO results_history (t, anomaly_ratio, leak_detected, scenario, run_id, timestamp)
        SELECT t, anomaly_ratio, leak_detected, scenario, run_id, timestamp
        FROM results
    """)
    conn.execute("DELETE FROM results")
    conn.commit()


def run_simulation_thread(scenario: str, run_id: str):
    try:
        simulation.main(scenario, run_id)
    finally:
        with simulation_lock:
            if simulation_state["run_id"] == run_id:
                simulation_state["thread"] = None


@app.post("/results")
def push_result(r: Result):
    conn.execute(
        "INSERT INTO results (t, anomaly_ratio, leak_detected, scenario, run_id, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        (r.t, r.anomaly_ratio, int(r.leak_detected), r.scenario, r.run_id, datetime.now().isoformat()),
    )
    conn.commit()
    return {"ok": True}


@app.get("/results")
def get_results():
    live_run_id = get_live_run_id()
    if live_run_id:
        rows = conn.execute(
            "SELECT t, anomaly_ratio, leak_detected, scenario, run_id, timestamp FROM results WHERE run_id = ? ORDER BY id",
            (live_run_id,),
        ).fetchall()
    else:
        row = conn.execute(
            "SELECT scenario FROM results ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row:
            rows = conn.execute(
                "SELECT t, anomaly_ratio, leak_detected, scenario, run_id, timestamp FROM results WHERE scenario = ? ORDER BY id",
                (row[0],),
            ).fetchall()
        else:
            rows = []
    return [
        {
            "t": t,
            "anomaly_ratio": ar,
            "leak_detected": bool(ld),
            "scenario": scenario,
            "run_id": run_id,
            "timestamp": ts,
        }
        for t, ar, ld, scenario, run_id, ts in rows
    ]


@app.get("/results/history")
def get_results_history():
    rows = conn.execute("""
        SELECT t, anomaly_ratio, leak_detected, scenario, run_id, timestamp, 0 AS source_order, id
        FROM results_history
        UNION ALL
        SELECT t, anomaly_ratio, leak_detected, scenario, run_id, timestamp, 1 AS source_order, id
        FROM results
        ORDER BY timestamp, source_order, id
    """).fetchall()
    return [
        {
            "t": t,
            "anomaly_ratio": ar,
            "leak_detected": bool(ld),
            "scenario": scenario,
            "run_id": run_id,
            "timestamp": ts,
        }
        for t, ar, ld, scenario, run_id, ts, _source_order, _id in rows
    ]


@app.get("/results/latest")
def get_latest():
    live_run_id = get_live_run_id()
    if live_run_id:
        row = conn.execute(
            "SELECT t, anomaly_ratio, leak_detected, scenario, run_id, timestamp FROM results WHERE run_id = ? ORDER BY id DESC LIMIT 1",
            (live_run_id,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT t, anomaly_ratio, leak_detected, scenario, run_id, timestamp FROM results ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if not row:
        return {"message": "Pas encore de resultats"}
    t, ar, ld, scenario, run_id, ts = row
    return {
        "t": t,
        "anomaly_ratio": ar,
        "leak_detected": bool(ld),
        "scenario": scenario,
        "run_id": run_id,
        "timestamp": ts,
    }


@app.delete("/results")
def reset_results():
    conn.execute("DELETE FROM results")
    conn.execute("DELETE FROM results_history")
    conn.commit()
    with simulation_lock:
        if not is_simulation_thread_running():
            simulation_state["run_id"] = None
            simulation_state["scenario"] = None
            simulation_state["started_at"] = None
    return {"ok": True}


@app.get("/scenarios")
def get_scenarios():
    return [
        {
            "id": scenario,
            "label": scenario.replace("_", " ").title(),
            "description": config["desc"],
        }
        for scenario, config in simulation.SCENARIOS.items()
    ]


@app.post("/simulation/start")
def start_simulation(payload: SimulationStart):
    if payload.scenario not in simulation.SCENARIOS:
        raise HTTPException(status_code=400, detail="Scenario inconnu")

    with simulation_lock:
        if is_simulation_thread_running():
            return {
                "ok": False,
                "running": True,
                "scenario": simulation_state["scenario"],
                "message": "Une simulation est deja en cours",
            }

        archive_current_results()

        run_id = uuid.uuid4().hex
        thread = threading.Thread(
            target=run_simulation_thread,
            args=(payload.scenario, run_id),
            daemon=True,
        )
        simulation_state["thread"] = thread
        simulation_state["scenario"] = payload.scenario
        simulation_state["run_id"] = run_id
        simulation_state["started_at"] = datetime.now().isoformat()
        thread.start()

    return {
        "ok": True,
        "running": True,
        "scenario": payload.scenario,
        "run_id": simulation_state["run_id"],
        "started_at": simulation_state["started_at"],
    }


@app.get("/status")
def get_status():
    live_run_id = get_live_run_id()
    if live_run_id:
        row = conn.execute(
            "SELECT leak_detected, scenario, run_id, timestamp FROM results WHERE run_id = ? ORDER BY id DESC LIMIT 1",
            (live_run_id,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT leak_detected, scenario, run_id, timestamp FROM results ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if not row:
        return {
            "running": is_simulation_thread_running(),
            "leak_detected": False,
            "scenario": simulation_state["scenario"],
            "run_id": simulation_state["run_id"],
        }

    ld, scenario, run_id, ts = row
    try:
        last_update = datetime.fromisoformat(ts)
        running = is_simulation_thread_running() or datetime.now() - last_update <= timedelta(seconds=RUNNING_TTL_SECONDS)
    except (TypeError, ValueError):
        running = is_simulation_thread_running()

    return {
        "running": running,
        "leak_detected": bool(ld),
        "scenario": simulation_state["scenario"] if is_simulation_thread_running() else scenario,
        "run_id": run_id,
        "last_update": ts,
    }
