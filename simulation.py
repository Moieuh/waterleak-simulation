# simulation.py
import time, threading
import numpy as np, pandas as pd, joblib
from pathlib import Path
from scipy import signal
import requests

# ---------- CONFIG ----------
FS = 25600
WINDOW_SEC = 1.0
STEP_RATIO = 0.25
CHUNK_SEC = 10          # on traite les 10 dernières secondes
TICK_SEC = 30           # calcul toutes les 30s fixes
LEAK_THRESHOLD = 0.70
FREQ_BANDS = [(0,1500),(1500,3000),(3000,5000),(5000,9000),(9000,12800)]


PARQUET_DIR = Path("data/parquet/Accelerometer/Branched")
NL_FILES = [

    "No-leak/BR_NL_ND_A1.parquet",
    "No-leak/BR_NL_Transient_A1.parquet",
    "No-leak/BR_NL_0.47 LPS_A1.parquet",
    "No-leak/BR_NL_0.18 LPS_A1.parquet",
]
LEAK_FILE = "Circumferential Crack/BR_CC_0.18 LPS_A1.parquet"

# ---------- FEATURES ----------
def extract_features(window, fs):
    rms = np.sqrt(np.mean(window**2))
    var = np.var(window)
    peak = np.max(np.abs(window))
    crest = peak/rms if rms>0 else 0
    kurt = pd.Series(window).kurtosis()
    freqs, psd = signal.welch(window, fs=fs, nperseg=min(1024,len(window)))
    bands = [np.sum(psd[np.where((freqs>=a)&(freqs<b))]) for a,b in FREQ_BANDS]
    return [rms,var,crest,kurt]+bands

def features_from_chunk(chunk, fs):
    w = int(WINDOW_SEC*fs); step = int(w*STEP_RATIO)
    rows = []
    i = 0
    while i+w <= len(chunk):
        rows.append(extract_features(chunk[i:i+w], fs))
        i += step
    return np.array(rows)

# ---------- BUILD SIGNAL ----------
def load(path): return pd.read_parquet(PARQUET_DIR/path)["Value"].values

def build_signal():
    parts = [load(f) for f in NL_FILES] + [load(LEAK_FILE), load(LEAK_FILE)]
    sig = np.concatenate(parts)
    leak_start = sum(len(load(f)) for f in NL_FILES)
    return sig, leak_start

# ---------- SIMULATION ----------
class Sim:
    def __init__(self, sig, leak_start, model, scaler):
        self.sig = sig; self.leak_start = leak_start
        self.model = model; self.scaler = scaler
        self.pos = 0; self.running = True

    def stream(self):
        # 1 "tick réel" = 1s simulée, accéléré ici
        while self.running and self.pos < len(self.sig):
            self.pos += FS  # +1s de points
            time.sleep(0.05)  # accélère la démo (1s simulée = 50ms réelles)

    def analyze_loop(self):
        while self.running and self.pos < len(self.sig):
            time.sleep(TICK_SEC*0.05)  # même accélération
            end = self.pos
            start = max(0, end - CHUNK_SEC*FS)
            chunk = self.sig[start:end]
            if len(chunk) < FS: continue
            X = features_from_chunk(chunk, FS)
            if len(X)==0: continue
            Xs = self.scaler.transform(X)
            preds = self.model.predict(Xs)        # -1 anomalie, 1 normal
            anomaly_ratio = np.mean(preds==-1)
            leak = anomaly_ratio >= LEAK_THRESHOLD
            t = end/FS
            zone = "FUITE" if start >= self.leak_start else "SAIN"
            print(f"[t={t:5.0f}s] zone réelle={zone:5} | anomalies={anomaly_ratio:4.0%} | "
                  f"{'🚨 FUITE DÉTECTÉE' if leak else '✅ normal'}")
            try:
                requests.post("http://localhost:8000/results", json={
                    "t": t,
                    "anomaly_ratio": float(anomaly_ratio),
                    "leak_detected": bool(leak)
                })
            except Exception:
                pass          
        self.running = False

def main():
    model = joblib.load("models/isolation_forest.pkl")
    scaler = joblib.load("models/scaler.pkl")
    sig, leak_start = build_signal()
    print(f"Signal: {len(sig)/FS:.0f}s | fuite injectée à t={leak_start/FS:.0f}s\n")
    sim = Sim(sig, leak_start, model, scaler)
    t1 = threading.Thread(target=sim.stream)
    t2 = threading.Thread(target=sim.analyze_loop)
    t1.start(); t2.start()
    t1.join(); t2.join()
    print("\nSimulation terminée.")

if __name__ == "__main__":
    main()