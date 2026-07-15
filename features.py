# features.py
"""
Nettoyage de signal + extraction de features.

Reprend telles quelles (verbatim) les fonctions du nouveau pipeline
(pipeline.ipynb, fourni avec le modele entraine par l'equipe) :
- detection/suppression des zones parasites (cellule 1 du notebook)
- extraction de features temporelles + frequentielles (cellule 2 du notebook)

fs=20000 Hz et BANDES_HZ correspondent aux parametres utilises pour
entrainer models/isolation_forest.pkl + models/scaler.pkl. Ne pas les
changer sans reentrainer le modele.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch
from scipy.ndimage import binary_closing, binary_dilation, label

# ---------- CONFIG (identique au notebook) ----------
FS = 20000
BANDES_HZ = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 7000), (7000, 10000)]
FEATURE_COLS = ["rms", "variance", "crest_factor", "kurtosis", "peak_to_peak"] + \
    [f"energie_{a}_{b}hz" for a, b in BANDES_HZ]

# Parametres de nettoyage utilises par l'ami pour entrainer le modele
# (cf. pipeline_complet(..., k=3, min_duration_ms=20, margin_ms=10) dans pipeline.ipynb)
CLEAN_K = 3
CLEAN_MIN_DURATION_MS = 20
CLEAN_MARGIN_MS = 10

# numpy >= 2.0 a renomme trapz en trapezoid (et supprime np.trapz) ; on gere les deux.
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


# ============================================================
# 1. DETECTION / SUPPRESSION DES ZONES PARASITES
# ============================================================
def detect_parasite_zones(signal, fs=FS, window_ms=5, smooth_ms=80,
                           k=CLEAN_K, min_duration_ms=CLEAN_MIN_DURATION_MS,
                           margin_ms=CLEAN_MARGIN_MS):
    signal = signal.astype(np.float64)
    n = len(signal)

    # 0. Retrait de l'offset DC
    offset = np.median(signal)
    signal_ac = signal - offset

    # 1. RMS fin
    window_size = max(1, int(fs * window_ms / 1000))
    kernel_fin = np.ones(window_size) / window_size
    pad = window_size // 2
    signal_padded = np.pad(signal_ac, pad, mode="reflect")
    rms_fin = np.sqrt(np.convolve(signal_padded**2, kernel_fin, mode="same"))
    rms_fin = rms_fin[pad:pad + n]

    # 2. RMS lisse
    smooth_size = max(1, int(fs * smooth_ms / 1000))
    kernel_smooth = np.ones(smooth_size) / smooth_size
    pad_s = smooth_size // 2
    rms_padded = np.pad(rms_fin, pad_s, mode="reflect")
    rms_smooth = np.convolve(rms_padded, kernel_smooth, mode="same")
    rms_smooth = rms_smooth[pad_s:pad_s + n]

    # 3. Seuil robuste
    med = np.median(rms_smooth)
    mad = np.median(np.abs(rms_smooth - med))
    threshold = med + k * mad * 1.4826

    mask = rms_smooth > threshold

    # 4. Nettoyage morphologique
    struct_close = np.ones(int(fs * 10 / 1000))
    mask = binary_closing(mask, structure=struct_close)

    struct_dilate = np.ones(int(fs * margin_ms / 1000))
    mask = binary_dilation(mask, structure=struct_dilate)

    labeled, n_zones = label(mask)
    zones = []
    min_duration_samples = int(fs * min_duration_ms / 1000)
    for i in range(1, n_zones + 1):
        idx = np.where(labeled == i)[0]
        if len(idx) >= min_duration_samples:
            zones.append((idx[0], idx[-1]))

    return zones, rms_smooth, threshold


def remove_parasite_zones(signal, zones):
    """
    Supprime les zones de parasite d'un signal et recolle les morceaux restants.
    zones : liste de tuples (start_idx, end_idx) a supprimer (bornes incluses)
    """
    if not zones:
        return signal.copy()

    zones_triees = sorted(zones, key=lambda z: z[0])

    morceaux = []
    curseur = 0

    for start, end in zones_triees:
        if start > curseur:
            morceaux.append(signal[curseur:start])
        curseur = end + 1

    if curseur < len(signal):
        morceaux.append(signal[curseur:])

    return np.concatenate(morceaux) if morceaux else np.array([], dtype=signal.dtype)


def clean_signal(signal, fs=FS):
    """Detecte puis retire les zones parasites d'un signal centre."""
    zones, _, _ = detect_parasite_zones(signal, fs=fs)
    return remove_parasite_zones(signal, zones)


# ============================================================
# 2. EXTRACTION DES FEATURES
# ============================================================
def extract_features(signal, fs=FS) -> dict:
    """
    Calcule le vecteur de features a partir d'un signal_clean :
    - temporelles : RMS, variance, crest factor, kurtosis, peak-to-peak
    - frequentielles : energie par bande (Welch), sur BANDES_HZ
    """
    signal = signal.astype(np.float64)
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))

    features = {
        "rms": rms,
        "variance": np.var(signal),
        "crest_factor": peak / rms if rms > 0 else 0.0,
        "kurtosis": stats.kurtosis(signal),
        "peak_to_peak": np.ptp(signal),
    }

    freqs, psd = welch(signal, fs=fs, nperseg=min(4096, len(signal)))
    for f_min, f_max in BANDES_HZ:
        mask = (freqs >= f_min) & (freqs < f_max)
        energie = _trapz(psd[mask], freqs[mask]) if mask.any() else 0.0
        features[f"energie_{f_min}_{f_max}hz"] = energie

    return features


def extract_feature_vector(signal, fs=FS) -> np.ndarray:
    """Meme chose que extract_features mais retourne un vecteur ordonne
    (FEATURE_COLS) pret pour scaler.transform / model.predict."""
    feats = extract_features(signal, fs=fs)
    return np.array([feats[c] for c in FEATURE_COLS])
