"""
Dataset Generator
==================
Generates synthetic microgrid measurement data for training and evaluation.

Output: data/processed/microgrid_dataset.csv
        data/sample_data/sample_50.csv (50-sample preview)

Columns (32 features + label + metadata):
  v_mean, v_std, ..., dt_max,  label, attack_name, split
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import json

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED         = 42
N_SAMPLES    = 10_000
NOMINAL_V    = 480.0    # V
NOMINAL_F    = 60.0     # Hz
FS           = 10.0     # Hz — IEC 61850 sampling rate
WINDOW_SIZE  = 10       # samples per feature window → 1 s

LABEL_MAP = {
    0: "Normal",
    1: "FDI",
    2: "DoS",
    3: "GPS_Spoofing",
    4: "Replay",
    5: "Meas_Manipulation",
}

# Attack injection ratios (% of total samples)
ATTACK_FRACTIONS = {1: 0.12, 2: 0.10, 3: 0.08, 4: 0.08, 5: 0.07}  # ~55% normal

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"
SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample_data"


# ---------------------------------------------------------------------------
# Raw signal generator
# ---------------------------------------------------------------------------
def generate_raw_signal(n: int, label: int, rng: np.random.Generator):
    t = np.arange(n) / FS

    # Base voltage (480 V + slow drift + noise)
    v = (NOMINAL_V
         + 2.0 * np.sin(2*np.pi*0.05*t)
         + rng.normal(0, 0.5, n))

    # Base frequency (60 Hz + slight variation + noise)
    f = (NOMINAL_F
         + 0.02 * np.cos(2*np.pi*0.1*t)
         + rng.normal(0, 0.01, n))

    # Current (nominal load ≈ 100 A)
    i = 100.0 + rng.normal(0, 2, n)

    # Active power
    p = v * i * 0.95  # PF ≈ 0.95

    # Reactive power
    q = v * i * np.sqrt(1 - 0.95**2) + rng.normal(0, 200, n)

    # Attack perturbations
    if label == 1:   # FDI — voltage spike
        v += rng.uniform(15, 40, n)
        f += rng.normal(0.3, 0.05, n)

    elif label == 2: # DoS — communication delay → timestamp jitter
        pass          # handled via dt features in extractor

    elif label == 3: # GPS Spoofing — timestamp offset → f deviation
        f += rng.uniform(-0.8, 0.8, n)

    elif label == 4: # Replay — repeated measurements
        # All samples are identical (near-zero variance)
        v = np.full(n, NOMINAL_V + rng.normal(0, 0.1))
        f = np.full(n, NOMINAL_F + rng.normal(0, 0.005))
        i = np.full(n, 100.0)
        p = v * i * 0.95
        q = v * i * 0.3

    elif label == 5: # Measurement Manipulation — gain scaling
        scale = rng.uniform(1.05, 1.15, n)
        v *= scale
        i *= scale

    return {"voltage": v, "frequency": f, "current": i, "active_p": p, "reactive_q": q}


# ---------------------------------------------------------------------------
# Feature extraction (self-contained, no external imports)
# ---------------------------------------------------------------------------
def extract_features(window: dict) -> np.ndarray:
    v  = np.asarray(window["voltage"])
    f  = np.asarray(window["frequency"])
    i  = np.asarray(window["current"])
    p  = np.asarray(window["active_p"])
    q  = np.asarray(window["reactive_q"])
    s  = np.sqrt(p**2 + q**2)
    pf = p / (s + 1e-9)

    def _thd(sig, nh=5):
        fm = np.abs(np.fft.rfft(sig - sig.mean()))
        return float(np.sqrt(np.sum(fm[2:nh+2]**2)) / (fm[1] + 1e-9))

    feats = np.array([
        # Voltage [0:8]
        np.mean(v)/NOMINAL_V, np.std(v)/NOMINAL_V,
        np.min(v)/NOMINAL_V,  np.max(v)/NOMINAL_V,
        np.sqrt(np.mean(v**2))/NOMINAL_V, _thd(v),
        np.max(np.abs(v))/(np.sqrt(np.mean(v**2))+1e-9),
        float(np.mean(((v-v.mean())/(v.std()+1e-9))**3)),
        # Frequency [8:16]
        (np.mean(f)-NOMINAL_F)/NOMINAL_F, np.std(f),
        np.min(f)-NOMINAL_F, np.max(f)-NOMINAL_F,
        float(np.mean(np.diff(f)*FS)) if len(f)>1 else 0.0,
        float(np.max(np.abs(np.diff(f)*FS))) if len(f)>1 else 0.0,
        float(np.mean(((f-f.mean())/(f.std()+1e-9))**3)),
        np.percentile(f,95)-np.percentile(f,5),
        # Power [16:24]
        np.mean(p)/1e5, np.std(p)/1e5, np.mean(q)/1e5, np.mean(s)/1e5,
        np.mean(pf), np.std(pf), np.max(p)/1e5, np.min(p)/1e5,
        # Current [24:28]
        np.mean(i)/100, np.std(i)/100, np.sqrt(np.mean(i**2))/100, _thd(i),
        # Temporal [28:32]
        1/FS, 0.0, 0.0, 1/FS,
    ], dtype=np.float32)
    return feats


FEATURE_NAMES = [
    "v_mean","v_std","v_min","v_max","v_rms","v_thd","v_crest","v_skew",
    "f_dev_mean","f_std","f_min_dev","f_max_dev","f_roc_mean","f_roc_max","f_skew","f_range",
    "p_mean","p_std","q_mean","s_mean","pf_mean","pf_std","p_max","p_min",
    "i_mean","i_std","i_rms","i_thd",
    "dt_mean","dt_std","missed_pkts","dt_max",
]


# ---------------------------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------------------------
def build_dataset(n_total: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    rng     = np.random.default_rng(seed)
    records = []

    # Compute per-class sample counts
    n_attack = {cls: int(n_total * frac) for cls, frac in ATTACK_FRACTIONS.items()}
    n_normal = n_total - sum(n_attack.values())
    counts   = {0: n_normal, **n_attack}

    for label, count in counts.items():
        for _ in range(count):
            raw  = generate_raw_signal(WINDOW_SIZE, label, rng)
            feat = extract_features(raw)
            row  = {name: float(val) for name, val in zip(FEATURE_NAMES, feat)}
            row["label"]       = label
            row["attack_name"] = LABEL_MAP[label]
            records.append(row)

    df = pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)

    # Train / val / test split stratified
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(df))
    y   = df["label"].values
    idx_tv, idx_test = train_test_split(idx, test_size=0.10, random_state=seed, stratify=y)
    idx_train, idx_val = train_test_split(idx_tv, test_size=0.111, random_state=seed,
                                          stratify=y[idx_tv])  # 0.111×0.9≈0.10
    df["split"] = "test"
    df.loc[idx_train, "split"] = "train"
    df.loc[idx_val,   "split"] = "val"

    return df


# ---------------------------------------------------------------------------
# Scaler export
# ---------------------------------------------------------------------------
def export_scaler(df: pd.DataFrame, out_path: Path):
    train = df[df["split"] == "train"]
    X     = train[FEATURE_NAMES].values
    sc    = StandardScaler().fit(X)
    with open(out_path, "w") as fh:
        json.dump({"mean": sc.mean_.tolist(), "std": sc.scale_.tolist()}, fh, indent=2)
    print(f"Scaler saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating {N_SAMPLES} samples …")
    df = build_dataset()
    print(df["attack_name"].value_counts().to_string())

    csv_path = OUTPUT_DIR / "microgrid_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"Dataset saved → {csv_path}  ({len(df)} rows, {len(df.columns)} cols)")

    # Sample preview
    sample_path = SAMPLE_DIR / "sample_50.csv"
    df.head(50).to_csv(sample_path, index=False)
    print(f"Sample saved  → {sample_path}")

    # Scaler
    export_scaler(df, OUTPUT_DIR / "scaler_params.json")
