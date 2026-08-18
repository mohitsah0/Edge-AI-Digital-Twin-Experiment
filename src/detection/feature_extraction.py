"""
Feature Extraction Module
==========================
Converts raw IEC 61850 sampled values (voltage, current, frequency, power)
into the 32-dimensional feature vector consumed by EdgeCNN1D.

Feature groups (8 features each × 4 groups = 32 total):
  [0:8]  Voltage statistics          (mean, std, min, max, RMS, THD, crest, skew)
  [8:16] Frequency statistics        (mean, std, min, max, rate-of-change, ...)
  [16:24] Power statistics           (active P, reactive Q, apparent S, PF, ...)
  [24:32] Temporal / metadata        (sequence idx, delta_t, attack_flag placeholder, ...)
"""

import numpy as np
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------
class FeatureExtractor:
    """
    Extracts a fixed 32-dimensional feature vector from a time-series window
    of microgrid measurements.

    Parameters
    ----------
    window_size : int
        Number of samples per feature extraction window (default 10 → 1 s at 10 Hz).
    nominal_v : float
        Nominal voltage in volts (default 480 V).
    nominal_f : float
        Nominal frequency in Hz (default 60 Hz).
    """

    NUM_FEATURES = 32

    def __init__(
        self,
        window_size: int = 10,
        nominal_v:   float = 480.0,
        nominal_f:   float = 60.0,
    ):
        self.window_size = window_size
        self.nominal_v   = nominal_v
        self.nominal_f   = nominal_f

    # ------------------------------------------------------------------
    def extract(self, window: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Parameters
        ----------
        window : dict with keys
            'voltage'   (V)  shape (window_size,)
            'current'   (A)  shape (window_size,)
            'frequency' (Hz) shape (window_size,)
            'active_p'  (W)  shape (window_size,)
            'reactive_q' (VAR) shape (window_size,)  [optional]
            'timestamp' (s)  shape (window_size,)    [optional]

        Returns
        -------
        np.ndarray  shape (32,)  float32
        """
        v  = np.asarray(window["voltage"],   dtype=np.float64)
        i  = np.asarray(window.get("current",   np.ones_like(v)), dtype=np.float64)
        f  = np.asarray(window["frequency"], dtype=np.float64)
        p  = np.asarray(window.get("active_p",  v * i), dtype=np.float64)
        q  = np.asarray(window.get("reactive_q", np.zeros_like(v)), dtype=np.float64)
        ts = np.asarray(window.get("timestamp",  np.arange(len(v)) / 10.0), dtype=np.float64)

        feats = np.zeros(self.NUM_FEATURES, dtype=np.float32)

        # --- Voltage features [0:8] ---
        feats[0] = np.mean(v) / self.nominal_v
        feats[1] = np.std(v)  / self.nominal_v
        feats[2] = np.min(v)  / self.nominal_v
        feats[3] = np.max(v)  / self.nominal_v
        feats[4] = np.sqrt(np.mean(v**2)) / self.nominal_v      # RMS
        feats[5] = self._thd(v)                                  # THD
        feats[6] = np.max(np.abs(v)) / (np.sqrt(np.mean(v**2)) + 1e-9)  # crest
        feats[7] = float(np.mean(((v - v.mean()) / (v.std() + 1e-9))**3))  # skewness

        # --- Frequency features [8:16] ---
        feats[8]  = (np.mean(f) - self.nominal_f) / self.nominal_f
        feats[9]  = np.std(f)
        feats[10] = np.min(f) - self.nominal_f
        feats[11] = np.max(f) - self.nominal_f
        roc_f     = np.diff(f) * 10.0  # rate of change [Hz/s] (×10 Hz rate)
        feats[12] = np.mean(roc_f)  if len(roc_f) else 0.0
        feats[13] = np.max(np.abs(roc_f)) if len(roc_f) else 0.0
        feats[14] = float(np.mean(((f - f.mean()) / (f.std() + 1e-9))**3))  # f-skew
        feats[15] = np.percentile(f, 95) - np.percentile(f, 5)              # f-range

        # --- Power features [16:24] ---
        s         = np.sqrt(p**2 + q**2)  # apparent power
        pf        = p / (s + 1e-9)
        feats[16] = np.mean(p)   / 1e4    # normalise by 10 kW
        feats[17] = np.std(p)    / 1e4
        feats[18] = np.mean(q)   / 1e4
        feats[19] = np.mean(s)   / 1e4
        feats[20] = np.mean(pf)
        feats[21] = np.std(pf)
        feats[22] = np.max(p)    / 1e4
        feats[23] = np.min(p)    / 1e4

        # --- Current features [24:28] ---
        feats[24] = np.mean(i)  / 100.0
        feats[25] = np.std(i)   / 100.0
        feats[26] = np.sqrt(np.mean(i**2)) / 100.0   # RMS current
        feats[27] = self._thd(i)

        # --- Temporal / delta features [28:32] ---
        delta_t   = np.diff(ts) if len(ts) > 1 else np.array([0.1])
        feats[28] = np.mean(delta_t)          # mean inter-sample interval
        feats[29] = np.std(delta_t)           # jitter (DoS indicator)
        feats[30] = float(np.sum(delta_t > 0.2))  # missed packets count
        feats[31] = float(np.max(delta_t))        # max gap (DoS indicator)

        return feats

    # ------------------------------------------------------------------
    @staticmethod
    def _thd(signal: np.ndarray, n_harmonics: int = 5) -> float:
        """Approximate Total Harmonic Distortion via FFT."""
        if len(signal) < 4:
            return 0.0
        fft_mag  = np.abs(np.fft.rfft(signal - signal.mean()))
        if fft_mag[1] < 1e-9:
            return 0.0
        harmonic = np.sqrt(np.sum(fft_mag[2:n_harmonics+2]**2))
        return float(harmonic / (fft_mag[1] + 1e-9))

    # ------------------------------------------------------------------
    def feature_names(self) -> list:
        return [
            "v_mean","v_std","v_min","v_max","v_rms","v_thd","v_crest","v_skew",
            "f_dev_mean","f_std","f_min_dev","f_max_dev","f_roc_mean","f_roc_max","f_skew","f_range",
            "p_mean","p_std","q_mean","s_mean","pf_mean","pf_std","p_max","p_min",
            "i_mean","i_std","i_rms","i_thd",
            "dt_mean","dt_std","missed_pkts","dt_max",
        ]


# ---------------------------------------------------------------------------
# Online data generator (simulates IEC 61850 stream for testing)
# ---------------------------------------------------------------------------
class MicrogridStreamSimulator:
    """
    Simulates a 10 Hz IEC 61850 data stream from a renewable microgrid.
    Injects FDI and DoS attacks at configurable time windows.
    """

    def __init__(
        self,
        duration_s: float = 60.0,
        fs: float = 10.0,
        fdi_window: tuple = (10.0, 20.0),
        dos_window: tuple = (35.0, 45.0),
        seed: int = 42,
    ):
        self.fs          = fs
        self.rng         = np.random.default_rng(seed)
        n                = int(duration_s * fs)
        self.t           = np.arange(n) / fs
        self._v          = self._gen_voltage()
        self._f          = self._gen_frequency()
        self._labels     = np.zeros(n, dtype=np.int64)
        self._inject_fdi(fdi_window)
        self._inject_dos(dos_window)
        self._idx = 0

    # ------------------------------------------------------------------
    def _gen_voltage(self):
        t = self.t
        n = len(t)
        return (480.0
                + 2.0  * np.sin(2*np.pi*0.05*t)
                + self.rng.normal(0, 0.5, n))

    def _gen_frequency(self):
        t = self.t
        n = len(t)
        return (60.0
                + 0.02 * np.cos(2*np.pi*0.1*t)
                + self.rng.normal(0, 0.01, n))

    def _inject_fdi(self, window):
        a = int(window[0] * self.fs)
        b = int(window[1] * self.fs)
        self._v[a:b] += self.rng.uniform(10, 30, b-a)   # voltage spike
        self._labels[a:b] = 1  # FDI class

    def _inject_dos(self, window):
        a = int(window[0] * self.fs)
        b = int(window[1] * self.fs)
        # Simulate dropped packets by adding large dt gaps
        self._labels[a:b] = 2  # DoS class

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        if self._idx >= len(self.t):
            raise StopIteration
        k = self._idx
        self._idx += 1
        # Simulate 100 ms interval with possible DoS jitter
        dt = 0.1 + (self.rng.uniform(0.1, 0.5) if self._labels[k] == 2 else 0.0)
        return {
            "voltage":    float(self._v[k]),
            "frequency":  float(self._f[k]),
            "current":    float(np.abs(self.rng.normal(20, 1))),
            "active_p":   float(self._v[k] * 20.0),
            "reactive_q": float(self.rng.normal(0, 500)),
            "timestamp":  float(k / self.fs),
            "label":      int(self._labels[k]),
            "dt":         float(dt),
        }


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    extractor = FeatureExtractor(window_size=10)
    sim       = MicrogridStreamSimulator(duration_s=5.0)
    window_buf: list = []

    for sample in sim:
        window_buf.append({
            "voltage":    np.array([sample["voltage"]]),
            "frequency":  np.array([sample["frequency"]]),
            "current":    np.array([sample["current"]]),
            "active_p":   np.array([sample["active_p"]]),
            "reactive_q": np.array([sample["reactive_q"]]),
            "timestamp":  np.array([sample["timestamp"]]),
        })
        if len(window_buf) == 10:
            combined = {k: np.concatenate([w[k] for w in window_buf])
                        for k in window_buf[0]}
            feats = extractor.extract(combined)
            print(f"t={sample['timestamp']:.1f}s  label={sample['label']}  "
                  f"feats[:4]={feats[:4].round(4)}")
            window_buf.clear()
