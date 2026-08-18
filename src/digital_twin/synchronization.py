"""
Digital Twin Synchronisation Module
=====================================
Keeps the physics-based Digital Twin (DT) state aligned with the physical
microgrid at a configurable sync interval (default 100 ms / 10 Hz).

Architecture
------------
Physical Layer  ──IEC 61850──►  CommunicationBridge
                                       │
                                       ▼
                              DTSynchronizer
                             ┌─────────────────┐
                             │  State estimator │  ← Kalman-like smoother
                             │  Anomaly gate    │  ← reject injected states
                             │  Shadow copy     │  ← last-known-good state
                             └─────────────────┘
                                       │
                                       ▼
                         Physics model (simplified DT)

On every synchronisation tick the bridge:
  1. Reads the latest sensor measurements.
  2. Validates them against physical constraints (voltage/freq limits).
  3. If valid → updates the DT state and computes the residual.
  4. If anomalous → flags the tick and holds the DT at its last-good state.
"""

import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DT State container
# ---------------------------------------------------------------------------
@dataclass
class DTState:
    timestamp:    float
    voltage_pu:   float          # normalised voltage (1.0 = nominal)
    frequency_hz: float
    active_p_kw:  float
    reactive_q_kvar: float
    soc_pct:      float          # battery state-of-charge
    irradiance:   float          # solar irradiance W/m²
    is_valid:     bool = True    # False when anomaly detected


@dataclass
class SyncStats:
    total_ticks:    int   = 0
    valid_ticks:    int   = 0
    anomaly_ticks:  int   = 0
    mean_residual:  float = 0.0
    max_residual:   float = 0.0
    _residuals:     List[float] = field(default_factory=list, repr=False)

    def update(self, residual: float, valid: bool):
        self.total_ticks += 1
        self._residuals.append(residual)
        if valid:
            self.valid_ticks += 1
        else:
            self.anomaly_ticks += 1
        self.mean_residual = float(np.mean(self._residuals[-100:]))
        self.max_residual  = float(np.max(self._residuals[-100:]))


# ---------------------------------------------------------------------------
# Physics validator (simple range checks + rate-of-change limits)
# ---------------------------------------------------------------------------
class PhysicsValidator:
    """
    Enforces IEEE 1547 / ANSI C84.1 operational envelopes.
    """

    V_MIN_PU    = 0.88
    V_MAX_PU    = 1.10
    F_MIN_HZ    = 59.3
    F_MAX_HZ    = 60.5
    ROCOF_LIMIT = 1.0    # Hz/s — fast-change limit (under-frequency load shedding)
    DV_LIMIT    = 0.05   # pu/tick — sudden voltage step limit

    def __init__(self, nominal_v: float = 480.0, nominal_f: float = 60.0):
        self.nominal_v = nominal_v
        self.nominal_f = nominal_f
        self._prev_f: Optional[float] = None

    def validate(self, v_pu: float, f_hz: float, dt: float = 0.1) -> Tuple[bool, str]:
        if not (self.V_MIN_PU <= v_pu <= self.V_MAX_PU):
            return False, f"Voltage {v_pu:.4f} pu out of [{self.V_MIN_PU},{self.V_MAX_PU}]"
        if not (self.F_MIN_HZ <= f_hz <= self.F_MAX_HZ):
            return False, f"Frequency {f_hz:.3f} Hz out of [{self.F_MIN_HZ},{self.F_MAX_HZ}]"
        if self._prev_f is not None and dt > 0:
            rocof = abs(f_hz - self._prev_f) / dt
            if rocof > self.ROCOF_LIMIT:
                return False, f"ROCOF {rocof:.3f} Hz/s exceeds {self.ROCOF_LIMIT}"
        self._prev_f = f_hz
        return True, "OK"


# ---------------------------------------------------------------------------
# Kalman-inspired state smoother (lightweight, no full covariance matrix)
# ---------------------------------------------------------------------------
class ScalarKalman:
    """1-D Kalman filter for online state estimation."""

    def __init__(self, q: float = 1e-4, r: float = 1e-3):
        self._x = None    # state estimate
        self._p = 1.0     # error covariance
        self.q  = q       # process noise
        self.r  = r       # measurement noise

    def update(self, z: float) -> float:
        if self._x is None:
            self._x = z
            return z
        # Predict
        self._p += self.q
        # Update
        k       = self._p / (self._p + self.r)
        self._x += k * (z - self._x)
        self._p  = (1 - k) * self._p
        return float(self._x)


# ---------------------------------------------------------------------------
# Main synchroniser
# ---------------------------------------------------------------------------
class DTSynchronizer:
    """
    Maintains a synchronised Digital Twin state and detects measurement
    anomalies before they are injected into the DT.

    Parameters
    ----------
    sync_interval_s : float
        Target synchronisation interval (default 0.1 s = 10 Hz).
    nominal_v : float
        Nominal microgrid bus voltage (V).
    nominal_f : float
        Nominal grid frequency (Hz).
    residual_threshold : float
        Normalised residual above which a tick is flagged as anomalous.
    """

    def __init__(
        self,
        sync_interval_s:    float = 0.1,
        nominal_v:          float = 480.0,
        nominal_f:          float = 60.0,
        residual_threshold: float = 0.03,
    ):
        self.interval    = sync_interval_s
        self.nominal_v   = nominal_v
        self.nominal_f   = nominal_f
        self.threshold   = residual_threshold

        self.validator   = PhysicsValidator(nominal_v, nominal_f)
        self._kf_v       = ScalarKalman(q=1e-5, r=5e-4)
        self._kf_f       = ScalarKalman(q=1e-6, r=1e-5)
        self.stats       = SyncStats()
        self._dt_state:  Optional[DTState] = None
        self._last_good: Optional[DTState] = None
        self._last_tick: float = 0.0

    # ------------------------------------------------------------------
    def sync(self, measurement: dict) -> DTState:
        """
        Synchronise one measurement tick.

        Parameters
        ----------
        measurement : dict
            Keys: voltage (V), frequency (Hz), active_p (W),
                  reactive_q (VAR), soc_pct (%), irradiance (W/m²).

        Returns
        -------
        DTState — current DT state (may be held at last-good on anomaly).
        """
        now = time.time()
        dt  = now - self._last_tick if self._last_tick else self.interval
        self._last_tick = now

        v_pu = measurement.get("voltage", self.nominal_v) / self.nominal_v
        f_hz = measurement.get("frequency", self.nominal_f)

        # --- Physics validation ---
        valid, reason = self.validator.validate(v_pu, f_hz, dt)

        # --- State smoothing ---
        v_smooth = self._kf_v.update(v_pu)
        f_smooth = self._kf_f.update(f_hz)

        # --- Residual (difference between raw and smoothed) ---
        residual = float(np.sqrt((v_pu - v_smooth)**2 + ((f_hz - f_smooth)/60.0)**2))

        # Flag anomaly if residual is large even if range check passed
        if residual > self.threshold:
            valid  = False
            reason = f"Residual {residual:.5f} exceeds threshold {self.threshold}"

        self.stats.update(residual, valid)

        if valid:
            state = DTState(
                timestamp    = now,
                voltage_pu   = v_smooth,
                frequency_hz = f_smooth,
                active_p_kw  = measurement.get("active_p", 0.0) / 1000.0,
                reactive_q_kvar = measurement.get("reactive_q", 0.0) / 1000.0,
                soc_pct      = measurement.get("soc_pct", 50.0),
                irradiance   = measurement.get("irradiance", 800.0),
                is_valid     = True,
            )
            self._dt_state = state
            self._last_good = state
            logger.debug("DT sync OK  v=%.4f pu  f=%.3f Hz  res=%.6f", v_smooth, f_smooth, residual)
        else:
            logger.warning("DT sync ANOMALY — %s", reason)
            # Return last known-good state rather than poisoning the DT
            state = DTState(
                timestamp    = now,
                voltage_pu   = self._last_good.voltage_pu   if self._last_good else 1.0,
                frequency_hz = self._last_good.frequency_hz if self._last_good else self.nominal_f,
                active_p_kw  = self._last_good.active_p_kw  if self._last_good else 0.0,
                reactive_q_kvar = self._last_good.reactive_q_kvar if self._last_good else 0.0,
                soc_pct      = self._last_good.soc_pct      if self._last_good else 50.0,
                irradiance   = self._last_good.irradiance   if self._last_good else 800.0,
                is_valid     = False,
            )
            self._dt_state = state

        return state

    def current_state(self) -> Optional[DTState]:
        return self._dt_state

    def anomaly_rate(self) -> float:
        if self.stats.total_ticks == 0:
            return 0.0
        return self.stats.anomaly_ticks / self.stats.total_ticks


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s  %(message)s")
    sync = DTSynchronizer()
    rng  = np.random.default_rng(7)

    print("=== Normal operation ===")
    for k in range(5):
        m = {"voltage": 480 + rng.normal(0, 1), "frequency": 60 + rng.normal(0, 0.02),
             "active_p": 30000, "reactive_q": 1000, "soc_pct": 70, "irradiance": 800}
        s = sync.sync(m)
        print(f"  tick {k:02d}  v={s.voltage_pu:.4f} pu  f={s.frequency_hz:.3f} Hz  valid={s.is_valid}")

    print("\n=== FDI attack (voltage spike) ===")
    for k in range(3):
        m = {"voltage": 530 + rng.normal(0, 1), "frequency": 60 + rng.normal(0, 0.02),
             "active_p": 30000, "reactive_q": 1000, "soc_pct": 70, "irradiance": 800}
        s = sync.sync(m)
        print(f"  tick {k:02d}  v={s.voltage_pu:.4f} pu  valid={s.is_valid}  (last-good held)")

    print(f"\nAnomaly rate: {sync.anomaly_rate()*100:.1f}%")
    print(f"Stats: {sync.stats}")
