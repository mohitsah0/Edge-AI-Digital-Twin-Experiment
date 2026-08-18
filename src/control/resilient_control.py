"""
Resilient Control Module
=========================
Implements the three-tier resilient control strategy described in Section V
of the paper:

  Tier 1 — Immediate mitigation   (< 50 ms)  : saturate/clamp bad setpoints
  Tier 2 — Active reconfiguration (< 200 ms) : redistribute generation
  Tier 3 — Recovery & restoration (< 500 ms) : restore nominal operation

The controller receives DetectionResult objects from EdgeInferenceEngine and
the current DTState from DTSynchronizer, then outputs updated setpoints for
the PV inverter, BESS, and diesel generator.
"""

import time
import logging
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attack class mapping (mirrors EdgeCNN1D NUM_CLASSES)
# ---------------------------------------------------------------------------
class AttackClass(IntEnum):
    NORMAL          = 0
    FDI             = 1
    DOS             = 2
    GPS_SPOOFING    = 3
    REPLAY          = 4
    MEAS_MANIP      = 5


# ---------------------------------------------------------------------------
# Microgrid setpoints container
# ---------------------------------------------------------------------------
@dataclass
class Setpoints:
    pv_power_kw:    float   # PV active power reference (0 – 50 kW)
    bess_power_kw:  float   # BESS active power (positive=discharge, negative=charge)
    diesel_power_kw: float  # Diesel generator power (0 – 40 kW)
    v_ref_pu:       float   # Voltage reference (per-unit)
    f_ref_hz:       float   # Frequency reference (Hz)
    mode:           str     # "normal" | "mitigation" | "reconfigure" | "recovery"
    timestamp:      float   = 0.0

    def clip(self):
        self.pv_power_kw     = float(np.clip(self.pv_power_kw,     0,   50))
        self.bess_power_kw   = float(np.clip(self.bess_power_kw,  -30,  30))
        self.diesel_power_kw = float(np.clip(self.diesel_power_kw,  0,  40))
        self.v_ref_pu        = float(np.clip(self.v_ref_pu,       0.95, 1.05))
        self.f_ref_hz        = float(np.clip(self.f_ref_hz,       59.5, 60.5))
        return self


# ---------------------------------------------------------------------------
# Resilient controller
# ---------------------------------------------------------------------------
class ResilientController:
    """
    Three-tier resilient control for a PV + BESS + Diesel microgrid.

    Parameters
    ----------
    total_load_kw : float
        Estimated total load for power balancing (kW).
    nominal_v_pu : float
        Nominal voltage reference.
    nominal_f_hz : float
        Nominal frequency reference.
    """

    RECOVERY_TIMEOUT_S = 2.0   # seconds before returning to normal mode

    def __init__(
        self,
        total_load_kw: float = 80.0,
        nominal_v_pu:  float = 1.0,
        nominal_f_hz:  float = 60.0,
    ):
        self.load        = total_load_kw
        self.v_nom       = nominal_v_pu
        self.f_nom       = nominal_f_hz
        self._mode       = "normal"
        self._attack_t   = None   # time attack was first detected
        self._last_sp    = self._nominal_setpoints()

    # ------------------------------------------------------------------
    def _nominal_setpoints(self) -> Setpoints:
        return Setpoints(
            pv_power_kw     = 40.0,
            bess_power_kw   =  0.0,
            diesel_power_kw = 40.0,
            v_ref_pu        = self.v_nom,
            f_ref_hz        = self.f_nom,
            mode            = "normal",
            timestamp       = time.time(),
        ).clip()

    # ------------------------------------------------------------------
    def compute(
        self,
        attack_class:  int,
        confidence:    float,
        v_pu:          float,
        f_hz:          float,
        soc_pct:       float = 60.0,
    ) -> Setpoints:
        """
        Compute updated setpoints based on detection result and current state.

        Parameters
        ----------
        attack_class : int
            Predicted attack class (0 = normal).
        confidence : float
            Softmax probability of the predicted class.
        v_pu : float
            Current normalised bus voltage.
        f_hz : float
            Current grid frequency (Hz).
        soc_pct : float
            BESS state-of-charge (%).

        Returns
        -------
        Setpoints — clipped to physical limits.
        """
        now = time.time()

        if attack_class == AttackClass.NORMAL or confidence < 0.5:
            # Check if we are returning from a previous attack
            if self._attack_t is not None:
                elapsed = now - self._attack_t
                if elapsed > self.RECOVERY_TIMEOUT_S:
                    logger.info("Recovery complete — returning to normal mode")
                    self._mode    = "normal"
                    self._attack_t = None
                    return self._nominal_setpoints()
                # Still in recovery ramp
                return self._recovery_setpoints(elapsed, v_pu, f_hz)
            return self._nominal_setpoints()

        # ---- Attack detected ----
        if self._attack_t is None:
            self._attack_t = now
            logger.warning("Attack detected: class=%d conf=%.3f", attack_class, confidence)

        # Tier 1: Immediate mitigation (< 50 ms)
        sp = self._tier1_mitigation(attack_class, v_pu, f_hz, soc_pct)

        # Tier 2: Active reconfiguration (fires ~50 ms after detection)
        elapsed = now - self._attack_t
        if elapsed > 0.05:
            sp = self._tier2_reconfigure(sp, attack_class, v_pu, f_hz, soc_pct)

        sp.timestamp = now
        self._last_sp = sp
        return sp.clip()

    # ------------------------------------------------------------------
    def _tier1_mitigation(
        self, attack_class: int, v_pu: float, f_hz: float, soc_pct: float
    ) -> Setpoints:
        """Clamp voltage/frequency references; no topology change."""
        sp = Setpoints(
            pv_power_kw     = self._last_sp.pv_power_kw,
            bess_power_kw   = self._last_sp.bess_power_kw,
            diesel_power_kw = self._last_sp.diesel_power_kw,
            v_ref_pu        = float(np.clip(v_pu, 0.98, 1.02)),  # tighter band
            f_ref_hz        = float(np.clip(f_hz, 59.8, 60.2)),
            mode            = "mitigation",
        )
        # FDI: voltage spike → reduce PV, absorb with BESS
        if attack_class == AttackClass.FDI:
            sp.pv_power_kw   = max(0, self._last_sp.pv_power_kw - 10.0)
            sp.bess_power_kw = -10.0  # charge to absorb excess

        # DoS: communication loss → switch to droop/islanded mode
        elif attack_class == AttackClass.DOS:
            sp.diesel_power_kw = min(40, self.load * 0.6)  # diesel covers base load

        return sp

    def _tier2_reconfigure(
        self, sp: Setpoints, attack_class: int, v_pu: float, f_hz: float, soc_pct: float
    ) -> Setpoints:
        """
        Power redistribution to maintain load balance while the DT is
        operating on its last-good state.
        """
        sp.mode = "reconfigure"

        # Estimate available generation headroom
        gen_avail = sp.pv_power_kw + abs(sp.bess_power_kw) + sp.diesel_power_kw
        deficit   = self.load - gen_avail

        if deficit > 0:
            # Engage more BESS if SOC allows
            if soc_pct > 20:
                extra_bess = min(deficit, 30 - abs(sp.bess_power_kw))
                sp.bess_power_kw += extra_bess  # positive = discharge
            # Ramp up diesel for the rest
            sp.diesel_power_kw = min(40, sp.diesel_power_kw + max(0, deficit - extra_bess
                                                                   if soc_pct > 20 else deficit))
        return sp

    def _recovery_setpoints(self, elapsed: float, v_pu: float, f_hz: float) -> Setpoints:
        """Linear ramp back to nominal over RECOVERY_TIMEOUT_S."""
        alpha = min(1.0, elapsed / self.RECOVERY_TIMEOUT_S)  # 0→1
        nom   = self._nominal_setpoints()
        sp    = Setpoints(
            pv_power_kw     = (1-alpha)*self._last_sp.pv_power_kw     + alpha*nom.pv_power_kw,
            bess_power_kw   = (1-alpha)*self._last_sp.bess_power_kw   + alpha*nom.bess_power_kw,
            diesel_power_kw = (1-alpha)*self._last_sp.diesel_power_kw + alpha*nom.diesel_power_kw,
            v_ref_pu        = (1-alpha)*self._last_sp.v_ref_pu        + alpha*nom.v_ref_pu,
            f_ref_hz        = (1-alpha)*self._last_sp.f_ref_hz        + alpha*nom.f_ref_hz,
            mode            = "recovery",
            timestamp       = time.time(),
        )
        return sp.clip()


# ---------------------------------------------------------------------------
# Recovery mechanism helper (exported as standalone for cosimulation)
# ---------------------------------------------------------------------------
class RecoveryMechanism:
    """
    Tracks mean time to recovery (MTTR) and success rate across multiple
    attack events for reporting.
    """

    def __init__(self):
        self._events: list = []     # list of (attack_class, start_t, end_t)
        self._active: dict = {}     # attack_class → start_t

    def on_attack_start(self, attack_class: int):
        self._active[attack_class] = time.time()

    def on_recovery_complete(self, attack_class: int):
        if attack_class in self._active:
            start = self._active.pop(attack_class)
            self._events.append((attack_class, start, time.time()))

    @property
    def mttr_ms(self) -> float:
        if not self._events:
            return 0.0
        return np.mean([(e[2]-e[1])*1000 for e in self._events])

    @property
    def success_rate(self) -> float:
        return 1.0  # every completed recovery is a success in this model

    def summary(self) -> dict:
        return {
            "total_events":   len(self._events),
            "mttr_ms":        round(self.mttr_ms, 2),
            "success_rate":   self.success_rate,
        }


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    ctrl = ResilientController()

    print("=== Normal ===")
    sp = ctrl.compute(0, 0.99, 1.0, 60.0, soc_pct=70)
    print(f"  PV={sp.pv_power_kw}kW  BESS={sp.bess_power_kw}kW  Diesel={sp.diesel_power_kw}kW  mode={sp.mode}")

    print("\n=== FDI Attack (v=1.08 pu) ===")
    sp = ctrl.compute(1, 0.95, 1.08, 60.0, soc_pct=70)
    print(f"  PV={sp.pv_power_kw}kW  BESS={sp.bess_power_kw}kW  Diesel={sp.diesel_power_kw}kW  mode={sp.mode}")
    print(f"  v_ref={sp.v_ref_pu:.4f} pu  f_ref={sp.f_ref_hz:.3f} Hz")

    print("\n=== DoS Attack ===")
    sp = ctrl.compute(2, 0.88, 0.99, 59.9, soc_pct=55)
    print(f"  PV={sp.pv_power_kw}kW  BESS={sp.bess_power_kw}kW  Diesel={sp.diesel_power_kw}kW  mode={sp.mode}")
