"""Low-altitude A2G wireless channel model and O(1) online Lyapunov drift-plus-penalty
escalation controller for CORIT-AD (Section III & Section IV-D, Proposition 2).
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class AlHouraniA2GChannel:
    """Probabilistic LoS/NLoS Air-to-Ground (A2G) channel model for UAV inspection.

    Parameters follow the standard Al-Hourani suburban/apron geometry:
      altitude h = 15 m, horizontal radius r_t in [8, 60] m,
      a = 9.61, b = 0.16, eta_LoS = 1 dB, eta_NLoS = 20 dB,
      reference SNR = 15 dB at d0 = 20 m, minimum link floor = 0.15 * R_nom.
    """

    altitude_m: float = 15.0
    r_min_m: float = 8.0
    r_max_m: float = 60.0
    env_a: float = 9.61
    env_b: float = 0.16
    eta_los_db: float = 1.0
    eta_nlos_db: float = 20.0
    ref_dist_m: float = 20.0
    ref_snr_db: float = 15.0
    min_rate_floor: float = 0.15

    def sample_trace(
        self,
        num_slots: int,
        nominal_mbps: float = 10.0,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates per-slot achievable A2G uplink rates (Mbps) and LoS indicators."""
        rng = np.random.RandomState(seed + 777)
        phase = np.linspace(0.0, 2.0 * np.pi * 3.0, num_slots)
        r_horiz = np.clip(
            25.0 + 18.0 * np.sin(phase) + rng.normal(0.0, 2.0, num_slots),
            self.r_min_m,
            self.r_max_m,
        )
        dist = np.sqrt(self.altitude_m**2 + r_horiz**2)
        theta_deg = np.degrees(np.arctan2(self.altitude_m, r_horiz))
        p_los = 1.0 / (1.0 + self.env_a * np.exp(-self.env_b * (theta_deg - self.env_a)))
        is_los = rng.rand(num_slots) < p_los

        pl_ref = 20.0 * np.log10(self.ref_dist_m) + self.eta_los_db
        pl = (
            20.0 * np.log10(dist)
            + np.where(is_los, self.eta_los_db, self.eta_nlos_db)
            + rng.normal(0.0, 2.0, num_slots)
        )
        snr_ref_lin = 10.0 ** (self.ref_snr_db / 10.0)
        snr_lin = snr_ref_lin * (10.0 ** (-(pl - pl_ref) / 10.0))
        cap_ratio = np.log2(1.0 + snr_lin) / np.log2(1.0 + snr_ref_lin)
        rates_mbps = np.maximum(self.min_rate_floor * nominal_mbps, nominal_mbps * cap_ratio)
        return rates_mbps, is_los


class StaticEscalationPolicy:
    """Normal-calibrated static escalation rule: escalate iff r(x) >= tau or E_conf(x) >= eta."""

    def __init__(self, tau: float, eta: float) -> None:
        self.tau = float(tau)
        self.eta = float(eta)

    def decide(self, risk: np.ndarray, conflict: np.ndarray) -> np.ndarray:
        return (np.asarray(risk) >= self.tau) | (np.asarray(conflict) >= self.eta)


class LyapunovEscalationController:
    """O(1) online Lyapunov drift-plus-penalty controller for joint review and A2G airtime budgets.

    At each inspection frame slot t, given escalation utility U_t in [0, 1],
    compressed image payload bytes L_t, and instantaneous A2G rate R_t (Mbps):

        u_t = 1  iff  V * U_t >= Q_rev(t) + Q_comm(t) * (tau_tx(t) / tau_max)

    where virtual queues evolve via:
        Q_rev(t+1)  = max(0, Q_rev(t) + u_t - rho_rev)
        Q_comm(t+1) = max(0, Q_comm(t) + (u_t * tau_tx(t) - tau_max) / tau_max)
    """

    def __init__(
        self,
        rho_rev: float = 0.25,
        acq_fps: float = 5.0,
        airtime_duty_cycle: float = 0.50,
        lyapunov_v: float = 6.0,
    ) -> None:
        self.rho_rev = float(rho_rev)
        self.slot_sec = 1.0 / float(acq_fps)
        self.airtime_cap_sec = float(airtime_duty_cycle) * self.slot_sec
        self.lyapunov_v = float(lyapunov_v)
        self.reset()

    def reset(self) -> None:
        self.q_rev: float = 0.0
        self.q_comm: float = 0.0

    def step(self, utility: float, payload_bytes: float, rate_mbps: float) -> Tuple[bool, float, float]:
        """Executes one O(1) online escalation decision and updates virtual queues."""
        tx_sec = (float(payload_bytes) * 8.0) / (max(float(rate_mbps), 1e-6) * 1e6)
        price = self.q_rev + self.q_comm * (tx_sec / self.airtime_cap_sec)
        escalate = (self.lyapunov_v * float(utility)) >= price
        act = 1.0 if escalate else 0.0
        self.q_rev = max(0.0, self.q_rev + act - self.rho_rev)
        self.q_comm = max(
            0.0,
            self.q_comm + (tx_sec * act - self.airtime_cap_sec) / self.airtime_cap_sec,
        )
        return escalate, self.q_rev, self.q_comm

    def run_stream(
        self,
        utilities: np.ndarray,
        payload_bytes: np.ndarray,
        rates_mbps: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Runs the online controller over an entire inspection frame stream."""
        self.reset()
        n = len(utilities)
        decisions = np.zeros(n, dtype=bool)
        q_rev_hist = np.zeros(n, dtype=np.float64)
        q_comm_hist = np.zeros(n, dtype=np.float64)
        for t in range(n):
            esc, qr, qc = self.step(utilities[t], payload_bytes[t], rates_mbps[t])
            decisions[t] = esc
            q_rev_hist[t] = qr
            q_comm_hist[t] = qc
        return {
            "escalate": decisions,
            "q_rev": q_rev_hist,
            "q_comm": q_comm_hist,
        }
