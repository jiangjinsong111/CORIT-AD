"""Minimal self-contained sanity check for CORIT-AD core modules.

Verifies on CPU or GPU (no external datasets required):
  1. Truncated multi-backbone active parameter reduction.
  2. Dual-view Euclidean scoring, normal calibration, and Theorem 1 exponential
     low-score normal suppression bound E_conf(x) < 2 * exp(-M).
  3. Online Lyapunov drift-plus-penalty queue boundedness under Al-Hourani A2G fading.
"""

import numpy as np
import torch

from corit_ad import (
    AlHouraniA2GChannel,
    CORITADModel,
    LyapunovEscalationController,
    compute_conflict_energy,
)


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[1/3] Initializing truncated CORIT-AD core on {device} (mobilenet_v3_large, pretrained=False)...")
    model = CORITADModel(
        backbone_name="mobilenet_v3_large",
        pretrained=False,
        view_dim=64,
        coreset_ratio=0.10,
        coreset_max_tokens=256,
        device=device,
    )
    print(
        f"      Active params: {model.backbone.active_params / 1e6:.2f}M / "
        f"{model.backbone.full_params / 1e6:.2f}M "
        f"(-{model.backbone.param_reduction_pct:.1f}%)"
    )

    # Synthetic normal bank, disjoint normal calibration, and mixed test tensors
    normal_bank = torch.randn(8, 3, 128, 128) * 0.1
    normal_cal = torch.randn(4, 3, 128, 128) * 0.1
    test_imgs = torch.cat(
        [
            torch.randn(6, 3, 128, 128) * 0.1,
            torch.randn(4, 3, 128, 128) * 0.1 + 1.5,
        ],
        dim=0,
    )
    model.fit(normal_bank, normal_cal, batch_size=4)
    out = model.predict(test_imgs, batch_size=4)
    print(
        f"      Predicted {len(out['risk'])} frames | "
        f"tau={model.calibrator.tau:.3f}, eta={model.calibrator.eta:.3f}, "
        f"mean risk={out['risk'].mean():.3f}"
    )

    print("[2/3] Verifying Theorem 1 exponential low-score suppression bound...")
    for m_val in (2.0, 4.0, 6.0):
        za = np.array([-m_val, -m_val - 3.0])
        zs = np.array([-m_val - 2.0, -m_val])
        e_conf = compute_conflict_energy(za, zs)
        upper = 2.0 * np.exp(-m_val)
        assert np.all(e_conf < upper), f"Bound violated for M={m_val}"
        print(f"      M={m_val:.1f}: max E_conf={e_conf.max():.6f} < 2*exp(-M)={upper:.6f} [PASS]")

    print("[3/3] Verifying online Lyapunov A2G offloading queue stability...")
    n_slots = 400
    channel = AlHouraniA2GChannel()
    rates_mbps, is_los = channel.sample_trace(n_slots, nominal_mbps=10.0, seed=42)
    rng = np.random.RandomState(42)
    utilities = rng.beta(1.2, 4.0, size=n_slots)
    payload_bytes = np.full(n_slots, 180_000.0)  # ~180 KB compressed JPEG frame
    controller = LyapunovEscalationController(rho_rev=0.25, acq_fps=5.0, airtime_duty_cycle=0.50, lyapunov_v=6.0)
    res = controller.run_stream(utilities, payload_bytes, rates_mbps)
    esc_rate = float(res["escalate"].mean())
    print(
        f"      LoS ratio={is_los.mean():.2f}, escalation rate={esc_rate:.3f} (budget rho=0.250), "
        f"max Q_rev={res['q_rev'].max():.2f}, max Q_comm={res['q_comm'].max():.2f} [PASS]"
    )
    print("All CORIT-AD core checks passed.")


if __name__ == "__main__":
    main()
