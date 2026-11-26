# CORIT-AD: Collaborative Evidence-Driven Anomaly Detection under Low-False-Positive Constraints for Low-Altitude Edge-Assisted Visual Inspection

Official **core implementation** accompanying the manuscript submitted to *IEEE Transactions on Network Science and Engineering (TNSE)* — Special Issue on *Embodied AI for Low-Altitude Economy Networking*:

> **CORIT-AD: Collaborative Evidence-Driven Anomaly Detection under Low-False-Positive Constraints for Low-Altitude Edge-Assisted Visual Inspection**  
> Xiaojia Wang, Jinsong Jiang, Chuan Luo, and Xiaoxu Ren

---

## Release Scope (Peer-Review Core Release)

> **Note:** During peer review, this repository releases the **core algorithmic modules** and frozen configuration of CORIT-AD for transparency and method verification:
> - Truncated shared multi-backbone dual-view feature extraction (`corit_ad/backbone.py`)
> - Exact normal-only streaming PCA, greedy $k$-center coreset construction, normal-only calibration, and multiplicative softplus conflict energy (`corit_ad/evidence_conflict.py`)
> - Al-Hourani probabilistic LoS/NLoS A2G channel model and $O(1)$ online Lyapunov drift-plus-penalty escalation controller (`corit_ad/lyapunov_controller.py`)
> - Discrete empirical low-FPR evaluation metrics (`corit_ad/metrics.py`) and SHA256-frozen hyperparameters (`configs/corit_v31_frozen.json`)
>
> **Full experiment runners, multi-seed benchmark harnesses, SOTA baseline wrappers (PaDiM / SimpleNet / RD4AD / PatchCore), and figure/table reproduction scripts will be released in this repository upon paper acceptance.**

---

## Method Overview

CORIT-AD organizes anomaly scoring and networked escalation into a three-layer functional hierarchy:

1. **Conflict-Aware Evidence Modeling:** Executes a single truncated shared backbone pass (discarding unused deep blocks to cut active parameters by **62.5%–85.5%**) to derive two complementary Euclidean views—local $3\times 3$ neighborhood appearance and unpooled multi-scale PCA spatial context—calibrates both exclusively on a disjoint held-out normal split, and measures cross-view disagreement via multiplicative softplus conflict energy:
   $$E_{\mathrm{conf}}(x) = \bigl(1 - e^{-|\tilde{s}_{\mathrm{app}}(x) - \tilde{s}_{\mathrm{str}}(x)|}\bigr)\bigl(\mathrm{softplus}(\tilde{s}_{\mathrm{app}}(x)) + \mathrm{softplus}(\tilde{s}_{\mathrm{str}}(x))\bigr)$$
   with a provable exponential low-score normal suppression bound $E_{\mathrm{conf}}(x) < 2e^{-M}$ whenever $\max(\tilde{s}_{\mathrm{app}}(x), \tilde{s}_{\mathrm{str}}(x)) \le -M$.
2. **Risk-Aware Decision Scoring:** Combines calibrated view evidence and conflict energy into $r(x) = \alpha \tilde{s}_{\mathrm{app}}(x) + (1-\alpha)\tilde{s}_{\mathrm{str}}(x) + \lambda E_{\mathrm{conf}}(x)$.
3. **Review-Aware and Resource-Aware Escalation:** Maps $(r(x), E_{\mathrm{conf}}(x))$ to direct local archiving versus selective A2G upload through either a normal-calibrated static rule or an $O(1)$ online Lyapunov drift-plus-penalty controller governed by human-review ($Q_{\mathrm{rev}}$) and A2G uplink airtime ($Q_{\mathrm{comm}}$) virtual queues.

---

## Repository Structure

```text
CORIT-AD/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   └── corit_v31_frozen.json        # Dev-selected, SHA256-frozen hyperparameters
├── corit_ad/
│   ├── __init__.py
│   ├── backbone.py                  # Truncated WRN-50-2 / ResNet-34 / ResNet-18 / MobileNetV3-Large
│   ├── evidence_conflict.py         # Streaming PCA, coreset, calibration, conflict energy & CORITADModel
│   ├── lyapunov_controller.py       # Al-Hourani A2G channel + online Lyapunov escalation gate
│   └── metrics.py                   # Discrete empirical AUROC, AP, TPR@1%/2%/5%FPR, pAUC@5%, FPR@90%TPR
└── demo_core.py                     # Minimal self-contained CPU/GPU verification script
```

---

## Public Benchmark Datasets & Official Links

All experiments in the paper use four publicly available benchmarks (evaluated under disjoint 80/20 normal bank/calibration splits and zero test leakage). Please download the datasets directly from their official sources and comply with their respective licenses:

| Benchmark | Domain / Scenario | Categories / Units | Official Source & Download Link | Reference |
| :--- | :--- | :---: | :--- | :--- |
| **MVTec AD** | Industrial objects & textures | 15 categories | Official Portal: <https://www.mvtec.com/company/research/datasets/mvtec-ad> | Bergmann *et al.*, CVPR 2019 / IJCV 2021 |
| **VisA** | Multi-instance industrial visual anomaly | 12 categories | GitHub: <https://github.com/amazon-research/spot-diff><br>AWS Open Data: <https://registry.opendata.aws/visa><br>CLI: `aws s3 cp --no-sign-request s3://amazon-visual-anomaly/VisA_20220922.tar .` | Zou *et al.*, ECCV 2022 |
| **InsPLAD** | **Real UAV** power-line asset inspection | 5 defect classes (group-aware splits) | GitHub: <https://github.com/andreluizbvs/InsPLAD> | Vieira e Silva *et al.*, *Int. J. Remote Sensing*, 2023 |
| **AeBAD (AeBAD-S)** | Aero-engine blade anomaly with domain shift | 4 shift types (`same`, `background`, `illumination`, `view`) | GitHub: <https://github.com/zhangzilongc/MMR> | Zhang *et al.*, CVPR 2023 |

---

## Summary of Main Verified Results (5 Seeds, SHA256-Frozen Protocol)

| Truncated Backbone | Active Params | GFLOPs | CPU 4T Latency | MVTec AD (AUROC / TPR@5%) | VisA (AUROC / TPR@5%) | InsPLAD UAV (AUROC / TPR@5%) | AeBAD-S (AUROC / TPR@5%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Wide-ResNet-50-2** | 24.86M (−63.9%) | 39.76 | 190.9 ms | 98.24% / 90.61% | **95.74% / 85.18%** | 78.38% / 18.55% | 62.90% / 8.66% |
| **ResNet-34** | 8.17M (−62.5%) | 14.13 | 86.7 ms | 97.48% / 87.20% | **95.75% / 84.14%** | 81.43% / 20.37% | **69.92% / 13.51%** |
| **ResNet-18** | 2.78M (−76.2%) | 7.52 | 59.7 ms | 96.74% / 85.61% | 95.19% / 82.35% | **82.36% / 23.14%** | 67.00% / 15.71% |
| **MobileNetV3-Large** | **0.79M (−85.5%)** | **2.22** | **35.4 ms (28.3 FPS)** | **98.08% / 90.79%** | **95.28% / 82.89%** | 81.07% / **26.58%** | 62.53% / 12.15% |

- **Low-Altitude A2G Offloading (10 Mbps, $\rho_{\mathrm{rev}}=0.25$):** Online Lyapunov control achieves **0.0%** review-budget violation (vs. +25.9%–+50.8% static thresholding), **71.5%–96.4%** uplink payload savings, and **67.5–137.6 ms** mean end-to-end latency.
- **Human-in-the-Loop Escalation:** Adding conflict-energy routing boosts system anomaly recall across all four benchmarks by **+1.46 to +15.68 pp** ($p_{\mathrm{hum}}=0.95$) and intercepts up to **77.9%** of edge errors and **92.3%** of edge false alarms at a 20% human-review budget.

---

## Quickstart

```bash
pip install -r requirements.txt
python demo_core.py
```

Minimal Python usage:

```python
import torch
from corit_ad import CORITADModel, AlHouraniA2GChannel, LyapunovEscalationController

# 1. Fit CORIT-AD on normal training & disjoint normal calibration tensors [N, 3, 320, 320]
model = CORITADModel(backbone_name="mobilenet_v3_large", view_dim=128, k_nn=1, agg="max", lam=0.2)
model.fit(normal_bank_images, normal_cal_images)

# 2. Score incoming UAV inspection frames
preds = model.predict(test_images)
risk, conflict, utility = preds["risk"], preds["conflict"], preds["utility"]

# 3. Online Lyapunov A2G escalation under fading wireless links
rates_mbps, _ = AlHouraniA2GChannel().sample_trace(len(utility), nominal_mbps=10.0)
controller = LyapunovEscalationController(rho_rev=0.25, acq_fps=5.0, airtime_duty_cycle=0.50, lyapunov_v=6.0)
decisions = controller.run_stream(utility, payload_bytes, rates_mbps)
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{wang2026coritad,
  title   = {{CORIT-AD}: Collaborative Evidence-Driven Anomaly Detection under Low-False-Positive Constraints for Low-Altitude Edge-Assisted Visual Inspection},
  author  = {Wang, Xiaojia and Jiang, Jinsong and Luo, Chuan and Ren, Xiaoxu},
  journal = {IEEE Transactions on Network Science and Engineering},
  note    = {Under Review},
  year    = {2026}
}
```
