"""CORIT-AD: Collaborative Evidence-Driven Anomaly Detection under Low-False-Positive
Constraints for Low-Altitude Edge-Assisted Visual Inspection.

Core algorithmic modules released for peer review.
"""

from .backbone import BACKBONE_SPECS, TruncatedDualViewBackbone, patch_neighborhood_embed
from .evidence_conflict import (
    CORITADModel,
    NormalCalibrator,
    compute_conflict_energy,
    compute_patch_distances,
    fit_streaming_pca,
    greedy_kcenter_coreset,
    pool_image_scores,
)
from .lyapunov_controller import (
    AlHouraniA2GChannel,
    LyapunovEscalationController,
    StaticEscalationPolicy,
)
from .metrics import operating_metrics, ranking_metrics

__all__ = [
    "BACKBONE_SPECS",
    "TruncatedDualViewBackbone",
    "patch_neighborhood_embed",
    "CORITADModel",
    "NormalCalibrator",
    "compute_conflict_energy",
    "compute_patch_distances",
    "fit_streaming_pca",
    "greedy_kcenter_coreset",
    "pool_image_scores",
    "AlHouraniA2GChannel",
    "LyapunovEscalationController",
    "StaticEscalationPolicy",
    "ranking_metrics",
    "operating_metrics",
]
