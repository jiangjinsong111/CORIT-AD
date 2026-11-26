"""Dual-view Euclidean evidence scoring, normal-only calibration, and
multiplicative softplus conflict energy for CORIT-AD.

Implements:
  - Exact streaming float64 PCA fitted strictly on defect-free training tokens.
  - Greedy k-center coreset subsampling for compact edge memory banks.
  - Exact nearest-neighbor Euclidean patch distances and image pooling.
  - Held-out normal-only z-score calibration and multiplicative softplus
    conflict energy E_conf(x) (Proposition 1 & Theorem 1).
"""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .backbone import TruncatedDualViewBackbone


def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def fit_streaming_pca(
    token_batches: Iterable[torch.Tensor],
    n_components: int = 128,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact full-population streaming PCA over normal training patch tokens.

    Args:
        token_batches: Iterable yielding tensors of shape [B, N, C] or [M, C].
        n_components: Maximum number of principal directions to retain.
        device: Computation device (defaults to first batch's device).

    Returns:
        (mean [1, 1, C] float32, basis [C, d_eff] float32) on CPU.
    """
    sx: Optional[torch.Tensor] = None
    sxx: Optional[torch.Tensor] = None
    count = 0

    for batch in token_batches:
        dev = device if device is not None else batch.device
        flat = batch.to(device=dev, dtype=torch.float64).reshape(-1, batch.shape[-1])
        if sx is None:
            dim = flat.shape[-1]
            sx = torch.zeros(dim, dtype=torch.float64, device=dev)
            sxx = torch.zeros((dim, dim), dtype=torch.float64, device=dev)
        sx += flat.sum(dim=0)
        sxx += flat.T @ flat
        count += flat.shape[0]

    if sx is None or sxx is None or count == 0:
        raise ValueError("token_batches must yield at least one non-empty batch.")

    mean = sx / count
    cov = (sxx - count * torch.outer(mean, mean)) / max(count - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    d_eff = min(n_components, mean.shape[0])
    order = torch.argsort(eigvals, descending=True)[:d_eff]
    basis = eigvecs[:, order].float().cpu()
    return mean.float().view(1, 1, -1).cpu(), basis


@torch.inference_mode()
def greedy_kcenter_coreset(
    descriptors: np.ndarray,
    ratio: float = 0.01,
    max_tokens: int = 4096,
    proj_dim: int = 128,
    seed: int = 42,
    n_start: int = 10,
    device: str = "cpu",
) -> np.ndarray:
    """Approximate greedy k-center coreset selection preserving Euclidean norm."""
    seed_everything(seed)
    arr = np.ascontiguousarray(descriptors.reshape(-1, descriptors.shape[-1]), dtype=np.float32)
    n_tokens, dim = arr.shape
    k = min(max_tokens, max(1, int(n_tokens * ratio)))
    if k >= n_tokens:
        return arr.copy()

    dev = torch.device(device)
    if dim > proj_dim:
        mapper = nn.Linear(dim, proj_dim, bias=False).to(dev)
        reduced = torch.empty((n_tokens, proj_dim), dtype=torch.float32, device=dev)
        for i in range(0, n_tokens, 200_000):
            xb = torch.from_numpy(arr[i : i + 200_000]).to(dev)
            reduced[i : i + len(xb)] = mapper(xb)
    else:
        reduced = torch.from_numpy(arr).to(dev)

    x_norm = reduced.square().sum(dim=1)
    ns = min(n_start, n_tokens)
    starts = np.random.choice(n_tokens, ns, replace=False)
    s_vec = reduced[starts]
    s_norm = x_norm[starts]
    d0 = (x_norm[:, None] + s_norm[None, :] - 2.0 * (reduced @ s_vec.T)).clamp_min_(0.0).sqrt_()
    min_dist = d0.mean(dim=1)

    chosen = np.empty(k, dtype=np.int64)
    for t in range(k):
        idx = int(torch.argmax(min_dist).item())
        chosen[t] = idx
        center = reduced[idx]
        d = (x_norm + x_norm[idx] - 2.0 * torch.mv(reduced, center)).clamp_min_(0.0).sqrt_()
        torch.minimum(min_dist, d, out=min_dist)

    return np.ascontiguousarray(arr[chosen], dtype=np.float32)


@torch.inference_mode()
def compute_patch_distances(
    descriptors: np.ndarray,
    memory_bank: np.ndarray,
    k_nn: int = 1,
    imgs_per_batch: int = 8,
    chunk: int = 4096,
    device: str = "cpu",
) -> np.ndarray:
    """Computes per-patch Euclidean k-NN distances against a coreset memory bank."""
    dev = torch.device(device)
    mem = torch.from_numpy(np.ascontiguousarray(memory_bank, dtype=np.float32)).to(dev)
    mem_norm = mem.square().sum(dim=1)
    n_imgs, n_patches, dim = descriptors.shape
    kk = min(k_nn, mem.shape[0])
    out = np.empty((n_imgs, n_patches), dtype=np.float32)

    for i in range(0, n_imgs, imgs_per_batch):
        bsz = min(imgs_per_batch, n_imgs - i)
        x = torch.from_numpy(
            np.ascontiguousarray(descriptors[i : i + bsz], dtype=np.float32).reshape(-1, dim)
        ).to(dev)
        parts = []
        for j in range(0, len(x), chunk):
            xb = x[j : j + chunk]
            sq = (xb.square().sum(dim=1, keepdim=True) + mem_norm[None] - 2.0 * xb @ mem.T).clamp_min_(0.0)
            vals = torch.topk(sq, kk, dim=1, largest=False).values.sqrt_()
            parts.append(vals[:, 0] if kk == 1 else vals.mean(dim=1))
        out[i : i + bsz] = torch.cat(parts).view(bsz, n_patches).cpu().numpy()
    return out


def pool_image_scores(patch_dists: np.ndarray, agg: str = "max", topp: float = 0.01) -> np.ndarray:
    """Pools patch distance maps [n_imgs, N] to image-level view evidence scores."""
    if agg == "max":
        return patch_dists.max(axis=1).astype(np.float64)
    if agg == "top1":
        n_patches = patch_dists.shape[1]
        kt = max(1, math.ceil(n_patches * topp))
        return np.partition(patch_dists, n_patches - kt, axis=1)[:, -kt:].mean(axis=1).astype(np.float64)
    raise ValueError(f"Unsupported aggregation '{agg}'. Choose 'max' or 'top1'.")


def compute_conflict_energy(
    s_app_cal: np.ndarray,
    s_str_cal: np.ndarray,
) -> np.ndarray:
    """Multiplicative softplus conflict energy (Eq. 9, Proposition 1 & Theorem 1):

        E_conf(x) = (1 - exp(-|s_app - s_str|)) * (softplus(s_app) + softplus(s_str))

    Satisfies:
      - Non-negativity, symmetry, consensus-zero, strict monotonicity.
      - Exponential low-score normal suppression: if max(s_app, s_str) <= -M,
        then E_conf(x) <= 2 (1 - exp(-|s_app - s_str|)) log(1 + exp(-M)) < 2 exp(-M).
    """
    za = np.asarray(s_app_cal, dtype=np.float64)
    zs = np.asarray(s_str_cal, dtype=np.float64)
    inconsistency = -np.expm1(-np.abs(za - zs))
    evidence_gate = np.logaddexp(0.0, za) + np.logaddexp(0.0, zs)
    return inconsistency * evidence_gate


@dataclass
class NormalCalibrator:
    """Held-out normal-only z-score calibrator and threshold estimator."""

    alpha: float = 0.5
    lam: float = 0.2
    risk_quantile: float = 0.95
    conflict_quantile: float = 0.90

    mu_app: float = 0.0
    sd_app: float = 1.0
    mu_str: float = 0.0
    sd_str: float = 1.0
    mu_risk: float = 0.0
    sd_risk: float = 1.0
    mu_conf: float = 0.0
    sd_conf: float = 1.0
    tau: float = 0.0
    eta: float = 0.0

    def fit(self, app_cal: np.ndarray, str_cal: np.ndarray) -> "NormalCalibrator":
        """Fits calibration statistics and quantiles strictly on held-out normal data."""
        app_cal = np.asarray(app_cal, dtype=np.float64)
        str_cal = np.asarray(str_cal, dtype=np.float64)
        self.mu_app = float(app_cal.mean())
        self.sd_app = max(float(app_cal.std()), 1e-8)
        self.mu_str = float(str_cal.mean())
        self.sd_str = max(float(str_cal.std()), 1e-8)

        cal_scores = self.score(app_cal, str_cal)
        r_cal = cal_scores["risk"]
        e_cal = cal_scores["conflict"]
        self.mu_risk = float(r_cal.mean())
        self.sd_risk = max(float(r_cal.std()), 1e-8)
        self.mu_conf = float(e_cal.mean())
        self.sd_conf = max(float(e_cal.std()), 1e-8)
        self.tau = float(np.quantile(r_cal, self.risk_quantile, method="higher"))
        self.eta = float(np.quantile(e_cal, self.conflict_quantile, method="higher"))
        return self

    def score(self, app_raw: np.ndarray, str_raw: np.ndarray) -> Dict[str, np.ndarray]:
        """Computes calibrated single-view scores, conflict energy, and risk score r(x)."""
        za = (np.asarray(app_raw, dtype=np.float64) - self.mu_app) / self.sd_app
        zs = (np.asarray(str_raw, dtype=np.float64) - self.mu_str) / self.sd_str
        e_conf = compute_conflict_energy(za, zs)
        risk = self.alpha * za + (1.0 - self.alpha) * zs + self.lam * e_conf
        return {
            "s_app": za,
            "s_str": zs,
            "conflict": e_conf,
            "risk": risk,
        }

    def escalation_utility(self, risk: np.ndarray, conflict: np.ndarray) -> np.ndarray:
        """Label-free escalation utility U(x) in [0, 1] used by the Lyapunov controller."""
        r_z = (np.asarray(risk, dtype=np.float64) - self.mu_risk) / self.sd_risk
        tau_z = (self.tau - self.mu_risk) / self.sd_risk
        e_z = (np.asarray(conflict, dtype=np.float64) - self.mu_conf) / self.sd_conf
        margin_sig = 1.0 / (1.0 + np.exp(-np.clip(r_z - tau_z, -30.0, 30.0)))
        conf_sig = 1.0 / (1.0 + np.exp(-np.clip(e_z - 1.28, -30.0, 30.0)))
        return np.clip(0.75 * margin_sig + 0.25 * conf_sig, 0.0, 1.0)


class CORITADModel:
    """End-to-end CORIT-AD core model (dev-selected v3.1 unpooled Euclidean design)."""

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        pretrained: bool = True,
        view_dim: int = 128,
        k_nn: int = 1,
        agg: str = "max",
        alpha: float = 0.5,
        lam: float = 0.2,
        coreset_ratio: float = 0.01,
        coreset_max_tokens: int = 4096,
        risk_quantile: float = 0.95,
        conflict_quantile: float = 0.90,
        seed: int = 42,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = TruncatedDualViewBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            view_dim=view_dim,
        ).to(self.device)
        self.view_dim = view_dim
        self.k_nn = k_nn
        self.agg = agg
        self.coreset_ratio = coreset_ratio
        self.coreset_max_tokens = coreset_max_tokens
        self.seed = seed
        self.calibrator = NormalCalibrator(
            alpha=alpha,
            lam=lam,
            risk_quantile=risk_quantile,
            conflict_quantile=conflict_quantile,
        )
        self.pca_params: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.bank_app: Optional[np.ndarray] = None
        self.bank_str: Optional[np.ndarray] = None

    @torch.inference_mode()
    def _extract_batched(
        self,
        images: torch.Tensor,
        batch_size: int = 16,
        use_pca: bool = True,
    ) -> Dict[str, np.ndarray]:
        apps, strs = [], []
        pca = self.pca_params if use_pca else None
        for i in range(0, len(images), batch_size):
            xb = images[i : i + batch_size].to(self.device)
            views = self.backbone.extract_views(xb, pca_params=pca)
            apps.append(views["app"].cpu().numpy())
            strs.append((views["str"] if use_pca else views["str_raw"]).cpu().numpy())
        return {
            "app": np.concatenate(apps, axis=0),
            "str": np.concatenate(strs, axis=0),
        }

    @torch.inference_mode()
    def fit(
        self,
        normal_bank_images: torch.Tensor,
        normal_cal_images: torch.Tensor,
        batch_size: int = 16,
    ) -> "CORITADModel":
        """Fits normal-only PCA, dual memory banks, and disjoint normal calibration."""
        def _str_batches():
            for i in range(0, len(normal_bank_images), batch_size):
                xb = normal_bank_images[i : i + batch_size].to(self.device)
                yield self.backbone.extract_views(xb, pca_params=None)["str_raw"]

        self.pca_params = fit_streaming_pca(_str_batches(), n_components=self.view_dim)
        bank_feats = self._extract_batched(normal_bank_images, batch_size=batch_size, use_pca=True)
        self.bank_app = greedy_kcenter_coreset(
            bank_feats["app"],
            ratio=self.coreset_ratio,
            max_tokens=self.coreset_max_tokens,
            seed=self.seed,
            device=self.device,
        )
        self.bank_str = greedy_kcenter_coreset(
            bank_feats["str"],
            ratio=self.coreset_ratio,
            max_tokens=self.coreset_max_tokens,
            seed=self.seed + 1,
            device=self.device,
        )

        cal_feats = self._extract_batched(normal_cal_images, batch_size=batch_size, use_pca=True)
        d_app_cal = compute_patch_distances(cal_feats["app"], self.bank_app, k_nn=self.k_nn, device=self.device)
        d_str_cal = compute_patch_distances(cal_feats["str"], self.bank_str, k_nn=self.k_nn, device=self.device)
        s_app_cal = pool_image_scores(d_app_cal, agg=self.agg)
        s_str_cal = pool_image_scores(d_str_cal, agg=self.agg)
        self.calibrator.fit(s_app_cal, s_str_cal)
        return self

    @torch.inference_mode()
    def predict(self, images: torch.Tensor, batch_size: int = 16) -> Dict[str, np.ndarray]:
        """Returns dual-view scores, conflict energy E_conf(x), risk r(x), and utility U(x)."""
        if self.bank_app is None or self.bank_str is None:
            raise RuntimeError("Call fit() before predict().")
        feats = self._extract_batched(images, batch_size=batch_size, use_pca=True)
        d_app = compute_patch_distances(feats["app"], self.bank_app, k_nn=self.k_nn, device=self.device)
        d_str = compute_patch_distances(feats["str"], self.bank_str, k_nn=self.k_nn, device=self.device)
        scores = self.calibrator.score(
            pool_image_scores(d_app, agg=self.agg),
            pool_image_scores(d_str, agg=self.agg),
        )
        scores["utility"] = self.calibrator.escalation_utility(scores["risk"], scores["conflict"])
        scores["static_escalate"] = (scores["risk"] >= self.calibrator.tau) | (
            scores["conflict"] >= self.calibrator.eta
        )
        return scores
