"""Truncated shared multi-backbone feature extraction for CORIT-AD.

Executes a single truncated forward pass up to stride-8 and stride-16 feature
stages (discarding unused deep classification blocks), reducing active
parameters by 62.5%--85.5% across Wide-ResNet-50-2, ResNet-34, ResNet-18, and
MobileNetV3-Large, and deriving two complementary Euclidean evidence views:
  1) Local neighborhood appearance embedding (3x3 spatial aggregation).
  2) Unpooled multi-scale spatial context ([f2, upsample(f3)]) prior to
     normal-only PCA projection.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BACKBONE_SPECS = {
    "wide_resnet50_2": (
        models.wide_resnet50_2,
        models.Wide_ResNet50_2_Weights.IMAGENET1K_V1,
        "resnet",
    ),
    "resnet34": (
        models.resnet34,
        models.ResNet34_Weights.IMAGENET1K_V1,
        "resnet",
    ),
    "resnet18": (
        models.resnet18,
        models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet",
    ),
    "mobilenet_v3_large": (
        models.mobilenet_v3_large,
        models.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        "mobilenet",
    ),
}


def adaptive_channel_reduce(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Reduces trailing channel dimension to target_dim via exact group averaging."""
    channels = x.shape[-1]
    if channels <= target_dim:
        return x
    if channels % target_dim == 0:
        return x.reshape(*x.shape[:-1], target_dim, channels // target_dim).mean(dim=-1)
    prefix = x.shape[:-1]
    return (
        F.adaptive_avg_pool1d(x.reshape(-1, 1, channels), target_dim)
        .squeeze(1)
        .reshape(*prefix, target_dim)
    )


def patch_neighborhood_embed(
    f2: torch.Tensor,
    f3: torch.Tensor,
    target_dim: int = 1024,
    patchsize: int = 3,
) -> torch.Tensor:
    """Computes 3x3 local-neighborhood appearance patch embeddings from (f2, f3)."""
    bsz, c2, h2, w2 = f2.shape
    pad = (patchsize - 1) // 2
    p2 = F.avg_pool2d(f2, patchsize, stride=1, padding=pad, count_include_pad=True)
    p3 = F.avg_pool2d(f3, patchsize, stride=1, padding=pad, count_include_pad=True)
    p3 = F.interpolate(p3, size=(h2, w2), mode="bilinear", align_corners=False)

    t2 = p2.permute(0, 2, 3, 1).reshape(bsz * h2 * w2, 1, c2)
    t3 = p3.permute(0, 2, 3, 1).reshape(bsz * h2 * w2, 1, p3.shape[1])
    dim = min(target_dim, max(c2, p3.shape[1]))
    e2 = F.adaptive_avg_pool1d(t2, dim).squeeze(1)
    e3 = F.adaptive_avg_pool1d(t3, dim).squeeze(1)
    emb = 0.5 * (e2 + e3)
    return emb.reshape(bsz, h2 * w2, dim)


class TruncatedDualViewBackbone(nn.Module):
    """Frozen truncated feature backbone producing dual Euclidean evidence views."""

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        pretrained: bool = True,
        view_dim: int = 128,
        embed_dim: int = 1024,
        patchsize: int = 3,
    ) -> None:
        super().__init__()
        if backbone_name not in BACKBONE_SPECS:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. "
                f"Choose from {sorted(BACKBONE_SPECS.keys())}."
            )
        ctor, default_weights, kind = BACKBONE_SPECS[backbone_name]
        weights = default_weights if pretrained else None
        full_model = ctor(weights=weights).eval()

        self.backbone_name = backbone_name
        self.kind = kind
        self.view_dim = view_dim
        self.embed_dim = embed_dim
        self.patchsize = patchsize
        self.full_params = sum(p.numel() for p in full_model.parameters())

        if kind == "resnet":
            self.stem = nn.Sequential(
                full_model.conv1,
                full_model.bn1,
                full_model.relu,
                full_model.maxpool,
                full_model.layer1,
            )
            self.stage1 = full_model.layer2
            self.stage2 = full_model.layer3
        else:
            self.stem = full_model.features[:4]
            self.stage1 = full_model.features[4:7]
            self.stage2 = full_model.features[7:13]

        for param in self.parameters():
            param.requires_grad_(False)
        self.active_params = sum(p.numel() for p in self.parameters())
        self.eval()

    @property
    def param_reduction_pct(self) -> float:
        return 100.0 * (1.0 - self.active_params / max(self.full_params, 1))

    @torch.inference_mode()
    def forward_stages(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single truncated forward pass returning (f2, f3)."""
        h = self.stem(x)
        f2 = self.stage1(h)
        f3 = self.stage2(f2)
        return f2, f3

    @torch.inference_mode()
    def extract_views(
        self,
        images: torch.Tensor,
        pca_params: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extracts Euclidean appearance ('app') and structure ('str_raw' / 'str') tokens.

        Args:
            images: Normalized RGB tensor [B, 3, H, W].
            pca_params: Optional tuple (mean [1, 1, C], V [C, d]) fitted solely
                on normal training tokens.

        Returns:
            Dictionary with keys:
              - 'app': [B, N, d_app] local-neighborhood appearance descriptors.
              - 'str_raw': [B, N, C2 + C3] unpooled multi-scale context tokens.
              - 'str': [B, N, d_str] PCA-projected structure descriptors (if pca_params given).
        """
        f2, f3 = self.forward_stages(images)
        bsz = images.shape[0]

        app_emb = patch_neighborhood_embed(
            f2, f3, target_dim=self.embed_dim, patchsize=self.patchsize
        )
        app = adaptive_channel_reduce(app_emb, min(self.view_dim, app_emb.shape[-1]))

        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        str_raw = (
            torch.cat([f2, f3_up], dim=1)
            .permute(0, 2, 3, 1)
            .reshape(bsz, -1, f2.shape[1] + f3.shape[1])
        )

        out: Dict[str, torch.Tensor] = {"app": app, "str_raw": str_raw}
        if pca_params is not None:
            mean, basis = pca_params
            mean = mean.to(device=str_raw.device, dtype=str_raw.dtype)
            basis = basis.to(device=str_raw.device, dtype=str_raw.dtype)
            d_eff = min(self.view_dim, basis.shape[1])
            out["str"] = (str_raw - mean) @ basis[:, :d_eff]
        return out
