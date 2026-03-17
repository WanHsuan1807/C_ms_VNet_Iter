# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F
from .vnet import VNet

# ---------------------------
# 1) CMS：Multi-scale classifier
#    Stage(≈x3,x4,x5) -> GAP -> L2 norm -> /C -> concat -> FC -> FC -> logits(2)
# ---------------------------
class MultiScaleClassifier(nn.Module):
    def __init__(self, in_dims: List[int], hidden: int = 256, num_classes: int = 2, eps: float = 1e-6, use_norm: bool = True):
        super().__init__()
        self.eps = eps
        self.use_norm = use_norm
        self.fc1 = nn.Linear(sum(in_dims), hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.fc_out = nn.Linear(hidden // 2, num_classes)

    @staticmethod
    def gap_3d(feat: torch.Tensor) -> torch.Tensor:
        return feat.mean(dim=(2, 3, 4))

    def l2norm_div_c(self, v: torch.Tensor) -> torch.Tensor:
        c = v.size(1)
        norm = v.norm(p=2, dim=1, keepdim=True).clamp_min(self.eps)
        return (v / norm) / float(c)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        vecs = []
        for f in feats:
            v = self.gap_3d(f)
            if self.use_norm:
                v = self.l2norm_div_c(v)
            vecs.append(v)

        x = torch.cat(vecs, dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class MultiScaleClassifierLarge(nn.Module):
    """Larger classifier variant with more capacity: deeper and wider."""
    def __init__(self, in_dims: List[int], hidden: int = 512, num_classes: int = 2, eps: float = 1e-6, use_norm: bool = True):
        super().__init__()
        self.eps = eps
        self.use_norm = use_norm
        # Larger network: in_dim -> 512 -> 256 -> 128 -> 2
        self.fc1 = nn.Linear(sum(in_dims), hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, hidden // 4)
        self.fc_out = nn.Linear(hidden // 4, num_classes)
        self.dropout = nn.Dropout(0.3)

    @staticmethod
    def gap_3d(feat: torch.Tensor) -> torch.Tensor:
        # [B,C,D,H,W] -> [B,C]
        return feat.mean(dim=(2, 3, 4))

    def l2norm_div_c(self, v: torch.Tensor) -> torch.Tensor:
        # v: [B,C]
        c = v.size(1)
        norm = v.norm(p=2, dim=1, keepdim=True).clamp_min(self.eps)
        return (v / norm) / float(c)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        vecs = []
        for f in feats:
            v = self.gap_3d(f)
            if self.use_norm:
                v = self.l2norm_div_c(v)
            vecs.append(v)

        x = torch.cat(vecs, dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        return self.fc_out(x)


# ============================================================
# 2) CMSVNet：VNet(seg) + CMS(cls)
#    backbone 必須支援 forward(x, return_encoder_features=True)
#    並回傳 (seg_logits, feats=[x3,x4,x5])
# ============================================================
class CMSVNet(nn.Module):
    def __init__(self, vnet_backbone: nn.Module, stage_channels: List[int], cls_hidden: int = 256, use_norm: bool = True, classifier_cls=None):
        super().__init__()
        self.vnet = vnet_backbone
        if classifier_cls is None:
            classifier_cls = MultiScaleClassifier
        self.classifier = classifier_cls(stage_channels, hidden=cls_hidden, num_classes=2, use_norm=use_norm)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        seg_logits, feats = self.vnet(x, return_encoder_features=True)  # <-- 你 vnet.py 已支援
        cls_logits = self.classifier(feats)
        return seg_logits, cls_logits, feats


# ============================================================
# 3) Iterative refinement 設定
# ============================================================
@dataclass
class IterConfig:
    n_iter: int = 2                 # 論文最佳 N=2
    lambda_cls: float = 0.3         # 論文用 0.3
    focal_gamma: float = 0.0        # 常用 2
    detach_probmap: bool = True     # 論文沒寫死；True 比較穩、比較省記憶體
    loss_on_all_iters: bool = True  # True: 每次迭代都算 loss 再平均；False: 只算最後一次
    use_cls_weights: bool = True    # 如果 False, classification loss 不做 class weighting


# ============================================================
# 4) 工具：Seg prob map（腫瘤通道機率）
# ============================================================
def tumor_prob_map(seg_logits_2c: torch.Tensor) -> torch.Tensor:
    """
    seg_logits_2c: [B,2,D,H,W] -> tumor prob: [B,1,D,H,W]
    """
    return F.softmax(seg_logits_2c, dim=1)[:, 1:2]


# ============================================================
# 5) Loss：weighted focal (cls) + soft dice (seg) + joint
#    你是 tumor-level classification，所以 y_cls 是 ROI-level label
# ============================================================
def weighted_focal_loss_from_logits(
    logits_2c: torch.Tensor,
    y: torch.Tensor,
    wm: float,
    wn: float,
    gamma: float = 0.0,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Args:
        logits_2c: [B,2]
        y: [B] in {0,1}，1=malignant
        wm/wn: class weight（依論文：由資料集 benign/malignant 數量算）
    """
    probs = F.softmax(logits_2c, dim=1).clamp(min=eps, max=1 - eps)
    p = probs[:, 1]  # malignant prob
    y = y.float().view(-1)

    loss_pos = -wm * ((1 - p) ** gamma) * y * torch.log(p)
    loss_neg = -wn * (p ** gamma) * (1 - y) * torch.log(1 - p)
    return (loss_pos + loss_neg).mean()


def soft_dice_loss(seg_logits_2c: torch.Tensor, seg_gt: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Args:
        seg_logits_2c: [B,2,D,H,W]
        seg_gt: [B,D,H,W] in {0,1}（腫瘤=1）
    """
    prob_t = F.softmax(seg_logits_2c, dim=1)[:, 1]  # [B,D,H,W]
    gt = seg_gt.float()

    inter = (prob_t * gt).sum(dim=(1, 2, 3))
    denom = prob_t.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))
    dice = (2 * inter + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


def compute_joint_loss(
    seg_logits_2c: torch.Tensor,
    cls_logits_2c: torch.Tensor,
    seg_gt: torch.Tensor,
    y_cls: torch.Tensor,
    cfg: IterConfig,
    nm: int,
    nn_: int,
) -> Dict[str, torch.Tensor]:
    """
    Args:
        nm: malignant count（訓練集惡性數量）
        nn_: benign count（訓練集良性數量）
    """
    # 論文權重：wm = Nn/(Nm+Nn), wn = Nm/(Nm+Nn)
    if cfg.use_cls_weights:
        wm = float(nn_) / float(nm + nn_)
        wn = float(nm) / float(nm + nn_)
    else:
        wm = 1.0
        wn = 1.0

    loss_seg = soft_dice_loss(seg_logits_2c, seg_gt)
    loss_cls = weighted_focal_loss_from_logits(
        cls_logits_2c, y_cls, wm=wm, wn=wn, gamma=cfg.focal_gamma
    )
    loss_joint = cfg.lambda_cls * loss_cls + (1.0 - cfg.lambda_cls) * loss_seg

    return {"loss_joint": loss_joint, "loss_seg": loss_seg, "loss_cls": loss_cls}


# ============================================================
# 6) Iterative forward：可選擇每次迭代都加 loss 或只用最後一次
# ============================================================
def forward_iterative_with_losses(
    model: CMSVNet,
    x: torch.Tensor,
    seg_gt: torch.Tensor,
    y_cls: torch.Tensor,
    cfg: IterConfig,
    nm: int,
    nn_: int,
    cls_on_orig: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Returns:
        loss_joint (scalar)
        loss_dict  (loss_joint/loss_seg/loss_cls)
        seg_logits_last: [B,2,D,H,W]
        cls_logits_last: [B,2]
    """
    prob_prev: Optional[torch.Tensor] = None
    orig_x = x

    loss_joint_all: List[torch.Tensor] = []
    loss_seg_all: List[torch.Tensor] = []
    loss_cls_all: List[torch.Tensor] = []

    seg_last: Optional[torch.Tensor] = None
    cls_last: Optional[torch.Tensor] = None

    for it in range(cfg.n_iter):
        # segmentation input is original volume plus previous probability map
        x_i = orig_x if it == 0 or prob_prev is None else (orig_x + prob_prev)
        seg_logits, cls_logits, _ = model(x_i)
        # optionally recompute classification using the raw image only
        if cls_on_orig and it > 0:
            # discard cls_logits computed on x_i and use original
            _, cls_logits, _ = model(orig_x)
        # 如果使用者希望分類不受概率地圖影響，可在 cfg 或外部控制
        # 現在我們沒有額外參數，所以 caller 可以在訓練時自行重新呼叫 model(orig_x)
        seg_last, cls_last = seg_logits, cls_logits

        # 下一次迭代的輸入
        prob_prev = tumor_prob_map(seg_logits)
        if cfg.detach_probmap:
            prob_prev = prob_prev.detach()

        # loss：每次都算 or 只算最後一次
        if cfg.loss_on_all_iters or (it == cfg.n_iter - 1):
            ld = compute_joint_loss(seg_logits, cls_logits, seg_gt, y_cls, cfg, nm=nm, nn_=nn_)
            loss_joint_all.append(ld["loss_joint"])
            loss_seg_all.append(ld["loss_seg"])
            loss_cls_all.append(ld["loss_cls"])

    loss_joint = torch.stack(loss_joint_all).mean()
    loss_seg = torch.stack(loss_seg_all).mean()
    loss_cls = torch.stack(loss_cls_all).mean()

    assert seg_last is not None and cls_last is not None
    return loss_joint, {"loss_joint": loss_joint, "loss_seg": loss_seg, "loss_cls": loss_cls}, seg_last, cls_last


# ============================================================
# 7) 常用 metrics（給 train.py 用）
# ============================================================
@torch.no_grad()
def cls_accuracy_from_logits(cls_logits_2c: torch.Tensor, y_cls: torch.Tensor) -> float:
    pred = cls_logits_2c.argmax(dim=1)
    y = y_cls.view(-1)
    return float((pred == y).float().mean().item())


@torch.no_grad()
def seg_soft_dice_from_logits(seg_logits_2c: torch.Tensor, seg_gt: torch.Tensor, smooth: float = 1.0) -> float:
    prob_t = F.softmax(seg_logits_2c, dim=1)[:, 1]  # [B,D,H,W]
    gt = seg_gt.float()
    inter = (prob_t * gt).sum(dim=(1, 2, 3))
    denom = prob_t.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))
    dice = (2 * inter + smooth) / (denom + smooth)
    return float(dice.mean().item())


@torch.no_grad()
def forward_iterative_inference(model: CMSVNet, x: torch.Tensor, cfg: IterConfig):
    """
    回傳最後一次 iter 的 (seg_logits_2c, cls_logits_2c)
    x: [B,1,D,H,W]
    """
    seg_logits = None
    cls_logits = None
    prev_prob = None

    for _ in range(cfg.n_iter):
        seg_logits, cls_logits, prev_prob = model(x, prev_prob=prev_prob, detach_probmap=cfg.detach_probmap)

    return seg_logits, cls_logits
# ============================================================
# 8) 最小自測：在沒有 VNet 時，用 dummy backbone 確認 CMS+Iter 能跑
# ============================================================
if __name__ == "__main__":
    class _DummyBackbone(nn.Module):
        """
        模擬你的 VNet 行為：
        forward(x, return_encoder_features=True) -> (seg_logits, [x3,x4,x5])
        """
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(1, 4, 3, padding=1)
            self.out = nn.Conv3d(4, 2, 1)

        def forward(self, x, return_encoder_features=False):
            f1 = F.relu(self.conv(x))          # [B,4,16,16,16]
            f2 = F.avg_pool3d(f1, 2)           # [B,4, 8, 8, 8]
            f3 = F.avg_pool3d(f2, 2)           # [B,4, 4, 4, 4]
            seg = self.out(f1)                 # [B,2,16,16,16]
            if return_encoder_features:
                return seg, [f1, f2, f3]
            return seg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bb = _DummyBackbone().to(device)
    model = CMSVNet(bb, stage_channels=[4, 4, 4], cls_hidden=16).to(device)

    cfg = IterConfig(n_iter=2, lambda_cls=0.3, detach_probmap=True, loss_on_all_iters=True)

    x = torch.randn(2, 1, 16, 16, 16, device=device)
    seg_gt = (torch.rand(2, 16, 16, 16, device=device) > 0.7).long()
    y_cls = torch.randint(0, 2, (2,), device=device)

    loss, ld, seg_last, cls_last = forward_iterative_with_losses(model, x, seg_gt, y_cls, cfg, nm=20, nn_=10)

    assert seg_last.shape == (2, 2, 16, 16, 16)
    assert cls_last.shape == (2, 2)

    print("[OK] cmsvnet_iter forward + loss works.")
    print({k: float(v.item()) for k, v in ld.items()}, "cls_acc=", cls_accuracy_from_logits(cls_last, y_cls),
          "seg_dice=", seg_soft_dice_from_logits(seg_last, seg_gt))