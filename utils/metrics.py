"""
常用 metrics（3D segmentation + 二分類 classification）

Segmentation（二分類：背景/腫瘤）
- soft dice（用 tumor 機率，常用於訓練/驗證）
- hard dice（用 argmax 或 threshold）
- IoU (Jaccard)
- precision / recall / specificity
- confusion matrix (TP, FP, TN, FN)

Classification（二分類：良/惡性）
- accuracy / precision / recall / specificity / F1
- AUROC / AUPRC（不依賴 sklearn，純 torch/numpy 計算）
- best threshold（可選：在 val set 找使 F1 最大的 threshold）

注意：
- segmentation logits 預期 shape: [B,2,D,H,W]
- seg_gt 預期 shape: [B,D,H,W]，值 {0,1}
- classification logits 預期 shape: [B,2]（或 [B] 機率）
- y_cls 預期 shape: [B]，值 {0,1}（1=malignant）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# 基礎工具
# ============================================================
def _to_numpy_1d(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        x = x.reshape(-1)
        return x.astype(np.float64)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().reshape(-1).astype(np.float64)
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a) / float(b + eps)


# ============================================================
# Segmentation metrics（二分類）
# ============================================================
@torch.no_grad()
def seg_prob_from_logits(seg_logits_2c: torch.Tensor) -> torch.Tensor:
    """
    seg_logits_2c: [B,2,D,H,W]
    return tumor prob: [B,D,H,W]
    """
    if seg_logits_2c.ndim != 5 or seg_logits_2c.size(1) != 2:
        raise ValueError(f"seg_logits_2c must be [B,2,D,H,W], got {tuple(seg_logits_2c.shape)}")
    return F.softmax(seg_logits_2c, dim=1)[:, 1]


@torch.no_grad()
def seg_pred_from_logits(
    seg_logits_2c: torch.Tensor,
    threshold: float = 0.5,
    use_argmax: bool = True
) -> torch.Tensor:
    """
    產生 hard prediction mask（0/1）
    - use_argmax=True: 直接 argmax（通常跟 threshold 類似但更保守）
    - use_argmax=False: tumor prob >= threshold
    return: [B,D,H,W] uint8
    """
    if use_argmax:
        pred = seg_logits_2c.argmax(dim=1)  # [B,D,H,W] in {0,1}
        return pred.to(torch.uint8)
    prob = seg_prob_from_logits(seg_logits_2c)
    return (prob >= threshold).to(torch.uint8)


@torch.no_grad()
def seg_confusion(
    seg_pred: torch.Tensor,
    seg_gt: torch.Tensor
) -> Dict[str, int]:
    """
    seg_pred: [B,D,H,W] 0/1
    seg_gt:   [B,D,H,W] 0/1
    """
    if seg_pred.shape != seg_gt.shape:
        raise ValueError(f"shape mismatch: pred={tuple(seg_pred.shape)} gt={tuple(seg_gt.shape)}")

    p = seg_pred.to(torch.bool)
    g = seg_gt.to(torch.bool)

    tp = int((p & g).sum().item())
    fp = int((p & ~g).sum().item())
    tn = int((~p & ~g).sum().item())
    fn = int((~p & g).sum().item())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


@torch.no_grad()
def seg_soft_dice(
    seg_logits_2c: torch.Tensor,
    seg_gt: torch.Tensor,
    smooth: float = 1.0
) -> float:
    """
    Soft Dice（用 tumor 機率，不是 argmax）
    seg_gt: [B,D,H,W] 0/1
    """
    prob = seg_prob_from_logits(seg_logits_2c)  # [B,D,H,W]
    gt = seg_gt.float()
    if prob.shape != gt.shape:
        raise ValueError(f"shape mismatch: prob={tuple(prob.shape)} gt={tuple(gt.shape)}")

    inter = (prob * gt).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))
    dice = (2 * inter + smooth) / (denom + smooth)
    return float(dice.mean().item())


@torch.no_grad()
def seg_hard_metrics(
    seg_logits_2c: torch.Tensor,
    seg_gt: torch.Tensor,
    threshold: float = 0.5,
    use_argmax: bool = True
) -> Dict[str, float]:
    """
    回傳 hard 指標：dice/iou/precision/recall/specificity
    """
    pred = seg_pred_from_logits(seg_logits_2c, threshold=threshold, use_argmax=use_argmax)
    gt = seg_gt.to(torch.uint8)
    conf = seg_confusion(pred, gt)

    tp, fp, tn, fn = conf["tp"], conf["fp"], conf["tn"], conf["fn"]
    dice = _safe_div(2 * tp, 2 * tp + fp + fn)
    iou = _safe_div(tp, tp + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


# ============================================================
# Classification metrics（二分類）
# ============================================================
@torch.no_grad()
def cls_prob_from_logits(cls_logits_2c: torch.Tensor) -> torch.Tensor:
    """
    cls_logits_2c: [B,2] -> malignant prob [B]
    """
    if cls_logits_2c.ndim != 2 or cls_logits_2c.size(1) != 2:
        raise ValueError(f"cls_logits_2c must be [B,2], got {tuple(cls_logits_2c.shape)}")
    return F.softmax(cls_logits_2c, dim=1)[:, 1]


@torch.no_grad()
def cls_pred_from_prob(prob: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    prob: [B] in [0,1] -> pred: [B] 0/1
    """
    return (prob >= threshold).to(torch.uint8)


@torch.no_grad()
def cls_confusion_from_prob(prob: torch.Tensor, y: torch.Tensor, threshold: float = 0.5) -> Dict[str, int]:
    """
    prob: [B] malignant prob
    y:   [B] 0/1
    """
    p = cls_pred_from_prob(prob, threshold=threshold).to(torch.bool)
    g = y.view(-1).to(torch.bool)

    tp = int((p & g).sum().item())
    fp = int((p & ~g).sum().item())
    tn = int((~p & ~g).sum().item())
    fn = int((~p & g).sum().item())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


@torch.no_grad()
def cls_basic_metrics_from_prob(prob: torch.Tensor, y: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    回傳 accuracy/precision/recall/specificity/F1
    """
    conf = cls_confusion_from_prob(prob, y, threshold=threshold)
    tp, fp, tn, fn = conf["tp"], conf["fp"], conf["tn"], conf["fn"]

    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "threshold": float(threshold),
    }


# ============================================================
# AUROC / AUPRC（不靠 sklearn）
# ============================================================
def _binary_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    回傳 (fpr, tpr, thresholds)
    """
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    # edge cases
    pos = (y_true == 1).sum()
    neg = (y_true == 0).sum()
    if pos == 0 or neg == 0:
        # 無法定義 ROC（全正或全負）
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])

    # sort by score desc
    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    # distinct thresholds
    distinct_idx = np.where(np.diff(y_score_sorted))[0]
    thr_idx = np.r_[distinct_idx, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted)[thr_idx]
    fps = (1 + thr_idx) - tps  # total predicted positive - tps

    tpr = tps / pos
    fpr = fps / neg
    thresholds = y_score_sorted[thr_idx]

    # prepend (0,0) point
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[np.inf, thresholds]
    return fpr, tpr, thresholds


def _binary_pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    回傳 (precision, recall, thresholds)
    """
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    pos = (y_true == 1).sum()
    if pos == 0:
        # 無法定義 PR（沒有正樣本）
        return np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])

    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    distinct_idx = np.where(np.diff(y_score_sorted))[0]
    thr_idx = np.r_[distinct_idx, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted)[thr_idx]
    fps = (1 + thr_idx) - tps

    recall = tps / pos
    precision = tps / np.maximum(tps + fps, 1)

    # prepend recall=0 with precision=1
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    thresholds = np.r_[np.inf, y_score_sorted[thr_idx]]
    return precision, recall, thresholds


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    """
    以梯形法則算 AUC（x 必須遞增）
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x/y must be 1D with same length")
    return float(np.trapz(y, x))


def auroc(y_true, y_score) -> Optional[float]:
    """
    回傳 AUROC；若全正或全負則回傳 None
    """
    yt = _to_numpy_1d(y_true)
    ys = _to_numpy_1d(y_score)

    if np.all(yt == 0) or np.all(yt == 1):
        return None

    fpr, tpr, _ = _binary_roc_curve(yt, ys)
    return auc_trapezoid(fpr, tpr)


def auprc(y_true, y_score) -> Optional[float]:
    """
    回傳 AUPRC（Average Precision 近似：precision-recall 曲線下面積）
    若沒有正樣本則回傳 None
    """
    yt = _to_numpy_1d(y_true)
    ys = _to_numpy_1d(y_score)

    if np.all(yt == 0):
        return None

    precision, recall, _ = _binary_pr_curve(yt, ys)
    # recall 是遞增，適合 trapezoid
    return auc_trapezoid(recall, precision)


def find_best_threshold_by_f1(y_true, y_score, thresholds: Optional[Iterable[float]] = None) -> Dict[str, float]:
    """
    在一組 threshold 上找 F1 最大者（常用於 val set）
    """
    yt = _to_numpy_1d(y_true).astype(np.int64)
    ys = _to_numpy_1d(y_score).astype(np.float64)

    if thresholds is None:
        # 使用分位數做一個不會太慢的掃描
        qs = np.linspace(0.0, 1.0, 101)
        thresholds = np.quantile(ys, qs)

    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for thr in thresholds:
        pred = (ys >= thr).astype(np.int64)
        tp = int(((pred == 1) & (yt == 1)).sum())
        fp = int(((pred == 1) & (yt == 0)).sum())
        fn = int(((pred == 0) & (yt == 1)).sum())

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        if f1 > best["f1"]:
            best = {"threshold": float(thr), "f1": float(f1), "precision": float(precision), "recall": float(recall)}
    return best


# ============================================================
# 聚合：一個 batch / epoch 常用輸出
# ============================================================
@torch.no_grad()
def compute_all_metrics(
    seg_logits_2c: torch.Tensor,
    seg_gt: torch.Tensor,
    cls_logits_2c: torch.Tensor,
    y_cls: torch.Tensor,
    seg_threshold: float = 0.5,
    cls_threshold: float = 0.5,
    seg_use_argmax: bool = True,
) -> Dict[str, float]:
    """
    一次算出常用 segmentation + classification 指標（不含 AUC，需要跨整個 val set 累積再算）
    """
    seg_m = seg_hard_metrics(seg_logits_2c, seg_gt, threshold=seg_threshold, use_argmax=seg_use_argmax)
    seg_sd = seg_soft_dice(seg_logits_2c, seg_gt)

    prob = cls_prob_from_logits(cls_logits_2c)
    cls_m = cls_basic_metrics_from_prob(prob, y_cls, threshold=cls_threshold)

    out = {
        "seg_soft_dice": float(seg_sd),
        "seg_dice": float(seg_m["dice"]),
        "seg_iou": float(seg_m["iou"]),
        "seg_precision": float(seg_m["precision"]),
        "seg_recall": float(seg_m["recall"]),
        "seg_specificity": float(seg_m["specificity"]),
        "cls_accuracy": float(cls_m["accuracy"]),
        "cls_precision": float(cls_m["precision"]),
        "cls_recall": float(cls_m["recall"]),
        "cls_specificity": float(cls_m["specificity"]),
        "cls_f1": float(cls_m["f1"]),
    }
    return out


# ============================================================
# 最小自測（可直接 python utils/metrics.py）
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    # --- segmentation dummy ---
    B, D, H, W = 2, 16, 16, 16
    seg_logits = torch.randn(B, 2, D, H, W)
    seg_gt = (torch.rand(B, D, H, W) > 0.85).long()

    print("seg_soft_dice:", seg_soft_dice(seg_logits, seg_gt))
    print("seg_hard_metrics:", seg_hard_metrics(seg_logits, seg_gt, use_argmax=True))

    # --- classification dummy ---
    cls_logits = torch.randn(B, 2)
    y_cls = torch.randint(0, 2, (B,))

    prob = cls_prob_from_logits(cls_logits)
    print("cls_basic:", cls_basic_metrics_from_prob(prob, y_cls, threshold=0.5))

    # --- AUC dummy (need more samples) ---
    N = 200
    y = torch.randint(0, 2, (N,))
    score = torch.rand(N) * 0.7 + y.float() * 0.3  # 讓正樣本分數稍高
    print("auroc:", auroc(y, score))
    print("auprc:", auprc(y, score))
    print("best_thr:", find_best_threshold_by_f1(y, score))
    print("[OK] metrics.py self-test done.")
