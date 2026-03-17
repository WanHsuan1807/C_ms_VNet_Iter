# train.py
# -*- coding: utf-8 -*-
"""
CMSVNetIter 訓練主程式（對接你本地 TDSC-ABUS2023 資料夾結構）

輸出：
- checkpoints/ 內會存 best_joint.pt / best_clsauc.pt / best_segdice.pt / last.pt
- 訓練過程會印每個 epoch 的 loss / dice / acc / AUC（val）

依賴：
- torch
- pandas
- (pynrrd 或 SimpleITK) 讀 nrrd
"""

from __future__ import annotations

import os
import math
import time
import json
import argparse
import random
from dataclasses import asdict
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 你前面我給你的檔案路徑
from models.vnet import VNet
from models.cmsvnet_iter import (
    CMSVNet,
    IterConfig,
    forward_iterative_with_losses,
    cls_accuracy_from_logits,
    seg_soft_dice_from_logits,
)
from data.dataset_abus import ABUSLocalConfig, ABUSLocalSegClsDataset, count_cls_labels
from utils.metrics import auroc, auprc  # 跨 epoch 累積後才算


# ============================================================
# 0) 小工具
# ============================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    # batch: (x, seg_gt, y_cls) 或 (x, seg_gt, y_cls, meta)
    if len(batch) == 3:
        x, seg, y = batch
        meta = None
    else:
        x, seg, y, meta = batch
    x = x.to(device, non_blocking=True).float()
    seg = seg.to(device, non_blocking=True).long()
    y = y.to(device, non_blocking=True).long().view(-1)
    return x, seg, y, meta


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_ckpt(path: str, state: Dict[str, Any]) -> None:
    torch.save(state, path)


def load_ckpt(path: str, model, optimizer=None, scaler=None) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def plot_training_curves(history: Dict[str, List[float]], out_dir: str) -> None:
    """
    繪製訓練/驗證曲線並儲存為圖片
    
    Args:
        history: 包含訓練歷史的字典，例如:
                {
                    'train_loss_joint': [...],
                    'train_loss_seg': [...],
                    'train_loss_cls': [...],
                    'val_loss_joint': [...],
                    'val_loss_seg': [...],
                    'val_loss_cls': [...],
                    'train_seg_dice': [...],
                    'val_seg_dice': [...],
                    'train_cls_acc': [...],
                    'val_cls_acc': [...],
                    'val_cls_auc': [...]
                }
        out_dir: 輸出目錄
    """
    # 設定中文字體支援
    rcParams['font.sans-serif'] = ['DejaVu Sans'] if 'SimHei' not in rcParams['font.sans-serif'] else ['SimHei']
    
    epochs = list(range(1, len(history['train_loss_joint']) + 1))
    
    # 建立 4 個子圖
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training and Validation Curves', fontsize=16, fontweight='bold')
    
    # 1) Loss (Joint, Seg, Cls)
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss_joint'], 'o-', label='train_loss_joint', markersize=4, linewidth=2)
    ax.plot(epochs, history['val_loss_joint'], 's-', label='val_loss_joint', markersize=4, linewidth=2)
    ax.plot(epochs, history['train_loss_seg'], '^-', label='train_loss_seg', markersize=4, linewidth=2, alpha=0.7)
    ax.plot(epochs, history['val_loss_seg'], 'v-', label='val_loss_seg', markersize=4, linewidth=2, alpha=0.7)
    ax.plot(epochs, history['train_loss_cls'], 'D-', label='train_loss_cls', markersize=4, linewidth=2, alpha=0.7)
    ax.plot(epochs, history['val_loss_cls'], 'd-', label='val_loss_cls', markersize=4, linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax.set_title('Loss Trends', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 2) Segmentation Dice
    ax = axes[0, 1]
    ax.plot(epochs, history['train_seg_dice'], 'o-', label='train_dice', markersize=4, linewidth=2)
    ax.plot(epochs, history['val_seg_dice'], 's-', label='val_dice', markersize=4, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Dice Score', fontsize=11, fontweight='bold')
    ax.set_title('Segmentation Dice', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 3) Classification Accuracy
    ax = axes[1, 0]
    ax.plot(epochs, history['train_cls_acc'], 'o-', label='train_acc', markersize=4, linewidth=2)
    ax.plot(epochs, history['val_cls_acc'], 's-', label='val_acc', markersize=4, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Classification Accuracy', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 4) Classification AUC
    ax = axes[1, 1]
    ax.plot(epochs, history['val_cls_auc'], 'o-', label='val_auc', markersize=4, linewidth=2, color='green')
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=11, fontweight='bold')
    ax.set_title('Classification AUC', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Training curves saved to {plot_path}")
    plt.close()
    
    # 分別儲存 Loss 對比圖
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, history['train_loss_joint'], 'o-', label='Train Loss (Joint)', markersize=5, linewidth=2.5)
    ax.plot(epochs, history['val_loss_joint'], 's-', label='Val Loss (Joint)', markersize=5, linewidth=2.5)
    ax.fill_between(epochs, history['train_loss_joint'], history['val_loss_joint'], alpha=0.1)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training vs Validation Loss (Joint)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(out_dir, "loss_comparison.png")
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    print(f"✓ Loss comparison saved to {loss_path}")
    plt.close()


# ============================================================
# 1) Train / Val 一個 epoch
# ============================================================
def train_one_epoch(
    model: CMSVNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    it_cfg: IterConfig,
    nm: int,
    nn_: int,
    use_amp: bool,
    grad_clip_norm: Optional[float] = 12.0,
    log_every: int = 20,
    debug_cls_grad: bool = False,
    cls_on_orig: bool = False,
) -> Dict[str, float]:
    model.train()

    t0 = time.time()
    n = 0

    sum_loss_joint = 0.0
    sum_loss_seg = 0.0
    sum_loss_cls = 0.0
    sum_dice = 0.0
    sum_acc = 0.0

    for step, batch in enumerate(loader, start=1):
        x, seg_gt, y_cls, _ = to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.amp.autocast("cuda"):
                loss_joint, loss_dict, seg_last, cls_last = forward_iterative_with_losses(
                    model=model, x=x, seg_gt=seg_gt, y_cls=y_cls,
                    cfg=it_cfg, nm=nm, nn_=nn_
                    , cls_on_orig=cls_on_orig
                )
            scaler.scale(loss_joint).backward()

            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss_joint, loss_dict, seg_last, cls_last = forward_iterative_with_losses(
                model=model, x=x, seg_gt=seg_gt, y_cls=y_cls,
                cfg=it_cfg, nm=nm, nn_=nn_
                , cls_on_orig=cls_on_orig
            )
            loss_joint.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        with torch.no_grad():
            dice = seg_soft_dice_from_logits(seg_last, seg_gt)
            acc = cls_accuracy_from_logits(cls_last, y_cls)

        bs = x.size(0)
        n += bs
        sum_loss_joint += float(loss_dict["loss_joint"].item()) * bs
        sum_loss_seg += float(loss_dict["loss_seg"].item()) * bs
        sum_loss_cls += float(loss_dict["loss_cls"].item()) * bs
        sum_dice += float(dice) * bs
        sum_acc += float(acc) * bs

        if log_every > 0 and step % log_every == 0:
            print(
                f"  [train] step {step:4d}/{len(loader)} | "
                f"loss_joint={sum_loss_joint/n:.4f} loss_seg={sum_loss_seg/n:.4f} loss_cls={sum_loss_cls/n:.4f} | "
                f"dice={sum_dice/n:.4f} acc={sum_acc/n:.4f}"
            )
            if debug_cls_grad:
                # also report current batch classification probability mean
                with torch.no_grad():
                    probs = torch.softmax(cls_last, dim=1)[:, 1]
                    print(f"    batch cls prob mean={probs.mean().item():.4f} std={probs.std().item():.4f}")
            if debug_cls_grad:
                # show classifier gradient norm so user can verify it's being updated
                total_grad = 0.0
                for name, p in model.named_parameters():
                    if "classifier" in name and p.grad is not None:
                        total_grad += float(p.grad.norm().item())
                print(f"    classifier grad norm {total_grad:.4e}")

    dt = time.time() - t0
    return {
        "loss_joint": sum_loss_joint / max(n, 1),
        "loss_seg": sum_loss_seg / max(n, 1),
        "loss_cls": sum_loss_cls / max(n, 1),
        "seg_dice": sum_dice / max(n, 1),
        "cls_acc": sum_acc / max(n, 1),
        "sec": dt,
    }


@torch.no_grad()
def validate_one_epoch(
    model: CMSVNet,
    loader: DataLoader,
    device: torch.device,
    it_cfg: IterConfig,
    nm: int,
    nn_: int,
    use_amp: bool,
    cls_on_orig: bool = False,
) -> Dict[str, float]:
    model.eval()

    n = 0
    sum_loss_joint = 0.0
    sum_loss_seg = 0.0
    sum_loss_cls = 0.0
    sum_dice = 0.0
    sum_acc = 0.0

    # 為了算 AUC：要收集所有 sample 的 y_true 與 y_score（malignant prob）
    y_true_all: List[int] = []
    y_score_all: List[float] = []

    for batch in loader:
        x, seg_gt, y_cls, _ = to_device(batch, device)

        if use_amp:
            with torch.amp.autocast("cuda"):
                loss_joint, loss_dict, seg_last, cls_last = forward_iterative_with_losses(
                    model=model, x=x, seg_gt=seg_gt, y_cls=y_cls,
                    cfg=it_cfg, nm=nm, nn_=nn_
                    , cls_on_orig=cls_on_orig
                )
        else:
            loss_joint, loss_dict, seg_last, cls_last = forward_iterative_with_losses(
                model=model, x=x, seg_gt=seg_gt, y_cls=y_cls,
                cfg=it_cfg, nm=nm, nn_=nn_
                , cls_on_orig=cls_on_orig
            )

        dice = seg_soft_dice_from_logits(seg_last, seg_gt)
        acc = cls_accuracy_from_logits(cls_last, y_cls)

        # malignant prob
        prob = torch.softmax(cls_last, dim=1)[:, 1].detach().cpu().numpy().tolist()
        y_true = y_cls.detach().cpu().numpy().tolist()

        y_true_all.extend([int(v) for v in y_true])
        y_score_all.extend([float(v) for v in prob])

        bs = x.size(0)
        n += bs
        sum_loss_joint += float(loss_dict["loss_joint"].item()) * bs
        sum_loss_seg += float(loss_dict["loss_seg"].item()) * bs
        sum_loss_cls += float(loss_dict["loss_cls"].item()) * bs
        sum_dice += float(dice) * bs
        sum_acc += float(acc) * bs

    # AUC / AUPRC（若全正或全負會回 None）
    auc = auroc(y_true_all, y_score_all)
    prc = auprc(y_true_all, y_score_all)

    return {
        "loss_joint": sum_loss_joint / max(n, 1),
        "loss_seg": sum_loss_seg / max(n, 1),
        "loss_cls": sum_loss_cls / max(n, 1),
        "seg_dice": sum_dice / max(n, 1),
        "cls_acc": sum_acc / max(n, 1),
        "cls_auc": float(auc) if auc is not None else float("nan"),
        "cls_auprc": float(prc) if prc is not None else float("nan"),
    }


# ============================================================
# 2) 主程式
# ============================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--roi_shape", type=str, default="128,128,64", help="D,H,W 例如 128,128,64 或 none")
    p.add_argument("--bbox_margin", type=str, default="8,16,16", help="z,y,x margin，例如 8,16,16")
    p.add_argument("--normalize", type=str, default="zscore", choices=["zscore", "minmax", "none"])

    # train
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="啟用 AMP（建議）")
    p.add_argument("--grad_clip", type=float, default=12.0)

    # model
    p.add_argument("--n_filters", type=int, default=16)
    p.add_argument("--norm", type=str, default="batchnorm", choices=["none", "batchnorm", "groupnorm", "instancenorm"])
    p.add_argument("--dropout", action="store_true")

    # iterative config
    p.add_argument("--n_iter", type=int, default=2)
    p.add_argument("--lambda_cls", type=float, default=0.3)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--detach_probmap", action="store_true")
    p.add_argument("--loss_on_all_iters", action="store_true")
    p.add_argument("--no_cls_weight", action="store_true", help="不要在分類 loss 中使用類別權重")
    p.add_argument("--cls_on_orig", action="store_true", help="迭代時分類只使用原始 volume，不加上前一次的 probability map")
    p.add_argument("--no_norm", action="store_true", help="分類頭不使用 L2 正規化，直接用 GAP")
    p.add_argument("--larger_cls", action="store_true", help="使用更大的分類頭 (容量較高)")

    # debugging / logging
    p.add_argument("--debug_cls_grad", action="store_true", help="在訓練過程中顯示分類分支梯度範數，用於排查是否有更新")

    # ckpt/log
    p.add_argument("--out_dir", type=str, default="./checkpoints")
    p.add_argument("--resume", type=str, default="", help="path to .pt")
    p.add_argument("--log_every", type=int, default=20)
    # convenience options for debugging / ablation
    p.add_argument("--freeze_backbone", action="store_true", help="凍結 VNet 編碼器權重，只訓練分類頭")
    p.add_argument("--only_cls", action="store_true", help="只計算分類 loss (等價於 --lambda_cls 1.0 並忽略 seg metrics)")

    return p


def parse_tuple_int(s: str) -> Optional[Tuple[int, int, int]]:
    if s.strip().lower() in ("none", "null", ""):
        return None
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"expect 3 ints like 128,128,128 but got {s!r}")
    return parts[0], parts[1], parts[2]


def main():
    args = build_argparser().parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    roi_shape = parse_tuple_int(args.roi_shape)
    bbox_margin = parse_tuple_int(args.bbox_margin) or (0, 0, 0)

    # ------------------------
    # Dataset / Loader
    # ------------------------
    train_ds = ABUSLocalSegClsDataset(ABUSLocalConfig(
        root=args.data_root,
        split="train",
        normalize=args.normalize,
        bbox_margin=bbox_margin,
        roi_shape=roi_shape,
        return_meta=False,
    ))
    val_ds = ABUSLocalSegClsDataset(ABUSLocalConfig(
        root=args.data_root,
        split="val",
        normalize=args.normalize,
        bbox_margin=bbox_margin,
        roi_shape=roi_shape,
        return_meta=False,
    ))


    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # val 建議 1，避免 shape/記憶體問題
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # benign/malignant 數量（給 weighted focal loss 用）
    counts = count_cls_labels(train_ds)
    nn_ = counts["benign"]
    nm = counts["malignant"]
    print(f"train label counts: benign={nn_} malignant={nm}")
    # 檢查驗證集的 class distribution，AUC 計算需要兩類樣本都存在
    val_counts = count_cls_labels(val_ds)
    print(f"val   label counts: benign={val_counts['benign']} malignant={val_counts['malignant']}")

    # ------------------------
    # Model
    # ------------------------
    vnet = VNet(
        n_channels=1,
        n_classes=2,
        n_filters=args.n_filters,
        normalization=args.norm,
        has_dropout=bool(args.dropout),
    )
    # 你的 VNet encoder features: x3=4f, x4=8f, x5=16f
    stage_channels = [4 * args.n_filters, 8 * args.n_filters, 16 * args.n_filters]
    
    # 選擇分類頭大小
    if args.larger_cls:
        from models.cmsvnet_iter import MultiScaleClassifierLarge
        model = CMSVNet(
            vnet_backbone=vnet, 
            stage_channels=stage_channels, 
            cls_hidden=512,
            use_norm=not bool(args.no_norm),
            classifier_cls=MultiScaleClassifierLarge
        ).to(device)
    else:
        model = CMSVNet(
            vnet_backbone=vnet, 
            stage_channels=stage_channels, 
            cls_hidden=256, 
            use_norm=not bool(args.no_norm)
        ).to(device)

    # ------------------------
    # Optimizer / Scaler
    # ------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None
    use_amp = (scaler is not None)

    # ------------------------
    # Iter config（你論文方法的核心）
    # ------------------------
    it_cfg = IterConfig(
        n_iter=args.n_iter,
        lambda_cls=args.lambda_cls,
        focal_gamma=args.focal_gamma,
        detach_probmap=bool(args.detach_probmap),
        loss_on_all_iters=bool(args.loss_on_all_iters),
        use_cls_weights=not bool(args.no_cls_weight),
    )
    print("IterConfig =", asdict(it_cfg))

    # freeze backbone if requested (useful for pretraining classifier)
    if args.freeze_backbone:
        print("Freezing VNet backbone parameters")
        for p in model.vnet.parameters():
            p.requires_grad = False

    # if only_cls, force lambda_cls=1.0 and ignore seg-related metrics later
    if args.only_cls:
        it_cfg.lambda_cls = 1.0
        print("Only classification loss (lambda_cls set to 1.0)")

    # ------------------------
    # Resume
    # ------------------------
    start_epoch = 1
    best_joint = float("inf")
    best_auc = -float("inf")
    best_dice = -float("inf")

    ensure_dir(args.out_dir)
    run_cfg_path = os.path.join(args.out_dir, "run_config.json")
    with open(run_cfg_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    if args.resume:
        ckpt = load_ckpt(args.resume, model, optimizer=optimizer, scaler=scaler)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_joint = float(ckpt.get("best_joint", best_joint))
        best_auc = float(ckpt.get("best_auc", best_auc))
        best_dice = float(ckpt.get("best_dice", best_dice))
        print(f"Resumed from {args.resume} (start_epoch={start_epoch})")

    # ------------------------
    # Training loop
    # ------------------------
    history = {
        'train_loss_joint': [],
        'train_loss_seg': [],
        'train_loss_cls': [],
        'val_loss_joint': [],
        'val_loss_seg': [],
        'val_loss_cls': [],
        'train_seg_dice': [],
        'val_seg_dice': [],
        'train_cls_acc': [],
        'val_cls_acc': [],
        'val_cls_auc': [],
        'val_cls_auprc': [],
    }
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n========== Epoch {epoch}/{args.epochs} ==========")

        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            it_cfg=it_cfg,
            nm=nm,
            nn_=nn_,
            use_amp=use_amp,
            grad_clip_norm=args.grad_clip,
            log_every=args.log_every,
            debug_cls_grad=args.debug_cls_grad,
            cls_on_orig=args.cls_on_orig,
        )
        va = validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            it_cfg=it_cfg,
            nm=nm,
            nn_=nn_,
            use_amp=use_amp,
            cls_on_orig=args.cls_on_orig,
        )

        print(
            f"[train] loss_joint={tr['loss_joint']:.4f} loss_seg={tr['loss_seg']:.4f} loss_cls={tr['loss_cls']:.4f} | "
            f"dice={tr['seg_dice']:.4f} acc={tr['cls_acc']:.4f} | {tr['sec']:.1f}s"
        )
        print(
            f"[ val ] loss_joint={va['loss_joint']:.4f} loss_seg={va['loss_seg']:.4f} loss_cls={va['loss_cls']:.4f} | "
            f"dice={va['seg_dice']:.4f} acc={va['cls_acc']:.4f} auc={va['cls_auc']:.4f} auprc={va['cls_auprc']:.4f}"
        )

        # Record history for plotting
        history['train_loss_joint'].append(tr['loss_joint'])
        history['train_loss_seg'].append(tr['loss_seg'])
        history['train_loss_cls'].append(tr['loss_cls'])
        history['val_loss_joint'].append(va['loss_joint'])

        history['val_loss_seg'].append(va['loss_seg'])
        history['val_loss_cls'].append(va['loss_cls'])
        history['train_seg_dice'].append(tr['seg_dice'])
        history['val_seg_dice'].append(va['seg_dice'])
        history['train_cls_acc'].append(tr['cls_acc'])
        history['val_cls_acc'].append(va['cls_acc'])
        history['val_cls_auc'].append(va['cls_auc'])
        history['val_cls_auprc'].append(va['cls_auprc'])

        # ------------------------
        # Save ckpt
        # ------------------------
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "args": vars(args),
            "iter_cfg": asdict(it_cfg),
            "label_counts": {"benign": nn_, "malignant": nm},
            "train": tr,
            "val": va,
            "best_joint": best_joint,
            "best_auc": best_auc,
            "best_dice": best_dice,
        }

        # last
        save_ckpt(os.path.join(args.out_dir, "last.pt"), state)

        # best by joint loss (越小越好)
        if va["loss_joint"] < best_joint:
            best_joint = va["loss_joint"]
            state["best_joint"] = best_joint
            save_ckpt(os.path.join(args.out_dir, "best_joint.pt"), state)
            print(f"  ✓ saved best_joint.pt (best_joint={best_joint:.4f})")

        # best by cls AUC (越大越好；若 nan 就跳過)
        if not math.isnan(va["cls_auc"]) and va["cls_auc"] > best_auc:
            best_auc = va["cls_auc"]
            state["best_auc"] = best_auc
            save_ckpt(os.path.join(args.out_dir, "best_clsauc.pt"), state)
            print(f"  ✓ saved best_clsauc.pt (best_auc={best_auc:.4f})")

        # best by seg dice (越大越好)
        if va["seg_dice"] > best_dice:
            best_dice = va["seg_dice"]
            state["best_dice"] = best_dice
            save_ckpt(os.path.join(args.out_dir, "best_segdice.pt"), state)
            print(f"  ✓ saved best_segdice.pt (best_dice={best_dice:.4f})")

    print("\nTraining finished.")
    print(f"Best joint loss: {best_joint:.4f}")
    print(f"Best cls AUC  : {best_auc:.4f}")
    print(f"Best seg dice : {best_dice:.4f}")
    
    # 生成訓練曲線圖
    print("\n" + "="*60)
    print("Generating training curves...")
    plot_training_curves(history, args.out_dir)
    print("="*60)


if __name__ == "__main__":
    main()
