# test.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import csv
import json
import math
import argparse
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.vnet import VNet
from models.cmsvnet_iter import CMSVNet  # 你的 forward(x) -> (seg_logits, cls_logits, prob_list)
from data.dataset_abus import ABUSLocalConfig, ABUSLocalSegClsDataset
from utils.metrics import auroc, auprc  # 你 train.py 也有用


# -------------------------
# utilities
# -------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_tuple_int(s: str) -> Optional[Tuple[int, int, int]]:
    if s.strip().lower() in ("none", "null", ""):
        return None
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"expect 3 ints like 128,128,64 but got {s!r}")
    return parts[0], parts[1], parts[2]


def load_ckpt(path: str, model: torch.nn.Module) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError as e:
        # if shapes mismatch/extra keys, try again non-strict and warn user
        print("Warning: strict load_state_dict failed; retrying with strict=False.")
        print("Original error:\n", e)
        model.load_state_dict(ckpt["model"], strict=False)
    return ckpt


# -------------------------
# Segmentation metrics
# -------------------------
def dice_iou_from_pred_gt(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> Tuple[float, float]:
    """
    pred, gt: binary {0,1} arrays, same shape [D,H,W] 或任意
    """
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    inter = float(np.sum(pred * gt))
    p = float(np.sum(pred))
    g = float(np.sum(gt))
    union = float(p + g - inter)

    dice = (2.0 * inter) / (p + g + eps)
    iou = inter / (union + eps)
    return dice, iou


# -------------------------
# Classification metrics
# -------------------------
def confusion_counts(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
    """
    return TP, FP, TN, FN with positive=1 (malignant)
    """
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 0 and p == 0:
            tn += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            raise ValueError(f"Unexpected labels: t={t}, p={p}")
    return tp, fp, tn, fn


def safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a) / float(b + eps)


def cls_metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    acc = safe_div(tp + tn, tp + fp + tn + fn)
    precision = safe_div(tp, tp + fp)  # PRC 我這裡用 Precision
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    fpr = safe_div(fp, fp + tn)
    return {
        "ACC": acc,
        "PRC": precision,  # Precision
        "FPR": fpr,
        "F1": f1,
        "REC": recall,
    }


@torch.no_grad()
def run_test(
    model: CMSVNet,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    thr: float = 0.5,
    save_pred_dir: str = "",
) -> Dict[str, Any]:
    model.eval()

    # segmentation aggregates
    dice_list: List[float] = []
    iou_list: List[float] = []

    # classification aggregates
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    y_score_all: List[float] = []

    per_case_rows = []

    if save_pred_dir:
        ensure_dir(save_pred_dir)

    for idx, batch in enumerate(loader):
        if len(batch) == 3:
            x, seg_gt, y_cls = batch
            meta = None
        else:
            x, seg_gt, y_cls, meta = batch

        x = x.to(device, non_blocking=True).float()
        seg_gt = seg_gt.to(device, non_blocking=True).long()
        y_cls = y_cls.to(device, non_blocking=True).long().view(-1)

        # forward：你的 signature 是 forward(x)->(seg_logits, cls_logits, prob_list)
        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                seg_logits, cls_logits, prob_list = model(x)
        else:
            seg_logits, cls_logits, prob_list = model(x)
    
        # cls prob + pred
        prob = torch.softmax(cls_logits, dim=1)[:, 1]  # malignant prob
        pred_cls = (prob >= thr).long()

        # seg pred
        seg_pred = seg_logits.argmax(dim=1)  # [B,D,H,W]

        # 這裡 DataLoader 通常 batch=1，仍保守寫成逐筆
        bsz = x.size(0)
        for b in range(bsz):
            # seg metrics
            pred_np = seg_pred[b].detach().cpu().numpy().astype(np.uint8)
            gt_np = seg_gt[b].detach().cpu().numpy().astype(np.uint8)
            dsc, ji = dice_iou_from_pred_gt(pred_np, gt_np)

            dice_list.append(float(dsc))
            iou_list.append(float(ji))

            # cls metrics
            t = int(y_cls[b].item())
            p = int(pred_cls[b].item())
            s = float(prob[b].item())

            y_true_all.append(t)
            y_pred_all.append(p)
            y_score_all.append(s)

            # meta id（如果你的 dataset return_meta=False，這裡就用 index）
            case_id = None
            if isinstance(meta, dict) and "id" in meta:
                case_id = str(meta["id"])
            elif isinstance(meta, (list, tuple)) and len(meta) > b and isinstance(meta[b], dict) and "id" in meta[b]:
                case_id = str(meta[b]["id"])
            else:
                case_id = f"sample_{idx:04d}_b{b}"

            per_case_rows.append({
                "case_id": case_id,
                "DSC": float(dsc),
                "JI": float(ji),
                "y_true": t,
                "y_prob": s,
                "y_pred": p,
            })

            # optional: save pred mask
            if save_pred_dir:
                np.savez_compressed(
                    os.path.join(save_pred_dir, f"{case_id}.npz"),
                    seg_pred=pred_np,  # 0/1
                    seg_gt=gt_np,
                    y_true=t,
                    y_prob=s,
                    y_pred=p,
                )

    # overall cls metrics
    tp, fp, tn, fn = confusion_counts(y_true_all, y_pred_all)
    cls_m = cls_metrics_from_counts(tp, fp, tn, fn)

    # overall AUC/AUPRC（可能會是 None -> nan）
    auc = auroc(y_true_all, y_score_all)
    prc_auc = auprc(y_true_all, y_score_all)

    out = {
        "N": len(per_case_rows),

        # Seg mean
        "DSC_mean": float(np.mean(dice_list)) if dice_list else float("nan"),
        "DSC_SD": float(np.std(dice_list)) if dice_list else float("nan"),  # SD
        "JI_mean": float(np.mean(iou_list)) if iou_list else float("nan"),
        "JI_SD": float(np.std(iou_list)) if iou_list else float("nan"),

        # Cls
        "ACC": cls_m["ACC"],
        "PRC": cls_m["PRC"],  # Precision
        "FPR": cls_m["FPR"],
        "F1": cls_m["F1"],
        "REC": cls_m["REC"],

        # Confusion
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,

        # Extra
        "AUC": float(auc) if auc is not None else float("nan"),
        "AUPRC": float(prc_auc) if prc_auc is not None else float("nan"),
    }

    return out, per_case_rows


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # ckpt & io
    p.add_argument("--ckpt", type=str, required=True, help="例如 ./checkpoints/best_joint.pt")
    p.add_argument("--out_dir", type=str, default="./test_outputs")
    p.add_argument("--save_pred", action="store_true", help="存每筆預測到 out_dir/preds_npz/")

    # data (must match training preprocessing)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--roi_shape", type=str, default="128,128,64")
    p.add_argument("--bbox_margin", type=str, default="8,16,16")
    p.add_argument("--normalize", type=str, default="zscore", choices=["zscore", "minmax", "none"])

    # model (must match training)
    p.add_argument("--n_filters", type=int, default=16)
    p.add_argument("--norm", type=str, default="batchnorm", choices=["none", "batchnorm", "groupnorm", "instancenorm"])
    p.add_argument("--dropout", action="store_true")
    # classifier options (should mirror train.py)
    p.add_argument("--no_norm", action="store_true", help="disable normalization inside classifier head")
    p.add_argument("--larger_cls", action="store_true", help="use larger classification head")

    # runtime
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--thr", type=float, default=0.5, help="classification threshold for malignant prob")

    return p


def main():
    args = build_argparser().parse_args()
    ensure_dir(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    roi_shape = parse_tuple_int(args.roi_shape)
    bbox_margin = parse_tuple_int(args.bbox_margin) or (0, 0, 0)

    # dataset: test split
    test_ds = ABUSLocalSegClsDataset(ABUSLocalConfig(
        root=args.data_root,
        split="test",
        normalize=args.normalize,
        bbox_margin=bbox_margin,
        roi_shape=roi_shape,
        return_meta=True,   # 讓 per-case csv 有 case_id
    ))
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # model init (same as train.py)
    vnet = VNet(
        n_channels=1,
        n_classes=2,
        n_filters=args.n_filters,
        normalization=args.norm,
        has_dropout=bool(args.dropout),
    )
    stage_channels = [4 * args.n_filters, 8 * args.n_filters, 16 * args.n_filters]
    # choose classifier size & normalization like train.py
    if args.larger_cls:
        from models.cmsvnet_iter import MultiScaleClassifierLarge
        model = CMSVNet(
            vnet_backbone=vnet,
            stage_channels=stage_channels,
            cls_hidden=512,
            use_norm=not bool(args.no_norm),
            classifier_cls=MultiScaleClassifierLarge,
        ).to(device)
    else:
        model = CMSVNet(
            vnet_backbone=vnet,
            stage_channels=stage_channels,
            cls_hidden=256,
            use_norm=not bool(args.no_norm),
        ).to(device)

    # load ckpt
    ckpt = load_ckpt(args.ckpt, model)
    print(f"Loaded ckpt: {args.ckpt}")
    if "epoch" in ckpt:
        print(f"  epoch = {ckpt['epoch']}")

    save_pred_dir = os.path.join(args.out_dir, "preds_npz") if args.save_pred else ""
    use_amp = bool(args.amp)

    report, per_case = run_test(
        model=model,
        loader=test_loader,
        device=device,
        use_amp=use_amp,
        thr=float(args.thr),
        save_pred_dir=save_pred_dir,
    )

    # print summary
    print("\n========== TEST SUMMARY ==========")
    print(f"N = {report['N']}")
    print(f"Seg  DSC(mean±SD) = {report['DSC_mean']:.4f} ± {report['DSC_SD']:.4f}")
    print(f"Seg  JI (mean±SD) = {report['JI_mean']:.4f} ± {report['JI_SD']:.4f}")
    print(f"Cls  ACC={report['ACC']:.4f}  PRC(Precision)={report['PRC']:.4f}  FPR={report['FPR']:.4f}  F1={report['F1']:.4f}")
    print(f"Cls  TP={report['TP']} FP={report['FP']} TN={report['TN']} FN={report['FN']}")
    print(f"Extra AUC={report['AUC']:.4f}  AUPRC={report['AUPRC']:.4f}")
    print("==================================\n")

    # save json report
    report_path = os.path.join(args.out_dir, "test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"✓ saved {report_path}")

    # save per-case csv
    csv_path = os.path.join(args.out_dir, "test_per_case.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "DSC", "JI", "y_true", "y_prob", "y_pred"])
        writer.writeheader()
        for r in per_case:
            writer.writerow(r)
    print(f"✓ saved {csv_path}")

    if save_pred_dir:
        print(f"✓ saved predictions to {save_pred_dir}")


if __name__ == "__main__":
    main()