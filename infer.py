# infer_and_export_nrrd.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import argparse
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# 讀寫 NRRD：優先 pynrrd，沒有就用 SimpleITK
try:
    import nrrd  # pip install pynrrd
except Exception:
    nrrd = None

try:
    import SimpleITK as sitk  # pip install SimpleITK
except Exception:
    sitk = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.vnet import VNet
from models.cmsvnet_iter import CMSVNet  # 你的 forward(x)->(seg_logits, cls_logits, prob_list)


# -------------------------
# I/O
# -------------------------
def read_nrrd(path: str) -> np.ndarray:
    if nrrd is not None:
        arr, _ = nrrd.read(path)
        return np.asarray(arr)
    if sitk is not None:
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)  # 常見 [D,H,W]
        return np.asarray(arr)
    raise ImportError("請安裝 pynrrd 或 SimpleITK 才能讀 nrrd")

def write_nrrd(path: str, arr: np.ndarray, ref_header_path: Optional[str] = None) -> None:
    """
    盡量保留 header（如果 ref_header_path 提供且使用 pynrrd）。
    若用 SimpleITK，會寫基本 nrrd，但不保證完整 header 一致。
    """
    arr = np.asarray(arr)
    if nrrd is not None:
        header = None
        if ref_header_path is not None and os.path.isfile(ref_header_path):
            _a, header = nrrd.read(ref_header_path)
        nrrd.write(path, arr, header=header)
        return
    if sitk is not None:
        img = sitk.GetImageFromArray(arr)
        sitk.WriteImage(img, path)
        return
    raise ImportError("請安裝 pynrrd 或 SimpleITK 才能寫 nrrd")


# -------------------------
# Preprocess (同你 dataset 的核心邏輯)
# -------------------------
def normalize(vol: np.ndarray, method: str = "zscore", eps: float = 1e-6) -> np.ndarray:
    v = vol.astype(np.float32, copy=False)
    if method == "none":
        return v
    if method == "zscore":
        m = float(v.mean())
        s = float(v.std())  # SD
        return (v - m) / (s + eps)
    if method == "minmax":
        vmin = float(v.min())
        vmax = float(v.max())
        return (v - vmin) / (vmax - vmin + eps)
    raise ValueError(method)

def compute_bbox_from_mask(msk: np.ndarray):
    idx = np.argwhere(msk > 0)
    if idx.size == 0:
        return None
    z0, y0, x0 = idx.min(axis=0)
    z1, y1, x1 = idx.max(axis=0) + 1
    return int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)

def clip_bbox(b, shape):
    D, H, W = shape
    z0, z1, y0, y1, x0, x1 = b
    z0 = max(0, min(z0, D)); z1 = max(0, min(z1, D))
    y0 = max(0, min(y0, H)); y1 = max(0, min(y1, H))
    x0 = max(0, min(x0, W)); x1 = max(0, min(x1, W))
    if z1 <= z0: z1 = min(D, z0 + 1)
    if y1 <= y0: y1 = min(H, y0 + 1)
    if x1 <= x0: x1 = min(W, x0 + 1)
    return z0, z1, y0, y1, x0, x1

def expand_bbox(b, margin_zyx, shape):
    mz, my, mx = margin_zyx
    z0, z1, y0, y1, x0, x1 = b
    return clip_bbox((z0 - mz, z1 + mz, y0 - my, y1 + my, x0 - mx, x1 + mx), shape)

def crop(vol: np.ndarray, b):
    z0, z1, y0, y1, x0, x1 = b
    return vol[z0:z1, y0:y1, x0:x1]

def center_crop_or_pad(vol: np.ndarray, out_shape: Tuple[int, int, int], pad_value: float = 0.0) -> np.ndarray:
    D, H, W = vol.shape
    od, oh, ow = out_shape
    cz, cy, cx = D // 2, H // 2, W // 2
    z0 = cz - od // 2
    y0 = cy - oh // 2
    x0 = cx - ow // 2
    z1 = z0 + od
    y1 = y0 + oh
    x1 = x0 + ow

    out = np.full((od, oh, ow), pad_value, dtype=vol.dtype)

    src_z0 = max(0, z0); src_y0 = max(0, y0); src_x0 = max(0, x0)
    src_z1 = min(D, z1); src_y1 = min(H, y1); src_x1 = min(W, x1)

    dst_z0 = src_z0 - z0; dst_y0 = src_y0 - y0; dst_x0 = src_x0 - x0
    dst_z1 = dst_z0 + (src_z1 - src_z0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    out[dst_z0:dst_z1, dst_y0:dst_y1, dst_x0:dst_x1] = vol[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]
    return out


# -------------------------
# Metrics
# -------------------------
def dice_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = float(np.sum(pred * gt))
    p = float(np.sum(pred))
    g = float(np.sum(gt))
    union = float(p + g - inter)
    dsc = (2 * inter) / (p + g + eps)
    ji = inter / (union + eps)
    return float(dsc), float(ji)


# -------------------------
# Outline (框線) 產生：用 6-neighborhood erosion 的 XOR
# -------------------------
def binary_erosion6(vol: np.ndarray, iters: int = 1) -> np.ndarray:
    """
    不依賴 scipy。用 6 邻域近似 erosion（保守）。
    vol: uint8 {0,1}
    """
    v = (vol > 0).astype(np.uint8)
    for _ in range(iters):
        c = v.copy()
        # 內縮：一個 voxel 要保留，必須自己與 6 方向鄰居都為 1
        er = c.copy()
        er[1:, :, :] &= c[:-1, :, :]
        er[:-1, :, :] &= c[1:, :, :]
        er[:, 1:, :] &= c[:, :-1, :]
        er[:, :-1, :] &= c[:, 1:, :]
        er[:, :, 1:] &= c[:, :, :-1]
        er[:, :, :-1] &= c[:, :, 1:]
        v = er
    return v

def outline_from_mask(msk: np.ndarray, erode_iters: int = 1) -> np.ndarray:
    m = (msk > 0).astype(np.uint8)
    er = binary_erosion6(m, iters=erode_iters)
    edge = (m ^ er).astype(np.uint8)  # XOR => 邊界
    return edge


# -------------------------
# Visualization PNG
# -------------------------
def save_overlay_png(
    img: np.ndarray,
    gt_edge: np.ndarray,
    pred_edge: np.ndarray,
    out_path: str,
    z_list: Optional[list] = None,
):
    """
    產生幾張切片的 overlay（灰階底，GT=綠框，Pred=紅框）
    """
    D = img.shape[0]
    if z_list is None:
        z_list = [D // 4, D // 2, (3 * D) // 4]
    z_list = [int(np.clip(z, 0, D - 1)) for z in z_list]

    n = len(z_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, z in zip(axes, z_list):
        sl = img[z]
        # normalize slice for display
        s = sl - sl.min()
        if s.max() > 0:
            s = s / s.max()

        ax.imshow(s, cmap="gray")

        # overlay edges
        g = gt_edge[z] > 0
        p = pred_edge[z] > 0

        # 畫框線：用 scatter 最穩（不用額外套件）
        gy, gx = np.where(g)
        py, px = np.where(p)

        if gy.size > 0:
            ax.scatter(gx, gy, s=1, c="lime", alpha=0.9, label="GT edge")
        if py.size > 0:
            ax.scatter(px, py, s=1, c="red", alpha=0.9, label="Pred edge")

        ax.set_title(f"z={z}")
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Build model (同 train.py)
# -------------------------
def build_model(n_filters: int, norm: str, dropout: bool, device: torch.device) -> CMSVNet:
    vnet = VNet(
        n_channels=1,
        n_classes=2,
        n_filters=n_filters,
        normalization=norm,
        has_dropout=bool(dropout),
    )
    stage_channels = [4 * n_filters, 8 * n_filters, 16 * n_filters]
    # default head size; caller can override later if needed
    model = CMSVNet(vnet_backbone=vnet, stage_channels=stage_channels, cls_hidden=256).to(device)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="例如 ./checkpoints/best_joint.pt")
    ap.add_argument("--image", type=str, required=True, help="image.nrrd")
    ap.add_argument("--gt", type=str, required=True, help="gt mask.nrrd (0/1)")
    ap.add_argument("--out_dir", type=str, default="./infer_outputs")
    ap.add_argument("--out_prefix", type=str, default="case")

    # must match training
    ap.add_argument("--roi_shape", type=str, default="128,128,64")
    ap.add_argument("--bbox_margin", type=str, default="8,16,16")
    ap.add_argument("--normalize", type=str, default="zscore", choices=["zscore", "minmax", "none"])
    ap.add_argument("--n_filters", type=int, default=16)
    ap.add_argument("--norm", type=str, default="batchnorm", choices=["none", "batchnorm", "groupnorm", "instancenorm"])
    ap.add_argument("--dropout", action="store_true")
    # classifier options (must match train.py)
    ap.add_argument("--no_norm", action="store_true", help="disable normalization inside classifier head")
    ap.add_argument("--larger_cls", action="store_true", help="use larger classification head")

    # outline settings
    ap.add_argument("--erode_iters", type=int, default=1, help="框線粗細（越大越粗）")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    roi_shape = tuple(int(x.strip()) for x in args.roi_shape.split(","))
    bbox_margin = tuple(int(x.strip()) for x in args.bbox_margin.split(","))

    # load image/gt
    img_full = read_nrrd(args.image)
    gt_full = read_nrrd(args.gt)

    # squeeze possible [1,D,H,W]
    if img_full.ndim == 4 and img_full.shape[0] == 1:
        img_full = img_full[0]
    if gt_full.ndim == 4 and gt_full.shape[0] == 1:
        gt_full = gt_full[0]

    if img_full.shape != gt_full.shape:
        raise ValueError(f"image shape {img_full.shape} != gt shape {gt_full.shape}")

    gt_bin = (gt_full > 0).astype(np.uint8)

    # ROI crop by GT bbox (跟你訓練一致)
    bbox = compute_bbox_from_mask(gt_bin)
    used_bbox = None
    img = img_full
    gt = gt_bin
    if bbox is not None:
        bbox = expand_bbox(clip_bbox(bbox, img_full.shape), bbox_margin, img_full.shape)
        used_bbox = bbox
        img = crop(img_full, bbox)
        gt = crop(gt_bin, bbox)

    # fixed roi
    img = center_crop_or_pad(img, roi_shape, pad_value=0.0)
    gt = center_crop_or_pad(gt, roi_shape, pad_value=0.0)

    # normalize
    img_norm = normalize(img, args.normalize)

    # model (mirror train.py choices)
    vnet = VNet(
        n_channels=1,
        n_classes=2,
        n_filters=args.n_filters,
        normalization=args.norm,
        has_dropout=bool(args.dropout),
    )
    stage_channels = [4 * args.n_filters, 8 * args.n_filters, 16 * args.n_filters]
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

    ckpt = torch.load(args.ckpt, map_location="cpu")
    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError as e:
        print("Warning: strict load_state_dict failed; retrying with strict=False.")
        print(e)
        model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # inference
    x = torch.from_numpy(img_norm).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,D,H,W]
    with torch.no_grad():
        seg_logits, cls_logits, prob_list = model(x)

    prob_malig = float(torch.softmax(cls_logits, dim=1)[0, 1].item())
    pred = seg_logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)  # [D,H,W]

    # metrics
    dsc, ji = dice_iou(pred, gt)
    print("===== Inference result =====")
    print(f"malignant_prob = {prob_malig:.6f}")
    print(f"DSC = {dsc:.6f}")
    print(f"JI  = {ji:.6f}")
    print(f"used_bbox_from_gt = {used_bbox}")
    print("============================")

    # outlines
    gt_edge = outline_from_mask(gt, erode_iters=args.erode_iters)
    pred_edge = outline_from_mask(pred, erode_iters=args.erode_iters)

    # label volume: 0=bg, 1=GT edge, 2=Pred edge, 3=overlap
    vis = np.zeros_like(gt_edge, dtype=np.uint8)
    vis[gt_edge > 0] = 1
    vis[pred_edge > 0] = np.maximum(vis[pred_edge > 0], 2)
    vis[(gt_edge > 0) & (pred_edge > 0)] = 3

    # save nrrd
    out_nrrd = os.path.join(args.out_dir, f"{args.out_prefix}_edges_vis.nrrd")
    write_nrrd(out_nrrd, vis, ref_header_path=args.image)
    print(f"✓ saved edges visualization nrrd: {out_nrrd}")
    print("  label meaning: 0=bg, 1=GT_edge, 2=Pred_edge, 3=Overlap_edge")

    # save pred/gt mask too (optional but useful)
    out_pred = os.path.join(args.out_dir, f"{args.out_prefix}_pred_mask.nrrd")
    out_gt = os.path.join(args.out_dir, f"{args.out_prefix}_gt_mask.nrrd")
    write_nrrd(out_pred, pred.astype(np.uint8), ref_header_path=args.image)
    write_nrrd(out_gt, gt.astype(np.uint8), ref_header_path=args.image)
    print(f"✓ saved pred mask nrrd: {out_pred}")
    print(f"✓ saved gt mask nrrd  : {out_gt}")

    # save png overlay
    out_png = os.path.join(args.out_dir, f"{args.out_prefix}_overlay.png")
    save_overlay_png(img_norm, gt_edge, pred_edge, out_png)
    print(f"✓ saved overlay png: {out_png}")


if __name__ == "__main__":
    main()