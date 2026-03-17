from __future__ import annotations

from typing import List, Tuple, Union, Optional

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    多層 3D Conv + (Norm) + ReLU 的堆疊
    """
    def __init__(self, n_stages: int, n_filters_in: int, n_filters_out: int, normalization: str = "none"):
        super().__init__()
        ops: List[nn.Module] = []
        for i in range(n_stages):
            in_ch = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(in_ch, n_filters_out, kernel_size=3, padding=1))

            if normalization == "batchnorm":
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == "groupnorm":
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == "instancenorm":
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization == "none":
                pass
            else:
                raise ValueError(f"Unknown normalization: {normalization}")

            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualConvBlock(nn.Module):
    """
    Residual 版本的 ConvBlock（你原本就有，但目前 VNet 主體未使用）
    """
    def __init__(self, n_stages: int, n_filters_in: int, n_filters_out: int, normalization: str = "none"):
        super().__init__()
        ops: List[nn.Module] = []
        for i in range(n_stages):
            in_ch = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(in_ch, n_filters_out, kernel_size=3, padding=1))

            if normalization == "batchnorm":
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == "groupnorm":
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == "instancenorm":
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization == "none":
                pass
            else:
                raise ValueError(f"Unknown normalization: {normalization}")

            # 最後一層 conv 後不馬上 ReLU，等加完 residual 再 ReLU
            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x) + x
        return self.relu(out)


class DownsamplingConvBlock(nn.Module):
    """
    透過 stride=2 的 Conv3d 做下採樣（常見 VNet 寫法：kernel_size=stride）
    """
    def __init__(self, n_filters_in: int, n_filters_out: int, stride: int = 2, normalization: str = "none"):
        super().__init__()
        ops: List[nn.Module] = []

        # 注意：這裡 kernel_size=stride 是你原本的寫法
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=stride, stride=stride, padding=0))

        if normalization == "batchnorm":
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == "groupnorm":
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == "instancenorm":
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization == "none":
            pass
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsamplingDeconvBlock(nn.Module):
    """
    透過 ConvTranspose3d 做上採樣（常見 VNet 寫法：kernel_size=stride）
    """
    def __init__(self, n_filters_in: int, n_filters_out: int, stride: int = 2, normalization: str = "none"):
        super().__init__()
        ops: List[nn.Module] = []

        ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, kernel_size=stride, stride=stride, padding=0))

        if normalization == "batchnorm":
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == "groupnorm":
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == "instancenorm":
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization == "none":
            pass
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsampling(nn.Module):
    """
    另一種上採樣方式：trilinear upsample + conv（你原本也有，但主體目前用 deconv）
    """
    def __init__(self, n_filters_in: int, n_filters_out: int, stride: int = 2, normalization: str = "none"):
        super().__init__()
        ops: List[nn.Module] = []
        ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == "batchnorm":
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == "groupnorm":
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == "instancenorm":
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization == "none":
            pass
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class VNet(nn.Module):
    """
    你提供的 VNet（加上 return_encoder_features 支援）
    - forward(x) -> seg_logits
    - forward(x, return_encoder_features=True) -> (seg_logits, [x3, x4, x5])
      其中 [x3,x4,x5] 作為分類分支（CMS）用的多尺度 encoder features
    """
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 2,
        n_filters: int = 16,
        normalization: str = "none",
        has_dropout: bool = False,
    ):
        super().__init__()
        self.has_dropout = has_dropout
        self.n_filters = n_filters

        # Encoder
        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, 2 * n_filters, 2 * n_filters, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(2 * n_filters, 4 * n_filters, normalization=normalization)

        self.block_three = ConvBlock(3, 4 * n_filters, 4 * n_filters, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(4 * n_filters, 8 * n_filters, normalization=normalization)

        self.block_four = ConvBlock(3, 8 * n_filters, 8 * n_filters, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(8 * n_filters, 16 * n_filters, normalization=normalization)

        self.block_five = ConvBlock(3, 16 * n_filters, 16 * n_filters, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(16 * n_filters, 8 * n_filters, normalization=normalization)

        # Decoder
        self.block_six = ConvBlock(3, 8 * n_filters, 8 * n_filters, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(8 * n_filters, 4 * n_filters, normalization=normalization)

        self.block_seven = ConvBlock(3, 4 * n_filters, 4 * n_filters, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(4 * n_filters, 2 * n_filters, normalization=normalization)

        self.block_eight = ConvBlock(2, 2 * n_filters, 2 * n_filters, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(2 * n_filters, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, kernel_size=1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def encoder(self, x: torch.Tensor) -> List[torch.Tensor]:
        x1 = self.block_one(x)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        if self.has_dropout:
            x5 = self.dropout(x5)

        return [x1, x2, x3, x4, x5]

    def decoder(self, feats: List[torch.Tensor]) -> torch.Tensor:
        x1, x2, x3, x4, x5 = feats

        x5_up = self.block_five_up(x5) + x4
        x6 = self.block_six(x5_up)

        x6_up = self.block_six_up(x6) + x3
        x7 = self.block_seven(x6_up)

        x7_up = self.block_seven_up(x7) + x2
        x8 = self.block_eight(x7_up)

        x8_up = self.block_eight_up(x8) + x1
        x9 = self.block_nine(x8_up)

        if self.has_dropout:
            x9 = self.dropout(x9)

        return self.out_conv(x9)

    def forward(
        self,
        x: torch.Tensor,
        turnoff_drop: bool = False,
        return_encoder_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Args:
            x: [B, C, D, H, W]
            turnoff_drop: 若 True，forward 時暫時關掉 dropout（你原本就有）
            return_encoder_features: 若 True，回傳 (seg_logits, [x3, x4, x5])

        Returns:
            seg_logits: [B, n_classes, D, H, W]
            optionally feats_for_cls: list of 3 tensors [x3, x4, x5]
        """
        if turnoff_drop:
            old = self.has_dropout
            self.has_dropout = False

        feats = self.encoder(x)
        seg_logits = self.decoder(feats)

        if turnoff_drop:
            self.has_dropout = old

        if return_encoder_features:
            # 用最後三個 encoder 尺度作為 CMS 的多尺度特徵
            feats_for_cls = [feats[2], feats[3], feats[4]]  # x3, x4, x5
            return seg_logits, feats_for_cls

        return seg_logits


if __name__ == "__main__":
    # 最小自測：確認 forward 維度正確
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VNet(n_channels=1, n_classes=2, n_filters=16, normalization="batchnorm", has_dropout=True).to(device)

    # 用 32^3 以上比較安全（多次 /2 不會變 0）
    x = torch.randn(2, 1, 32, 32, 32, device=device)

    y = model(x)
    assert y.shape == (2, 2, 32, 32, 32), f"seg_logits shape mismatch: {y.shape}"

    y2, feats = model(x, return_encoder_features=True)
    assert y2.shape == (2, 2, 32, 32, 32)
    assert len(feats) == 3
    # x3: 4*n_filters, x4: 8*n_filters, x5: 16*n_filters
    assert feats[0].shape[1] == 4 * model.n_filters
    assert feats[1].shape[1] == 8 * model.n_filters
    assert feats[2].shape[1] == 16 * model.n_filters

    print("[OK] VNet forward & return_encoder_features shapes are correct.")