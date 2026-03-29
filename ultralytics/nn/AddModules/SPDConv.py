import torch
import torch.nn as nn


class SPDConv(nn.Module):
    """SPD-Conv: PixelUnshuffle(2) + Conv1x1, 替代stride-2 Conv实现零信息丢失下采样。

    输入 [B, c1, H, W] -> PixelUnshuffle(2) -> [B, 4*c1, H/2, W/2] -> Conv1x1 -> [B, c2, H/2, W/2]
    """

    def __init__(self, c1, c2):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2)
        self.conv = nn.Conv2d(4 * c1, c2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(self.pixel_unshuffle(x))))
