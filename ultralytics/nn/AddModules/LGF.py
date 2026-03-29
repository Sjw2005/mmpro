import torch
import torch.nn as nn


class LGF(nn.Module):
    """Local-Global Fusion placeholder (与LAEF共用parse_model通道处理)。"""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        a, b = x[0], x[1]
        return self.conv(torch.cat([a, b], dim=1))
