import torch
import torch.nn as nn


class LSKA(nn.Module):
    """LSKA: 大核可分离注意力模块 (Large Separable Kernel Attention)。

    使用大核(默认35x35)DW卷积分解为多个小核，提供大范围上下文信息，
    增强小目标检测能力。

    分解方式: 35x35 DW Conv -> 5x5 DW Conv + 7x1 膨胀DW Conv + 1x7 膨胀DW Conv
    """

    def __init__(self, c1, k=35):
        super().__init__()
        self.c1 = c1

        # 大核分解
        # 35 = 5 * 7, 分解为: 5x5 DW + 7x1 dilated DW (d=5) + 1x7 dilated DW (d=5)
        k0 = 5   # 基础核
        k1 = 7   # 膨胀核
        d1 = (k0 + 1) // 2  # 膨胀率 = 3 使 effective RF = 5 + (7-1)*3 = 23?
        # 更精确: k=35 -> k0=5, d1 = k // k0 // (k1//k0) 不太对
        # 直接用: 5x5 + dilated(d=5) 7x1 + dilated(d=5) 1x7 -> effective = 5 + (7-1)*5 = 35 ✓
        d1 = k // k1  # = 5

        # 空间注意力通道压缩
        self.proj_in = nn.Conv2d(c1, c1, 1, bias=False)

        # 大核DW卷积分解
        self.dw_base = nn.Conv2d(c1, c1, k0, padding=k0 // 2, groups=c1, bias=False)
        self.dw_d_h = nn.Conv2d(c1, c1, (k1, 1), padding=((k1 // 2) * d1, 0),
                                dilation=(d1, 1), groups=c1, bias=False)
        self.dw_d_w = nn.Conv2d(c1, c1, (1, k1), padding=(0, (k1 // 2) * d1),
                                dilation=(1, d1), groups=c1, bias=False)

        self.proj_out = nn.Conv2d(c1, c1, 1, bias=False)

    def forward(self, x):
        shortcut = x
        # 生成注意力权重
        attn = self.proj_in(x)
        attn = self.dw_base(attn)
        attn = self.dw_d_h(attn)
        attn = self.dw_d_w(attn)
        attn = self.proj_out(attn)
        # 注意力加权
        return shortcut * attn
