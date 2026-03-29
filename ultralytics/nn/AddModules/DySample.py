import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    """DySample: 轻量级可学习动态上采样模块。

    通过学习采样偏移量实现动态上采样，替代nn.Upsample(nearest)。
    输入 [B, c1, H, W] -> 生成偏移 -> grid_sample -> [B, c1, scale*H, scale*W]
    """

    def __init__(self, in_channels, scale_factor=2, groups=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.groups = groups

        # 偏移量生成网络: 为每个输出像素预测(dx, dy)偏移
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // groups, 1, bias=False),
            nn.BatchNorm2d(in_channels // groups),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // groups, 2 * scale_factor * scale_factor, 1, bias=False),
        )
        # 初始化偏移为0 (退化为最近邻)
        nn.init.zeros_(self.offset_conv[-1].weight)

    def forward(self, x):
        B, C, H, W = x.shape
        sH, sW = H * self.scale_factor, W * self.scale_factor

        # 生成偏移 [B, 2*s*s, H, W]
        offset = self.offset_conv(x)
        # reshape -> [B, 2, sH, sW]
        offset = offset.view(B, 2, self.scale_factor, self.scale_factor, H, W)
        offset = offset.permute(0, 1, 4, 2, 5, 3).contiguous()  # [B, 2, H, s, W, s]
        offset = offset.view(B, 2, sH, sW)

        # 构建基础网格 (归一化坐标 [-1, 1])
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, sH, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, sW, device=x.device, dtype=x.dtype),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # [1, 2, sH, sW]

        # 将偏移缩放到合理范围 (每个像素最多偏移1个像素)
        offset_scale_x = 2.0 / sW
        offset_scale_y = 2.0 / sH
        offset = offset * torch.tensor([offset_scale_x, offset_scale_y],
                                        device=x.device, dtype=x.dtype).view(1, 2, 1, 1)

        # 最终采样网格
        grid = base_grid + offset  # [B, 2, sH, sW]
        grid = grid.permute(0, 2, 3, 1)  # [B, sH, sW, 2]

        # 双线性插值采样
        out = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return out
