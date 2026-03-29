import torch
import torch.nn as nn
import torch.nn.functional as F


class MDAF(nn.Module):
    """Multi-Dimensional Attention Fusion placeholder (与MDAFP共用parse_model通道处理)。"""

    def __init__(self, dim, num_heads=4, bias_type='WithBias', attn_sizes=None, dropout=0.0, last_stage=False):
        super().__init__()
        self.dim = dim
        if attn_sizes is None:
            attn_sizes = [3, 5, 7]
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        a, b = x[0], x[1]
        return self.fusion(torch.cat([a, b], dim=1))


class MDAFP3D(nn.Module):
    """
    三维注意力跨模态融合模块 (3-Dimensional Attention Fusion):
      1) Channel Attention: 对每个模态做SE通道注意力，学习通道维度上的模态可靠性
      2) Spatial Attention:  跨模态空间注意力图，学习空间维度上哪个模态更重要
      3) Cross-Modal Interaction: 轻量跨模态交叉注意力，语义级信息交换
    最终通过自适应门控融合两模态特征。
    """

    def __init__(self, dim, num_heads=4, bias_type='WithBias', attn_sizes=None, dropout=0.0, last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        r = max(dim // 16, 4)  # SE reduction ratio

        # ========== 1) Channel Attention (SE-style, 每个模态独立) ==========
        self.ca_x = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, r, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(r, dim, 1, bias=True),
            nn.Sigmoid()
        )
        self.ca_y = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, r, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(r, dim, 1, bias=True),
            nn.Sigmoid()
        )

        # ========== 2) Spatial Attention (跨模态空间注意力) ==========
        # 对两模态concat后生成两张空间注意力图，分别加权两个模态
        self.sa = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 2, 1, bias=False),
            nn.BatchNorm2d(dim // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim // 2, dim // 2, 7, padding=3, groups=dim // 2, bias=False),
            nn.BatchNorm2d(dim // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim // 2, 2, 1, bias=True),  # 2通道: 分别给x,y
            nn.Sigmoid()
        )

        # ========== 3) Cross-Modal Interaction (轻量交叉注意力) ==========
        # 将特征投影到低维空间做 cross-attention
        self.head_dim = max(dim // num_heads, 8)
        inner_dim = self.head_dim * num_heads
        self.q_proj = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.cross_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.cross_scale = nn.Parameter(torch.tensor(0.1))  # 初始小权重，稳定训练

        # ========== 4) 自适应融合门控 ==========
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.Sigmoid()
        )

        # ========== 5) 输出投影 ==========
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

    def _cross_attn(self, q_feat, kv_feat):
        """轻量跨模态交叉注意力: q来自一个模态, k/v来自另一个模态"""
        B, C, H, W = q_feat.shape
        # 投影
        q = self.q_proj(q_feat)   # [B, inner, H, W]
        k = self.k_proj(kv_feat)
        v = self.v_proj(kv_feat)

        # Reshape to [B, heads, head_dim, H*W]
        hd = self.head_dim
        nh = self.num_heads
        q = q.view(B, nh, hd, H * W)
        k = k.view(B, nh, hd, H * W)
        v = v.view(B, nh, hd, H * W)

        # 注意力: 在通道维度做(不是空间维度)，复杂度 O(hd^2) 而非 O((HW)^2)
        # 这是 Linear Attention 的思路，适合高分辨率特征图
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        # [B, nh, hd, hd] = q * k^T (在空间维度聚合)
        attn = torch.matmul(q, k.transpose(-2, -1))  # [B, nh, hd, hd]
        attn = attn.softmax(dim=-1)
        # [B, nh, hd, HW] = attn * v
        out = torch.matmul(attn, v)  # [B, nh, hd, HW]
        out = out.view(B, nh * hd, H, W)
        return self.cross_out(out)

    def forward(self, data):
        x, y = data  # x: 模态A, y: 模态B
        x0, y0 = x, y  # 保留残差

        # === Dim1: Channel Attention ===
        x_ca = x * self.ca_x(x)  # 通道加权
        y_ca = y * self.ca_y(y)

        # === Dim2: Spatial Attention ===
        sa_map = self.sa(torch.cat([x_ca, y_ca], dim=1))  # [B, 2, H, W]
        x_sa = x_ca * sa_map[:, 0:1, :, :]  # 空间加权
        y_sa = y_ca * sa_map[:, 1:2, :, :]

        # === Dim3: Cross-Modal Interaction ===
        x_cross = x_sa + self.cross_scale * self._cross_attn(x_sa, y_sa)  # x query, y key/value
        y_cross = y_sa + self.cross_scale * self._cross_attn(y_sa, x_sa)  # y query, x key/value

        # === Adaptive Gate Fusion ===
        g = self.gate(torch.cat([x_cross, y_cross], dim=1))  # [B, dim, H, W]
        fused = g * x_cross + (1.0 - g) * y_cross

        # === Output with residual ===
        out = self.out_proj(fused)
        return out + 0.5 * (x0 + y0)  # 残差保护
