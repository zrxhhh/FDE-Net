import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from torch.nn import BatchNorm2d

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 6, 12, 18]):
        super().__init__()
        modules = []
        for rate in dilation_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(dilation_rates)*out_channels, out_channels, 1, bias=False),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def create_wavelet_filter(wave, in_size, out_size, dtype=torch.float):

    in_size = int(in_size)
    out_size = int(out_size)

    # 获取小波滤波器
    w = pywt.Wavelet(wave)

    # 创建分解滤波器
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)

    # 创建4个方向的2D滤波器
    dec_filters = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0)  # [4, 2, 2]

    # 扩展滤波器维度并重复通道
    dec_filters = dec_filters.unsqueeze(1)  # [4, 1, 2, 2]
    dec_filters = dec_filters.repeat(1, in_size, 1, 1)  # [4, in_size, 2, 2]
    dec_filters = dec_filters.reshape(4 * in_size, 1, 2, 2)  # [4*in_size, 1, 2, 2]

    # 创建重构滤波器
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype).flip(dims=[0])

    rec_filters = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
    ], dim=0)  # [4, 2, 2]

    # 扩展滤波器维度并重复通道
    rec_filters = rec_filters.unsqueeze(1)  # [4, 1, 2, 2]
    rec_filters = rec_filters.repeat(1, out_size, 1, 1)  # [4, out_size, 2, 2]
    rec_filters = rec_filters.reshape(4 * out_size, 1, 2, 2)  # [4*out_size, 1, 2, 2]

    return dec_filters, rec_filters


def wavelet_transform(x, wt_filter):
    b, c, h, w = x.shape

    if h % 2 != 0 or w % 2 != 0:
        pad_h = 0 if h % 2 == 0 else 1
        pad_w = 0 if w % 2 == 0 else 1
        x = F.pad(x, (0, pad_w, 0, pad_h))
        h, w = x.shape[2], x.shape[3]

    # 执行小波变换
    out = F.conv2d(x, wt_filter, stride=2, padding=0, groups=c)
    h2, w2 = out.shape[2], out.shape[3]


    total_elements = b * 4 * c * h2 * w2
    assert out.numel() == total_elements, (
        f"Wavelet transform shape mismatch: input={x.shape}, filter={wt_filter.shape}, "
        f"output={out.shape}, expected elements={total_elements}"
    )


    out = out.view(b, 4, c, h2, w2).permute(0, 2, 1, 3, 4).contiguous()
    return out


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    x = x.view(b, c * 4, h_half, w_half)


    h_out = h_half * 2
    w_out = w_half * 2


    out = F.conv_transpose2d(
        x, filters,
        stride=2,
        padding=0,
        groups=c
    )

    assert out.shape[2] == h_out and out.shape[3] == w_out, (
        f"逆小波变换尺寸错误: 输入 {x.shape} -> 输出 {out.shape}, "
        f"预期 ({h_out}, {w_out})"
    )

    return out


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return torch.mul(self.weight, x)


class WTB(nn.Module):
    def __init__(self, in_channels, wt_levels=1, wt_type='db1', reduction_ratio=16):
        super().__init__()
        # 处理通道数表达式
        if isinstance(in_channels, (str, list)):
            self.in_channels = int(in_channels[0]) if isinstance(in_channels, list) else int(in_channels)
        else:
            self.in_channels = int(in_channels)

        self.wt_levels = wt_levels

        # 创建小波滤波器
        wt_filter, iwt_filter = create_wavelet_filter(
            wt_type, self.in_channels, self.in_channels, torch.float
        )
        self.register_buffer('wt_filter_buf', wt_filter)
        self.register_buffer('iwt_filter_buf', iwt_filter)

        # 1. 基础卷积路径
        self.base_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 3, padding=1, groups=self.in_channels, bias=False),
            BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True)
        )

        # 2. 小波变换路径
        self.wavelet_convs = nn.ModuleList()
        self.wavelet_se = nn.ModuleList()
        for _ in range(wt_levels):
            self.wavelet_convs.append(nn.Sequential(
                nn.Conv2d(self.in_channels * 4, self.in_channels * 4, 3,
                          padding=1, groups=self.in_channels * 4, bias=False),
                BatchNorm2d(self.in_channels * 4),
                nn.ReLU(inplace=True)
            ))
            self.wavelet_se.append(SEBlock(self.in_channels * 4, reduction_ratio))

        # 3. 多尺度融合模块
        self.aspp = ASPP(self.in_channels, self.in_channels)

        # 4. 最终融合层
        self.final_se = SEBlock(self.in_channels, reduction_ratio)
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 1, bias=False),
            BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True)
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x

        # --- 小波变换路径 ---
        if self.wt_levels > 0:
            x_ll_in_levels, x_h_in_levels = [], []
            curr_x_ll = x


            orig_h, orig_w = x.shape[2], x.shape[3]

            for i in range(self.wt_levels):
                # 小波分解
                pad_h = 0 if curr_x_ll.shape[2] % 2 == 0 else 1
                pad_w = 0 if curr_x_ll.shape[3] % 2 == 0 else 1
                curr_x_ll = F.pad(curr_x_ll, (0, pad_w, 0, pad_h))

                curr_x = wavelet_transform(curr_x_ll, self.wt_filter_buf)
                curr_x_ll = curr_x[:, :, 0, :, :]

                # 小波系数处理
                b, c, _, h, w = curr_x.shape
                curr_x_tag = curr_x.view(b, c * 4, h, w)
                curr_x_tag = self.wavelet_convs[i](curr_x_tag)
                curr_x_tag = self.wavelet_se[i](curr_x_tag)
                curr_x_tag = curr_x_tag.view(b, c, 4, h, w)

                x_ll_in_levels.append(curr_x_tag[:, :, 0:1, :, :])
                x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

            # 小波重构
            next_x_ll = 0
            for i in range(self.wt_levels - 1, -1, -1):
                curr_x_ll = x_ll_in_levels[i]
                curr_x_h = x_h_in_levels[i]
                curr_x = torch.cat([curr_x_ll, curr_x_h], dim=2)
                reconstructed = inverse_wavelet_transform(curr_x, self.iwt_filter_buf)
                next_x_ll = reconstructed + next_x_ll

            # 裁剪尺寸并应用多尺度融合
            wavelet_out = next_x_ll[:, :, :orig_h, :orig_w]
            wavelet_out = self.aspp(wavelet_out)
        else:
            wavelet_out = 0

        # --- 基础卷积路径 ---
        conv_out = self.base_conv(x)

        # --- 双路径融合 ---
        fused = conv_out + wavelet_out


        out = self.final_conv(fused)
        out = self.final_se(out)

        # 残差连接
        return identity + out



def angle_transform(x, sin, cos):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    return (x * cos) + (torch.stack([-x2, x1], dim=-1).flatten(-2) * sin)


class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class full_GES_light(nn.Module):
    def __init__(self, embed_dim, num_heads=8, kernel_size=1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, 'embed_dim 能被 num_heads 整除'


        self.q_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size,
                                padding=kernel_size // 2, groups=num_heads)
        self.k_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size,
                                padding=kernel_size // 2, groups=num_heads)
        self.v_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size,
                                padding=kernel_size // 2, groups=num_heads)

        # Local Enhancement
        self.lepe = nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim)

        # Value特征增强
        self.eca = ECABlock(embed_dim)

        # 输出投影：仅 1x1 conv
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, 1)

        self.reset_parameters()

    def forward(self, x):
        identity = x

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 简化注意力映射：sigmoid(q + k)
        attn_map = torch.sigmoid(q + k)

        # Value 特征增强
        v = self.eca(v)

        # 注意力加权
        out = attn_map * v

        # Local Enhancement
        out = out + self.lepe(v)

        # 输出
        out = self.out_proj(out)

        # 残差连接
        out = out + identity
        return out

    def reset_parameters(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.lepe, self.out_proj]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class GES(nn.Module):
    def __init__(self, embed_dim=None, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ges = None

    def forward(self, x):
        if self.ges is None:
            c = x.shape[1]
            adjusted_embed_dim = make_divisible(c, self.num_heads)
            self.ges = full_GES_light(adjusted_embed_dim, self.num_heads).to(x.device)
        return self.ges(x)


def make_divisible(x, divisor):
    return max(divisor, int(x + divisor / 2) // divisor * divisor)




# BiasFree LayerNorm
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-4):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + self.eps) * self.weight

# WithBias LayerNorm
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-4):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + self.eps) * self.weight + self.bias

# Ultra-stable Token Mixer
class StableTokenMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.dwconv5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.pwconv = nn.Conv2d(dim, dim, 1)
        for m in [self.dwconv3, self.dwconv5, self.pwconv]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        u = x
        x = F.relu6(self.dwconv3(x))
        x = F.relu6(self.dwconv5(x))
        x = self.pwconv(x)
        return u + x

# 残差卷积块
class ResidualConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.relu = nn.ReLU6(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


class PRS(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        self.norm1 = (WithBias_LayerNorm if LayerNorm_type == 'WithBias' else BiasFree_LayerNorm)(dim)
        self.token_mixer = StableTokenMixer(dim)
        self.norm2 = (WithBias_LayerNorm if LayerNorm_type == 'WithBias' else BiasFree_LayerNorm)(dim)
        self.res_block = ResidualConvBlock(dim)

    def forward(self, x):
        b, c, h, w = x.shape

        # Normalize
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm1(x).permute(0, 3, 1, 2).contiguous()

        # Token Mixing
        x = self.token_mixer(x)

        # Normalize again
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm2(x).permute(0, 3, 1, 2).contiguous()

        # Residual Convolution Block
        x = self.res_block(x)

        return x

