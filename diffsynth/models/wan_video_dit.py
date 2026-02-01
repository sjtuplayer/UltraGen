import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .utils import hash_state_dict_keys
from torchvision import transforms
import torch.nn.functional as F
import pdb

try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn

    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else: #this brach
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                             [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs,grid_size, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    f,h,w=grid_size
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs_x=freqs[:f,:h,:w].reshape(f * h * w, 1, -1)
    x_out = torch.view_as_real(x_out * freqs_x).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x

class LoRA(nn.Module):
    def __init__(self, in_dim,out_dim, rank=64):
        super(LoRA, self).__init__()
        self.rank = rank
        self.lora_A = nn.Linear(in_dim,rank)
        self.lora_B = nn.Linear(rank,out_dim)

    def forward(self, x):
        return self.lora_B(self.lora_A(x))


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, block_id=0, down_factor=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.full_index = [0, 1,2, 27,28,29]
        self.wh = 1536
        #self.wh=7168
        if not block_id in self.full_index:
            self.q_lora_1 = LoRA(dim, dim)
            self.k_lora_1 = LoRA(dim, dim)
            self.v_lora_1 = LoRA(dim, dim)

            self.q_lora_2 = LoRA(dim, dim)
            self.k_lora_2 = LoRA(dim, dim)
            self.v_lora_2 = LoRA(dim, dim)
            self.alpha_time_embedding = nn.Sequential(
                nn.Linear(256, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            self.alpha_time_projection = nn.Sequential(
                nn.SiLU(), nn.Linear(dim, self.wh * 1), nn.Sigmoid())
            self.alpha_time_embedding2 = nn.Sequential(
                nn.Linear(256, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            self.alpha_time_projection2 = nn.Sequential(
                nn.SiLU(), nn.Linear(dim, self.wh * 1), nn.Sigmoid())

            self.downsample_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=4,
                                               stride=4, padding=0,groups=dim)
            self.upsample_conv = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                           padding=(1, 1, 1),groups=dim)

            self.downsample_conv_hierachcal = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=2, stride=2,padding=0,groups=dim)
            self.upsample_conv_hierachcal = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=(3, 3, 3),
                                                      stride=(1, 1, 1), padding=(1, 1, 1),groups=dim)

        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.attn = AttentionModule(self.num_heads)


    def split_attn(self, x,freqs, grid_size, down_factor=4):

        def closest_divisible(x, y):
            # 计算商
            quotient = x // y
            # 找到两个候选数
            lower = quotient * y
            upper = (quotient + 1) * y
            # 比较距离并返回最近的数
            if abs(x - lower) <= abs(x - upper):
                return lower
            else:
                return upper
        batch_size, dim = x.size(0),x.size(-1)
        t, w, h = grid_size
        x = x.view(batch_size, t, w, h, dim)
        ori_w, ori_h = w, h
        if closest_divisible(w,down_factor)!=w or closest_divisible(h,down_factor)!=h:
            new_w = closest_divisible(w, down_factor)
            new_h = closest_divisible(h, down_factor)
            x = x.permute(0, 1, 4, 2, 3).view(batch_size*t,dim,w,h)
            resize = transforms.Resize([new_w, new_h])
            x = resize(x)

            w,h = new_w, new_h
            x = x.view(batch_size, t, dim, w, h).permute(0, 1, 3, 4, 2)

        # 计算每个分块的大小
        w_step = w // down_factor
        h_step = h // down_factor

        x = rearrange(x, "b t (n1 w) (n2 h) c -> (b n1 n2) (t w h) c", n1=down_factor, n2=down_factor, w=w_step, h=h_step)
        # 应用注意力机制

        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)

        q = rope_apply(q, freqs,(t,w_step,h_step), self.num_heads)
        k = rope_apply(k, freqs,(t,w_step,h_step), self.num_heads)

        attended_parts = self.attn(q, k, v)
        # 重新组合分块
        attended_parts = rearrange(attended_parts, "(b n1 n2) (t w h) c -> b t (n1 w) (n2 h) c", n1=down_factor, n2=down_factor, w=w_step, h=h_step)

        if (ori_w, ori_h) != (w, h):
            attended_parts = attended_parts.view(batch_size, t, w,h, dim).permute(0, 1, 4, 2,3).contiguous().view(batch_size * t, dim, w ,h)
            resize = transforms.Resize([ori_w, ori_h])
            attended_parts = resize(attended_parts).view(batch_size, t, dim, ori_w, ori_h).permute(0, 1, 3, 4, 2).contiguous()

        attended_parts = attended_parts.view(batch_size, -1, dim)
        return attended_parts

    def hierarchical_attn(self, x_sample, freqs, grid_size, block_num=2, down_factor=2):
        # x_sample: (B t) d w h
        def closest_divisible(x, y):
            # 计算商
            quotient = x // y
            # 找到两个候选数
            lower = quotient * y
            upper = (quotient + 1) * y
            # 比较距离并返回最近的数
            if abs(x - lower) <= abs(x - upper):
                return lower
            else:
                return upper

        t, w, h = grid_size
        batch_size, dim = 1, x_sample.size(1)
        x_hierarchical = self.downsample_conv_hierachcal(x_sample)
        w, h = x_hierarchical.size(-2), x_hierarchical.size(-1)
        ori_w, ori_h = w, h

        if closest_divisible(w, block_num) != w or closest_divisible(h, block_num) != h:
            new_w = closest_divisible(w, block_num)
            new_h = closest_divisible(h, block_num)
            resize = transforms.Resize([new_w, new_h])
            x_hierarchical = resize(x_hierarchical)
            w, h = new_w, new_h

        x_hierarchical = rearrange(x_hierarchical, "(B t) d w h -> B (t w h) d", t=t)

        q = self.norm_q(self.q(x_hierarchical) + self.q_lora_2(x_hierarchical))
        k = self.norm_k(self.k(x_hierarchical) + self.k_lora_2(x_hierarchical))
        v = self.v(x_hierarchical) + self.v_lora_2(x_hierarchical)

        w_step = w // block_num
        h_step = h // block_num
        assert w_step * block_num == w, f'wrong split size w={w},down_factor={block_num}'
        assert h_step * block_num == h, f'wrong split size h={h},down_factor={block_num}'

        q = rearrange(q, "b (t n1 w n2 h) d -> (n1 n2 b) (t w h) d", b=batch_size, n1=block_num, n2=block_num, t=t, w=w_step, h=h_step, d=dim).contiguous()

        k = rearrange(k, "b (t n1 w n2 h) d -> (n1 n2 b) (t w h) d", b=batch_size, n1=block_num, n2=block_num, t=t,
                       w=w_step, h=h_step, d=dim).contiguous()

        v = rearrange(v, "b (t n1 w n2 h) d -> (n1 n2 b) (t w h) d", b=batch_size, n1=block_num, n2=block_num, t=t,
                       w=w_step, h=h_step, d=dim).contiguous()

        q = rope_apply(q, freqs, (t, w_step, h_step), self.num_heads)
        k = rope_apply(k, freqs, (t, w_step, h_step), self.num_heads)

        # 应用注意力机制
        attended_parts = self.attn(q, k, v)

        attended_parts = rearrange(attended_parts, "(n1 n2 b) (t w h) d -> (b t) d (n1 w) (n2 h)", t=t, w=w_step, h=h_step, d=dim,n1=block_num,n2=block_num)
        resize = transforms.Resize([grid_size[1], grid_size[2]])
        attended_parts = resize(attended_parts)
        attended_parts = rearrange(attended_parts, "(b t) d w h -> b d t w h", t=t)

        attended_parts = self.upsample_conv_hierachcal(attended_parts)

        attended_parts = rearrange(attended_parts, "b d t w h -> b (t w h) d").contiguous()

        return attended_parts


    def global_attn(self,x_sample,freqs,grid_size,down_factor=4):

        t, w, h = grid_size
        dim=x_sample.size(1)
        down_w, down_h=w//down_factor,h//down_factor

        x_global = self.downsample_conv(x_sample)
        x_global = rearrange(x_global, "(B t) d w h -> B (t w h) d", t=t)
        q = self.norm_q(self.q(x_global) + self.q_lora_1(x_global))
        k = self.norm_k(self.q(x_global) + self.k_lora_1(x_global))
        v = self.v(x_global) + self.v_lora_1(x_global)
        q = rope_apply(q, freqs,(t,down_w,down_h), self.num_heads)
        k = rope_apply(k, freqs,(t,down_w,down_h), self.num_heads)

        # Apply attention
        attended = self.attn(q, k, v)

        attended = rearrange(attended, " B (t w h) d -> (B t) d w h", d=dim,t=t,w=down_w,h=down_h)
        resize=transforms.Resize([w,h])
        attended=resize(attended)   #
        #
        attended = rearrange(attended,"(B t) d w h -> B d t w h",t=t).contiguous()

        attended=self.upsample_conv(attended)
        attended = rearrange(attended, "B d t w h -> B (t w h) d", d=dim, t=t, w=w, h=h).contiguous()
        # Final linear projection
        return attended

    def forward_ori(self, x, freqs,grid_size,*args):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs,grid_size, self.num_heads)
        k = rope_apply(k, freqs,grid_size, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)

    def forward(self, x, freqs,grid_size,time_step,block_id,down_factor=4):   #block_id=[0,29]
        if block_id in self.full_index:
            return self.forward_ori(x,freqs,grid_size)
        # Normalize and projectßßß
        t = self.alpha_time_embedding(
            sinusoidal_embedding_1d(256, time_step))
        t_mod = self.alpha_time_projection(t).unflatten(1, (1, self.dim))

        t2 = self.alpha_time_embedding2(
            sinusoidal_embedding_1d(256, time_step))
        t_mod2 = self.alpha_time_projection2(t2).unflatten(1, (1, self.dim))
        t,w,h=grid_size

        x_reshaped = rearrange(x, "B (t w h) d -> (B t) d w h", t=t, w=w)

        batch_size, seq_len, dim = x.size()
        assert seq_len == t * w * h, "Sequence length must be equal to t * w * h"

        split_attn_results = self.split_attn(x,freqs, grid_size, down_factor+block_id%2)

        global_attn_results=self.global_attn(x_reshaped,freqs,grid_size,down_factor)

        hierachical_attn_results=self.hierarchical_attn(x_reshaped,freqs,grid_size,block_num=2+block_id%2,down_factor=2)

        attn_results = t_mod2 *split_attn_results+ (1-t_mod2)*hierachical_attn_results

        attn_results = t_mod * attn_results + (1 - t_mod) * global_attn_results

        return self.o(attn_results)


class SelfAttention0(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs,*args):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6,*args):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps,*args)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, *args):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, *args))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):
    def __init__(
            self,
            dim: int,
            in_dim: int,
            ffn_dim: int,
            out_dim: int,
            text_dim: int,
            freq_dim: int,
            eps: float,
            patch_size: Tuple[int, int, int],
            num_heads: int,
            num_layers: int,
            has_image_input: bool,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps,layer_idx)
            for layer_idx in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim)  # clip_feature_dim = 1280

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2],
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(256, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, 1536))
        context = self.text_embedding(context)

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, grid_size = self.patchify(x)
        (f, h, w) = grid_size
        # freqs = torch.cat([
        #     self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        #     self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        #     self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        # ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        f,h,w=f+100,h+100,w+100
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).to(x.device)

        (f, h, w) = grid_size

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        for block_id,block in enumerate(self.blocks):
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,(f, h, w),timestep,block_id,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,(f, h, w),timestep,block_id,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()


class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config

    def from_civitai(self, state_dict):
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6d6ccde6845b95ad9114ab993d917893":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "349723183fc063b2bfc10bb2835cf677":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "efa44cddf936c70abd0ea28b6cbe946c":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        else:
            config = {}
        return state_dict, config
