import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmseg.models.builder import BACKBONES
from engine.utils import rearrange, memory_efficient_attention
from engine.timers import Timer

# Assuming pdyrelu is defined somewhere in the code
def pdyrelu(x, condition_feature):
    # Placeholder for the actual implementation of pdyrelu
    return F.relu(x)  # Replace with the actual pdyrelu logic

class TwoLayerMLP(nn.Module):

    def __init__(self, embed_dim, mlp_dim):
        super(TwoLayerMLP, self).__init__()
        self.lin1 = nn.Linear(embed_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embed_dim)

    def forward(self, x: torch.Tensor, condition_feature: torch.Tensor) -> torch.Tensor:
        return self.lin2(pdyrelu(self.lin1(x), condition_feature))


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        dims = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList([
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in
            zip([input_dim] + dims, dims + [output_dim])])

    def forward(self, x, condition_feature):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = pdyrelu(x, condition_feature) if i < self.num_layers - 1 else x
        return x


class LayerNorm2d(nn.Module):

    def __init__(self, num_channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).square().mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = rearrange(self.weight, 'c -> c () ()') * x + \
            rearrange(self.bias, 'c -> c () ()')
        return x


class PatchEmbed(nn.Module):

    def __init__(self,
                 kernel_size=(16, 16),
                 stride=(16, 16),
                 padding=(0, 0),
                 in_dim=3,
                 embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.proj = nn.Conv2d(
            in_dim, embed_dim,
            kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b h w c')
        return x


class Attention(nn.Module):

    def __init__(self,
                 embed_dim,
                 num_heads=8,
                 qkv_bias=False,
                 use_rel_pos_embed=False,
                 input_size=None):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.use_rel_pos_embed = use_rel_pos_embed
        if self.use_rel_pos_embed:
            if input_size is None:
                raise ValueError(
                    "Input size must be provided if "
                    "using relative positional encoding.")
            self.rel_pos_h = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(
                torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor, condition_feature: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        q, k, v = rearrange(
            self.qkv(x), 'b h w (n3 hn c) -> n3 (b hn) (h w) c',
            n3=3, hn=self.num_heads)

        if self.use_rel_pos_embed:
            rel_pos_bias = self.decomposed_rel_pos(
                q, self.rel_pos