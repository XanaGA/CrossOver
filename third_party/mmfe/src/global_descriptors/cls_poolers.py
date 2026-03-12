import torch
import torch.nn as nn
import torch.nn.functional as F

def power_mean_pooling(descs, p=3.0, eps=1e-6):
    # descs: (B, N, D)
    return torch.pow(
        torch.mean(torch.pow(descs.clamp(min=eps), p), dim=1),
        1.0 / p
    )

class GeMPool(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):  # x: [B, C, H, W]
        x = x.clamp_min(self.eps)
        return x.pow(self.p).mean(dim=(-1, -2)).pow(1.0 / self.p)  # [B, C]


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)         # [B, HW, C]
        weights = torch.softmax(self.attn(x), dim=1)    # [B, HW, 1]
        pooled = (weights * x).sum(dim=1)               # [B, C]
        return pooled


class MHSA_Pooler(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)  # [B, HW, C]
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, C]
        out, _ = self.attn(query=cls, key=x, value=x)
        return out[:, 0]  # [B, C]
