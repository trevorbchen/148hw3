"""Vision Transformer — §2 (with RoPE option for §6).

You implement: PatchEmbeddings, ViT.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from basics.model import Block

PosEncoding = Literal["learned", "rope1d", "rope2d"]


class PatchEmbeddings(nn.Module):
    """Split an image into non-overlapping patches and project each to d_model.

    Implemented with a strided Conv2d whose kernel size and stride both equal
    `patch_size`.

    Args:
        img_size:   Input image side length (assumed square). Must be divisible
                    by patch_size.
        patch_size: Side length of each patch in pixels.
        d_model:    Output embedding dimension per patch.

    Forward:
        x: (B, 3, img_size, img_size) float tensor.
        returns: (B, num_patches, d_model) where num_patches = (img_size // patch_size) ** 2.
    """

    def __init__(self, img_size: int, patch_size: int, d_model: int) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels=3,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)              # (B, d_model, H/p, W/p)
        x = x.flatten(2)              # (B, d_model, num_patches)
        x = x.transpose(1, 2)         # (B, num_patches, d_model)
        return x


# ---------------------------------------------------------------------------
# RoPE-aware attention block (only used when pos_encoding != "learned").
# Mirrors basics.model.Block but applies RoPE to q,k inside attention.
# ---------------------------------------------------------------------------


class _RoPEHead(nn.Module):
    def __init__(self, d_model: int, head_dim: int, dropout: float) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, head_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rope_apply) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # RoPE expects (B, num_heads, T, head_dim); we treat per-head as num_heads=1.
        q4 = q.unsqueeze(1)
        k4 = k.unsqueeze(1)
        q4 = rope_apply(q4)
        k4 = rope_apply(k4)
        q = q4.squeeze(1)
        k = k4.squeeze(1)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        return attn @ v


class _RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        head_dim = d_model // num_heads
        self.heads = nn.ModuleList(
            [_RoPEHead(d_model, head_dim, dropout) for _ in range(num_heads)]
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rope_apply) -> torch.Tensor:
        out = torch.cat([h(x, rope_apply) for h in self.heads], dim=-1)
        return self.dropout(self.out_proj(out))


class _RoPEBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _RoPEMultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        from basics.model import MLP

        self.mlp = MLP(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, rope_apply) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), rope_apply)
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer.

    Pipeline (learned PE):
      1. Patchify with `PatchEmbeddings`.
      2. Prepend a learnable [CLS] token.
      3. Add a learnable positional embedding (interpolated if image size differs).
      4. Pass through `num_blocks` Transformer Blocks (is_decoder=False).
      5. Final LayerNorm; return [CLS] (or all tokens if requested).

    With pos_encoding="rope1d" or "rope2d" the learned pos_embed is dropped and
    RoPE is applied to q,k inside each attention head. The CLS token is given
    position 0 / coordinate (0,0); patches use 1..N (rope1d) or grid coords
    (rope2d).

    Args:
        img_size, patch_size, d_model, num_heads, num_blocks, dropout
        pos_encoding: "learned" | "rope1d" | "rope2d"
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        d_model: int,
        num_heads: int,
        num_blocks: int,
        dropout: float = 0.1,
        pos_encoding: PosEncoding = "learned",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout_p = dropout
        self.pos_encoding = pos_encoding
        self.patch_embed = PatchEmbeddings(img_size, patch_size, d_model)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.train_grid = int(math.sqrt(num_patches))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.dropout = nn.Dropout(dropout)

        if pos_encoding == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.blocks = nn.ModuleList(
                [
                    Block(
                        d_model=d_model, num_heads=num_heads,
                        block_size=num_patches + 1,
                        is_decoder=False, dropout=dropout,
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:
            from basics.rope import RoPE1D, RoPE2D

            head_dim = d_model // num_heads
            if pos_encoding == "rope1d":
                self.rope = RoPE1D(head_dim=head_dim, max_seq_len=4096)
            else:  # rope2d
                self.rope = RoPE2D(head_dim=head_dim, grid_size=64)
            self.blocks = nn.ModuleList(
                [_RoPEBlock(d_model, num_heads, dropout) for _ in range(num_blocks)]
            )
        self.ln = nn.LayerNorm(d_model)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    # ----------------------------------------------------------- positional
    def _interpolate_learned_pos_embed(self, num_patches: int) -> torch.Tensor:
        """Interpolate a learned (1, N_train+1, d) pos_embed to N_eval patches."""
        if num_patches == self.num_patches:
            return self.pos_embed
        cls_pe = self.pos_embed[:, :1]                         # (1, 1, d)
        patch_pe = self.pos_embed[:, 1:]                       # (1, N_train, d)
        d = patch_pe.shape[-1]
        train_grid = self.train_grid
        eval_grid = int(math.sqrt(num_patches))
        assert eval_grid * eval_grid == num_patches
        # bilinear interp on the grid
        patch_pe = patch_pe.reshape(1, train_grid, train_grid, d).permute(0, 3, 1, 2)
        patch_pe = F.interpolate(
            patch_pe, size=(eval_grid, eval_grid), mode="bilinear", align_corners=False
        )
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, num_patches, d)
        return torch.cat([cls_pe, patch_pe], dim=1)

    def _make_rope_apply(self, num_patches: int):
        """Return a closure rope_apply(x4) for the current sequence length."""
        device = self.cls_token.device
        if self.pos_encoding == "rope1d":
            positions = torch.arange(num_patches + 1, device=device)
            return lambda x4: self.rope(x4, positions)
        # rope2d: CLS at (0,0); patches at (gx, gy) with 1-based offsets.
        grid = int(math.sqrt(num_patches))
        assert grid * grid == num_patches
        ys, xs = torch.meshgrid(
            torch.arange(grid, device=device),
            torch.arange(grid, device=device),
            indexing="ij",
        )
        # CLS coords (0,0); patches use 1..grid so they don't collide with CLS.
        x_coords = torch.cat([torch.zeros(1, device=device, dtype=torch.long), xs.flatten() + 1])
        y_coords = torch.cat([torch.zeros(1, device=device, dtype=torch.long), ys.flatten() + 1])
        return lambda x4: self.rope(x4, x_coords, y_coords)

    # ----------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        B = x.shape[0]
        feats = self.patch_embed(x)                                # (B, N, d_model)
        N = feats.shape[1]
        cls = self.cls_token.expand(B, -1, -1)
        feats = torch.cat([cls, feats], dim=1)                     # (B, N+1, d_model)

        if self.pos_encoding == "learned":
            feats = feats + self._interpolate_learned_pos_embed(N)
            feats = self.dropout(feats)
            for block in self.blocks:
                feats = block(feats)
        else:
            feats = self.dropout(feats)
            rope_apply = self._make_rope_apply(N)
            for block in self.blocks:
                feats = block(feats, rope_apply)

        feats = self.ln(feats)
        if return_all_tokens:
            return feats                                           # (B, N+1, d_model)
        return feats[:, 0]                                         # (B, d_model)
