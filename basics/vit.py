"""Vision Transformer — §2.

You implement: PatchEmbeddings, ViT.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from basics.model import Block


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


class ViT(nn.Module):
    """Vision Transformer.

    Pipeline:
      1. Patchify with `PatchEmbeddings`.
      2. Prepend a learnable [CLS] token.
      3. Add a learnable positional embedding of shape (1, num_patches+1, d_model).
      4. Pass the sequence through `num_blocks` Transformer Blocks
         (with is_decoder=False).
      5. Apply a final LayerNorm.
      6. Return only the [CLS] slice — shape (B, d_model).

    For §5 (VLM), `return_all_tokens=True` returns the full
    (B, num_patches+1, d_model) sequence instead.

    Args:
        img_size, patch_size, d_model, num_heads, num_blocks, dropout
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        d_model: int,
        num_heads: int,
        num_blocks: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_embed = PatchEmbeddings(img_size, patch_size, d_model)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.dropout = nn.Dropout(dropout)

        block_size = num_patches + 1
        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model=d_model,
                    num_heads=num_heads,
                    block_size=block_size,
                    is_decoder=False,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.ln = nn.LayerNorm(d_model)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)                                  # (B, N, d_model)
        cls = self.cls_token.expand(B, -1, -1)                   # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                           # (B, N+1, d_model)
        x = x + self.pos_embed
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        if return_all_tokens:
            return x                                             # (B, N+1, d_model)
        return x[:, 0]                                           # (B, d_model)
