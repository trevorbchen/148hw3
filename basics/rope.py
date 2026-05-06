"""Rotary Position Embeddings — §6.

You implement: RoPE1D, RoPE2D.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to the last dim of x using interleaved pairs.

    Args:
        x:   (..., T, D) where D is even.
        cos: (T, D/2) cosine table aligned with the T axis.
        sin: (T, D/2) sine table.

    Returns: (..., T, D) with each (x[..., 2i], x[..., 2i+1]) pair rotated.
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    # Broadcast cos/sin (T, D/2) over leading dims of x.
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return out


class RoPE1D(nn.Module):
    """1D Rotary Position Embedding.

    For a vector x at position m, RoPE groups dimensions into d/2 pairs and
    rotates each pair (x_{2i}, x_{2i+1}) by angle m * theta_i, where
        theta_i = base ** (-2i / head_dim).

    Apply RoPE to queries and keys (not values) inside attention, before
    computing q @ k^T.

    Args:
        head_dim:    Dimensionality of each attention head. Must be even.
        max_seq_len: Maximum sequence length to precompute angles for.
        base:        Base of the geometric progression (typically 10_000).

    Forward:
        x:         (B, num_heads, T, head_dim)
        positions: (T,) integer tensor of token positions.
        returns:   (B, num_heads, T, head_dim) with RoPE applied.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim // 2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[positions].to(x.dtype)  # (T, head_dim/2)
        sin = self.sin_cached[positions].to(x.dtype)
        return _apply_rotary(x, cos, sin)


class RoPE2D(nn.Module):
    """2D Rotary Position Embedding for image patches.

    Splits head_dim in half. The first half rotates by the patch's x-coordinate
    using 1D RoPE; the second half rotates by the patch's y-coordinate. After
    rotation, dot products depend on the 2D *relative* offset between patches.

    Args:
        head_dim:  Must be divisible by 4 (since each half is split into
                   real/imaginary pairs).
        grid_size: Maximum grid side (patches per row).
        base:      Base of the geometric progression.

    Forward:
        x:        (B, num_heads, T, head_dim)
        x_coords: (T,) integer tensor of x positions on the grid.
        y_coords: (T,) integer tensor of y positions on the grid.
        returns:  (B, num_heads, T, head_dim) with 2D RoPE applied.
    """

    def __init__(self, head_dim: int, grid_size: int, base: float = 10_000.0) -> None:
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        self.head_dim = head_dim
        self.half_dim = head_dim // 2
        self.grid_size = grid_size
        self.base = base

        inv_freq = base ** (-torch.arange(0, self.half_dim, 2).float() / self.half_dim)
        t = torch.arange(grid_size).float()
        freqs = torch.outer(t, inv_freq)  # (grid_size, head_dim // 4)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        x_coords: torch.Tensor,
        y_coords: torch.Tensor,
    ) -> torch.Tensor:
        x_first = x[..., : self.half_dim]
        x_second = x[..., self.half_dim :]

        cos_x = self.cos_cached[x_coords].to(x.dtype)
        sin_x = self.sin_cached[x_coords].to(x.dtype)
        cos_y = self.cos_cached[y_coords].to(x.dtype)
        sin_y = self.sin_cached[y_coords].to(x.dtype)

        rot_first = _apply_rotary(x_first, cos_x, sin_x)
        rot_second = _apply_rotary(x_second, cos_y, sin_y)
        return torch.cat([rot_first, rot_second], dim=-1)
