# mypy: allow-untyped-defs
"""Learning-rate adjustment helpers shared with Muon.

Copied (logic only) from ``torch.optim._muon`` in PyTorch, which in turn
credits Keller Jordan's Muon write-up:
https://kellerjordan.github.io/posts/muon/
Echelon mirrors Muon's ``adjust_lr_fn`` conventions, so it reuses the same
validation and scaling rather than redefining them.
"""

import math

import torch

# Constants from Keller Jordan's Muon post.
EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5


def _validate_adjust_lr_fn(adjust_lr_fn: str | None) -> None:
    if adjust_lr_fn not in (None, "original", "match_rms_adamw", "spectral_unclamped"):
        raise ValueError(
            f"Adjust learning rate function {adjust_lr_fn} is not supported"
        )


def _adjust_lr(lr: float, adjust_lr_fn: str | None, param_shape: torch.Size) -> float:
    """Default learning rate adjustment used by Muon."""
    A, B = param_shape[:2]

    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    elif adjust_lr_fn == "spectral_unclamped":
        adjusted_ratio = math.sqrt(A / B)
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio
