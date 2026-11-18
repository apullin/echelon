"""Standalone Echelon optimizer for PyTorch.

Echelon is a rank-anchored spectral-clip optimizer: it tracks an orthonormal
top-``r`` singular frame of the momentum, orthogonalizes the small head with
quintic Newton-Schulz, and divides the orthogonal complement by an estimate
of its largest singular value. See https://evolvingmachines.ai/blog/echelon
(the algorithm is not the author's original work; this package is an
independent implementation following the public description).

This package was extracted from PyTorch PR #205345, which PyTorch Core
closed without merging: new optimizers are expected to prove themselves
as separate libraries first (see
https://github.com/pytorch/pytorch/issues/205526). The API mirrors
``torch.optim.Muon`` so the two can be swapped for hidden-layer weights.
"""

from importlib.metadata import PackageNotFoundError, version

from ._echelon import DEFAULT_MARGIN, DEFAULT_RANK, DEFAULT_TAU_ITERS, Echelon, echelon
from .partition import echelon_adamw_pair, partition_parameters

__all__ = [
    "DEFAULT_MARGIN",
    "DEFAULT_RANK",
    "DEFAULT_TAU_ITERS",
    "Echelon",
    "echelon",
    "echelon_adamw_pair",
    "partition_parameters",
]

try:
    __version__ = version("echelon-optimizer")
except PackageNotFoundError:
    __version__ = "0.3.0"
