"""Splitting a model's parameters between Echelon and another optimizer.

``Echelon`` only accepts 2D parameters, so the usual setup is two
optimizers: Echelon for hidden-layer matrices and AdamW for everything
else (embeddings, heads, biases, norms). Hand-writing that filter for
every model is the most common friction in using this package, so it
lives here, in one tested place.
"""

from collections.abc import Iterable

import torch
from torch import Tensor
from torch.nn import Module

__all__ = ["partition_parameters", "echelon_adamw_pair"]


def _named_parameters(
    model_or_named: Module | Iterable[tuple[str, Tensor]],
) -> Iterable[tuple[str, Tensor]]:
    if isinstance(model_or_named, Module):
        return model_or_named.named_parameters()
    return model_or_named


def partition_parameters(
    model_or_named: Module | Iterable[tuple[str, Tensor]],
    *,
    exclude_substrings: tuple[str, ...] = ("embed", "head"),
) -> tuple[list[Tensor], list[Tensor]]:
    """Split parameters into ``(hidden_2d, rest)``.

    A parameter goes to ``hidden_2d`` when it is 2D and none of
    ``exclude_substrings`` (matched case-insensitively against the
    parameter name) appears in its name. Everything else -- 1D/0D
    parameters, and 2D parameters whose name matches an excluded
    substring (embeddings, output heads) -- goes to ``rest``.

    Args:
        model_or_named: an ``nn.Module``, or an iterable of
            ``(name, parameter)`` pairs as yielded by
            ``named_parameters()``.
        exclude_substrings: name fragments identifying parameters that
            must not be optimized by Echelon even though they are 2D.
            The defaults cover the usual ``embed``/``head`` names and
            also match e.g. ``lm_head`` and ``embedding``.

    Returns:
        ``(hidden_2d, rest)`` as two lists of parameters, in
        ``named_parameters()`` order.

    Example:
        >>> hidden, rest = partition_parameters(model)
        >>> opt_echelon = Echelon(hidden, lr=0.02, weight_decay=0.0, rank=64)
        >>> opt_adamw = torch.optim.AdamW(rest, lr=3e-4, weight_decay=0.01)
    """
    excluded = tuple(s.lower() for s in exclude_substrings)
    hidden: list[Tensor] = []
    rest: list[Tensor] = []
    for name, p in _named_parameters(model_or_named):
        lname = name.lower()
        if p.ndim == 2 and not any(s in lname for s in excluded):
            hidden.append(p)
        else:
            rest.append(p)
    return hidden, rest


def echelon_adamw_pair(
    model_or_named: Module | Iterable[tuple[str, Tensor]],
    *,
    exclude_substrings: tuple[str, ...] = ("embed", "head"),
    echelon_kwargs: dict | None = None,
    adamw_kwargs: dict | None = None,
) -> tuple["Echelon", torch.optim.AdamW]:
    """Build the conventional ``(Echelon, AdamW)`` optimizer pair.

    This is :func:`partition_parameters` plus one constructor call per
    optimizer, with the defaults this package recommends for a first
    try (Echelon ``lr=0.02, weight_decay=0.0, rank=64`` on hidden
    weights; AdamW ``lr=3e-4, weight_decay=0.01`` on the rest).
    Pass ``echelon_kwargs`` / ``adamw_kwargs`` to override either.
    Step both optimizers (and any schedulers) each training step.
    """
    # Local import: avoids a package-level import cycle (``_echelon``
    # does not import this module).
    from ._echelon import Echelon

    hidden, rest = partition_parameters(
        model_or_named, exclude_substrings=exclude_substrings
    )
    echelon_kw = {"lr": 0.02, "weight_decay": 0.0, "rank": 64}
    echelon_kw.update(echelon_kwargs or {})
    adamw_kw = {"lr": 3e-4, "weight_decay": 0.01}
    adamw_kw.update(adamw_kwargs or {})
    return Echelon(hidden, **echelon_kw), torch.optim.AdamW(rest, **adamw_kw)
