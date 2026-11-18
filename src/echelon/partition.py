"""Splitting a model's parameters between Echelon and another optimizer.

``Echelon`` only accepts 2D parameters, so the usual setup is two
optimizers: Echelon for hidden-layer matrices and AdamW for everything
else (embeddings, heads, biases, norms). Hand-writing that filter for
every model is the most common friction in using this package, so it
lives here, in one tested place.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

from ._echelon import Echelon

__all__ = ["echelon_adamw_pair", "partition_parameters"]

_DEFAULT_EXCLUDE = ("embed", "head")
_TOKEN_SPLIT_RE = re.compile(r"[._-]+")
_DEFAULT_HEAD_TOKENS = frozenset({"head", "wte", "wpe", "classifier"})


def _is_default_excluded_name(name: str) -> bool:
    lname = name.lower()
    tokens = [t for t in _TOKEN_SPLIT_RE.split(lname) if t]
    if any(t.startswith("embed") or t in _DEFAULT_HEAD_TOKENS for t in tokens):
        return True
    # Llama / torchtune / torchtitan name the final vocabulary projection
    # ``output.weight`` (or ``model.output.weight``), whereas BERT-style
    # hidden projections use ``attention.output.dense.weight`` (leaf ``dense``).
    parts = lname.split(".")
    return len(parts) >= 2 and parts[-2] == "output" and parts[-1] == "weight"


def _validate_exclude_substrings(exclude_substrings: Iterable[str]) -> tuple[str, ...]:
    if not isinstance(exclude_substrings, Iterable) or isinstance(
        exclude_substrings, (str, bytes)
    ):
        msg = (
            "exclude_substrings must be an iterable of non-empty strings, "
            f"not a single {type(exclude_substrings).__name__}"
        )
        raise ValueError(msg)  # noqa: TRY004
    out: list[str] = []
    for s in exclude_substrings:
        if not isinstance(s, str) or not s:
            raise ValueError(
                f"exclude_substrings entries must be non-empty strings, got: {s!r}"
            )
        out.append(s.lower())
    return tuple(out)


def partition_parameters(
    model_or_named: Module | Iterable[tuple[str, Tensor]],
    *,
    exclude_substrings: tuple[str, ...] = _DEFAULT_EXCLUDE,
    exclude_predicate: Callable[[str, Tensor], bool] | None = None,
    exclude_embeddings: bool = True,
    include_frozen: bool = False,
) -> tuple[list[Tensor], list[Tensor]]:
    """Split parameters into ``(hidden_2d, rest)``.

    A trainable parameter goes to ``hidden_2d`` when it is 2D, does not
    belong to an ``nn.Embedding`` / ``nn.EmbeddingBag`` submodule (when
    ``model_or_named`` is an ``nn.Module`` and ``exclude_embeddings=True``),
    and none of its parameter-name aliases are excluded. Everything else --
    non-2D parameters, embeddings, and excluded 2D output heads -- goes to
    ``rest``.

    Shared / tied parameters are deduplicated by tensor identity across all
    of their aliases (via ``named_parameters(remove_duplicate=False)`` on
    modules, and by ``id(p)`` on explicit iterables): a shared parameter
    never appears in both lists, and if *any* alias is excluded (for example
    a tied ``wte.weight`` / ``lm_head.weight``), the parameter goes to
    ``rest``.

    With the default ``exclude_substrings=("embed", "head")``, matching is
    token-aware across ``.`` / ``_`` / ``-`` boundaries so hidden attention
    layers such as ``multihead_attn.in_proj_weight`` stay in ``hidden_2d``,
    while ``embed*``, ``*head*`` tokens (``head``, ``lm_head``), ``wte``,
    ``wpe``, ``classifier``, and leaf ``output.weight`` heads go to ``rest``.
    Passing a custom ``exclude_substrings`` tuple matches those substrings
    case-insensitively instead.

    Args:
        model_or_named: an ``nn.Module``, or an iterable of
            ``(name, parameter)`` pairs as yielded by
            ``named_parameters()``.
        exclude_substrings: name fragments identifying 2D parameters that
            should not be optimized by Echelon.
        exclude_predicate: optional ``fn(name, param) -> bool``; if it
            returns ``True`` for any alias of a 2D parameter, that parameter
            goes to ``rest``.
        exclude_embeddings: when ``model_or_named`` is an ``nn.Module``,
            automatically route ``nn.Embedding`` and ``nn.EmbeddingBag``
            weights (including weights tied to them) to ``rest``
            (default: ``True``).
        include_frozen: include parameters with ``requires_grad=False``
            (default: ``False``).

    Returns:
        ``(hidden_2d, rest)`` as two disjoint lists of parameters, in
        first-seen order.

    Example:
        >>> hidden, rest = partition_parameters(model)
        >>> opt_echelon = Echelon(hidden, lr=0.02, weight_decay=0.0, rank=64)
        >>> opt_adamw = torch.optim.AdamW(rest, lr=3e-4, weight_decay=0.01)
    """
    use_default_rules = exclude_substrings == _DEFAULT_EXCLUDE
    excluded = () if use_default_rules else _validate_exclude_substrings(exclude_substrings)

    embedding_ids: set[int] = set()
    if isinstance(model_or_named, Module):
        if exclude_embeddings:
            for submod in model_or_named.modules():
                if isinstance(submod, (torch.nn.Embedding, torch.nn.EmbeddingBag)):
                    for emb_p in submod.parameters(recurse=False):
                        embedding_ids.add(id(emb_p))
        raw_pairs: Iterable[tuple[str, Tensor]] = model_or_named.named_parameters(
            remove_duplicate=False
        )
    else:
        raw_pairs = model_or_named

    ordered_params: list[Tensor] = []
    aliases_by_id: dict[int, list[str]] = {}
    for name, p in raw_pairs:
        if not include_frozen and not p.requires_grad:
            continue
        pid = id(p)
        if pid not in aliases_by_id:
            aliases_by_id[pid] = [name]
            ordered_params.append(p)
        else:
            aliases_by_id[pid].append(name)

    def _alias_excluded(alias: str, param: Tensor) -> bool:
        if exclude_predicate is not None and exclude_predicate(alias, param):
            return True
        if use_default_rules:
            return _is_default_excluded_name(alias)
        lname = alias.lower()
        return any(s in lname for s in excluded)

    hidden: list[Tensor] = []
    rest: list[Tensor] = []
    for p in ordered_params:
        pid = id(p)
        if (
            p.ndim == 2
            and pid not in embedding_ids
            and not any(_alias_excluded(a, p) for a in aliases_by_id[pid])
        ):
            hidden.append(p)
        else:
            rest.append(p)
    return hidden, rest


def echelon_adamw_pair(
    model_or_named: Module | Iterable[tuple[str, Tensor]],
    *,
    exclude_substrings: tuple[str, ...] = _DEFAULT_EXCLUDE,
    exclude_predicate: Callable[[str, Tensor], bool] | None = None,
    exclude_embeddings: bool = True,
    include_frozen: bool = False,
    echelon_kwargs: dict[str, Any] | None = None,
    adamw_kwargs: dict[str, Any] | None = None,
) -> tuple[Echelon, torch.optim.AdamW]:
    """Build the conventional ``(Echelon, AdamW)`` optimizer pair.

    This is :func:`partition_parameters` plus one constructor call per
    optimizer, with the defaults this package recommends for a first
    try (Echelon ``lr=0.02, weight_decay=0.0, rank=64`` on hidden
    weights; AdamW ``lr=3e-4, weight_decay=0.01`` on the rest).
    Pass ``echelon_kwargs`` / ``adamw_kwargs`` to override either.
    Step both optimizers (and any schedulers) each training step.

    If all trainable parameters fall on one side of the split (for example
    a bias-free MLP with no AdamW parameters, or a norm-only module with no
    2D hidden matrices), the empty side receives an empty parameter group
    ``[{"params": []}]`` so ``zero_grad()``, ``step()``, ``state_dict()``,
    and learning-rate schedulers continue to work without special cases.
    Raises ``ValueError`` if there are no parameters to optimize at all.
    """
    if echelon_kwargs and "params" in echelon_kwargs:
        raise ValueError("Pass parameters via model_or_named, not echelon_kwargs['params']")
    if adamw_kwargs and "params" in adamw_kwargs:
        raise ValueError("Pass parameters via model_or_named, not adamw_kwargs['params']")

    hidden, rest = partition_parameters(
        model_or_named,
        exclude_substrings=exclude_substrings,
        exclude_predicate=exclude_predicate,
        exclude_embeddings=exclude_embeddings,
        include_frozen=include_frozen,
    )
    if not hidden and not rest:
        raise ValueError(
            "model_or_named yielded no parameters to optimize "
            "(check if the model is empty or all parameters have requires_grad=False)"
        )
    echelon_kw: dict[str, Any] = {"lr": 0.02, "weight_decay": 0.0, "rank": 64}
    echelon_kw.update(echelon_kwargs or {})
    adamw_kw: dict[str, Any] = {"lr": 3e-4, "weight_decay": 0.01}
    adamw_kw.update(adamw_kwargs or {})

    echelon_params: Any = hidden if hidden else [{"params": []}]
    adamw_params: Any = rest if rest else [{"params": []}]
    return Echelon(echelon_params, **echelon_kw), torch.optim.AdamW(adamw_params, **adamw_kw)
