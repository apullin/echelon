# mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
"""Implementation of the Echelon optimizer."""

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, MutableMapping
from contextlib import contextmanager, nullcontext
from typing import Any, TypeAlias, cast

import torch
from torch import Tensor
from torch.optim import Optimizer

from ._muon_utils import (
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_C,
    DEFAULT_NS_STEPS,
    EPS,
    _adjust_lr,
    _validate_adjust_lr_fn,
)

_functional_api_doc = (
    "Functional API that performs {optimizer} algorithm computation.\n\n"
    "This function updates the provided parameters and optimizer state in place.\n"
    "The caller must initialize and retain optimizer state."
)
_params_doc = (
    "params (iterable): iterable of Tensors or dicts defining parameter groups"
)
_foreach_doc = (
    "foreach (bool, optional): whether to use the foreach implementation "
    "(default: None, which defaults to foreach on CUDA)."
)

try:
    # Same helpers torch.optim.Muon uses internally; fall back to local
    # equivalents on torch versions that do not expose them.
    from torch.optim.optimizer import (
        _default_to_fused_or_foreach,
        _disable_dynamo_if_unsupported,
        _to_scalar,
    )
except ImportError:  # pragma: no cover
    from collections.abc import Callable as _Callable

    def _disable_dynamo_if_unsupported(  # type: ignore[misc]
        single_tensor_fn: _Callable | None = None,
    ) -> _Callable:
        def wrap(fn: _Callable) -> _Callable:
            return fn

        return wrap

    def _default_to_fused_or_foreach(params, differentiable=False, use_fused=False):  # type: ignore[misc,no-redef]
        foreach = bool(params) and all(p.is_cuda for p in params)
        return False, foreach

    def _to_scalar(x):  # type: ignore[misc,no-redef]
        return x.squeeze() if isinstance(x, Tensor) and x.dim() != 0 else x


ParamsT: TypeAlias = (
    Iterable[Tensor]
    | Iterable[dict[str, Any]]
    | Iterable[tuple[str, Tensor]]
)

__all__ = ["Echelon", "echelon"]

# Reference Echelon specification: https://evolvingmachines.ai/blog/echelon
DEFAULT_RANK = 64
DEFAULT_TAU_ITERS = 2
DEFAULT_MARGIN = 1.05
_WARMUP_STEPS = 3
_TAU_FLOOR_MULT = 16.0
_TINY = 1e-30
_FP32_EPS = torch.finfo(torch.float32).eps

# Cached DTensor class, resolved lazily: importing torch.distributed on
# every step would be wasteful, and it may be unavailable in some builds.
_DTensor = None
_DTensor_checked = False


def _is_dtensor(t: Tensor) -> bool:
    global _DTensor, _DTensor_checked
    if not _DTensor_checked:
        _DTensor_checked = True
        try:
            from torch.distributed.tensor import DTensor as _DTensor_cls

            _DTensor = _DTensor_cls
        except (ImportError, AttributeError, RuntimeError):
            _DTensor = None
    return _DTensor is not None and isinstance(t, _DTensor)


# Relative diagonal shift of the shifted Cholesky-QR used by the foreach path.
_CHOLESKY_QR_SHIFT = 1e-6
# Maximum entrywise Gram error max|U^T U - I| accepted for a refreshed frame;
# passes IEEE fp32 (~1e-6) and TF32 (~3e-4) orthonormal frames while rejecting
# rank-deficient / collapsed Cholesky-QR outputs (error ~1.0).
_ORTH_TOL = 1e-2
# The foreach path stacks same-shape matrices up to this many elements; larger
# ones gain little from stacking and are stepped individually (still using
# Cholesky-QR2).
_FOREACH_MAX_NUMEL = 2**22
_FOREACH_MAX_BATCH_NUMEL = 2**26


@contextmanager
def _disable_autocast(device_type: str) -> Iterator[None]:
    available = torch.amp.autocast_mode.is_autocast_available(device_type)
    if available and torch.is_autocast_enabled(device_type):
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    else:
        yield


def _quintic_newton_schulz(
    X: Tensor,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
) -> Tensor:
    """Apply `ns_steps` odd-quintic Newton-Schulz iterations on pre-normalized `X` (`... x rows x cols`, `rows <= cols`)."""
    a, b, c = ns_coefficients
    mm_add = torch.addmm if X.ndim == 2 else torch.baddbmm
    for _ in range(ns_steps):
        A = X @ X.mT
        B = mm_add(A, A, A, beta=b, alpha=c)
        X = mm_add(X, B, X, beta=a)
    return X


def _householder_qr(Y: Tensor) -> Tensor:
    """Orthonormal basis of the columns of `Y`, signed so that `R` has a nonnegative diagonal."""
    Q, R = torch.linalg.qr(Y, mode="reduced")
    signs = torch.where(R.diagonal(dim1=-2, dim2=-1) < 0, -1.0, 1.0).to(Q.dtype)
    return Q * signs.unsqueeze(-2)


def _cholesky_qr2(Y: Tensor) -> Tensor:
    """Orthonormalize the columns of `Y` (`... x n x r`) with two shifted Cholesky-QR passes.

    Unlike Householder QR, this is a few GEMMs plus `r x r` Cholesky/triangular solves,
    making it substantially faster on GPU both when batched and on large single matrices.
    A rank-deficient or severely ill-conditioned `Y` either fails Cholesky (`NaN`) or
    leaves missing directions near zero (`U^T U != I`); `_echelon_step_impl` checks both
    finiteness and `max|U^T U - I| <= _ORTH_TOL` and retains the previous orthonormal
    frame when either fails.
    """
    for _ in range(2):
        G = Y.mT @ Y
        G_diag = G.diagonal(dim1=-2, dim2=-1)
        shift = (_CHOLESKY_QR_SHIFT * G_diag.mean(-1, keepdim=True)).clamp_min(_TINY)
        G_diag.add_(shift)
        L, info = torch.linalg.cholesky_ex(G)
        L = torch.where((info == 0)[..., None, None], L, torch.nan)
        Y = torch.linalg.solve_triangular(L, Y.mT, upper=False).mT
    return Y


def _subspace_step(X: Tensor, U: Tensor, orth: Callable[[Tensor], Tensor]) -> Tensor:
    """One subspace power step `U <- orth(X X^T U)` as two orthonormalized half-steps."""
    return orth(X @ orth(X.mT @ U))


def _frob_norm(X: Tensor) -> Tensor:
    """Frobenius norm over the last two dims, kept for broadcasting."""
    return torch.linalg.vector_norm(X, dim=(-2, -1), keepdim=True)


def _prepare_unit_momentum(
    X: Tensor, eps: float, owns_input: bool
) -> tuple[Tensor, Tensor, Tensor]:
    """Normalize ``X`` to unit Frobenius norm in ``float32`` without overflow, underflow, or NaN leakage.

    Pre-scaling by the infinity norm keeps entries in ``[-1, 1]`` before squaring so
    ``float32`` inputs near ``1e20``--``1e38`` do not overflow ``||X||_F`` to ``inf`` and
    tiny ``float32`` inputs near ``1e-25``--``1e-42`` (or ``float64`` inputs outside
    ``float32`` range) do not underflow to ``0``. Non-finite or zero matrices are
    masked to exact ``0.0`` with ``active=False`` and ``scale=0.0``.
    """
    if X.dtype == torch.float64:
        amax64 = torch.linalg.vector_norm(
            X, ord=float("inf"), dim=(-2, -1), keepdim=True
        )
        active = torch.isfinite(amax64) & (amax64 > 0)
        Xf64 = X.masked_fill(~active, 0.0)
        Xf64.div_(torch.where(active, amax64, 1.0))
        Xf = Xf64.to(dtype=torch.float32, memory_format=torch.preserve_format)
        scaled_norm = _frob_norm(Xf)
        if eps > 0.0:
            ratio = torch.where(amax64 < eps, amax64 / eps, 1.0).to(dtype=torch.float32)
            scale = torch.where(active, (ratio * scaled_norm).clamp_max(1.0), 0.0)
        else:
            scale = torch.where(active, 1.0, 0.0)
        Xf.div_(scaled_norm.clamp_min(_TINY))
        return Xf, scale, active

    if X.dtype == torch.float32:
        Xf = X if owns_input else X.clone(memory_format=torch.preserve_format)
    else:
        Xf = X.to(dtype=torch.float32, memory_format=torch.preserve_format)

    amax = torch.linalg.vector_norm(Xf, ord=float("inf"), dim=(-2, -1), keepdim=True)
    active = torch.isfinite(amax) & (amax > 0)
    Xf.masked_fill_(~active, 0.0)
    Xf.div_(torch.where(active, amax, 1.0))
    scaled_norm = _frob_norm(Xf)
    if eps > 0.0:
        ratio = torch.where(amax < eps, amax / eps, 1.0)
        scale = torch.where(active, (ratio * scaled_norm).clamp_max(1.0), 0.0)
    else:
        scale = torch.where(active, 1.0, 0.0)
    Xf.div_(scaled_norm.clamp_min(_TINY))
    return Xf, scale, active


def _echelon_step(
    X: Tensor,
    U: Tensor | None,
    z: Tensor | None,
    *,
    rank: int,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    tau_iters: int,
    margin: float,
    eps: float,
    orth: Callable[[Tensor], Tensor] = _householder_qr,
    autocast_disabled: bool = False,
    _owns_input: bool = False,
) -> Tensor:
    """Compute Echelon updates for short-side-first momenta `X` (`... x s x l`, `s <= l`), updating `U` and `z` in place.

    ``autocast_disabled`` is an internal escape hatch for callers (the warmup
    in :meth:`Echelon._init_group`) that already entered the disabled-autocast
    context, so the context is entered once per step rather than once per
    parameter. ``_owns_input`` indicates that ``X`` is an unaliased temporary
    that may be mutated in place when already ``float32``."""
    ctx = nullcontext() if autocast_disabled else _disable_autocast(X.device.type)
    with ctx:
        return _echelon_step_impl(
            X,
            U,
            z,
            rank=rank,
            ns_coefficients=ns_coefficients,
            ns_steps=ns_steps,
            tau_iters=tau_iters,
            margin=margin,
            eps=eps,
            orth=orth,
            owns_input=_owns_input,
        )


def _echelon_step_impl(
    X: Tensor,
    U: Tensor | None,
    z: Tensor | None,
    *,
    rank: int,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    tau_iters: int,
    margin: float,
    eps: float,
    orth: Callable[[Tensor], Tensor],
    owns_input: bool = False,
) -> Tensor:
    s = X.size(-2)
    T, scale, active = _prepare_unit_momentum(X, eps, owns_input)
    if rank >= s:
        T.mul_(scale)
        out = _quintic_newton_schulz(T, ns_coefficients, ns_steps)
        out.masked_fill_(~active, 0.0)
        return out

    if U is None or z is None:
        raise ValueError(
            f"subspace_frame and tail_vec must be provided when rank ({rank}) < short_side ({s})"
        )

    Uf = U.to(dtype=torch.float32)

    # 1. Refresh tracked top-r subspace: V = orth(X^T U), U = orth(X V),
    # keeping U if orth produced non-finite entries or lost orthonormality.
    U_next = _subspace_step(T, Uf, orth)
    gram_err = U_next.mT @ U_next
    gram_err.diagonal(dim1=-2, dim2=-1).sub_(1.0)
    orth_ok = gram_err.abs().amax(dim=(-2, -1), keepdim=True) <= _ORTH_TOL
    refresh = active & torch.isfinite(U_next).all(dim=(-2, -1), keepdim=True) & orth_ok
    Uf = torch.where(refresh, U_next, Uf)

    # 2. Head: H = U^T X, P = NS5(H)
    H = Uf.mT @ T
    H_norm = _frob_norm(H)
    P = _quintic_newton_schulz(
        (H / H_norm.clamp_min(_TINY)) * scale, ns_coefficients, ns_steps
    )

    # 3. Tail: T = X - U H (re-projected once to keep U^T T = 0 to fp32 precision).
    # Fuse both projections directly into T via addmm_/baddbmm_ to avoid allocating
    # s x l temporaries and preserve X's memory layout on transposed views.
    if T.ndim == 2:
        T.addmm_(Uf, H, beta=1.0, alpha=-1.0)
        T.addmm_(Uf, Uf.mT @ T, beta=1.0, alpha=-1.0)
    else:
        T.baddbmm_(Uf, H, beta=1.0, alpha=-1.0)
        T.baddbmm_(Uf, Uf.mT @ T, beta=1.0, alpha=-1.0)
    t_frob = _frob_norm(T)
    tau_floor = _TAU_FLOOR_MULT * math.sqrt(s) * _FP32_EPS * H_norm

    zf = z.to(dtype=torch.float32)
    z_cur = zf / _frob_norm(zf).clamp_min(_TINY)
    est = t_frob
    for _ in range(tau_iters):
        w = T @ z_cur
        w_norm = _frob_norm(w)
        z_next = T.mT @ w
        z_next_norm = _frob_norm(z_next)
        # ||T^T w|| / ||w|| = ||T^T (w / ||w||)|| <= sigma_max(T) gives an extra
        # half power step of accuracy over ||w|| with no additional matmul.
        est = torch.where(w_norm > 0, z_next_norm / w_norm.clamp_min(_TINY), 0.0)
        z_cur = torch.where(
            z_next_norm > 0, z_next / z_next_norm.clamp_min(_TINY), z_cur
        )

    U.copy_(Uf)
    z.copy_(torch.where(active & (t_frob > tau_floor), z_cur, zf))

    # 4. Output: O = U P + T / tau (with lower bound ||T||_F / sqrt(s - r) and ker(T) fallback ||T||_F)
    tail_lb = t_frob / math.sqrt(s - rank)
    est = torch.where(est > 0, torch.maximum(est, tail_lb), t_frob)
    tau = margin * torch.maximum(est, tau_floor).clamp_min(_TINY)
    T.mul_(scale / tau)
    if T.ndim == 2:
        T.addmm_(Uf, P)
    else:
        T.baddbmm_(Uf, P)
    T.masked_fill_(~active, 0.0)
    return T


def _is_real_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _validate_echelon_scalar_hyperparameters(
    *,
    momentum: float,
    weight_decay: float,
    adjust_lr_fn: str | None,
) -> None:
    if not _is_real_number(momentum) or not (0.0 <= momentum < 1.0):
        raise ValueError(f"momentum must be a finite number in [0, 1), got: {momentum!r}")
    if (
        not _is_real_number(weight_decay)
        or not math.isfinite(weight_decay)
        or weight_decay < 0.0
    ):
        raise ValueError(f"weight decay should be finite and >= 0, got: {weight_decay!r}")
    _validate_adjust_lr_fn(adjust_lr_fn)


def _validate_echelon_hyperparameters(
    *,
    lr: float | Tensor,
    momentum: float,
    weight_decay: float,
    adjust_lr_fn: str | None,
) -> None:
    if isinstance(lr, Tensor):
        if torch.is_complex(lr) or lr.dtype == torch.bool or lr.numel() != 1:
            raise ValueError("Tensor lr must be a real 1-element tensor")
        if not bool(torch.isfinite(lr) & (lr >= 0.0)):
            raise ValueError(f"Learning rate should be finite and >= 0 but is: {lr}")
    elif not _is_real_number(lr) or not math.isfinite(lr) or lr < 0.0:
        raise ValueError(f"Learning rate should be finite and >= 0 but is: {lr!r}")
    _validate_echelon_scalar_hyperparameters(
        momentum=momentum,
        weight_decay=weight_decay,
        adjust_lr_fn=adjust_lr_fn,
    )


def _validate_echelon_options(
    *,
    rank: int,
    tau_iters: int,
    margin: float,
    ns_steps: int,
    ns_coefficients: tuple[float, float, float],
    eps: float,
) -> tuple[float, float, float]:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError(f"rank must be an integer >= 1, got: {rank!r}")
    if isinstance(tau_iters, bool) or not isinstance(tau_iters, int) or tau_iters < 1:
        raise ValueError(f"tau_iters must be an integer >= 1, got: {tau_iters!r}")
    if not _is_real_number(margin) or not math.isfinite(margin) or margin <= 0.0:
        raise ValueError(f"margin must be finite and > 0, got: {margin!r}")
    if (
        isinstance(ns_steps, bool)
        or not isinstance(ns_steps, int)
        or not 1 <= ns_steps < 100
    ):
        raise ValueError(
            f"ns_steps must be an integer in [1, 100) for computational efficiency, got: {ns_steps!r}"
        )
    if not isinstance(ns_coefficients, Iterable) or isinstance(
        ns_coefficients, (str, bytes)
    ):
        msg = (
            "Coefficients must be a tuple of exactly 3 finite numbers, "
            f"got: {ns_coefficients!r}"
        )
        raise ValueError(msg)  # noqa: TRY004
    raw_coeffs = list(ns_coefficients)
    if (
        len(raw_coeffs) != 3
        or not all(_is_real_number(c) and math.isfinite(float(c)) for c in raw_coeffs)
    ):
        raise ValueError(
            "Coefficients must be a tuple of exactly 3 finite numbers, "
            f"got: {ns_coefficients!r}"
        )
    coeffs = (float(raw_coeffs[0]), float(raw_coeffs[1]), float(raw_coeffs[2]))
    if not _is_real_number(eps) or not math.isfinite(eps) or eps < 0.0:
        raise ValueError(f"eps should be finite and >= 0 but is: {eps!r}")
    return coeffs


def _validate_foreach(foreach: bool | None) -> None:
    if foreach is not None and not isinstance(foreach, bool):
        raise ValueError(f"foreach must be True, False, or None, got: {foreach!r}")


def _warmup_frame(
    group: MutableMapping,
    p: Tensor,
    state: MutableMapping,
    U: Tensor,
    z: Tensor,
    *,
    rank: int,
) -> bool:
    """Warm up ``U``/``z`` on this step's update; return whether it was usable.

    A zero or non-finite update cannot warm the frame up, so it stays at its
    random start and the caller records the warmup as still pending: it is
    retried on the first finite, nonzero update (see ``_init_group``).
    The active check is a host sync, but only on frame creation or while a
    warmup is pending -- never in the steady state.
    """
    grad = cast(Tensor, p.grad)
    m0 = state["momentum_buffer"].lerp(grad, 1 - group["momentum"])
    if group["nesterov"]:
        m0 = grad.lerp(m0, group["momentum"])
    X0 = m0.mT if p.size(0) > p.size(1) else m0
    amax0 = torch.linalg.vector_norm(X0, ord=float("inf"))
    active = bool(torch.isfinite(amax0) & (amax0 > 0))
    if active:
        with _disable_autocast(p.device.type):
            for _ in range(_WARMUP_STEPS):
                _echelon_step(
                    X0,
                    U,
                    z,
                    rank=rank,
                    ns_coefficients=group["ns_coefficients"],
                    ns_steps=group["ns_steps"],
                    tau_iters=group["tau_iters"],
                    margin=group["margin"],
                    eps=group["eps"],
                    autocast_disabled=True,
                )
    return active


def _check_group_inputs(group: MutableMapping) -> None:
    """Reject DTensor/complex/sparse/non-2D inputs before any group is stepped.

    ``Echelon.step`` calls this for every group up front (and again after a
    closure runs): the per-group loop would otherwise leave earlier groups
    stepped if a later group fails validation.
    """
    lr = group["lr"]
    if isinstance(lr, Tensor):
        if torch.is_complex(lr) or lr.dtype == torch.bool or lr.numel() != 1:
            raise ValueError("Tensor lr must be a real 1-element tensor")
    elif not _is_real_number(lr) or not math.isfinite(lr) or lr < 0.0:
        raise ValueError(f"Learning rate should be finite and >= 0 but is: {lr!r}")

    for p in group["params"]:
        if _is_dtensor(p):
            raise NotImplementedError(
                "Echelon does not support DTensor parameters "
                "(for example under FSDP2)"
            )
        if p.ndim != 2:
            raise ValueError(
                f"Echelon only supports 2D parameters whereas we found a parameter with size: {p.size()}"
            )
        if torch.is_complex(p):
            raise RuntimeError("Echelon does not support complex parameters")
        g = p.grad
        if g is not None:
            if _is_dtensor(g):
                raise NotImplementedError(
                    "Echelon does not support DTensor gradients "
                    "(for example under FSDP2)"
                )
            if torch.is_complex(g):
                raise RuntimeError("Echelon does not support complex gradients")
            if g.is_sparse:
                raise RuntimeError("Echelon does not support sparse gradients")
            if g.shape != p.shape:
                raise ValueError(
                    f"Gradient shape {tuple(g.shape)} does not match parameter shape {tuple(p.shape)}"
                )
            if g.device != p.device:
                raise ValueError(
                    f"Gradient device {g.device} does not match parameter device {p.device}"
                )


class Echelon(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        *,
        rank: int = DEFAULT_RANK,
        tau_iters: int = DEFAULT_TAU_ITERS,
        margin: float = DEFAULT_MARGIN,
        ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = None,
        foreach: bool | None = None,
    ) -> None:
        _validate_echelon_hyperparameters(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            adjust_lr_fn=adjust_lr_fn,
        )
        ns_coefficients = _validate_echelon_options(
            rank=rank,
            tau_iters=tau_iters,
            margin=margin,
            ns_steps=ns_steps,
            ns_coefficients=ns_coefficients,
            eps=eps,
        )
        _validate_foreach(foreach)

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "rank": rank,
            "tau_iters": tau_iters,
            "margin": margin,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
            "foreach": foreach,
        }
        super().__init__(params, defaults)

    @torch._disable_dynamo
    def add_param_group(self, param_group: dict) -> None:
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        As :meth:`torch.optim.Optimizer.add_param_group`, but the group's hyperparameters get the
        same checks as the constructor's and all of its parameters must be 2D; a group that fails
        a check raises and is not added.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        try:
            _validate_echelon_hyperparameters(
                lr=group["lr"],
                momentum=group["momentum"],
                weight_decay=group["weight_decay"],
                adjust_lr_fn=group["adjust_lr_fn"],
            )
            group["ns_coefficients"] = _validate_echelon_options(
                rank=group["rank"],
                tau_iters=group["tau_iters"],
                margin=group["margin"],
                ns_steps=group["ns_steps"],
                ns_coefficients=group["ns_coefficients"],
                eps=group["eps"],
            )
            _validate_foreach(group["foreach"])
            for p in group["params"]:
                if _is_dtensor(p):
                    raise NotImplementedError(
                        "Echelon does not support DTensor parameters "
                        "(for example under FSDP2)"
                    )
                if p.ndim != 2:
                    raise ValueError(
                        f"Echelon only supports 2D parameters whereas we found a parameter with size: {p.size()}. "
                        "Pass only 2D hidden weights to Echelon (see echelon.partition_parameters) "
                        "and use another optimizer, e.g. AdamW, for the rest."
                    )
                if torch.is_complex(p):
                    raise RuntimeError("Echelon does not support complex parameters")
        except Exception:
            self.param_groups.pop()
            raise

    @torch._disable_dynamo
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""Load optimizer state, preserving ``float32`` precision for ``subspace_frame`` and ``tail_vec``.

        :meth:`torch.optim.Optimizer.load_state_dict` casts floating-point state
        tensors to the parameter's dtype (e.g. ``bfloat16``), which would round
        orthonormal subspace frames on checkpoint load. When the incoming state
        dict stores ``subspace_frame`` or ``tail_vec`` in ``torch.float32``,
        this override restores them in ``torch.float32`` on the parameter's
        device without rounding.
        """
        super().load_state_dict(state_dict)
        saved_state_map = state_dict.get("state", {})
        saved_groups = state_dict.get("param_groups", [])
        for saved_group, group in zip(saved_groups, self.param_groups):
            for saved_id, p in zip(saved_group.get("params", []), group["params"]):
                saved_p_state = saved_state_map.get(saved_id)
                if saved_p_state is None or p not in self.state:
                    continue
                p_state = self.state[p]
                for key in ("subspace_frame", "tail_vec"):
                    orig = saved_p_state.get(key)
                    if isinstance(orig, Tensor) and orig.dtype == torch.float32:
                        p_state[key] = (
                            orig.to(device=p.device, dtype=torch.float32)
                            .contiguous()
                            .clone()
                        )

    def _init_group(
        self,
        group: MutableMapping,
        group_idx: int,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        momentum_bufs: list[Tensor],
        subspace_frames: list[Tensor | None],
        tail_vecs: list[Tensor | None],
    ) -> bool:
        rank = group["rank"]
        for param_idx, p in enumerate(group["params"]):
            s, l = min(p.size(0), p.size(1)), max(p.size(0), p.size(1))
            low_rank = p.numel() > 0 and rank < s
            if not low_rank and p in self.state:
                # A frame kept from a smaller `rank` would later be reused without a warmup.
                self.state[p].pop("subspace_frame", None)
                self.state[p].pop("tail_vec", None)
                self.state[p].pop("warmed", None)
            if p.grad is None:
                continue

            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    p.grad, memory_format=torch.preserve_format
                )
            momentum_bufs.append(state["momentum_buffer"])

            if low_rank:
                frame = state.get("subspace_frame")
                # Initialize on the first step and after a change of `rank`.
                if frame is None or frame.shape != (s, rank):
                    # A CPU generator gives the same initial frame on every device.
                    gen = torch.Generator(device="cpu")
                    gen.manual_seed(group_idx * 1000003 + param_idx)
                    u0 = torch.randn(
                        (s, rank), generator=gen, device="cpu", dtype=torch.float32
                    )
                    z0 = torch.randn(
                        (l, 1), generator=gen, device="cpu", dtype=torch.float32
                    )
                    with _disable_autocast(p.device.type):
                        U0, _ = torch.linalg.qr(u0.to(device=p.device), mode="reduced")
                    # linalg.qr returns a column-major Q. Store the frame row-major: later steps
                    # copy into it in place, and some checkpointers write state as raw storage.
                    U0 = U0.contiguous()
                    z0 = z0.to(device=p.device)
                    state["subspace_frame"] = U0
                    state["tail_vec"] = z0
                    # Warm up on the update this step orthogonalizes, discarding
                    # the updates. A zero or non-finite update leaves the frame
                    # at its random start and the warmup pending (see below).
                    state["warmed"] = _warmup_frame(group, p, state, U0, z0, rank=rank)
                else:
                    if frame.dtype != torch.float32:
                        # Fallback if a caller loaded a pre-cast state dict directly.
                        state["subspace_frame"] = frame.to(torch.float32).contiguous()
                        state["tail_vec"] = state["tail_vec"].to(torch.float32)
                    if not state.get("warmed", True):
                        # First finite, nonzero update since the frame was
                        # created: warm up now, so it does not follow a random
                        # frame. Checkpoints saved before the flag existed
                        # (missing key) are treated as already warmed.
                        state["warmed"] = _warmup_frame(
                            group,
                            p,
                            state,
                            state["subspace_frame"],
                            state["tail_vec"],
                            rank=rank,
                        )
                subspace_frames.append(state["subspace_frame"])
                tail_vecs.append(state["tail_vec"])
            else:
                subspace_frames.append(None)
                tail_vecs.append(None)

        return False  # has_complex

    def _validate_all_groups(self) -> None:
        for group in self.param_groups:
            group["ns_coefficients"] = _validate_echelon_options(
                rank=group["rank"],
                tau_iters=group["tau_iters"],
                margin=group["margin"],
                ns_steps=group["ns_steps"],
                ns_coefficients=group["ns_coefficients"],
                eps=group["eps"],
            )
            # Tensor lr value checks are skipped here to avoid a CUDA host sync
            # on every step; scalar hyperparameters and shapes/dtypes are sync-free.
            _validate_echelon_scalar_hyperparameters(
                momentum=group["momentum"],
                weight_decay=group["weight_decay"],
                adjust_lr_fn=group["adjust_lr_fn"],
            )
            _validate_foreach(group["foreach"])
            _check_group_inputs(group)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        # Validate every group before running a closure or touching state: an
        # invalid option or input in a later group must never leave earlier
        # groups partially stepped.
        self._validate_all_groups()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
            # Re-validate after the closure in case it created invalid (e.g.
            # sparse) gradients or mutated group options.
            self._validate_all_groups()

        for group_idx, group in enumerate(self.param_groups):
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            momentum_bufs: list[Tensor] = []
            subspace_frames: list[Tensor | None] = []
            tail_vecs: list[Tensor | None] = []

            has_complex = self._init_group(
                group,
                group_idx,
                params_with_grad,
                grads,
                momentum_bufs,
                subspace_frames,
                tail_vecs,
            )

            if not params_with_grad:
                continue

            echelon(
                params_with_grad,
                grads,
                momentum_bufs,
                subspace_frames,
                tail_vecs,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                nesterov=group["nesterov"],
                rank=group["rank"],
                tau_iters=group["tau_iters"],
                margin=group["margin"],
                ns_coefficients=group["ns_coefficients"],
                eps=group["eps"],
                ns_steps=group["ns_steps"],
                adjust_lr_fn=group["adjust_lr_fn"],
                foreach=group["foreach"],
                has_complex=has_complex,
                _prevalidated=True,
                _skip_input_checks=True,
            )
        return loss


Echelon.__doc__ = (
    r"""Implements the Echelon optimizer: rank-anchored spectral clipping in :math:`O(mnr)` FLOPs.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt} \\
            &\textbf{input}      : \gamma \text{ (lr)},\ \lambda \text{ (weight decay)},\
               \mu \text{ (momentum)},\ \textit{nesterov}\in\{\text{True},\text{False}\},\\
            &\hspace{13mm} r \text{ (rank)},\ k_{\tau} \text{ (tail iters)},\ c_{\tau} \text{ (margin)},\
               \theta_0 \in \mathbb{R}^{m \times n} \\
            &\textbf{initialize} : B_0 \leftarrow 0,\ U_0 \in \mathbb{R}^{s \times r}\ (U_0^\top U_0 = I_r),\
               z_0 \in \mathbb{R}^{l \times 1} \\[-1.ex]
            &\rule{110mm}{0.4pt} \\
            &\textbf{for}\ t=1\ \textbf{to}\ \ldots\ \textbf{do} \\[0.25ex]
            &\hspace{5mm} g_t \leftarrow \nabla_{\theta} f_t(\theta_{t-1}) \\[0.25ex]
            &\hspace{5mm} B_t \leftarrow \mu B_{t-1} + (1 - \mu) g_t \\[0.25ex]
            &\hspace{5mm} \widetilde{B}_t \leftarrow
                \begin{cases}
                   (1 - \mu) g_t + \mu B_t, & \text{if nesterov}=\text{True} \\
                   B_t,                     & \text{if nesterov}=\text{False}
                \end{cases} \\[0.75ex]
            &\hspace{5mm} X_t \leftarrow \widetilde{B}_t \text{ or } \widetilde{B}_t^\top \in \mathbb{R}^{s \times l}\
               (s = \min(m, n),\ l = \max(m, n)) \\[0.25ex]
            &\hspace{5mm} V_t \leftarrow \mathrm{orth}(X_t^\top U_{t-1}),\quad
               U_t \leftarrow \mathrm{orth}(X_t V_t) \\[0.25ex]
            &\hspace{5mm} H_t \leftarrow U_t^\top X_t,\quad
               P_t \leftarrow \mathrm{NS}_k(H_t),\quad
               T_t \leftarrow X_t - U_t H_t \\[0.25ex]
            &\hspace{5mm} \tau_t \leftarrow c_{\tau}\max\big(\widehat{\sigma}_{\max}(T_t),\
               \lVert T_t \rVert_F / \sqrt{s - r},\ \epsilon_{\tau}\big)\
               \text{(} \widehat{\sigma}_{\max} \text{ via } k_{\tau} \text{ power steps)} \\[0.25ex]
            &\hspace{5mm} O_t \leftarrow U_t P_t + T_t / \tau_t\ \text{(transposed back if } m > n\text{)} \\[0.25ex]
            &\hspace{5mm} \theta_t \leftarrow (1 - \gamma\lambda)\theta_{t-1}
               - \mathrm{AdjustLR}(\gamma; \mathrm{shape}(\theta_{t-1}))\, O_t \\
            &\rule{110mm}{0.4pt}
       \end{aligned}

    With an exact top-:math:`r` frame, an exact head polar factor, and :math:`\tau_t = c_{\tau}\sigma_{r+1}`,
    the update equalizes the top-:math:`r` singular values and scales the tail by :math:`1 / (c_{\tau}\sigma_{r+1})`.
    When :math:`r \ge \min(m, n)`, the tail vanishes and Echelon reduces directly to full-rank Newton-Schulz.
    :math:`\mathrm{AdjustLR}` is the learning-rate scaling selected by ``adjust_lr_fn``, as in
    :class:`torch.optim.Muon`, and :math:`\epsilon` (``eps``) floors the Frobenius norm that
    normalizes :math:`X_t` before these steps (matching Muon's ``NS_k(X / max(||X||_F, eps))``
    convention on the head and full-rank path while scaling the tail by ``min(1, ||X||_F / eps)``).

    .. note::
        This implementation computes orthonormalization, head Newton-Schulz, and explicit tail
        deflation in ``torch.float32``, with autocast disabled inside the step. The roundoff floor
        :math:`\epsilon_{\tau} = 16\sqrt{s}\,\epsilon_{\text{fp32}}\lVert H_t \rVert_F` keeps
        rank-deficient momenta from amplifying float32 cancellation noise, and the lower bound
        :math:`\lVert T_t \rVert_F / \sqrt{s - r}` bounds worst-case tail amplification (to at most
        :math:`\sqrt{s - r} / c_{\tau}` if the probe vector happens to be orthogonal to the dominant
        tail direction); if the power iteration returns zero, :math:`\lVert T_t \rVert_F` replaces it.
        The floor assumes IEEE float32 matmuls. As for other optimizers, matmul precision follows the
        caller's settings (``torch.backends.cuda.matmul.fp32_precision``,
        ``torch.backends.mkldnn.matmul.fp32_precision`` and :func:`torch.set_float32_matmul_precision`);
        with TF32 or bfloat16 matmuls, the roundoff tail of a rank-deficient momentum is amplified
        (update spectral norm up to about 1.5 instead of 1.1). The first step warms up :math:`U` and
        :math:`z` on :math:`X_t`, discarding the updates; a zero or non-finite first update leaves
        them at their random start and the warmup pending, retried on the first finite, nonzero update
        (recorded in optimizer state, so a checkpoint preserves it). Changing a group's ``rank``
        re-initializes them the same way at the next step, and parameters with ``rank >= s`` drop them.
        ``subspace_frame`` and ``tail_vec`` are kept in ``torch.float32``, and :meth:`load_state_dict`
        preserves their ``float32`` precision even on ``bfloat16``/``float16`` parameters. DTensor
        parameters, for example under FSDP2, are rejected with ``NotImplementedError`` rather than
        failing inside the step.
        Defaults (``lr=1e-3``, ``weight_decay=0.1``) mirror :class:`torch.optim.Muon`;
        the Echelon blog post uses ``lr=0.02``, ``weight_decay=0.0``, and ``rank = max(64, width // 16)``.

    For further details see `The Echelon Optimizer`_.
    """
    + rf"""
    Args:
        {_params_doc}. Echelon supports 2D matrix parameters and is intended for hidden-layer weights.
            Non-2D parameters (biases, norms), as well as 2D embeddings and output heads, should use
            a standard optimizer such as AdamW.
        lr (float, Tensor, optional): learning rate (default: 1e-3). Supports 0D/1-element CUDA
            tensors without host synchronization or breaking CUDA graph capture.
        weight_decay (float, optional): decoupled weight decay penalty (default: 0.1).
        momentum (float, optional): momentum factor in ``[0, 1)`` (default: 0.95).
        nesterov (bool, optional): enables Nesterov momentum (default: True).
        rank (int, optional): tracked top singular subspace rank :math:`r`, a Python ``int`` >= 1
            (default: {DEFAULT_RANK}).
        tau_iters (int, optional): power iterations used to estimate the tail spectral norm, a Python
            ``int`` >= 1 (default: {DEFAULT_TAU_ITERS}).
        margin (float, optional): multiplicative safety margin on the tail spectral norm estimate (default: {DEFAULT_MARGIN}).
        ns_coefficients (tuple of three floats, optional): coefficients :math:`(a, b, c)` for the
            Newton-Schulz orthogonalization of the head, or of the whole update when
            :math:`r \ge \min(m, n)` (default: ({DEFAULT_A}, {DEFAULT_B}, {DEFAULT_C})).
        eps (float, optional): floor on the Frobenius norm used to normalize the momentum
            (default: {EPS}).
        ns_steps (int, optional): number of those Newton-Schulz steps, a Python ``int`` in [1, 100)
            (default: {DEFAULT_NS_STEPS}).
        adjust_lr_fn (str, optional): learning-rate scaling rule. One of ``"original"``,
            ``"match_rms_adamw"``, or ``"spectral_unclamped"`` (default: None, equivalent to ``"original"``).
        {_foreach_doc} The foreach implementation orthonormalizes with shifted Cholesky-QR2 instead
            of Householder QR and batches same-shape parameters of up to :math:`2^{{22}}` elements
            (computing Nesterov updates per chunk to bound temporary memory); larger parameters are
            stepped one at a time without stacking. A parameter whose Cholesky-QR2 output is
            non-finite or non-orthonormal (rank-deficient or very ill-conditioned momentum) keeps
            its previous orthonormal frame for that step.

    Example:
        >>> # xdoctest: +SKIP
        >>> from echelon import echelon_adamw_pair
        >>> optim_echelon, optim_adamw = echelon_adamw_pair(model)

    .. _The Echelon Optimizer:
        https://evolvingmachines.ai/blog/echelon
    """
)


def _check_echelon_inputs(
    params: list[Tensor],
    grads: list[Tensor],
    momentum_bufs: list[Tensor],
    subspace_frames: list[Tensor | None],
    tail_vecs: list[Tensor | None],
    *,
    rank: int,
    tau_iters: int,
    margin: float,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    has_complex: bool,
    validate_options: bool = True,
) -> tuple[float, float, float]:
    if has_complex:
        raise RuntimeError("Echelon does not support complex parameters")
    if not (
        len(params)
        == len(grads)
        == len(momentum_bufs)
        == len(subspace_frames)
        == len(tail_vecs)
    ):
        raise ValueError(
            "params, grads, momentum_bufs, subspace_frames, and tail_vecs must have matching lengths"
        )
    if validate_options:
        ns_coefficients = _validate_echelon_options(
            rank=rank,
            tau_iters=tau_iters,
            margin=margin,
            ns_steps=ns_steps,
            ns_coefficients=ns_coefficients,
            eps=eps,
        )
    for i, (param, grad, buf) in enumerate(zip(params, grads, momentum_bufs)):
        if _is_dtensor(param) or _is_dtensor(grad) or _is_dtensor(buf):
            raise NotImplementedError(
                "Echelon does not support DTensor parameters "
                "(for example under FSDP2)"
            )
        if torch.is_complex(param) or torch.is_complex(grad) or torch.is_complex(buf):
            raise RuntimeError("Echelon does not support complex parameters")
        if grad.is_sparse or buf.is_sparse:
            raise RuntimeError("Echelon does not support sparse gradients")
        if param.ndim != 2 or grad.ndim != 2 or buf.ndim != 2:
            raise ValueError("Param, gradient, and momentum_buffer must be 2D matrices")
        if grad.shape != param.shape or buf.shape != param.shape:
            raise ValueError(
                f"Expected grad and momentum_buffer shape {tuple(param.shape)} at index {i}, "
                f"got {tuple(grad.shape)} and {tuple(buf.shape)}"
            )
        if grad.device != param.device or buf.device != param.device:
            raise ValueError(
                f"Param, grad, and momentum_buffer at index {i} must be on the same device, "
                f"got {param.device}, {grad.device}, and {buf.device}"
            )
        s, l = min(param.size(0), param.size(1)), max(param.size(0), param.size(1))
        if param.numel() > 0 and rank < s:
            u_i, z_i = subspace_frames[i], tail_vecs[i]
            if u_i is None or z_i is None:
                raise ValueError(
                    f"subspace_frames[{i}] and tail_vecs[{i}] must be provided when rank ({rank}) < short_side ({s})"
                )
            if torch.is_complex(u_i) or torch.is_complex(z_i):
                raise RuntimeError("Echelon does not support complex state tensors")
            if u_i.shape != (s, rank) or z_i.shape != (l, 1):
                raise ValueError(
                    f"Expected subspace_frames[{i}] shape {(s, rank)} and tail_vecs[{i}] shape {(l, 1)}, "
                    f"got {tuple(u_i.shape)} and {tuple(z_i.shape)}"
                )
            if u_i.device != param.device or z_i.device != param.device:
                raise ValueError(
                    f"subspace_frames[{i}] and tail_vecs[{i}] must be on {param.device}, "
                    f"got {u_i.device} and {z_i.device}"
                )
    return ns_coefficients


def _single_tensor_echelon(
    params: list[Tensor],
    grads: list[Tensor],
    momentum_bufs: list[Tensor],
    subspace_frames: list[Tensor | None],
    tail_vecs: list[Tensor | None],
    *,
    lr: float | Tensor,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    rank: int,
    tau_iters: int,
    margin: float,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
    _prevalidated: bool = False,
    _skip_input_checks: bool = False,
) -> None:
    if not _skip_input_checks:
        ns_coefficients = _check_echelon_inputs(
            params,
            grads,
            momentum_bufs,
            subspace_frames,
            tail_vecs,
            rank=rank,
            tau_iters=tau_iters,
            margin=margin,
            ns_coefficients=ns_coefficients,
            ns_steps=ns_steps,
            eps=eps,
            has_complex=has_complex,
            validate_options=not _prevalidated,
        )
    if len(params) == 0:
        return

    lr_val = _to_scalar(lr)
    is_tensor_lr = isinstance(lr_val, Tensor)
    lr_by_dev: dict[torch.device, Tensor] = {}

    def _lr_on(dev: torch.device) -> Tensor:
        cached = lr_by_dev.get(dev)
        if cached is None:
            t = cast(Tensor, lr_val)
            cached = (
                t.to(dtype=torch.float32)
                if t.device == dev
                else t.to(device=dev, dtype=torch.float32)
            )
            lr_by_dev[dev] = cached
        return cached

    for i, param in enumerate(params):
        grad = grads[i]
        buf = momentum_bufs[i]
        buf.lerp_(grad, 1 - momentum)
        if param.numel() == 0:
            continue
        update = grad.lerp(buf, momentum) if nesterov else buf

        transposed = update.size(0) > update.size(1)
        X = update.mT if transposed else update
        O = _echelon_step(
            X,
            subspace_frames[i],
            tail_vecs[i],
            rank=rank,
            ns_coefficients=ns_coefficients,
            ns_steps=ns_steps,
            tau_iters=tau_iters,
            margin=margin,
            eps=eps,
            _owns_input=nesterov,
        )
        if transposed:
            O = O.mT

        if is_tensor_lr:
            lr_dev = _lr_on(param.device)
            if weight_decay != 0.0:
                param.mul_(1.0 - lr_dev * weight_decay)
            O.mul_(-_adjust_lr(lr_dev, adjust_lr_fn, param.shape))
            param.add_(O)
        else:
            lr_f = cast(float, lr_val)
            if weight_decay != 0.0:
                param.mul_(1.0 - lr_f * weight_decay)
            adjusted_lr = cast(float, _adjust_lr(lr_f, adjust_lr_fn, param.shape))
            param.add_(O, alpha=-adjusted_lr)


def _multi_tensor_echelon(
    params: list[Tensor],
    grads: list[Tensor],
    momentum_bufs: list[Tensor],
    subspace_frames: list[Tensor | None],
    tail_vecs: list[Tensor | None],
    *,
    lr: float | Tensor,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    rank: int,
    tau_iters: int,
    margin: float,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
    _prevalidated: bool = False,
    _skip_input_checks: bool = False,
) -> None:
    if not _skip_input_checks:
        ns_coefficients = _check_echelon_inputs(
            params,
            grads,
            momentum_bufs,
            subspace_frames,
            tail_vecs,
            rank=rank,
            tau_iters=tau_iters,
            margin=margin,
            ns_coefficients=ns_coefficients,
            ns_steps=ns_steps,
            eps=eps,
            has_complex=has_complex,
            validate_options=not _prevalidated,
        )
    if len(params) == 0:
        return

    lr_val = _to_scalar(lr)
    is_tensor_lr = isinstance(lr_val, Tensor)
    lr_by_dev: dict[torch.device, Tensor] = {}

    def _lr_on(dev: torch.device) -> Tensor:
        cached = lr_by_dev.get(dev)
        if cached is None:
            t = cast(Tensor, lr_val)
            cached = (
                t.to(dtype=torch.float32)
                if t.device == dev
                else t.to(device=dev, dtype=torch.float32)
            )
            lr_by_dev[dev] = cached
        return cached

    step_kwargs = {
        "rank": rank,
        "ns_coefficients": ns_coefficients,
        "ns_steps": ns_steps,
        "tau_iters": tau_iters,
        "margin": margin,
        "eps": eps,
    }
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [
            cast(list[Tensor | None], params),
            cast(list[Tensor | None], grads),
            cast(list[Tensor | None], momentum_bufs),
        ],
        with_indices=True,
    )
    for (dev_params_opt, dev_grads_opt, dev_bufs_opt), indices in grouped_tensors.values():
        device_params = cast(list[Tensor], dev_params_opt)
        device_grads = cast(list[Tensor], dev_grads_opt)
        device_bufs = cast(list[Tensor], dev_bufs_opt)
        torch._foreach_lerp_(device_bufs, device_grads, 1 - momentum)

        # Under torch.compile, _group_tensors_by_device_and_dtype returns a single group, so
        # batches (and per-device tensor lr decay) are keyed by device as well.
        if weight_decay != 0.0:
            if is_tensor_lr:
                by_dev: dict[torch.device, list[Tensor]] = defaultdict(list)
                for p in device_params:
                    if p.numel() > 0:
                        by_dev[p.device].append(p)
                for dev, dev_ps in by_dev.items():
                    torch._foreach_mul_(dev_ps, 1.0 - _lr_on(dev) * weight_decay)
            else:
                torch._foreach_mul_(
                    device_params, 1.0 - cast(float, lr_val) * weight_decay
                )

        batches: dict[tuple[torch.device, torch.dtype, torch.Size], list[int]]
        batches = defaultdict(list)
        for j, p in enumerate(device_params):
            if p.numel() > 0:
                batches[(p.device, p.dtype, p.shape)].append(j)

        for (dev, _, shape), js in batches.items():
            transposed = shape[0] > shape[1]
            lr_eff: float | Tensor = _lr_on(dev) if is_tensor_lr else cast(float, lr_val)
            adjusted_lr = _adjust_lr(lr_eff, adjust_lr_fn, shape)
            frames = [subspace_frames[indices[j]] for j in js]
            tails = [tail_vecs[indices[j]] for j in js]
            if len(js) == 1 or shape.numel() > _FOREACH_MAX_NUMEL:
                for j, U, z in zip(js, frames, tails):
                    upd = (
                        device_grads[j].lerp(device_bufs[j], momentum)
                        if nesterov
                        else device_bufs[j]
                    )
                    X = upd.mT if transposed else upd
                    O = _echelon_step(
                        X,
                        U,
                        z,
                        orth=_cholesky_qr2,
                        _owns_input=nesterov,
                        **step_kwargs,
                    )
                    O_out = O.mT if transposed else O
                    p = device_params[j]
                    if is_tensor_lr:
                        O_out.mul_(-adjusted_lr)
                        p.add_(O_out)
                    else:
                        p.add_(O_out, alpha=-cast(float, adjusted_lr))
                continue

            low_rank = rank < min(shape)
            chunk = max(1, _FOREACH_MAX_BATCH_NUMEL // shape.numel())
            for c in range(0, len(js), chunk):
                chunk_js = js[c : c + chunk]
                chunk_frames = cast(list[Tensor], frames[c : c + chunk])
                chunk_tails = cast(list[Tensor], tails[c : c + chunk])
                if nesterov:
                    upd_stack = torch.stack([device_grads[j] for j in chunk_js])
                    upd_stack.lerp_(
                        torch.stack([device_bufs[j] for j in chunk_js]), momentum
                    )
                else:
                    upd_stack = torch.stack([device_bufs[j] for j in chunk_js])
                X_batch = upd_stack.mT if transposed else upd_stack
                U = torch.stack(chunk_frames) if low_rank else None
                z = torch.stack(chunk_tails) if low_rank else None
                O = _echelon_step(
                    X_batch,
                    U,
                    z,
                    orth=_cholesky_qr2,
                    _owns_input=True,
                    **step_kwargs,
                )
                if U is not None and z is not None:
                    torch._foreach_copy_(chunk_frames, list(U.unbind(0)))
                    torch._foreach_copy_(chunk_tails, list(z.unbind(0)))
                O_batch = O.mT if transposed else O
                chunk_params = [device_params[j] for j in chunk_js]
                if is_tensor_lr:
                    O_batch.mul_(-adjusted_lr)
                    torch._foreach_add_(chunk_params, list(O_batch.unbind(0)))
                else:
                    torch._foreach_add_(
                        chunk_params,
                        list(O_batch.unbind(0)),
                        alpha=-cast(float, adjusted_lr),
                    )


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_echelon)
def echelon(
    params: list[Tensor],
    grads: list[Tensor],
    momentum_bufs: list[Tensor],
    subspace_frames: list[Tensor | None],
    tail_vecs: list[Tensor | None],
    *,
    foreach: bool | None = None,
    lr: float | Tensor,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    rank: int,
    tau_iters: int,
    margin: float,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
    _prevalidated: bool = False,
    _skip_input_checks: bool = False,
) -> None:
    if not _prevalidated:
        _validate_echelon_scalar_hyperparameters(
            momentum=momentum,
            weight_decay=weight_decay,
            adjust_lr_fn=adjust_lr_fn,
        )
        _validate_foreach(foreach)
        ns_coefficients = _validate_echelon_options(
            rank=rank,
            tau_iters=tau_iters,
            margin=margin,
            ns_steps=ns_steps,
            ns_coefficients=ns_coefficients,
            eps=eps,
        )
        if isinstance(lr, Tensor):
            if torch.is_complex(lr) or lr.dtype == torch.bool or lr.numel() != 1:
                raise ValueError("Tensor lr must be a real 1-element tensor")
        elif not _is_real_number(lr) or not math.isfinite(lr) or lr < 0.0:
            raise ValueError(f"Learning rate should be finite and >= 0 but is: {lr!r}")
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable=False, use_fused=False
        )
    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    func = (
        _multi_tensor_echelon
        if foreach and not torch.jit.is_scripting()
        else _single_tensor_echelon
    )
    func(
        params,
        grads,
        momentum_bufs,
        subspace_frames,
        tail_vecs,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        rank=rank,
        tau_iters=tau_iters,
        margin=margin,
        ns_coefficients=ns_coefficients,
        ns_steps=ns_steps,
        eps=eps,
        adjust_lr_fn=adjust_lr_fn,
        has_complex=has_complex,
        _prevalidated=True,
        _skip_input_checks=_skip_input_checks,
    )


echelon.__doc__ = (
    _functional_api_doc.format(optimizer="Echelon")
    + r"""

.. note::
    For each 2D parameter ``p`` with ``s = min(p.shape)`` and ``l = max(p.shape)``:
    when ``rank < s``, ``subspace_frames[i]`` must be a tensor of shape ``(s, rank)``
    with orthonormal columns and ``tail_vecs[i]`` a nonzero tensor of shape ``(l, 1)``.
    Both are updated in place and should be ``torch.float32``: a lower-precision frame is
    rounded to its dtype again at every step. Their shapes and devices are checked, but not
    that the initial frame is orthonormal or that the initial tail vector is nonzero. Valid
    starting values are
    ``torch.linalg.qr(torch.randn(s, rank, device=p.device, dtype=torch.float32))[0].contiguous()`` and
    ``torch.randn(l, 1, device=p.device, dtype=torch.float32)``. :class:`Echelon` also warms them up
    on the first step's update (deferred to the first finite, nonzero update if that
    update is zero or non-finite); without that warmup the first updates follow a random frame
    and can overshoot. When ``rank >= s``, ``None`` may be passed for both entries.
"""
)
