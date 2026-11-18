# Echelon optimizer for PyTorch

Standalone implementation of the **Echelon** optimizer: rank-anchored
spectral clipping in `O(mnr)` FLOPs. It tracks an orthonormal top-`r`
singular frame of the momentum, orthogonalizes the small head with quintic
Newton–Schulz, and divides the orthogonal complement by an estimate of its
largest singular value.

- Algorithm description: https://evolvingmachines.ai/blog/echelon
  (technical report: https://evolvingmachines.ai/echelon/technical-report.pdf).
  The algorithm is not the author's original work; this package is an
  independent implementation following the public description.
- History: this code was first submitted to PyTorch Core as
  [pytorch/pytorch#205345](https://github.com/pytorch/pytorch/pull/205345),
  which Core closed without merging — new optimizers are expected to prove
  themselves as separate libraries first (see the discussion in
  [pytorch/pytorch#205526](https://github.com/pytorch/pytorch/issues/205526),
  which this package does not maintain). The API mirrors
  `torch.optim.Muon` so the two can be swapped for hidden-layer weights.

## Install

```sh
pip install echelon-optimizer
```

Requires Python ≥ 3.11 and PyTorch ≥ 2.6.

## Usage

```python
from echelon import Echelon
import torch

hidden = [p for n, p in model.named_parameters()
          if p.ndim == 2 and "embed" not in n and "head" not in n]
other = [p for n, p in model.named_parameters()
         if p.ndim != 2 or "embed" in n or "head" in n]

opt_echelon = Echelon(hidden, lr=0.02, weight_decay=0.0, rank=64)
opt_adamw = torch.optim.AdamW(other, lr=3e-4, weight_decay=0.01)
```

The functional API mirrors `torch.optim.functional`:

```python
from echelon import echelon
echelon(params, grads, momentum_bufs, subspace_frames, tail_vecs,
        lr=..., weight_decay=..., momentum=..., nesterov=...,
        rank=..., tau_iters=..., margin=..., ns_coefficients=...,
        ns_steps=..., eps=..., adjust_lr_fn=..., has_complex=False)
```

`Echelon` supports 2D parameters only, intended for hidden-layer weights.
DTensor parameters (e.g. under FSDP2) are not supported. The `foreach`
implementation batches same-shape parameters and orthonormalizes them with
shifted Cholesky-QR instead of Householder QR; larger parameters are stepped
one at a time. See `Echelon`'s docstring for the full algorithm, defaults
(`lr=1e-3`, `weight_decay=0.1`, mirroring Muon; the blog post uses
`lr=0.02`, `weight_decay=0.0`, `rank = max(64, width // 16)`), and the
TF32/autocast notes.

## Development

All commands run in a [uv](https://docs.astral.sh/uv/) environment:

```sh
uv venv
uv pip install --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple "torch>=2.6" pytest
uv pip install -e . --no-deps
uv run --no-sync pytest
```

## Status

- `0.1.0`: the unbatched (for-loop) optimizer from PR #205345, plus the
  batched `foreach` path from internal diff D132660022 (second commit).
- Known limitations are documented in `Echelon`'s docstring (DTensor,
  TF32 matmul precision, cold start after a zero/non-finite first update
  until the first finite, nonzero update warms the frame up).
