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

The package is not on PyPI yet; install it from GitHub (this installs
PyTorch too, if it is not already installed):

```sh
pip install "git+https://github.com/apullin/echelon.git"
```

Requires Python ≥ 3.11 and PyTorch ≥ 2.9.

## Usage

`Echelon` only accepts 2D parameters, so the usual setup is two
optimizers. The one-call version is `echelon_adamw_pair`, which
partitions a model's parameters (2D matrices that are not embeddings
or heads go to Echelon, everything else to AdamW) and builds both
optimizers; step both each training step. See
`examples/train_toy.py` for a complete runnable version.

```python
from echelon import echelon_adamw_pair

opt_echelon, opt_adamw = echelon_adamw_pair(model)
# ... loss.backward(); clip_grad_norm_(...); opt_echelon.step(); opt_adamw.step()
```

Or partition by hand, e.g. to use a different second optimizer:

```python
from echelon import Echelon, partition_parameters
import torch

hidden, rest = partition_parameters(model)  # exclude_substrings=("embed", "head")
opt_echelon = Echelon(hidden, lr=0.02, weight_decay=0.0, rank=64)
opt_adamw = torch.optim.AdamW(rest, lr=3e-4, weight_decay=0.01)
```

Passing `model.parameters()` directly to `Echelon` raises a `ValueError`
pointing at `partition_parameters`, rather than failing obscurely.

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
    --extra-index-url https://pypi.org/simple "torch>=2.9" pytest
uv pip install -e . --no-deps
uv run --no-sync pytest
```

## Benchmarks

`benchmarks/bench_gpu.py` is the GPU correctness + perf script; it
can only run on a GPU OD -- see `docs/gpu-od.md` for the runbook
(it cannot be driven from the sandboxed laptop session, and GPU
remote execution only runs buck targets in fbsource, whose copy of
the optimizer is stale).

`benchmarks/bench_cpu.py` measures `step()`-only time on CPU and
`benchmarks/cpu_results.md` records a laptop run, with its caveats
(CPU QR costs and Muon's bf16 CPU GEMMs do not transfer to GPU).
There are no GPU benchmark numbers from this package yet.

## Status

- `0.2.0`: the unbatched (for-loop) optimizer from PR #205345, plus the
  batched `foreach` path from internal diff D132660022 (second commit),
  plus `partition_parameters` / `echelon_adamw_pair` and the audit fixes
  in the fourth commit.
- The 9 non-blocking findings from the last automated review of PR #205345
  (0594e6e) are addressed in the third commit: up-front `adjust_lr_fn`
  validation, deferred warmup after a zero/non-finite first update,
  early DTensor rejection, the contiguous-frame docstring example, the
  argument-validation test placement, and the fused/in-place step with
  single option validation. There is no XPU tolerance override in this
  package (that finding is not applicable here).
- Known limitations are documented in `Echelon`'s docstring (DTensor,
  TF32 matmul precision).
