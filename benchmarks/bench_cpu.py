"""CPU step-time benchmark (laptop-scale; NOT a GPU claim).

Measures optimizer ``step()`` only, on synthetic gradients, after
warmup: for-loop Echelon vs foreach Echelon vs torch.optim.Muon vs
AdamW. On CPU the Householder QR in the for-loop path is the dominant
cost, so expect Echelon (for-loop) to be much slower than Muon here;
the foreach path batches QR via Cholesky-QR and is the interesting
comparison at many-small-matrices shapes. GPU numbers (where QR is a
cuSOLVER call and GEMMs are Tensor-Core) will differ substantially --
see benchmarks/cpu_results.md for the run that produced it.

Usage (uv env, repo root):

    uv run --no-sync python benchmarks/bench_cpu.py [--iters N] [--warmup N]
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from torch.nn import Parameter

from echelon import Echelon


def _time_step(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3


def _step_only(params, grads, opt):
    for p, g in zip(params, grads):
        p.grad = g
    opt.step()


def bench_single(shape, rank, iters, warmup):
    torch.manual_seed(0)
    results = {}
    for name, make_opt in [
        ("echelon", lambda ps: Echelon(ps, lr=0.02, rank=rank, foreach=False)),
        ("echelon-foreach", lambda ps: Echelon(ps, lr=0.02, rank=rank, foreach=True)),
        ("muon", lambda ps: torch.optim.Muon(ps, lr=0.02)),
        ("adamw", lambda ps: torch.optim.AdamW(ps, lr=1e-3)),
    ]:
        params = [Parameter(torch.randn(shape)) for _ in range(1)]
        grads = [torch.randn(shape) for _ in range(1)]
        opt = make_opt(params)
        ms = _time_step(lambda: _step_only(params, grads, opt), iters, warmup)
        results[name] = ms
    return results


def bench_many(n_layers, shape, rank, iters, warmup):
    """Many same-shape matrices, as in a transformer stack: the foreach case."""
    torch.manual_seed(0)
    results = {}
    for name, make_opt in [
        ("echelon", lambda ps: Echelon(ps, lr=0.02, rank=rank, foreach=False)),
        ("echelon-foreach", lambda ps: Echelon(ps, lr=0.02, rank=rank, foreach=True)),
        ("muon", lambda ps: torch.optim.Muon(ps, lr=0.02)),
        ("adamw", lambda ps: torch.optim.AdamW(ps, lr=1e-3)),
    ]:
        params = [Parameter(torch.randn(shape)) for _ in range(n_layers)]
        grads = [torch.randn(shape) for _ in range(n_layers)]
        opt = make_opt(params)
        ms = _time_step(lambda: _step_only(params, grads, opt), iters, warmup)
        results[name] = ms
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()
    print(f"torch {torch.__version__}  device cpu  threads {torch.get_num_threads()}")
    print(f"median of {args.iters} steps, {args.warmup} warmup, synthetic grads, step() only\n")
    print("single matrix (ms/step):")
    print(f"{'shape':>12} {'rank':>4}  {'echelon':>9} {'foreach':>9} {'muon':>9} {'adamw':>9}  foreach/echelon")
    for shape, rank in [((64, 64), 16), ((128, 256), 16), ((256, 256), 32), ((512, 512), 64)]:
        r = bench_single(shape, rank, args.iters, args.warmup)
        print(
            f"{str(shape):>12} {rank:>4}  {r['echelon']:9.2f} {r['echelon-foreach']:9.2f} "
            f"{r['muon']:9.2f} {r['adamw']:9.2f}  {r['echelon-foreach'] / r['echelon']:6.2f}x"
        )
    print("\nmany same-shape matrices, 12 layers (ms/step):")
    print(f"{'shape':>12} {'rank':>4}  {'echelon':>9} {'foreach':>9} {'muon':>9} {'adamw':>9}  foreach/echelon")
    for shape, rank in [((128, 128), 16), ((256, 256), 32)]:
        r = bench_many(12, shape, rank, args.iters, args.warmup)
        print(
            f"{str(shape):>12} {rank:>4}  {r['echelon']:9.2f} {r['echelon-foreach']:9.2f} "
            f"{r['muon']:9.2f} {r['adamw']:9.2f}  {r['echelon-foreach'] / r['echelon']:6.2f}x"
        )


if __name__ == "__main__":
    main()
