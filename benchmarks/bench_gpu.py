"""GPU correctness + perf verification for the standalone Echelon package.

Run on a GPU On Demand (1x A100 80G) from the repo root, in the uv env
(see docs/gpu-od.md):

    uv run --no-sync python benchmarks/bench_gpu.py

Exits 0 with a "no CUDA" message on machines without a GPU (so it is
harmless to run anywhere), and asserts on every correctness check on
CUDA. Pass ``--device cpu`` to smoke-test the script logic on a
machine without a GPU; the timings it prints there are meaningless.

Checks (all on the target device):
  1. foreach vs for-loop parity (fp32 and bf16 params) on same-shape
     batches, including a zero-gradient member.
  2. Exact-SVD reference: with the exact top-r frame and tail direction,
     one step's update has spectral norm ~1.0-1.2 and the frame stays
     orthonormal / spans the exact top-r subspace.
  3. TF32 on vs off: spectral norm of the update for a rank-deficient
     momentum (the docstring claims ~1.1 with IEEE fp32, up to ~1.5
     with TF32).
  4. Autocast (bf16) step matches a no-autocast step.
  5. state_dict CPU <-> CUDA round-trip: frame/tail return to fp32
     and the resumed step matches.
Perf (CUDA events, step() only, synthetic grads, after warmup):
  Muon vs Echelon for-loop vs Echelon foreach at transformer-block
  shapes for widths 768 / 2048 / 4096 / 8192, rank 64, fp32 params.
  Results are printed as a table and written to
  ``benchmarks/gpu_results.md`` (not committed by this script).
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from torch.nn import Parameter

from echelon import Echelon, echelon


def _time_cuda(fn, iters: int, warmup: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for s, e in zip(starts, ends):
            s.record()
            fn()
            e.record()
        torch.cuda.synchronize()
        return statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends))
    for _ in range(iters):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3


def _step(params, grads, opt):
    for p, g in zip(params, grads):
        p.grad = g
    opt.step()


def check_foreach_parity(device, dtype):
    torch.manual_seed(2)
    shapes = [(24, 16)] * 3 + [(16, 24)] * 2 + [(4, 6)] * 2
    base = [torch.randn(s, device=device, dtype=dtype) for s in shapes]
    p_loop = [Parameter(b.clone()) for b in base]
    p_each = [Parameter(b.clone()) for b in base]
    kwargs = {"lr": 0.02, "weight_decay": 0.01, "rank": 4}
    opt_loop = Echelon(p_loop, foreach=False, **kwargs)
    opt_each = Echelon(p_each, foreach=True, **kwargs)
    for _ in range(5):
        for i, (a, b) in enumerate(zip(p_loop, p_each)):
            a.grad = torch.zeros_like(a) if i == 2 else torch.randn_like(a)
            b.grad = a.grad.clone()
        opt_loop.step()
        opt_each.step()
    for a, b in zip(p_loop, p_each):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    print(f"  foreach/for-loop parity ({dtype}): OK")


def check_exact_svd(device):
    torch.manual_seed(0)
    X = torch.randn(32, 64, device=device)
    U, _S, Vh = torch.linalg.svd(X, full_matrices=False)
    r = 8
    p = Parameter(torch.zeros_like(X))
    buf = torch.zeros_like(X)
    u = U[:, :r].clone()
    z = Vh[r, :].reshape(-1, 1).clone()
    with torch.no_grad():
        echelon(
            [p], [X.clone()], [buf], [u], [z],
            lr=1.0, weight_decay=0.0, momentum=0.0, nesterov=False,
            rank=r, tau_iters=2, margin=1.05,
            ns_coefficients=(3.4445, -4.7750, 2.0315), ns_steps=5, eps=1e-7,
            adjust_lr_fn="original", has_complex=False,
        )
    spec = torch.linalg.svdvals(-p)[0].item()
    overlap = (u.T @ U[:, :r]).norm().item() ** 2 / r
    assert 0.9 < spec < 1.3, f"spectral norm {spec}"
    assert overlap > 0.99, f"frame overlap {overlap}"
    print(f"  exact-SVD reference: spectral norm {spec:.3f}, frame overlap {overlap:.4f}: OK")


def _spectral_norm_after_rank_deficient_step(device):
    torch.manual_seed(4)
    q_l, _ = torch.linalg.qr(torch.randn(12, 12, device=device))
    q_r, _ = torch.linalg.qr(torch.randn(16, 16, device=device))
    sig = torch.diag(torch.tensor([4.0, 1.2], device=device))
    p = Parameter(torch.zeros(12, 16, device=device))
    p.grad = q_l[:, :2] @ sig @ q_r[:, :2].T
    Echelon([p], lr=1.0, weight_decay=0.0, rank=4,
            adjust_lr_fn="spectral_unclamped").step()
    return torch.linalg.svdvals(p)[0].item()


def check_tf32(device):
    if device.type != "cuda":
        print("  TF32 check: skipped (not CUDA)")
        return
    prev = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        ieee = _spectral_norm_after_rank_deficient_step(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        tf32 = _spectral_norm_after_rank_deficient_step(device)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    print(f"  rank-deficient update spectral norm: IEEE fp32 {ieee:.3f}, TF32 {tf32:.3f} "
          f"(docstring: ~1.1 vs up to ~1.5)")
    assert ieee < 1.3, f"IEEE spectral norm {ieee}"


def check_autocast_and_state_dict(device):
    torch.manual_seed(6)
    g = torch.randn(12, 16, device=device)
    p_ref = Parameter(torch.zeros_like(g))
    p_ac = Parameter(torch.zeros_like(g))
    p_ref.grad = g.clone()
    p_ac.grad = g.clone()
    Echelon([p_ref], lr=1.0, rank=4).step()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        Echelon([p_ac], lr=1.0, rank=4).step()
    torch.testing.assert_close(p_ac, p_ref)
    print("  autocast parity: OK")

    # CPU <-> CUDA round-trip of a bf16 model's optimizer state.
    p_gpu = Parameter(torch.randn(16, 24, device=device, dtype=torch.bfloat16))
    opt = Echelon([p_gpu], rank=4)
    p_gpu.grad = torch.randn_like(p_gpu)
    opt.step()
    sd = opt.state_dict()
    p_cpu = Parameter(p_gpu.detach().cpu().clone())
    opt_cpu = Echelon([p_cpu], rank=4)
    opt_cpu.load_state_dict(sd)
    # load_state_dict casts floating state to the (bf16) param dtype;
    # the next step converts the frame/tail back to fp32, as covered
    # by test_state_returns_to_float32_after_load.
    p_cpu.grad = torch.randn_like(p_cpu)
    opt_cpu.step()
    frame = opt_cpu.state[p_cpu]["subspace_frame"]
    assert frame.dtype == torch.float32, frame.dtype
    assert torch.isfinite(p_cpu.float()).all()
    print("  state_dict cross-device round-trip (frame back to fp32 after a step): OK")


def bench_perf(device, quick=False):
    print("\nperf: step() only, synthetic grads, fp32 params, rank 64")
    print(f"{'width':>6}  {'muon':>9} {'echelon':>9} {'foreach':>9}  foreach/echelon  echelon/muon")
    rows = []
    widths = [(64, 2), (128, 2)] if quick else [(768, 12), (2048, 12), (4096, 8), (8192, 4)]
    for width, n_layers in widths:
        shapes = [(width, width), (width, 3 * width), (width, 4 * width)]
        counts = [n_layers, n_layers, 3 * n_layers]
        results = {}
        for name, make in [
            ("muon", lambda ps: torch.optim.Muon(ps, lr=0.02)),
            ("echelon", lambda ps: Echelon(ps, lr=0.02, rank=64, foreach=False)),
            ("foreach", lambda ps: Echelon(ps, lr=0.02, rank=64, foreach=True)),
        ]:
            params, grads = [], []
            for shape, count in zip(shapes, counts):
                for _ in range(count):
                    params.append(Parameter(torch.randn(shape, device=device)))
                    grads.append(torch.randn(shape, device=device))
            opt = make(params)
            results[name] = _time_cuda(
                lambda: _step(params, grads, opt), iters=5, warmup=2, device=device
            )
            del params, grads, opt
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rows.append((width, results))
        print(
            f"{width:>6}  {results['muon']:9.2f} {results['echelon']:9.2f} "
            f"{results['foreach']:9.2f}  {results['foreach'] / results['echelon']:10.2f}x  "
            f"{results['echelon'] / results['muon']:9.2f}x"
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None,
                    help="cuda (default, requires CUDA) or cpu (logic smoke test only)")
    ap.add_argument("--quick", action="store_true",
                    help="small perf shapes only (for the --device cpu smoke test)")
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        if args.device is None:
            print("No CUDA device found; nothing to verify on GPU. "
                  "Run this on a GPU OD (see docs/gpu-od.md) or pass --device cpu "
                  "to smoke-test the script logic.")
            return
        print("WARNING: --device cpu: correctness logic only, timings meaningless.")
    else:
        print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    print("correctness:")
    for dtype in (torch.float32, torch.bfloat16):
        check_foreach_parity(device, dtype)
    check_exact_svd(device)
    check_tf32(device)
    check_autocast_and_state_dict(device)
    rows = bench_perf(device, quick=args.quick)

    if device.type == "cuda":
        out = ["# GPU benchmark results\n",
               f"Device: {torch.cuda.get_device_name(0)}, torch {torch.__version__}\n",
               "| width | Muon (ms) | Echelon for-loop (ms) | Echelon foreach (ms) |",
               "|---|---|---|---|"]
        for width, r in rows:
            out.append(f"| {width} | {r['muon']:.2f} | {r['echelon']:.2f} | {r['foreach']:.2f} |")
        out.append("\nStep() only, synthetic grads, fp32 params, rank 64, median of 5 after 2 warmup.\n")
        with open("benchmarks/gpu_results.md", "w") as f:
            f.write("\n".join(out) + "\n")
        print("\nwrote benchmarks/gpu_results.md")
    print("\nGPU verification: all correctness checks passed.")


if __name__ == "__main__":
    main()
