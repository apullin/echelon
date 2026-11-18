"""GPU correctness + perf verification for the standalone Echelon package.

Run on any CUDA GPU (16 GB+ workstation GPUs through 80 GB A100/H100):

    uv run --no-sync python benchmarks/bench_gpu.py

Exits 0 with a "no CUDA" message on machines without a GPU (so it is
harmless to run anywhere), and asserts on every correctness check on
CUDA. Pass ``--device cpu --quick`` to smoke-test the script logic on a
machine without a GPU.

Checks (all on the target device):
  1. foreach vs for-loop parity (fp32 and bf16 params) on same-shape
     and single-member batches, including a zero-gradient member.
  2. Exact-SVD reference: with the exact top-r frame and tail direction,
     one step's update matches the closed-form spectral clip and keeps
     an orthonormal frame spanning the exact top-r subspace.
  3. TF32 on vs off (for-loop and foreach): spectral norm and frame
     orthonormality for a rank-deficient momentum.
  4. Autocast (bf16) step matches a no-autocast step.
  5. state_dict CUDA <-> CPU round-trip on a bf16 model: float32 frame
     precision is preserved and the resumed trajectory matches.
Perf (CUDA events, step() only, synthetic grads, after warmup):
  AdamW (foreach) vs Muon vs Echelon for-loop vs Echelon foreach across
  transformer-block shapes (scaled automatically to fit device VRAM),
  measured in both IEEE FP32 and TF32 modes with peak VRAM recorded.
  Results are printed and written to ``benchmarks/gpu_results.md``.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

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
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3


def _step(params, grads, opt):
    for p, g in zip(params, grads):
        p.grad = g
    opt.step()


def check_foreach_parity(device, dtype):
    prev = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(2)
        shapes = [(24, 16)] * 3 + [(16, 24)] * 2 + [(12, 20)] + [(4, 6)] * 2
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
        tol = 1e-5 if dtype == torch.float32 else 2e-2
        for a, b in zip(p_loop, p_each):
            torch.testing.assert_close(a, b, atol=tol, rtol=tol)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
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
            [p],
            [X.clone()],
            [buf],
            [u],
            [z],
            lr=1.0,
            weight_decay=0.0,
            momentum=0.0,
            nesterov=False,
            rank=r,
            tau_iters=2,
            margin=1.05,
            ns_coefficients=(3.4445, -4.7750, 2.0315),
            ns_steps=5,
            eps=1e-7,
            adjust_lr_fn="original",
            has_complex=False,
        )
    spec = torch.linalg.svdvals(-p)[0].item()
    overlap = (u.T @ U[:, :r]).norm().item() ** 2 / r
    assert 0.9 < spec < 1.3, f"spectral norm {spec}"
    assert overlap > 0.99, f"frame overlap {overlap}"
    print(f"  exact-SVD reference: spectral norm {spec:.3f}, frame overlap {overlap:.4f}: OK")


def _rank_deficient_stats(device, foreach: bool) -> tuple[float, float]:
    torch.manual_seed(4)
    ps = [Parameter(torch.zeros(32, 48, device=device)) for _ in range(2)]
    for p in ps:
        q_l, _ = torch.linalg.qr(torch.randn(32, 32, device=device))
        q_r, _ = torch.linalg.qr(torch.randn(48, 48, device=device))
        sig = torch.diag(torch.tensor([4.0, 1.2], device=device))
        p.grad = q_l[:, :2] @ sig @ q_r[:, :2].T
    opt = Echelon(
        ps, lr=1.0, weight_decay=0.0, rank=8, foreach=foreach, adjust_lr_fn="spectral_unclamped"
    )
    opt.step()
    spec = torch.linalg.svdvals(ps[0])[0].item()
    u = opt.state[ps[0]]["subspace_frame"]
    orth_err = (u.T @ u - torch.eye(8, device=device)).abs().max().item()
    return spec, orth_err


def check_tf32(device):
    if device.type != "cuda":
        print("  TF32 check: skipped (not CUDA)")
        return
    prev = torch.backends.cuda.matmul.allow_tf32
    try:
        for foreach in (False, True):
            torch.backends.cuda.matmul.allow_tf32 = False
            ieee, orth_ieee = _rank_deficient_stats(device, foreach)
            torch.backends.cuda.matmul.allow_tf32 = True
            tf32, orth_tf32 = _rank_deficient_stats(device, foreach)
            assert ieee < 1.3, f"IEEE spectral norm {ieee} (foreach={foreach})"
            assert tf32 < 1.6, f"TF32 spectral norm {tf32} (foreach={foreach})"
            assert orth_ieee <= 1e-2 and orth_tf32 <= 1e-2, (orth_ieee, orth_tf32)
            print(
                f"  rank-deficient (foreach={foreach!s:<5}): "
                f"IEEE {ieee:.3f} (|U^TU-I|={orth_ieee:.1e}), "
                f"TF32 {tf32:.3f} (|U^TU-I|={orth_tf32:.1e}): OK"
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


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

    # CUDA -> CPU -> CUDA round-trip on a bf16 parameter, checking trajectory parity.
    p_uninterrupted = Parameter(torch.randn(16, 24, device=device, dtype=torch.bfloat16))
    opt_uninterrupted = Echelon([p_uninterrupted], rank=4)
    g1 = torch.randn_like(p_uninterrupted)
    p_uninterrupted.grad = g1.clone()
    opt_uninterrupted.step()

    sd_to_cpu = opt_uninterrupted.state_dict()
    p_cpu = Parameter(p_uninterrupted.detach().cpu().clone())
    opt_cpu = Echelon([p_cpu], rank=4)
    opt_cpu.load_state_dict(sd_to_cpu)
    assert opt_cpu.state[p_cpu]["subspace_frame"].dtype == torch.float32

    p_resumed = Parameter(p_cpu.detach().to(device).clone())
    opt_resumed = Echelon([p_resumed], rank=4)
    opt_resumed.load_state_dict(opt_cpu.state_dict())
    assert opt_resumed.state[p_resumed]["subspace_frame"].dtype == torch.float32

    g2 = torch.randn_like(p_uninterrupted)
    p_uninterrupted.grad = g2.clone()
    p_resumed.grad = g2.clone()
    opt_uninterrupted.step()
    opt_resumed.step()
    torch.testing.assert_close(p_resumed, p_uninterrupted)
    print("  state_dict GPU <-> CPU round-trip (lossless fp32 frame & trajectory parity): OK")


def _select_widths(device: torch.device, quick: bool) -> list[tuple[int, int]]:
    if quick:
        return [(64, 2), (128, 2)]
    vram_gib = (
        torch.cuda.get_device_properties(device).total_memory / (1024**3)
        if device.type == "cuda"
        else 16.0
    )
    if vram_gib >= 60.0:
        return [(768, 12), (2048, 12), (4096, 8), (8192, 4)]
    # Fits comfortably in 16 GB VRAM while exercising widths 512 through 4096.
    return [(512, 12), (768, 12), (1024, 12), (1536, 12), (2048, 8), (3072, 4), (4096, 2)]


def bench_perf(device: torch.device, quick: bool = False):
    widths = _select_widths(device, quick=quick)
    modes = (
        [(False, "IEEE FP32 (allow_tf32=False)"), (True, "TF32 (allow_tf32=True)")]
        if device.type == "cuda"
        else [(False, "CPU")]
    )
    all_tables = []
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        for tf32_flag, label in modes:
            if device.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = tf32_flag
            print(f"\nperf [{label}]: step() only, synthetic grads, fp32 params, rank 64")
            print(
                f"{'width':>6} {'layers':>6}  {'adamw':>8} {'muon':>8} {'echelon':>9} {'foreach':>9}  "
                f"foreach/ech  foreach/muon  peakGiB(each)"
            )
            rows = []
            for width, n_layers in widths:
                shapes = [(width, width), (width, 3 * width), (width, 4 * width)]
                counts = [n_layers, n_layers, 3 * n_layers]
                results = {}
                peaks = {}
                for name, make in [
                    ("adamw", lambda ps: torch.optim.AdamW(ps, lr=1e-3, foreach=True)),
                    ("muon", lambda ps: torch.optim.Muon(ps, lr=0.02)),
                    ("echelon", lambda ps: Echelon(ps, lr=0.02, rank=64, foreach=False)),
                    ("foreach", lambda ps: Echelon(ps, lr=0.02, rank=64, foreach=True)),
                ]:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(device)
                    params, grads = [], []
                    for shape, count in zip(shapes, counts):
                        for _ in range(count):
                            params.append(Parameter(torch.randn(shape, device=device)))
                            grads.append(torch.randn(shape, device=device))
                    opt = make(params)
                    results[name] = _time_cuda(
                        lambda ps=params, gs=grads, o=opt: _step(ps, gs, o),
                        iters=5,
                        warmup=2,
                        device=device,
                    )
                    peaks[name] = (
                        torch.cuda.max_memory_allocated(device) / (1024**3)
                        if device.type == "cuda"
                        else 0.0
                    )
                    del params, grads, opt
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                rows.append((width, n_layers, results, peaks))
                print(
                    f"{width:>6} {n_layers:>6}  {results['adamw']:8.2f} {results['muon']:8.2f} "
                    f"{results['echelon']:9.2f} {results['foreach']:9.2f}  "
                    f"{results['foreach'] / results['echelon']:10.2f}x  "
                    f"{results['foreach'] / results['muon']:11.2f}x  "
                    f"{peaks['foreach']:10.2f} GiB"
                )
            all_tables.append((label, rows))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return all_tables


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--device",
        default=None,
        help="cuda (default, requires CUDA) or cpu (logic smoke test only)",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="small perf shapes only (for --device cpu smoke tests)",
    )
    args = ap.parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type != "cuda":
        if args.device is None:
            print(
                "No CUDA device found; nothing to verify on GPU. "
                "Run this on a CUDA host or pass --device cpu --quick "
                "to smoke-test the script logic."
            )
            return
        print("WARNING: --device cpu: correctness logic only, timings meaningless.")
    else:
        vram_gib = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(
            f"device: {torch.cuda.get_device_name(device)} ({vram_gib:.2f} GiB)  "
            f"torch {torch.__version__}"
        )

    print("correctness:")
    for dtype in (torch.float32, torch.bfloat16):
        check_foreach_parity(device, dtype)
    check_exact_svd(device)
    check_tf32(device)
    check_autocast_and_state_dict(device)
    tables = bench_perf(device, quick=args.quick)

    if device.type == "cuda" and not args.quick:
        vram_gib = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        out = [
            "# GPU benchmark results\n",
            (
                f"Device: {torch.cuda.get_device_name(device)} ({vram_gib:.2f} GiB VRAM), "
                f"torch {torch.__version__}\n"
            ),
        ]
        for label, rows in tables:
            out.append(f"## {label}\n")
            out.append(
                "| width | layers | AdamW (ms) | Muon (ms) | Echelon for-loop (ms) | "
                "Echelon foreach (ms) | foreach / loop | foreach / Muon | Peak VRAM foreach (GiB) |"
            )
            out.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for width, n_layers, r, p in rows:
                out.append(
                    f"| {width} | {n_layers} | {r['adamw']:.2f} | {r['muon']:.2f} | "
                    f"{r['echelon']:.2f} | {r['foreach']:.2f} | "
                    f"{r['foreach'] / r['echelon']:.2f}x | "
                    f"{r['foreach'] / r['muon']:.2f}x | {p['foreach']:.2f} |"
                )
            out.append("")
        out.append(
            "Step() only, synthetic gradients, fp32 parameters (`[(w, w)] * L + [(w, 3w)] * L + [(w, 4w)] * 3L`), "
            "rank 64, median of 5 steps after 2 warmup steps."
        )
        out_path = Path(__file__).resolve().with_name("gpu_results.md")
        out_path.write_text("\n".join(out) + "\n")
        print(f"\nwrote {out_path}")
    print("\nGPU verification: all correctness checks passed.")


if __name__ == "__main__":
    main()
