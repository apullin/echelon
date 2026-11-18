# GPU verification runbook

GPU verification runs on any Linux CUDA workstation (16 GB+ VRAM) or an
80 GB data-center GPU (such as a GPU On Demand A100/H100).

## 1. Install CUDA PyTorch and Echelon

```sh
uv venv
uv pip install --python .venv/bin/python "torch>=2.9" pytest
uv pip install --python .venv/bin/python -e . --no-deps
uv run --no-sync python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## 2. Run the unit and ecosystem test suite (CPU + CUDA)

When CUDA is available, `pytest` automatically runs all numerical,
ecosystem, `torch.compile`, sync-free steady-state, and `CUDAGraph`
tests on both `cpu` and `cuda`:

```sh
uv run --no-sync pytest -q
```

## 3. Run the GPU correctness + performance benchmark

```sh
uv run --no-sync python benchmarks/bench_gpu.py
```

It asserts:
- `foreach` / for-loop parity (`fp32` + `bf16`),
- exact-SVD reference (update spectral norm ~1.0–1.2 and frame overlap >0.99),
- IEEE `fp32` vs `TF32` rank-deficient spectral norms and frame orthonormality (`foreach=False` and `foreach=True`),
- `bfloat16` autocast parity, and
- lossless `float32` frame preservation across a `CUDA <-> CPU` `state_dict` round-trip on a `bfloat16` model.

It then benchmarks `AdamW (foreach)` vs `Muon` vs `Echelon (for-loop)` vs
`Echelon (foreach)` in both IEEE `fp32` and `TF32` modes (automatically
scaling layer counts to fit available GPU memory, e.g. widths `512–4096`
on 16 GB GPUs and `768–8192` on 60 GB+ GPUs) and writes
`benchmarks/gpu_results.md`.

## 4. Optional: GPU On Demand (internal)

If running on an internal GPU OD instead of a local CUDA host:

```sh
ondemand connect -t gpu --expiration 1 --name echelon-gpu
# run steps 1-3 above, then release when finished:
ondemand release
```

## 5. Known non-goals

- Multi-GPU / FSDP2: `DTensor` parameters are rejected by design with
  `NotImplementedError`.
- XPU: no XPU device is targeted by this standalone package.
