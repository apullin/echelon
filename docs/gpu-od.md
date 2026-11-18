# GPU verification runbook (GPU On Demand)

Local laptops have no GPU, and the standalone package cannot be tested
on GPU remote execution: buck2/RE only runs buck targets inside
fbsource, whose copy of `_echelon.py` is the stale pre-standalone diff
version. GPU verification therefore runs interactively on a GPU On
Demand (1× A100 80G). It cannot be driven from the sandboxed `pi`
session on a laptop: `ondemand connect -t gpu` fails there with a
DevEnv API stream error before any 2FA prompt, and connecting to an
existing devserver fails the same way. Run the steps below in your
own terminal, where the OD tooling and 2FA work.

## 1. Reserve the GPU OD

```sh
ondemand connect -t gpu --expiration 1 --name echelon-gpu
```

Keep that terminal open for the session. Release it when done:

```sh
ondemand release
```

## 2. On the OD: install and test

The OD has CUDA; install the CUDA torch wheel (cu132 matches torch
2.15; use the index for whatever torch version `uv pip` resolves if
you change it), then the package in editable mode:

```sh
gh repo clone apullin/echelon ~/echelon && cd ~/echelon
uv venv
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu132 \
  --extra-index-url https://pypi.org/simple "torch>=2.9" pytest
uv pip install --python .venv/bin/python -e . --no-deps
uv run --no-sync python -c "import torch; print(torch.cuda.get_device_name(0))"
uv run --no-sync pytest -q
```

## 3. Run the GPU correctness + perf script

```sh
uv run --no-sync python benchmarks/bench_gpu.py
```

It asserts: foreach/for-loop parity (fp32 + bf16), the exact-SVD
reference (update spectral norm ~1.0–1.2), TF32 on/off spectral norms
for a rank-deficient momentum, autocast parity, and a CPU↔CUDA
state_dict round-trip; then it times Muon vs Echelon for-loop vs
foreach at widths 768/2048/4096/8192 (rank 64) and writes
`benchmarks/gpu_results.md`. Commit that file from the OD (or copy it
back) with the device name and torch version in the commit message.

## 4. What is *not* covered here

- Multi-GPU / FSDP2: DTensor parameters are rejected by design; a
  2-GPU run would only re-confirm the rejection.
- `torch.compile` parity on CUDA: not scripted yet; the CPU test suite
  is the only compiled-parity coverage so far.
- XPU: no XPU device is available on GPU ODs; the standalone package
  deliberately has no XPU tolerance override.
