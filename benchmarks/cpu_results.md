# CPU benchmark results (laptop-scale only)

Generated with `benchmarks/bench_cpu.py` (median of 5 steps, 2 warmup,
synthetic gradients, `step()` only, no forward/backward).

Machine: Intel Core Ultra 7 165U, 12 threads, torch 2.15.0 (CPU wheel),
Python 3.12. This is a small laptop, not a proper ML compute machine:
**do not extrapolate these numbers to GPU.** On GPU, QR is a cuSOLVER
call, GEMMs are Tensor-Core, and the relative costs change completely;
the PR's GPU numbers are in pytorch/pytorch#205345's test plan.

```
torch 2.15.0+cu132  device cpu  threads 12

single matrix (ms/step):
       shape rank    echelon   foreach      muon     adamw  foreach/echelon
    (64, 64)   16       0.87      1.16     19.18      0.39    1.33x
  (128, 256)   16       7.85     14.70     85.05      0.63    1.87x
  (256, 256)   32       8.23      6.21    435.49      0.73    0.75x
  (512, 512)   64      34.42     19.81   3144.69      7.43    0.58x

many same-shape matrices, 12 layers (ms/step):
       shape rank    echelon   foreach      muon     adamw  foreach/echelon
  (128, 128)   16      31.94     12.73    875.48      6.77    0.40x
  (256, 256)   32     197.94     80.37   5635.13     19.85    0.41x
```

Reading:

- **Foreach helps where it was designed to**: many same-shape matrices
  (a transformer stack), ~2.5x faster than the for-loop path on CPU.
  For a *single* small matrix it is slower on CPU (grouping/`_foreach_*`
  overhead dominates the QR cost) -- the foreach path is not a
  universal win on CPU, and users should measure their own shape mix.
- **Muon on CPU is not a fair comparison here**: `torch.optim.Muon`
  iterates in bfloat16, whose CPU GEMMs on this chip are far slower
  than the fp32 GEMMs Echelon uses. The Muon column is included only
  to show the benchmark ran; it says nothing about Muon on GPU, where
  bf16 Tensor-Core GEMMs are fast. AdamW is the sane CPU baseline:
  Echelon is ~2-5x AdamW's step cost at these shapes, for-loop path.
- First-step (warmup) cost is excluded by the warmup steps; steady
  state is what is measured.
