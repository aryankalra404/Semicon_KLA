# KLA public dataset layout

The official extracted arrays are organized without renaming any files:

```text
data/
├── train/
│   ├── GT/       # 3,200 clean 256x256 float32 targets
│   └── NoisyLR/  # 3,200 degraded 128x128 float32 inputs
└── test/
    └── NoisyLR/  # 400 degraded 128x128 float32 inputs
```

Training pairs match exactly by filename. The test files are numbered
`000000.npy` through `000399.npy`; matching ground-truth arrays exist under
`train/GT`, while the corresponding test degradations differ from the arrays
under `train/NoisyLR`.

To prevent target leakage in honest evaluation, reserve IDs 0-399 for testing
and train only on IDs 400-3199 unless the competition later publishes a
different split protocol.

Observed value ranges:

- Ground truth is float32 and bounded to `[0, 1]`.
- Degraded inputs can be below 0 or above 1; do not clip or independently
  min-max normalize them before the model.
