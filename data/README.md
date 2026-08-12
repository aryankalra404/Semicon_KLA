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

Training pairs match exactly by filename. The blind test files are independently
numbered `000000.npy` through `000399.npy`; those names do **not** identify the
same images as the equally numbered training targets. Signal-correlation checks
confirm that the test ground truths are withheld.

Use paired training IDs 0-2879 for training and 2880-3199 for validation. Run
final inference on all 400 blind test inputs without computing reference-based
metrics on them.

Observed value ranges:

- Ground truth is float32 and bounded to `[0, 1]`.
- Degraded inputs can be below 0 or above 1; do not clip or independently
  min-max normalize them before the model.
