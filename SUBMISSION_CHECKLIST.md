# KLA Phase 1 submission checklist

Run every item from a clean clone before uploading. Do not replace the frozen
`weights/final.pt` or the 400 restored outputs unless a challenger passes the
promotion gates in `select_candidate.py`.

## Repository

- [ ] Public GitHub repository opens without authentication.
- [ ] `README.md` documents setup, training, inference, metrics, and limitations.
- [ ] `weights/final.pt` is present and its SHA-256 is
      `89223db798de64c675385102250ef8a5cdbad2cbf5f893a8d759e7eb2f56b798`.
- [ ] `outputs/restored/` contains exactly 400 float32 arrays with the original
      test filenames and shape `(256, 256)`.
- [ ] `outputs/output_manifest.json` matches those outputs.
- [ ] No dataset, credentials, tokens, passwords, or private paths are committed.

## Evaluator dry run

```bash
docker build -t kla-restorenet:latest .
docker run --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$PWD/data":/workspace/project/data:ro \
  kla-restorenet:latest \
  python validate_submission.py --device cuda

python make_output_manifest.py
python submission_audit.py
```

- [ ] Test once from a new clone, not only the working directory.
- [ ] Record the exact Docker command, GPU, latency protocol, checkpoint hash,
      and output contract in the deck.

## Evidence

- [ ] Report the untouched fixed-split PSNR, SSIM, LPIPS, 95% confidence
      intervals, batch-1 p50/p95 latency, parameters, and peak VRAM.
- [ ] Include bicubic, v2, v3, randomized-order fine-tune, and any promoted
      challenger in an ablation table.
- [ ] Show best, median, and worst validation examples; label the failure mode.
- [ ] State clearly that blind-test ground truth was never used.

## Official deck and video

- [ ] Use the official KLA idea-submission template and remove its instruction
      slide.
- [ ] Keep 8-9 content slides: Team, Problem, Idea, Solution, Innovation,
      Results, Feasibility, GitHub/demo, References.
- [ ] Export to PDF and name it `TeamName_KLA_PS01.pdf`.
- [ ] Optional demo video is five minutes or less and its link is public.
- [ ] Every external figure, model idea, and metric is cited.

## Portal

- [ ] GitHub URL, PDF, and optional video URL open in an incognito window.
- [ ] Submit before 16 August 2026 and retain the confirmation screenshot.
