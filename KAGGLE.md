# Kaggle Phase 1 Training - Quickstart

Train the **Budget-Adaptive Decoder** (Phase 1: hybrid distillation) on
Vimeo90k via Kaggle GPU.

## Prerequisites (Kaggle UI)

1. Create a Kaggle **Dataset** (private) and upload one of:
   - The **DCVC checkpoint** `.pth.tar` file (Microsoft DCVC quality-3 PSNR
     weights).
   - Optional: the full unzipped DCVC-family source tree (`DCVC/src/`,
     `DCVC/checkpoints/`, etc.) — needed only if your Kaggle input path
     does not already include the DCVC source.
2. Create a second Kaggle **Dataset** (private) and upload the Vimeo90k
   septuplet distribution. Final on-disk shape inside the dataset root:
   ```
   vimeo_septuplet/
   ├── sequences/
   │   ├── 00001/
   │   │   └── 0001/
   │   │       ├── im1.png  im2.png  im3.png  im4.png  im5.png  im6.png  im7.png
   │   ├── 00002/
   │   │   └── 0001/  ...  im1..im7.png
   │   └── ... up to 00096
   ├── sep_trainlist.txt
   └── sep_testlist.txt
   ```
   Each line of `sep_trainlist.txt`/`sep_testlist.txt`:
   ```
   <SSSSS>/<CCCC> <frame_type_int>
   e.g.  00001/0001 1
   ```
3. Add both datasets as **Inputs** in your Kaggle notebook/notebook
   settings. The two dataset folders will appear under
   `/kaggle/input/<DATASET_SLUG>/`.

---

## Notebook Cell 1 — Clone the branch

```python
!git clone -b fix/data-loading-real-video https://github.com/Sathvik-G15/Neural-Codec.git
%cd Neural-Codec
```

---

## Notebook Cell 2 — Sanity-check data layout (1 × 15 sec)

```python
import sys
sys.path.insert(0, '/kaggle/working/Neural-Codec')
from pathlib import Path

# Adjust this path to where your Vimeo90k septuplet dataset was mounted by Kaggle.
DATA_ROOT = '/kaggle/input/datasets/sathvikml/vimeo90k/vimeo_septuplet'

from budget_adaptive_decoder.data.vimeo90k import Vimeo90kDataset
ds = Vimeo90kDataset(root=DATA_ROOT, split='train')
print(f"Train clips: {len(ds)}")
frames, _bs, prev = ds[0]
print(f"frames.shape={frames.shape}, frames[0].mean={frames[0].mean():.4f}, prev_frame.mean={prev.mean():.4f}")
print(f"first frame == base reconstruction (cnt): {(frames[0] - prev).abs().max().item() == 0}")
```

Expected output:
```
[Vimeo90kDataset] <N> clips from /kaggle/input/datasets/sathvikml/vimeo90k/vimeo_septuplet/sep_trainlist.txt (Kaggle layout: sequences/SSSSS/CCCC/im{1..7}.png).
Train clips: <N>
frames.shape=torch.Size([7, 3, 256, 448]), frames[0].mean=0.6363, prev_frame.mean=0.7408
first frame == base reconstruction (cnt): True
```
The `frames[0]` and `prev_frame` are real pixel values (mean around 0.5-0.8,
not the synthetic 0.5 default).

---

## Notebook Cell 3 — DCVC source location

DCVC is imported as `from src.models.DCVC_net import DCVC_net`, so the
**directory containing `src/`** must be on PYTHONPATH.

If your Kaggle DCVC dataset zip contains the full DCVC-family tree:
```
dcvc-baseline/
└── DCVC-family/DCVC/{src, checkpoints, ...}/
```
then `/kaggle/input/<DCVC_SLUG>/dcvc-baseline/DCVC-family/DCVC` is the
src directory. Point `--dcvc_src` (or env `DCVC_SRC_PATH`) to it.

If you only uploaded the `.pth.tar` checkpoint and not the DCVC source,
the run will fail at `from src.models.DCVC_net import DCVC_net` — you
must include the DCVC source in a Kaggle Dataset. The pybind11
extension is built on first import; the rebuild can take ~10 min on
first call. Subsequent calls re-use the cached `.so`.

---

## Notebook Cell 4 — Phase 1 training (full)

```python
!python -m budget_adaptive_decoder.training.phase1_distill \
    --pretrained /kaggle/input/datasets/sathvikml/checkpoint/model_dcvc_quality_3_psnr.pth \
    --data_root   /kaggle/input/datasets/sathvikml/vimeo90k/vimeo_septuplet \
    --output_dir  /kaggle/working/checkpoints/phase1 \
    --batch_size  2 --epochs 30 --lr 1e-4 --num_workers 2
```

Notes:
- `batch_size=2` is chosen as the safe default for a 16 GB GPU
  (DCVC dominates VRAM during training).
- `epochs=30` is a starting point. **You should monitor**
  `train/s1_teacher` (Stage 1 MSE to DCVC) and stop early when it
  plateaus (<30% improvement across consecutive 3 epochs).
- Stages 2-4 (`loss_s2_gt`, `loss_s3_gt`, `loss_s4_gt`) start **worse**
  than Stage 1 early in training. This is expected — see the README
  section "Per-stage ΔQ distribution" below.

---

## Notebook Cell 5 — Resume from a checkpoint

If a previously-saved checkpoint exists:
```python
!python -m budget_adaptive_decoder.training.phase1_distill \
    --pretrained /kaggle/input/.../model_dcvc_quality_3_psnr.pth \
    --data_root   /kaggle/input/.../vimeo_septuplet \
    --output_dir  /kaggle/working/checkpoints/phase1_resume \
    --batch_size  2 --epochs 30 --lr 1e-4 --num_workers 2 \
    --resume /kaggle/working/checkpoints/phase1/phase1_final.pt
```

---

## Expected Outputs

Inside `--output_dir` after training:
- `phase1_epoch1.pt`, `phase1_epoch2.pt`, ...    (one per `--save_every` epoch boundary)
- `phase1_final.pt`                              (final state)

Each checkpoint contains:
```python
{
    "epoch": int,
    "decoder_state_dict": <state>,
    "optimizer_state_dict": <state>,
    "metrics": {        # last epoch's train + val metrics
        "loss_total": float, "loss_s1_teacher": float, ...,
        "val_loss_total": float, ...
    },
    "best_loss": float,
}
```

---

## What to check before committing to the long run (smoke test)

Before training for 30 epochs, do a **1-epoch dry-run** that exercises
all the system plumbing:
```python
!python -m budget_adaptive_decoder.training.phase1_distill \
    --pretrained /kaggle/input/.../model_dcvc_quality_3_psnr.pth \
    --data_root   /kaggle/input/.../vimeo_septuplet \
    --output_dir  /kaggle/working/checkpoints/phase1_dryrun \
    --batch_size  2 --epochs 1 --num_workers 2
```
Watch the per-step `loss=...` line. After ~50 steps it should be
stabilising in the 0.0X range (a few thousand pixels × MSE per channel).

If you see `NaN`:
- Drop learning rate to `--lr 5e-5`.
- Check that `--dcvc_src` resolves correctly (run a single Python step
  that does `from src.models.DCVC_net import DCVC_net` from the chosen
  directory).

---

## Per-stage expected loss curves

Phase 1 trained decoders have an inverted-quality profile during early
training — Stage 1 (teacher alignment) converges first, Stages 2-4
catch up later:

| Stage | After 5 epochs | After 30 epochs (target) |
|-------|----------------|--------------------------|
| Stage 1 loss (vs teacher) | 0.01-0.04 | 0.005-0.015 |
| Stage 2 loss (vs ground truth) | high (≈0.30) | 0.02-0.05 |
| Stage 3 loss (vs ground truth) | high (≈0.35) | 0.02-0.05 |
| Stage 4 loss (vs ground truth) | high (≈0.30) | 0.02-0.05 |

This is consistent with the cumulative-vs-marginal loss design (see
DESIGN_DOCUMENT.md §6.4).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: src.models.DCVC_net` | `--dcvc_src` not set | Run with explicit `--dcvc_src /path/to/DCVC-family/DCVC` |
| `RuntimeError: DCVC checkpoint not found at ...` | Path wrong or file not uploaded | `ls /kaggle/input/<slugs>` and adjust `--pretrained` |
| Loss stays high for both stages | First-run pybind11 compile | Let it run, the cost is one-time per Kaggle session |
| `dataset length = 0` | Wrong `--data_root` | Confirm dataset structure matches `sequences/SSSSS/CCCC/im{1..7}.png` |
| `RuntimeError: Expected 3D or 4D (batch mode)` | Wrong classifier style applied to wrong tensor | re-run Cell 2 and verify frame tensor shape |
| OOM after 2-3 epochs | DCVC + student at full 448x256 | drop `--batch_size` to 1, or use `--num_workers 0` |
