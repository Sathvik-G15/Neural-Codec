# DCVC-Progressive: Compute-Scalable Neural Video Decoding

A neural video codec with a **progressive residual decoder** that exposes controllable runtime complexity from a single model.

## Key Idea

Standard neural video codecs (e.g., DCVC) assume **fixed decode compute**. This project replaces the fixed decoder with a progressive residual architecture:

```
Compressed Bitstream
        │
        ▼
   ┌─────────────┐
   │ Base Decoder │  ← ~70-80% of compute (produces usable reconstruction)
   └──────┬──────┘
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
  [R_1] [R_2] [R_3]  ← Each refinement adds ~5-10% compute
    │     │     │
    └─────┼─────┘
          │
          ▼
    Refined Output
```

**Compute scaling**: Execute 1-4 refinement stages based on available budget.

## Features

- **Single model, multiple compute points**: One checkpoint serves all device tiers
- **Progressive refinement**: Diminishing returns curve enables principled compute allocation
- **Bitstream compatible**: Uses DCVC encoder/entropy model unchanged
- **Deep supervision training**: Weighted multi-stage loss ensures all tiers produce useful output

## Setup

```bash
cd DCVC-Scalable
pip install -r requirements.txt
```

## Download Dataset

```bash
python scripts/download_uvg.py --output_dir data/uvg
```

## Profile Decoder FLOPs

```bash
python scripts/profile_decoder.py
```

## Training

```bash
python src/train.py \
    --data_root data/uvg \
    --epochs 600 \
    --batch_size 1 \
    --lr 1e-4 \
    --log_dir logs \
    --checkpoint_dir checkpoints
```

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/checkpoint_final.pt \
    --data data/uvg \
    --output results
```

## Architecture

### ProgressiveDecoder

- **Base decoder**: Mirrors DCVC contextual decoder (~70-80% of FLOPs)
- **Refinement blocks**: Small residual CNNs (~5-10% each)
- **Forward**: Returns list of reconstructions at each depth

### Training Loss

```python
weights = [1.0, 0.9, 0.8, 0.7]  # Earlier stages weighted higher

for i, recon in enumerate(outputs):
    loss += weights[i] * MSE(recon, target)
```

### Curriculum

| Phase | Epochs | Depth Strategy |
|-------|--------|----------------|
| 1 | 0-200 | Fixed depth=4 |
| 2 | 200-400 | Random [1,4] |
| 3 | 400-600 | Random [1,4] |

## Expected Results

After training, you should observe:

| Tier | FLOPs | VMAF Range | FPS Target |
|------|-------|------------|------------|
| 1 (Minimal) | ~70% | baseline | 60+ FPS |
| 2 (Low) | ~80% | +1-2 | 50+ FPS |
| 3 (Medium) | ~90% | +2-3 | 40+ FPS |
| 4 (Full) | 100% | +3-5 | 30+ FPS |

## Project Structure

```
DCVC-Scalable/
├── src/
│   ├── __init__.py
│   ├── progressive_decoder.py   # Core decoder architecture
│   ├── dcvc_progressive.py      # Full model (encoder + decoder)
│   └── train.py                  # Training script
├── scripts/
│   ├── profile_decoder.py        # FLOPs profiling
│   ├── download_uvg.py           # Dataset download
│   └── evaluate.py               # Evaluation script
├── data/                          # Data directory
├── checkpoints/                   # Model checkpoints
├── logs/                          # TensorBoard logs
├── config.json                    # Training configuration
└── README.md
```

## Citation

If you use this code, please cite:

```bibtex
@misc{DCVC-Progressive,
    title={Progressive Residual Decoding for Compute-Adaptive Neural Video Streaming},
    author={},
    year={2026}
}
```

## References

- Li et al., "DCVC: Deep Contextual Video Compression," NeurIPS 2021
- Li et al., "DCVC-RT: Real-Time Neural Video Coding," CVPR 2025