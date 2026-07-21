import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
import sys

DCVC_DIR = Path(__file__).parent / "DCVC-repo" / "DCVC-family" / "DCVC"
sys.path.insert(0, str(DCVC_DIR))
from src.models.DCVC_net import DCVC_net

CHECKPOINT_DIR = DCVC_DIR / "checkpoints"
UVG_DATA_DIR = Path(__file__).parent / "DCVC-Scalable" / "data" / "uvg"

def _load_image(path):
    with Image.open(str(path)) as img:
        img = img.convert('RGB')
        arr = np.array(img, copy=True)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(torch.float32).mul_(1.0 / 255.0)
    return tensor.cuda()

def main():
    print("Loading base DCVC weights...")
    model = DCVC_net().cuda()
    base_ckpt_path = CHECKPOINT_DIR / "model_dcvc_quality_3_psnr.pth"
    if not base_ckpt_path.exists():
        extracted = CHECKPOINT_DIR / "extracted" / "model_dcvc_quality_3_psnr.pth"
        if extracted.exists():
            base_ckpt_path = extracted
    base_ckpt = torch.load(base_ckpt_path, map_location='cuda', weights_only=False)
    if 'state_dict' in base_ckpt:
        model.load_state_dict(base_ckpt['state_dict'], strict=False)
    else:
        model.load_state_dict(base_ckpt, strict=False)
        
    print("Loading progressive decoder checkpoint...")
    prog_ckpt_path = CHECKPOINT_DIR / "progressive" / "progressive_best.pth.tar"
    prog_ckpt = torch.load(prog_ckpt_path, map_location='cuda', weights_only=False)
    model.progressive_decoder.load_state_dict(prog_ckpt['progressive_decoder_state'])

    model.eval()

    print("Loading UVG Beauty sequence (first 2 frames)...")
    uvg_dir = UVG_DATA_DIR / "Beauty"
    pngs = sorted(list(uvg_dir.glob("*.png")))
    if len(pngs) < 2:
        print("Could not find enough PNGs in UVG Beauty directory.")
        return
        
    ref = _load_image(pngs[0])
    cur = _load_image(pngs[1])
    
    # Pad to multiple of 64
    H, W = ref.shape[2:]
    pad_h = (64 - H % 64) % 64
    pad_w = (64 - W % 64) % 64
    if pad_h > 0 or pad_w > 0:
        import torch.nn.functional as F
        ref = F.pad(ref, (0, pad_w, 0, pad_h))
        cur = F.pad(cur, (0, pad_w, 0, pad_h))
    
    print(f"Running forward pass... Shape: {ref.shape}")
    with torch.no_grad():
        out = model(ref, cur)
        
    recon_images = out['recon_images']
    
    # Crop back
    if pad_h > 0 or pad_w > 0:
        cur = cur[:, :, :H, :W]
        recon_images = [r[:, :, :H, :W] for r in recon_images]
        
    print("\n--- Progressive Tier Results ---")
    for i, recon in enumerate(recon_images):
        mse = nn.functional.mse_loss(recon, cur).item()
        psnr = 10 * np.log10(1.0 / (mse + 1e-8))
        print(f"Tier {i+1} PSNR: {psnr:.3f} dB")
        
    if len(recon_images) >= 4:
        mse1 = nn.functional.mse_loss(recon_images[0], cur).item()
        mse4 = nn.functional.mse_loss(recon_images[-1], cur).item()
        psnr1 = 10 * np.log10(1.0 / (mse1 + 1e-8))
        psnr4 = 10 * np.log10(1.0 / (mse4 + 1e-8))
        print(f"\nDelta PSNR (Tier {len(recon_images)} - Tier 1): {psnr4 - psnr1:.3f} dB")

if __name__ == '__main__':
    main()
