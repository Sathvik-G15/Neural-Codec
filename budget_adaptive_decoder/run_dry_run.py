"""
Phase 1 Dry-Run - Test training pipeline with synthetic data.
"""
import sys
from pathlib import Path

# Add budget_adaptive_decoder to path
BAD_PATH = Path(__file__).parent
if str(BAD_PATH) not in sys.path:
    sys.path.insert(0, str(BAD_PATH))

import torch
import torch.nn.functional as F
import tempfile
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.decoder import BudgetAdaptiveDecoder
from models.teacher import DCVCWrapper
from data.generic_video_dataset import GenericVideoDataset

GT_WEIGHTS = {2: 0.5, 3: 0.75, 4: 1.0}


def make_loader(root, batch_size=4, shuffle=True):
    dataset = GenericVideoDataset(root=root, height=256, width=448, corruption_sigma=0.0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


def compute_phase1_loss(student_recons, teacher_recon, ground_truth, K, device):
    loss_s1 = F.mse_loss(student_recons[0], teacher_recon)
    loss_gt = torch.tensor(0.0, device=device)

    for k in range(1, K):
        stage_gt_loss = F.mse_loss(student_recons[k], ground_truth)
        weight = GT_WEIGHTS.get(k + 1, 1.0)
        loss_gt = loss_gt + weight * stage_gt_loss

    return loss_s1 + loss_gt, {'s1': loss_s1.item(), 'gt': loss_gt.item()}


def train_epoch(decoder, teacher, loader, optimizer, device, K):
    decoder.train()
    total_loss = 0
    n_batches = 0
    losses = {'s1': 0, 'gt': 0}

    for curr, ref in tqdm(loader, desc="Training"):
        curr = curr.to(device)
        ref = ref.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            teacher_out = teacher.get_stage_targets(curr, ref)
            # teacher_out['stage_recons'] is list of 4 reconstructions
            # stage_recons[0] is the base (teacher's single output for K=1)
            teacher_recon = teacher_out['stage_recons'][0]
            context = teacher_out['context']

        student_recons = decoder.decode_all_stages(context, ref)

        loss, loss_comps = compute_phase1_loss(student_recons, teacher_recon, curr, K, device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        losses['s1'] += loss_comps['s1']
        losses['gt'] += loss_comps['gt']
        n_batches += 1

    return {
        'loss_total': total_loss / n_batches,
        'loss_s1_teacher': losses['s1'] / n_batches,
        'loss_s2_gt': 0,
        'loss_s3_gt': 0,
        'loss_s4_gt': 0,
    }


def validate(decoder, teacher, loader, device, K):
    decoder.eval()
    total_loss = 0
    n_batches = 0
    losses = {'s1': 0, 'gt': 0}

    with torch.no_grad():
        for curr, ref in loader:
            curr = curr.to(device)
            ref = ref.to(device)

            teacher_out = teacher.get_stage_targets(curr, ref)
            teacher_recon = teacher_out['stage_recons'][0]
            context = teacher_out['context']

            student_recons = decoder.decode_all_stages(context, ref)

            loss, loss_comps = compute_phase1_loss(student_recons, teacher_recon, curr, K, device)

            total_loss += loss.item()
            losses['s1'] += loss_comps['s1']
            losses['gt'] += loss_comps['gt']
            n_batches += 1

    return {
        'val_loss_total': total_loss / n_batches,
        'val_loss_s1_teacher': losses['s1'] / n_batches,
    }


def main():
    print('='*60)
    print('Phase 1 Dry-Run with Synthetic Data')
    print('='*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(20):
            arr = np.random.randint(0, 255, (256, 448, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            img.save(Path(tmpdir) / f'frame_{i:03d}.png')
        print(f'Created 20 synthetic frames in {tmpdir}')

        train_loader = make_loader(tmpdir, batch_size=4, shuffle=True)
        val_loader = make_loader(tmpdir, batch_size=4, shuffle=False)
        print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')

        print('Loading teacher (DCVC)...')
        teacher = DCVCWrapper(load_pretrained=True).to(device)
        print(f'Teacher loaded. DCVC is None: {teacher.dcvc is None}')

        print('Creating decoder...')
        decoder = BudgetAdaptiveDecoder().to(device)
        K = decoder.K
        print(f'Decoder params: {sum(p.numel() for p in decoder.parameters()):,}')
        print(f'Decoder K={K}')

        optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)

        print('\nTraining 3 epochs...')
        for epoch in range(1, 4):
            train_m = train_epoch(decoder, teacher, train_loader, optimizer, device, K)
            val_m = validate(decoder, teacher, val_loader, device, K)

            print(f'\nEpoch {epoch}:')
            print(f'  Train Loss: {train_m["loss_total"]:.4f}')
            print(f'    S1 (teacher): {train_m["loss_s1_teacher"]:.4f}')
            print(f'  Val Loss: {val_m["val_loss_total"]:.4f}')

            if np.isnan(train_m['loss_total']):
                print('NaN detected - stopping')
                break

    print('\n' + '='*60)
    print('Dry-run completed successfully!')
    print('='*60)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)