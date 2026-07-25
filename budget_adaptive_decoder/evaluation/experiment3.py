"""
Experiment 3: Quality-Compute Pareto Curve vs. Static Tiers

As specified in Section 13 of the design document.

Purpose: Demonstrate that the adaptive system Pareto-dominates static
execution tiers at every budget level, and that quality increases with budget.

Method:
For B in {0.2, 0.3, 0.44, 0.5, 0.67, 0.8, 0.89, 0.95, 1.0}, evaluate:
    (a) Proposed system (adaptive)
    (b) Static tier T1 (always 1 stage)
    (c) Static tier T2 (always 2 stages)
    (d) Static tier T3 (always 3 stages)
    (e) Static tier T4 (always 4 stages)
    (f) Oracle (best possible at each budget)

Budget values rationale (from design doc):
    {0.44, 0.67, 0.89, 1.0} are stage cost boundaries for C_1:C_2:C_3:C_4 = 4:2:2:1
    {0.2, 0.3, 0.5, 0.8, 0.95} fall between boundaries

Quality monotonicity verification:
    For each consecutive pair B_i < B_{i+1}, verify PSNR(B_{i+1}) >= PSNR(B_i)
    for >95% of consecutive pairs.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm

from ..models.decoder import BudgetAdaptiveDecoder
from ..models.policy import PolicyNetwork
from ..models.extractor import BitstreamContentExtractor
from ..models.scheduler import Scheduler
from ..data.vimeo90k import Vimeo90kDataset
from ..evaluation.metrics import compute_psnr


DESIGN_BUDGETS = [0.2, 0.3, 0.44, 0.5, 0.67, 0.8, 0.89, 0.95, 1.0]
MONOTONICITY_THRESHOLD = 0.95


def evaluate_static_tier(
    decoder: BudgetAdaptiveDecoder,
    depth: int,
    context: torch.Tensor,
    prev_frame: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Evaluate decoder at a fixed depth (static tier)."""
    decoder.eval()
    with torch.no_grad():
        return decoder.run_to_depth(context, prev_frame, depth)


def evaluate_oracle(
    decoder: BudgetAdaptiveDecoder,
    budget: float,
    stage_costs: List[float],
    context: torch.Tensor,
    prev_frame: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    """
    Evaluate oracle: select best depth within budget.

    Returns reconstruction and the depth selected.
    """
    decoder.eval()
    K = len(stage_costs) - 1
    C_0 = stage_costs[0]
    C_total = sum(stage_costs)
    budget_limit = budget * C_total

    stage_costs_only = np.array(stage_costs[1:])
    cumulative_costs = np.cumsum(stage_costs_only)

    best_depth = 0
    best_psnr = float("-inf")
    best_reconstruction = None

    with torch.no_grad():
        reconstructions = decoder.decode_all_stages(context, prev_frame)

        for depth in range(K + 1):
            if depth == 0:
                cum_cost = C_0
            else:
                cum_cost = C_0 + cumulative_costs[depth - 1]

            if cum_cost <= budget_limit + 1e-6:
                psnr = compute_psnr(reconstructions[depth], reconstructions[-1])
                if psnr > best_psnr:
                    best_psnr = psnr
                    best_depth = depth
                    best_reconstruction = reconstructions[depth]

    return best_reconstruction, best_depth


def run_experiment3(
    decoder: BudgetAdaptiveDecoder,
    policy: PolicyNetwork,
    extractor: BitstreamContentExtractor,
    scheduler: Scheduler,
    device: torch.device,
    stage_costs: List[float],
    num_samples: int = 500,
    data_root: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Run Experiment 3: Quality-Compute Pareto Curve.

    Args:
        decoder: BudgetAdaptiveDecoder
        policy: Trained PolicyNetwork
        extractor: Trained BitstreamContentExtractor
        scheduler: Scheduler
        device: Device
        stage_costs: Stage costs [C_1, ..., C_K]
        num_samples: Number of samples
        data_root: Dataset root
        output_dir: Output directory

    Returns:
        Dictionary with Pareto curve data
    """
    dataset = Vimeo90kDataset(
        root=data_root or "./vimeo90k",
        split="test",
        apply_corruption=False,
    )

    results = {budget: {} for budget in DESIGN_BUDGETS}

    for budget in tqdm(DESIGN_BUDGETS, desc="Budget levels"):
        adaptive_psnrs = []
        tier_psnrs = {1: [], 2: [], 3: [], 4: []}
        oracle_psnrs = []

        for idx in range(min(len(dataset), num_samples)):
            frames, bitstream_dict, prev_frame = dataset[idx]

            latent = bitstream_dict["latent"].unsqueeze(0).to(device)
            motion = bitstream_dict["motion"].unsqueeze(0).to(device)
            prev_frame_tensor = prev_frame.unsqueeze(0).to(device)

            context = torch.zeros(1, 64, prev_frame_tensor.shape[2] // 4, prev_frame_tensor.shape[3] // 4, device=device)
            recon_image = torch.zeros_like(prev_frame_tensor)

            with torch.no_grad():
                content_features = extractor(
                    context, recon_image, prev_frame_tensor, latent
                )
                delta_q_pred = policy(content_features, torch.tensor([budget], device=device))
                k_star = scheduler.select_depth(delta_q_pred, budget).item()

                adaptive_reconstruction = decoder.run_to_depth(
                    context, prev_frame_tensor, k_star
                )

                for tier in range(1, 5):
                    tier_reconstruction = decoder.run_to_depth(
                        context, prev_frame_tensor, tier
                    )
                    psnr = compute_psnr(tier_reconstruction, frames[-1].unsqueeze(0).to(device))
                    tier_psnrs[tier].append(psnr)

                oracle_reconstruction, oracle_depth = evaluate_oracle(
                    decoder, budget, stage_costs, context, prev_frame_tensor, device
                )

            adaptive_psnr = compute_psnr(adaptive_reconstruction, frames[-1].unsqueeze(0).to(device))
            oracle_psnr = compute_psnr(oracle_reconstruction, frames[-1].unsqueeze(0).to(device))

            adaptive_psnrs.append(adaptive_psnr)
            oracle_psnrs.append(oracle_psnr)

        results[budget] = {
            "adaptive": {
                "mean_psnr": np.mean(adaptive_psnrs),
                "std_psnr": np.std(adaptive_psnrs),
            },
            "tiers": {
                tier: {
                    "mean_psnr": np.mean(psnrs),
                    "std_psnr": np.std(psnrs),
                }
                for tier, psnrs in tier_psnrs.items()
            },
            "oracle": {
                "mean_psnr": np.mean(oracle_psnrs),
                "std_psnr": np.std(oracle_psnrs),
            },
            "num_samples": len(adaptive_psnrs),
        }

    monotonicity_check = check_quality_monotonicity(results)
    results["monotonicity"] = monotonicity_check

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "experiment3_results.json", "w") as f:
            json.dump(results, f, indent=2)

    return results


def check_quality_monotonicity(results: Dict) -> Dict:
    """
    Check that PSNR(B_{i+1}) >= PSNR(B_i) for >95% of consecutive pairs.

    As per design doc: quality must increase with budget (O4).
    """
    budgets = [b for b in DESIGN_BUDGETS if b in results]
    monotonic_pairs = 0
    total_pairs = 0

    for i in range(len(budgets) - 1):
        b1, b2 = budgets[i], budgets[i + 1]

        if "adaptive" in results[b1] and "adaptive" in results[b2]:
            psnr1 = results[b1]["adaptive"]["mean_psnr"]
            psnr2 = results[b2]["adaptive"]["mean_psnr"]

            total_pairs += 1
            if psnr2 >= psnr1:
                monotonic_pairs += 1

    ratio = monotonic_pairs / max(total_pairs, 1)

    return {
        "monotonic_pairs": monotonic_pairs,
        "total_pairs": total_pairs,
        "monotonicity_ratio": ratio,
        "PASS": ratio > MONOTONICITY_THRESHOLD,
    }


if __name__ == "__main__":
    print("Experiment 3: Quality-Compute Pareto Curve")