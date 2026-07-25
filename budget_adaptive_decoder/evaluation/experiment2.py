"""
Experiment 2: k* vs. Content at Fixed Budget

As specified in Section 13 of the design document.

Purpose: Verify that the system allocates more compute to harder frames
at any fixed budget B.

Method: For each fixed B in {0.3, 0.5, 0.7}, run the full system and
record k* selected for each frame and its content features.

Feasibility check: Before computing correlations, verify that the budget
B permits at least 2 depth values for more than 50% of frames.

Success criterion: At each fixed B where feasibility check passes,
frames with higher latent energy or motion magnitude receive higher k*
on average (positive correlation, r > 0.2).
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


FIXED_BUDGETS = [0.3, 0.5, 0.7]
CORRELATION_THRESHOLD = 0.2
FEASIBILITY_THRESHOLD = 0.5


def verify_budget_feasibility(
    budget: float,
    stage_costs: List[float],
    dataset: Vimeo90kDataset,
    decoder: BudgetAdaptiveDecoder,
    device: torch.device,
    num_samples: int = 500,
) -> Dict:
    """
    Verify that budget B permits at least 2 depth values for >50% of frames.

    Args:
        budget: Budget ratio B
        stage_costs: List of K stage costs
        dataset: Dataset to evaluate
        decoder: BudgetAdaptiveDecoder
        device: Device

    Returns:
        Dictionary with feasibility results
    """
    K = len(stage_costs) - 1
    C_0 = stage_costs[0]
    C_total = sum(stage_costs)
    budget_limit = budget * C_total

    stage_costs_only = np.array(stage_costs[1:])
    cumulative_costs = np.cumsum(stage_costs_only)

    depth_counts = {k: 0 for k in range(K + 1)}

    for idx in range(min(len(dataset), num_samples)):
        frames, bitstream_dict, prev_frame = dataset[idx]

        for depth in range(K + 1):
            if depth == 0:
                cum_cost = C_0
            else:
                cum_cost = C_0 + cumulative_costs[depth - 1]

            if cum_cost <= budget_limit:
                depth_counts[depth] += 1

    total_samples = min(len(dataset), num_samples)

    fraction_at_least_1 = sum(depth_counts[k] for k in range(1, K + 1)) / total_samples
    fraction_at_least_2 = sum(depth_counts[k] for k in range(2, K + 1)) / total_samples

    return {
        "budget": budget,
        "depth_counts": depth_counts,
        "fraction_at_least_1": fraction_at_least_1,
        "fraction_at_least_2": fraction_at_least_2,
        "PASS": fraction_at_least_2 > FEASIBILITY_THRESHOLD,
    }


def run_experiment2(
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
    Run Experiment 2: k* vs Content at Fixed Budget.

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
        Dictionary with results per budget level
    """
    from ..verify.check_correlation import ContentFeatureExtractor

    dataset = Vimeo90kDataset(
        root=data_root or "./vimeo90k",
        split="test",
        apply_corruption=False,
    )

    feature_extractor = ContentFeatureExtractor()

    results = {}

    for budget in FIXED_BUDGETS:
        print(f"\n--- Testing budget B = {budget} ---")

        feasibility = verify_budget_feasibility(
            budget, stage_costs, dataset, decoder, device, num_samples
        )

        print(f"Feasibility: {feasibility['fraction_at_least_2']:.2%} of frames have ≥2 depths")

        if not feasibility["PASS"]:
            print(f"  SKIPPED: Budget {budget} too tight for content analysis")
            results[budget] = {**feasibility, "correlation": None, "PASS": False}
            continue

        all_features = []
        all_k_star = []

        for idx in tqdm(range(min(len(dataset), num_samples)), desc=f"B={budget}"):
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

            residual = (recon_image - prev_frame_tensor).abs().mean().item()
            motion_mag = latent.std().item() if latent is not None else 0.0
            features = {
                "latent_energy": motion_mag,
                "motion_magnitude": abs(motion.std().item()) if motion is not None else 0.0,
                "residual_proxy": residual,
            }

            all_features.append(features)
            all_k_star.append(k_star)

        feature_array = np.array([[f[k] for f in all_features] for k in all_features[0].keys()]).T
        k_star_array = np.array(all_k_star)

        correlations = {}
        for feat_idx, feat_name in enumerate(feature_names := list(all_features[0].keys())):
            corr, _ = pearsonr(feature_array[:, feat_idx], k_star_array)
            correlations[feat_name] = corr

        max_corr = max(abs(v) for v in correlations.values())
        has_positive_correlation = any(v > CORRELATION_THRESHOLD for v in correlations.values())

        result = {
            **feasibility,
            "correlations": correlations,
            "max_correlation": max_corr,
            "PASS": has_positive_correlation,
        }

        results[budget] = result

        print(f"  Max correlation: {max_corr:.4f}")
        print(f"  PASS: {has_positive_correlation}")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "experiment2_results.json", "w") as f:
            json.dump(results, f, indent=2)

    return results


def pearsonr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Compute Pearson correlation coefficient and p-value."""
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 1.0
    corr = np.corrcoef(x, y)[0, 1]
    n = len(x)
    if n > 2:
        t_stat = corr * np.sqrt(n - 2) / np.sqrt(1 - corr ** 2 + 1e-8)
        from scipy.stats import t as t_dist
        p_val = 2 * (1 - t_dist.cdf(abs(t_stat), n - 2))
    else:
        p_val = 1.0
    return corr, p_val


if __name__ == "__main__":
    print("Experiment 2: k* vs Content at Fixed Budget")