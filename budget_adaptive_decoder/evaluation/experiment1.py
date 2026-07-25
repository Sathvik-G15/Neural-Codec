"""
Experiment 1: ΔQ–Content Correlation

As specified in Section 13 of the design document.

Purpose: Validate the core assumption that marginal quality gains ΔQ(k)
correlate with observable bitstream features.

Success criterion: At least one content feature has |r| > 0.3 correlation
with at least one ΔQ(k).

If FAIL: STOP — Do not proceed to training.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm

from ..models.decoder import BudgetAdaptiveDecoder
from ..models.extractor import BitstreamContentExtractor
from ..data.vimeo90k import Vimeo90kDataset
from ..evaluation.metrics import compute_psnr


CORRELATION_THRESHOLD = 0.3


def run_experiment1(
    decoder: BudgetAdaptiveDecoder,
    device: torch.device,
    num_samples: int = 1000,
    data_root: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Run Experiment 1: ΔQ-Content Correlation check.

    Args:
        decoder: Trained BudgetAdaptiveDecoder
        device: Device to use
        num_samples: Number of samples to evaluate
        data_root: Dataset root directory
        output_dir: Directory to save results

    Returns:
        Dictionary with results and PASS/FAIL status
    """
    from ..verify.check_correlation import (
        ContentFeatureExtractor,
        compute_delta_q_targets,
        compute_correlation,
        pearson_correlation,
    )

    decoder.eval()
    feature_extractor = ContentFeatureExtractor()

    all_features = []
    all_delta_q = []

    dataset = Vimeo90kDataset(
        root=data_root or "./vimeo90k",
        split="test",
        apply_corruption=False,
    )

    print(f"Running Experiment 1 on {min(len(dataset), num_samples)} samples...")

    for idx in tqdm(range(min(len(dataset), num_samples))):
        frames, bitstream_dict, prev_frame = dataset[idx]

        latent = bitstream_dict["latent"].unsqueeze(0).to(device)
        motion = bitstream_dict["motion"].unsqueeze(0).to(device)
        prev_frame_tensor = prev_frame.unsqueeze(0).to(device)
        frame_type_id = bitstream_dict["frame_type_id"].to(device)

        features = feature_extractor.extract_features(
            latent, motion, prev_frame_tensor, frame_type_id
        )
        all_features.append(features)

        delta_q, _ = compute_delta_q_targets(
            decoder, latent, motion, prev_frame_tensor, device
        )
        all_delta_q.append(delta_q.cpu().numpy())

    feature_array = np.array([[f[k] for f in all_features] for k in all_features[0].keys()]).T
    delta_q_array = np.concatenate(all_delta_q, axis=0)

    correlation_matrix, p_values = compute_correlation(feature_array, delta_q_array)

    feature_names = list(all_features[0].keys())
    stage_names = [f"ΔQ(k={k+1})" for k in range(delta_q_array.shape[1])]

    max_corr = 0.0
    max_corr_info = {}
    for feat_idx, feat_name in enumerate(feature_names):
        for k_idx in range(len(stage_names)):
            corr = correlation_matrix[feat_idx, k_idx]
            if abs(corr) > abs(max_corr):
                max_corr = corr
                max_corr_info = {
                    "feature": feat_name,
                    "stage": stage_names[k_idx],
                    "correlation": corr,
                    "p_value": p_values[feat_idx, k_idx],
                }

    result = {
        "experiment": 1,
        "correlation_matrix": correlation_matrix.tolist(),
        "feature_names": feature_names,
        "stage_names": stage_names,
        "max_correlation": max_corr,
        "max_correlation_info": max_corr_info,
        "num_samples": len(all_features),
        "PASS": abs(max_corr) > CORRELATION_THRESHOLD,
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "experiment1_results.json", "w") as f:
            json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    print("Experiment 1: ΔQ-Content Correlation")
    print("See verify/check_correlation.py for standalone execution")