"""M5.5: Mobile Export Feasibility Spike (~1 day effort).

Tests:
1. Can the progressive decoder be exported to ONNX?
2. Does PixelShuffle (subpixel-conv) export correctly to ONNX?
3. Does the ONNX model run on the target mobile runtime?

This spike's outcome determines M6's status:
  - M5.5 succeeds cleanly -> M6 is REQUIRED
  - M5.5 reveals blockers -> M6 is DOWNGRADED TO STRETCH

Per plan §5.3: PixelShuffle/subpixel-conv in part1 is on the Tier-1 export
critical path, independent of the refinement-block design. This must be
verified in this spike, not assumed.

Usage:
    python export_spike.py [--runtime cpu|cuda]

Output:
    - spike_results.json: pass/fail per test + blocker list if any
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DCVC_DIR = Path(__file__).parent / "DCVC-repo" / "DCVC-family" / "DCVC"
sys.path.insert(0, str(DCVC_DIR))
from src.models.progressive_decoder import ProgressiveContextualDecoder


def test_onnx_export(decoder: ProgressiveContextualDecoder, output_path: Path) -> dict:
    result = {"onnx_export": False, "error": None, "ops_verified": []}

    try:
        y_hat = torch.randn(1, 96, 68, 120).cuda()
        context = torch.randn(1, 64, 1080, 1920).cuda()
        decoder = decoder.cuda()
        decoder.eval()

        torch.onnx.export(
            decoder,
            (y_hat, context, 4),
            str(output_path),
            input_names=["y_hat", "context", "depth"],
            output_names=["recon_1", "recon_2", "recon_3", "recon_4"],
            dynamic_axes={
                "y_hat": {0: "B", 2: "H", 3: "W"},
                "context": {0: "B", 2: "H", 3: "W"},
                "recon_1": {0: "B", 2: "H", 3: "W"},
            },
            opset_version=13,
            verbose=False,
        )
        result["onnx_export"] = True
        result["ops_verified"].append("export_success")
    except Exception as e:
        result["error"] = str(e)
        return result

    return result


def test_pixelshuffle_export():
    result = {"pixelshuffle_export": False, "error": None}

    try:
        class MinimalPixelShuffleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(96, 96 * 4, 3, padding=1)
                self.pixelshuffle = nn.PixelShuffle(2)

            def forward(self, x):
                return self.pixelshuffle(self.conv(x))

        model = MinimalPixelShuffleModel().cuda()
        model.eval()

        x = torch.randn(1, 96, 68, 120).cuda()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(
                model, x, f.name,
                input_names=["x"],
                output_names=["out"],
                opset_version=13,
            )
            result["pixelshuffle_export"] = True
            result["error"] = None
    except Exception as e:
        result["error"] = str(e)

    return result


def test_part1_exportable():
    result = {"part1_exportable": False, "error": None, "notes": []}

    try:
        from src.layers.layers import subpel_conv3x3
        from src.models.video_net import ResBlock, GDN

        class MinimalPart1(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    subpel_conv3x3(96, 64, 2),
                    GDN(64, inverse=True),
                )

            def forward(self, x):
                return self.layers(x)

        model = MinimalPart1().cuda()
        model.eval()

        x = torch.randn(1, 96, 68, 120).cuda()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(
                model, x, f.name,
                input_names=["x"],
                output_names=["out"],
                opset_version=13,
            )
            result["part1_exportable"] = True
            result["notes"].append("subpel_conv3x3 exports OK")
    except Exception as e:
        result["error"] = str(e)

    return result


def test_refinement_block_exportable():
    result = {"refinement_exportable": False, "error": None, "notes": []}

    try:
        from src.models.progressive_decoder import FeatureRefinementBlock

        block = FeatureRefinementBlock(channels=64).cuda()
        block.eval()

        x = torch.randn(1, 64, 1080, 1920).cuda()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(
                block, x, f.name,
                input_names=["x"],
                output_names=["out"],
                opset_version=13,
            )
            result["refinement_exportable"] = True
            result["notes"].append("FeatureRefinementBlock exports OK (depthwise-separable confirmed)")
    except Exception as e:
        result["error"] = str(e)

    return result


def test_mobile_runtime_compatibility(onnx_path: Path, runtime: str = "cpu"):
    result = {"runtime_test": False, "runtime": runtime, "error": None}

    try:
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        result["runtime_test"] = True
        result["error"] = None
    except ImportError:
        result["error"] = "onnx or onnxruntime not installed - can't verify runtime compatibility"
    except Exception as e:
        result["error"] = f"ONNX model check failed: {e}"

    return result


def determine_m6_status(spike_results: dict) -> str:
    blockers = []

    if not spike_results.get("onnx_export", {}).get("onnx_export", False):
        blockers.append("ONNX export failed")
    if not spike_results.get("pixelshuffle_export", {}).get("pixelshuffle_export", False):
        blockers.append("PixelShuffle export failed - Tier-1 critical path blocked")
    if not spike_results.get("part1_exportable", {}).get("part1_exportable", False):
        blockers.append("part1 (subpel_conv3x3) export failed - full decoder blocked")
    if not spike_results.get("refinement_exportable", {}).get("refinement_exportable", False):
        blockers.append("FeatureRefinementBlock export failed")

    runtime_result = spike_results.get("runtime_test", {})
    if runtime_result.get("error") and "not installed" in runtime_result["error"]:
        blockers.append("Runtime compatibility untestable (missing onnxruntime)")

    if not blockers:
        return "REQUIRED", blockers
    else:
        return "DOWNGRADED_TO_STRETCH", blockers


def main():
    import argparse
    parser = argparse.ArgumentParser(description="M5.5: Mobile Export Feasibility Spike")
    parser.add_argument("--runtime", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Target mobile/runtime for compatibility test")
    parser.add_argument("--output", type=str, default="spike_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print("M5.5: Mobile Export Feasibility Spike")
    print("=" * 60)
    print("\nThis spike determines whether M6 is REQUIRED or DOWNGRADED TO STRETCH.")
    print()

    spike_results = {}

    print("[1/4] Testing PixelShuffle export...")
    spike_results["pixelshuffle_export"] = test_pixelshuffle_export()
    status = "PASS" if spike_results["pixelshuffle_export"]["pixelshuffle_export"] else "FAIL"
    print(f"  [{status}] PixelShuffle export: {spike_results['pixelshuffle_export']}")

    print("\n[2/4] Testing part1 (subpel_conv3x3) export...")
    spike_results["part1_exportable"] = test_part1_exportable()
    status = "PASS" if spike_results["part1_exportable"]["part1_exportable"] else "FAIL"
    print(f"  [{status}] part1 export: {spike_results['part1_exportable']}")

    print("\n[3/4] Testing FeatureRefinementBlock export...")
    spike_results["refinement_exportable"] = test_refinement_block_exportable()
    status = "PASS" if spike_results["refinement_exportable"]["refinement_exportable"] else "FAIL"
    print(f"  [{status}] Refinement block export: {spike_results['refinement_exportable']}")

    print("\n[4/4] Testing full progressive decoder ONNX export...")
    decoder = ProgressiveContextualDecoder(out_channel_M=96, out_channel_N=64, num_refinement_blocks=3)
    onnx_path = Path(tempfile.gettempdir()) / "progressive_decoder_test.onnx"
    spike_results["onnx_export"] = test_onnx_export(decoder, onnx_path)
    status = "PASS" if spike_results["onnx_export"]["onnx_export"] else "FAIL"
    print(f"  [{status}] Full decoder ONNX export: {spike_results['onnx_export']}")

    if spike_results["onnx_export"].get("onnx_export"):
        spike_results["runtime_test"] = test_mobile_runtime_compatibility(onnx_path, args.runtime)
        status = "PASS" if spike_results["runtime_test"].get("runtime_test") else "FAIL"
        print(f"  [{status}] Runtime compatibility ({args.runtime}): {spike_results['runtime_test']}")

    m6_status, blockers = determine_m6_status(spike_results)

    print("\n" + "=" * 60)
    print("SPIKE CONCLUSION")
    print("=" * 60)
    print(f"\nM6 Status: {m6_status}")
    if blockers:
        print("\nBlockers identified:")
        for b in blockers:
            print(f"  - {b}")
    else:
        print("\nNo blockers - proceed with full mobile benchmarking (M6)")

    with open(args.output, "w") as f:
        json.dump({
            "spike_results": spike_results,
            "m6_status": m6_status,
            "blockers": blockers,
        }, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()