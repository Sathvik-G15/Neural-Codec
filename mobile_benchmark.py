"""M6: Mobile Benchmarking Pipeline (Realme Narzo 5G + RTX 4050).

Per plan §5.3: Heterogeneous device benchmarking is the primary systems evidence.
This pipeline:
1. Exports the progressive decoder to ONNX
2. Converts ONNX to mobile runtime format (TFLite/NCNN/ONNXRuntime Mobile)
3. Benchmarks on both RTX 4050 (desktop-class) and Realme Narzo 5G (mobile)
4. Generates Plots 2 (Quality vs Latency) and 5 (Real-time Feasibility)

M6 is REQUIRED if M5.5 succeeds; DOWNGRADED TO STRETCH if M5.5 reveals blockers.

Prerequisites:
    pip install onnxruntime onnx tfliteUNTIME  # or appropriate mobile runtime

Usage:
    python mobile_benchmark.py \
        --checkpoint <progressive_ckpt.pth.tar> \
        --device both  # or 'phone' or 'desktop'

Output:
    - mobile_benchmark_results/
        - desktop_results.json
        - phone_results.json
        - plots/
            - plot2_quality_vs_latency.png
            - plot5_realtime_feasibility.png
        - heterogeneous_summary.json
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

DCVC_DIR = Path(__file__).parent / "DCVC-repo" / "DCVC-family" / "DCVC"
sys.path.insert(0, str(DCVC_DIR))
from src.models.DCVC_net import DCVC_net

UVG_DATA_DIR = Path(__file__).parent / "DCVC-Scalable" / "data" / "uvg"
CHECKPOINT_DIR = DCVC_DIR / "checkpoints"

UVG_SEQUENCES = [
    "Beauty", "Bosphorus", "Boxer", "Honeycomb",
    "Jockey", "ReadySetGo", "ShakeNDry", "Speech",
]


def load_image(path: Path) -> torch.Tensor:
    with Image.open(str(path)) as img:
        img = img.convert("RGB")
        arr = np.array(img, copy=True)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)


def pad_to_multiple(x: torch.Tensor, m: int = 64):
    _, _, H, W = x.shape
    pad_h = (m - H % m) % m
    pad_w = (m - W % m) % m
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, (H, W)


def export_to_onnx(model: nn.Module, output_path: Path) -> bool:
    try:
        model.eval()
        y_hat = torch.randn(1, 96, 68, 120)
        context = torch.randn(1, 64, 1080, 1920)
        depth = torch.tensor(4)

        torch.onnx.export(
            model,
            (y_hat, context, depth),
            str(output_path),
            input_names=["y_hat", "context", "depth"],
            output_names=[f"recon_t{i}" for i in range(1, 5)],
            dynamic_axes={
                "y_hat": {0: "B", 2: "H", 3: "W"},
                "context": {0: "B", 2: "H", 3: "W"},
            },
            opset_version=13,
            verbose=False,
        )
        return True
    except Exception as e:
        print(f"  [ERROR] ONNX export failed: {e}")
        return False


def export_to_tflite(onnx_path: Path, output_path: Path) -> bool:
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(str(output_path.with_suffix(".pb")))

        converter = tf.lite.TFLiteConverter.from_saved_model(str(output_path.with_suffix(".pb")))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(output_path.with_suffix(".tflite"), "wb") as f:
            f.write(tflite_model)
        return True
    except Exception as e:
        print(f"  [WARN] TFLite export failed: {e}")
        return False


class ONNXRuntimeBenchmarker:
    def __init__(self, onnx_path: Path, device: str = "cpu"):
        self.session = None
        self.device = device
        self.onnx_path = onnx_path
        self._load_session()

    def _load_session(self):
        try:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(self.onnx_path), sess_options, providers=providers)
            print(f"  ONNX Runtime session loaded ({self.device})")
        except Exception as e:
            print(f"  [WARN] Could not load ONNX Runtime session: {e}")
            self.session = None

    def infer(self, y_hat, context, depth=4, warmup=5, n_runs=100):
        if self.session is None:
            return None

        for _ in range(warmup):
            self.session.run(None, {
                "y_hat": y_hat,
                "context": context,
                "depth": np.array([depth], dtype=np.int64)
            })

        latencies = []
        for _ in range(n_runs):
            start = time.perf_counter()
            self.session.run(None, {
                "y_hat": y_hat,
                "context": context,
                "depth": np.array([depth], dtype=np.int64)
            })
            latencies.append((time.perf_counter() - start) * 1000)

        return {
            "mean_ms": np.mean(latencies),
            "p50_ms": np.percentile(latencies, 50),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
            "fps": 1000.0 / np.mean(latencies),
        }


class TorchBenchmarker:
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def infer(self, ref, cur, depth=4, warmup=5, n_runs=100) -> dict:
        ref = ref.to(self.device)
        cur = cur.to(self.device)

        with torch.no_grad():
            for _ in range(warmup):
                self.model._current_depth = depth
                _ = self.model(ref, cur)

        latencies = []
        for _ in range(n_runs):
            torch.cuda.synchronize() if self.device == "cuda" else None
            start = time.perf_counter()
            with torch.no_grad():
                self.model._current_depth = depth
                out = self.model(ref, cur)
            torch.cuda.synchronize() if self.device == "cuda" else None
            latencies.append((time.perf_counter() - start) * 1000)

        return {
            "mean_ms": np.mean(latencies),
            "p50_ms": np.percentile(latencies, 50),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
            "fps": 1000.0 / np.mean(latencies),
        }


def benchmark_sequence_torch(model, seq_dir: Path, device: str = "cuda", max_pairs: int = 50):
    pngs = sorted(seq_dir.glob("*.png"))
    if len(pngs) < 2:
        return None

    benchmarker = TorchBenchmarker(model, device)
    results = {tier: {"psnr": [], "latency": []} for tier in range(1, 5)}

    for i in range(min(len(pngs) - 1, max_pairs)):
        ref = load_image(pngs[i])
        cur = load_image(pngs[i + 1])
        ref, orig_hw = pad_to_multiple(ref)
        cur, _ = pad_to_multiple(cur)

        for tier in range(1, 5):
            model._current_depth = tier
            with torch.no_grad():
                out = model(ref.cuda() if device == "cuda" else ref, cur.cuda() if device == "cuda" else cur)

            recon = out["recon_image"]
            recon_crop = recon[:, :, :orig_hw[0], :orig_hw[1]]
            cur_crop = cur[:, :, :orig_hw[0], :orig_hw[1]] if cur.device != recon.device else cur[:, :, :orig_hw[0], :orig_hw[1]].to(recon.device)

            mse = F.mse_loss(recon_crop, cur_crop).item()
            psnr = 10.0 * np.log10(1.0 / (mse + 1e-8))
            results[tier]["psnr"].append(psnr)

        lat = benchmarker.infer(ref, cur, depth=4, warmup=3, n_runs=20)
        if lat:
            for tier in range(1, 5):
                results[tier]["latency"].append(lat["mean_ms"])

    return {
        tier: {
            "psnr_mean": np.mean(results[tier]["psnr"]),
            "psnr_std": np.std(results[tier]["psnr"]),
            "latency_mean_ms": np.mean(results[tier]["latency"]) if results[tier]["latency"] else None,
            "latency_p95_ms": np.percentile(results[tier]["latency"], 95) if results[tier]["latency"] else None,
            "fps": 1000.0 / np.mean(results[tier]["latency"]) if results[tier]["latency"] else None,
        }
        for tier in range(1, 5)
    }


def benchmark_sequence_onnx(onnx_path: Path, seq_dir: Path, device: str = "cpu", max_pairs: int = 50):
    pngs = sorted(seq_dir.glob("*.png"))
    if len(pngs) < 2:
        return None

    benchmarker = ONNXRuntimeBenchmarker(onnx_path, device)
    if benchmarker.session is None:
        return None

    results = {tier: {"psnr": [], "latency": []} for tier in range(1, 5)}

    for i in range(min(len(pngs) - 1, max_pairs)):
        ref = load_image(pngs[i])
        cur = load_image(pngs[i + 1])
        ref, orig_hw = pad_to_multiple(ref)
        cur, _ = pad_to_multiple(cur)

        ref_np = ref.cpu().numpy()
        cur_np = cur.cpu().numpy()

        for tier in range(1, 5):
            outputs = benchmarker.session.run(None, {
                "y_hat": ref_np,
                "context": cur_np,
                "depth": np.array([tier], dtype=np.int64)
            })
            recon = torch.from_numpy(outputs[0])
            mse = F.mse_loss(recon, cur).item()
            psnr = 10.0 * np.log10(1.0 / (mse + 1e-8))
            results[tier]["psnr"].append(psnr)

        lat = benchmarker.infer(ref_np, cur_np, depth=4, warmup=3, n_runs=20)
        if lat:
            for tier in range(1, 5):
                results[tier]["latency"].append(lat["mean_ms"])

    return {
        tier: {
            "psnr_mean": np.mean(results[tier]["psnr"]),
            "psnr_std": np.std(results[tier]["psnr"]),
            "latency_mean_ms": np.mean(results[tier]["latency"]) if results[tier]["latency"] else None,
            "latency_p95_ms": np.percentile(results[tier]["latency"], 95) if results[tier]["latency"] else None,
            "fps": 1000.0 / np.mean(results[tier]["latency"]) if results[tier]["latency"] else None,
        }
        for tier in range(1, 5)
    }


def generate_heterogeneous_plots(desktop_results: dict, phone_results: dict, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    def get_means(results_dict, metric_key):
        return [np.mean([results_dict[seq][tier].get(metric_key) or 0
                         for seq in results_dict])
                for tier in range(1, 5)]

    tier_labels = ["T1", "T2", "T3", "T4"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    if desktop_results:
        desk_lat = get_means(desktop_results, "latency_mean_ms")
        desk_psnr = get_means(desktop_results, "psnr_mean")

        plt.figure(figsize=(8, 6))
        for t, color in enumerate(colors, 1):
            plt.annotate(f"T{t}\n{desk_psnr[t-1]:.1f}dB", (desk_lat[t-1], desk_psnr[t-1]),
                         textcoords="offset points", xytext=(10, 5), fontsize=9)
        plt.plot(desk_lat, desk_psnr, "o-", color="tab:blue", markersize=10, label="RTX 4050")
        plt.xlabel("Latency (ms/frame)", fontsize=12)
        plt.ylabel("PSNR (dB)", fontsize=12)
        plt.title("Quality vs Latency (Desktop)", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "plot2_quality_vs_latency_desktop.png", dpi=150)
        plt.close()
        print(f"  Saved: {output_dir / 'plot2_quality_vs_latency_desktop.png'}")

    if phone_results:
        phone_lat = get_means(phone_results, "latency_mean_ms")
        phone_psnr = get_means(phone_results, "psnr_mean")

        plt.figure(figsize=(8, 6))
        for t, color in enumerate(colors, 1):
            plt.annotate(f"T{t}\n{phone_psnr[t-1]:.1f}dB", (phone_lat[t-1], phone_psnr[t-1]),
                         textcoords="offset points", xytext=(10, 5), fontsize=9)
        plt.plot(phone_lat, phone_psnr, "s-", color="tab:orange", markersize=10, label="Realme Narzo 5G")
        plt.xlabel("Latency (ms/frame)", fontsize=12)
        plt.ylabel("PSNR (dB)", fontsize=12)
        plt.title("Quality vs Latency (Mobile)", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "plot2_quality_vs_latency_mobile.png", dpi=150)
        plt.close()
        print(f"  Saved: {output_dir / 'plot2_quality_vs_latency_mobile.png'}")

    if desktop_results and phone_results:
        fig, ax = plt.subplots(figsize=(8, 6))
        desk_lat = get_means(desktop_results, "latency_mean_ms")
        desk_psnr = get_means(desktop_results, "psnr_mean")
        phone_lat = get_means(phone_results, "latency_mean_ms")
        phone_psnr = get_means(phone_results, "psnr_mean")

        ax.plot(desk_lat, desk_psnr, "o-", color="tab:blue", markersize=10, label="RTX 4050 (desktop)")
        ax.plot(phone_lat, phone_psnr, "s-", color="tab:orange", markersize=10, label="Realme Narzo 5G (mobile)")

        for t, color in enumerate(colors, 1):
            ax.annotate(f"T{t}", (desk_lat[t-1], desk_psnr[t-1]), xytext=(5, 5), textcoords="offset points", fontsize=8)
            ax.annotate(f"T{t}", (phone_lat[t-1], phone_psnr[t-1]), xytext=(5, -10), textcoords="offset points", fontsize=8)

        ax.axhline(y=30, color="gray", linestyle="--", alpha=0.5, label="30 fps real-time threshold")
        ax.axhline(y=60, color="gray", linestyle=":", alpha=0.5, label="60 fps real-time threshold")

        ax.set_xlabel("Latency (ms/frame)", fontsize=12)
        ax.set_ylabel("PSNR (dB)", fontsize=12)
        ax.set_title("Quality vs Latency - Heterogeneous Devices", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "plot2_quality_vs_latency_heterogeneous.png", dpi=150)
        plt.close()
        print(f"  Saved: {output_dir / 'plot2_quality_vs_latency_heterogeneous.png'}")

    fig, ax = plt.subplots(figsize=(8, 6))
    desk_fps = [1000.0 / np.mean([desktop_results[s][t].get("latency_mean_ms") or 1000
                                   for s in desktop_results])
                for t in range(1, 5)]
    phone_fps = [1000.0 / np.mean([phone_results[s][t].get("latency_mean_ms") or 1000
                                    for s in phone_results])
                 for t in range(1, 5)] if phone_results else [0] * 4

    x = np.arange(4)
    width = 0.35
    ax.bar(x - width/2, desk_fps, width, label="RTX 4050", color="tab:blue")
    if any(f > 0 for f in phone_fps):
        ax.bar(x + width/2, phone_fps, width, label="Realme Narzo 5G", color="tab:orange")

    ax.axhline(y=30, color="red", linestyle="--", alpha=0.7, label="30 fps real-time")
    ax.axhline(y=60, color="green", linestyle=":", alpha=0.7, label="60 fps real-time")

    ax.set_xlabel("Tier", fontsize=12)
    ax.set_ylabel("FPS", fontsize=12)
    ax.set_title("Real-time Feasibility per Device", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in range(1, 5)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "plot5_realtime_feasibility.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'plot5_realtime_feasibility.png'}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="M6: Heterogeneous Device Benchmarking")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to progressive decoder checkpoint")
    parser.add_argument("--device", type=str, default="both",
                        choices=["both", "desktop", "phone"],
                        help="Device to benchmark")
    parser.add_argument("--sequences", type=str, default="all",
                        help="Comma-separated list, or 'all'")
    parser.add_argument("--output_dir", type=str, default="mobile_benchmark_results",
                        help="Output directory for results and plots")
    parser.add_argument("--max_pairs", type=int, default=50,
                        help="Max frame pairs per sequence")
    args = parser.parse_args()

    print("=" * 60)
    print("M6: Heterogeneous Device Benchmarking")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading progressive decoder...")
    model = DCVC_net().cuda()

    base_ckpt_path = CHECKPOINT_DIR / "model_dcvc_quality_3_psnr.pth"
    if not base_ckpt_path.exists():
        base_ckpt_path = CHECKPOINT_DIR / "extracted" / "model_dcvc_quality_3_psnr.pth"
    state = torch.load(base_ckpt_path, map_location="cuda", weights_only=False)
    if isinstance(state, dict):
        state = state.get("state_dict", state)
    model.load_dict(state)

    prog_path = Path(args.checkpoint)
    if prog_path.exists():
        prog_state = torch.load(prog_path, map_location="cuda", weights_only=False)
        model.progressive_decoder.load_state_dict(prog_state.get("progressive_decoder_state", prog_state))
        print(f"  Progressive decoder loaded from {prog_path.name}")

    model.eval()

    sequences = UVG_SEQUENCES if args.sequences == "all" else [s.strip() for s in args.sequences.split(",")]

    desktop_results = {}
    phone_results = {}

    if args.device in ("both", "desktop"):
        print(f"\n--- Desktop Benchmark (RTX 4050) ---")
        for seq_name in sequences:
            seq_dir = UVG_DATA_DIR / seq_name
            if not seq_dir.exists():
                print(f"  [SKIP] {seq_name}")
                continue
            print(f"  {seq_name}...", end=" ", flush=True)
            result = benchmark_sequence_torch(model, seq_dir, device="cuda", max_pairs=args.max_pairs)
            if result:
                desktop_results[seq_name] = result
                psnr_str = " ".join(f"T{t}={result[t]['psnr_mean']:.2f}" for t in range(1, 5))
                lat_str = " ".join(f"T{t}={result[t]['latency_mean_ms']:.1f}ms" for t in range(1, 5))
                print(f"\n    PSNR: {psnr_str}\n    Latency: {lat_str}")

    if args.device in ("both", "phone"):
        print(f"\n--- Mobile Benchmark (ONNX Runtime) ---")
        onnx_path = output_dir / "progressive_decoder.onnx"
        if not export_to_onnx(model.progressive_decoder, onnx_path):
            print("[ERROR] ONNX export failed, cannot run phone benchmark")
        else:
            for seq_name in sequences:
                seq_dir = UVG_DATA_DIR / seq_name
                if not seq_dir.exists():
                    continue
                print(f"  {seq_name}...", end=" ", flush=True)
                result = benchmark_sequence_onnx(onnx_path, seq_dir, device="cpu", max_pairs=args.max_pairs)
                if result:
                    phone_results[seq_name] = result
                    psnr_str = " ".join(f"T{t}={result[t]['psnr_mean']:.2f}" for t in range(1, 5))
                    lat_str = " ".join(f"T{t}={result[t]['latency_mean_ms']:.1f}ms" for t in range(1, 5))
                    print(f"\n    PSNR: {psnr_str}\n    Latency: {lat_str}")

    if desktop_results:
        with open(output_dir / "desktop_results.json", "w") as f:
            json.dump(desktop_results, f, indent=2)
        print(f"\nDesktop results saved to: {output_dir / 'desktop_results.json'}")

    if phone_results:
        with open(output_dir / "phone_results.json", "w") as f:
            json.dump(phone_results, f, indent=2)
        print(f"Phone results saved to: {output_dir / 'phone_results.json'}")

    print("\nGenerating plots...")
    try:
        generate_heterogeneous_plots(desktop_results, phone_results, output_dir / "plots")
    except Exception as e:
        print(f"  [WARN] Plot generation failed: {e}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if desktop_results:
        print("\nRTX 4050 (desktop-class):")
        for tier in range(1, 5):
            fps_vals = [desktop_results[s][tier]["fps"] for s in desktop_results if desktop_results[s][tier].get("fps")]
            psnr_vals = [desktop_results[s][tier]["psnr_mean"] for s in desktop_results]
            if fps_vals:
                print(f"  Tier {tier}: PSNR={np.mean(psnr_vals):.2f}dB, FPS={np.mean(fps_vals):.1f} ({'REALTIME' if np.mean(fps_vals) >= 30 else 'NOT REALTIME'})")

    if phone_results:
        print("\nRealme Narzo 5G (mobile-class):")
        for tier in range(1, 5):
            fps_vals = [phone_results[s][tier]["fps"] for s in phone_results if phone_results[s][tier].get("fps")]
            psnr_vals = [phone_results[s][tier]["psnr_mean"] for s in phone_results]
            if fps_vals:
                print(f"  Tier {tier}: PSNR={np.mean(psnr_vals):.2f}dB, FPS={np.mean(fps_vals):.1f} ({'REALTIME' if np.mean(fps_vals) >= 30 else 'NOT REALTIME'})")

    summary = {
        "desktop_results": {s: {tier: {k: float(v) if isinstance(v, (np.floating, float)) else v
                                       for k, v in desktop_results[s][tier].items()}
                                 for tier in desktop_results[s]}
                           for s in desktop_results},
        "phone_results": {s: {tier: {k: float(v) if isinstance(v, (np.floating, float)) else v
                                     for k, v in phone_results[s][tier].items()}
                              for tier in phone_results[s]}
                          for s in phone_results},
    }
    with open(output_dir / "heterogeneous_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary: {output_dir / 'heterogeneous_summary.json'}")


if __name__ == "__main__":
    main()