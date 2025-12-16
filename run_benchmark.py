#!/usr/bin/env python3
"""
Extended Benchmark Script with Frame Ablation.

Enhancements over original:
- Supports ablation over max_images (number of frames)
- Aggregates runtime, cost, and accuracy per frame count
- Produces clearer, publication-ready plots

This is a DROP-IN replacement / extension of the original run_benchmark.py.
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from benchmark import BenchmarkRunner
from benchmark_visualizer import BenchmarkVisualizer
import matplotlib.pyplot as plt


# ============================================================================
# Dataset Configuration (Video-based)
# ============================================================================

DATASET_DIR = "dataset"

DEFAULT_PROMPT = """
You are analyzing a short egocentric video clip.

Based on the observed frames, predict the NEXT most likely action.
Respond ONLY in the following format:

Verb: <verb>
Noun: <noun>
"""


def load_video_dataset(dataset_dir: str) -> List[Dict[str, Any]]:
    """
    Load videos from a dataset directory.
    Expected filename format: verb_noun.mov
    """
    samples = []

    for video_path in Path(dataset_dir).glob("*.mov"):
        name = video_path.stem  # verb_noun
        if "_" not in name:
            raise ValueError(f"Invalid filename (expected verb_noun.mov): {video_path.name}")

        verb, noun = name.split("_", 1)

        samples.append({
            "id": name,
            "video_path": str(video_path),
            "ground_truth": {
                "verb": verb,
                "noun": noun,
                "text": f"{verb} {noun}",
            },
        })

    if not samples:
        raise RuntimeError(f"No .mov files found in {dataset_dir}")

    return samples


# ============================================================================
# Ablation Benchmark
# ============================================================================

def run_frame_ablation(
    model: str,
    samples: List[Dict[str, Any]],
    prompt: str,
    max_images_list: List[int],
    output_dir: str = "benchmark_results",
) -> List[str]:
    """
    Run an ablation study over different numbers of input frames.
    """

    print("\n🔬 Running frame ablation study")
    print(f"   Model: {model}")
    print(f"   Frame counts: {max_images_list}")

    runner = BenchmarkRunner(
        prediction_model=model,
        judge_model="gpt-4o-mini"
    )

    aggregated = defaultdict(list)
    run_files = []

    for max_images in max_images_list:
        print(f"\n➡️  max_images = {max_images}")

        benchmark = runner.run_benchmark(
            samples=samples,
            prompt=prompt,
            max_images=max_images,
            verbose=True
        )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"{benchmark.run_id}_n{max_images}.json"
        runner.save_results(benchmark, str(result_file))
        run_files.append(str(result_file))

        aggregated["max_images"].append(max_images)
        aggregated["avg_runtime"].append(benchmark.avg_processing_time)
        aggregated["avg_cost"].append(benchmark.total_cost_usd / benchmark.total_samples)
        aggregated["avg_accuracy"].append(benchmark.avg_accuracy)

    _plot_ablation(aggregated, output_dir)
    return run_files


# ============================================================================
# Plotting
# ============================================================================

def _plot_ablation(stats: Dict[str, List], output_dir: str):
    """Generate ablation study plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Runtime
    plt.figure(figsize=(7, 4))
    plt.plot(stats["max_images"], stats["avg_runtime"], marker="o")
    plt.xlabel("Number of Frames")
    plt.ylabel("Avg Runtime (s)")
    plt.title("Runtime vs Number of Frames")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "ablation_runtime.png", dpi=150)
    plt.close()

    # Cost
    plt.figure(figsize=(7, 4))
    plt.plot(stats["max_images"], stats["avg_cost"], marker="o")
    plt.xlabel("Number of Frames")
    plt.ylabel("Avg Cost per Sample ($)")
    plt.title("Cost vs Number of Frames")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "ablation_cost.png", dpi=150)
    plt.close()

    # Accuracy
    if all(v is not None for v in stats["avg_accuracy"]):
        plt.figure(figsize=(7, 4))
        plt.plot(stats["max_images"], stats["avg_accuracy"], marker="o")
        plt.xlabel("Number of Frames")
        plt.ylabel("Avg Accuracy (LLM Judge)")
        plt.title("Accuracy vs Number of Frames")
        plt.ylim(0, 100)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "ablation_accuracy.png", dpi=150)
        plt.close()

    print(f"📊 Ablation plots saved to {output_dir}/")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extended VLM Benchmark with Frame Ablation")

    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--frames", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--output", default="benchmark_results")
    parser.add_argument("-p", "--prompt", default=None)

    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    # Load dataset
    samples = load_video_dataset(DATASET_DIR)

    run_frame_ablation(
        model=args.model,
        samples=samples,
        prompt=args.prompt or DEFAULT_PROMPT,
        max_images_list=args.frames,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()