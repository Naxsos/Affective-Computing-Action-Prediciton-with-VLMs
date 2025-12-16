#!/usr/bin/env python3
"""
Example script demonstrating how to use the VLM Benchmarking Pipeline.

This script shows how to:
1. Configure samples with ground truth labels
2. Run benchmarks across different models
3. Compare results and generate visualizations

Usage:
    python run_benchmark.py
    python run_benchmark.py --models gpt-4o gpt-4o-mini
    python run_benchmark.py --compare
"""

import os
import argparse
from pathlib import Path

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

from benchmark import BenchmarkRunner
from benchmark_visualizer import BenchmarkVisualizer


# ============================================================================
# Sample Configuration
# ============================================================================

# Define your samples with ground truth labels for scoring
# Each sample is a directory of frames with an optional ground truth action

SAMPLE_CONFIGS = [
    {
        "id": "P01_01_t104.0s_n30_i7",
        "image_dir": "output_frames/P01_01_t104.0s_n30_i7_20251124_112114",
        "ground_truth": "The person will pick up a cucumber and place it on the cutting board"  # Update with actual ground truth
    },
]

# Default prompt for action prediction
DEFAULT_PROMPT = """
    You are analyzing a sequence of video frames from an egocentric (first-person) perspective.

    Based on the sequence of actions visible in these frames, predict the next 2 most plausible actions the person will perform.

    Format your response as:
    1. [Most likely action]: Brief description
    2. [Second most likely action]: Brief description

    Be specific about the objects and actions involved."""


# ============================================================================
# Benchmark Functions
# ============================================================================

def run_single_model_benchmark(
    model: str = "gpt-4o",
    samples: list = None,
    prompt: str = None,
    output_dir: str = "benchmark_results",
    max_images: int = None
):
    """Run benchmark with a single model."""
    samples = samples or SAMPLE_CONFIGS
    prompt = prompt or DEFAULT_PROMPT
    
    print(f"\n🚀 Running benchmark with model: {model}")
    print(f"   Samples: {len(samples)}")
    
    runner = BenchmarkRunner(
        prediction_model=model,
        judge_model="gpt-4o"  # Use cheaper model for judging
    )
    
    benchmark = runner.run_benchmark(
        samples=samples,
        prompt=prompt,
        max_images=max_images,
        verbose=True
    )
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / f"{benchmark.run_id}.json"
    runner.save_results(benchmark, str(results_file))
    
    # Generate visualizations
    visualizer = BenchmarkVisualizer()
    visualizer.plot_single_run(benchmark, output_dir=str(output_path))
    
    return benchmark, str(results_file)


def run_multi_model_comparison(
    models: list[str] = None,
    samples: list = None,
    prompt: str = None,
    output_dir: str = "benchmark_results"
):
    """Run benchmark across multiple models and compare."""
    models = models or ["gpt-4o", "gpt-4o-mini"]
    samples = samples or SAMPLE_CONFIGS
    prompt = prompt or DEFAULT_PROMPT
    
    print(f"\n🔬 Running multi-model comparison")
    print(f"   Models: {models}")
    print(f"   Samples: {len(samples)}")
    
    result_files = []
    
    for model in models:
        _, result_file = run_single_model_benchmark(
            model=model,
            samples=samples,
            prompt=prompt,
            output_dir=output_dir
        )
        result_files.append(result_file)
    
    # Generate comparison plot
    if len(result_files) > 1:
        visualizer = BenchmarkVisualizer()
        visualizer.compare_runs(result_files, output_dir=output_dir)
    
    return result_files


def quick_test(image_dir: str = None, ground_truth: str = None):
    """Run a quick test on a single directory."""
    if image_dir is None:
        # Use first available sample
        sample_dir = Path("output_frames")
        if sample_dir.exists():
            subdirs = [d for d in sample_dir.iterdir() if d.is_dir()]
            if subdirs:
                image_dir = str(subdirs[0])
            else:
                print("❌ No sample directories found in output_frames/")
                return
        else:
            print("❌ No output_frames directory found. Run extract_frames.py first.")
            return
    
    samples = [{
        "id": Path(image_dir).name,
        "image_dir": image_dir,
        "ground_truth": ground_truth or "No ground truth provided"
    }]
    
    run_single_model_benchmark(
        model="gpt-4o-mini",  # Use cheaper model for quick test
        samples=samples,
        max_images=10  # Limit images for faster test
    )


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run VLM action prediction benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with a single directory
  python run_benchmark.py --quick-test output_frames/P01_01_t104.0s_n20_i5_20251124_111814

  # Run benchmark with default samples
  python run_benchmark.py

  # Compare multiple models
  python run_benchmark.py --compare --models gpt-4o gpt-4o-mini

  # Custom prompt
  python run_benchmark.py -p "What will happen next in this video?"
        """
    )
    
    parser.add_argument(
        "--quick-test",
        type=str,
        nargs='?',
        const='auto',
        help="Run a quick test on a single directory (uses first available if not specified)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison across multiple models"
    )
    parser.add_argument(
        "--models",
        nargs='+',
        default=["gpt-4o"],
        help="Model(s) to benchmark (default: gpt-4o)"
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default=None,
        help="Custom prediction prompt"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="benchmark_results",
        help="Output directory (default: benchmark_results)"
    )
    parser.add_argument(
        "-n", "--max-images",
        type=int,
        default=None,
        help="Maximum images per sample"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Ground truth for quick test"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY environment variable not set.")
        print("\n💡 Set it with:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return 1
    
    try:
        if args.quick_test:
            image_dir = None if args.quick_test == 'auto' else args.quick_test
            quick_test(image_dir, args.ground_truth)
        elif args.compare:
            run_multi_model_comparison(
                models=args.models,
                prompt=args.prompt,
                output_dir=args.output
            )
        else:
            for model in args.models:
                run_single_model_benchmark(
                    model=model,
                    prompt=args.prompt,
                    output_dir=args.output,
                    max_images=args.max_images
                )
        
        print(f"\n✅ Benchmark complete! Results saved to: {args.output}/")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

