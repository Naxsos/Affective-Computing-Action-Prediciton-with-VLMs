#!/usr/bin/env python3
"""
Benchmarking Pipeline for VLM Action Prediction.

This module provides a complete benchmarking system that:
1. Runs VLM predictions on video frames
2. Uses LLM-as-a-judge to score predictions
3. Tracks metrics (tokens, cost, processing time, accuracy)
4. Generates visualizations
"""

import os
import json
import time
import base64
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
import statistics

from openai import OpenAI


# ============================================================================
# Data Classes for Metrics
# ============================================================================

@dataclass
class TokenUsage:
    """Track token usage for a single API call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    
    def calculate_cost(self, input_price_per_million: float = 2.50, output_price_per_million: float = 10.00) -> float:
        """Calculate cost based on pricing (default: GPT-4o pricing as of 2024)."""
        input_cost = (self.prompt_tokens / 1_000_000) * input_price_per_million
        output_cost = (self.completion_tokens / 1_000_000) * output_price_per_million
        self.estimated_cost_usd = input_cost + output_cost
        return self.estimated_cost_usd


@dataclass
class PredictionResult:
    """Result from a single prediction run."""
    sample_id: str
    model: str
    prompt: str
    prediction: str
    ground_truth: Optional[str] = None
    processing_time_seconds: float = 0.0
    tokens: TokenUsage = field(default_factory=TokenUsage)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Scoring fields (filled by LLM judge)
    accuracy_score: Optional[float] = None  # 0-100 scale
    judge_reasoning: Optional[str] = None
    judge_model: Optional[str] = None


@dataclass
class BenchmarkRun:
    """Complete benchmark run with multiple samples."""
    run_id: str
    model: str
    prompt: str
    results: list[PredictionResult] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    
    @property
    def total_samples(self) -> int:
        return len(self.results)
    
    @property
    def total_tokens(self) -> int:
        return sum(r.tokens.total_tokens for r in self.results)
    
    @property
    def total_cost_usd(self) -> float:
        return sum(r.tokens.estimated_cost_usd for r in self.results)
    
    @property
    def avg_processing_time(self) -> float:
        if not self.results:
            return 0.0
        return statistics.mean(r.processing_time_seconds for r in self.results)
    
    @property
    def avg_accuracy(self) -> Optional[float]:
        scores = [r.accuracy_score for r in self.results if r.accuracy_score is not None]
        return statistics.mean(scores) if scores else None


# ============================================================================
# Image Encoding Utilities
# ============================================================================

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_images_from_directory(directory: str, max_images: Optional[int] = None) -> list[str]:
    """Load image file paths from a directory, sorted by name."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise ValueError(f"Directory not found: {directory}")
    
    image_paths = sorted([
        str(f) for f in directory_path.iterdir()
        if f.suffix.lower() in image_extensions
    ])
    
    if not image_paths:
        raise ValueError(f"No images found in directory: {directory}")
    
    return image_paths[:max_images] if max_images else image_paths


# ============================================================================
# VLM Prediction Engine
# ============================================================================

class VLMPredictor:
    """Handles predictions using Vision Language Models."""
    
    # Pricing per 1M tokens (input, output) - update as needed
    MODEL_PRICING = {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-4o-2024-11-20": (2.50, 10.00),
        # GPT-4.1 models (2025)
        "gpt-4.1": (2.00, 8.00),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1-nano": (0.10, 0.40),
        # O-series reasoning models
        "o1": (15.00, 60.00),
        "o1-mini": (1.10, 4.40),
        "o1-preview": (15.00, 60.00),
        "o3-mini": (1.10, 4.40),
        # GPT-5 series (Responses API)
        "gpt-5.1": (5.00, 20.00),  # Placeholder pricing - update when known
        "gpt-5.1-mini": (1.00, 4.00),
    }
    
    # Models that use the new Responses API
    RESPONSES_API_MODELS = {"gpt-5.1", "gpt-5.1-mini"}
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
    
    def predict(
        self,
        image_paths: list[str],
        prompt: str,
        detail: str = "auto"
    ) -> tuple[str, TokenUsage, float]:
        """
        Run prediction on images.
        
        Returns:
            tuple: (prediction_text, token_usage, processing_time_seconds)
        """
        # Check if model uses new Responses API
        if self.model in self.RESPONSES_API_MODELS:
            return self._predict_responses_api(image_paths, prompt, detail)
        else:
            return self._predict_chat_api(image_paths, prompt, detail)
    
    def _predict_chat_api(
        self,
        image_paths: list[str],
        prompt: str,
        detail: str = "auto"
    ) -> tuple[str, TokenUsage, float]:
        """Use the Chat Completions API (gpt-4o, gpt-4.1, etc.)"""
        # Build content with prompt and images
        content = [{"type": "text", "text": prompt}]
        
        for image_path in image_paths:
            base64_image = encode_image_to_base64(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": detail
                }
            })
        
        # Time the API call
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=4096
        )
        
        processing_time = time.time() - start_time
        
        # Extract token usage
        usage = response.usage
        tokens = TokenUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens
        )
        
        # Apply model-specific pricing
        if self.model in self.MODEL_PRICING:
            input_price, output_price = self.MODEL_PRICING[self.model]
            tokens.calculate_cost(input_price, output_price)
        else:
            tokens.calculate_cost()  # Use default GPT-4o pricing
        
        prediction = response.choices[0].message.content
        
        return prediction, tokens, processing_time
    
    def _predict_responses_api(
        self,
        image_paths: list[str],
        prompt: str,
        detail: str = "auto"
    ) -> tuple[str, TokenUsage, float]:
        """Use the new Responses API (gpt-5.1, etc.)"""
        # Build input with images for Responses API
        input_content = [{"type": "input_text", "text": prompt}]
        
        for image_path in image_paths:
            base64_image = encode_image_to_base64(image_path)
            input_content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{base64_image}",
                "detail": detail
            })
        
        # Time the API call
        start_time = time.time()
        from openai import OpenAI
        client = OpenAI()
        
        response = client.responses.create(
            model=self.model,
            input=input_content,
            reasoning={"effort": "low"},
            text={"verbosity": "medium"}
        )
        
        processing_time = time.time() - start_time
        
        # Extract token usage from Responses API
        usage = response.usage
        tokens = TokenUsage(
            prompt_tokens=getattr(usage, 'input_tokens', 0),
            completion_tokens=getattr(usage, 'output_tokens', 0),
            total_tokens=getattr(usage, 'input_tokens', 0) + getattr(usage, 'output_tokens', 0)
        )
        
        # Apply model-specific pricing
        if self.model in self.MODEL_PRICING:
            input_price, output_price = self.MODEL_PRICING[self.model]
            tokens.calculate_cost(input_price, output_price)
        else:
            tokens.calculate_cost()
        
        prediction = response.output_text
        
        return prediction, tokens, processing_time


# ============================================================================
# LLM-as-Judge Evaluator
# ============================================================================

class LLMJudge:
    """Uses an LLM to evaluate prediction quality against ground truth."""
    
    JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for action prediction tasks. 
Your job is to assess how well a model's prediction matches the ground truth action.

You will receive:
1. The predicted action(s) from a Vision Language Model
2. The ground truth action(s) that actually occurred

Score the prediction on a scale of 0-100:
- 100: Perfect match - prediction exactly matches ground truth
- 80-99: Excellent - prediction captures the main action correctly with minor differences
- 60-79: Good - prediction is mostly correct but misses some details
- 40-59: Partial - prediction captures some aspects but misses key elements
- 20-39: Poor - prediction has minimal overlap with ground truth
- 0-19: Wrong - prediction is completely incorrect or unrelated

Respond in JSON format:
{
    "score": <0-100>,
    "reasoning": "<brief explanation of your scoring>"
}"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required for LLM judge.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
    
    def evaluate(self, prediction: str, ground_truth: str) -> tuple[float, str]:
        """
        Evaluate a prediction against ground truth.
        
        Returns:
            tuple: (score 0-100, reasoning string)
        """
        user_prompt = f"""Please evaluate this action prediction:

**Predicted Action(s):**
{prediction}

**Ground Truth Action(s):**
{ground_truth}

Provide your evaluation in JSON format with 'score' (0-100) and 'reasoning' fields."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        
        try:
            result = json.loads(result_text)
            score = float(result.get("score", 0))
            reasoning = result.get("reasoning", "No reasoning provided")
        except (json.JSONDecodeError, ValueError):
            # Fallback if JSON parsing fails
            score = 0.0
            reasoning = f"Failed to parse judge response: {result_text}"
        
        return score, reasoning


# ============================================================================
# Benchmark Runner
# ============================================================================

class BenchmarkRunner:
    """Orchestrates the complete benchmarking pipeline."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        prediction_model: str = "gpt-4o",
        judge_model: str = "gpt-4o-mini"
    ):
        self.predictor = VLMPredictor(api_key=api_key, model=prediction_model)
        self.judge = LLMJudge(api_key=api_key, model=judge_model)
        self.prediction_model = prediction_model
        self.judge_model = judge_model
    
    def run_single_sample(
        self,
        sample_id: str,
        image_dir: str,
        prompt: str,
        ground_truth: Optional[str] = None,
        max_images: Optional[int] = None,
        detail: str = "auto",
        verbose: bool = True
    ) -> PredictionResult:
        """Run benchmark on a single sample (directory of frames)."""
        
        if verbose:
            print(f"\n{'─'*60}")
            print(f"📊 Processing sample: {sample_id}")
        
        # Load images
        image_paths = load_images_from_directory(image_dir, max_images)
        if verbose:
            print(f"   🖼️  Loaded {len(image_paths)} images")
        
        # Run prediction
        if verbose:
            print(f"   🤖 Running prediction with {self.prediction_model}...")
        
        prediction, tokens, proc_time = self.predictor.predict(
            image_paths, prompt, detail
        )
        
        if verbose:
            print(f"   ⏱️  Processing time: {proc_time:.2f}s")
            print(f"   📝 Tokens: {tokens.total_tokens} (${tokens.estimated_cost_usd:.4f})")
        
        # Create result
        result = PredictionResult(
            sample_id=sample_id,
            model=self.prediction_model,
            prompt=prompt,
            prediction=prediction,
            ground_truth=ground_truth,
            processing_time_seconds=proc_time,
            tokens=tokens
        )
        
        # Score with LLM judge if ground truth is available
        if ground_truth:
            if verbose:
                print(f"   ⚖️  Evaluating with LLM judge ({self.judge_model})...")
            
            score, reasoning = self.judge.evaluate(prediction, ground_truth)
            result.accuracy_score = score
            result.judge_reasoning = reasoning
            result.judge_model = self.judge_model
            
            if verbose:
                print(f"   🎯 Accuracy score: {score:.1f}/100")
        
        return result
    
    def run_benchmark(
        self,
        samples: list[dict],
        prompt: str,
        max_images: Optional[int] = None,
        detail: str = "auto",
        verbose: bool = True
    ) -> BenchmarkRun:
        """
        Run benchmark on multiple samples.
        
        Args:
            samples: List of dicts with 'id', 'image_dir', and optionally 'ground_truth'
            prompt: The prediction prompt to use
            max_images: Max images per sample
            detail: Image detail level
            verbose: Print progress
        
        Returns:
            BenchmarkRun with all results
        """
        run_id = f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if verbose:
            print(f"\n{'═'*60}")
            print(f"🚀 Starting Benchmark Run: {run_id}")
            print(f"   Model: {self.prediction_model}")
            print(f"   Judge: {self.judge_model}")
            print(f"   Samples: {len(samples)}")
            print(f"{'═'*60}")
        
        benchmark = BenchmarkRun(
            run_id=run_id,
            model=self.prediction_model,
            prompt=prompt
        )
        
        for sample in samples:
            result = self.run_single_sample(
                sample_id=sample['id'],
                image_dir=sample['image_dir'],
                prompt=prompt,
                ground_truth=sample.get('ground_truth'),
                max_images=max_images,
                detail=detail,
                verbose=verbose
            )
            benchmark.results.append(result)
        
        benchmark.end_time = datetime.now().isoformat()
        
        if verbose:
            self._print_summary(benchmark)
        
        return benchmark
    
    def _print_summary(self, benchmark: BenchmarkRun):
        """Print a summary of benchmark results."""
        print(f"\n{'═'*60}")
        print("📈 BENCHMARK SUMMARY")
        print(f"{'═'*60}")
        print(f"   Run ID: {benchmark.run_id}")
        print(f"   Model: {benchmark.model}")
        print(f"   Total Samples: {benchmark.total_samples}")
        print(f"   Total Tokens: {benchmark.total_tokens:,}")
        print(f"   Total Cost: ${benchmark.total_cost_usd:.4f}")
        print(f"   Avg Processing Time: {benchmark.avg_processing_time:.2f}s")
        
        if benchmark.avg_accuracy is not None:
            print(f"   Avg Accuracy Score: {benchmark.avg_accuracy:.1f}/100")
        
        print(f"{'═'*60}\n")
    
    def save_results(self, benchmark: BenchmarkRun, output_path: str):
        """Save benchmark results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclasses to dicts
        data = {
            "run_id": benchmark.run_id,
            "model": benchmark.model,
            "prompt": benchmark.prompt,
            "start_time": benchmark.start_time,
            "end_time": benchmark.end_time,
            "summary": {
                "total_samples": benchmark.total_samples,
                "total_tokens": benchmark.total_tokens,
                "total_cost_usd": benchmark.total_cost_usd,
                "avg_processing_time_seconds": benchmark.avg_processing_time,
                "avg_accuracy_score": benchmark.avg_accuracy
            },
            "results": [
                {
                    "sample_id": r.sample_id,
                    "model": r.model,
                    "prediction": r.prediction,
                    "ground_truth": r.ground_truth,
                    "processing_time_seconds": r.processing_time_seconds,
                    "tokens": {
                        "prompt_tokens": r.tokens.prompt_tokens,
                        "completion_tokens": r.tokens.completion_tokens,
                        "total_tokens": r.tokens.total_tokens,
                        "estimated_cost_usd": r.tokens.estimated_cost_usd
                    },
                    "accuracy_score": r.accuracy_score,
                    "judge_reasoning": r.judge_reasoning,
                    "judge_model": r.judge_model,
                    "timestamp": r.timestamp
                }
                for r in benchmark.results
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_file}")
        return str(output_file)


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark VLM action prediction models."
    )
    parser.add_argument(
        "image_directories",
        nargs='+',
        help="Directories containing images to benchmark"
    )
    parser.add_argument(
        "-p", "--prompt",
        default="Predict the next 2 plausible actions based on this sequence of images.",
        help="Prediction prompt"
    )
    parser.add_argument(
        "-g", "--ground-truth",
        nargs='+',
        default=None,
        help="Ground truth labels (one per directory, in order)"
    )
    parser.add_argument(
        "-m", "--model",
        default="gpt-4o",
        help="VLM model to use for predictions (default: gpt-4o)"
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="LLM model to use as judge (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "-n", "--max-images",
        type=int,
        default=None,
        help="Maximum images per sample"
    )
    parser.add_argument(
        "-o", "--output",
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots"
    )
    
    args = parser.parse_args()
    
    # Build samples list
    samples = []
    ground_truths = args.ground_truth or []
    
    for i, img_dir in enumerate(args.image_directories):
        sample = {
            'id': Path(img_dir).name,
            'image_dir': img_dir
        }
        if i < len(ground_truths):
            sample['ground_truth'] = ground_truths[i]
        samples.append(sample)
    
    # Run benchmark
    runner = BenchmarkRunner(
        prediction_model=args.model,
        judge_model=args.judge_model
    )
    
    benchmark = runner.run_benchmark(
        samples=samples,
        prompt=args.prompt,
        max_images=args.max_images
    )
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f"{benchmark.run_id}.json"
    runner.save_results(benchmark, str(results_file))
    
    # Generate plots
    if not args.no_plot:
        try:
            from benchmark_visualizer import BenchmarkVisualizer
            visualizer = BenchmarkVisualizer()
            visualizer.plot_single_run(benchmark, output_dir=str(output_dir))
            print(f"📊 Plots saved to: {output_dir}")
        except ImportError:
            print("⚠️  Visualization module not available. Run: pip install matplotlib")


if __name__ == "__main__":
    main()

