import time
import json
import csv
from pathlib import Path
from analyze_frames import load_images_from_directory, analyze_frames_with_openai
import matplotlib.pyplot as plt


# Estimate cost --> TODO: API usage details would be needed for accurate calculation
PRICES = {
    "qwen3-vl": 0.001,
    "": 0.002, 
}

# Estimate cost --> TODO: API usage details would be needed for accurate calculation
def estimate_cost(model, prompt_tokens, completion_tokens):
    price_per_1k = PRICES.get(model)
    total_tokens = prompt_tokens + completion_tokens
    cost = (total_tokens / 1000) * price_per_1k
    return cost

# TODO: Maybe better evaluation function --> with llm?
def evaluate_accuracy(response, expected_label):
    response_lower = response.lower()
    expected_lower = expected_label.lower()

    return 1.0 if expected_lower in response_lower else 0.0


def run_benchmark(
    directory,
    model,
    prompt,
    num_images_list,
    label=None,
    output_csv="benchmark_results.csv"
):

    results = []

    for N in num_images_list:
        print(f"\n=== Running benchmark with {N} images ===")

        images = load_images_from_directory(directory, max_images=N)

        start_time = time.time()
        response = analyze_frames_with_openai(images, prompt, model=model, api_key=None)
        runtime = time.time() - start_time

        # Estimate cost --> TODO: API usage details would be needed for accurate calculation
        prompt_tokens = len(prompt) / 4  # rough estimation
        response_tokens = len(response) / 4

        cost = estimate_cost(model, prompt_tokens, response_tokens)

        accuracy = evaluate_accuracy(response, label) if label else None

        print(f"Runtime: {runtime:.2f}s | Cost: {cost:.4f}$ | Accuracy: {accuracy}")

        results.append({
            "num_images": N,
            "runtime_seconds": runtime,
            "estimated_cost_usd": cost,
            "accuracy": accuracy,
        })

    # Save results to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nBenchmark saved to: {output_csv}")


def plot_benchmark(csv_file="benchmark_results.csv"):
    import pandas as pd
    df = pd.read_csv(csv_file)

    # Runtime vs Num Images
    plt.figure(figsize=(8,5))
    plt.plot(df['num_images'], df['runtime_seconds'], marker='o')
    plt.title("Runtime vs Number of Images")
    plt.xlabel("Number of Images")
    plt.ylabel("Runtime (s)")
    plt.grid(True)
    plt.show()

    # Cost vs Num Images
    plt.figure(figsize=(8,5))
    plt.plot(df['num_images'], df['estimated_cost_usd'], marker='o', color='orange')
    plt.title("Estimated Cost vs Number of Images")
    plt.xlabel("Number of Images")
    plt.ylabel("Estimated Cost ($)")
    plt.grid(True)
    plt.show()

    # Accuracy vs Num Images
    if 'accuracy' in df.columns and df['accuracy'].notnull().all():
        plt.figure(figsize=(8,5))
        plt.plot(df['num_images'], df['accuracy'], marker='o', color='green')
        plt.title("Accuracy vs Number of Images")
        plt.xlabel("Number of Images")
        plt.ylabel("Accuracy")
        plt.ylim(0,1.1)
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # TODO: Adjust parameters as needed
    run_benchmark(
        directory="",
        model="qwen3-vl",
        prompt="What is the next Action of the person in the images?",
        num_images_list=[1, 3, 5],
        label="cutting onions",             # TODO: set expected label for accuracy
        output_csv="benchmark_results.csv"
    )
