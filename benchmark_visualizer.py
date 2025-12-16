#!/usr/bin/env python3
"""
Visualization module for VLM Benchmark Results.

Creates beautiful, informative plots for analyzing benchmark performance.
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set a modern, clean style
plt.style.use('seaborn-v0_8-whitegrid')

# Custom color palette - vibrant yet professional
COLORS = {
    'primary': '#2563eb',      # Blue
    'secondary': '#7c3aed',    # Purple
    'accent': '#06b6d4',       # Cyan
    'success': '#10b981',      # Emerald
    'warning': '#f59e0b',      # Amber
    'danger': '#ef4444',       # Red
    'neutral': '#6b7280',      # Gray
    'background': '#f8fafc',   # Light slate
    'text': '#1e293b',         # Dark slate
}

# Gradient colors for bars
BAR_COLORS = ['#3b82f6', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444']


@dataclass
class BenchmarkData:
    """Parsed benchmark data for visualization."""
    run_id: str
    model: str
    prompt: str
    sample_ids: list[str]
    processing_times: list[float]
    token_counts: list[int]
    costs: list[float]
    accuracy_scores: list[float]
    
    @classmethod
    def from_json(cls, json_path: str) -> 'BenchmarkData':
        """Load benchmark data from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        
        return cls(
            run_id=data.get('run_id', 'unknown'),
            model=data.get('model', 'unknown'),
            prompt=data.get('prompt', ''),
            sample_ids=[r['sample_id'] for r in results],
            processing_times=[r['processing_time_seconds'] for r in results],
            token_counts=[r['tokens']['total_tokens'] for r in results],
            costs=[r['tokens']['estimated_cost_usd'] for r in results],
            accuracy_scores=[r.get('accuracy_score') or 0 for r in results]
        )


class BenchmarkVisualizer:
    """Creates visualizations for benchmark results."""
    
    def __init__(self, figsize: tuple[int, int] = (14, 10)):
        self.figsize = figsize
        self.dpi = 150
    
    def plot_single_run(
        self,
        benchmark_data,
        output_dir: str = "benchmark_results",
        show: bool = False
    ) -> list[str]:
        """
        Generate all plots for a single benchmark run.
        
        Args:
            benchmark_data: Either BenchmarkRun object or path to JSON file
            output_dir: Directory to save plots
            show: Whether to display plots interactively
        
        Returns:
            List of saved plot file paths
        """
        # Handle different input types
        if isinstance(benchmark_data, str):
            data = BenchmarkData.from_json(benchmark_data)
        elif hasattr(benchmark_data, 'results'):
            # It's a BenchmarkRun object
            data = BenchmarkData(
                run_id=benchmark_data.run_id,
                model=benchmark_data.model,
                prompt=benchmark_data.prompt,
                sample_ids=[r.sample_id for r in benchmark_data.results],
                processing_times=[r.processing_time_seconds for r in benchmark_data.results],
                token_counts=[r.tokens.total_tokens for r in benchmark_data.results],
                costs=[r.tokens.estimated_cost_usd for r in benchmark_data.results],
                accuracy_scores=[r.accuracy_score or 0 for r in benchmark_data.results]
            )
        else:
            data = benchmark_data
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        # 1. Dashboard overview
        dashboard_path = self._plot_dashboard(data, output_path)
        saved_files.append(dashboard_path)
        
        # 2. Individual metric plots
        metrics_path = self._plot_metrics_breakdown(data, output_path)
        saved_files.append(metrics_path)
        
        # 3. Cost analysis
        cost_path = self._plot_cost_analysis(data, output_path)
        saved_files.append(cost_path)
        
        if show:
            plt.show()
        
        return saved_files
    
    def _plot_dashboard(self, data: BenchmarkData, output_path: Path) -> str:
        """Create a comprehensive dashboard with all key metrics."""
        fig = plt.figure(figsize=(16, 12), facecolor=COLORS['background'])
        fig.suptitle(
            f'VLM Benchmark Dashboard\n{data.model}',
            fontsize=20,
            fontweight='bold',
            color=COLORS['text'],
            y=0.98
        )
        
        # Create grid for subplots
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3, 
                             left=0.08, right=0.95, top=0.90, bottom=0.08)
        
        # Shorten sample IDs for display
        short_ids = [self._shorten_id(sid) for sid in data.sample_ids]
        x = np.arange(len(short_ids))
        
        # 1. Processing Time (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.bar(x, data.processing_times, color=COLORS['primary'], 
                       alpha=0.85, edgecolor='white', linewidth=1.5)
        ax1.set_ylabel('Time (seconds)', fontsize=11, color=COLORS['text'])
        ax1.set_title('⏱️ Processing Time', fontsize=13, fontweight='bold', 
                     color=COLORS['text'], pad=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(short_ids, rotation=45, ha='right', fontsize=9)
        ax1.axhline(y=np.mean(data.processing_times), color=COLORS['danger'], 
                   linestyle='--', linewidth=2, label=f'Avg: {np.mean(data.processing_times):.2f}s')
        ax1.legend(loc='upper right', fontsize=9)
        self._style_axis(ax1)
        
        # 2. Token Count (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(x, data.token_counts, color=COLORS['secondary'], 
                       alpha=0.85, edgecolor='white', linewidth=1.5)
        ax2.set_ylabel('Tokens', fontsize=11, color=COLORS['text'])
        ax2.set_title('📝 Token Usage', fontsize=13, fontweight='bold', 
                     color=COLORS['text'], pad=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(short_ids, rotation=45, ha='right', fontsize=9)
        ax2.axhline(y=np.mean(data.token_counts), color=COLORS['danger'], 
                   linestyle='--', linewidth=2, label=f'Avg: {np.mean(data.token_counts):.0f}')
        ax2.legend(loc='upper right', fontsize=9)
        self._style_axis(ax2)
        
        # 3. Cost (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        bars3 = ax3.bar(x, [c * 100 for c in data.costs], color=COLORS['warning'], 
                       alpha=0.85, edgecolor='white', linewidth=1.5)
        ax3.set_ylabel('Cost (cents)', fontsize=11, color=COLORS['text'])
        ax3.set_title('💰 API Cost', fontsize=13, fontweight='bold', 
                     color=COLORS['text'], pad=10)
        ax3.set_xticks(x)
        ax3.set_xticklabels(short_ids, rotation=45, ha='right', fontsize=9)
        total_cost = sum(data.costs) * 100
        ax3.axhline(y=np.mean(data.costs) * 100, color=COLORS['danger'], 
                   linestyle='--', linewidth=2, label=f'Total: {total_cost:.2f}¢')
        ax3.legend(loc='upper right', fontsize=9)
        self._style_axis(ax3)
        
        # 4. Accuracy Scores (middle, spanning 2 columns)
        ax4 = fig.add_subplot(gs[1, :2])
        if any(s > 0 for s in data.accuracy_scores):
            colors = [self._score_to_color(s) for s in data.accuracy_scores]
            bars4 = ax4.bar(x, data.accuracy_scores, color=colors, 
                           alpha=0.85, edgecolor='white', linewidth=1.5)
            ax4.axhline(y=np.mean(data.accuracy_scores), color=COLORS['text'], 
                       linestyle='--', linewidth=2, alpha=0.7,
                       label=f'Avg: {np.mean(data.accuracy_scores):.1f}')
            ax4.legend(loc='upper right', fontsize=10)
        else:
            ax4.text(0.5, 0.5, 'No accuracy scores available\n(Ground truth not provided)',
                    ha='center', va='center', fontsize=14, color=COLORS['neutral'],
                    transform=ax4.transAxes)
        ax4.set_ylabel('Score (0-100)', fontsize=11, color=COLORS['text'])
        ax4.set_title('🎯 Accuracy Scores (LLM Judge)', fontsize=13, fontweight='bold', 
                     color=COLORS['text'], pad=10)
        ax4.set_xticks(x)
        ax4.set_xticklabels(short_ids, rotation=45, ha='right', fontsize=9)
        ax4.set_ylim(0, 105)
        self._style_axis(ax4)
        
        # 5. Score distribution (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        if any(s > 0 for s in data.accuracy_scores):
            bins = [0, 20, 40, 60, 80, 100]
            counts, _ = np.histogram(data.accuracy_scores, bins=bins)
            labels = ['0-19\nWrong', '20-39\nPoor', '40-59\nPartial', 
                     '60-79\nGood', '80-100\nExcellent']
            bar_colors = [COLORS['danger'], COLORS['warning'], '#fbbf24', 
                         COLORS['accent'], COLORS['success']]
            ax5.bar(range(len(counts)), counts, color=bar_colors, 
                   alpha=0.85, edgecolor='white', linewidth=1.5)
            ax5.set_xticks(range(len(labels)))
            ax5.set_xticklabels(labels, fontsize=8)
        else:
            ax5.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                    fontsize=20, color=COLORS['neutral'], transform=ax5.transAxes)
        ax5.set_ylabel('Count', fontsize=11, color=COLORS['text'])
        ax5.set_title('📊 Score Distribution', fontsize=13, fontweight='bold', 
                     color=COLORS['text'], pad=10)
        self._style_axis(ax5)
        
        # 6. Summary statistics (bottom, spanning all columns)
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        # Create summary cards
        summary_data = [
            ('Total Samples', f'{len(data.sample_ids)}', COLORS['primary']),
            ('Avg Time', f'{np.mean(data.processing_times):.2f}s', COLORS['secondary']),
            ('Total Tokens', f'{sum(data.token_counts):,}', COLORS['accent']),
            ('Total Cost', f'${sum(data.costs):.4f}', COLORS['warning']),
            ('Avg Accuracy', f'{np.mean(data.accuracy_scores):.1f}/100' if any(s > 0 for s in data.accuracy_scores) else 'N/A', COLORS['success']),
        ]
        
        card_width = 0.18
        start_x = 0.05
        for i, (label, value, color) in enumerate(summary_data):
            x_pos = start_x + i * (card_width + 0.02)
            # Card background
            rect = mpatches.FancyBboxPatch(
                (x_pos, 0.2), card_width, 0.6,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                facecolor=color, alpha=0.15, edgecolor=color, linewidth=2,
                transform=ax6.transAxes
            )
            ax6.add_patch(rect)
            # Label
            ax6.text(x_pos + card_width/2, 0.7, label,
                    ha='center', va='center', fontsize=11, 
                    color=COLORS['neutral'], transform=ax6.transAxes)
            # Value
            ax6.text(x_pos + card_width/2, 0.4, value,
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    color=color, transform=ax6.transAxes)
        
        # Save
        filepath = output_path / f'{data.run_id}_dashboard.png'
        plt.savefig(filepath, dpi=self.dpi, facecolor=COLORS['background'], 
                   edgecolor='none', bbox_inches='tight')
        plt.close(fig)
        
        print(f"   📊 Saved dashboard: {filepath}")
        return str(filepath)
    
    def _plot_metrics_breakdown(self, data: BenchmarkData, output_path: Path) -> str:
        """Create detailed metrics comparison plot."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=COLORS['background'])
        fig.suptitle(f'Metrics Breakdown - {data.model}', fontsize=16, 
                    fontweight='bold', color=COLORS['text'], y=0.98)
        
        short_ids = [self._shorten_id(sid) for sid in data.sample_ids]
        
        # 1. Processing time with percentiles
        ax1 = axes[0, 0]
        times = data.processing_times
        ax1.boxplot([times], positions=[0], widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor=COLORS['primary'], alpha=0.6),
                   medianprops=dict(color=COLORS['danger'], linewidth=2))
        ax1.scatter(np.zeros(len(times)), times, c=COLORS['primary'], 
                   alpha=0.7, s=100, zorder=5, edgecolor='white', linewidth=2)
        ax1.set_ylabel('Seconds', fontsize=11)
        ax1.set_title('Processing Time Distribution', fontsize=12, fontweight='bold')
        ax1.set_xticks([])
        
        # Add stats
        stats_text = f'Min: {min(times):.2f}s\nMax: {max(times):.2f}s\nMean: {np.mean(times):.2f}s\nStd: {np.std(times):.2f}s'
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        self._style_axis(ax1)
        
        # 2. Token breakdown (could be expanded with prompt vs completion)
        ax2 = axes[0, 1]
        tokens = data.token_counts
        ax2.hist(tokens, bins=max(5, len(tokens)//2), color=COLORS['secondary'], 
                alpha=0.7, edgecolor='white', linewidth=1.5)
        ax2.axvline(x=np.mean(tokens), color=COLORS['danger'], linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(tokens):.0f}')
        ax2.set_xlabel('Total Tokens', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Token Usage Distribution', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')
        self._style_axis(ax2)
        
        # 3. Time vs Tokens correlation
        ax3 = axes[1, 0]
        ax3.scatter(tokens, times, c=COLORS['accent'], s=120, alpha=0.7,
                   edgecolor='white', linewidth=2)
        if len(tokens) > 1:
            z = np.polyfit(tokens, times, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(tokens), max(tokens), 100)
            ax3.plot(x_line, p(x_line), '--', color=COLORS['danger'], 
                    linewidth=2, label='Trend')
            ax3.legend(loc='upper left')
        ax3.set_xlabel('Total Tokens', fontsize=11)
        ax3.set_ylabel('Processing Time (s)', fontsize=11)
        ax3.set_title('Tokens vs Processing Time', fontsize=12, fontweight='bold')
        self._style_axis(ax3)
        
        # 4. Cost per sample
        ax4 = axes[1, 1]
        costs_cents = [c * 100 for c in data.costs]
        cumulative = np.cumsum(costs_cents)
        ax4.fill_between(range(len(cumulative)), cumulative, 
                        color=COLORS['warning'], alpha=0.3)
        ax4.plot(cumulative, color=COLORS['warning'], linewidth=3, marker='o',
                markersize=8, markerfacecolor='white', markeredgewidth=2)
        ax4.set_xlabel('Sample Index', fontsize=11)
        ax4.set_ylabel('Cumulative Cost (cents)', fontsize=11)
        ax4.set_title('Cumulative Cost Over Samples', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(len(short_ids)))
        ax4.set_xticklabels(short_ids, rotation=45, ha='right', fontsize=9)
        self._style_axis(ax4)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        filepath = output_path / f'{data.run_id}_metrics.png'
        plt.savefig(filepath, dpi=self.dpi, facecolor=COLORS['background'],
                   edgecolor='none', bbox_inches='tight')
        plt.close(fig)
        
        print(f"   📈 Saved metrics breakdown: {filepath}")
        return str(filepath)
    
    def _plot_cost_analysis(self, data: BenchmarkData, output_path: Path) -> str:
        """Create cost analysis visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=COLORS['background'])
        fig.suptitle(f'Cost Analysis - {data.model}', fontsize=16, 
                    fontweight='bold', color=COLORS['text'], y=1.02)
        
        # 1. Cost per sample
        ax1 = axes[0]
        short_ids = [self._shorten_id(sid) for sid in data.sample_ids]
        costs_cents = [c * 100 for c in data.costs]
        
        bars = ax1.barh(range(len(short_ids)), costs_cents, color=COLORS['warning'],
                       alpha=0.85, edgecolor='white', linewidth=1.5)
        ax1.set_yticks(range(len(short_ids)))
        ax1.set_yticklabels(short_ids, fontsize=10)
        ax1.set_xlabel('Cost (cents)', fontsize=11)
        ax1.set_title('Cost per Sample', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        
        # Add value labels
        for i, (bar, cost) in enumerate(zip(bars, costs_cents)):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{cost:.3f}¢', va='center', fontsize=9, color=COLORS['text'])
        
        self._style_axis(ax1)
        
        # 2. Cost breakdown pie (if we had prompt vs completion breakdown)
        ax2 = axes[1]
        total_cost = sum(data.costs) * 100
        avg_cost = np.mean(costs_cents)
        
        # Create a donut chart with total cost
        sizes = costs_cents if len(costs_cents) <= 8 else costs_cents[:8]
        labels = short_ids if len(short_ids) <= 8 else short_ids[:8]
        
        colors_pie = plt.cm.Blues(np.linspace(0.3, 0.9, len(sizes)))
        wedges, texts, autotexts = ax2.pie(
            sizes, labels=None, autopct='%1.1f%%',
            colors=colors_pie, startangle=90,
            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
            pctdistance=0.75
        )
        
        # Add center text
        ax2.text(0, 0, f'Total\n{total_cost:.2f}¢', 
                ha='center', va='center', fontsize=14, fontweight='bold',
                color=COLORS['text'])
        
        ax2.set_title('Cost Distribution', fontsize=12, fontweight='bold')
        ax2.legend(wedges, labels, title="Samples", loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
        
        plt.tight_layout()
        
        filepath = output_path / f'{data.run_id}_cost.png'
        plt.savefig(filepath, dpi=self.dpi, facecolor=COLORS['background'],
                   edgecolor='none', bbox_inches='tight')
        plt.close(fig)
        
        print(f"   💰 Saved cost analysis: {filepath}")
        return str(filepath)
    
    def compare_runs(
        self,
        benchmark_files: list[str],
        output_dir: str = "benchmark_results",
        show: bool = False
    ) -> str:
        """Compare multiple benchmark runs in a single plot."""
        datasets = [BenchmarkData.from_json(f) for f in benchmark_files]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=COLORS['background'])
        fig.suptitle('Multi-Run Comparison', fontsize=18, fontweight='bold',
                    color=COLORS['text'], y=0.98)
        
        models = [d.model for d in datasets]
        x = np.arange(len(models))
        width = 0.6
        
        # 1. Average processing time
        ax1 = axes[0, 0]
        avg_times = [np.mean(d.processing_times) for d in datasets]
        bars = ax1.bar(x, avg_times, width, color=BAR_COLORS[:len(models)], 
                      alpha=0.85, edgecolor='white', linewidth=2)
        ax1.set_ylabel('Avg Time (s)', fontsize=11)
        ax1.set_title('⏱️ Average Processing Time', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=15, ha='right')
        self._style_axis(ax1)
        
        # 2. Average tokens
        ax2 = axes[0, 1]
        avg_tokens = [np.mean(d.token_counts) for d in datasets]
        bars = ax2.bar(x, avg_tokens, width, color=BAR_COLORS[:len(models)],
                      alpha=0.85, edgecolor='white', linewidth=2)
        ax2.set_ylabel('Avg Tokens', fontsize=11)
        ax2.set_title('📝 Average Token Usage', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=15, ha='right')
        self._style_axis(ax2)
        
        # 3. Total cost
        ax3 = axes[1, 0]
        total_costs = [sum(d.costs) * 100 for d in datasets]
        bars = ax3.bar(x, total_costs, width, color=BAR_COLORS[:len(models)],
                      alpha=0.85, edgecolor='white', linewidth=2)
        ax3.set_ylabel('Total Cost (cents)', fontsize=11)
        ax3.set_title('💰 Total Cost', fontsize=13, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=15, ha='right')
        self._style_axis(ax3)
        
        # 4. Average accuracy
        ax4 = axes[1, 1]
        avg_acc = [np.mean(d.accuracy_scores) if any(s > 0 for s in d.accuracy_scores) 
                   else 0 for d in datasets]
        colors = [self._score_to_color(a) for a in avg_acc]
        bars = ax4.bar(x, avg_acc, width, color=colors,
                      alpha=0.85, edgecolor='white', linewidth=2)
        ax4.set_ylabel('Avg Accuracy', fontsize=11)
        ax4.set_title('🎯 Average Accuracy Score', fontsize=13, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models, rotation=15, ha='right')
        ax4.set_ylim(0, 105)
        self._style_axis(ax4)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / 'comparison.png'
        plt.savefig(filepath, dpi=self.dpi, facecolor=COLORS['background'],
                   edgecolor='none', bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        print(f"📊 Saved comparison plot: {filepath}")
        return str(filepath)
    
    def _shorten_id(self, sample_id: str, max_len: int = 20) -> str:
        """Shorten sample ID for display."""
        if len(sample_id) <= max_len:
            return sample_id
        return sample_id[:max_len-3] + '...'
    
    def _score_to_color(self, score: float) -> str:
        """Convert accuracy score to appropriate color."""
        if score >= 80:
            return COLORS['success']
        elif score >= 60:
            return COLORS['accent']
        elif score >= 40:
            return '#fbbf24'  # Light amber
        elif score >= 20:
            return COLORS['warning']
        else:
            return COLORS['danger']
    
    def _style_axis(self, ax):
        """Apply consistent styling to an axis."""
        ax.set_facecolor('white')
        ax.tick_params(colors=COLORS['text'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(COLORS['neutral'])
        ax.spines['bottom'].set_color(COLORS['neutral'])
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize VLM benchmark results."
    )
    parser.add_argument(
        "json_files",
        nargs='+',
        help="Benchmark result JSON files to visualize"
    )
    parser.add_argument(
        "-o", "--output",
        default="benchmark_results",
        help="Output directory for plots (default: benchmark_results)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate comparison plot for multiple runs"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively"
    )
    
    args = parser.parse_args()
    
    visualizer = BenchmarkVisualizer()
    
    if args.compare and len(args.json_files) > 1:
        visualizer.compare_runs(args.json_files, args.output, args.show)
    else:
        for json_file in args.json_files:
            visualizer.plot_single_run(json_file, args.output, args.show)


if __name__ == "__main__":
    main()

