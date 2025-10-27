"""
Visualization tools for strategy optimization results.

Provides heatmap generation, table formatting, and CSV export.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import csv

from src.strategy_optimizer import OptimizationResult, OptimizationSummary

logger = logging.getLogger(__name__)


def export_to_csv(results: List[OptimizationResult], output_path: str):
    """
    Export optimization results to CSV file.

    Args:
        results: List of optimization results
        output_path: Path to output CSV file
    """
    if not results:
        logger.warning("No results to export")
        return

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Get all fields from first result
    fieldnames = list(results[0].to_dict().keys())

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(result.to_dict())

    logger.info(f"Exported {len(results)} results to {output_path}")


def print_top_strategies_table(
    summary: OptimizationSummary,
    metric: str,
    metric_name: str,
    n: int = 10
):
    """
    Print a formatted table of top N strategies by a given metric.

    Args:
        summary: Optimization summary
        metric: Attribute name to sort by
        metric_name: Display name for the metric
        n: Number of strategies to show
    """
    # Sort by metric
    sorted_results = sorted(
        summary.strategy_results,
        key=lambda r: getattr(r, metric),
        reverse=True
    )[:n]

    print(f"\n{'='*100}")
    print(f"TOP {n} STRATEGIES BY {metric_name.upper()}")
    print(f"{'='*100}")

    # Table header
    print(f"{'Rank':<6} {'Threshold':<12} {'Interval':<12} {metric_name:<18} {'Sharpe':<10} {'Fees':<12} {'Rebal.':<8}")
    print(f"{'':<6} {'(%)':<12} {'(min)':<12} {'':<18} {'Ratio':<10} {'($)':<12} {'Count':<8}")
    print(f"{'-'*100}")

    # Table rows
    for i, result in enumerate(sorted_results, 1):
        threshold = f"{result.threshold_percent:.1f}%"
        interval = f"{result.interval_minutes:.0f}min"

        # Format metric value
        metric_val = getattr(result, metric)
        if 'percent' in metric or 'pct' in metric:
            metric_str = f"{metric_val:+.2f}%"
        elif metric == 'sharpe_ratio':
            metric_str = f"{metric_val:.3f}"
        else:
            metric_str = f"{metric_val:.2f}"

        sharpe_str = f"{result.sharpe_ratio:.3f}"
        fees_str = f"${result.total_fees_paid:.2f}"
        rebal_str = f"{result.num_rebalances}"

        print(f"{i:<6} {threshold:<12} {interval:<12} {metric_str:<18} {sharpe_str:<10} {fees_str:<12} {rebal_str:<8}")

    print(f"{'='*100}\n")


def print_optimization_summary(summary: OptimizationSummary):
    """Print overall optimization summary."""
    print("\n" + "="*100)
    print("OPTIMIZATION SUMMARY")
    print("="*100)

    print(f"\nTotal Combinations Tested: {summary.total_combinations}")
    print(f"Execution Time: {summary.execution_time_seconds:.1f} seconds")
    print(f"Average Time per Simulation: {summary.execution_time_seconds / summary.total_combinations:.3f} seconds")

    print(f"\nBuy-and-Hold Baseline:")
    print(f"  Total Return: {summary.baseline_result.total_return_percent:+.2f}%")
    print(f"  Sharpe Ratio: {summary.baseline_result.sharpe_ratio:.3f}")
    print(f"  Max Drawdown: {summary.baseline_result.max_drawdown_percent:.2f}%")

    print(f"\nBest Strategies:")
    print(f"  Highest Return: {summary.best_return.threshold_percent:.1f}% threshold, "
          f"{summary.best_return.interval_minutes:.0f}min interval "
          f"({summary.best_return.total_return_percent:+.2f}%)")

    print(f"  Best Sharpe Ratio: {summary.best_sharpe.threshold_percent:.1f}% threshold, "
          f"{summary.best_sharpe.interval_minutes:.0f}min interval "
          f"(Sharpe: {summary.best_sharpe.sharpe_ratio:.3f})")

    print(f"  Best Net Return: {summary.best_net_return.threshold_percent:.1f}% threshold, "
          f"{summary.best_net_return.interval_minutes:.0f}min interval "
          f"({summary.best_net_return.net_return_percent:+.2f}%)")

    print(f"  Lowest Fees: {summary.lowest_fees.threshold_percent:.1f}% threshold, "
          f"{summary.lowest_fees.interval_minutes:.0f}min interval "
          f"(${summary.lowest_fees.total_fees_paid:.2f})")

    print(f"  Lowest Drawdown: {summary.lowest_drawdown.threshold_percent:.1f}% threshold, "
          f"{summary.lowest_drawdown.interval_minutes:.0f}min interval "
          f"({summary.lowest_drawdown.max_drawdown_percent:.2f}%)")

    print("="*100 + "\n")


def generate_heatmaps(
    summary: OptimizationSummary,
    output_dir: str,
    metrics: Optional[List[str]] = None
):
    """
    Generate heatmap visualizations for optimization results.

    Args:
        summary: Optimization summary with results
        output_dir: Directory to save heatmap images
        metrics: List of metrics to visualize (default: all main metrics)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.error("matplotlib is required for heatmap generation")
        logger.error("Install with: pip install matplotlib")
        return

    if metrics is None:
        metrics = [
            ('total_return_percent', 'Total Return (%)', 'RdYlGn'),
            ('sharpe_ratio', 'Sharpe Ratio', 'RdYlGn'),
            ('net_return_percent', 'Net Return After Fees (%)', 'RdYlGn'),
            ('total_fees_paid', 'Total Fees ($)', 'YlOrRd_r'),
            ('max_drawdown_percent', 'Max Drawdown (%)', 'YlOrRd_r'),
        ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract unique thresholds and intervals
    thresholds = sorted(set(r.threshold_percent for r in summary.strategy_results))
    intervals = sorted(set(r.interval_minutes for r in summary.strategy_results))

    logger.info(f"Generating heatmaps with {len(thresholds)} thresholds × {len(intervals)} intervals")

    for metric_attr, metric_name, colormap in metrics:
        # Create data matrix
        data = np.zeros((len(thresholds), len(intervals)))

        for result in summary.strategy_results:
            i = thresholds.index(result.threshold_percent)
            j = intervals.index(result.interval_minutes)
            data[i, j] = getattr(result, metric_attr)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))

        im = ax.imshow(data, cmap=colormap, aspect='auto')

        # Set ticks and labels
        ax.set_xticks(np.arange(len(intervals)))
        ax.set_yticks(np.arange(len(thresholds)))

        ax.set_xticklabels([f"{int(i)}min" for i in intervals], rotation=45, ha='right')
        ax.set_yticklabels([f"{t:.1f}%" for t in thresholds])

        ax.set_xlabel('Rebalance Interval', fontsize=12)
        ax.set_ylabel('Threshold (%)', fontsize=12)
        ax.set_title(f'Strategy Optimization: {metric_name}', fontsize=14, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric_name, rotation=270, labelpad=20)

        # Add value annotations for smaller grids
        if len(thresholds) * len(intervals) <= 100:  # Only annotate if grid isn't too large
            for i in range(len(thresholds)):
                for j in range(len(intervals)):
                    value = data[i, j]
                    if 'percent' in metric_attr or 'pct' in metric_attr:
                        text = f"{value:.1f}"
                    elif metric_attr == 'sharpe_ratio':
                        text = f"{value:.2f}"
                    else:
                        text = f"{value:.0f}"

                    ax.text(j, i, text,
                           ha="center", va="center",
                           color="black" if abs(value) < data.max()/2 else "white",
                           fontsize=8)

        plt.tight_layout()

        # Save figure
        filename = f"{metric_attr}_heatmap.png"
        filepath = output_path / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved heatmap: {filepath}")

    logger.info(f"Generated {len(metrics)} heatmaps in {output_dir}/")


def print_comparison_vs_baseline(summary: OptimizationSummary, n: int = 5):
    """Print top strategies that beat buy-and-hold baseline."""
    # Filter strategies that beat baseline
    beating_baseline = [
        r for r in summary.strategy_results
        if r.return_vs_baseline > 0
    ]

    print("\n" + "="*100)
    print(f"STRATEGIES THAT OUTPERFORM BUY-AND-HOLD (Total: {len(beating_baseline)})")
    print("="*100)

    if not beating_baseline:
        print("\nNo rebalancing strategies outperformed buy-and-hold for this period.")
        print("This suggests the market had strong directional trends where holding was optimal.\n")
        return

    # Sort by return vs baseline
    top_beaters = sorted(beating_baseline, key=lambda r: r.return_vs_baseline, reverse=True)[:n]

    print(f"\n{'Rank':<6} {'Threshold':<12} {'Interval':<12} {'Return':<15} {'vs Baseline':<15} {'Sharpe':<10} {'Fees':<10}")
    print(f"{'':<6} {'(%)':<12} {'(min)':<12} {'(%)':<15} {'(pp)':<15} {'Ratio':<10} {'($)':<10}")
    print(f"{'-'*100}")

    for i, result in enumerate(top_beaters, 1):
        threshold = f"{result.threshold_percent:.1f}%"
        interval = f"{result.interval_minutes:.0f}min"
        ret = f"{result.total_return_percent:+.2f}%"
        vs_base = f"{result.return_vs_baseline:+.2f}pp"
        sharpe = f"{result.sharpe_ratio:.3f}"
        fees = f"${result.total_fees_paid:.2f}"

        print(f"{i:<6} {threshold:<12} {interval:<12} {ret:<15} {vs_base:<15} {sharpe:<10} {fees:<10}")

    print("="*100 + "\n")
