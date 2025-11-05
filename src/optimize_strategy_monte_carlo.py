#!/usr/bin/env python3
"""
Optimize rebalancing parameters using Monte Carlo simulations.

Unlike optimize_strategy.py which tests on actual historical data,
this tests each parameter combination across many shuffled timelines
to find the most ROBUST parameters across different market sequences.

Usage:
    python -m src.optimize_strategy_monte_carlo \
        --days 365 \
        --threshold-min 1.0 --threshold-max 5.0 --threshold-step 1.0 \
        --interval-min 1440 --interval-max 10080 --interval-step 1440 \
        --simulations 1000 \
        --chunk-days 30 \
        --output results.csv
"""

import argparse
import logging
import json
import sys
import warnings
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np

# Suppress urllib3 OpenSSL warnings
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')

from src.coinbase_client import CoinbaseClient
from src.monte_carlo_simulator import (
    SimulationConfig,
    HistoricalPriceFetcher,
    HybridStrategy
)
from src.run_monte_carlo import run_monte_carlo

logger = logging.getLogger(__name__)


def setup_logging(quiet: bool = False):
    """Configure logging."""
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str) -> dict:
    """Load portfolio configuration."""
    config_file = Path(config_path)

    if not config_file.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    with open(config_file, 'r') as f:
        config = json.load(f)

    return config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Optimize rebalancing parameters using Monte Carlo simulations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test 5 thresholds × 7 intervals × 1000 simulations = 35,000 total runs
  python -m src.optimize_strategy_monte_carlo \\
      --days 365 \\
      --threshold-min 1.0 --threshold-max 5.0 --threshold-step 1.0 \\
      --interval-min 1440 --interval-max 10080 --interval-step 1440 \\
      --simulations 1000 \\
      --chunk-days 30 \\
      --output results.csv

  # Quick test: 3 thresholds × 3 intervals × 100 simulations = 900 runs
  python -m src.optimize_strategy_monte_carlo \\
      --days 180 \\
      --threshold-min 2.0 --threshold-max 4.0 --threshold-step 1.0 \\
      --interval-min 2880 --interval-max 5760 --interval-step 1440 \\
      --simulations 100 \\
      --chunk-days 30
        """
    )

    # Time period options
    time_group = parser.add_mutually_exclusive_group(required=True)
    time_group.add_argument(
        '--days',
        type=int,
        help='Number of days of historical data to use'
    )
    time_group.add_argument(
        '--start',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end',
        type=str,
        help='End date (YYYY-MM-DD), defaults to today'
    )

    # Threshold range parameters
    parser.add_argument(
        '--threshold-min',
        type=float,
        required=True,
        help='Minimum threshold to test (%%) (e.g., 1.0)'
    )
    parser.add_argument(
        '--threshold-max',
        type=float,
        required=True,
        help='Maximum threshold to test (%%) (e.g., 5.0)'
    )
    parser.add_argument(
        '--threshold-step',
        type=float,
        default=0.5,
        help='Step size for threshold (%%) (default: 0.5)'
    )

    # Interval range parameters (minutes)
    parser.add_argument(
        '--interval-min',
        type=float,
        required=True,
        help='Minimum interval to test (minutes) (e.g., 1440 = 1 day)'
    )
    parser.add_argument(
        '--interval-max',
        type=float,
        required=True,
        help='Maximum interval to test (minutes) (e.g., 10080 = 7 days)'
    )
    parser.add_argument(
        '--interval-step',
        type=float,
        default=1440,
        help='Step size for interval (minutes) (default: 1440 = 1 day)'
    )

    # Monte Carlo parameters
    parser.add_argument(
        '--simulations',
        type=int,
        default=1000,
        help='Number of Monte Carlo simulations per parameter combination (default: 1000)'
    )
    parser.add_argument(
        '--chunk-days',
        type=int,
        default=30,
        help='Size of chunks to shuffle in days (default: 30)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )

    # Simulation parameters
    parser.add_argument(
        '--fee-rate',
        type=float,
        default=0.006,
        help='Trading fee rate as decimal (default: 0.006 = 0.6%%)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/portfolio.json',
        help='Path to portfolio configuration file (default: config/portfolio.json)'
    )
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=10000.0,
        help='Initial portfolio value in USD (default: 10000)'
    )

    # Cache options
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Skip cache and force fresh API fetch'
    )
    parser.add_argument(
        '--cache-only',
        action='store_true',
        help='Only use cached data, fail if cache not available'
    )

    # Performance options
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count)'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar'
    )

    # Output options
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to CSV file (optional)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top parameter combinations to display (default: 10)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed logging'
    )

    return parser.parse_args()


def main():
    """Run Monte Carlo parameter optimization."""
    args = parse_arguments()

    setup_logging(args.quiet)

    # Load portfolio configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    target_allocation = config.get('target_allocation', {})
    if not target_allocation:
        logger.error("No target_allocation found in config file")
        sys.exit(1)

    # Validate allocation sums to 100%
    total_allocation = sum(target_allocation.values())
    if not 99.9 <= total_allocation <= 100.1:
        logger.error(f"Target allocation must sum to 100%, got {total_allocation}%")
        sys.exit(1)

    # Parse dates
    if args.days:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
    else:
        start_date = datetime.fromisoformat(args.start)
        if args.end:
            end_date = datetime.fromisoformat(args.end)
        else:
            end_date = datetime.now()

    # Calculate parameter grid
    thresholds = np.arange(args.threshold_min, args.threshold_max + args.threshold_step/2, args.threshold_step)
    intervals_min = np.arange(args.interval_min, args.interval_max + args.interval_step/2, args.interval_step)

    print("\n" + "="*100)
    print("MONTE CARLO PARAMETER OPTIMIZATION")
    print("="*100)
    print(f"\nSimulation Period: {start_date.date()} to {end_date.date()}")
    print(f"Target Allocation: {target_allocation}")
    print(f"\nParameter Ranges:")
    print(f"  Thresholds: {args.threshold_min}% to {args.threshold_max}% (step: {args.threshold_step}%)")
    print(f"    → Testing {len(thresholds)} threshold values: {[f'{t:.1f}%' for t in thresholds]}")
    print(f"  Intervals: {args.interval_min/1440:.1f} to {args.interval_max/1440:.1f} days (step: {args.interval_step/1440:.1f} days)")
    print(f"    → Testing {len(intervals_min)} interval values: {[f'{int(i/1440)}d' for i in intervals_min]}")
    print(f"\nMonte Carlo Settings:")
    print(f"  Simulations per combination: {args.simulations:,}")
    print(f"  Chunk size: {args.chunk_days} days")
    print(f"  Random seed: {args.seed if args.seed else 'auto-generated'}")
    print(f"\nExecution:")
    total_combos = len(thresholds) * len(intervals_min)
    total_sims = total_combos * args.simulations
    print(f"  Parameter combinations: {total_combos} ({len(thresholds)} × {len(intervals_min)})")
    print(f"  Total simulations: {total_sims:,}")
    print(f"  Parallel workers: {args.workers or 'auto (CPU count)'}")
    print("="*100 + "\n")

    # Initialize Coinbase client
    logger.info("Initializing Coinbase client...")
    try:
        client = CoinbaseClient()
    except Exception as e:
        logger.error(f"Failed to initialize Coinbase client: {e}")
        sys.exit(1)

    # Fetch historical price data (once, shared across all)
    logger.info(f"Fetching historical price data for {len(target_allocation)} assets...")
    fetcher = HistoricalPriceFetcher(
        client,
        use_cache=not args.no_cache,
        cache_max_age_days=7 if not args.no_cache else None
    )

    try:
        price_data = fetcher.fetch_historical_prices(
            assets=list(target_allocation.keys()),
            start_date=start_date,
            end_date=end_date,
            granularity='ONE_DAY',
            cache_only=args.cache_only
        )
    except Exception as e:
        logger.error(f"Failed to fetch price data: {e}")
        sys.exit(1)

    # Validate we got data
    for asset, prices in price_data.items():
        if not prices:
            logger.error(f"No price data retrieved for {asset}")
            sys.exit(1)
        logger.info(f"{asset}: {len(prices)} data points")

    print("\n" + "="*100)
    print("RUNNING MONTE CARLO OPTIMIZATIONS")
    print("="*100 + "\n")

    # Run Monte Carlo for each parameter combination
    results = []
    start_time_total = time.time()

    for combo_idx, (threshold, interval_min) in enumerate(
        [(t, i) for t in thresholds for i in intervals_min], 1
    ):
        interval_days = interval_min / 1440

        print(f"\n[{combo_idx}/{total_combos}] Testing: Threshold={threshold:.1f}%, Interval={interval_days:.1f} days")
        print(f"Running {args.simulations:,} Monte Carlo simulations...")

        # Create strategy for this parameter combination
        strategy = HybridStrategy(
            rebalance_interval_days=interval_days,
            threshold_percent=threshold
        )

        # Run Monte Carlo simulations
        try:
            mc_result, original_result = run_monte_carlo(
                price_data=price_data,
                strategy=strategy,
                target_allocation=target_allocation,
                initial_capital=args.initial_capital,
                num_simulations=args.simulations,
                chunk_days=args.chunk_days,
                fee_rate=args.fee_rate,
                seed=args.seed,
                max_workers=args.workers,
                show_progress=not args.no_progress
            )

            # Store results
            results.append({
                'threshold_percent': threshold,
                'interval_days': interval_days,
                'interval_minutes': interval_min,
                # Monte Carlo statistics
                'median_return': mc_result.median_return,
                'mean_return': mc_result.mean_return,
                'std_return': mc_result.std_return,
                'percentile_5_return': mc_result.percentile_5_return,
                'percentile_95_return': mc_result.percentile_95_return,
                'median_sharpe': mc_result.median_sharpe,
                'mean_sharpe': mc_result.mean_sharpe,
                'median_max_drawdown': mc_result.median_max_drawdown,
                'mean_max_drawdown': mc_result.mean_max_drawdown,
                'worst_max_drawdown': mc_result.worst_max_drawdown,
                'mean_num_rebalances': mc_result.mean_num_rebalances,
                'mean_total_fees': mc_result.mean_total_fees,
                # Original (unshuffled) for reference
                'original_return': original_result.total_return_percent,
                'original_sharpe': original_result.sharpe_ratio,
                'original_max_drawdown': original_result.max_drawdown_percent,
                'num_simulations': args.simulations
            })

            print(f"✓ Complete - Median Return: {mc_result.median_return:.2f}%, Median Sharpe: {mc_result.median_sharpe:.3f}")

        except Exception as e:
            logger.error(f"Error running Monte Carlo for threshold={threshold}%, interval={interval_days}d: {e}")
            continue

    elapsed_total = time.time() - start_time_total
    print(f"\n✅ All {total_combos} parameter combinations tested in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")

    # Print results
    print("\n" + "="*100)
    print("OPTIMIZATION RESULTS")
    print("="*100)

    # Sort by median return
    results_sorted = sorted(results, key=lambda r: r['median_return'], reverse=True)

    print(f"\nTop {args.top_n} Parameter Combinations by Median Return:")
    print("-"*100)
    print(f"{'Rank':<6} {'Threshold':<12} {'Interval':<12} {'Median Ret':<13} {'Med Sharpe':<12} {'Med DD':<10} {'Worst DD':<10}")
    print("-"*100)

    for rank, result in enumerate(results_sorted[:args.top_n], 1):
        star = " ⭐" if rank == 1 else ""
        print(f"{rank:<6} {result['threshold_percent']:>5.1f}%{'':<6} "
              f"{result['interval_days']:>5.1f} days{'':<3} "
              f"{result['median_return']:>+6.2f}%{star:<6} "
              f"{result['median_sharpe']:>6.3f}{'':<6} "
              f"{result['median_max_drawdown']:>5.2f}%{'':<5} "
              f"{result['worst_max_drawdown']:>5.2f}%")

    # Sort by median Sharpe
    results_by_sharpe = sorted(results, key=lambda r: r['median_sharpe'], reverse=True)

    print(f"\nTop {args.top_n} Parameter Combinations by Median Sharpe Ratio:")
    print("-"*100)
    print(f"{'Rank':<6} {'Threshold':<12} {'Interval':<12} {'Med Sharpe':<12} {'Median Ret':<13} {'Med DD':<10}")
    print("-"*100)

    for rank, result in enumerate(results_by_sharpe[:args.top_n], 1):
        star = " ⭐" if rank == 1 else ""
        print(f"{rank:<6} {result['threshold_percent']:>5.1f}%{'':<6} "
              f"{result['interval_days']:>5.1f} days{'':<3} "
              f"{result['median_sharpe']:>6.3f}{star:<6} "
              f"{result['median_return']:>+6.2f}%{'':<6} "
              f"{result['median_max_drawdown']:>5.2f}%")

    # Overall winner
    print("\n" + "="*100)
    print("RECOMMENDED PARAMETERS")
    print("="*100)
    winner = results_sorted[0]
    print(f"\n🏆 Best Parameter Combination (by median return):")
    print(f"   Threshold: {winner['threshold_percent']:.1f}%")
    print(f"   Interval: {winner['interval_days']:.1f} days ({int(winner['interval_minutes'])} minutes)")
    print(f"\n   Monte Carlo Performance ({args.simulations:,} simulations):")
    print(f"   • Median Return: {winner['median_return']:+.2f}%")
    print(f"   • Mean Return: {winner['mean_return']:+.2f}%")
    print(f"   • Return Range: {winner['percentile_5_return']:.2f}% to {winner['percentile_95_return']:.2f}% (5th-95th %ile)")
    print(f"   • Median Sharpe: {winner['median_sharpe']:.3f}")
    print(f"   • Median Max Drawdown: {winner['median_max_drawdown']:.2f}%")
    print(f"   • Worst Max Drawdown: {winner['worst_max_drawdown']:.2f}%")
    print(f"   • Avg Rebalances: {winner['mean_num_rebalances']:.1f}")
    print(f"   • Avg Fees: ${winner['mean_total_fees']:.2f}")
    print(f"\n   Original Historical Performance:")
    print(f"   • Return: {winner['original_return']:+.2f}%")
    print(f"   • Sharpe: {winner['original_sharpe']:.3f}")
    print(f"   • Max Drawdown: {winner['original_max_drawdown']:.2f}%")

    # Export to CSV if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import csv
        with open(output_path, 'w', newline='') as f:
            if results:
                fieldnames = list(results[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results_sorted)

        print(f"\n✓ Results exported to: {args.output}")

    print("\n" + "="*100)
    print("OPTIMIZATION COMPLETE")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
