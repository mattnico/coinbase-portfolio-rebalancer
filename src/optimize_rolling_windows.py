#!/usr/bin/env python3
"""
CLI tool for rolling window optimization across time periods.

Tests portfolio allocations and rebalancing parameters across multiple
overlapping or non-overlapping time windows to find consistently winning
strategies across different market conditions.

Usage:
    # Quick validation: 4 portfolios, coarse grid, monthly steps (~2 hours)
    python -m src.optimize_rolling_windows \
        --days 730 \
        --window-size 90 \
        --step-size 30 \
        --portfolios config/top_portfolios.json \
        --threshold-min 5.0 --threshold-max 20.0 --threshold-step 5.0 \
        --interval-min 60 --interval-max 360 --interval-step 60 \
        --output rolling_results.csv
"""

import argparse
import logging
import json
import sys
import warnings
from datetime import datetime, timedelta

# Suppress urllib3 OpenSSL warnings
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import time

from src.coinbase_client import CoinbaseClient
from src.monte_carlo_simulator import (
    SimulationConfig,
    HistoricalPriceFetcher,
)
from src.strategy_optimizer import StrategyOptimizer, OptimizationResult


logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    """Results for a single time window."""
    window_num: int
    start_date: datetime
    end_date: datetime
    winning_portfolio: str
    winning_threshold: float
    winning_interval: float
    winning_return: float
    winning_sharpe: float
    baseline_return: float
    return_vs_baseline: float
    all_portfolio_results: Dict[str, OptimizationResult] = field(default_factory=dict)


@dataclass
class RollingAnalysisSummary:
    """Summary of rolling window analysis."""
    total_windows: int
    window_results: List[WindowResult]
    portfolio_win_counts: Dict[str, int]
    portfolio_avg_returns: Dict[str, float]
    portfolio_avg_sharpe: Dict[str, float]
    most_common_thresholds: List[Tuple[float, int]]
    most_common_intervals: List[Tuple[float, int]]
    execution_time_seconds: float


def setup_logging(quiet: bool = False):
    """Configure logging."""
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_portfolios(portfolios_file: str) -> Dict[str, Dict]:
    """Load portfolio definitions from JSON file."""
    config_file = Path(portfolios_file)

    if not config_file.exists():
        print(f"Error: Portfolios file not found at {portfolios_file}")
        sys.exit(1)

    with open(config_file, 'r') as f:
        data = json.load(f)

    if 'portfolios' not in data:
        print(f"Error: Portfolios file must have 'portfolios' key")
        sys.exit(1)

    portfolios = {}
    for name, portfolio_data in data['portfolios'].items():
        allocation = {k: v for k, v in portfolio_data.items() if k != 'description'}
        total = sum(allocation.values())
        if not 99.9 <= total <= 100.1:
            print(f"Error: Portfolio '{name}' allocation sums to {total}%, must be ~100%")
            sys.exit(1)

        portfolios[name] = {
            'allocation': allocation,
            'description': portfolio_data.get('description', '')
        }

    return portfolios


def auto_detect_granularity(interval_min: float) -> str:
    """Auto-detect appropriate granularity based on minimum interval."""
    if interval_min <= 1:
        return 'ONE_MINUTE'
    elif interval_min <= 5:
        return 'FIVE_MINUTE'
    elif interval_min <= 15:
        return 'FIFTEEN_MINUTE'
    elif interval_min <= 30:
        return 'THIRTY_MINUTE'
    elif interval_min <= 60:
        return 'ONE_HOUR'
    elif interval_min <= 120:
        return 'TWO_HOUR'
    elif interval_min <= 360:
        return 'SIX_HOUR'
    else:
        return 'ONE_DAY'


def optimize_single_window(
    window_num: int,
    start_date: datetime,
    end_date: datetime,
    portfolios: Dict[str, Dict],
    price_data: Dict,
    args
) -> WindowResult:
    """
    Optimize a single time window.

    Returns the winning portfolio and parameters for this window.
    """
    logger.info(f"\nWindow {window_num}: {start_date.date()} to {end_date.date()}")

    # Filter price data to window
    window_price_data = {}
    for asset, prices in price_data.items():
        window_prices = [
            (ts, price) for ts, price in prices
            if start_date <= ts <= end_date
        ]
        window_price_data[asset] = window_prices

    # Track best result across all portfolios
    best_portfolio = None
    best_result = None
    best_return = float('-inf')
    portfolio_results = {}

    check_interval_hours = args.interval_min / 60.0

    # Optimize each portfolio
    for portfolio_name, portfolio_data in portfolios.items():
        allocation = portfolio_data['allocation']

        # Create simulation config
        sim_config = SimulationConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital_usd=args.initial_capital,
            target_allocation=allocation,
            fee_rate=args.fee_rate,
            price_check_interval_hours=check_interval_hours
        )

        # Run optimization
        optimizer = StrategyOptimizer(
            price_data=window_price_data,
            sim_config=sim_config,
            max_workers=args.workers
        )

        summary = optimizer.optimize(
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
            interval_min=args.interval_min,
            interval_max=args.interval_max,
            interval_step=args.interval_step,
            show_progress=False  # Suppress per-portfolio progress
        )

        # Store best result for this portfolio
        best_for_portfolio = summary.best_return
        portfolio_results[portfolio_name] = best_for_portfolio

        # Check if this is overall best
        if best_for_portfolio.total_return_percent > best_return:
            best_return = best_for_portfolio.total_return_percent
            best_portfolio = portfolio_name
            best_result = best_for_portfolio
            baseline_return = summary.baseline_result.total_return_percent

    return WindowResult(
        window_num=window_num,
        start_date=start_date,
        end_date=end_date,
        winning_portfolio=best_portfolio,
        winning_threshold=best_result.threshold_percent,
        winning_interval=best_result.interval_minutes,
        winning_return=best_result.total_return_percent,
        winning_sharpe=best_result.sharpe_ratio,
        baseline_return=baseline_return,
        return_vs_baseline=best_result.total_return_percent - baseline_return,
        all_portfolio_results=portfolio_results
    )


def analyze_results(window_results: List[WindowResult]) -> RollingAnalysisSummary:
    """Analyze results across all windows."""

    # Count portfolio wins
    portfolio_wins = Counter(r.winning_portfolio for r in window_results)

    # Calculate average returns per portfolio
    portfolio_returns = defaultdict(list)
    portfolio_sharpes = defaultdict(list)

    for result in window_results:
        for portfolio_name, opt_result in result.all_portfolio_results.items():
            portfolio_returns[portfolio_name].append(opt_result.total_return_percent)
            portfolio_sharpes[portfolio_name].append(opt_result.sharpe_ratio)

    portfolio_avg_returns = {
        name: sum(returns) / len(returns)
        for name, returns in portfolio_returns.items()
    }

    portfolio_avg_sharpe = {
        name: sum(sharpes) / len(sharpes)
        for name, sharpes in portfolio_sharpes.items()
    }

    # Count most common parameters
    threshold_counts = Counter(r.winning_threshold for r in window_results)
    interval_counts = Counter(r.winning_interval for r in window_results)

    return RollingAnalysisSummary(
        total_windows=len(window_results),
        window_results=window_results,
        portfolio_win_counts=dict(portfolio_wins),
        portfolio_avg_returns=portfolio_avg_returns,
        portfolio_avg_sharpe=portfolio_avg_sharpe,
        most_common_thresholds=threshold_counts.most_common(5),
        most_common_intervals=interval_counts.most_common(5),
        execution_time_seconds=0  # Set by caller
    )


def print_summary(summary: RollingAnalysisSummary, portfolios: Dict[str, Dict]):
    """Print comprehensive summary of rolling window analysis."""

    print("\n" + "="*100)
    print("ROLLING WINDOW ANALYSIS SUMMARY")
    print("="*100)

    print(f"\nTotal Windows Tested: {summary.total_windows}")
    print(f"Execution Time: {summary.execution_time_seconds/60:.1f} minutes")

    # Portfolio win rates
    print("\n" + "="*100)
    print("PORTFOLIO WIN RATES")
    print("="*100)

    sorted_portfolios = sorted(
        summary.portfolio_win_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    max_wins = max(summary.portfolio_win_counts.values())

    for rank, (portfolio_name, wins) in enumerate(sorted_portfolios, 1):
        win_rate = (wins / summary.total_windows) * 100
        bar_length = int((wins / max_wins) * 40)
        bar = "█" * bar_length
        star = " 🏆" if rank == 1 else ""

        desc = portfolios[portfolio_name]['description']
        print(f"{rank}. {portfolio_name:<20} {bar:<40} {win_rate:>5.1f}% ({wins}/{summary.total_windows} wins){star}")
        if desc:
            print(f"   {desc}")

    # Average performance
    print("\n" + "="*100)
    print("AVERAGE PERFORMANCE ACROSS ALL WINDOWS")
    print("="*100)
    print(f"{'Portfolio':<20} {'Avg Return':<15} {'Avg Sharpe':<12} {'Win Rate'}")
    print("-"*100)

    for portfolio_name in sorted(summary.portfolio_avg_returns.keys()):
        avg_return = summary.portfolio_avg_returns[portfolio_name]
        avg_sharpe = summary.portfolio_avg_sharpe[portfolio_name]
        wins = summary.portfolio_win_counts.get(portfolio_name, 0)
        win_rate = (wins / summary.total_windows) * 100

        print(f"{portfolio_name:<20} {avg_return:>+6.2f}%{'':<8} {avg_sharpe:>6.3f}{'':<6} {win_rate:>5.1f}%")

    # Most common winning parameters
    print("\n" + "="*100)
    print("MOST COMMON WINNING PARAMETERS")
    print("="*100)

    print("\nThresholds:")
    for threshold, count in summary.most_common_thresholds:
        pct = (count / summary.total_windows) * 100
        print(f"  {threshold:>5.1f}%  appeared in {count:>3} windows ({pct:>5.1f}%)")

    print("\nIntervals:")
    for interval, count in summary.most_common_intervals:
        pct = (count / summary.total_windows) * 100
        print(f"  {interval:>5.0f} min  appeared in {count:>3} windows ({pct:>5.1f}%)")

    # Best and worst windows
    print("\n" + "="*100)
    print("BEST AND WORST WINDOWS")
    print("="*100)

    sorted_by_return = sorted(summary.window_results, key=lambda r: r.winning_return, reverse=True)

    print("\nBest 5 Windows:")
    for i, result in enumerate(sorted_by_return[:5], 1):
        print(f"{i}. {result.start_date.date()} to {result.end_date.date()}")
        print(f"   Winner: {result.winning_portfolio} ({result.winning_threshold:.1f}%, {result.winning_interval:.0f}min)")
        print(f"   Return: {result.winning_return:+.2f}% (vs baseline: {result.baseline_return:+.2f}%)")

    print("\nWorst 5 Windows:")
    for i, result in enumerate(reversed(sorted_by_return[-5:]), 1):
        print(f"{i}. {result.start_date.date()} to {result.end_date.date()}")
        print(f"   Winner: {result.winning_portfolio} ({result.winning_threshold:.1f}%, {result.winning_interval:.0f}min)")
        print(f"   Return: {result.winning_return:+.2f}% (vs baseline: {result.baseline_return:+.2f}%)")

    # Overall recommendation
    print("\n" + "="*100)
    print("RECOMMENDATION")
    print("="*100)

    winner_name = sorted_portfolios[0][0]
    winner_wins = sorted_portfolios[0][1]
    winner_rate = (winner_wins / summary.total_windows) * 100

    most_common_threshold = summary.most_common_thresholds[0][0]
    most_common_interval = summary.most_common_intervals[0][0]

    print(f"\n🏆 Most Consistent Portfolio: {winner_name}")
    print(f"   Won {winner_wins}/{summary.total_windows} windows ({winner_rate:.1f}%)")
    print(f"   Average return: {summary.portfolio_avg_returns[winner_name]:+.2f}%")
    print(f"   Average Sharpe: {summary.portfolio_avg_sharpe[winner_name]:.3f}")

    print(f"\n📊 Most Common Winning Parameters:")
    print(f"   Threshold: {most_common_threshold:.1f}%")
    print(f"   Interval: {most_common_interval:.0f} minutes")

    print(f"\n💡 Strategy: Use {winner_name} with {most_common_threshold:.1f}% threshold")
    print(f"   and {most_common_interval:.0f}-minute rebalancing interval")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Rolling window optimization across time periods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test: 4 portfolios, coarse grid, monthly steps (~2 hours)
  python -m src.optimize_rolling_windows \\
      --days 730 \\
      --window-size 90 \\
      --step-size 30 \\
      --portfolios config/top_portfolios.json \\
      --threshold-min 5.0 --threshold-max 20.0 --threshold-step 5.0 \\
      --interval-min 60 --interval-max 360 --interval-step 60

  # Weekly rolling windows (slower but more comprehensive)
  python -m src.optimize_rolling_windows \\
      --days 365 \\
      --window-size 90 \\
      --step-size 7 \\
      --portfolios config/test_portfolios.json \\
      --threshold-min 10.0 --threshold-max 20.0 --threshold-step 5.0 \\
      --interval-min 120 --interval-max 360 --interval-step 60
        """
    )

    # Time period options
    parser.add_argument(
        '--days',
        type=int,
        required=True,
        help='Total historical period in days (e.g., 730 for 2 years)'
    )

    parser.add_argument(
        '--window-size',
        type=int,
        default=90,
        help='Size of each rolling window in days (default: 90)'
    )

    parser.add_argument(
        '--step-size',
        type=int,
        default=30,
        help='Days to step forward between windows (default: 30, use 1 for daily rolling)'
    )

    # Portfolio options
    parser.add_argument(
        '--portfolios',
        type=str,
        required=True,
        help='Path to portfolios JSON file'
    )

    # Threshold range parameters
    parser.add_argument(
        '--threshold-min',
        type=float,
        required=True,
        help='Minimum threshold to test (%%) (e.g., 5.0)'
    )
    parser.add_argument(
        '--threshold-max',
        type=float,
        required=True,
        help='Maximum threshold to test (%%) (e.g., 20.0)'
    )
    parser.add_argument(
        '--threshold-step',
        type=float,
        default=5.0,
        help='Step size for threshold (%%) (default: 5.0)'
    )

    # Interval range parameters
    parser.add_argument(
        '--interval-min',
        type=float,
        required=True,
        help='Minimum interval to test (minutes) (e.g., 60)'
    )
    parser.add_argument(
        '--interval-max',
        type=float,
        required=True,
        help='Maximum interval to test (minutes) (e.g., 360)'
    )
    parser.add_argument(
        '--interval-step',
        type=float,
        default=60,
        help='Step size for interval (minutes) (default: 60)'
    )

    # Simulation parameters
    parser.add_argument(
        '--fee-rate',
        type=float,
        default=0.006,
        help='Trading fee rate as decimal (default: 0.006 = 0.6%%)'
    )

    parser.add_argument(
        '--initial-capital',
        type=float,
        default=10000.0,
        help='Initial portfolio value in USD (default: 10000)'
    )

    parser.add_argument(
        '--granularity',
        type=str,
        default=None,
        choices=['ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 'THIRTY_MINUTE',
                 'ONE_HOUR', 'TWO_HOUR', 'SIX_HOUR', 'ONE_DAY'],
        help='Price data granularity (default: auto-detect based on interval-min)'
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

    parser.add_argument(
        '--cache-max-age',
        type=int,
        default=7,
        help='Maximum age of cached data in days (default: 7)'
    )

    parser.add_argument(
        '--request-delay',
        type=float,
        default=0.3,
        help='Delay between API requests in seconds (default: 0.3)'
    )

    # Performance options
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count)'
    )

    # Output options
    parser.add_argument(
        '--output',
        type=str,
        help='Save detailed results to CSV file (optional)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed logging'
    )

    return parser.parse_args()


def main():
    """Run rolling window analysis."""
    args = parse_arguments()

    start_time = time.time()

    # Set up logging
    setup_logging(args.quiet)

    # Load portfolios
    logger.info(f"Loading portfolios from {args.portfolios}")
    portfolios = load_portfolios(args.portfolios)

    # Calculate windows
    total_days = args.days
    window_size = args.window_size
    step_size = args.step_size

    if window_size > total_days:
        print(f"Error: Window size ({window_size}) cannot be larger than total days ({total_days})")
        sys.exit(1)

    num_windows = ((total_days - window_size) // step_size) + 1

    print("\n" + "="*100)
    print("ROLLING WINDOW OPTIMIZATION")
    print("="*100)
    print(f"\nConfiguration:")
    print(f"  Total Period: {total_days} days")
    print(f"  Window Size: {window_size} days")
    print(f"  Step Size: {step_size} days")
    print(f"  Number of Windows: {num_windows}")
    print(f"  Portfolios: {len(portfolios)}")

    for name in portfolios.keys():
        print(f"    • {name}")

    # Calculate combinations
    num_thresholds = int((args.threshold_max - args.threshold_min) / args.threshold_step) + 1
    num_intervals = int((args.interval_max - args.interval_min) / args.interval_step) + 1
    combos_per_portfolio = num_thresholds * num_intervals
    total_combos_per_window = combos_per_portfolio * len(portfolios)
    total_simulations = total_combos_per_window * num_windows

    print(f"\nParameter Grid:")
    print(f"  Thresholds: {args.threshold_min}% to {args.threshold_max}% (step: {args.threshold_step}%) = {num_thresholds} values")
    print(f"  Intervals: {args.interval_min} to {args.interval_max} min (step: {args.interval_step}) = {num_intervals} values")
    print(f"  Combinations per portfolio: {combos_per_portfolio}")
    print(f"  Total per window: {total_combos_per_window}")
    print(f"  GRAND TOTAL: {total_simulations} simulations")

    # Auto-detect granularity
    if args.granularity is None:
        granularity = auto_detect_granularity(args.interval_min)
        print(f"\nGranularity: {granularity} (auto-detected)")
    else:
        granularity = args.granularity
        print(f"\nGranularity: {granularity}")

    print("="*100 + "\n")

    # Initialize Coinbase client
    logger.info("Initializing Coinbase client...")
    try:
        client = CoinbaseClient()
    except Exception as e:
        logger.error(f"Failed to initialize Coinbase client: {e}")
        sys.exit(1)

    # Calculate full date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=total_days)

    # Fetch ALL historical price data once
    all_assets = set()
    for portfolio_data in portfolios.values():
        all_assets.update(portfolio_data['allocation'].keys())

    use_cache = not args.no_cache
    if args.cache_only and args.no_cache:
        logger.error("Cannot specify both --no-cache and --cache-only")
        sys.exit(1)

    cache_mode = "disabled" if args.no_cache else ("cache-only" if args.cache_only else "enabled")
    print(f"Fetching {total_days} days of price data for {len(all_assets)} assets...")
    print(f"Cache mode: {cache_mode}")
    print("This may take a few minutes...\n")

    fetcher = HistoricalPriceFetcher(
        client,
        use_cache=use_cache,
        cache_max_age_days=args.cache_max_age if use_cache else None,
        request_delay=args.request_delay
    )

    try:
        price_data = fetcher.fetch_historical_prices(
            assets=list(all_assets),
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
            cache_only=args.cache_only
        )
    except Exception as e:
        logger.error(f"Failed to fetch price data: {e}")
        sys.exit(1)

    # Validate data
    for asset, prices in price_data.items():
        if not prices:
            logger.error(f"No price data retrieved for {asset}")
            sys.exit(1)
        logger.info(f"{asset}: {len(prices)} data points")

    print("\n" + "="*100)
    print(f"RUNNING {num_windows} WINDOW OPTIMIZATIONS")
    print("="*100 + "\n")

    # Run optimization for each window
    window_results = []

    for i in range(num_windows):
        window_start = start_date + timedelta(days=i * step_size)
        window_end = window_start + timedelta(days=window_size)

        print(f"\n{'='*100}")
        print(f"Window {i+1}/{num_windows}: {window_start.date()} to {window_end.date()}")
        print(f"{'='*100}")

        result = optimize_single_window(
            window_num=i+1,
            start_date=window_start,
            end_date=window_end,
            portfolios=portfolios,
            price_data=price_data,
            args=args
        )

        window_results.append(result)

        print(f"\n✓ Window {i+1} complete:")
        print(f"  Winner: {result.winning_portfolio}")
        print(f"  Parameters: {result.winning_threshold:.1f}% threshold, {result.winning_interval:.0f}min interval")
        print(f"  Return: {result.winning_return:+.2f}% (vs baseline {result.baseline_return:+.2f}%)")

    # Analyze results
    execution_time = time.time() - start_time
    summary = analyze_results(window_results)
    summary.execution_time_seconds = execution_time

    # Print summary
    print_summary(summary, portfolios)

    # Export to CSV if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import csv
        with open(output_path, 'w', newline='') as f:
            fieldnames = [
                'window_num', 'start_date', 'end_date',
                'winning_portfolio', 'winning_threshold', 'winning_interval',
                'winning_return', 'winning_sharpe', 'baseline_return', 'return_vs_baseline'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in window_results:
                writer.writerow({
                    'window_num': result.window_num,
                    'start_date': result.start_date.strftime('%Y-%m-%d'),
                    'end_date': result.end_date.strftime('%Y-%m-%d'),
                    'winning_portfolio': result.winning_portfolio,
                    'winning_threshold': result.winning_threshold,
                    'winning_interval': result.winning_interval,
                    'winning_return': result.winning_return,
                    'winning_sharpe': result.winning_sharpe,
                    'baseline_return': result.baseline_return,
                    'return_vs_baseline': result.return_vs_baseline,
                })

        print(f"\n✓ Detailed results exported to: {args.output}")

    print("\n" + "="*100)
    print("ROLLING WINDOW ANALYSIS COMPLETE")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
