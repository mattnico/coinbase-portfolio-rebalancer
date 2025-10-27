#!/usr/bin/env python3
"""
CLI tool for optimizing both portfolio allocations AND rebalancing parameters.

Tests multiple predefined portfolio allocations to find which combination
of allocation + threshold + interval produces the best results.

Usage:
    python -m src.optimize_portfolios \
        --days 14 \
        --portfolios config/test_portfolios.json \
        --threshold-min 0.5 --threshold-max 2.0 --threshold-step 0.5 \
        --interval-min 5 --interval-max 30 --interval-step 5 \
        --output results/portfolio_optimization.csv
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
from typing import Dict, List
from dataclasses import dataclass

from src.coinbase_client import CoinbaseClient
from src.monte_carlo_simulator import (
    SimulationConfig,
    HistoricalPriceFetcher,
)
from src.strategy_optimizer import StrategyOptimizer, OptimizationResult, OptimizationSummary


logger = logging.getLogger(__name__)


@dataclass
class PortfolioOptimizationResult:
    """Results for a single portfolio allocation."""
    portfolio_name: str
    portfolio_description: str
    allocation: Dict[str, float]
    optimization_summary: OptimizationSummary
    best_return_strategy: OptimizationResult
    best_sharpe_strategy: OptimizationResult
    best_net_return_strategy: OptimizationResult


def setup_logging(quiet: bool = False):
    """Configure logging."""
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_portfolios(portfolios_file: str) -> Dict[str, Dict[str, float]]:
    """
    Load portfolio definitions from JSON file.

    Args:
        portfolios_file: Path to portfolios JSON file

    Returns:
        Dict mapping portfolio name to allocation dict
    """
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
        # Extract allocation (exclude 'description' field)
        allocation = {k: v for k, v in portfolio_data.items() if k != 'description'}

        # Validate allocation sums to ~100%
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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Optimize portfolio allocations AND rebalancing parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test 7 portfolios × 4 thresholds × 6 intervals = 168 combinations
  python -m src.optimize_portfolios \\
      --days 14 \\
      --portfolios config/test_portfolios.json \\
      --threshold-min 0.5 --threshold-max 2.0 --threshold-step 0.5 \\
      --interval-min 5 --interval-max 30 --interval-step 5 \\
      --output results.csv
        """
    )

    # Time period options
    time_group = parser.add_mutually_exclusive_group(required=True)
    time_group.add_argument(
        '--days',
        type=int,
        help='Number of days to simulate (from today backward)'
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
        help='Minimum threshold to test (%%) (e.g., 0.5)'
    )
    parser.add_argument(
        '--threshold-max',
        type=float,
        required=True,
        help='Maximum threshold to test (%%) (e.g., 2.5)'
    )
    parser.add_argument(
        '--threshold-step',
        type=float,
        default=0.1,
        help='Step size for threshold (%%) (default: 0.1)'
    )

    # Interval range parameters (minutes)
    parser.add_argument(
        '--interval-min',
        type=float,
        required=True,
        help='Minimum interval to test (minutes) (e.g., 5)'
    )
    parser.add_argument(
        '--interval-max',
        type=float,
        required=True,
        help='Maximum interval to test (minutes) (e.g., 60)'
    )
    parser.add_argument(
        '--interval-step',
        type=float,
        default=5,
        help='Step size for interval (minutes) (default: 5)'
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
        default=5,
        help='Number of top strategies to display per portfolio (default: 5)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed logging'
    )

    return parser.parse_args()


def main():
    """Run portfolio + parameter optimization."""
    args = parse_arguments()

    # Set up logging
    setup_logging(args.quiet)

    # Load portfolios
    logger.info(f"Loading portfolios from {args.portfolios}")
    portfolios = load_portfolios(args.portfolios)

    print("\n" + "="*100)
    print("PORTFOLIO ALLOCATION + PARAMETER OPTIMIZATION")
    print("="*100)
    print(f"\nTesting {len(portfolios)} portfolio allocations:")
    for name, data in portfolios.items():
        desc = data['description']
        print(f"  • {name}: {desc if desc else 'No description'}")

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

    print(f"\nSimulation Period: {start_date.date()} to {end_date.date()}")
    print(f"\nOptimization Ranges:")
    print(f"  Threshold: {args.threshold_min}% to {args.threshold_max}% (step: {args.threshold_step}%)")
    print(f"  Interval: {args.interval_min} to {args.interval_max} minutes (step: {args.interval_step} min)")

    # Calculate total combinations
    num_thresholds = int((args.threshold_max - args.threshold_min) / args.threshold_step) + 1
    num_intervals = int((args.interval_max - args.interval_min) / args.interval_step) + 1
    combos_per_portfolio = num_thresholds * num_intervals
    total_combinations = combos_per_portfolio * len(portfolios)

    print(f"\nCombinations per portfolio: {combos_per_portfolio} ({num_thresholds} thresholds × {num_intervals} intervals)")
    print(f"Total combinations: {total_combinations} ({len(portfolios)} portfolios × {combos_per_portfolio})")
    print(f"Parallel Workers: {args.workers or 'auto (CPU count)'}")

    # Auto-detect granularity
    if args.granularity is None:
        granularity = auto_detect_granularity(args.interval_min)
        logger.info(f"Auto-detected granularity: {granularity}")
        print(f"Granularity: {granularity} (auto-detected)")
    else:
        granularity = args.granularity
        print(f"Granularity: {granularity}")

    print("="*100 + "\n")

    # Initialize Coinbase client
    logger.info("Initializing Coinbase client...")
    try:
        client = CoinbaseClient()
    except Exception as e:
        logger.error(f"Failed to initialize Coinbase client: {e}")
        sys.exit(1)

    # Fetch historical price data ONCE (will be reused for all portfolios)
    # Get union of all assets across all portfolios
    all_assets = set()
    for portfolio_data in portfolios.values():
        all_assets.update(portfolio_data['allocation'].keys())

    use_cache = not args.no_cache
    if args.cache_only and args.no_cache:
        logger.error("Cannot specify both --no-cache and --cache-only")
        sys.exit(1)

    cache_mode = "disabled" if args.no_cache else ("cache-only" if args.cache_only else "enabled")
    logger.info(f"Fetching historical price data for {len(all_assets)} assets (granularity: {granularity}, cache: {cache_mode})...")

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

    # Validate we got data
    for asset, prices in price_data.items():
        if not prices:
            logger.error(f"No price data retrieved for {asset}")
            sys.exit(1)
        logger.info(f"{asset}: {len(prices)} data points")

    print("\n" + "="*100)
    print("RUNNING OPTIMIZATIONS")
    print("="*100 + "\n")

    # Run optimization for each portfolio
    portfolio_results = []

    for idx, (portfolio_name, portfolio_data) in enumerate(portfolios.items(), 1):
        allocation = portfolio_data['allocation']
        description = portfolio_data['description']

        print(f"\n[{idx}/{len(portfolios)}] Optimizing: {portfolio_name}")
        print(f"Description: {description}")
        print(f"Allocation: {allocation}")
        print(f"Running {combos_per_portfolio} parameter combinations...\n")

        # Auto-set check interval based on minimum interval
        check_interval_hours = args.interval_min / 60.0

        # Create simulation config for this portfolio
        sim_config = SimulationConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital_usd=args.initial_capital,
            target_allocation=allocation,
            fee_rate=args.fee_rate,
            price_check_interval_hours=check_interval_hours
        )

        # Initialize optimizer
        optimizer = StrategyOptimizer(
            price_data=price_data,
            sim_config=sim_config,
            max_workers=args.workers
        )

        # Run optimization
        summary = optimizer.optimize(
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
            interval_min=args.interval_min,
            interval_max=args.interval_max,
            interval_step=args.interval_step,
            show_progress=not args.no_progress
        )

        # Store results
        portfolio_results.append(PortfolioOptimizationResult(
            portfolio_name=portfolio_name,
            portfolio_description=description,
            allocation=allocation,
            optimization_summary=summary,
            best_return_strategy=summary.best_return,
            best_sharpe_strategy=summary.best_sharpe,
            best_net_return_strategy=summary.best_net_return
        ))

        print(f"✓ {portfolio_name} optimization complete")
        print(f"  Best return: {summary.best_return.total_return_percent:+.2f}% "
              f"({summary.best_return.threshold_percent:.1f}% threshold, {summary.best_return.interval_minutes:.0f}min interval)")

    # Print comparison across all portfolios
    print("\n\n" + "="*100)
    print("CROSS-PORTFOLIO COMPARISON")
    print("="*100)

    # Sort by best total return
    sorted_by_return = sorted(portfolio_results, key=lambda r: r.best_return_strategy.total_return_percent, reverse=True)

    print("\n" + "="*100)
    print("BEST PORTFOLIOS BY TOTAL RETURN")
    print("="*100)
    print(f"{'Rank':<6} {'Portfolio':<20} {'Best Return':<15} {'Threshold':<12} {'Interval':<12} {'Sharpe':<10}")
    print("-"*100)

    for rank, result in enumerate(sorted_by_return, 1):
        best = result.best_return_strategy
        star = " ⭐" if rank == 1 else ""
        print(f"{rank:<6} {result.portfolio_name:<20} {best.total_return_percent:>+6.2f}%{star:<8} "
              f"{best.threshold_percent:>5.1f}%{'':<7} {best.interval_minutes:>5.0f}min{'':<7} "
              f"{best.sharpe_ratio:>6.3f}")

    # Sort by best Sharpe ratio
    sorted_by_sharpe = sorted(portfolio_results, key=lambda r: r.best_sharpe_strategy.sharpe_ratio, reverse=True)

    print("\n" + "="*100)
    print("BEST PORTFOLIOS BY SHARPE RATIO (Risk-Adjusted Returns)")
    print("="*100)
    print(f"{'Rank':<6} {'Portfolio':<20} {'Sharpe Ratio':<15} {'Threshold':<12} {'Interval':<12} {'Return':<10}")
    print("-"*100)

    for rank, result in enumerate(sorted_by_sharpe, 1):
        best = result.best_sharpe_strategy
        star = " ⭐" if rank == 1 else ""
        print(f"{rank:<6} {result.portfolio_name:<20} {best.sharpe_ratio:>6.3f}{star:<9} "
              f"{best.threshold_percent:>5.1f}%{'':<7} {best.interval_minutes:>5.0f}min{'':<7} "
              f"{best.total_return_percent:>+6.2f}%")

    # Overall winner
    print("\n" + "="*100)
    print("OVERALL WINNER")
    print("="*100)
    winner = sorted_by_return[0]
    best_strat = winner.best_return_strategy

    print(f"\n🏆 Best Portfolio: {winner.portfolio_name}")
    print(f"   Description: {winner.portfolio_description}")
    print(f"   Allocation: {winner.allocation}")
    print(f"\n   Best Parameters:")
    print(f"   • Threshold: {best_strat.threshold_percent:.1f}%")
    print(f"   • Interval: {best_strat.interval_minutes:.0f} minutes")
    print(f"\n   Performance:")
    print(f"   • Total Return: {best_strat.total_return_percent:+.2f}%")
    print(f"   • Annualized Return: {best_strat.annualized_return_percent:+.2f}%")
    print(f"   • Sharpe Ratio: {best_strat.sharpe_ratio:.3f}")
    print(f"   • Max Drawdown: {best_strat.max_drawdown_percent:.2f}%")
    print(f"   • Total Fees: ${best_strat.total_fees_paid:.2f}")
    print(f"   • Rebalances: {best_strat.num_rebalances}")

    baseline = winner.optimization_summary.baseline_result
    print(f"\n   vs Buy-and-Hold:")
    print(f"   • Return difference: {best_strat.total_return_percent - baseline.total_return_percent:+.2f} pp")
    print(f"   • Sharpe difference: {best_strat.sharpe_ratio - baseline.sharpe_ratio:+.3f}")

    # Export to CSV if requested
    if args.output:
        print(f"\n{'='*100}")
        print("EXPORTING RESULTS")
        print("="*100)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Combine all results into one CSV
        all_results = []
        for portfolio_result in portfolio_results:
            for strategy_result in portfolio_result.optimization_summary.strategy_results:
                row = strategy_result.to_dict()
                row['portfolio_name'] = portfolio_result.portfolio_name
                row['portfolio_description'] = portfolio_result.portfolio_description
                # Add allocation as JSON string
                row['portfolio_allocation'] = json.dumps(portfolio_result.allocation)
                all_results.append(row)

        # Write to CSV
        if all_results:
            import csv
            with open(output_path, 'w', newline='') as f:
                fieldnames = list(all_results[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)

            print(f"✓ Exported {len(all_results)} results to: {args.output}")

    print("\n" + "="*100)
    print("OPTIMIZATION COMPLETE")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
