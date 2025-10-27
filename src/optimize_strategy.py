#!/usr/bin/env python3
"""
CLI tool for optimizing rebalancing strategy parameters.

Performs grid search over threshold and interval ranges to find
optimal parameter combinations for maximizing returns, Sharpe ratio,
or other metrics.

Usage:
    python -m src.optimize_strategy \
        --days 7 \
        --threshold-min 0.5 --threshold-max 2.5 --threshold-step 0.1 \
        --interval-min 5 --interval-max 60 --interval-step 5 \
        --granularity ONE_HOUR \
        --output results.csv \
        --heatmap heatmaps/
"""

import argparse
import logging
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from src.coinbase_client import CoinbaseClient
from src.monte_carlo_simulator import (
    SimulationConfig,
    HistoricalPriceFetcher,
)
from src.strategy_optimizer import StrategyOptimizer
from src.visualization import (
    export_to_csv,
    print_top_strategies_table,
    print_optimization_summary,
    generate_heatmaps,
    print_comparison_vs_baseline,
)


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


def auto_detect_granularity(interval_min: float) -> str:
    """
    Auto-detect appropriate granularity based on minimum interval.

    Matches granularity to interval for accurate simulation:
    - 1-min interval needs 1-min granularity
    - 5-min interval needs 5-min granularity
    - etc.

    Args:
        interval_min: Minimum rebalancing interval in minutes

    Returns:
        Recommended granularity string
    """
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
        description='Optimize rebalancing strategy parameters using grid search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize threshold (0.5-2.5%) and interval (5-60min) for last 7 days
  python -m src.optimize_strategy \\
      --days 7 \\
      --threshold-min 0.5 --threshold-max 2.5 --threshold-step 0.1 \\
      --interval-min 5 --interval-max 60 --interval-step 5 \\
      --granularity ONE_HOUR

  # Save results and generate heatmaps
  python -m src.optimize_strategy \\
      --days 14 \\
      --threshold-min 1.0 --threshold-max 5.0 --threshold-step 0.5 \\
      --interval-min 15 --interval-max 120 --interval-step 15 \\
      --output optimization_results.csv \\
      --heatmap heatmaps/
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

    # Interval range parameters (minutes only for now)
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
        '--heatmap',
        type=str,
        help='Directory to save heatmap visualizations (optional)'
    )

    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top strategies to display (default: 10)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed logging'
    )

    return parser.parse_args()


def main():
    """Run optimization."""
    args = parse_arguments()

    # Set up logging
    setup_logging(args.quiet)
    logger = logging.getLogger(__name__)

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

    # Get portfolio_id from config
    portfolio_id = config.get('portfolio_id')
    if not portfolio_id:
        logger.warning("No portfolio_id in config, using default portfolio")

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

    # Auto-detect granularity if not specified
    if args.granularity is None:
        granularity = auto_detect_granularity(args.interval_min)
        logger.info(f"Auto-detected granularity: {granularity} (based on {args.interval_min}min interval)")
    else:
        granularity = args.granularity
        logger.info(f"Using specified granularity: {granularity}")

    print("\n" + "="*100)
    print("STRATEGY PARAMETER OPTIMIZATION")
    print("="*100)
    print(f"\nSimulation Period: {start_date.date()} to {end_date.date()}")
    print(f"Target Allocation: {target_allocation}")
    print(f"\nOptimization Ranges:")
    print(f"  Threshold: {args.threshold_min}% to {args.threshold_max}% (step: {args.threshold_step}%)")
    print(f"  Interval: {args.interval_min} to {args.interval_max} minutes (step: {args.interval_step} min)")
    print(f"  Granularity: {granularity}")

    # Calculate total combinations
    num_thresholds = int((args.threshold_max - args.threshold_min) / args.threshold_step) + 1
    num_intervals = int((args.interval_max - args.interval_min) / args.interval_step) + 1
    total_combinations = num_thresholds * num_intervals

    print(f"\nTotal Combinations: {total_combinations} ({num_thresholds} thresholds × {num_intervals} intervals)")
    print(f"Parallel Workers: {args.workers or 'auto (CPU count)'}")
    print("="*100 + "\n")

    # Validation: Check if granularity is too coarse for the interval
    granularity_minutes = {
        'ONE_MINUTE': 1, 'FIVE_MINUTE': 5, 'FIFTEEN_MINUTE': 15, 'THIRTY_MINUTE': 30,
        'ONE_HOUR': 60, 'TWO_HOUR': 120, 'SIX_HOUR': 360, 'ONE_DAY': 1440
    }

    if granularity in granularity_minutes and args.interval_min < granularity_minutes[granularity]:
        logger.warning(f"Granularity ({granularity}) is coarser than minimum interval ({args.interval_min}min)")
        logger.warning(f"This will result in missing {int(granularity_minutes[granularity] / args.interval_min)} out of every {int(granularity_minutes[granularity] / args.interval_min)} rebalancing decision points")
        logger.warning(f"Consider using finer granularity or larger intervals for accurate results")

    # Initialize Coinbase client
    logger.info("Initializing Coinbase client...")
    try:
        client = CoinbaseClient(portfolio_id=portfolio_id)
    except Exception as e:
        logger.error(f"Failed to initialize Coinbase client: {e}")
        sys.exit(1)

    # Fetch historical price data (once, shared across all simulations)
    use_cache = not args.no_cache
    if args.cache_only and args.no_cache:
        logger.error("Cannot specify both --no-cache and --cache-only")
        sys.exit(1)

    cache_mode = "disabled" if args.no_cache else ("cache-only" if args.cache_only else "enabled")
    logger.info(f"Fetching historical price data (granularity: {granularity}, cache: {cache_mode})...")

    fetcher = HistoricalPriceFetcher(
        client,
        use_cache=use_cache,
        cache_max_age_days=args.cache_max_age if use_cache else None,
        request_delay=args.request_delay
    )

    try:
        price_data = fetcher.fetch_historical_prices(
            assets=list(target_allocation.keys()),
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

    # Auto-set check interval based on minimum interval
    check_interval_hours = args.interval_min / 60.0

    # Create simulation config
    sim_config = SimulationConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital_usd=args.initial_capital,
        target_allocation=target_allocation,
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

    # Print overall summary
    print_optimization_summary(summary)

    # Print top strategies by different metrics
    print_top_strategies_table(summary, 'total_return_percent', 'Total Return', n=args.top_n)
    print_top_strategies_table(summary, 'sharpe_ratio', 'Sharpe Ratio', n=args.top_n)
    print_top_strategies_table(summary, 'net_return_percent', 'Net Return (After Fees)', n=args.top_n)

    # Show comparison vs baseline
    print_comparison_vs_baseline(summary, n=min(5, args.top_n))

    # Export to CSV if requested
    if args.output:
        # Sort results by total return for CSV
        sorted_results = sorted(
            summary.strategy_results,
            key=lambda r: r.total_return_percent,
            reverse=True
        )
        export_to_csv(sorted_results, args.output)
        print(f"\n✓ Results exported to: {args.output}")

    # Generate heatmaps if requested
    if args.heatmap:
        generate_heatmaps(summary, args.heatmap)
        print(f"✓ Heatmaps saved to: {args.heatmap}/")

    print("\n" + "="*100)
    print("OPTIMIZATION COMPLETE")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
