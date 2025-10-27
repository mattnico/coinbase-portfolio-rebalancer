#!/usr/bin/env python3
"""
CLI tool for running rebalancing strategy simulations.

Usage:
    python -m src.simulate_strategies --days 90
    python -m src.simulate_strategies --start 2024-01-01 --end 2024-03-31
    python -m src.simulate_strategies --days 180 --interval 7 --threshold 2.5
    python -m src.simulate_strategies --config config/portfolio.json --days 60
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

from src.coinbase_client import CoinbaseClient
from src.monte_carlo_simulator import (
    SimulationConfig,
    HistoricalPriceFetcher,
    PortfolioSimulator,
    BuyAndHoldStrategy,
    HybridStrategy,
    print_simulation_report,
    save_simulation_results,
)


def setup_logging():
    """Configure logging for simulation."""
    logging.basicConfig(
        level=logging.INFO,
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


def auto_detect_granularity(interval_minutes: float) -> str:
    """
    Auto-detect appropriate granularity based on rebalancing interval.

    Args:
        interval_minutes: Rebalancing interval in minutes

    Returns:
        Recommended granularity string
    """
    if interval_minutes <= 1:
        return 'ONE_MINUTE'
    elif interval_minutes <= 5:
        return 'FIVE_MINUTE'
    elif interval_minutes <= 15:
        return 'FIFTEEN_MINUTE'
    elif interval_minutes <= 30:
        return 'THIRTY_MINUTE'
    elif interval_minutes <= 60:
        return 'ONE_HOUR'
    elif interval_minutes <= 120:
        return 'TWO_HOUR'
    elif interval_minutes <= 360:
        return 'SIX_HOUR'
    else:
        return 'ONE_DAY'


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Simulate and compare rebalancing strategies using historical data'
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

    # Strategy parameters - interval options (mutually exclusive)
    interval_group = parser.add_mutually_exclusive_group()
    interval_group.add_argument(
        '--interval-days',
        type=int,
        help='Rebalance check interval in days (e.g., 7 for weekly)'
    )
    interval_group.add_argument(
        '--interval-hours',
        type=int,
        help='Rebalance check interval in hours (e.g., 6 for every 6 hours)'
    )
    interval_group.add_argument(
        '--interval-minutes',
        type=int,
        help='Rebalance check interval in minutes (e.g., 5 for every 5 minutes)'
    )

    # Keep legacy --interval for backward compatibility (treated as days)
    interval_group.add_argument(
        '--interval',
        type=int,
        help='Rebalance check interval in days (legacy, use --interval-days instead)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=2.5,
        help='Deviation threshold percentage for hybrid strategy (default: 2.5)'
    )

    parser.add_argument(
        '--fee-rate',
        type=float,
        default=0.006,
        help='Trading fee rate as decimal (default: 0.006 = 0.6%%)'
    )

    # Configuration
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
        help='Price data granularity (default: auto-detect based on interval)'
    )

    parser.add_argument(
        '--check-interval-hours',
        type=float,
        default=None,
        help='How often to check for rebalancing in hours (default: auto-set based on rebalance interval)'
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

    # Output options
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file (optional)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed logging'
    )

    return parser.parse_args()


def main():
    """Run simulation."""
    args = parse_arguments()

    # Set up logging
    if args.quiet:
        logging.basicConfig(level=logging.WARNING)
    else:
        setup_logging()

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

    logger.info(f"Simulation period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Target allocation: {target_allocation}")

    # Determine rebalancing interval from args FIRST (needed for auto-detecting granularity)
    # Priority: minutes > hours > days > legacy interval (default 7 days)
    if args.interval_minutes:
        interval_minutes = args.interval_minutes
    elif args.interval_hours:
        interval_minutes = args.interval_hours * 60
    elif args.interval_days:
        interval_minutes = args.interval_days * 1440
    elif args.interval:
        interval_minutes = args.interval * 1440  # Legacy, treat as days
    else:
        interval_minutes = 7 * 1440  # Default 7 days

    # Auto-detect granularity if not specified
    if args.granularity is None:
        granularity = auto_detect_granularity(interval_minutes)
        logger.info(f"Auto-detected granularity: {granularity} (based on {interval_minutes}min interval)")
    else:
        granularity = args.granularity
        logger.info(f"Using specified granularity: {granularity}")

    # Initialize Coinbase client
    logger.info("Initializing Coinbase client...")
    try:
        client = CoinbaseClient(portfolio_id=portfolio_id)
    except Exception as e:
        logger.error(f"Failed to initialize Coinbase client: {e}")
        sys.exit(1)

    # Fetch historical price data
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

    # Build strategy configuration based on interval arguments
    if args.interval_minutes:
        hybrid_kwargs = {
            'rebalance_interval_minutes': args.interval_minutes,
            'threshold_percent': args.threshold
        }
        check_interval_hours = args.check_interval_hours or (args.interval_minutes / 60.0)
    elif args.interval_hours:
        hybrid_kwargs = {
            'rebalance_interval_hours': args.interval_hours,
            'threshold_percent': args.threshold
        }
        check_interval_hours = args.check_interval_hours or args.interval_hours
    elif args.interval_days:
        hybrid_kwargs = {
            'rebalance_interval_days': args.interval_days,
            'threshold_percent': args.threshold
        }
        check_interval_hours = args.check_interval_hours or (args.interval_days * 24)
    elif args.interval:
        hybrid_kwargs = {
            'rebalance_interval_days': args.interval,
            'threshold_percent': args.threshold
        }
        check_interval_hours = args.check_interval_hours or (args.interval * 24)
    else:
        hybrid_kwargs = {
            'rebalance_interval_days': 7,
            'threshold_percent': args.threshold
        }
        check_interval_hours = args.check_interval_hours or 24

    logger.info(f"Check interval: every {check_interval_hours} hours")

    # Create simulation config with determined check interval
    sim_config = SimulationConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital_usd=args.initial_capital,
        target_allocation=target_allocation,
        fee_rate=args.fee_rate,
        price_check_interval_hours=int(check_interval_hours) if check_interval_hours >= 1 else check_interval_hours
    )

    # Define strategies to compare
    strategies = [
        BuyAndHoldStrategy(),
        HybridStrategy(**hybrid_kwargs)
    ]

    # Run simulations
    results = []
    for strategy in strategies:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running simulation: {strategy.name}")
        logger.info(f"{'='*60}")

        simulator = PortfolioSimulator(
            config=sim_config,
            strategy=strategy,
            price_data=price_data
        )

        result = simulator.run()
        results.append(result)

        logger.info(f"Simulation complete for {strategy.name}")

    # Print comparison report
    print_simulation_report(results)

    # Save to file if requested
    if args.output:
        save_simulation_results(results, args.output)
        print(f"\nResults saved to {args.output}")

    # Print key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    hybrid = next(r for r in results if 'Hybrid' in r.strategy_name)
    buy_hold = next(r for r in results if 'Buy and Hold' in r.strategy_name)

    return_diff = hybrid.total_return_percent - buy_hold.total_return_percent
    if return_diff > 0:
        print(f"✓ Hybrid strategy OUTPERFORMED by {return_diff:.2f}%")
    else:
        print(f"✗ Hybrid strategy UNDERPERFORMED by {abs(return_diff):.2f}%")

    print(f"✓ Hybrid strategy traded {hybrid.num_rebalances} times, paying ${hybrid.total_fees_paid:.2f} in fees")
    print(f"✓ Buy and Hold avoided {hybrid.num_rebalances} rebalances and saved ${hybrid.total_fees_paid:.2f} in fees")

    # Net performance after fees
    net_diff = return_diff
    print(f"\nNet performance difference (after fees): {net_diff:+.2f}%")

    if hybrid.sharpe_ratio > buy_hold.sharpe_ratio:
        print(f"✓ Hybrid had better risk-adjusted returns (Sharpe: {hybrid.sharpe_ratio:.3f} vs {buy_hold.sharpe_ratio:.3f})")
    else:
        print(f"✗ Buy and Hold had better risk-adjusted returns (Sharpe: {buy_hold.sharpe_ratio:.3f} vs {hybrid.sharpe_ratio:.3f})")

    print("="*80)


if __name__ == '__main__':
    main()
