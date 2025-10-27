#!/usr/bin/env python3
"""
Run Monte Carlo simulations with chunk-shuffled historical data.

This script runs multiple simulations by shuffling historical price data in chunks,
testing strategy robustness to different sequences of market conditions.

Usage:
    python -m src.run_monte_carlo --days 1825 --simulations 1000 --chunk-days 30
    python -m src.run_monte_carlo --days 1825 --simulations 100 --compare-strategies
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from src.coinbase_client import CoinbaseClient
from src.monte_carlo_simulator import (
    SimulationConfig,
    PortfolioSimulator,
    RebalancingStrategy,
    HybridStrategy,
    HistoricalPriceFetcher
)
from src.monte_carlo_chunk_shuffle import (
    ChunkShuffler,
    MonteCarloConfig,
    aggregate_simulation_results
)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def run_monte_carlo(
    price_data: Dict[str, List[Tuple[datetime, float]]],
    strategy: RebalancingStrategy,
    target_allocation: Dict[str, float],
    initial_capital: float,
    num_simulations: int,
    chunk_days: int,
    fee_rate: float = 0.006,
    seed: int = None
):
    """
    Run Monte Carlo simulations with shuffled historical data.

    Args:
        price_data: Historical price data
        strategy: Rebalancing strategy to test
        target_allocation: Target portfolio allocation
        initial_capital: Starting capital
        num_simulations: Number of shuffled simulations to run
        chunk_days: Size of chunks to shuffle
        fee_rate: Trading fee rate
        seed: Random seed for reproducibility

    Returns:
        MonteCarloResult with aggregated statistics
    """
    logger = logging.getLogger(__name__)

    # Create chunk shuffler
    mc_config = MonteCarloConfig(
        chunk_days=chunk_days,
        num_simulations=num_simulations,
        preserve_start_chunk=True,
        seed=seed
    )
    shuffler = ChunkShuffler(mc_config)

    # Determine date range from original data
    all_timestamps = [ts for prices in price_data.values() for ts, _ in prices]
    start_date = min(all_timestamps)
    end_date = max(all_timestamps)

    logger.info(f"\n{'='*80}")
    logger.info(f"STARTING MONTE CARLO SIMULATION")
    logger.info(f"{'='*80}")
    logger.info(f"Strategy: {strategy.name}")
    logger.info(f"Simulations: {num_simulations}")
    logger.info(f"Chunk Size: {chunk_days} days")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Initial Capital: ${initial_capital:,.2f}")
    logger.info(f"Target Allocation: {target_allocation}")
    logger.info(f"{'='*80}\n")

    # Run simulations
    results = []
    for i in range(num_simulations):
        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"Running simulation {i+1}/{num_simulations}...")

        # Generate shuffled timeline
        shuffled_data = shuffler.generate_shuffled_timeline(
            price_data,
            new_start_date=start_date
        )

        # Create simulation config
        sim_config = SimulationConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital_usd=initial_capital,
            target_allocation=target_allocation,
            fee_rate=fee_rate
        )

        # Run simulation
        simulator = PortfolioSimulator(
            config=sim_config,
            strategy=strategy,
            price_data=shuffled_data
        )

        result = simulator.run()
        results.append(result)

    logger.info(f"\n✅ Completed {num_simulations} simulations")

    # Aggregate results
    mc_result = aggregate_simulation_results(
        results,
        strategy_name=strategy.name,
        chunk_days=chunk_days
    )

    return mc_result


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Monte Carlo simulations with chunk-shuffled historical data'
    )

    # Time period
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

    # Monte Carlo parameters
    parser.add_argument(
        '--simulations',
        type=int,
        default=100,
        help='Number of Monte Carlo simulations (default: 100)'
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

    # Portfolio parameters
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=10000.0,
        help='Initial capital in USD (default: 10000)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=5.0,
        help='Rebalancing threshold percentage (default: 5.0)'
    )

    parser.add_argument(
        '--fee-rate',
        type=float,
        default=0.006,
        help='Trading fee rate (default: 0.006 = 0.6%%)'
    )

    # Strategy comparison
    parser.add_argument(
        '--compare-strategies',
        action='store_true',
        help='Compare multiple portfolio strategies'
    )

    # Cache options
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Skip cache'
    )

    parser.add_argument(
        '--cache-only',
        action='store_true',
        help='Only use cached data'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main():
    """Run Monte Carlo simulations."""
    args = parse_arguments()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

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

    # Initialize Coinbase client and fetch data
    logger.info("Initializing Coinbase client...")
    try:
        client = CoinbaseClient()
    except Exception as e:
        logger.error(f"Failed to initialize client: {e}")
        sys.exit(1)

    # Define portfolios to test
    portfolios = {
        'Top_Three': {
            'BTC': 40.0,
            'ETH': 40.0,
            'SOL': 20.0
        },
        'Aggressive': {
            'BTC': 30.0,
            'ETH': 30.0,
            'SOL': 25.0,
            'AVAX': 15.0
        },
        'Conservative': {
            'BTC': 30.0,
            'ETH': 30.0,
            'SOL': 15.0,
            'USDC': 25.0
        }
    }

    # Determine which portfolios to test
    if args.compare_strategies:
        portfolios_to_test = portfolios
    else:
        # Just test Top_Three
        portfolios_to_test = {'Top_Three': portfolios['Top_Three']}

    # Fetch historical price data for all assets
    all_assets = set()
    for portfolio in portfolios_to_test.values():
        all_assets.update(portfolio.keys())

    logger.info(f"Fetching historical data for {len(all_assets)} assets: {sorted(all_assets)}")

    fetcher = HistoricalPriceFetcher(
        client,
        use_cache=not args.no_cache,
        cache_max_age_days=7 if not args.no_cache else None
    )

    try:
        price_data = fetcher.fetch_historical_prices(
            assets=list(all_assets),
            start_date=start_date,
            end_date=end_date,
            granularity='ONE_DAY',
            show_progress=not args.verbose,
            cache_only=args.cache_only
        )
    except Exception as e:
        logger.error(f"Failed to fetch price data: {e}")
        sys.exit(1)

    # Verify we have data for all assets
    missing_assets = all_assets - set(price_data.keys())
    if missing_assets:
        logger.error(f"Missing price data for assets: {missing_assets}")
        sys.exit(1)

    logger.info(f"Loaded price data for {len(price_data)} assets")

    # Run Monte Carlo for each portfolio
    all_results = {}

    for portfolio_name, target_allocation in portfolios_to_test.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"TESTING PORTFOLIO: {portfolio_name}")
        logger.info(f"{'='*80}\n")

        # Filter price data to only include assets in this portfolio
        filtered_price_data = {
            asset: prices for asset, prices in price_data.items()
            if asset in target_allocation
        }

        # Create rebalancing strategy (check weekly, rebalance if threshold exceeded)
        strategy = HybridStrategy(
            rebalance_interval_days=7,
            threshold_percent=args.threshold
        )

        # Run Monte Carlo
        mc_result = run_monte_carlo(
            price_data=filtered_price_data,
            strategy=strategy,
            target_allocation=target_allocation,
            initial_capital=args.initial_capital,
            num_simulations=args.simulations,
            chunk_days=args.chunk_days,
            fee_rate=args.fee_rate,
            seed=args.seed
        )

        all_results[portfolio_name] = mc_result

        # Print results
        mc_result.print_summary()

    # Comparison summary if multiple strategies
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("STRATEGY COMPARISON SUMMARY")
        print("=" * 80)
        print(f"{'Strategy':<20} {'Mean Return':<15} {'Median Return':<15} {'Mean Sharpe':<12} {'Worst DD':<12}")
        print("-" * 80)

        for name, result in all_results.items():
            print(f"{name:<20} {result.mean_return:>13.2f}% "
                  f"{result.median_return:>13.2f}% "
                  f"{result.mean_sharpe:>10.3f} "
                  f"{result.worst_max_drawdown:>10.2f}%")

        print("=" * 80)

        # Find winner
        winner_name = max(all_results.items(), key=lambda x: x[1].median_return)[0]
        print(f"\n🏆 Winner (by median return): {winner_name}")
        print(f"   Median Return: {all_results[winner_name].median_return:.2f}%")
        print(f"   Mean Sharpe: {all_results[winner_name].mean_sharpe:.3f}")
        print()


if __name__ == '__main__':
    main()
