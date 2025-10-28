#!/usr/bin/env python3
"""
Run Monte Carlo simulations with chunk-shuffled historical data.

This script runs multiple simulations by shuffling historical price data in chunks,
testing strategy robustness to different sequences of market conditions.

Usage:
    # Run 1000 simulations using all CPU cores
    python -m src.run_monte_carlo --days 1825 --simulations 1000 --chunk-days 30

    # Run 100 simulations using 8 workers
    python -m src.run_monte_carlo --days 1825 --simulations 100 --workers 8

    # Run with progress bar and custom seed
    python -m src.run_monte_carlo --days 365 --simulations 10000 --seed 42

    # Compare strategies
    python -m src.run_monte_carlo --days 1825 --simulations 100 --compare-strategies

Performance Notes:
    - Parallelization provides ~3-4x speedup on 8-core machines
    - For 150k simulations: ~6-8 minutes with 12 cores vs ~27 minutes single-threaded
    - Memory usage: ~50-100 MB per 1000 simulations (plan for 5-10 GB for 100k sims)
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

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
from src.adaptive_strategy import AdaptiveRebalancingStrategy
from src.regime_detector import MarketRegime, ReturnDetector
from src.monte_carlo_visualization import generate_and_save_chart


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def _run_single_monte_carlo_simulation(
    sim_index: int,
    price_data: Dict[str, List[Tuple[datetime, float]]],
    shuffler: ChunkShuffler,
    strategy: RebalancingStrategy,
    target_allocation: Dict[str, float],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float,
    fee_rate: float
):
    """
    Run a single Monte Carlo simulation.

    This function is designed to be pickable for multiprocessing.

    Args:
        sim_index: Simulation index (for logging)
        price_data: Original price data to shuffle
        shuffler: ChunkShuffler instance
        strategy: Rebalancing strategy
        target_allocation: Target portfolio allocation
        start_date: Simulation start date
        end_date: Simulation end date
        initial_capital: Initial capital in USD
        fee_rate: Trading fee rate

    Returns:
        SimulationResult from the simulation
    """
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

    return result


def run_monte_carlo(
    price_data: Dict[str, List[Tuple[datetime, float]]],
    strategy: RebalancingStrategy,
    target_allocation: Dict[str, float],
    initial_capital: float,
    num_simulations: int,
    chunk_days: int,
    fee_rate: float = 0.006,
    seed: int = None,
    max_workers: int = None,
    show_progress: bool = True
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
        max_workers: Number of parallel workers (default: CPU count)
        show_progress: Whether to show progress bar

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
    logger.info(f"Random Seed: {shuffler.seed} (use --seed {shuffler.seed} to reproduce)")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Initial Capital: ${initial_capital:,.2f}")
    logger.info(f"Target Allocation: {target_allocation}")
    logger.info(f"{'='*80}\n")

    # First, run one simulation with original (unshuffled) historical data
    logger.info("Running original (unshuffled) historical simulation...")
    original_sim_config = SimulationConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital_usd=initial_capital,
        target_allocation=target_allocation,
        fee_rate=fee_rate
    )
    original_simulator = PortfolioSimulator(
        config=original_sim_config,
        strategy=strategy,
        price_data=price_data  # Use original unshuffled data
    )
    original_result = original_simulator.run()
    logger.info(f"Original strategy return: {original_result.total_return_percent:.2f}%\n")

    # Run Monte Carlo simulations with shuffled data in parallel
    cpu_count = os.cpu_count() or 1
    workers_to_use = max_workers if max_workers else cpu_count
    logger.info(f"Running {num_simulations} simulations using {workers_to_use} workers (system has {cpu_count} CPUs)...")

    start_time_sims = time.time()
    results = []

    # Initialize progress bar if enabled
    if show_progress:
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=num_simulations, desc="Monte Carlo", unit="sim")
        except ImportError:
            logger.warning("tqdm not installed, progress bar disabled. Install with: pip install tqdm")
            progress_bar = None
    else:
        progress_bar = None

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        futures = {
            executor.submit(
                _run_single_monte_carlo_simulation,
                i,
                price_data,
                shuffler,
                strategy,
                target_allocation,
                start_date,
                end_date,
                initial_capital,
                fee_rate
            ): i
            for i in range(num_simulations)
        }

        # Collect results as they complete
        for future in as_completed(futures):
            sim_index = futures[future]
            try:
                result = future.result()

                # Debug: Check for extreme returns
                if result.total_return_percent > 100000:  # More than 1000x return
                    logger.warning(f"⚠️  Simulation {sim_index+1} has extreme return: {result.total_return_percent:.2f}%")
                    logger.warning(f"   Initial: ${result.initial_value:.2f}, Final: ${result.final_value:.2f}")
                    logger.warning(f"   Rebalances: {result.num_rebalances}, Fees: ${result.total_fees_paid:.2f}")

                results.append(result)

                if progress_bar:
                    progress_bar.update(1)

            except Exception as e:
                logger.error(f"Error in simulation {sim_index+1}: {e}", exc_info=True)

    if progress_bar:
        progress_bar.close()

    elapsed_time = time.time() - start_time_sims
    sims_per_sec = num_simulations / elapsed_time if elapsed_time > 0 else 0
    logger.info(f"\n✅ Completed {num_simulations} simulations in {elapsed_time:.1f}s ({sims_per_sec:.1f} sims/sec)")

    # Aggregate results
    mc_result = aggregate_simulation_results(
        results,
        strategy_name=strategy.name,
        chunk_days=chunk_days,
        seed=shuffler.seed
    )

    # Generate equity curve visualization
    logger.info("Generating equity curve visualization...")
    try:
        chart_path = generate_and_save_chart(
            mc_results=results,
            original_result=original_result,
            strategy_name=strategy.name,
            start_date=start_date,
            end_date=end_date,
            num_simulations=num_simulations,
            chunk_days=chunk_days,
            seed=shuffler.seed
        )
        if chart_path:
            logger.info(f"Visualization saved to: {chart_path}")
    except Exception as e:
        logger.warning(f"Could not generate visualization: {e}")

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

    # Strategy options
    parser.add_argument(
        '--adaptive',
        action='store_true',
        help='Use adaptive regime-based strategy'
    )

    parser.add_argument(
        '--compare-strategies',
        action='store_true',
        help='Compare multiple portfolio strategies'
    )

    parser.add_argument(
        '--compare-adaptive-vs-static',
        action='store_true',
        help='Compare adaptive strategy vs static allocation'
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

    parser.add_argument(
        '--workers',
        type=int,
        help='Number of parallel workers (default: CPU count)'
    )

    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar'
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

    # Define regime-based portfolios for adaptive strategy
    regime_portfolios = {
        MarketRegime.BULL: {
            'BTC': 40.0,
            'ETH': 40.0,
            'SOL': 20.0
        },
        MarketRegime.BEAR: {
            'BTC': 20.0,
            'ETH': 20.0,
            'SOL': 10.0,
            'USDC': 40.0,
            'DOGE': 5.0,
            'AVAX': 5.0
        },
        MarketRegime.NEUTRAL: {
            'BTC': 40.0,
            'ETH': 40.0,
            'SOL': 20.0
        }
    }

    # Determine which portfolios to test
    if args.compare_strategies:
        portfolios_to_test = portfolios
    elif args.compare_adaptive_vs_static:
        portfolios_to_test = {'Top_Three': portfolios['Top_Three']}
    else:
        # Just test Top_Three or adaptive
        portfolios_to_test = {'Top_Three': portfolios['Top_Three']}

    # Fetch historical price data for all assets
    all_assets = set()
    for portfolio in portfolios_to_test.values():
        all_assets.update(portfolio.keys())

    # Add assets needed for adaptive strategy if requested
    if args.adaptive or args.compare_adaptive_vs_static:
        for regime_allocation in regime_portfolios.values():
            all_assets.update(regime_allocation.keys())

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

    # Test adaptive strategy if requested
    if args.adaptive or args.compare_adaptive_vs_static:
        logger.info(f"\n{'='*80}")
        logger.info(f"TESTING ADAPTIVE STRATEGY")
        logger.info(f"{'='*80}\n")

        # Get all assets needed for adaptive (union of all regime portfolios)
        all_adaptive_assets = set()
        for regime_allocation in regime_portfolios.values():
            all_adaptive_assets.update(regime_allocation.keys())

        # Filter price data to include all assets needed
        adaptive_price_data = {
            asset: prices for asset, prices in price_data.items()
            if asset in all_adaptive_assets
        }

        # Create adaptive strategy
        detector = ReturnDetector(
            window_days=30,
            bull_threshold=15.0,
            bear_threshold=-10.0
        )

        adaptive_strategy = AdaptiveRebalancingStrategy(
            regime_portfolios=regime_portfolios,
            detector=detector,
            check_frequency_days=7,
            persistence_days=14,
            threshold_percent=args.threshold,
            name="Adaptive (Bear Protection)"
        )

        # Use neutral portfolio as base allocation (doesn't matter, will be dynamic)
        base_allocation = regime_portfolios[MarketRegime.NEUTRAL]

        # Run Monte Carlo
        mc_result = run_monte_carlo(
            price_data=adaptive_price_data,
            strategy=adaptive_strategy,
            target_allocation=base_allocation,
            initial_capital=args.initial_capital,
            num_simulations=args.simulations,
            chunk_days=args.chunk_days,
            fee_rate=args.fee_rate,
            seed=args.seed,
            max_workers=args.workers,
            show_progress=not args.no_progress
        )

        all_results['Adaptive'] = mc_result

        # Print results
        mc_result.print_summary()

    # Test static portfolios if requested
    if not args.adaptive or args.compare_adaptive_vs_static:
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
                seed=args.seed,
                max_workers=args.workers,
                show_progress=not args.no_progress
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
