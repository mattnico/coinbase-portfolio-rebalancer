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
    - Memory-efficient: Keeps only summary metrics + 1000 sampled results for visualization
    - Memory usage: ~50-100 MB regardless of simulation count (previously scaled linearly)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
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
    aggregate_simulation_results,
    aggregate_from_metrics,
    load_monte_carlo_results,
    OriginalResult
)
from src.adaptive_strategy import AdaptiveRebalancingStrategy
from src.regime_detector import MarketRegime, ReturnDetector
from src.monte_carlo_visualization import generate_and_save_chart

# Global variables for worker processes (shared via initializer)
_worker_price_data: Optional[Dict[str, List[Tuple[datetime, float]]]] = None
_worker_strategy: Optional[RebalancingStrategy] = None
_worker_target_allocation: Optional[Dict[str, float]] = None
_worker_start_date: Optional[datetime] = None
_worker_end_date: Optional[datetime] = None
_worker_initial_capital: Optional[float] = None
_worker_fee_rate: Optional[float] = None


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def _init_worker(
    price_data: Dict[str, List[Tuple[datetime, float]]],
    strategy: RebalancingStrategy,
    target_allocation: Dict[str, float],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float,
    fee_rate: float
):
    """Initialize worker process with shared data (called once per worker)."""
    global _worker_price_data, _worker_strategy, _worker_target_allocation
    global _worker_start_date, _worker_end_date, _worker_initial_capital, _worker_fee_rate

    _worker_price_data = price_data
    _worker_strategy = strategy
    _worker_target_allocation = target_allocation
    _worker_start_date = start_date
    _worker_end_date = end_date
    _worker_initial_capital = initial_capital
    _worker_fee_rate = fee_rate

    # Suppress verbose logging in worker processes
    logging.getLogger('src.monte_carlo_chunk_shuffle').setLevel(logging.WARNING)
    logging.getLogger('src.monte_carlo_simulator').setLevel(logging.WARNING)


def _run_single_monte_carlo_simulation(
    sim_index: int,
    chunk_days: int,
    base_seed: int,
    keep_full_result: bool = False
):
    """
    Run a single Monte Carlo simulation using worker globals.

    This function is designed to be pickable for multiprocessing.
    Data is passed via globals set by _init_worker() to avoid repeated pickling.

    Args:
        sim_index: Simulation index (for uniqueness)
        chunk_days: Size of chunks to shuffle in days
        base_seed: Base random seed (sim_index will be added for uniqueness)
        keep_full_result: If True, return full SimulationResult; else return metrics only

    Returns:
        Either full SimulationResult or dict of metrics (memory-efficient)
    """
    # Access globals set by initializer
    price_data = _worker_price_data
    strategy = _worker_strategy
    target_allocation = _worker_target_allocation
    start_date = _worker_start_date
    end_date = _worker_end_date
    initial_capital = _worker_initial_capital
    fee_rate = _worker_fee_rate

    # Create a unique seed for this simulation
    sim_seed = base_seed + sim_index

    # Create shuffler for this specific simulation
    mc_config = MonteCarloConfig(
        chunk_days=chunk_days,
        num_simulations=1,
        preserve_start_chunk=True,
        seed=sim_seed
    )
    shuffler = ChunkShuffler(mc_config)

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

    # Memory optimization: Return only lightweight metrics unless full result requested
    if keep_full_result:
        return result
    else:
        # Extract only what we need for aggregation (tiny memory footprint)
        return {
            'total_return_percent': result.total_return_percent,
            'final_value': result.final_value,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown_percent': result.max_drawdown_percent,
            'num_rebalances': result.num_rebalances,
            'total_fees_paid': result.total_fees_paid
        }


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
    show_progress: bool = True,
    max_full_results_for_viz: int = 1000
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
        max_full_results_for_viz: Max number of full results to keep for visualization (default: 1000)

    Returns:
        Tuple of (MonteCarloResult, original_result)
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

    # Memory-efficient: Keep only summary metrics + sample of full results
    summary_metrics = {
        'returns': [],
        'final_values': [],
        'sharpe_ratios': [],
        'max_drawdowns': [],
        'num_rebalances': [],
        'total_fees': []
    }
    full_results_for_viz = []  # Keep small sample for visualization

    # Determine sampling rate for visualization
    viz_sample_rate = max(1, num_simulations // max_full_results_for_viz)

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

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(price_data, strategy, target_allocation, start_date, end_date, initial_capital, fee_rate)
    ) as executor:
        # Submit all jobs - only request full results for sampled indices
        # Data is passed via initializer to avoid repeated pickling
        futures = {
            executor.submit(
                _run_single_monte_carlo_simulation,
                i,
                chunk_days,
                shuffler.seed,  # Pass base seed, worker will add sim_index for uniqueness
                keep_full_result=(i % viz_sample_rate == 0 and i // viz_sample_rate < max_full_results_for_viz)
            ): i
            for i in range(num_simulations)
        }

        # Collect results as they complete
        for future in as_completed(futures):
            sim_index = futures[future]
            try:
                result = future.result()

                # Result is either dict (metrics only) or SimulationResult (full)
                if isinstance(result, dict):
                    # Lightweight metrics dict
                    metrics = result

                    # Debug: Check for extreme returns
                    if metrics['total_return_percent'] > 100000:
                        logger.warning(f"⚠️  Simulation {sim_index+1} has extreme return: {metrics['total_return_percent']:.2f}%")
                        logger.warning(f"   Final: ${metrics['final_value']:.2f}")
                        logger.warning(f"   Rebalances: {metrics['num_rebalances']}, Fees: ${metrics['total_fees_paid']:.2f}")

                    # Store metrics
                    summary_metrics['returns'].append(metrics['total_return_percent'])
                    summary_metrics['final_values'].append(metrics['final_value'])
                    summary_metrics['sharpe_ratios'].append(metrics['sharpe_ratio'])
                    summary_metrics['max_drawdowns'].append(metrics['max_drawdown_percent'])
                    summary_metrics['num_rebalances'].append(metrics['num_rebalances'])
                    summary_metrics['total_fees'].append(metrics['total_fees_paid'])
                else:
                    # Full SimulationResult (sampled for visualization)
                    # Debug: Check for extreme returns
                    if result.total_return_percent > 100000:
                        logger.warning(f"⚠️  Simulation {sim_index+1} has extreme return: {result.total_return_percent:.2f}%")
                        logger.warning(f"   Initial: ${result.initial_value:.2f}, Final: ${result.final_value:.2f}")
                        logger.warning(f"   Rebalances: {result.num_rebalances}, Fees: ${result.total_fees_paid:.2f}")

                    # Extract summary metrics
                    summary_metrics['returns'].append(result.total_return_percent)
                    summary_metrics['final_values'].append(result.final_value)
                    summary_metrics['sharpe_ratios'].append(result.sharpe_ratio)
                    summary_metrics['max_drawdowns'].append(result.max_drawdown_percent)
                    summary_metrics['num_rebalances'].append(result.num_rebalances)
                    summary_metrics['total_fees'].append(result.total_fees_paid)

                    # Keep for visualization
                    full_results_for_viz.append(result)

                if progress_bar:
                    progress_bar.update(1)

            except Exception as e:
                logger.error(f"Error in simulation {sim_index+1}: {e}", exc_info=True)

    if progress_bar:
        progress_bar.close()

    elapsed_time = time.time() - start_time_sims
    sims_per_sec = num_simulations / elapsed_time if elapsed_time > 0 else 0
    logger.info(f"\n✅ Completed {num_simulations} simulations in {elapsed_time:.1f}s ({sims_per_sec:.1f} sims/sec)")
    logger.info(f"   Kept {len(full_results_for_viz)} full results for visualization (sampled every {viz_sample_rate} simulations)")

    # Aggregate results (memory-efficient version)
    mc_result = aggregate_from_metrics(
        metrics=summary_metrics,
        strategy_name=strategy.name,
        chunk_days=chunk_days,
        seed=shuffler.seed
    )

    # Generate equity curve visualization using sampled results
    logger.info("Generating equity curve visualization...")
    try:
        chart_path = generate_and_save_chart(
            mc_results=full_results_for_viz,
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
            logger.info(f"   (based on {len(full_results_for_viz)} sampled simulations)")
    except Exception as e:
        logger.warning(f"Could not generate visualization: {e}")

    # Auto-save results to JSON
    logger.info("Saving results to JSON...")
    try:
        # Create filename based on strategy, dates, and simulations
        strategy_clean = strategy.name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('%', 'pct')
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        json_filename = f"mc_{strategy_clean}_{start_str}_{end_str}_{num_simulations}sims_seed{shuffler.seed}.json"
        json_path = Path("results") / json_filename

        mc_result.save_to_json(str(json_path), original_result=original_result)
        logger.info(f"Results saved to: {json_path}")
    except Exception as e:
        logger.warning(f"Could not save results to JSON: {e}")

    return mc_result, original_result


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Monte Carlo simulations with chunk-shuffled historical data'
    )

    # Time period (not required if loading results)
    time_group = parser.add_mutually_exclusive_group(required=False)
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

    parser.add_argument(
        '--load-results',
        type=str,
        help='Load and display previously saved Monte Carlo results from JSON file'
    )

    return parser.parse_args()


def main():
    """Run Monte Carlo simulations."""
    args = parse_arguments()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Handle --load-results mode
    if args.load_results:
        logger.info(f"Loading results from {args.load_results}")
        try:
            mc_result, original_result = load_monte_carlo_results(args.load_results)
            mc_result.print_summary(original_result=original_result)
            print()
            logger.info("Results displayed successfully")
            return
        except FileNotFoundError:
            logger.error(f"Results file not found: {args.load_results}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading results: {e}", exc_info=True)
            sys.exit(1)

    # Validate that either --load-results or time period is provided
    if not args.days and not args.start:
        logger.error("Either --days/--start or --load-results must be provided")
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
        mc_result, original_result = run_monte_carlo(
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

        # Print results with original comparison
        mc_result.print_summary(original_result=original_result)

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
            mc_result, original_result = run_monte_carlo(
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

            # Print results with original comparison
            mc_result.print_summary(original_result=original_result)

    # Comparison summary if multiple strategies
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("STRATEGY COMPARISON SUMMARY (Median-Focused)")
        print("=" * 80)
        print(f"{'Strategy':<20} {'Median Return':<15} {'Median Sharpe':<14} {'Median DD':<12} {'Worst DD':<12}")
        print("-" * 80)

        for name, result in all_results.items():
            print(f"{name:<20} {result.median_return:>13.2f}% "
                  f"{result.median_sharpe:>12.3f} "
                  f"{result.median_max_drawdown:>10.2f}% "
                  f"{result.worst_max_drawdown:>10.2f}%")

        print("=" * 80)

        # Find winner by median return
        winner_name = max(all_results.items(), key=lambda x: x[1].median_return)[0]
        winner = all_results[winner_name]
        print(f"\n🏆 Winner (by median return): {winner_name}")
        print(f"   Median Return:      {winner.median_return:>10.2f}%")
        print(f"   Median Sharpe:      {winner.median_sharpe:>10.3f}")
        print(f"   Median Max DD:      {winner.median_max_drawdown:>10.2f}%")
        print(f"   Worst Drawdown:     {winner.worst_max_drawdown:>10.2f}%")
        print()


if __name__ == '__main__':
    main()
