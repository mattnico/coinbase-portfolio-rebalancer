#!/usr/bin/env python3
"""
Example script showing how to use the Monte Carlo simulator programmatically.

This demonstrates how to run simulations without using the CLI,
which is useful for automation or custom analysis.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.coinbase_client import CoinbaseClient
from src.monte_carlo_simulator import (
    SimulationConfig,
    HistoricalPriceFetcher,
    PortfolioSimulator,
    BuyAndHoldStrategy,
    HybridStrategy,
    print_simulation_report,
)


def main():
    """Run example simulation."""

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Define simulation parameters
    END_DATE = datetime.now()
    START_DATE = END_DATE - timedelta(days=90)
    INITIAL_CAPITAL = 10000.0
    TARGET_ALLOCATION = {
        'BTC': 40.0,
        'ETH': 30.0,
        'SOL': 20.0,
        'AVAX': 10.0,
    }

    logger.info("="*60)
    logger.info("Monte Carlo Simulation Example")
    logger.info("="*60)
    logger.info(f"Period: {START_DATE.date()} to {END_DATE.date()}")
    logger.info(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    logger.info(f"Target Allocation: {TARGET_ALLOCATION}")

    # Initialize Coinbase client
    logger.info("\nInitializing Coinbase client...")
    try:
        client = CoinbaseClient()
    except Exception as e:
        logger.error(f"Failed to initialize client: {e}")
        logger.error("Make sure your API credentials are set in config/.env")
        return

    # Fetch historical price data
    logger.info("\nFetching historical price data...")
    fetcher = HistoricalPriceFetcher(client)

    try:
        price_data = fetcher.fetch_historical_prices(
            assets=list(TARGET_ALLOCATION.keys()),
            start_date=START_DATE,
            end_date=END_DATE,
            granularity="ONE_DAY"
        )
    except Exception as e:
        logger.error(f"Failed to fetch price data: {e}")
        return

    # Validate data
    for asset, prices in price_data.items():
        logger.info(f"{asset}: {len(prices)} data points")
        if not prices:
            logger.error(f"No data for {asset}")
            return

    # Create simulation config
    config = SimulationConfig(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital_usd=INITIAL_CAPITAL,
        target_allocation=TARGET_ALLOCATION,
        fee_rate=0.006,  # 0.6% per trade
        price_check_interval_hours=24
    )

    # Define strategies to test
    strategies = [
        BuyAndHoldStrategy(),
        # Daily rebalancing with 2.5% threshold
        HybridStrategy(rebalance_interval_days=1, threshold_percent=2.5),
        # Weekly rebalancing with 2.5% threshold (traditional approach)
        HybridStrategy(rebalance_interval_days=7, threshold_percent=2.5),
        # Bi-weekly rebalancing with 5% threshold (less frequent)
        HybridStrategy(rebalance_interval_days=14, threshold_percent=5.0),
        # Hourly rebalancing (if using hourly data) - uncomment if testing intra-day
        # HybridStrategy(rebalance_interval_hours=6, threshold_percent=2.5),
        # High-frequency (5-minute intervals) - uncomment if testing minute-level
        # HybridStrategy(rebalance_interval_minutes=5, threshold_percent=2.5),
    ]

    # Run simulations
    results = []
    logger.info("\n" + "="*60)
    logger.info("Running Simulations")
    logger.info("="*60)

    for strategy in strategies:
        logger.info(f"\nStrategy: {strategy.name}")

        simulator = PortfolioSimulator(
            config=config,
            strategy=strategy,
            price_data=price_data
        )

        result = simulator.run()
        results.append(result)

        logger.info(f"✓ Complete - Return: {result.total_return_percent:+.2f}%, "
                   f"Fees: ${result.total_fees_paid:.2f}, "
                   f"Rebalances: {result.num_rebalances}")

    # Print comparison report
    print_simulation_report(results)

    # Additional analysis
    print("\n" + "="*80)
    print("ADDITIONAL ANALYSIS")
    print("="*80)

    buy_hold = results[0]
    best_hybrid = max(results[1:], key=lambda r: r.total_return_percent)

    return_diff = best_hybrid.total_return_percent - buy_hold.total_return_percent
    fee_cost = best_hybrid.total_fees_paid

    print(f"\nBest Hybrid Strategy: {best_hybrid.strategy_name}")
    print(f"Return vs Buy & Hold: {return_diff:+.2f}%")
    print(f"Total Fees Paid: ${fee_cost:.2f}")
    print(f"Fee Impact on $10k portfolio: {(fee_cost / INITIAL_CAPITAL) * 100:.2f}%")

    if return_diff > 0:
        print(f"\n✓ Rebalancing ADDED value of {return_diff:.2f}% despite ${fee_cost:.2f} in fees")
    else:
        print(f"\n✗ Rebalancing COST {abs(return_diff):.2f}% after paying ${fee_cost:.2f} in fees")

    print("\nConclusion:")
    if return_diff > (fee_cost / INITIAL_CAPITAL) * 100:
        print("✓ Rebalancing was worthwhile for this period")
    else:
        print("✗ Buy and hold would have been better for this period")

    print("="*80)


if __name__ == '__main__':
    main()
