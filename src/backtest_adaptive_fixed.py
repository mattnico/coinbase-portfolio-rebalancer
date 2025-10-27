#!/usr/bin/env python3
"""
Fixed adaptive strategy backtest with proper holdings tracking.

Focuses on bear market protection by switching to stablecoins during corrections.

Usage:
    python -m src.backtest_adaptive_fixed --days 1825 --granularity ONE_DAY
"""

import argparse
import logging
import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

# Suppress urllib3 OpenSSL warnings
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')

from src.coinbase_client import CoinbaseClient
from src.monte_carlo_simulator import (
    HistoricalPriceFetcher,
    SimulationConfig,
    PortfolioSimulator,
    BuyAndHoldStrategy,
    SimulationResult
)
from src.regime_detector import ReturnDetector, MarketRegime


@dataclass
class AdaptiveResult:
    """Results from adaptive backtest."""
    strategy_name: str
    initial_value: float
    final_value: float
    total_return_percent: float
    annualized_return_percent: float
    sharpe_ratio: float
    max_drawdown_percent: float
    total_fees_paid: float
    num_switches: int
    days_in_bull: int
    days_in_bear: int
    days_in_neutral: int


def setup_logging(quiet: bool = False):
    """Configure logging."""
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_portfolios(config_path: str) -> Dict[str, Dict[str, float]]:
    """Load portfolio configurations."""
    with open(config_path) as f:
        config = json.load(f)

    portfolios = {}
    for name, portfolio in config.get('portfolios', {}).items():
        # Remove description, keep only allocations
        alloc = {k: v for k, v in portfolio.items() if k != 'description'}
        portfolios[name] = alloc

    return portfolios


def get_price_at_date(price_data: Dict[str, List[Tuple[datetime, float]]], asset: str, target_date: datetime) -> float:
    """Get price for asset closest to target date."""
    if asset not in price_data or not price_data[asset]:
        return 0.0

    prices = price_data[asset]

    # Find closest price
    closest = min(prices, key=lambda p: abs((p[0] - target_date).total_seconds()))

    return closest[1]


def backtest_adaptive(
    price_data: Dict[str, List[Tuple[datetime, float]]],
    portfolios: Dict[str, Dict[str, float]],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float,
    fee_rate: float,
    check_days: int = 7,
    persistence_days: int = 14
) -> AdaptiveResult:
    """
    Backtest adaptive strategy with proper holdings tracking.

    Args:
        price_data: Historical prices
        portfolios: Portfolio allocations
        start_date: Start date
        end_date: End date
        initial_capital: Starting capital
        fee_rate: Trading fee rate
        check_days: Days between regime checks
        persistence_days: Days regime must persist before switching

    Returns:
        AdaptiveResult with metrics
    """
    logger = logging.getLogger(__name__)

    # Get all dates
    all_dates = sorted(set(ts for prices in price_data.values() for ts, _ in prices))
    dates = [d for d in all_dates if start_date <= d <= end_date]

    if not dates:
        raise ValueError("No dates in range")

    # Initialize detector
    detector = ReturnDetector(window_days=30, bull_threshold=15.0, bear_threshold=-10.0)

    # Portfolio mapping (focus on bear protection)
    regime_portfolios = {
        MarketRegime.BULL: 'Top_Three',           # Ride bulls
        MarketRegime.BEAR: 'Stablecoin_Heavy',    # Protect in bears
        MarketRegime.NEUTRAL: 'Top_Three'         # Default to Top_Three
    }

    # Get all unique assets across all portfolios
    all_assets = set()
    for portfolio in portfolios.values():
        all_assets.update(portfolio.keys())

    # Initialize with Top_Three
    current_portfolio = 'Top_Three'
    current_allocation = portfolios[current_portfolio]

    # Buy initial holdings
    holdings = {}  # asset -> quantity
    first_date = dates[0]

    for asset in all_assets:
        price = get_price_at_date(price_data, asset, first_date)
        target_pct = current_allocation.get(asset, 0.0)

        if target_pct > 0 and price > 0:
            holdings[asset] = (initial_capital * target_pct / 100.0) / price
        else:
            holdings[asset] = 0.0

    logger.info(f"Initialized holdings: {holdings}")
    logger.info(f"Starting portfolio: {current_portfolio}")

    # Track state
    portfolio_values = []
    total_fees = 0.0
    num_switches = 0

    # Regime tracking
    confirmed_regime = MarketRegime.NEUTRAL
    regime_buffer = []
    days_in_bull = 0
    days_in_bear = 0
    days_in_neutral = 0
    check_counter = 0

    # Simulate day by day
    for i, date in enumerate(dates):
        # Get current prices for all assets
        current_prices = {}
        for asset in all_assets:
            current_prices[asset] = get_price_at_date(price_data, asset, date)

        # Calculate portfolio value
        portfolio_value = sum(holdings[asset] * current_prices[asset] for asset in all_assets)

        # Check for regime change periodically
        check_counter += 1
        if check_counter >= check_days or i == 0:
            check_counter = 0

            # Detect regime using BTC prices
            btc_prices = price_data.get('BTC', [])
            window_start = date - timedelta(days=30)
            window_prices = [(ts, p) for ts, p in btc_prices if window_start <= ts <= date]

            if len(window_prices) >= 20:
                prices = [p[1] for p in window_prices]
                timestamps = [p[0] for p in window_prices]
                detection = detector.detect(prices, timestamps)
                detected_regime = detection.regime

                # Add to buffer
                regime_buffer.append(detected_regime)
                if len(regime_buffer) > persistence_days // check_days:
                    regime_buffer.pop(0)

                # Check if regime has persisted
                if len(regime_buffer) >= persistence_days // check_days:
                    # Count regime occurrences
                    regime_counts = {}
                    for r in regime_buffer:
                        regime_counts[r] = regime_counts.get(r, 0) + 1

                    # Get most common
                    most_common = max(regime_counts, key=regime_counts.get)
                    count = regime_counts[most_common]

                    # Require majority
                    if count >= len(regime_buffer) // 2 + 1:
                        if most_common != confirmed_regime:
                            # Regime change confirmed!
                            old_regime = confirmed_regime
                            confirmed_regime = most_common

                            new_portfolio = regime_portfolios[confirmed_regime]

                            if new_portfolio != current_portfolio:
                                logger.info(f"Date {date.date()}: Regime {old_regime.value} -> {confirmed_regime.value}")
                                logger.info(f"  Switching: {current_portfolio} -> {new_portfolio}")
                                logger.info(f"  Portfolio value before switch: ${portfolio_value:,.2f}")

                                # Rebalance to new portfolio
                                old_allocation = portfolios[current_portfolio]
                                new_allocation = portfolios[new_portfolio]

                                # Calculate trade volume
                                trade_volume = 0.0
                                for asset in all_assets:
                                    old_pct = old_allocation.get(asset, 0.0)
                                    new_pct = new_allocation.get(asset, 0.0)
                                    change = abs(new_pct - old_pct) / 100.0
                                    trade_volume += portfolio_value * change

                                # Charge fees on trade volume
                                fee = trade_volume * fee_rate
                                total_fees += fee
                                portfolio_value -= fee

                                logger.info(f"  Trade volume: ${trade_volume:,.2f}, Fee: ${fee:,.2f}")

                                # Rebalance holdings
                                for asset in all_assets:
                                    target_pct = new_allocation.get(asset, 0.0)
                                    price = current_prices[asset]

                                    if price > 0:
                                        target_value = portfolio_value * (target_pct / 100.0)
                                        holdings[asset] = target_value / price
                                    else:
                                        holdings[asset] = 0.0

                                current_portfolio = new_portfolio
                                num_switches += 1

                                # Recalculate value after rebalancing
                                portfolio_value = sum(holdings[asset] * current_prices[asset] for asset in all_assets)
                                logger.info(f"  Portfolio value after switch: ${portfolio_value:,.2f}")

        # Count regime days
        if confirmed_regime == MarketRegime.BULL:
            days_in_bull += 1
        elif confirmed_regime == MarketRegime.BEAR:
            days_in_bear += 1
        else:
            days_in_neutral += 1

        # Record value
        portfolio_values.append(portfolio_value)

    # Calculate metrics
    final_value = portfolio_values[-1]
    total_return = ((final_value - initial_capital) / initial_capital) * 100

    days = (end_date - start_date).days
    years = days / 365.25
    annualized_return = ((final_value / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Sharpe ratio
    returns = []
    for i in range(1, len(portfolio_values)):
        if portfolio_values[i-1] > 0:
            ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            returns.append(ret)

    if returns:
        avg_return = sum(returns) / len(returns)
        std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
        sharpe = (avg_return / std_return) * (252 ** 0.5) if std_return > 0 else 0
    else:
        sharpe = 0.0

    # Max drawdown
    max_dd = 0.0
    peak = portfolio_values[0]
    for value in portfolio_values:
        if value > peak:
            peak = value
        if peak > 0:
            dd = ((peak - value) / peak) * 100
            if dd > max_dd:
                max_dd = dd

    return AdaptiveResult(
        strategy_name="Adaptive (Bear Protection)",
        initial_value=initial_capital,
        final_value=final_value,
        total_return_percent=total_return,
        annualized_return_percent=annualized_return,
        sharpe_ratio=sharpe,
        max_drawdown_percent=max_dd,
        total_fees_paid=total_fees,
        num_switches=num_switches,
        days_in_bull=days_in_bull,
        days_in_bear=days_in_bear,
        days_in_neutral=days_in_neutral
    )


def print_comparison(adaptive: AdaptiveResult, static_results: List[SimulationResult]):
    """Print comparison."""
    print("\n" + "="*100)
    print("ADAPTIVE (BEAR PROTECTION) VS STATIC BUY-AND-HOLD")
    print("="*100)

    # Adaptive
    print(f"\n{'ADAPTIVE STRATEGY (BEAR PROTECTION)':^100}")
    print("-" * 100)
    print(f"Initial Value:        ${adaptive.initial_value:,.2f}")
    print(f"Final Value:          ${adaptive.final_value:,.2f}")
    print(f"Total Return:         {adaptive.total_return_percent:+.2f}%")
    print(f"Annualized Return:    {adaptive.annualized_return_percent:+.2f}%")
    print(f"Sharpe Ratio:         {adaptive.sharpe_ratio:.3f}")
    print(f"Max Drawdown:         {adaptive.max_drawdown_percent:.2f}%")
    print(f"Total Fees:           ${adaptive.total_fees_paid:,.2f}")
    print(f"Portfolio Switches:   {adaptive.num_switches}")
    total_days = adaptive.days_in_bull + adaptive.days_in_bear + adaptive.days_in_neutral
    print(f"\nRegime Breakdown:")
    print(f"  BULL:    {adaptive.days_in_bull} days ({adaptive.days_in_bull/total_days*100:.1f}%)")
    print(f"  BEAR:    {adaptive.days_in_bear} days ({adaptive.days_in_bear/total_days*100:.1f}%)")
    print(f"  NEUTRAL: {adaptive.days_in_neutral} days ({adaptive.days_in_neutral/total_days*100:.1f}%)")

    # Static strategies
    print(f"\n{'STATIC STRATEGIES (BUY-AND-HOLD)':^100}")
    print("-" * 100)

    for result in static_results:
        print(f"\n{result.strategy_name}:")
        print(f"  Final Value:       ${result.final_value:,.2f}")
        print(f"  Total Return:      {result.total_return_percent:+.2f}%")
        print(f"  Annualized Return: {result.annualized_return_percent:+.2f}%")
        print(f"  Sharpe Ratio:      {result.sharpe_ratio:.3f}")
        print(f"  Max Drawdown:      {result.max_drawdown_percent:.2f}%")

    # Comparison table
    print("\n" + "="*100)
    print("COMPARISON TABLE")
    print("="*100)

    print(f"\n{'Strategy':<40} {'Return':<15} {'Sharpe':<12} {'Max DD':<12}")
    print("-" * 100)

    # Adaptive
    print(f"{'Adaptive (Bear Protection)':<40} {adaptive.total_return_percent:>+8.2f}%      "
          f"{adaptive.sharpe_ratio:>8.3f}    {adaptive.max_drawdown_percent:>8.2f}%")

    # Static
    for result in static_results:
        print(f"{result.strategy_name:<40} {result.total_return_percent:>+8.2f}%      "
              f"{result.sharpe_ratio:>8.3f}    {result.max_drawdown_percent:>8.2f}%")

    # Key insights
    print("\n" + "="*100)
    print("DRAWDOWN REDUCTION ANALYSIS")
    print("="*100)

    # Find Top_Three for comparison
    top_three = next((r for r in static_results if 'Top_Three' in r.strategy_name), None)

    if top_three:
        dd_reduction = top_three.max_drawdown_percent - adaptive.max_drawdown_percent
        return_difference = adaptive.total_return_percent - top_three.total_return_percent

        print(f"\nAdaptive vs Static Top_Three:")
        print(f"  Return:           {adaptive.total_return_percent:+.2f}% vs {top_three.total_return_percent:+.2f}% "
              f"(difference: {return_difference:+.2f}%)")
        print(f"  Max Drawdown:     {adaptive.max_drawdown_percent:.2f}% vs {top_three.max_drawdown_percent:.2f}% "
              f"(reduction: {dd_reduction:+.2f}%)")
        print(f"  Sharpe Ratio:     {adaptive.sharpe_ratio:.3f} vs {top_three.sharpe_ratio:.3f}")
        print(f"  Fees Paid:        ${adaptive.total_fees_paid:,.2f}")
        print(f"  Regime Switches:  {adaptive.num_switches}")

        if dd_reduction > 0:
            print(f"\n✓ Adaptive REDUCED drawdown by {dd_reduction:.2f}%")
        else:
            print(f"\n✗ Adaptive did NOT reduce drawdown (worse by {abs(dd_reduction):.2f}%)")

        if return_difference > 0:
            print(f"✓ Adaptive INCREASED returns by {return_difference:.2f}%")
        else:
            print(f"✗ Adaptive DECREASED returns by {abs(return_difference):.2f}%")

    print("="*100 + "\n")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Backtest adaptive bear protection strategy'
    )

    time_group = parser.add_mutually_exclusive_group(required=True)
    time_group.add_argument('--days', type=int, help='Number of days to backtest')
    time_group.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')

    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--config', type=str, default='config/test_portfolios.json')
    parser.add_argument('--initial-capital', type=float, default=10000.0)
    parser.add_argument('--fee-rate', type=float, default=0.006)
    parser.add_argument('--granularity', type=str, default='ONE_DAY')
    parser.add_argument('--check-days', type=int, default=7, help='Days between regime checks')
    parser.add_argument('--persistence-days', type=int, default=14, help='Days regime must persist')
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--quiet', action='store_true')

    return parser.parse_args()


def main():
    """Run backtest."""
    args = parse_arguments()
    setup_logging(args.quiet)
    logger = logging.getLogger(__name__)

    # Parse dates
    if args.days:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
    else:
        start_date = datetime.fromisoformat(args.start)
        end_date = datetime.fromisoformat(args.end) if args.end else datetime.now()

    print("\n" + "="*100)
    print("ADAPTIVE BEAR PROTECTION BACKTEST")
    print("="*100)
    print(f"\nPeriod: {start_date.date()} to {end_date.date()}")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    print(f"Fee Rate: {args.fee_rate * 100:.2f}%")
    print(f"Regime Check Frequency: Every {args.check_days} days")
    print(f"Regime Persistence Required: {args.persistence_days} days")
    print("="*100 + "\n")

    # Load portfolios
    portfolios = load_portfolios(args.config)
    logger.info(f"Loaded {len(portfolios)} portfolios")

    # Get all assets
    all_assets = set()
    for portfolio in portfolios.values():
        all_assets.update(portfolio.keys())

    # Fetch price data
    logger.info(f"Fetching price data for {len(all_assets)} assets...")
    client = CoinbaseClient()
    fetcher = HistoricalPriceFetcher(client, use_cache=not args.no_cache)

    price_data = fetcher.fetch_historical_prices(
        assets=list(all_assets),
        start_date=start_date,
        end_date=end_date,
        granularity=args.granularity,
        show_progress=not args.quiet
    )

    for asset, prices in price_data.items():
        logger.info(f"{asset}: {len(prices)} data points")

    # Run adaptive backtest
    logger.info("Running adaptive strategy...")
    adaptive_result = backtest_adaptive(
        price_data=price_data,
        portfolios=portfolios,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.initial_capital,
        fee_rate=args.fee_rate,
        check_days=args.check_days,
        persistence_days=args.persistence_days
    )

    # Run static backtests for comparison
    logger.info("Running static strategies for comparison...")
    static_results = []

    for name in ['Top_Three', 'Stablecoin_Heavy']:
        if name in portfolios:
            sim_config = SimulationConfig(
                start_date=start_date,
                end_date=end_date,
                initial_capital_usd=args.initial_capital,
                target_allocation=portfolios[name],
                fee_rate=args.fee_rate,
                price_check_interval_hours=24
            )

            simulator = PortfolioSimulator(
                config=sim_config,
                strategy=BuyAndHoldStrategy(),
                price_data=price_data
            )

            result = simulator.run()
            result.strategy_name = f"Static {name}"
            static_results.append(result)

    # Print comparison
    print_comparison(adaptive_result, static_results)


if __name__ == '__main__':
    main()
