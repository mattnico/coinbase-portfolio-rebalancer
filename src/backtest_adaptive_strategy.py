#!/usr/bin/env python3
"""
Backtest regime-adaptive portfolio strategy vs static allocations.

Compares:
- Adaptive: Switches portfolios based on detected regime
- Static Top_Three: Always uses Top_Three allocation
- Static Equal_Weight: Always uses Equal_Weight allocation
- Static Stablecoin_Heavy: Always uses Stablecoin_Heavy allocation

Shows which approach would have generated highest returns with lowest risk.

Usage:
    python -m src.backtest_adaptive_strategy --days 1825 --granularity ONE_DAY
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
class AdaptiveBacktestResult:
    """Results from adaptive strategy backtest."""
    strategy_name: str
    initial_value: float
    final_value: float
    total_return_percent: float
    annualized_return_percent: float
    sharpe_ratio: float
    max_drawdown_percent: float
    total_fees_paid: float
    num_regime_switches: int
    regime_history: List[Dict] = field(default_factory=list)
    portfolio_history: List[Dict] = field(default_factory=list)

    # Regime breakdown
    days_in_bull: int = 0
    days_in_bear: int = 0
    days_in_neutral: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'strategy_name': self.strategy_name,
            'initial_value': self.initial_value,
            'final_value': self.final_value,
            'total_return_percent': self.total_return_percent,
            'annualized_return_percent': self.annualized_return_percent,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown_percent': self.max_drawdown_percent,
            'total_fees_paid': self.total_fees_paid,
            'num_regime_switches': self.num_regime_switches,
            'days_in_bull': self.days_in_bull,
            'days_in_bear': self.days_in_bear,
            'days_in_neutral': self.days_in_neutral
        }


def setup_logging(quiet: bool = False):
    """Configure logging."""
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_portfolio_configs(config_path: str) -> Dict[str, Dict[str, float]]:
    """Load portfolio configurations."""
    config_file = Path(config_path)

    if not config_file.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Extract allocations from portfolios
    portfolios = config.get('portfolios', {})
    allocations = {}

    for name, portfolio in portfolios.items():
        # Remove description if present
        allocation = {k: v for k, v in portfolio.items() if k != 'description'}
        allocations[name] = allocation

    return allocations


def detect_regime_at_timestamp(
    detector: ReturnDetector,
    btc_prices: List[Tuple[datetime, float]],
    current_timestamp: datetime
) -> MarketRegime:
    """
    Detect market regime at a specific timestamp using 30-day lookback.

    Args:
        detector: ReturnDetector instance
        btc_prices: Full BTC price history
        current_timestamp: Timestamp to detect regime for

    Returns:
        Detected MarketRegime
    """
    # Get 30 days of prices before current timestamp
    lookback = timedelta(days=30)
    start_window = current_timestamp - lookback

    window_prices = [
        (ts, price) for ts, price in btc_prices
        if start_window <= ts <= current_timestamp
    ]

    if len(window_prices) < 20:  # Need at least 20 days
        return MarketRegime.NEUTRAL  # Default to neutral if insufficient data

    prices = [p[1] for p in window_prices]
    timestamps = [p[0] for p in window_prices]

    detection = detector.detect(prices, timestamps)
    return detection.regime


def backtest_adaptive_strategy(
    price_data: Dict[str, List[Tuple[datetime, float]]],
    portfolios: Dict[str, Dict[str, float]],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float,
    fee_rate: float,
    detector: ReturnDetector,
    regime_persistence_days: int = 7,
    check_frequency_days: int = 7
) -> AdaptiveBacktestResult:
    """
    Backtest adaptive portfolio strategy that switches based on regime.

    Args:
        price_data: Historical price data for all assets
        portfolios: Dict mapping portfolio name to allocation
        start_date: Start of backtest period
        end_date: End of backtest period
        initial_capital: Starting capital in USD
        fee_rate: Trading fee rate
        detector: ReturnDetector instance
        regime_persistence_days: Days regime must persist before switching (default: 7)
        check_frequency_days: Days between regime checks (default: 7)

    Returns:
        AdaptiveBacktestResult with performance metrics
    """
    # Portfolio mapping
    regime_to_portfolio = {
        MarketRegime.BULL: 'Equal_Weight',
        MarketRegime.BEAR: 'Stablecoin_Heavy',
        MarketRegime.NEUTRAL: 'Top_Three'
    }

    # Get BTC prices for regime detection
    btc_prices = price_data.get('BTC', [])

    # Initialize holdings (track actual units of each asset)
    holdings = {}  # asset -> quantity
    current_portfolio_name = 'Top_Three'  # Start with neutral
    current_allocation = portfolios[current_portfolio_name]

    # Get all unique timestamps from price data
    all_timestamps = sorted(set(ts for prices in price_data.values() for ts, _ in prices))
    timestamps_in_range = [ts for ts in all_timestamps if start_date <= ts <= end_date]

    # Initialize holdings with first day prices
    first_prices = {}
    first_ts = timestamps_in_range[0]

    for asset, prices in price_data.items():
        # Find closest price to first timestamp
        if prices:
            # Try exact match first
            matching = [(ts, p) for ts, p in prices if ts == first_ts]
            if matching:
                first_prices[asset] = matching[0][1]
            else:
                # Use first available price if no exact match
                first_prices[asset] = prices[0][1]

    # Buy initial positions
    for asset, target_pct in current_allocation.items():
        if target_pct > 0:  # Only buy assets with positive allocation
            if asset in first_prices and first_prices[asset] > 0:
                usd_to_invest = initial_capital * (target_pct / 100.0)
                quantity = usd_to_invest / first_prices[asset]
                holdings[asset] = quantity
            elif target_pct > 0:
                # Log warning but continue
                import logging
                logging.warning(f"No price data for {asset} at start date, skipping")

    # Verify we have holdings
    if not holdings:
        raise ValueError(f"Failed to initialize holdings. first_prices: {first_prices}, allocation: {current_allocation}")

    # Track state
    portfolio_history = []
    regime_history = []
    total_fees = 0.0
    num_switches = 0

    # Regime persistence tracking
    current_detected_regime = MarketRegime.NEUTRAL
    regime_detection_buffer = []  # Last N detections
    confirmed_regime = MarketRegime.NEUTRAL

    # Counters
    days_in_bull = 0
    days_in_bear = 0
    days_in_neutral = 0
    days_since_last_check = 0

    # Simulate day by day
    for i, current_ts in enumerate(timestamps_in_range):
        # Get current prices
        current_prices = {}
        for asset, prices in price_data.items():
            matching = [(ts, p) for ts, p in prices if ts == current_ts]
            if matching:
                current_prices[asset] = matching[0][1]

        # Calculate current portfolio value
        current_value = sum(
            holdings.get(asset, 0) * current_prices.get(asset, 0)
            for asset in holdings.keys()
        )

        # Check regime periodically (not every day)
        days_since_last_check += 1
        if days_since_last_check >= check_frequency_days or i == 0:
            days_since_last_check = 0

            # Detect current regime
            detected_regime = detect_regime_at_timestamp(detector, btc_prices, current_ts)

            # Add to detection buffer for persistence check
            regime_detection_buffer.append(detected_regime)
            if len(regime_detection_buffer) > regime_persistence_days // check_frequency_days:
                regime_detection_buffer.pop(0)

            # Check if regime has persisted long enough
            if len(regime_detection_buffer) >= regime_persistence_days // check_frequency_days:
                # Require majority of recent detections to agree
                regime_counts = {}
                for r in regime_detection_buffer:
                    regime_counts[r] = regime_counts.get(r, 0) + 1

                most_common_regime = max(regime_counts, key=regime_counts.get)
                most_common_count = regime_counts[most_common_regime]

                # Switch only if majority agrees and it's different from current
                if most_common_count >= len(regime_detection_buffer) // 2 + 1:
                    if most_common_regime != confirmed_regime:
                        # Regime change confirmed - rebalance portfolio
                        old_portfolio = current_portfolio_name
                        new_portfolio = regime_to_portfolio[most_common_regime]

                        if new_portfolio != current_portfolio_name:
                            # Calculate rebalancing trades
                            old_allocation = portfolios[current_portfolio_name]
                            new_allocation = portfolios[new_portfolio]

                            # Calculate trade volume (sum of absolute allocation changes)
                            trade_volume_usd = 0.0
                            for asset in set(old_allocation.keys()) | set(new_allocation.keys()):
                                old_pct = old_allocation.get(asset, 0.0)
                                new_pct = new_allocation.get(asset, 0.0)
                                change_pct = abs(new_pct - old_pct)
                                trade_volume_usd += current_value * (change_pct / 100.0)

                            # Charge realistic fees (only on traded volume, not entire portfolio)
                            rebalance_fee = trade_volume_usd * fee_rate
                            total_fees += rebalance_fee
                            current_value -= rebalance_fee

                            # Execute rebalancing - sell all and buy new allocation
                            # This is simpler and more realistic than tracking partial rebalances
                            new_holdings = {}
                            for asset in set(list(new_allocation.keys()) + list(holdings.keys())):
                                target_pct = new_allocation.get(asset, 0.0)
                                if target_pct > 0 and asset in current_prices and current_prices[asset] > 0:
                                    target_value = current_value * (target_pct / 100.0)
                                    new_holdings[asset] = target_value / current_prices[asset]
                                else:
                                    new_holdings[asset] = 0.0  # Sell assets not in new allocation

                            holdings = new_holdings
                            current_portfolio_name = new_portfolio
                            confirmed_regime = most_common_regime
                            num_switches += 1

                            regime_history.append({
                                'timestamp': current_ts,
                                'from_regime': old_portfolio,
                                'to_regime': most_common_regime.value,
                                'portfolio_switched_to': new_portfolio,
                                'fee_paid': rebalance_fee
                            })

        # Count regime days based on confirmed regime
        if confirmed_regime == MarketRegime.BULL:
            days_in_bull += 1
        elif confirmed_regime == MarketRegime.BEAR:
            days_in_bear += 1
        else:
            days_in_neutral += 1

        # Record portfolio state
        portfolio_history.append({
            'timestamp': current_ts,
            'value': current_value,
            'regime': confirmed_regime.value,
            'portfolio': current_portfolio_name
        })

    # Calculate metrics
    final_value = current_value
    total_return = ((final_value - initial_capital) / initial_capital) * 100

    # Calculate annualized return
    days = (end_date - start_date).days
    years = days / 365.25
    annualized_return = ((final_value / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Calculate Sharpe ratio (simplified)
    if len(portfolio_history) > 1:
        values = [p['value'] for p in portfolio_history]
        returns = []
        for i in range(1, len(values)):
            if values[i-1] > 0:
                ret = (values[i] - values[i-1]) / values[i-1]
                returns.append(ret)

        if returns:
            avg_return = sum(returns) / len(returns)
            std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            sharpe_ratio = (avg_return / std_return) * (252 ** 0.5) if std_return > 0 else 0
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    # Calculate max drawdown
    max_drawdown = 0.0
    peak = portfolio_history[0]['value'] if portfolio_history else initial_capital
    if peak == 0:
        peak = initial_capital  # Fallback to initial capital if first value is 0

    for point in portfolio_history:
        value = point['value']
        if value > peak:
            peak = value
        if peak > 0:
            drawdown = ((peak - value) / peak) * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

    return AdaptiveBacktestResult(
        strategy_name="Adaptive (Regime-Switching)",
        initial_value=initial_capital,
        final_value=final_value,
        total_return_percent=total_return,
        annualized_return_percent=annualized_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown_percent=max_drawdown,
        total_fees_paid=total_fees,
        num_regime_switches=num_switches,
        regime_history=regime_history,
        portfolio_history=portfolio_history,
        days_in_bull=days_in_bull,
        days_in_bear=days_in_bear,
        days_in_neutral=days_in_neutral
    )


def backtest_static_strategy(
    price_data: Dict[str, List[Tuple[datetime, float]]],
    allocation: Dict[str, float],
    strategy_name: str,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float,
    fee_rate: float
) -> SimulationResult:
    """
    Backtest static portfolio allocation (buy and hold).

    Args:
        price_data: Historical price data
        allocation: Target allocation percentages
        strategy_name: Name of strategy
        start_date: Start date
        end_date: End date
        initial_capital: Starting capital
        fee_rate: Trading fee rate

    Returns:
        SimulationResult
    """
    sim_config = SimulationConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital_usd=initial_capital,
        target_allocation=allocation,
        fee_rate=fee_rate,
        price_check_interval_hours=24
    )

    strategy = BuyAndHoldStrategy()
    simulator = PortfolioSimulator(
        config=sim_config,
        strategy=strategy,
        price_data=price_data
    )

    result = simulator.run()
    result.strategy_name = strategy_name
    return result


def print_backtest_comparison(
    adaptive_result: AdaptiveBacktestResult,
    static_results: List[SimulationResult]
):
    """Print comprehensive comparison of backtest results."""
    print("\n" + "="*100)
    print("ADAPTIVE VS STATIC PORTFOLIO BACKTEST RESULTS")
    print("="*100)

    # Adaptive strategy
    print(f"\n{'ADAPTIVE STRATEGY (REGIME-SWITCHING)':^100}")
    print("-" * 100)
    print(f"Initial Value:        ${adaptive_result.initial_value:,.2f}")
    print(f"Final Value:          ${adaptive_result.final_value:,.2f}")
    print(f"Total Return:         {adaptive_result.total_return_percent:+.2f}%")
    print(f"Annualized Return:    {adaptive_result.annualized_return_percent:+.2f}%")
    print(f"Sharpe Ratio:         {adaptive_result.sharpe_ratio:.3f}")
    print(f"Max Drawdown:         {adaptive_result.max_drawdown_percent:.2f}%")
    print(f"Total Fees Paid:      ${adaptive_result.total_fees_paid:,.2f}")
    print(f"Regime Switches:      {adaptive_result.num_regime_switches}")
    print(f"\nRegime Breakdown:")
    print(f"  BULL:    {adaptive_result.days_in_bull} days ({adaptive_result.days_in_bull / (adaptive_result.days_in_bull + adaptive_result.days_in_bear + adaptive_result.days_in_neutral) * 100:.1f}%)")
    print(f"  BEAR:    {adaptive_result.days_in_bear} days ({adaptive_result.days_in_bear / (adaptive_result.days_in_bull + adaptive_result.days_in_bear + adaptive_result.days_in_neutral) * 100:.1f}%)")
    print(f"  NEUTRAL: {adaptive_result.days_in_neutral} days ({adaptive_result.days_in_neutral / (adaptive_result.days_in_bull + adaptive_result.days_in_bear + adaptive_result.days_in_neutral) * 100:.1f}%)")

    # Static strategies
    print(f"\n{'STATIC STRATEGIES (BUY AND HOLD)':^100}")
    print("-" * 100)

    for result in static_results:
        print(f"\n{result.strategy_name}:")
        print(f"  Final Value:       ${result.final_value:,.2f}")
        print(f"  Total Return:      {result.total_return_percent:+.2f}%")
        print(f"  Annualized Return: {result.annualized_return_percent:+.2f}%")
        print(f"  Sharpe Ratio:      {result.sharpe_ratio:.3f}")
        print(f"  Max Drawdown:      {result.max_drawdown_percent:.2f}%")
        print(f"  Total Fees:        ${result.total_fees_paid:,.2f}")

    # Comparison table
    print("\n" + "="*100)
    print("STRATEGY COMPARISON")
    print("="*100)

    all_results = [adaptive_result] + static_results
    best_return = max(all_results, key=lambda r: r.total_return_percent if hasattr(r, 'total_return_percent') else r.total_return_percent)
    best_sharpe = max(all_results, key=lambda r: r.sharpe_ratio)
    lowest_drawdown = min(all_results, key=lambda r: r.max_drawdown_percent)

    print(f"\n{'Strategy':<40} {'Return':<15} {'Sharpe':<12} {'Max DD':<12}")
    print("-" * 100)

    for result in all_results:
        name = result.strategy_name if hasattr(result, 'strategy_name') else "Unknown"
        ret = result.total_return_percent
        sharpe = result.sharpe_ratio
        dd = result.max_drawdown_percent

        marker_ret = " 🏆" if result == best_return else ""
        marker_sharpe = " 🏆" if result == best_sharpe else ""
        marker_dd = " 🏆" if result == lowest_drawdown else ""

        print(f"{name:<40} {ret:>+8.2f}%{marker_ret:<6} {sharpe:>8.3f}{marker_sharpe:<5} {dd:>8.2f}%{marker_dd}")

    # Key insights
    print("\n" + "="*100)
    print("KEY INSIGHTS")
    print("="*100)

    adaptive_vs_best_static = adaptive_result.total_return_percent - max(r.total_return_percent for r in static_results)

    if adaptive_vs_best_static > 0:
        print(f"✓ Adaptive strategy OUTPERFORMED best static by {adaptive_vs_best_static:+.2f}%")
    else:
        print(f"✗ Adaptive strategy UNDERPERFORMED best static by {abs(adaptive_vs_best_static):.2f}%")

    print(f"✓ Adaptive strategy switched regimes {adaptive_result.num_regime_switches} times")
    print(f"✓ Regime detection fees: ${adaptive_result.total_fees_paid:,.2f}")

    # Risk-adjusted performance
    adaptive_sharpe = adaptive_result.sharpe_ratio
    best_static_sharpe = max(r.sharpe_ratio for r in static_results)

    if adaptive_sharpe > best_static_sharpe:
        print(f"✓ Adaptive had better risk-adjusted returns (Sharpe: {adaptive_sharpe:.3f} vs {best_static_sharpe:.3f})")
    else:
        print(f"✗ Static had better risk-adjusted returns (Sharpe: {best_static_sharpe:.3f} vs {adaptive_sharpe:.3f})")

    print("="*100 + "\n")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Backtest adaptive regime-switching strategy vs static allocations'
    )

    # Time period
    time_group = parser.add_mutually_exclusive_group(required=True)
    time_group.add_argument(
        '--days',
        type=int,
        help='Number of days to backtest'
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

    parser.add_argument(
        '--config',
        type=str,
        default='config/top_four_portfolios.json',
        help='Path to portfolio configuration file'
    )

    parser.add_argument(
        '--initial-capital',
        type=float,
        default=10000.0,
        help='Initial portfolio value in USD (default: 10000)'
    )

    parser.add_argument(
        '--fee-rate',
        type=float,
        default=0.006,
        help='Trading fee rate (default: 0.006 = 0.6%%)'
    )

    parser.add_argument(
        '--granularity',
        type=str,
        default='ONE_DAY',
        choices=['ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 'THIRTY_MINUTE',
                 'ONE_HOUR', 'TWO_HOUR', 'SIX_HOUR', 'ONE_DAY'],
        help='Price data granularity (default: ONE_DAY)'
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
        if args.end:
            end_date = datetime.fromisoformat(args.end)
        else:
            end_date = datetime.now()

    print("\n" + "="*100)
    print("ADAPTIVE PORTFOLIO STRATEGY BACKTEST")
    print("="*100)
    print(f"\nPeriod: {start_date.date()} to {end_date.date()}")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    print(f"Fee Rate: {args.fee_rate * 100:.2f}%")
    print(f"Granularity: {args.granularity}")
    print("="*100 + "\n")

    # Load portfolio configurations
    logger.info(f"Loading portfolio configurations from {args.config}")
    portfolios = load_portfolio_configs(args.config)

    # Initialize Coinbase client
    logger.info("Initializing Coinbase client...")
    try:
        client = CoinbaseClient()
    except Exception as e:
        logger.error(f"Failed to initialize client: {e}")
        sys.exit(1)

    # Fetch historical price data for all assets
    all_assets = set()
    for allocation in portfolios.values():
        all_assets.update(allocation.keys())

    logger.info(f"Fetching historical price data for {len(all_assets)} assets...")
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
            granularity=args.granularity,
            show_progress=not args.quiet,
            cache_only=args.cache_only
        )
    except Exception as e:
        logger.error(f"Failed to fetch price data: {e}")
        sys.exit(1)

    # Validate we got data
    for asset, prices in price_data.items():
        if not prices:
            logger.error(f"No price data for {asset}")
            sys.exit(1)
        logger.info(f"{asset}: {len(prices)} data points")

    # Initialize regime detector with optimal thresholds
    detector = ReturnDetector(
        window_days=30,
        bull_threshold=15.0,
        bear_threshold=-10.0
    )

    # Run adaptive backtest
    logger.info("Running adaptive strategy backtest...")
    adaptive_result = backtest_adaptive_strategy(
        price_data=price_data,
        portfolios=portfolios,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.initial_capital,
        fee_rate=args.fee_rate,
        detector=detector
    )

    # Run static strategy backtests
    logger.info("Running static strategy backtests...")
    static_results = []

    for portfolio_name, allocation in portfolios.items():
        logger.info(f"  Testing {portfolio_name}...")
        result = backtest_static_strategy(
            price_data=price_data,
            allocation=allocation,
            strategy_name=f"Static {portfolio_name}",
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.initial_capital,
            fee_rate=args.fee_rate
        )
        static_results.append(result)

    # Print comparison
    print_backtest_comparison(adaptive_result, static_results)

    # Save to file if requested
    if args.output:
        output_data = {
            'backtest_date': datetime.now().isoformat(),
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'adaptive_strategy': adaptive_result.to_dict(),
            'static_strategies': [r.to_dict() for r in static_results]
        }

        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
