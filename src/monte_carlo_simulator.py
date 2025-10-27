"""
Monte Carlo simulator for testing cryptocurrency rebalancing strategies.

This module provides historical backtesting capabilities to compare different
rebalancing approaches and optimize configuration parameters.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for simulation runs."""
    start_date: datetime
    end_date: datetime
    initial_capital_usd: float
    target_allocation: Dict[str, float]  # Asset: target percentage
    fee_rate: float = 0.006  # 0.6% flat fee per trade
    price_check_interval_hours: int = 24  # How often to check if rebalancing needed

    def __post_init__(self):
        """Validate configuration."""
        total_allocation = sum(self.target_allocation.values())
        if not 99.9 <= total_allocation <= 100.1:
            raise ValueError(f"Target allocation must sum to 100%, got {total_allocation}%")


@dataclass
class PortfolioState:
    """Snapshot of portfolio state at a point in time."""
    timestamp: datetime
    holdings: Dict[str, float]  # Asset: quantity
    prices: Dict[str, float]  # Asset: price in USD
    total_value_usd: float = 0.0

    def __post_init__(self):
        """Calculate total portfolio value."""
        self.total_value_usd = sum(
            qty * self.prices.get(asset, 0.0)
            for asset, qty in self.holdings.items()
        )

    def get_current_allocation(self) -> Dict[str, float]:
        """Get current allocation percentages."""
        if self.total_value_usd == 0:
            return {asset: 0.0 for asset in self.holdings.keys()}

        return {
            asset: (qty * self.prices.get(asset, 0.0) / self.total_value_usd * 100)
            for asset, qty in self.holdings.items()
        }

    def get_deviation_from_target(self, target: Dict[str, float]) -> Dict[str, float]:
        """Calculate how far each asset deviates from target allocation."""
        current = self.get_current_allocation()
        return {
            asset: current.get(asset, 0.0) - target.get(asset, 0.0)
            for asset in set(current.keys()) | set(target.keys())
        }


@dataclass
class Trade:
    """Record of a simulated trade."""
    timestamp: datetime
    asset: str
    action: str  # 'buy' or 'sell'
    quantity: float
    price: float
    value_usd: float
    fee_usd: float
    reason: str = ""


@dataclass
class SimulationResult:
    """Results from a simulation run."""
    strategy_name: str
    config: SimulationConfig
    initial_value: float
    final_value: float
    total_return_percent: float
    annualized_return_percent: float
    sharpe_ratio: float
    max_drawdown_percent: float
    total_fees_paid: float
    num_rebalances: int
    trades: List[Trade] = field(default_factory=list)
    portfolio_history: List[PortfolioState] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'strategy_name': self.strategy_name,
            'initial_value': self.initial_value,
            'final_value': self.final_value,
            'total_return_percent': self.total_return_percent,
            'annualized_return_percent': self.annualized_return_percent,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown_percent': self.max_drawdown_percent,
            'total_fees_paid': self.total_fees_paid,
            'num_rebalances': self.num_rebalances,
            'num_trades': len(self.trades),
            'start_date': self.config.start_date.isoformat(),
            'end_date': self.config.end_date.isoformat(),
        }


class RebalancingStrategy(ABC):
    """Abstract base class for rebalancing strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def should_rebalance(
        self,
        current_state: PortfolioState,
        target_allocation: Dict[str, float],
        last_rebalance_time: Optional[datetime],
        **kwargs
    ) -> Tuple[bool, str]:
        """
        Determine if portfolio should be rebalanced.

        Returns:
            (should_rebalance: bool, reason: str)
        """
        pass

    @abstractmethod
    def calculate_trades(
        self,
        current_state: PortfolioState,
        target_allocation: Dict[str, float],
        fee_rate: float,
        min_trade_value_usd: float = 10.0
    ) -> List[Trade]:
        """
        Calculate trades needed to rebalance portfolio.

        Returns:
            List of Trade objects
        """
        pass


class BuyAndHoldStrategy(RebalancingStrategy):
    """Never rebalances - used as baseline."""

    def __init__(self):
        super().__init__("Buy and Hold")

    def should_rebalance(
        self,
        current_state: PortfolioState,
        target_allocation: Dict[str, float],
        last_rebalance_time: Optional[datetime],
        **kwargs
    ) -> Tuple[bool, str]:
        """Never rebalance."""
        return False, "Buy and hold strategy"

    def calculate_trades(
        self,
        current_state: PortfolioState,
        target_allocation: Dict[str, float],
        fee_rate: float,
        min_trade_value_usd: float = 10.0
    ) -> List[Trade]:
        """No trades."""
        return []


class HybridStrategy(RebalancingStrategy):
    """
    Rebalances at fixed time intervals only if threshold is exceeded.

    Combines time-based (e.g., hourly, daily, weekly) with threshold-based (e.g., ±2.5%) logic.
    """

    def __init__(
        self,
        rebalance_interval: timedelta = None,
        rebalance_interval_days: int = None,
        rebalance_interval_hours: int = None,
        rebalance_interval_minutes: int = None,
        threshold_percent: float = 2.5
    ):
        # Accept interval in multiple formats for flexibility
        if rebalance_interval is not None:
            self.interval = rebalance_interval
        elif rebalance_interval_minutes is not None:
            self.interval = timedelta(minutes=rebalance_interval_minutes)
        elif rebalance_interval_hours is not None:
            self.interval = timedelta(hours=rebalance_interval_hours)
        elif rebalance_interval_days is not None:
            self.interval = timedelta(days=rebalance_interval_days)
        else:
            # Default to 7 days
            self.interval = timedelta(days=7)

        self.threshold_percent = threshold_percent

        # Format name based on interval
        total_seconds = self.interval.total_seconds()
        if total_seconds < 3600:  # Less than 1 hour
            minutes = int(total_seconds / 60)
            interval_str = f"{minutes}min"
        elif total_seconds < 86400:  # Less than 1 day
            hours = int(total_seconds / 3600)
            interval_str = f"{hours}h"
        else:
            days = int(total_seconds / 86400)
            interval_str = f"{days}d"

        super().__init__(f"Hybrid (every {interval_str}, ±{threshold_percent}%)")

    def should_rebalance(
        self,
        current_state: PortfolioState,
        target_allocation: Dict[str, float],
        last_rebalance_time: Optional[datetime],
        **kwargs
    ) -> Tuple[bool, str]:
        """Check if interval has passed AND threshold is exceeded."""

        # Check time interval
        if last_rebalance_time is None:
            time_to_rebalance = True
            time_reason = "Initial rebalance"
        else:
            time_since_rebalance = current_state.timestamp - last_rebalance_time
            time_to_rebalance = time_since_rebalance >= self.interval

            # Format time reason based on interval
            total_seconds = time_since_rebalance.total_seconds()
            if self.interval.total_seconds() < 3600:  # Minutes
                time_amount = int(total_seconds / 60)
                time_unit = "minutes"
            elif self.interval.total_seconds() < 86400:  # Hours
                time_amount = int(total_seconds / 3600)
                time_unit = "hours"
            else:  # Days
                time_amount = int(total_seconds / 86400)
                time_unit = "days"

            time_reason = f"{time_amount} {time_unit} since last rebalance"

        if not time_to_rebalance:
            return False, time_reason

        # Time interval met - now check threshold
        deviations = current_state.get_deviation_from_target(target_allocation)
        max_deviation = max(abs(dev) for dev in deviations.values())

        if max_deviation >= self.threshold_percent:
            reason = f"{time_reason}, max deviation {max_deviation:.2f}% >= {self.threshold_percent}%"
            return True, reason
        else:
            reason = f"{time_reason}, but max deviation {max_deviation:.2f}% < {self.threshold_percent}%"
            return False, reason

    def calculate_trades(
        self,
        current_state: PortfolioState,
        target_allocation: Dict[str, float],
        fee_rate: float,
        min_trade_value_usd: float = 10.0
    ) -> List[Trade]:
        """Calculate trades to return to target allocation."""

        trades = []
        current_alloc = current_state.get_current_allocation()

        # Calculate target values
        target_values = {
            asset: current_state.total_value_usd * (pct / 100)
            for asset, pct in target_allocation.items()
        }

        # Calculate current values
        current_values = {
            asset: current_state.holdings.get(asset, 0.0) * current_state.prices.get(asset, 0.0)
            for asset in target_allocation.keys()
        }

        # Determine what to buy and sell
        to_buy = {}
        to_sell = {}

        for asset in target_allocation.keys():
            current_val = current_values[asset]
            target_val = target_values[asset]
            diff = target_val - current_val

            if diff > min_trade_value_usd:
                to_buy[asset] = diff
            elif diff < -min_trade_value_usd:
                to_sell[asset] = abs(diff)

        # Create sell trades (these generate cash)
        for asset, value_usd in to_sell.items():
            price = current_state.prices[asset]
            quantity = value_usd / price
            fee = value_usd * fee_rate

            trades.append(Trade(
                timestamp=current_state.timestamp,
                asset=asset,
                action='sell',
                quantity=quantity,
                price=price,
                value_usd=value_usd,
                fee_usd=fee,
                reason='Rebalancing'
            ))

        # Create buy trades (these consume cash)
        for asset, value_usd in to_buy.items():
            price = current_state.prices[asset]
            # Account for fee in purchase
            value_after_fee = value_usd / (1 + fee_rate)
            quantity = value_after_fee / price
            fee = value_after_fee * fee_rate

            trades.append(Trade(
                timestamp=current_state.timestamp,
                asset=asset,
                action='buy',
                quantity=quantity,
                price=price,
                value_usd=value_usd,
                fee_usd=fee,
                reason='Rebalancing'
            ))

        return trades


class PortfolioSimulator:
    """Simulates portfolio performance over time with different strategies."""

    def __init__(
        self,
        config: SimulationConfig,
        strategy: RebalancingStrategy,
        price_data: Dict[str, List[Tuple[datetime, float]]]
    ):
        """
        Initialize simulator.

        Args:
            config: Simulation configuration
            strategy: Rebalancing strategy to test
            price_data: Historical prices {asset: [(timestamp, price), ...]}
        """
        self.config = config
        self.strategy = strategy
        self.price_data = price_data

        # Initialize portfolio with target allocation
        self.current_holdings = self._initialize_portfolio()
        self.portfolio_history: List[PortfolioState] = []
        self.trades: List[Trade] = []
        self.last_rebalance_time: Optional[datetime] = None
        self.total_fees_paid = 0.0

    def _initialize_portfolio(self) -> Dict[str, float]:
        """Set up initial portfolio based on target allocation."""
        initial_prices = {
            asset: prices[0][1]
            for asset, prices in self.price_data.items()
        }

        holdings = {}
        for asset, target_pct in self.config.target_allocation.items():
            target_value = self.config.initial_capital_usd * (target_pct / 100)
            holdings[asset] = target_value / initial_prices[asset]

        logger.info(f"Initialized portfolio with {self.config.initial_capital_usd} USD")
        logger.info(f"Initial holdings: {holdings}")

        return holdings

    def _get_prices_at_time(self, timestamp: datetime) -> Dict[str, float]:
        """Get prices for all assets at a specific time."""
        prices = {}
        for asset, price_series in self.price_data.items():
            # Find closest price on or before timestamp
            valid_prices = [p for p in price_series if p[0] <= timestamp]
            if valid_prices:
                prices[asset] = valid_prices[-1][1]
            else:
                # Use first available price if timestamp is before data start
                prices[asset] = price_series[0][1]
        return prices

    def _execute_trades(self, trades: List[Trade]):
        """Execute trades and update portfolio holdings."""
        for trade in trades:
            if trade.action == 'sell':
                self.current_holdings[trade.asset] -= trade.quantity
            else:  # buy
                self.current_holdings[trade.asset] = (
                    self.current_holdings.get(trade.asset, 0.0) + trade.quantity
                )

            self.total_fees_paid += trade.fee_usd
            self.trades.append(trade)

        if trades:
            self.last_rebalance_time = trades[0].timestamp

    def run(self) -> SimulationResult:
        """Run the simulation and return results."""
        logger.info(f"Starting simulation with {self.strategy.name} strategy")
        logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")

        # Get all timestamps we need to check
        timestamps = self._get_check_timestamps()

        for ts in timestamps:
            prices = self._get_prices_at_time(ts)

            # Create current state
            state = PortfolioState(
                timestamp=ts,
                holdings=self.current_holdings.copy(),
                prices=prices
            )
            self.portfolio_history.append(state)

            # Check if we should rebalance
            should_rebalance, reason = self.strategy.should_rebalance(
                current_state=state,
                target_allocation=self.config.target_allocation,
                last_rebalance_time=self.last_rebalance_time
            )

            if should_rebalance:
                logger.info(f"Rebalancing at {ts}: {reason}")
                trades = self.strategy.calculate_trades(
                    current_state=state,
                    target_allocation=self.config.target_allocation,
                    fee_rate=self.config.fee_rate
                )
                self._execute_trades(trades)
                logger.info(f"Executed {len(trades)} trades, fees: ${sum(t.fee_usd for t in trades):.2f}")

        # Calculate performance metrics
        return self._calculate_results()

    def _get_check_timestamps(self) -> List[datetime]:
        """Generate list of timestamps to check for rebalancing."""
        timestamps = []
        current = self.config.start_date
        interval = timedelta(hours=self.config.price_check_interval_hours)

        while current <= self.config.end_date:
            timestamps.append(current)
            current += interval

        return timestamps

    def _calculate_results(self) -> SimulationResult:
        """Calculate performance metrics from simulation."""
        if not self.portfolio_history:
            raise ValueError("No portfolio history - simulation not run?")

        initial_value = self.portfolio_history[0].total_value_usd
        final_value = self.portfolio_history[-1].total_value_usd

        # Calculate returns
        total_return_pct = ((final_value - initial_value) / initial_value) * 100

        # Calculate annualized return
        days = (self.config.end_date - self.config.start_date).days
        years = days / 365.25
        annualized_return_pct = ((final_value / initial_value) ** (1 / years) - 1) * 100 if years > 0 else 0

        # Calculate daily returns for Sharpe ratio
        daily_returns = []
        for i in range(1, len(self.portfolio_history)):
            prev_val = self.portfolio_history[i-1].total_value_usd
            curr_val = self.portfolio_history[i].total_value_usd
            daily_returns.append((curr_val - prev_val) / prev_val)

        # Sharpe ratio (assuming 0% risk-free rate, annualized)
        if daily_returns:
            import statistics
            mean_return = statistics.mean(daily_returns)
            std_return = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0
            sharpe = (mean_return / std_return * (365.25 ** 0.5)) if std_return > 0 else 0
        else:
            sharpe = 0

        # Calculate max drawdown
        peak_value = initial_value
        max_drawdown = 0
        for state in self.portfolio_history:
            if state.total_value_usd > peak_value:
                peak_value = state.total_value_usd
            drawdown = ((peak_value - state.total_value_usd) / peak_value) * 100
            max_drawdown = max(max_drawdown, drawdown)

        # Count rebalancing events
        num_rebalances = len(set(t.timestamp for t in self.trades)) if self.trades else 0

        return SimulationResult(
            strategy_name=self.strategy.name,
            config=self.config,
            initial_value=initial_value,
            final_value=final_value,
            total_return_percent=total_return_pct,
            annualized_return_percent=annualized_return_pct,
            sharpe_ratio=sharpe,
            max_drawdown_percent=max_drawdown,
            total_fees_paid=self.total_fees_paid,
            num_rebalances=num_rebalances,
            trades=self.trades,
            portfolio_history=self.portfolio_history
        )


class HistoricalPriceFetcher:
    """Fetches historical price data from Coinbase API with batching and caching."""

    # Coinbase API typically returns ~300 candles max per request
    MAX_CANDLES_PER_REQUEST = 300

    def __init__(
        self,
        coinbase_client,
        use_cache: bool = True,
        cache_max_age_days: Optional[int] = 7,
        request_delay: float = 0.3
    ):
        """
        Initialize fetcher.

        Args:
            coinbase_client: CoinbaseClient instance
            use_cache: Whether to use disk caching
            cache_max_age_days: Maximum age of cache to use (None = no limit)
            request_delay: Delay between API requests in seconds
        """
        self.client = coinbase_client
        self.use_cache = use_cache
        self.cache_max_age_days = cache_max_age_days
        self.request_delay = request_delay

        if self.use_cache:
            from src.price_cache import PriceCache
            self.cache = PriceCache()
        else:
            self.cache = None

    def _calculate_candles_needed(
        self,
        start_date: datetime,
        end_date: datetime,
        granularity: str
    ) -> int:
        """Calculate number of candles needed for time period."""
        duration = end_date - start_date

        if granularity == "ONE_MINUTE":
            return int(duration.total_seconds() / 60)
        elif granularity == "FIVE_MINUTE":
            return int(duration.total_seconds() / 300)
        elif granularity == "FIFTEEN_MINUTE":
            return int(duration.total_seconds() / 900)
        elif granularity == "THIRTY_MINUTE":
            return int(duration.total_seconds() / 1800)
        elif granularity == "ONE_HOUR":
            return int(duration.total_seconds() / 3600)
        elif granularity == "TWO_HOUR":
            return int(duration.total_seconds() / 7200)
        elif granularity == "SIX_HOUR":
            return int(duration.total_seconds() / 21600)
        elif granularity == "ONE_DAY":
            return duration.days + 1
        else:
            # Default to hourly
            return int(duration.total_seconds() / 3600)

    def _calculate_chunks(
        self,
        start_date: datetime,
        end_date: datetime,
        granularity: str
    ) -> List[Tuple[datetime, datetime]]:
        """
        Calculate time chunks for batched requests.

        Returns:
            List of (chunk_start, chunk_end) tuples
        """
        candles_needed = self._calculate_candles_needed(start_date, end_date, granularity)

        if candles_needed <= self.MAX_CANDLES_PER_REQUEST:
            # Single request is enough
            return [(start_date, end_date)]

        # Need multiple chunks
        import math
        num_chunks = math.ceil(candles_needed / self.MAX_CANDLES_PER_REQUEST)

        total_seconds = (end_date - start_date).total_seconds()
        chunk_seconds = total_seconds / num_chunks

        chunks = []
        for i in range(num_chunks):
            chunk_start = start_date + timedelta(seconds=i * chunk_seconds)
            chunk_end = start_date + timedelta(seconds=(i + 1) * chunk_seconds)

            # Last chunk should end exactly at end_date
            if i == num_chunks - 1:
                chunk_end = end_date

            chunks.append((chunk_start, chunk_end))

        return chunks

    def _fetch_chunk(
        self,
        product_id: str,
        start: datetime,
        end: datetime,
        granularity: str
    ) -> List[Tuple[datetime, float]]:
        """Fetch a single chunk of candle data."""
        try:
            candles = self.client.get_candles(
                product_id=product_id,
                start=int(start.timestamp()),
                end=int(end.timestamp()),
                granularity=granularity
            )

            prices = []
            for candle in candles.get('candles', []):
                timestamp = datetime.fromtimestamp(int(candle['start']))
                close_price = float(candle['close'])
                prices.append((timestamp, close_price))

            return prices

        except Exception as e:
            logger.error(f"Error fetching chunk for {product_id} ({start} to {end}): {e}")
            raise

    def fetch_asset_prices(
        self,
        asset: str,
        start_date: datetime,
        end_date: datetime,
        granularity: str,
        show_progress: bool = True,
        cache_only: bool = False
    ) -> List[Tuple[datetime, float]]:
        """
        Fetch prices for a single asset with batching and caching.

        Args:
            asset: Asset symbol (e.g., 'BTC')
            start_date: Start of period
            end_date: End of period
            granularity: Data granularity
            show_progress: Whether to show progress bar
            cache_only: If True, only use cache and fail if not available

        Returns:
            List of (timestamp, price) tuples
        """
        # Stablecoins pegged to USD (price = 1.0)
        STABLECOINS = {'USDC', 'USDT', 'DAI', 'USD'}

        if asset in STABLECOINS:
            logger.info(f"{asset}: Stablecoin, using $1.00")

            # Generate timestamps
            prices = []
            current = start_date

            # Map granularity to timedelta
            granularity_map = {
                "ONE_MINUTE": timedelta(minutes=1),
                "FIVE_MINUTE": timedelta(minutes=5),
                "FIFTEEN_MINUTE": timedelta(minutes=15),
                "THIRTY_MINUTE": timedelta(minutes=30),
                "ONE_HOUR": timedelta(hours=1),
                "TWO_HOUR": timedelta(hours=2),
                "SIX_HOUR": timedelta(hours=6),
                "ONE_DAY": timedelta(days=1),
            }

            delta = granularity_map.get(granularity, timedelta(hours=1))

            while current <= end_date:
                prices.append((current, 1.0))
                current += delta

            logger.info(f"{asset}: Generated {len(prices)} candles")
            return prices

        # Check cache first
        if self.cache:
            cached_prices = self.cache.get(
                asset, start_date, end_date, granularity, self.cache_max_age_days
            )
            if cached_prices is not None:
                return cached_prices
            elif cache_only:
                raise ValueError(
                    f"Cache-only mode enabled but no cached data found for {asset} "
                    f"({start_date.date()} to {end_date.date()}, {granularity})"
                )
        elif cache_only:
            raise ValueError("Cache-only mode enabled but cache is disabled (use --no-cache)")

        # Not in cache, need to fetch
        product_id = f"{asset}-USD"
        chunks = self._calculate_chunks(start_date, end_date, granularity)

        logger.info(f"{asset}: Fetching {len(chunks)} chunk(s)")

        all_prices = []

        # Set up progress bar if requested
        if show_progress:
            try:
                from tqdm import tqdm
                chunk_iter = tqdm(chunks, desc=f"{asset:6s}", unit="chunk")
            except ImportError:
                chunk_iter = chunks
        else:
            chunk_iter = chunks

        for chunk_start, chunk_end in chunk_iter:
            chunk_prices = self._fetch_chunk(product_id, chunk_start, chunk_end, granularity)
            all_prices.extend(chunk_prices)

            # Rate limiting between chunks
            if len(chunks) > 1:
                time.sleep(self.request_delay)

        # Sort by timestamp and remove duplicates
        all_prices.sort(key=lambda x: x[0])

        # Remove duplicates (keep first occurrence)
        seen_timestamps = set()
        unique_prices = []
        for ts, price in all_prices:
            if ts not in seen_timestamps:
                seen_timestamps.add(ts)
                unique_prices.append((ts, price))

        logger.info(f"{asset}: Fetched {len(unique_prices)} candles total")

        # Save to cache
        if self.cache:
            self.cache.set(asset, start_date, end_date, granularity, unique_prices)

        return unique_prices

    def fetch_historical_prices(
        self,
        assets: List[str],
        start_date: datetime,
        end_date: datetime,
        granularity: str = "ONE_DAY",
        show_progress: bool = True,
        cache_only: bool = False
    ) -> Dict[str, List[Tuple[datetime, float]]]:
        """
        Fetch historical prices for multiple assets.

        Args:
            assets: List of asset symbols (e.g., ['BTC', 'ETH'])
            start_date: Start of historical period
            end_date: End of historical period
            granularity: Candle granularity
            show_progress: Whether to show progress bars
            cache_only: If True, only use cache and fail if not available

        Returns:
            Dict mapping asset -> [(timestamp, price), ...]
        """
        logger.info(f"Fetching historical prices for {len(assets)} assets")
        logger.info(f"Period: {start_date.date()} to {end_date.date()}, granularity: {granularity}")

        # Calculate total candles needed
        candles_per_asset = self._calculate_candles_needed(start_date, end_date, granularity)
        logger.info(f"~{candles_per_asset} candles per asset")

        price_data = {}

        for asset in assets:
            price_data[asset] = self.fetch_asset_prices(
                asset, start_date, end_date, granularity, show_progress, cache_only
            )

        return price_data


def print_simulation_report(results: List[SimulationResult]):
    """Print a formatted comparison report of simulation results."""

    print("\n" + "="*80)
    print("REBALANCING STRATEGY SIMULATION RESULTS")
    print("="*80)

    # Find best performers
    best_return = max(results, key=lambda r: r.total_return_percent)
    best_sharpe = max(results, key=lambda r: r.sharpe_ratio)
    lowest_drawdown = min(results, key=lambda r: r.max_drawdown_percent)
    lowest_fees = min(results, key=lambda r: r.total_fees_paid)

    # Print each strategy
    for result in results:
        print(f"\n{result.strategy_name}")
        print("-" * 80)
        print(f"Period: {result.config.start_date.date()} to {result.config.end_date.date()}")
        print(f"Initial Value: ${result.initial_value:,.2f}")
        print(f"Final Value: ${result.final_value:,.2f}")
        print(f"Total Return: {result.total_return_percent:+.2f}% {'⭐ BEST' if result == best_return else ''}")
        print(f"Annualized Return: {result.annualized_return_percent:+.2f}%")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.3f} {'⭐ BEST' if result == best_sharpe else ''}")
        print(f"Max Drawdown: {result.max_drawdown_percent:.2f}% {'⭐ BEST' if result == lowest_drawdown else ''}")
        print(f"Total Fees Paid: ${result.total_fees_paid:,.2f} {'⭐ LOWEST' if result == lowest_fees else ''}")
        print(f"Number of Rebalances: {result.num_rebalances}")
        print(f"Total Trades: {len(result.trades)}")

        if result.num_rebalances > 0:
            avg_fee_per_rebalance = result.total_fees_paid / result.num_rebalances
            print(f"Avg Fee per Rebalance: ${avg_fee_per_rebalance:.2f}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Best Total Return: {best_return.strategy_name} ({best_return.total_return_percent:+.2f}%)")
    print(f"Best Risk-Adjusted (Sharpe): {best_sharpe.strategy_name} ({best_sharpe.sharpe_ratio:.3f})")
    print(f"Lowest Drawdown: {lowest_drawdown.strategy_name} ({lowest_drawdown.max_drawdown_percent:.2f}%)")
    print(f"Lowest Fees: {lowest_fees.strategy_name} (${lowest_fees.total_fees_paid:,.2f})")
    print("="*80 + "\n")


def save_simulation_results(results: List[SimulationResult], output_path: str):
    """Save simulation results to JSON file."""
    data = {
        'simulation_date': datetime.now().isoformat(),
        'results': [r.to_dict() for r in results]
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved simulation results to {output_path}")
