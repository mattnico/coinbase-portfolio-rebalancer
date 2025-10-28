"""
Adaptive rebalancing strategy for Monte Carlo simulations.

This strategy uses regime detection to dynamically switch between different
portfolio allocations based on market conditions.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import Counter

from src.monte_carlo_simulator import RebalancingStrategy, PortfolioState, Trade
from src.regime_detector import ReturnDetector, MarketRegime

logger = logging.getLogger(__name__)


class AdaptiveRebalancingStrategy(RebalancingStrategy):
    """
    Rebalancing strategy that adapts allocation based on detected market regime.

    Switches between different portfolio allocations depending on whether the
    market is in a BULL, BEAR, or NEUTRAL regime.
    """

    def __init__(
        self,
        regime_portfolios: Dict[MarketRegime, Dict[str, float]],
        detector: Optional[ReturnDetector] = None,
        check_frequency_days: int = 7,
        persistence_days: int = 14,
        threshold_percent: float = 5.0,
        name: str = "Adaptive"
    ):
        """
        Initialize adaptive strategy.

        Args:
            regime_portfolios: Mapping of regime to target allocation
                Example: {
                    MarketRegime.BULL: {'BTC': 40, 'ETH': 40, 'SOL': 20},
                    MarketRegime.BEAR: {'BTC': 20, 'ETH': 20, 'USDC': 60},
                    MarketRegime.NEUTRAL: {'BTC': 40, 'ETH': 40, 'SOL': 20}
                }
            detector: ReturnDetector instance (default: bull=15%, bear=-10%)
            check_frequency_days: Days between regime checks
            persistence_days: Days regime must persist before switching
            threshold_percent: Allocation deviation threshold for rebalancing
            name: Strategy name
        """
        super().__init__(name)

        self.regime_portfolios = regime_portfolios
        self.detector = detector or ReturnDetector(
            window_days=30,
            bull_threshold=15.0,
            bear_threshold=-10.0
        )
        self.check_frequency_days = check_frequency_days
        self.persistence_days = persistence_days
        self.threshold_percent = threshold_percent

        # State tracking
        self.current_regime = MarketRegime.NEUTRAL
        self.current_allocation = regime_portfolios[MarketRegime.NEUTRAL]
        self.regime_buffer: List[MarketRegime] = []
        self.last_regime_check: Optional[datetime] = None
        self.last_regime_switch: Optional[datetime] = None

        logger.debug(f"Initialized {name} strategy")
        logger.debug(f"  Check frequency: {check_frequency_days} days")
        logger.debug(f"  Persistence: {persistence_days} days")
        logger.debug(f"  Threshold: {threshold_percent}%")

    def _get_price_history(
        self,
        timestamp: datetime,
        price_data: Dict[str, List[Tuple[datetime, float]]],
        window_days: int = 30
    ) -> Tuple[List[float], List[datetime]]:
        """
        Extract price history for regime detection.

        Args:
            timestamp: Current timestamp
            price_data: Full price history
            window_days: Number of days to look back

        Returns:
            Tuple of (prices, timestamps) for the window
        """
        # Use BTC as reference asset
        if 'BTC' not in price_data:
            logger.warning("BTC not in price data, cannot detect regime")
            return [], []

        btc_prices = price_data['BTC']

        # Find prices in the window before current timestamp
        window_start = timestamp - timedelta(days=window_days)

        window_prices = [
            (ts, price) for ts, price in btc_prices
            if window_start <= ts <= timestamp
        ]

        if not window_prices:
            return [], []

        prices = [p[1] for p in window_prices]
        timestamps = [p[0] for p in window_prices]

        return prices, timestamps

    def _detect_regime(
        self,
        timestamp: datetime,
        price_data: Dict[str, List[Tuple[datetime, float]]]
    ) -> Optional[MarketRegime]:
        """
        Detect current market regime.

        Args:
            timestamp: Current timestamp
            price_data: Full price history

        Returns:
            Detected regime or None if detection fails
        """
        prices, timestamps = self._get_price_history(timestamp, price_data)

        if not prices or len(prices) < 2:
            logger.debug(f"Insufficient price data at {timestamp}")
            return None

        try:
            detection = self.detector.detect(prices, timestamps)
            logger.debug(f"Detected {detection.regime.value} at {timestamp} "
                        f"(return: {detection.return_30d:.2f}%, confidence: {detection.confidence:.3f})")
            return detection.regime
        except Exception as e:
            logger.warning(f"Error detecting regime: {e}")
            return None

    def _update_regime(
        self,
        timestamp: datetime,
        price_data: Dict[str, List[Tuple[datetime, float]]]
    ) -> bool:
        """
        Update current regime based on detection and persistence.

        Args:
            timestamp: Current timestamp
            price_data: Full price history

        Returns:
            True if regime changed, False otherwise
        """
        # Check if it's time to check regime
        if self.last_regime_check is not None:
            days_since_check = (timestamp - self.last_regime_check).days
            if days_since_check < self.check_frequency_days:
                return False

        # Detect regime
        detected_regime = self._detect_regime(timestamp, price_data)

        if detected_regime is None:
            return False

        self.last_regime_check = timestamp

        # Update regime buffer
        self.regime_buffer.append(detected_regime)
        buffer_size = max(2, self.persistence_days // self.check_frequency_days)
        if len(self.regime_buffer) > buffer_size:
            self.regime_buffer.pop(0)

        # Check if regime change is confirmed
        if len(self.regime_buffer) >= 2:
            regime_counts = Counter(self.regime_buffer)
            most_common_regime = regime_counts.most_common(1)[0][0]

            if most_common_regime != self.current_regime:
                logger.info(f"🔄 Regime change at {timestamp}: {self.current_regime.value} → {most_common_regime.value}")
                logger.info(f"   Buffer: {[r.value for r in self.regime_buffer]}")

                old_regime = self.current_regime
                self.current_regime = most_common_regime
                self.current_allocation = self.regime_portfolios[self.current_regime]
                self.last_regime_switch = timestamp

                return True

        return False

    def should_rebalance(
        self,
        current_state: PortfolioState,
        target_allocation: Dict[str, float],
        last_rebalance_time: Optional[datetime],
        price_data: Optional[Dict[str, List[Tuple[datetime, float]]]] = None,
        **kwargs
    ) -> Tuple[bool, str]:
        """
        Determine if portfolio should be rebalanced.

        This checks:
        1. If regime has changed (forces rebalance)
        2. If allocation threshold is exceeded (normal rebalance)

        Args:
            current_state: Current portfolio state
            target_allocation: Ignored (we use dynamic allocation)
            last_rebalance_time: When last rebalanced
            price_data: Full price history (needed for regime detection)

        Returns:
            Tuple of (should_rebalance, reason)
        """
        if price_data is None:
            logger.warning("No price data provided to adaptive strategy")
            return False, "No price data"

        # Update regime (may trigger regime change)
        regime_changed = self._update_regime(current_state.timestamp, price_data)

        if regime_changed:
            return True, f"Regime changed to {self.current_regime.value}"

        # Use current dynamic allocation (not the static one passed in)
        dynamic_target = self.current_allocation

        # Check threshold-based rebalancing
        deviations = current_state.get_deviation_from_target(dynamic_target)
        max_deviation = max(abs(d) for d in deviations.values())

        if max_deviation >= self.threshold_percent:
            return True, f"Threshold exceeded: {max_deviation:.2f}% >= {self.threshold_percent}%"

        return False, f"Within threshold ({max_deviation:.2f}% < {self.threshold_percent}%)"

    def calculate_trades(
        self,
        current_state: PortfolioState,
        target_allocation: Dict[str, float],
        fee_rate: float,
        min_trade_value_usd: float = 10.0,
        price_data: Optional[Dict[str, List[Tuple[datetime, float]]]] = None,
        **kwargs
    ) -> List[Trade]:
        """
        Calculate trades needed to rebalance to current dynamic allocation.

        Args:
            current_state: Current portfolio state
            target_allocation: Ignored (we use dynamic allocation)
            fee_rate: Trading fee rate
            min_trade_value_usd: Minimum trade value
            price_data: Price data (not needed here, but accepted for compatibility)

        Returns:
            List of trades to execute
        """
        # Use dynamic allocation instead of static
        dynamic_target = self.current_allocation

        trades = []
        current_alloc = current_state.get_current_allocation()

        # Get all assets (both in target and currently held)
        all_assets = set(dynamic_target.keys()) | set(current_state.holdings.keys())

        # Calculate target values (0 for assets not in target = sell them)
        target_values = {
            asset: current_state.total_value_usd * (dynamic_target.get(asset, 0.0) / 100)
            for asset in all_assets
        }

        # Calculate current values for all assets
        current_values = {
            asset: current_state.holdings.get(asset, 0.0) * current_state.prices.get(asset, 0.0)
            for asset in all_assets
        }

        # Determine what to buy and sell
        to_buy = {}
        to_sell = {}

        for asset in all_assets:
            current_val = current_values.get(asset, 0.0)
            target_val = target_values.get(asset, 0.0)
            diff = target_val - current_val

            if diff > min_trade_value_usd:
                to_buy[asset] = diff
            elif diff < -min_trade_value_usd:
                to_sell[asset] = abs(diff)

        # Calculate total sell and buy amounts
        total_sell_value = sum(to_sell.values())
        total_buy_value = sum(to_buy.values())

        # Fee is charged once on the total rebalanced amount (models direct pair trading)
        rebalance_amount = min(total_sell_value, total_buy_value)
        total_fee = rebalance_amount * fee_rate

        # Distribute fee proportionally across sell trades
        for asset, value_usd in to_sell.items():
            price = current_state.prices.get(asset, 0.0)
            if price <= 0:
                continue
            quantity = value_usd / price
            # Proportional fee for this sell
            fee = (value_usd / total_sell_value) * total_fee if total_sell_value > 0 else 0

            trades.append(Trade(
                timestamp=current_state.timestamp,
                asset=asset,
                action='sell',
                quantity=quantity,
                price=price,
                value_usd=value_usd,
                fee_usd=fee,
                reason=f'Rebalancing to {self.current_regime.value} regime'
            ))

        # Create buy trades with no additional fees (fee already charged on sells)
        for asset, value_usd in to_buy.items():
            price = current_state.prices.get(asset, 0.0)
            if price <= 0:
                continue
            # Reduce buy amount to account for fees paid on sells
            available_for_this_buy = value_usd - (value_usd / total_buy_value) * total_fee if total_buy_value > 0 else value_usd
            quantity = available_for_this_buy / price

            trades.append(Trade(
                timestamp=current_state.timestamp,
                asset=asset,
                action='buy',
                quantity=quantity,
                price=price,
                value_usd=available_for_this_buy,
                fee_usd=0.0,
                reason=f'Rebalancing to {self.current_regime.value} regime'
            ))

        return trades

    def get_current_target_allocation(self) -> Dict[str, float]:
        """
        Get the current dynamic target allocation based on regime.

        Returns:
            Current target allocation dict
        """
        return self.current_allocation.copy()

    def get_regime_history(self) -> List[MarketRegime]:
        """
        Get history of detected regimes.

        Returns:
            List of regimes in buffer
        """
        return self.regime_buffer.copy()
