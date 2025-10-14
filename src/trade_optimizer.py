"""Trade optimization logic for capital-efficient rebalancing."""
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class TradeRoute:
    """Represents a single trade route."""
    from_asset: str
    to_asset: str
    value_usd: float
    product_id: str
    is_direct: bool  # True if direct pair, False if routed through USD


class TradeOptimizer:
    """Optimizes rebalancing trades to minimize fees and slippage."""

    def __init__(self):
        """Initialize trade optimizer."""
        self.logger = logging.getLogger(__name__)

    def calculate_optimal_trades(
        self,
        to_sell: Dict[str, float],  # {asset: value_to_sell_usd}
        to_buy: Dict[str, float],   # {asset: value_to_buy_usd}
        available_pairs: Dict[str, bool],  # Available trading pairs
        prefer_direct: bool = True
    ) -> List[Dict]:
        """
        Calculate optimal trade routes to minimize fees.

        Args:
            to_sell: Assets to sell with USD values
            to_buy: Assets to buy with USD values
            available_pairs: Available trading pairs on exchange
            prefer_direct: Whether to prefer direct trades over USD routing

        Returns:
            List of optimized trade dictionaries
        """
        trades = []

        # Make copies to track remaining balances
        remaining_sells = dict(to_sell)
        remaining_buys = dict(to_buy)

        self.logger.info("Starting trade optimization")
        self.logger.info(f"Assets to sell: {list(remaining_sells.keys())}")
        self.logger.info(f"Assets to buy: {list(remaining_buys.keys())}")

        # Step 1: Find direct trading pair opportunities
        if prefer_direct:
            direct_trades = self._find_direct_trades(
                remaining_sells, remaining_buys, available_pairs
            )
            trades.extend(direct_trades)

            self.logger.info(f"Found {len(direct_trades)} direct trade opportunities")

            # Update remaining amounts after direct trades
            for trade in direct_trades:
                from_asset = trade['from_asset']
                to_asset = trade['to_asset']
                value = trade['value_usd']

                if from_asset in remaining_sells:
                    remaining_sells[from_asset] -= value
                    if remaining_sells[from_asset] <= 0:
                        del remaining_sells[from_asset]

                if to_asset in remaining_buys:
                    remaining_buys[to_asset] -= value
                    if remaining_buys[to_asset] <= 0:
                        del remaining_buys[to_asset]

        # Step 2: Route remaining trades through USD
        usd_routed_trades = self._route_through_usd(remaining_sells, remaining_buys)
        trades.extend(usd_routed_trades)

        self.logger.info(f"Added {len(usd_routed_trades)} USD-routed trades")
        self.logger.info(f"Total optimized trades: {len(trades)}")

        # Calculate savings
        self._log_optimization_savings(trades, to_sell, to_buy)

        return trades

    def _find_direct_trades(
        self,
        to_sell: Dict[str, float],
        to_buy: Dict[str, float],
        available_pairs: Dict[str, bool]
    ) -> List[Dict]:
        """
        Find opportunities for direct trades between assets.

        Args:
            to_sell: Assets to sell
            to_buy: Assets to buy
            available_pairs: Available trading pairs

        Returns:
            List of direct trade dictionaries
        """
        direct_trades = []

        # Try to match each seller with each buyer
        for sell_asset, sell_value in list(to_sell.items()):
            for buy_asset, buy_value in list(to_buy.items()):
                if sell_asset == buy_asset:
                    continue

                # Check if direct pair exists
                pair_format = self._get_pair_format(sell_asset, buy_asset, available_pairs)

                if pair_format:
                    # Calculate trade amount (minimum of sell and buy)
                    trade_value = min(sell_value, buy_value)

                    # Determine trade direction based on pair format
                    base, quote = pair_format.split('-')

                    if base == sell_asset:
                        # Selling base, buying quote
                        side = 'SELL'
                        from_asset = sell_asset
                        to_asset = buy_asset
                    else:
                        # Buying base, selling quote
                        side = 'BUY'
                        from_asset = sell_asset
                        to_asset = buy_asset

                    direct_trades.append({
                        'type': 'direct',
                        'from_asset': from_asset,
                        'to_asset': to_asset,
                        'product_id': pair_format,
                        'side': side,
                        'value_usd': trade_value,
                        'is_direct': True,
                        'reason': f"Direct swap: {from_asset} → {to_asset}"
                    })

                    self.logger.info(
                        f"Direct trade: {from_asset} → {to_asset} "
                        f"({pair_format}) ${trade_value:.2f}"
                    )

        return direct_trades

    def _route_through_usd(
        self,
        to_sell: Dict[str, float],
        to_buy: Dict[str, float]
    ) -> List[Dict]:
        """
        Create trades routed through USD for remaining amounts.

        Args:
            to_sell: Remaining assets to sell
            to_buy: Remaining assets to buy

        Returns:
            List of USD-routed trades
        """
        trades = []

        # Create sell orders (ASSET → USD)
        for asset, value in to_sell.items():
            if value > 0 and asset not in ['USD', 'USDC']:
                trades.append({
                    'type': 'usd_sell',
                    'from_asset': asset,
                    'to_asset': 'USD',
                    'product_id': f"{asset}-USD",
                    'side': 'SELL',
                    'value_usd': value,
                    'is_direct': False,
                    'reason': f"Sell {asset} to USD"
                })

        # Create buy orders (USD → ASSET)
        for asset, value in to_buy.items():
            if value > 0 and asset not in ['USD', 'USDC']:
                trades.append({
                    'type': 'usd_buy',
                    'from_asset': 'USD',
                    'to_asset': asset,
                    'product_id': f"{asset}-USD",
                    'side': 'BUY',
                    'value_usd': value,
                    'is_direct': False,
                    'reason': f"Buy {asset} with USD"
                })

        return trades

    def _get_pair_format(
        self,
        asset1: str,
        asset2: str,
        available_pairs: Dict[str, bool]
    ) -> Optional[str]:
        """
        Get the correct trading pair format if it exists.

        Args:
            asset1: First asset
            asset2: Second asset
            available_pairs: Available pairs dictionary

        Returns:
            Product ID string or None
        """
        pair_1 = f"{asset1}-{asset2}"
        pair_2 = f"{asset2}-{asset1}"

        if pair_1 in available_pairs:
            return pair_1
        elif pair_2 in available_pairs:
            return pair_2
        return None

    def _log_optimization_savings(
        self,
        optimized_trades: List[Dict],
        original_sells: Dict[str, float],
        original_buys: Dict[str, float]
    ) -> None:
        """
        Log the estimated savings from trade optimization.

        Args:
            optimized_trades: Optimized trade list
            original_sells: Original sell amounts
            original_buys: Original buy amounts
        """
        # Count direct vs routed trades
        direct_count = sum(1 for t in optimized_trades if t.get('is_direct', False))
        total_count = len(optimized_trades)

        # Calculate naive approach trade count
        naive_count = len(original_sells) + len(original_buys)

        # Estimate fee savings (assuming 0.6% fee per trade)
        fee_rate = 0.006

        # Calculate total value traded via direct pairs
        direct_value = sum(t['value_usd'] for t in optimized_trades if t.get('is_direct', False))

        # Fees saved: direct trades pay 1 fee instead of 2
        estimated_savings = direct_value * fee_rate

        self.logger.info("=" * 60)
        self.logger.info("TRADE OPTIMIZATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Naive approach trades: {naive_count}")
        self.logger.info(f"Optimized trades: {total_count}")
        self.logger.info(f"Direct pair trades: {direct_count}")
        self.logger.info(f"USD-routed trades: {total_count - direct_count}")
        self.logger.info(f"Direct trade volume: ${direct_value:.2f}")
        self.logger.info(f"Estimated fee savings: ${estimated_savings:.2f}")
        self.logger.info("=" * 60)

    def convert_to_execution_format(
        self,
        optimized_trades: List[Dict],
        current_balances: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> List[Dict]:
        """
        Convert optimized trades to execution format with sizes.

        Args:
            optimized_trades: List of optimized trades
            current_balances: Current asset balances
            current_prices: Current asset prices in USD

        Returns:
            List of trades ready for execution
        """
        execution_trades = []

        for trade in optimized_trades:
            product_id = trade['product_id']
            side = trade['side']
            value_usd = trade['value_usd']
            from_asset = trade['from_asset']
            to_asset = trade['to_asset']

            execution_trade = {
                'asset': from_asset if side == 'SELL' else to_asset,
                'action': side,
                'product_id': product_id,
                'value_usd': value_usd,
                'reason': trade['reason'],
                'is_direct': trade.get('is_direct', False),
                'from_asset': from_asset,
                'to_asset': to_asset,
            }

            # Calculate size based on side and available balance
            if side == 'SELL':
                # Selling from_asset
                if from_asset in current_balances and from_asset in current_prices:
                    current_value = current_balances[from_asset] * current_prices.get(from_asset, 0)
                    if current_value > 0:
                        size = (value_usd / current_value) * current_balances[from_asset]
                        execution_trade['size'] = size
            else:  # BUY
                # Buying with USD value
                execution_trade['quote_size'] = value_usd

            execution_trades.append(execution_trade)

        return execution_trades
