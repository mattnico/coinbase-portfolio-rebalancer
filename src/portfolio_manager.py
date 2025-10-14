"""Portfolio management and rebalancing logic."""
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from src.coinbase_client import CoinbaseClient
from src.transaction_logger import TransactionLogger
from src.trade_optimizer import TradeOptimizer
from src.utils import load_json, format_currency, format_percentage


class PortfolioManager:
    """Manages portfolio rebalancing operations."""

    def __init__(self, config_path: str = "config/portfolio.json",
                 coinbase_client: Optional[CoinbaseClient] = None,
                 transaction_logger: Optional[TransactionLogger] = None):
        """Initialize portfolio manager."""
        self.config = load_json(config_path)
        self.config_path = config_path

        # Extract portfolio_id from config and pass to CoinbaseClient
        portfolio_id = self.config.get('portfolio_id')
        self.client = coinbase_client or CoinbaseClient(portfolio_id=portfolio_id)
        self.tx_logger = transaction_logger or TransactionLogger()
        self.optimizer = TradeOptimizer()
        self.logger = logging.getLogger(__name__)

    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.config = load_json(self.config_path)
        self.logger.info("Configuration reloaded")

    def get_current_allocation(self) -> Tuple[Dict[str, float], float, Dict[str, float]]:
        """
        Get current portfolio allocation percentages.

        Returns:
            Tuple of (allocation_percentages, total_value_usd, individual_values_usd)
        """
        balances = self.client.get_accounts()
        total_value, values = self.client.get_portfolio_value_usd(balances)

        allocation = {}
        for asset, value in values.items():
            if total_value > 0:
                allocation[asset] = (value / total_value) * 100
            else:
                allocation[asset] = 0.0

        return allocation, total_value, values

    def calculate_rebalancing_trades(self, current_allocation: Dict[str, float],
                                    current_values: Dict[str, float],
                                    total_value: float) -> List[Dict]:
        """
        Calculate capital-efficient trades needed to rebalance portfolio.
        Uses direct trading pairs when available to minimize fees.
        Automatically sells assets not in target allocation.

        Args:
            current_allocation: Current allocation percentages
            current_values: Current USD values per asset
            total_value: Total portfolio value in USD

        Returns:
            List of optimized trade dictionaries
        """
        target_allocation = self.config['target_allocation']
        threshold = self.config['rebalancing']['threshold_percent']
        min_trade_value = self.config['rebalancing']['min_trade_value_usd']

        # Get optimization preference from config (default to True)
        prefer_direct = self.config['rebalancing'].get('prefer_direct_routes', True)

        # Get unknown asset handling preference (default to 'sell')
        handle_unknown = self.config['rebalancing'].get('handle_unknown_assets', 'sell')

        # Identify unknown assets (not in target allocation)
        unknown_assets = {}
        for asset, value in current_values.items():
            if asset not in target_allocation and asset not in ['USD', 'USDC']:
                unknown_assets[asset] = value

        if unknown_assets:
            self.logger.warning(f"Found {len(unknown_assets)} asset(s) not in target allocation: {list(unknown_assets.keys())}")
            for asset, value in unknown_assets.items():
                self.logger.warning(f"  {asset}: {format_currency(value)} ({current_allocation.get(asset, 0):.2f}%)")

        # Calculate deviations for target assets
        deviations = {}
        for asset, target_pct in target_allocation.items():
            current_pct = current_allocation.get(asset, 0.0)
            deviation = current_pct - target_pct
            deviations[asset] = {
                'current_pct': current_pct,
                'target_pct': target_pct,
                'deviation_pct': deviation,
                'current_value': current_values.get(asset, 0.0),
                'target_value': (target_pct / 100) * total_value
            }

        # Identify assets to buy and sell (amounts in USD)
        to_sell_dict = {}
        to_buy_dict = {}

        # Handle unknown assets based on configuration
        if handle_unknown == 'sell' and unknown_assets:
            self.logger.info(f"Selling {len(unknown_assets)} unknown asset(s) per configuration")
            for asset, value in unknown_assets.items():
                if value >= min_trade_value:
                    to_sell_dict[asset] = value
                    self.logger.info(f"  Queuing {asset} for sale: {format_currency(value)}")
                else:
                    self.logger.info(f"  Skipping {asset} (below minimum trade value)")

        # Add target assets that need rebalancing
        for asset, data in deviations.items():
            if data['deviation_pct'] > threshold:
                value_to_sell = data['current_value'] - data['target_value']
                if value_to_sell >= min_trade_value:
                    to_sell_dict[asset] = value_to_sell

            elif data['deviation_pct'] < -threshold:
                value_to_buy = data['target_value'] - data['current_value']
                if value_to_buy >= min_trade_value:
                    to_buy_dict[asset] = value_to_buy

        # If we sold unknown assets, distribute proceeds to target allocation
        if unknown_assets and handle_unknown == 'sell':
            total_unknown_value = sum(unknown_assets.values())
            if total_unknown_value >= min_trade_value:
                self.logger.info(f"Distributing {format_currency(total_unknown_value)} from unknown assets to target allocation")

                # Calculate how much to add to each target asset based on target percentages
                for asset, target_pct in target_allocation.items():
                    additional_buy = (target_pct / 100) * total_unknown_value
                    if additional_buy >= min_trade_value / 10:  # Lower threshold for distribution
                        current_buy = to_buy_dict.get(asset, 0)
                        to_buy_dict[asset] = current_buy + additional_buy
                        self.logger.info(f"  Adding {format_currency(additional_buy)} to {asset} buy order")

        if not to_sell_dict and not to_buy_dict:
            return []

        # Get available trading pairs
        available_pairs = self.client.get_available_trading_pairs()

        # Use optimizer to find best trade routes
        optimized_trades = self.optimizer.calculate_optimal_trades(
            to_sell=to_sell_dict,
            to_buy=to_buy_dict,
            available_pairs=available_pairs,
            prefer_direct=prefer_direct
        )

        # Get current balances and prices for size calculations
        balances = self.client.get_accounts()
        prices = {}
        for asset in set(list(to_sell_dict.keys()) + list(to_buy_dict.keys())):
            try:
                if asset not in ['USD', 'USDC']:
                    prices[asset] = self.client.get_product_price(f"{asset}-USD")
                    time.sleep(0.1)
            except Exception as e:
                self.logger.warning(f"Could not get price for {asset}: {e}")

        # Convert to execution format
        execution_trades = self.optimizer.convert_to_execution_format(
            optimized_trades=optimized_trades,
            current_balances=balances,
            current_prices=prices
        )

        return execution_trades

    def execute_trade(self, trade: Dict, dry_run: bool = True) -> Dict:
        """
        Execute a single trade.

        Args:
            trade: Trade dictionary
            dry_run: If True, simulate trade without execution

        Returns:
            Trade result dictionary
        """
        if dry_run:
            self.logger.info(f"[DRY RUN] {trade['action']} {trade['asset']}: "
                           f"{format_currency(trade['value_usd'])} - {trade['reason']}")
            return {
                'success': True,
                'dry_run': True,
                'trade': trade,
                'order_id': f"dry_run_{uuid.uuid4().hex[:8]}",
                'timestamp': datetime.now().isoformat()
            }

        try:
            self.logger.info(f"Executing {trade['action']} order for {trade['asset']}")

            # Get pre-trade price for slippage calculation
            pre_trade_price = self.client.get_product_price(trade['product_id'])

            if trade['action'] == 'SELL':
                result = self.client.place_market_order(
                    product_id=trade['product_id'],
                    side='SELL',
                    size=trade['size']
                )
            else:  # BUY
                result = self.client.place_market_order(
                    product_id=trade['product_id'],
                    side='BUY',
                    quote_size=trade['quote_size']
                )

            # Wait briefly for order to be processed
            time.sleep(0.5)

            # Fetch complete order details including fees
            order_id = result.get('order_id')
            order_details = None
            if order_id:
                try:
                    order_details = self.client.get_order(order_id)
                except Exception as e:
                    self.logger.warning(f"Could not fetch order details for {order_id}: {e}")

            # Build comprehensive transaction record
            transaction = {
                'timestamp': datetime.now().isoformat(),
                'asset': trade['asset'],
                'action': trade['action'],
                'product_id': trade['product_id'],
                'size': trade.get('size'),
                'quote_size': trade.get('quote_size'),
                'value_usd': trade['value_usd'],
                'reason': trade['reason'],
                'order_id': order_id,
                'success': result.get('success', True),
                'pre_trade_price': pre_trade_price,
            }

            # Add detailed order information if available
            if order_details:
                transaction.update({
                    'filled_size': order_details.get('filled_size'),
                    'average_filled_price': order_details.get('average_filled_price'),
                    'filled_value': order_details.get('filled_value'),
                    'total_fees': order_details.get('total_fees'),
                    'fee_currency': order_details.get('fee_currency'),
                    'number_of_fills': order_details.get('number_of_fills'),
                    'order_status': order_details.get('status'),
                    'fills': order_details.get('fills', []),
                })

                # Calculate slippage if we have execution data
                if order_details.get('average_filled_price'):
                    avg_price = order_details['average_filled_price']
                    if trade['action'] == 'BUY':
                        slippage_pct = ((avg_price - pre_trade_price) / pre_trade_price) * 100
                    else:  # SELL
                        slippage_pct = ((pre_trade_price - avg_price) / pre_trade_price) * 100
                    transaction['slippage_percent'] = slippage_pct
                    transaction['slippage_usd'] = (avg_price - pre_trade_price) * order_details.get('filled_size', 0)

            self.tx_logger.log_transaction(transaction)

            return {
                'success': True,
                'dry_run': False,
                'trade': trade,
                'order_id': result.get('order_id'),
                'timestamp': transaction['timestamp']
            }

        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return {
                'success': False,
                'dry_run': False,
                'trade': trade,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def rebalance(self) -> Dict:
        """
        Execute portfolio rebalancing.

        Returns:
            Dictionary with rebalancing results
        """
        session_id = uuid.uuid4().hex[:8]
        self.logger.info(f"Starting rebalancing session: {session_id}")

        # Reload config to get latest settings
        self.reload_config()
        dry_run = self.config['rebalancing']['dry_run']

        try:
            # Get current state
            current_allocation, total_value, current_values = self.get_current_allocation()

            self.logger.info(f"Current portfolio value: {format_currency(total_value)}")
            self.logger.info("Current allocation:")
            for asset, pct in current_allocation.items():
                self.logger.info(f"  {asset}: {format_percentage(pct)} "
                               f"({format_currency(current_values[asset])})")

            # Calculate trades
            trades = self.calculate_rebalancing_trades(
                current_allocation, current_values, total_value
            )

            if not trades:
                self.logger.info("Portfolio is balanced. No trades needed.")
                return {
                    'session_id': session_id,
                    'trades_executed': 0,
                    'message': 'Portfolio is balanced',
                    'portfolio_value': total_value
                }

            # Execute trades
            results = []
            for trade in trades:
                result = self.execute_trade(trade, dry_run=dry_run)
                results.append(result)

                # Rate limiting between trades
                if not dry_run:
                    time.sleep(1)

            # Get final state
            final_allocation, final_value, final_values = self.get_current_allocation()

            # Log session
            session_data = {
                'session_id': session_id,
                'dry_run': dry_run,
                'portfolio_value_before': total_value,
                'portfolio_value_after': final_value,
                'trades': results
            }
            self.tx_logger.log_rebalance_session(session_data)

            successful_trades = sum(1 for r in results if r['success'])

            self.logger.info(f"Rebalancing complete: {successful_trades}/{len(results)} trades successful")

            return {
                'session_id': session_id,
                'trades_executed': successful_trades,
                'total_trades': len(results),
                'dry_run': dry_run,
                'portfolio_value_before': total_value,
                'portfolio_value_after': final_value,
                'results': results
            }

        except Exception as e:
            self.logger.error(f"Error during rebalancing: {e}")
            raise

    def get_portfolio_status(self) -> Dict:
        """
        Get current portfolio status without rebalancing.

        Returns:
            Dictionary with portfolio information
        """
        current_allocation, total_value, current_values = self.get_current_allocation()
        target_allocation = self.config['target_allocation']

        status = {
            'total_value': total_value,
            'assets': []
        }

        for asset in set(list(current_allocation.keys()) + list(target_allocation.keys())):
            current_pct = current_allocation.get(asset, 0.0)
            target_pct = target_allocation.get(asset, 0.0)
            deviation = current_pct - target_pct

            status['assets'].append({
                'asset': asset,
                'current_value': current_values.get(asset, 0.0),
                'current_allocation': current_pct,
                'target_allocation': target_pct,
                'deviation': deviation
            })

        return status
