"""Integration tests for portfolio rebalancing with optimization."""
import unittest
from unittest.mock import Mock, patch, MagicMock
from src.portfolio_manager import PortfolioManager
from src.coinbase_client import CoinbaseClient
from src.transaction_logger import TransactionLogger


class MockCoinbaseClient:
    """Mock Coinbase client for testing."""

    def __init__(self):
        """Initialize mock client."""
        self.trades_executed = []

    def get_accounts(self):
        """Return mock account balances."""
        return {
            'BTC': 0.05,    # ~$3,500 at $70k
            'ETH': 1.0,     # ~$3,500 at $3.5k
            'SOL': 10.0,    # ~$2,000 at $200
            'USDC': 1000.0  # $1,000
        }

    def get_product_price(self, product_id):
        """Return mock prices."""
        prices = {
            'BTC-USD': 70000.0,
            'ETH-USD': 3500.0,
            'SOL-USD': 200.0,
        }
        return prices.get(product_id, 1.0)

    def get_portfolio_value_usd(self, balances=None):
        """Calculate mock portfolio value."""
        if balances is None:
            balances = self.get_accounts()

        values = {
            'BTC': 0.05 * 70000.0,
            'ETH': 1.0 * 3500.0,
            'SOL': 10.0 * 200.0,
            'USDC': 1000.0
        }

        total = sum(values.values())
        return total, values

    def get_available_trading_pairs(self, force_refresh=False):
        """Return mock available trading pairs."""
        return {
            'BTC-USD': True,
            'ETH-USD': True,
            'SOL-USD': True,
            'BTC-ETH': True,
            'ETH-SOL': True,
            'SOL-BTC': True,
        }

    def place_market_order(self, product_id, side, size=None, quote_size=None):
        """Mock order placement."""
        order = {
            'success': True,
            'order_id': f'mock_order_{len(self.trades_executed)}',
            'product_id': product_id,
            'side': side,
            'size': size,
            'quote_size': quote_size,
        }
        self.trades_executed.append(order)
        return order

    def get_order(self, order_id):
        """Mock order details."""
        return {
            'order_id': order_id,
            'status': 'FILLED',
            'filled_size': 0.1,
            'average_filled_price': 1000.0,
            'filled_value': 100.0,
            'total_fees': 0.6,
            'fee_currency': 'USD',
            'number_of_fills': 1,
            'fills': []
        }


class TestPortfolioRebalancingIntegration(unittest.TestCase):
    """Integration tests for portfolio rebalancing."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MockCoinbaseClient()
        self.mock_logger = Mock(spec=TransactionLogger)

    def test_simple_rebalance_with_direct_pair(self):
        """Test rebalancing that should use direct trading pair."""
        # Create test config with BTC underweight, ETH overweight
        config = {
            'target_allocation': {
                'BTC': 40.0,  # Currently 35%
                'ETH': 30.0,  # Currently 35%
                'SOL': 20.0,  # Currently 20%
                'USDC': 10.0  # Currently 10%
            },
            'rebalancing': {
                'threshold_percent': 3.0,
                'min_trade_value_usd': 10.0,
                'prefer_direct_routes': True,
                'dry_run': True
            }
        }

        with patch('src.portfolio_manager.load_json', return_value=config):
            manager = PortfolioManager(
                config_path='mock_config.json',
                coinbase_client=self.mock_client,
                transaction_logger=self.mock_logger
            )

            # Get current allocation
            current_alloc, total_value, current_values = manager.get_current_allocation()

            # Calculate trades
            trades = manager.calculate_rebalancing_trades(
                current_alloc, current_values, total_value
            )

            # Should generate trades
            self.assertGreater(len(trades), 0)

            # Check for direct trade opportunity (ETH->BTC)
            direct_trades = [t for t in trades if t.get('is_direct', False)]

            # Log trade details for inspection
            print(f"\nTotal trades: {len(trades)}")
            print(f"Direct trades: {len(direct_trades)}")
            for trade in trades:
                print(f"  {trade.get('from_asset')} -> {trade.get('to_asset')}: "
                      f"${trade['value_usd']:.2f} (direct: {trade.get('is_direct', False)})")

    def test_rebalance_without_direct_pairs(self):
        """Test rebalancing when direct pairs aren't available."""
        config = {
            'target_allocation': {
                'BTC': 40.0,
                'ETH': 30.0,
                'SOL': 20.0,
                'USDC': 10.0
            },
            'rebalancing': {
                'threshold_percent': 3.0,
                'min_trade_value_usd': 10.0,
                'prefer_direct_routes': False,  # Force USD routing
                'dry_run': True
            }
        }

        # Mock client with no direct pairs
        self.mock_client.get_available_trading_pairs = lambda: {
            'BTC-USD': True,
            'ETH-USD': True,
            'SOL-USD': True,
        }

        with patch('src.portfolio_manager.load_json', return_value=config):
            manager = PortfolioManager(
                config_path='mock_config.json',
                coinbase_client=self.mock_client,
                transaction_logger=self.mock_logger
            )

            current_alloc, total_value, current_values = manager.get_current_allocation()
            trades = manager.calculate_rebalancing_trades(
                current_alloc, current_values, total_value
            )

            # All trades should be USD-routed
            direct_trades = [t for t in trades if t.get('is_direct', False)]
            self.assertEqual(len(direct_trades), 0)

            print(f"\nUSD-routed trades: {len(trades)}")
            for trade in trades:
                print(f"  {trade['action']} {trade['asset']}: ${trade['value_usd']:.2f}")

    def test_balanced_portfolio_no_trades(self):
        """Test that balanced portfolio generates no trades."""
        # Portfolio is already at target
        config = {
            'target_allocation': {
                'BTC': 35.0,  # Matches current
                'ETH': 35.0,  # Matches current
                'SOL': 20.0,  # Matches current
                'USDC': 10.0  # Matches current
            },
            'rebalancing': {
                'threshold_percent': 5.0,
                'min_trade_value_usd': 10.0,
                'prefer_direct_routes': True,
                'dry_run': True
            }
        }

        with patch('src.portfolio_manager.load_json', return_value=config):
            manager = PortfolioManager(
                config_path='mock_config.json',
                coinbase_client=self.mock_client,
                transaction_logger=self.mock_logger
            )

            current_alloc, total_value, current_values = manager.get_current_allocation()
            trades = manager.calculate_rebalancing_trades(
                current_alloc, current_values, total_value
            )

            # Should generate no trades
            self.assertEqual(len(trades), 0)
            print("\nBalanced portfolio - no trades needed ✓")

    def test_min_trade_value_filtering(self):
        """Test that trades below minimum value are filtered out."""
        config = {
            'target_allocation': {
                'BTC': 35.5,  # Tiny deviation
                'ETH': 34.5,  # Tiny deviation
                'SOL': 20.0,
                'USDC': 10.0
            },
            'rebalancing': {
                'threshold_percent': 0.1,  # Very low threshold
                'min_trade_value_usd': 500.0,  # High minimum
                'prefer_direct_routes': True,
                'dry_run': True
            }
        }

        with patch('src.portfolio_manager.load_json', return_value=config):
            manager = PortfolioManager(
                config_path='mock_config.json',
                coinbase_client=self.mock_client,
                transaction_logger=self.mock_logger
            )

            current_alloc, total_value, current_values = manager.get_current_allocation()
            trades = manager.calculate_rebalancing_trades(
                current_alloc, current_values, total_value
            )

            # Trades below $500 should be filtered
            for trade in trades:
                self.assertGreaterEqual(trade['value_usd'], 500.0)

            print(f"\nTrades above ${500} minimum: {len(trades)}")


class TestOptimizationBenefits(unittest.TestCase):
    """Test to demonstrate optimization benefits."""

    def test_fee_savings_calculation(self):
        """Demonstrate fee savings from direct routing."""
        mock_client = MockCoinbaseClient()

        # Scenario: Need to move $500 from ETH to BTC
        to_sell = {'ETH': 500.0}
        to_buy = {'BTC': 500.0}

        # Test with direct routing
        config_optimized = {
            'target_allocation': {'BTC': 40.0, 'ETH': 30.0, 'SOL': 20.0, 'USDC': 10.0},
            'rebalancing': {
                'threshold_percent': 3.0,
                'min_trade_value_usd': 10.0,
                'prefer_direct_routes': True,
                'dry_run': True
            }
        }

        # Test without direct routing
        config_naive = config_optimized.copy()
        config_naive['rebalancing']['prefer_direct_routes'] = False

        print("\n" + "="*60)
        print("FEE SAVINGS DEMONSTRATION")
        print("="*60)
        print(f"Rebalancing scenario: Sell $500 ETH, Buy $500 BTC")
        print(f"Assumed fee rate: 0.6%")
        print()

        # Calculate for optimized approach
        from src.trade_optimizer import TradeOptimizer
        optimizer = TradeOptimizer()

        available_pairs = mock_client.get_available_trading_pairs()

        # Optimized trades
        trades_opt = optimizer.calculate_optimal_trades(
            to_sell=to_sell,
            to_buy=to_buy,
            available_pairs=available_pairs,
            prefer_direct=True
        )

        # Naive trades
        trades_naive = optimizer.calculate_optimal_trades(
            to_sell=to_sell,
            to_buy=to_buy,
            available_pairs=available_pairs,
            prefer_direct=False
        )

        fee_rate = 0.006  # 0.6%

        # Calculate fees
        direct_trade_count = sum(1 for t in trades_opt if t.get('is_direct', False))
        opt_fees = len(trades_opt) * 500.0 * fee_rate
        naive_fees = len(trades_naive) * 500.0 * fee_rate

        print(f"OPTIMIZED APPROACH:")
        print(f"  Trades: {len(trades_opt)}")
        print(f"  Direct pairs used: {direct_trade_count}")
        print(f"  Estimated fees: ${opt_fees:.2f}")
        print()
        print(f"NAIVE APPROACH:")
        print(f"  Trades: {len(trades_naive)}")
        print(f"  Direct pairs used: 0")
        print(f"  Estimated fees: ${naive_fees:.2f}")
        print()
        print(f"💰 SAVINGS: ${naive_fees - opt_fees:.2f} ({((naive_fees - opt_fees) / naive_fees * 100):.0f}%)")
        print("="*60)

        self.assertLess(len(trades_opt), len(trades_naive))
        self.assertLess(opt_fees, naive_fees)


if __name__ == '__main__':
    unittest.main(verbosity=2)
