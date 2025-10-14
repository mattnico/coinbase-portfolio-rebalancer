"""Tests for trade optimization logic."""
import unittest
from src.trade_optimizer import TradeOptimizer


class TestTradeOptimizer(unittest.TestCase):
    """Test cases for TradeOptimizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = TradeOptimizer()

    def test_direct_trade_single_pair(self):
        """Test optimization with single direct trading pair."""
        to_sell = {'ETH': 500.0}
        to_buy = {'BTC': 500.0}
        available_pairs = {
            'BTC-ETH': True,
            'ETH-USD': True,
            'BTC-USD': True,
        }

        trades = self.optimizer.calculate_optimal_trades(
            to_sell=to_sell,
            to_buy=to_buy,
            available_pairs=available_pairs,
            prefer_direct=True
        )

        # Should have 1 direct trade
        self.assertEqual(len(trades), 1)
        self.assertTrue(trades[0]['is_direct'])
        self.assertEqual(trades[0]['value_usd'], 500.0)
        self.assertIn('BTC', trades[0]['product_id'])
        self.assertIn('ETH', trades[0]['product_id'])

    def test_no_direct_pair_available(self):
        """Test when no direct pair exists - should route through USD."""
        to_sell = {'SOL': 500.0}
        to_buy = {'BTC': 500.0}
        available_pairs = {
            'SOL-USD': True,
            'BTC-USD': True,
        }

        trades = self.optimizer.calculate_optimal_trades(
            to_sell=to_sell,
            to_buy=to_buy,
            available_pairs=available_pairs,
            prefer_direct=True
        )

        # Should have 2 trades: sell SOL to USD, buy BTC with USD
        self.assertEqual(len(trades), 2)
        self.assertFalse(trades[0]['is_direct'])
        self.assertFalse(trades[1]['is_direct'])

        # Check sell trade
        sell_trade = next(t for t in trades if t['type'] == 'usd_sell')
        self.assertEqual(sell_trade['from_asset'], 'SOL')
        self.assertEqual(sell_trade['to_asset'], 'USD')

        # Check buy trade
        buy_trade = next(t for t in trades if t['type'] == 'usd_buy')
        self.assertEqual(buy_trade['from_asset'], 'USD')
        self.assertEqual(buy_trade['to_asset'], 'BTC')

    def test_partial_direct_match(self):
        """Test when direct pair can only match part of the needed trades."""
        to_sell = {'ETH': 1000.0}
        to_buy = {'BTC': 500.0, 'SOL': 500.0}
        available_pairs = {
            'BTC-ETH': True,
            'ETH-USD': True,
            'BTC-USD': True,
            'SOL-USD': True,
        }

        trades = self.optimizer.calculate_optimal_trades(
            to_sell=to_sell,
            to_buy=to_buy,
            available_pairs=available_pairs,
            prefer_direct=True
        )

        # Should have: 1 direct ETH->BTC, 1 sell ETH->USD, 1 buy SOL with USD
        direct_trades = [t for t in trades if t.get('is_direct', False)]
        usd_trades = [t for t in trades if not t.get('is_direct', False)]

        self.assertEqual(len(direct_trades), 1)
        self.assertGreater(len(usd_trades), 0)

        # Direct trade should be ETH->BTC for $500
        direct_trade = direct_trades[0]
        self.assertEqual(direct_trade['value_usd'], 500.0)
        self.assertIn('ETH', direct_trade['product_id'])
        self.assertIn('BTC', direct_trade['product_id'])

    def test_multiple_sellers_multiple_buyers(self):
        """Test complex scenario with multiple assets to sell and buy."""
        to_sell = {'BTC': 300.0, 'ETH': 200.0}
        to_buy = {'SOL': 250.0, 'MATIC': 250.0}
        available_pairs = {
            'BTC-USD': True,
            'ETH-USD': True,
            'SOL-USD': True,
            'MATIC-USD': True,
            'ETH-BTC': True,
            'SOL-ETH': True,
        }

        trades = self.optimizer.calculate_optimal_trades(
            to_sell=to_sell,
            to_buy=to_buy,
            available_pairs=available_pairs,
            prefer_direct=True
        )

        # Should find some direct opportunities
        direct_trades = [t for t in trades if t.get('is_direct', False)]
        self.assertGreater(len(direct_trades), 0)

        # Total value should balance
        total_sell_value = sum(t['value_usd'] for t in trades if t['from_asset'] in to_sell)
        total_buy_value = sum(t['value_usd'] for t in trades if t['to_asset'] in to_buy)

    def test_prefer_direct_false(self):
        """Test that prefer_direct=False routes everything through USD."""
        to_sell = {'ETH': 500.0}
        to_buy = {'BTC': 500.0}
        available_pairs = {
            'BTC-ETH': True,
            'ETH-USD': True,
            'BTC-USD': True,
        }

        trades = self.optimizer.calculate_optimal_trades(
            to_sell=to_sell,
            to_buy=to_buy,
            available_pairs=available_pairs,
            prefer_direct=False
        )

        # Should route through USD
        self.assertEqual(len(trades), 2)
        self.assertFalse(any(t.get('is_direct', False) for t in trades))

    def test_convert_to_execution_format(self):
        """Test conversion to execution format with sizes."""
        optimized_trades = [{
            'type': 'usd_sell',
            'from_asset': 'BTC',
            'to_asset': 'USD',
            'product_id': 'BTC-USD',
            'side': 'SELL',
            'value_usd': 1000.0,
            'is_direct': False,
            'reason': 'Test sell'
        }]

        current_balances = {'BTC': 0.02}
        current_prices = {'BTC': 50000.0}

        execution_trades = self.optimizer.convert_to_execution_format(
            optimized_trades=optimized_trades,
            current_balances=current_balances,
            current_prices=current_prices
        )

        self.assertEqual(len(execution_trades), 1)
        trade = execution_trades[0]

        # Should calculate size
        self.assertIn('size', trade)
        self.assertEqual(trade['action'], 'SELL')
        self.assertEqual(trade['value_usd'], 1000.0)

    def test_empty_inputs(self):
        """Test handling of empty trade lists."""
        trades = self.optimizer.calculate_optimal_trades(
            to_sell={},
            to_buy={},
            available_pairs={},
            prefer_direct=True
        )

        self.assertEqual(len(trades), 0)

    def test_single_asset_to_sell_multiple_to_buy(self):
        """Test selling one asset to buy multiple."""
        to_sell = {'BTC': 1000.0}
        to_buy = {'ETH': 400.0, 'SOL': 300.0, 'MATIC': 300.0}
        available_pairs = {
            'BTC-USD': True,
            'ETH-USD': True,
            'SOL-USD': True,
            'MATIC-USD': True,
            'BTC-ETH': True,
            'BTC-SOL': True,
        }

        trades = self.optimizer.calculate_optimal_trades(
            to_sell=to_sell,
            to_buy=to_buy,
            available_pairs=available_pairs,
            prefer_direct=True
        )

        # Should have direct trades for BTC->ETH and BTC->SOL
        direct_trades = [t for t in trades if t.get('is_direct', False)]
        self.assertGreater(len(direct_trades), 0)

        # Verify direct trade pairs involve BTC
        for trade in direct_trades:
            self.assertIn('BTC', trade['product_id'])


class TestTradeOptimizerEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = TradeOptimizer()

    def test_reverse_pair_detection(self):
        """Test that optimizer detects pairs in both directions."""
        to_sell = {'BTC': 500.0}
        to_buy = {'ETH': 500.0}

        # Pair exists as ETH-BTC instead of BTC-ETH
        available_pairs = {
            'ETH-BTC': True,
            'BTC-USD': True,
            'ETH-USD': True,
        }

        trades = self.optimizer.calculate_optimal_trades(
            to_sell=to_sell,
            to_buy=to_buy,
            available_pairs=available_pairs,
            prefer_direct=True
        )

        # Should still find the direct pair
        self.assertEqual(len(trades), 1)
        self.assertTrue(trades[0]['is_direct'])
        self.assertEqual(trades[0]['product_id'], 'ETH-BTC')

    def test_stablecoin_handling(self):
        """Test handling of USD and USDC in trades."""
        to_sell = {'BTC': 500.0}
        to_buy = {'USDC': 500.0}
        available_pairs = {
            'BTC-USD': True,
            'BTC-USDC': True,
        }

        trades = self.optimizer.calculate_optimal_trades(
            to_sell=to_sell,
            to_buy=to_buy,
            available_pairs=available_pairs,
            prefer_direct=True
        )

        # Should work normally
        self.assertGreater(len(trades), 0)


if __name__ == '__main__':
    unittest.main()
