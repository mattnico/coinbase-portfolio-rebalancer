"""Test unknown asset handling functionality."""
import unittest
from unittest.mock import Mock, patch
from src.portfolio_manager import PortfolioManager


class TestUnknownAssetHandling(unittest.TestCase):
    """Test that unknown assets are properly detected and handled."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'target_allocation': {
                'BTC': 40.0,
                'ETH': 30.0,
                'SOL': 20.0,
                'USDC': 10.0
            },
            'rebalancing': {
                'threshold_percent': 5.0,
                'min_trade_value_usd': 10.0,
                'prefer_direct_routes': True,
                'handle_unknown_assets': 'sell'
            }
        }

    @patch('src.portfolio_manager.CoinbaseClient')
    @patch('src.portfolio_manager.TransactionLogger')
    @patch('src.portfolio_manager.load_json')
    def test_unknown_asset_detection(self, mock_load_json, mock_logger, mock_client):
        """Test that unknown assets are properly identified."""
        mock_load_json.return_value = self.config

        # Mock client methods
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        # Portfolio has LINK which is not in target allocation
        mock_client_instance.get_accounts.return_value = {
            'BTC': 0.05,
            'ETH': 1.5,
            'SOL': 10.0,
            'USDC': 1000.0,
            'LINK': 50.0  # Unknown asset!
        }

        mock_client_instance.get_portfolio_value_usd.return_value = (10000.0, {
            'BTC': 3500.0,  # 35%
            'ETH': 3000.0,  # 30%
            'SOL': 2000.0,  # 20%
            'USDC': 1000.0, # 10%
            'LINK': 500.0   # 5% - should be detected as unknown
        })

        # Mock available trading pairs
        mock_client_instance.get_available_trading_pairs.return_value = {
            'BTC-USD': True,
            'ETH-USD': True,
            'SOL-USD': True,
            'LINK-USD': True,
            'BTC-ETH': True,
            'ETH-BTC': True
        }

        # Mock prices
        mock_client_instance.get_product_price.side_effect = lambda x: {
            'BTC-USD': 70000.0,
            'ETH-USD': 2000.0,
            'SOL-USD': 200.0,
            'LINK-USD': 10.0
        }.get(x, 1.0)

        manager = PortfolioManager(config_path='dummy.json')
        manager.config = self.config

        # Get current allocation
        current_allocation = {
            'BTC': 35.0,
            'ETH': 30.0,
            'SOL': 20.0,
            'USDC': 10.0,
            'LINK': 5.0
        }

        current_values = {
            'BTC': 3500.0,
            'ETH': 3000.0,
            'SOL': 2000.0,
            'USDC': 1000.0,
            'LINK': 500.0
        }

        # Calculate trades
        trades = manager.calculate_rebalancing_trades(
            current_allocation=current_allocation,
            current_values=current_values,
            total_value=10000.0
        )

        # Verify LINK is being sold
        link_trades = [t for t in trades if t['asset'] == 'LINK']
        self.assertTrue(len(link_trades) > 0, "LINK should be queued for sale")
        self.assertEqual(link_trades[0]['action'], 'SELL', "LINK trade should be a SELL")

        # Verify proceeds are being distributed
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        self.assertTrue(len(buy_trades) > 0, "Should have buy trades to distribute LINK proceeds")

    @patch('src.portfolio_manager.CoinbaseClient')
    @patch('src.portfolio_manager.TransactionLogger')
    @patch('src.portfolio_manager.load_json')
    def test_unknown_asset_distribution(self, mock_load_json, mock_logger, mock_client):
        """Test that unknown asset proceeds are distributed according to target allocation."""
        mock_load_json.return_value = self.config

        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        # Portfolio is perfectly balanced except for unknown LINK
        mock_client_instance.get_accounts.return_value = {
            'BTC': 0.05714,  # Exactly 40% of 10000 at $70k = $4000
            'ETH': 1.5,      # Exactly 30% of 10000 at $2k = $3000
            'SOL': 10.0,     # Exactly 20% of 10000 at $200 = $2000
            'USDC': 1000.0,  # Exactly 10% = $1000
            'LINK': 100.0    # Unknown asset worth $1000
        }

        mock_client_instance.get_portfolio_value_usd.return_value = (11000.0, {
            'BTC': 4000.0,
            'ETH': 3000.0,
            'SOL': 2000.0,
            'USDC': 1000.0,
            'LINK': 1000.0
        })

        mock_client_instance.get_available_trading_pairs.return_value = {
            'BTC-USD': True,
            'ETH-USD': True,
            'SOL-USD': True,
            'LINK-USD': True
        }

        mock_client_instance.get_product_price.side_effect = lambda x: {
            'BTC-USD': 70000.0,
            'ETH-USD': 2000.0,
            'SOL-USD': 200.0,
            'LINK-USD': 10.0
        }.get(x, 1.0)

        manager = PortfolioManager(config_path='dummy.json')
        manager.config = self.config

        current_allocation = {
            'BTC': 36.36,  # 4000/11000
            'ETH': 27.27,  # 3000/11000
            'SOL': 18.18,  # 2000/11000
            'USDC': 9.09,  # 1000/11000
            'LINK': 9.09   # 1000/11000
        }

        current_values = {
            'BTC': 4000.0,
            'ETH': 3000.0,
            'SOL': 2000.0,
            'USDC': 1000.0,
            'LINK': 1000.0
        }

        trades = manager.calculate_rebalancing_trades(
            current_allocation=current_allocation,
            current_values=current_values,
            total_value=11000.0
        )

        # LINK should be sold
        link_trades = [t for t in trades if t['asset'] == 'LINK']
        self.assertEqual(len(link_trades), 1, "Should have one LINK sell trade")
        self.assertEqual(link_trades[0]['action'], 'SELL')

        # Calculate expected distribution of $1000 from LINK
        # BTC should get 40% = $400
        # ETH should get 30% = $300
        # SOL should get 20% = $200
        # USDC should get 10% = $100

        buy_trades = [t for t in trades if t['action'] == 'BUY']

        # Group by asset
        buy_by_asset = {}
        for trade in buy_trades:
            asset = trade['asset']
            if asset not in buy_by_asset:
                buy_by_asset[asset] = 0
            buy_by_asset[asset] += trade['value_usd']

        # Verify distribution (allowing for small rounding errors)
        if 'BTC' in buy_by_asset:
            self.assertAlmostEqual(buy_by_asset['BTC'], 400.0, delta=50.0,
                                 msg="BTC should receive ~40% of LINK proceeds")

        if 'ETH' in buy_by_asset:
            self.assertAlmostEqual(buy_by_asset['ETH'], 300.0, delta=50.0,
                                 msg="ETH should receive ~30% of LINK proceeds")

    @patch('src.portfolio_manager.CoinbaseClient')
    @patch('src.portfolio_manager.TransactionLogger')
    @patch('src.portfolio_manager.load_json')
    def test_ignore_unknown_assets(self, mock_load_json, mock_logger, mock_client):
        """Test that unknown assets are ignored when configured to do so."""
        config_ignore = self.config.copy()
        config_ignore['rebalancing']['handle_unknown_assets'] = 'ignore'
        mock_load_json.return_value = config_ignore

        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_client_instance.get_accounts.return_value = {
            'BTC': 0.05,
            'ETH': 1.5,
            'SOL': 10.0,
            'USDC': 1000.0,
            'LINK': 50.0
        }

        mock_client_instance.get_portfolio_value_usd.return_value = (10000.0, {
            'BTC': 4000.0,  # 40% - balanced
            'ETH': 3000.0,  # 30% - balanced
            'SOL': 2000.0,  # 20% - balanced
            'USDC': 1000.0, # 10% - balanced
            'LINK': 500.0   # Should be ignored
        })

        mock_client_instance.get_available_trading_pairs.return_value = {}

        manager = PortfolioManager(config_path='dummy.json')
        manager.config = config_ignore

        current_allocation = {
            'BTC': 40.0,
            'ETH': 30.0,
            'SOL': 20.0,
            'USDC': 10.0,
            'LINK': 5.0
        }

        current_values = {
            'BTC': 4000.0,
            'ETH': 3000.0,
            'SOL': 2000.0,
            'USDC': 1000.0,
            'LINK': 500.0
        }

        trades = manager.calculate_rebalancing_trades(
            current_allocation=current_allocation,
            current_values=current_values,
            total_value=10000.0
        )

        # No LINK trades should be generated
        link_trades = [t for t in trades if t['asset'] == 'LINK']
        self.assertEqual(len(link_trades), 0, "LINK should be ignored when handle_unknown_assets='ignore'")

    @patch('src.portfolio_manager.CoinbaseClient')
    @patch('src.portfolio_manager.TransactionLogger')
    @patch('src.portfolio_manager.load_json')
    def test_multiple_unknown_assets(self, mock_load_json, mock_logger, mock_client):
        """Test handling of multiple unknown assets simultaneously."""
        mock_load_json.return_value = self.config

        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        # Portfolio has LINK and DOGE which are not in target
        mock_client_instance.get_accounts.return_value = {
            'BTC': 0.05,
            'ETH': 1.5,
            'SOL': 10.0,
            'USDC': 1000.0,
            'LINK': 50.0,
            'DOGE': 1000.0
        }

        mock_client_instance.get_portfolio_value_usd.return_value = (10000.0, {
            'BTC': 3200.0,  # 32%
            'ETH': 2800.0,  # 28%
            'SOL': 1800.0,  # 18%
            'USDC': 1000.0, # 10%
            'LINK': 600.0,  # 6% - unknown
            'DOGE': 600.0   # 6% - unknown
        })

        mock_client_instance.get_available_trading_pairs.return_value = {
            'BTC-USD': True,
            'ETH-USD': True,
            'SOL-USD': True,
            'LINK-USD': True,
            'DOGE-USD': True
        }

        mock_client_instance.get_product_price.side_effect = lambda x: {
            'BTC-USD': 70000.0,
            'ETH-USD': 2000.0,
            'SOL-USD': 200.0,
            'LINK-USD': 10.0,
            'DOGE-USD': 0.60
        }.get(x, 1.0)

        manager = PortfolioManager(config_path='dummy.json')
        manager.config = self.config

        current_allocation = {
            'BTC': 32.0,
            'ETH': 28.0,
            'SOL': 18.0,
            'USDC': 10.0,
            'LINK': 6.0,
            'DOGE': 6.0
        }

        current_values = {
            'BTC': 3200.0,
            'ETH': 2800.0,
            'SOL': 1800.0,
            'USDC': 1000.0,
            'LINK': 600.0,
            'DOGE': 600.0
        }

        trades = manager.calculate_rebalancing_trades(
            current_allocation=current_allocation,
            current_values=current_values,
            total_value=10000.0
        )

        # Both LINK and DOGE should be sold
        link_trades = [t for t in trades if t['asset'] == 'LINK']
        doge_trades = [t for t in trades if t['asset'] == 'DOGE']

        self.assertTrue(len(link_trades) > 0, "LINK should be sold")
        self.assertTrue(len(doge_trades) > 0, "DOGE should be sold")

        # Total unknown value is $1200, should be distributed
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        self.assertTrue(len(buy_trades) > 0, "Should have buy trades for distribution")


if __name__ == '__main__':
    unittest.main()
