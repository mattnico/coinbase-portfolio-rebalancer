"""Unit tests for Monte Carlo simulator."""

import unittest
from datetime import datetime, timedelta
from src.monte_carlo_simulator import (
    SimulationConfig,
    PortfolioState,
    PortfolioSimulator,
    BuyAndHoldStrategy,
    HybridStrategy,
    Trade,
)


class TestSimulationConfig(unittest.TestCase):
    """Test SimulationConfig validation."""

    def test_valid_config(self):
        """Test creating valid configuration."""
        config = SimulationConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_capital_usd=10000.0,
            target_allocation={'BTC': 50.0, 'ETH': 50.0}
        )
        self.assertEqual(config.fee_rate, 0.006)

    def test_invalid_allocation_sum(self):
        """Test that allocation must sum to 100%."""
        with self.assertRaises(ValueError):
            SimulationConfig(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),
                initial_capital_usd=10000.0,
                target_allocation={'BTC': 40.0, 'ETH': 50.0}  # Only 90%
            )


class TestPortfolioState(unittest.TestCase):
    """Test PortfolioState calculations."""

    def test_total_value_calculation(self):
        """Test portfolio value is calculated correctly."""
        state = PortfolioState(
            timestamp=datetime.now(),
            holdings={'BTC': 0.5, 'ETH': 10.0},
            prices={'BTC': 50000.0, 'ETH': 3000.0}
        )
        # 0.5 * 50000 + 10 * 3000 = 25000 + 30000 = 55000
        self.assertEqual(state.total_value_usd, 55000.0)

    def test_current_allocation(self):
        """Test allocation percentages are calculated correctly."""
        state = PortfolioState(
            timestamp=datetime.now(),
            holdings={'BTC': 1.0, 'ETH': 10.0},
            prices={'BTC': 50000.0, 'ETH': 5000.0}
        )
        # Total: 50000 + 50000 = 100000
        # BTC: 50000/100000 = 50%
        # ETH: 50000/100000 = 50%
        allocation = state.get_current_allocation()
        self.assertAlmostEqual(allocation['BTC'], 50.0, places=2)
        self.assertAlmostEqual(allocation['ETH'], 50.0, places=2)

    def test_deviation_from_target(self):
        """Test deviation calculation."""
        state = PortfolioState(
            timestamp=datetime.now(),
            holdings={'BTC': 0.6, 'ETH': 8.0},
            prices={'BTC': 50000.0, 'ETH': 5000.0}
        )
        # BTC: 30000 / 70000 = 42.86%
        # ETH: 40000 / 70000 = 57.14%
        target = {'BTC': 50.0, 'ETH': 50.0}
        deviation = state.get_deviation_from_target(target)

        # BTC: 42.86 - 50 = -7.14% (underweight)
        # ETH: 57.14 - 50 = +7.14% (overweight)
        self.assertAlmostEqual(deviation['BTC'], -7.14, places=1)
        self.assertAlmostEqual(deviation['ETH'], 7.14, places=1)


class TestBuyAndHoldStrategy(unittest.TestCase):
    """Test buy and hold strategy."""

    def test_never_rebalances(self):
        """Test that buy and hold never rebalances."""
        strategy = BuyAndHoldStrategy()
        state = PortfolioState(
            timestamp=datetime.now(),
            holdings={'BTC': 0.5, 'ETH': 10.0},
            prices={'BTC': 50000.0, 'ETH': 3000.0}
        )

        should_rebalance, reason = strategy.should_rebalance(
            current_state=state,
            target_allocation={'BTC': 50.0, 'ETH': 50.0},
            last_rebalance_time=None
        )

        self.assertFalse(should_rebalance)
        self.assertIn('Buy and hold', reason)

    def test_no_trades(self):
        """Test that buy and hold generates no trades."""
        strategy = BuyAndHoldStrategy()
        state = PortfolioState(
            timestamp=datetime.now(),
            holdings={'BTC': 0.5, 'ETH': 10.0},
            prices={'BTC': 50000.0, 'ETH': 3000.0}
        )

        trades = strategy.calculate_trades(
            current_state=state,
            target_allocation={'BTC': 50.0, 'ETH': 50.0},
            fee_rate=0.006
        )

        self.assertEqual(len(trades), 0)


class TestHybridStrategy(unittest.TestCase):
    """Test hybrid strategy logic."""

    def test_initial_rebalance(self):
        """Test that strategy rebalances on first run if threshold exceeded."""
        strategy = HybridStrategy(
            rebalance_interval_days=7,
            threshold_percent=2.5
        )
        self.assertEqual(strategy.name, "Hybrid (every 7d, ±2.5%)")

        # Create state with 60/40 split (10% deviation from 50/50 target)
        state = PortfolioState(
            timestamp=datetime(2024, 1, 1),
            holdings={'BTC': 0.6, 'ETH': 8.0},
            prices={'BTC': 50000.0, 'ETH': 5000.0}
        )

        should_rebalance, reason = strategy.should_rebalance(
            current_state=state,
            target_allocation={'BTC': 50.0, 'ETH': 50.0},
            last_rebalance_time=None
        )

        self.assertTrue(should_rebalance)
        self.assertIn('Initial rebalance', reason)

    def test_time_not_met(self):
        """Test that strategy doesn't rebalance before interval."""
        strategy = HybridStrategy(
            rebalance_interval_days=7,
            threshold_percent=2.5
        )

        state = PortfolioState(
            timestamp=datetime(2024, 1, 5),  # Only 4 days later
            holdings={'BTC': 0.7, 'ETH': 6.0},
            prices={'BTC': 50000.0, 'ETH': 5000.0}
        )

        should_rebalance, reason = strategy.should_rebalance(
            current_state=state,
            target_allocation={'BTC': 50.0, 'ETH': 50.0},
            last_rebalance_time=datetime(2024, 1, 1)
        )

        self.assertFalse(should_rebalance)
        self.assertIn('4 days', reason)

    def test_time_met_threshold_not_met(self):
        """Test interval met but threshold not exceeded."""
        strategy = HybridStrategy(
            rebalance_interval_days=7,
            threshold_percent=5.0  # High threshold
        )

        # 51/49 split - only 1% deviation
        state = PortfolioState(
            timestamp=datetime(2024, 1, 10),  # 9 days later
            holdings={'BTC': 1.02, 'ETH': 9.8},
            prices={'BTC': 50000.0, 'ETH': 5000.0}
        )
        # BTC: 1.02 * 50000 = 51000 (51%)
        # ETH: 9.8 * 5000 = 49000 (49%)
        # Max deviation: 1%

        should_rebalance, reason = strategy.should_rebalance(
            current_state=state,
            target_allocation={'BTC': 50.0, 'ETH': 50.0},
            last_rebalance_time=datetime(2024, 1, 1)
        )

        self.assertFalse(should_rebalance)
        self.assertIn('but max deviation', reason)

    def test_both_conditions_met(self):
        """Test rebalancing when both time and threshold are met."""
        strategy = HybridStrategy(
            rebalance_interval_days=7,
            threshold_percent=2.5
        )

        # 60/40 split - 10% deviation
        state = PortfolioState(
            timestamp=datetime(2024, 1, 10),  # 9 days later
            holdings={'BTC': 0.6, 'ETH': 8.0},
            prices={'BTC': 50000.0, 'ETH': 5000.0}
        )

        should_rebalance, reason = strategy.should_rebalance(
            current_state=state,
            target_allocation={'BTC': 50.0, 'ETH': 50.0},
            last_rebalance_time=datetime(2024, 1, 1)
        )

        self.assertTrue(should_rebalance)
        self.assertIn('max deviation', reason)

    def test_calculate_trades(self):
        """Test trade calculation logic."""
        strategy = HybridStrategy(
            rebalance_interval_days=7,
            threshold_percent=2.5
        )

        # Portfolio: 60% BTC ($60k), 40% ETH ($40k)
        # Target: 50% BTC ($50k), 50% ETH ($50k)
        # Need to: Sell $10k BTC, Buy $10k ETH
        state = PortfolioState(
            timestamp=datetime.now(),
            holdings={'BTC': 1.2, 'ETH': 10.0},
            prices={'BTC': 50000.0, 'ETH': 4000.0}
        )

        trades = strategy.calculate_trades(
            current_state=state,
            target_allocation={'BTC': 50.0, 'ETH': 50.0},
            fee_rate=0.006,
            min_trade_value_usd=10.0
        )

        # Should have 2 trades: sell BTC, buy ETH
        self.assertEqual(len(trades), 2)

        sell_trade = next(t for t in trades if t.action == 'sell')
        buy_trade = next(t for t in trades if t.action == 'buy')

        self.assertEqual(sell_trade.asset, 'BTC')
        self.assertEqual(buy_trade.asset, 'ETH')

        # Check values are approximately correct
        self.assertAlmostEqual(sell_trade.value_usd, 10000.0, delta=100)
        self.assertAlmostEqual(buy_trade.value_usd, 10000.0, delta=100)

    def test_hourly_interval(self):
        """Test strategy with hourly rebalancing interval."""
        strategy = HybridStrategy(
            rebalance_interval_hours=6,
            threshold_percent=2.5
        )

        self.assertEqual(strategy.name, "Hybrid (every 6h, ±2.5%)")

        # Test that 5 hours is not enough
        state = PortfolioState(
            timestamp=datetime(2024, 1, 1, 5, 0),  # 5 hours later
            holdings={'BTC': 0.6, 'ETH': 8.0},
            prices={'BTC': 50000.0, 'ETH': 5000.0}
        )

        should_rebalance, reason = strategy.should_rebalance(
            current_state=state,
            target_allocation={'BTC': 50.0, 'ETH': 50.0},
            last_rebalance_time=datetime(2024, 1, 1, 0, 0)
        )

        self.assertFalse(should_rebalance)
        self.assertIn('5 hours', reason)

        # Test that 6+ hours with threshold exceeded triggers rebalance
        state2 = PortfolioState(
            timestamp=datetime(2024, 1, 1, 7, 0),  # 7 hours later
            holdings={'BTC': 0.6, 'ETH': 8.0},
            prices={'BTC': 50000.0, 'ETH': 5000.0}
        )

        should_rebalance, reason = strategy.should_rebalance(
            current_state=state2,
            target_allocation={'BTC': 50.0, 'ETH': 50.0},
            last_rebalance_time=datetime(2024, 1, 1, 0, 0)
        )

        self.assertTrue(should_rebalance)
        self.assertIn('7 hours', reason)

    def test_minute_interval(self):
        """Test strategy with minute-level rebalancing interval."""
        strategy = HybridStrategy(
            rebalance_interval_minutes=5,
            threshold_percent=2.5
        )

        self.assertEqual(strategy.name, "Hybrid (every 5min, ±2.5%)")

        # Test initial rebalance
        state = PortfolioState(
            timestamp=datetime(2024, 1, 1, 0, 0),
            holdings={'BTC': 0.6, 'ETH': 8.0},
            prices={'BTC': 50000.0, 'ETH': 5000.0}
        )

        should_rebalance, reason = strategy.should_rebalance(
            current_state=state,
            target_allocation={'BTC': 50.0, 'ETH': 50.0},
            last_rebalance_time=None
        )

        self.assertTrue(should_rebalance)
        self.assertIn('Initial', reason)

        # Test that 4 minutes is not enough
        state2 = PortfolioState(
            timestamp=datetime(2024, 1, 1, 0, 4),  # 4 minutes later
            holdings={'BTC': 0.6, 'ETH': 8.0},
            prices={'BTC': 50000.0, 'ETH': 5000.0}
        )

        should_rebalance, reason = strategy.should_rebalance(
            current_state=state2,
            target_allocation={'BTC': 50.0, 'ETH': 50.0},
            last_rebalance_time=datetime(2024, 1, 1, 0, 0)
        )

        self.assertFalse(should_rebalance)
        self.assertIn('4 minutes', reason)

        # Test that 5+ minutes with threshold triggers rebalance
        state3 = PortfolioState(
            timestamp=datetime(2024, 1, 1, 0, 6),  # 6 minutes later
            holdings={'BTC': 0.6, 'ETH': 8.0},
            prices={'BTC': 50000.0, 'ETH': 5000.0}
        )

        should_rebalance, reason = strategy.should_rebalance(
            current_state=state3,
            target_allocation={'BTC': 50.0, 'ETH': 50.0},
            last_rebalance_time=datetime(2024, 1, 1, 0, 0)
        )

        self.assertTrue(should_rebalance)
        self.assertIn('6 minutes', reason)


class TestPortfolioSimulator(unittest.TestCase):
    """Test portfolio simulator."""

    def setUp(self):
        """Set up test data."""
        # Create 30 days of price data
        start_date = datetime(2024, 1, 1)
        self.price_data = {
            'BTC': [],
            'ETH': []
        }

        for i in range(30):
            date = start_date + timedelta(days=i)
            # BTC goes from $40k to $50k (25% gain)
            btc_price = 40000 + (i * 333.33)
            # ETH goes from $2k to $3k (50% gain)
            eth_price = 2000 + (i * 33.33)

            self.price_data['BTC'].append((date, btc_price))
            self.price_data['ETH'].append((date, eth_price))

    def test_buy_and_hold_simulation(self):
        """Test buy and hold simulation."""
        config = SimulationConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 30),
            initial_capital_usd=10000.0,
            target_allocation={'BTC': 50.0, 'ETH': 50.0},
            fee_rate=0.006,
            price_check_interval_hours=24
        )

        strategy = BuyAndHoldStrategy()
        simulator = PortfolioSimulator(config, strategy, self.price_data)
        result = simulator.run()

        # Check basic properties
        self.assertEqual(result.strategy_name, 'Buy and Hold')
        self.assertEqual(result.initial_value, 10000.0)
        self.assertEqual(result.num_rebalances, 0)
        self.assertEqual(result.total_fees_paid, 0.0)
        self.assertEqual(len(result.trades), 0)

        # Portfolio should have gained (BTC +25%, ETH +50% = avg ~37.5%)
        self.assertGreater(result.total_return_percent, 30.0)
        self.assertLess(result.total_return_percent, 45.0)

    def test_hybrid_strategy_simulation(self):
        """Test hybrid strategy simulation."""
        config = SimulationConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 30),
            initial_capital_usd=10000.0,
            target_allocation={'BTC': 50.0, 'ETH': 50.0},
            fee_rate=0.006,
            price_check_interval_hours=24
        )

        # Rebalance weekly if deviation > 2.5%
        strategy = HybridStrategy(
            rebalance_interval_days=7,
            threshold_percent=2.5
        )

        simulator = PortfolioSimulator(config, strategy, self.price_data)
        result = simulator.run()

        # Check basic properties
        self.assertEqual(result.strategy_name, 'Hybrid (every 7d, ±2.5%)')
        self.assertEqual(result.initial_value, 10000.0)

        # Should have rebalanced at least once (ETH growing faster)
        self.assertGreater(result.num_rebalances, 0)
        self.assertGreater(result.total_fees_paid, 0)
        self.assertGreater(len(result.trades), 0)

        # Should have positive return
        self.assertGreater(result.total_return_percent, 0)

    def test_portfolio_history_tracking(self):
        """Test that portfolio history is tracked correctly."""
        config = SimulationConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5),  # 5 days
            initial_capital_usd=10000.0,
            target_allocation={'BTC': 50.0, 'ETH': 50.0},
            fee_rate=0.006,
            price_check_interval_hours=24
        )

        strategy = BuyAndHoldStrategy()
        simulator = PortfolioSimulator(config, strategy, self.price_data)
        result = simulator.run()

        # Should have 5 portfolio states (Jan 1, 2, 3, 4, 5)
        self.assertEqual(len(result.portfolio_history), 5)

        # First state should be at start date
        self.assertEqual(
            result.portfolio_history[0].timestamp.date(),
            datetime(2024, 1, 1).date()
        )

        # Last state should be at end date
        self.assertEqual(
            result.portfolio_history[-1].timestamp.date(),
            datetime(2024, 1, 5).date()
        )

    def test_performance_metrics(self):
        """Test that performance metrics are calculated."""
        config = SimulationConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 30),
            initial_capital_usd=10000.0,
            target_allocation={'BTC': 50.0, 'ETH': 50.0},
            fee_rate=0.006,
            price_check_interval_hours=24
        )

        strategy = BuyAndHoldStrategy()
        simulator = PortfolioSimulator(config, strategy, self.price_data)
        result = simulator.run()

        # All metrics should be calculated
        self.assertIsNotNone(result.total_return_percent)
        self.assertIsNotNone(result.annualized_return_percent)
        self.assertIsNotNone(result.sharpe_ratio)
        self.assertIsNotNone(result.max_drawdown_percent)

        # Sharpe can be high with consistent positive returns and low volatility
        # Just verify it's a reasonable number
        self.assertGreater(result.sharpe_ratio, -100)
        self.assertLess(result.sharpe_ratio, 1000)

        # Drawdown should be non-negative
        self.assertGreaterEqual(result.max_drawdown_percent, 0)


class TestTrade(unittest.TestCase):
    """Test Trade data class."""

    def test_trade_creation(self):
        """Test creating trade objects."""
        trade = Trade(
            timestamp=datetime(2024, 1, 1),
            asset='BTC',
            action='buy',
            quantity=0.1,
            price=50000.0,
            value_usd=5000.0,
            fee_usd=30.0,
            reason='Rebalancing'
        )

        self.assertEqual(trade.asset, 'BTC')
        self.assertEqual(trade.action, 'buy')
        self.assertEqual(trade.quantity, 0.1)
        self.assertEqual(trade.fee_usd, 30.0)


class TestStablecoinHandling(unittest.TestCase):
    """Test handling of stablecoins in simulations."""

    def test_portfolio_with_stablecoin(self):
        """Test that portfolios with USDC work correctly."""
        # Create simple price data including a stablecoin
        price_data = {
            'BTC': [(datetime(2024, 1, i), 40000.0 + i*1000) for i in range(1, 6)],
            'USDC': [(datetime(2024, 1, i), 1.0) for i in range(1, 6)]
        }

        config = SimulationConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5),
            initial_capital_usd=10000.0,
            target_allocation={'BTC': 50.0, 'USDC': 50.0},
            fee_rate=0.006,
            price_check_interval_hours=24
        )

        strategy = BuyAndHoldStrategy()
        simulator = PortfolioSimulator(config, strategy, price_data)
        result = simulator.run()

        # Should complete successfully
        self.assertEqual(result.initial_value, 10000.0)
        self.assertEqual(result.num_rebalances, 0)

        # USDC should remain stable at $1
        for state in result.portfolio_history:
            self.assertEqual(state.prices['USDC'], 1.0)


if __name__ == '__main__':
    unittest.main()
