"""Unit tests for strategy optimizer."""

import unittest
from datetime import datetime, timedelta

from src.monte_carlo_simulator import SimulationConfig
from src.strategy_optimizer import StrategyOptimizer, OptimizationResult


class TestStrategyOptimizer(unittest.TestCase):
    """Test strategy optimizer functionality."""

    def setUp(self):
        """Set up test data."""
        # Create simple price data
        start_date = datetime(2024, 1, 1)
        self.price_data = {
            'BTC': [(start_date + timedelta(days=i), 40000.0 + i*1000) for i in range(30)],
            'ETH': [(start_date + timedelta(days=i), 2000.0 + i*50) for i in range(30)]
        }

        # Create simulation config
        self.sim_config = SimulationConfig(
            start_date=start_date,
            end_date=start_date + timedelta(days=29),
            initial_capital_usd=10000.0,
            target_allocation={'BTC': 50.0, 'ETH': 50.0},
            fee_rate=0.006,
            price_check_interval_hours=24
        )

    def test_parameter_grid_generation(self):
        """Test that parameter grid is generated correctly."""
        optimizer = StrategyOptimizer(self.price_data, self.sim_config)

        grid = optimizer.generate_parameter_grid(
            threshold_min=0.5,
            threshold_max=1.5,
            threshold_step=0.5,
            interval_min=60,
            interval_max=120,
            interval_step=30
        )

        # Should have 3 thresholds (0.5, 1.0, 1.5) x 3 intervals (60, 90, 120) = 9 combinations
        self.assertEqual(len(grid), 9)

        # Check first and last combinations
        self.assertEqual(grid[0], (0.5, 60.0))
        self.assertEqual(grid[-1], (1.5, 120.0))

    def test_baseline_simulation(self):
        """Test that baseline simulation runs successfully."""
        optimizer = StrategyOptimizer(self.price_data, self.sim_config)

        result = optimizer.run_baseline()

        self.assertIsNotNone(result)
        self.assertEqual(result.strategy_name, 'Buy and Hold')
        self.assertGreater(result.initial_value, 0)

    def test_optimization_result_dataclass(self):
        """Test OptimizationResult dataclass."""
        result = OptimizationResult(
            threshold_percent=1.0,
            interval_minutes=60,
            total_return_percent=5.0,
            annualized_return_percent=65.0,
            sharpe_ratio=1.5,
            max_drawdown_percent=10.0,
            total_fees_paid=50.0,
            net_return_percent=4.5,
            num_rebalances=10,
            num_trades=20
        )

        # Test to_dict conversion
        result_dict = result.to_dict()

        self.assertEqual(result_dict['threshold_pct'], 1.0)
        self.assertEqual(result_dict['interval_min'], 60)
        self.assertEqual(result_dict['total_return_pct'], 5.0)
        self.assertEqual(result_dict['num_rebalances'], 10)

    def test_small_optimization_run(self):
        """Test running a small optimization."""
        optimizer = StrategyOptimizer(
            self.price_data,
            self.sim_config,
            max_workers=1  # Single worker for testing
        )

        # Run small optimization (2x2 grid = 4 combinations)
        summary = optimizer.optimize(
            threshold_min=1.0,
            threshold_max=2.0,
            threshold_step=1.0,
            interval_min=1440,  # Daily
            interval_max=2880,  # 2 days
            interval_step=1440,
            show_progress=False
        )

        # Check summary
        self.assertEqual(summary.total_combinations, 4)
        self.assertEqual(len(summary.strategy_results), 4)
        self.assertIsNotNone(summary.baseline_result)
        self.assertIsNotNone(summary.best_return)
        self.assertIsNotNone(summary.best_sharpe)

        # Check that best performers are identified
        self.assertGreater(summary.best_return.total_return_percent, -1000)  # Sanity check

    def test_get_top_strategies(self):
        """Test getting top strategies by metric."""
        optimizer = StrategyOptimizer(self.price_data, self.sim_config, max_workers=1)

        summary = optimizer.optimize(
            threshold_min=1.0,
            threshold_max=2.0,
            threshold_step=1.0,
            interval_min=1440,
            interval_max=2880,
            interval_step=1440,
            show_progress=False
        )

        # Get top 2 by return
        top_strategies = optimizer.get_top_strategies(summary, 'total_return_percent', n=2)

        self.assertEqual(len(top_strategies), 2)
        # First should have higher or equal return than second
        self.assertGreaterEqual(
            top_strategies[0].total_return_percent,
            top_strategies[1].total_return_percent
        )


class TestVisualization(unittest.TestCase):
    """Test visualization functions."""

    def test_csv_export(self):
        """Test CSV export functionality."""
        import tempfile
        import csv
        from src.visualization import export_to_csv

        # Create mock results
        results = [
            OptimizationResult(
                threshold_percent=1.0,
                interval_minutes=60,
                total_return_percent=5.0,
                annualized_return_percent=65.0,
                sharpe_ratio=1.5,
                max_drawdown_percent=10.0,
                total_fees_paid=50.0,
                net_return_percent=4.5,
                num_rebalances=10,
                num_trades=20
            ),
            OptimizationResult(
                threshold_percent=2.0,
                interval_minutes=120,
                total_return_percent=4.0,
                annualized_return_percent=55.0,
                sharpe_ratio=1.3,
                max_drawdown_percent=12.0,
                total_fees_paid=30.0,
                net_return_percent=3.7,
                num_rebalances=5,
                num_trades=10
            )
        ]

        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_path = f.name

        try:
            export_to_csv(results, temp_path)

            # Read back and verify
            with open(temp_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 2)
            self.assertEqual(float(rows[0]['threshold_pct']), 1.0)
            self.assertEqual(float(rows[1]['threshold_pct']), 2.0)

        finally:
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()
