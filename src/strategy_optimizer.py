"""
Strategy parameter optimizer using grid search.

This module provides tools for finding optimal rebalancing parameters
by testing multiple threshold and interval combinations in parallel.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

from src.monte_carlo_simulator import (
    SimulationConfig,
    PortfolioSimulator,
    HybridStrategy,
    BuyAndHoldStrategy,
    SimulationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results from a single parameter combination."""
    threshold_percent: float
    interval_minutes: float
    total_return_percent: float
    annualized_return_percent: float
    sharpe_ratio: float
    max_drawdown_percent: float
    total_fees_paid: float
    net_return_percent: float  # Return minus fee impact
    num_rebalances: int
    num_trades: int

    # Comparison to baseline
    return_vs_baseline: float = 0.0
    sharpe_vs_baseline: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for export."""
        return {
            'threshold_pct': self.threshold_percent,
            'interval_min': self.interval_minutes,
            'total_return_pct': self.total_return_percent,
            'annualized_return_pct': self.annualized_return_percent,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown_pct': self.max_drawdown_percent,
            'total_fees': self.total_fees_paid,
            'net_return_pct': self.net_return_percent,
            'num_rebalances': self.num_rebalances,
            'num_trades': self.num_trades,
            'return_vs_baseline': self.return_vs_baseline,
            'sharpe_vs_baseline': self.sharpe_vs_baseline,
        }


@dataclass
class OptimizationSummary:
    """Summary of optimization run."""
    baseline_result: SimulationResult
    strategy_results: List[OptimizationResult]
    total_combinations: int
    execution_time_seconds: float

    # Best performers by metric
    best_return: OptimizationResult = None
    best_sharpe: OptimizationResult = None
    best_net_return: OptimizationResult = None
    lowest_fees: OptimizationResult = None
    lowest_drawdown: OptimizationResult = None

    def __post_init__(self):
        """Identify best performers."""
        if not self.strategy_results:
            return

        self.best_return = max(self.strategy_results, key=lambda r: r.total_return_percent)
        self.best_sharpe = max(self.strategy_results, key=lambda r: r.sharpe_ratio)
        self.best_net_return = max(self.strategy_results, key=lambda r: r.net_return_percent)
        self.lowest_fees = min(self.strategy_results, key=lambda r: r.total_fees_paid)
        self.lowest_drawdown = min(self.strategy_results, key=lambda r: r.max_drawdown_percent)


def _run_single_simulation(
    params: Tuple[float, float],
    sim_config: SimulationConfig,
    price_data: Dict
) -> OptimizationResult:
    """
    Run a single simulation with given parameters.

    This function is designed to be pickable for multiprocessing.

    Args:
        params: (threshold_percent, interval_minutes)
        sim_config: Simulation configuration
        price_data: Historical price data

    Returns:
        OptimizationResult with metrics
    """
    threshold, interval_minutes = params

    # Create strategy with these parameters
    strategy = HybridStrategy(
        rebalance_interval_minutes=int(interval_minutes),
        threshold_percent=threshold
    )

    # Run simulation
    simulator = PortfolioSimulator(
        config=sim_config,
        strategy=strategy,
        price_data=price_data
    )

    result = simulator.run()

    # Calculate net return (return minus fee impact)
    initial_capital = sim_config.initial_capital_usd
    fee_impact_percent = (result.total_fees_paid / initial_capital) * 100
    net_return = result.total_return_percent - fee_impact_percent

    # Create optimization result
    return OptimizationResult(
        threshold_percent=threshold,
        interval_minutes=interval_minutes,
        total_return_percent=result.total_return_percent,
        annualized_return_percent=result.annualized_return_percent,
        sharpe_ratio=result.sharpe_ratio,
        max_drawdown_percent=result.max_drawdown_percent,
        total_fees_paid=result.total_fees_paid,
        net_return_percent=net_return,
        num_rebalances=result.num_rebalances,
        num_trades=len(result.trades)
    )


class StrategyOptimizer:
    """Optimizer for finding best rebalancing parameters."""

    def __init__(
        self,
        price_data: Dict,
        sim_config: SimulationConfig,
        max_workers: Optional[int] = None
    ):
        """
        Initialize optimizer.

        Args:
            price_data: Historical price data (shared across simulations)
            sim_config: Base simulation configuration
            max_workers: Number of parallel workers (default: CPU count)
        """
        self.price_data = price_data
        self.sim_config = sim_config
        self.max_workers = max_workers

    def generate_parameter_grid(
        self,
        threshold_min: float,
        threshold_max: float,
        threshold_step: float,
        interval_min: float,
        interval_max: float,
        interval_step: float
    ) -> List[Tuple[float, float]]:
        """
        Generate all parameter combinations to test.

        Returns:
            List of (threshold, interval) tuples
        """
        # Generate threshold values using plain Python
        thresholds = []
        current = threshold_min
        while current <= threshold_max + threshold_step/10:  # Small epsilon for float comparison
            thresholds.append(round(current, 2))
            current += threshold_step

        # Generate interval values
        intervals = []
        current = interval_min
        while current <= interval_max + interval_step/10:  # Small epsilon for float comparison
            intervals.append(round(current, 1))
            current += interval_step

        # Create all combinations
        combinations = list(itertools.product(thresholds, intervals))

        logger.info(f"Generated {len(combinations)} parameter combinations")
        logger.info(f"Thresholds: {len(thresholds)} values from {threshold_min}% to {threshold_max}%")
        logger.info(f"Intervals: {len(intervals)} values from {interval_min} to {interval_max} minutes")

        return combinations

    def run_baseline(self) -> SimulationResult:
        """Run buy-and-hold baseline simulation."""
        logger.info("Running buy-and-hold baseline...")

        strategy = BuyAndHoldStrategy()
        simulator = PortfolioSimulator(
            config=self.sim_config,
            strategy=strategy,
            price_data=self.price_data
        )

        result = simulator.run()
        logger.info(f"Baseline return: {result.total_return_percent:+.2f}%")

        return result

    def optimize(
        self,
        threshold_min: float,
        threshold_max: float,
        threshold_step: float,
        interval_min: float,
        interval_max: float,
        interval_step: float,
        show_progress: bool = True
    ) -> OptimizationSummary:
        """
        Run grid search optimization.

        Args:
            threshold_min: Minimum threshold to test (%)
            threshold_max: Maximum threshold to test (%)
            threshold_step: Step size for threshold (%)
            interval_min: Minimum interval to test (minutes)
            interval_max: Maximum interval to test (minutes)
            interval_step: Step size for interval (minutes)
            show_progress: Whether to show progress bar

        Returns:
            OptimizationSummary with results
        """
        import time
        start_time = time.time()

        # Generate parameter grid
        param_grid = self.generate_parameter_grid(
            threshold_min, threshold_max, threshold_step,
            interval_min, interval_max, interval_step
        )

        # Run baseline
        baseline_result = self.run_baseline()

        # Run simulations in parallel
        logger.info(f"Running {len(param_grid)} simulations using {self.max_workers or 'auto'} workers...")

        results = []

        if show_progress:
            try:
                from tqdm import tqdm
                progress_bar = tqdm(total=len(param_grid), desc="Optimizing", unit="sim")
            except ImportError:
                logger.warning("tqdm not installed, progress bar disabled")
                progress_bar = None
        else:
            progress_bar = None

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            futures = {
                executor.submit(_run_single_simulation, params, self.sim_config, self.price_data): params
                for params in param_grid
            }

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    result = future.result()

                    # Calculate vs baseline
                    result.return_vs_baseline = result.total_return_percent - baseline_result.total_return_percent
                    result.sharpe_vs_baseline = result.sharpe_ratio - baseline_result.sharpe_ratio

                    results.append(result)

                    if progress_bar:
                        progress_bar.update(1)

                except Exception as e:
                    params = futures[future]
                    logger.error(f"Error simulating {params}: {e}")

        if progress_bar:
            progress_bar.close()

        execution_time = time.time() - start_time

        logger.info(f"Optimization complete in {execution_time:.1f} seconds")
        logger.info(f"Tested {len(results)} parameter combinations")

        # Create summary
        summary = OptimizationSummary(
            baseline_result=baseline_result,
            strategy_results=results,
            total_combinations=len(param_grid),
            execution_time_seconds=execution_time
        )

        return summary

    def get_top_strategies(
        self,
        summary: OptimizationSummary,
        metric: str = 'total_return_percent',
        n: int = 10
    ) -> List[OptimizationResult]:
        """
        Get top N strategies by a given metric.

        Args:
            summary: Optimization summary
            metric: Metric to sort by
            n: Number of top strategies to return

        Returns:
            List of top OptimizationResult objects
        """
        sorted_results = sorted(
            summary.strategy_results,
            key=lambda r: getattr(r, metric),
            reverse=True
        )

        return sorted_results[:n]
