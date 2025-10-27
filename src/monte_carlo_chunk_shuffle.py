"""
Monte Carlo simulation using chunk-shuffled historical data.

This approach uses real historical price data but shuffles the order of time periods
to create alternative historical paths. This preserves:
- Real price movements and volatility
- Asset correlations within chunks
- Market microstructure

But randomizes:
- The sequence of bull/bear periods
- The timing of market events

This tests strategy robustness to sequence risk and provides distributions of outcomes.
"""

import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

if TYPE_CHECKING:
    from src.monte_carlo_simulator import SimulationResult

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation with chunk shuffling."""
    chunk_days: int = 30  # Size of chunks to shuffle
    num_simulations: int = 1000  # Number of shuffled sequences to test
    preserve_start_chunk: bool = True  # Keep first chunk in place (start from known state)
    preserve_end_chunk: bool = False  # Keep last chunk in place
    seed: Optional[int] = None  # Random seed for reproducibility


class ChunkShuffler:
    """Shuffles historical price data in chunks to create alternative timelines."""

    def __init__(self, config: MonteCarloConfig):
        """
        Initialize chunk shuffler.

        Args:
            config: Monte Carlo configuration
        """
        self.config = config
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)

    def split_into_chunks(
        self,
        price_data: Dict[str, List[Tuple[datetime, float]]],
        chunk_days: int
    ) -> List[Dict[str, List[Tuple[datetime, float]]]]:
        """
        Split price data into time-based chunks.

        Args:
            price_data: Original price data {asset: [(timestamp, price), ...]}
            chunk_days: Number of days per chunk

        Returns:
            List of chunks, each chunk is a dict like price_data
        """
        # Find the date range
        all_timestamps = []
        for asset, prices in price_data.items():
            all_timestamps.extend([ts for ts, _ in prices])

        if not all_timestamps:
            logger.warning("No price data provided")
            return []

        start_date = min(all_timestamps)
        end_date = max(all_timestamps)

        # Create chunk boundaries
        chunks = []
        current_date = start_date

        while current_date < end_date:
            chunk_end = current_date + timedelta(days=chunk_days)

            # Extract data for this chunk for all assets
            chunk_data = {}
            for asset, prices in price_data.items():
                chunk_prices = [
                    (ts, price) for ts, price in prices
                    if current_date <= ts < chunk_end
                ]
                if chunk_prices:  # Only include if we have data for this chunk
                    chunk_data[asset] = chunk_prices

            if chunk_data:  # Only add chunk if it has data
                chunks.append(chunk_data)

            current_date = chunk_end

        logger.info(f"Split data into {len(chunks)} chunks of ~{chunk_days} days each")
        return chunks

    def shuffle_chunks(
        self,
        chunks: List[Dict[str, List[Tuple[datetime, float]]]],
        preserve_start: bool = True,
        preserve_end: bool = False
    ) -> List[Dict[str, List[Tuple[datetime, float]]]]:
        """
        Shuffle chunks randomly while optionally preserving start/end.

        Args:
            chunks: List of data chunks
            preserve_start: If True, keep first chunk in position
            preserve_end: If True, keep last chunk in position

        Returns:
            Shuffled list of chunks
        """
        if len(chunks) <= 1:
            return chunks.copy()

        # Determine which chunks to shuffle
        start_idx = 1 if preserve_start else 0
        end_idx = len(chunks) - 1 if preserve_end else len(chunks)

        if start_idx >= end_idx:
            return chunks.copy()

        # Extract chunks to shuffle
        chunks_to_shuffle = chunks[start_idx:end_idx]
        random.shuffle(chunks_to_shuffle)

        # Reassemble
        result = []
        if preserve_start:
            result.append(chunks[0])
        result.extend(chunks_to_shuffle)
        if preserve_end:
            result.append(chunks[-1])

        return result

    def reassemble_chunks(
        self,
        chunks: List[Dict[str, List[Tuple[datetime, float]]]],
        new_start_date: datetime
    ) -> Dict[str, List[Tuple[datetime, float]]]:
        """
        Reassemble shuffled chunks into continuous price data with new timestamps.

        Args:
            chunks: List of shuffled chunks
            new_start_date: New starting date for the timeline

        Returns:
            Price data with reassembled chunks and sequential timestamps
        """
        # Collect all assets
        all_assets = set()
        for chunk in chunks:
            all_assets.update(chunk.keys())

        # Reassemble each asset's price series
        reassembled_data = {}
        current_date = new_start_date

        for asset in all_assets:
            asset_prices = []

            for chunk in chunks:
                if asset in chunk:
                    chunk_prices = chunk[asset]

                    # Calculate original time deltas within chunk
                    if len(chunk_prices) > 1:
                        original_start = chunk_prices[0][0]
                        for ts, price in chunk_prices:
                            # Preserve relative timing within chunk
                            time_offset = (ts - original_start).total_seconds()
                            new_ts = current_date + timedelta(seconds=time_offset)
                            asset_prices.append((new_ts, price))
                    elif len(chunk_prices) == 1:
                        asset_prices.append((current_date, chunk_prices[0][1]))

                # Move current_date forward by chunk duration
                if asset in chunk and chunk[asset]:
                    chunk_duration = (chunk[asset][-1][0] - chunk[asset][0][0]).total_seconds()
                    current_date += timedelta(seconds=chunk_duration)

            reassembled_data[asset] = asset_prices

            # Reset current_date for next asset
            current_date = new_start_date

        logger.debug(f"Reassembled {len(reassembled_data)} assets from {len(chunks)} chunks")
        return reassembled_data

    def generate_shuffled_timeline(
        self,
        price_data: Dict[str, List[Tuple[datetime, float]]],
        new_start_date: Optional[datetime] = None
    ) -> Dict[str, List[Tuple[datetime, float]]]:
        """
        Generate one shuffled timeline from historical data.

        Args:
            price_data: Original historical price data
            new_start_date: Optional new start date (defaults to original start)

        Returns:
            Shuffled price data with same format as input
        """
        # Determine start date
        if new_start_date is None:
            all_timestamps = [ts for prices in price_data.values() for ts, _ in prices]
            new_start_date = min(all_timestamps) if all_timestamps else datetime.now()

        # Split into chunks
        chunks = self.split_into_chunks(price_data, self.config.chunk_days)

        if not chunks:
            logger.warning("No chunks created, returning original data")
            return price_data

        # Shuffle chunks
        shuffled_chunks = self.shuffle_chunks(
            chunks,
            preserve_start=self.config.preserve_start_chunk,
            preserve_end=self.config.preserve_end_chunk
        )

        # Reassemble
        shuffled_data = self.reassemble_chunks(shuffled_chunks, new_start_date)

        return shuffled_data


@dataclass
class MonteCarloResult:
    """Aggregated results from multiple Monte Carlo simulations."""
    strategy_name: str
    num_simulations: int
    chunk_days: int

    # Return distribution
    mean_return: float
    median_return: float
    std_return: float
    percentile_5_return: float
    percentile_95_return: float
    min_return: float
    max_return: float

    # Final value distribution
    mean_final_value: float
    median_final_value: float
    std_final_value: float
    percentile_5_final_value: float
    percentile_95_final_value: float

    # Risk metrics distribution
    mean_sharpe: float
    median_sharpe: float
    mean_max_drawdown: float
    median_max_drawdown: float
    worst_max_drawdown: float

    # Trading metrics
    mean_num_rebalances: float
    mean_total_fees: float

    # Raw simulation results
    all_returns: List[float]
    all_final_values: List[float]
    all_sharpe_ratios: List[float]
    all_max_drawdowns: List[float]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'strategy_name': self.strategy_name,
            'num_simulations': self.num_simulations,
            'chunk_days': self.chunk_days,
            'returns': {
                'mean': self.mean_return,
                'median': self.median_return,
                'std': self.std_return,
                'percentile_5': self.percentile_5_return,
                'percentile_95': self.percentile_95_return,
                'min': self.min_return,
                'max': self.max_return
            },
            'final_values': {
                'mean': self.mean_final_value,
                'median': self.median_final_value,
                'std': self.std_final_value,
                'percentile_5': self.percentile_5_final_value,
                'percentile_95': self.percentile_95_final_value
            },
            'risk_metrics': {
                'mean_sharpe': self.mean_sharpe,
                'median_sharpe': self.median_sharpe,
                'mean_max_drawdown': self.mean_max_drawdown,
                'median_max_drawdown': self.median_max_drawdown,
                'worst_max_drawdown': self.worst_max_drawdown
            },
            'trading': {
                'mean_num_rebalances': self.mean_num_rebalances,
                'mean_total_fees': self.mean_total_fees
            }
        }

    def print_summary(self):
        """Print formatted summary of results."""
        print("\n" + "=" * 80)
        print(f"MONTE CARLO RESULTS: {self.strategy_name}")
        print("=" * 80)
        print(f"Simulations: {self.num_simulations}")
        print(f"Chunk Size: {self.chunk_days} days")
        print()
        print("RETURNS:")
        print(f"  Mean:       {self.mean_return:>10.2f}%")
        print(f"  Median:     {self.median_return:>10.2f}%")
        print(f"  Std Dev:    {self.std_return:>10.2f}%")
        print(f"  5th %ile:   {self.percentile_5_return:>10.2f}%")
        print(f"  95th %ile:  {self.percentile_95_return:>10.2f}%")
        print(f"  Range:      {self.min_return:>10.2f}% to {self.max_return:>10.2f}%")
        print()
        print("FINAL VALUES:")
        print(f"  Mean:       ${self.mean_final_value:>12,.2f}")
        print(f"  Median:     ${self.median_final_value:>12,.2f}")
        print(f"  5th %ile:   ${self.percentile_5_final_value:>12,.2f}")
        print(f"  95th %ile:  ${self.percentile_95_final_value:>12,.2f}")
        print()
        print("RISK METRICS:")
        print(f"  Mean Sharpe:        {self.mean_sharpe:>8.3f}")
        print(f"  Median Sharpe:      {self.median_sharpe:>8.3f}")
        print(f"  Mean Max Drawdown:  {self.mean_max_drawdown:>8.2f}%")
        print(f"  Worst Drawdown:     {self.worst_max_drawdown:>8.2f}%")
        print()
        print("TRADING:")
        print(f"  Mean Rebalances:    {self.mean_num_rebalances:>8.1f}")
        print(f"  Mean Fees Paid:     ${self.mean_total_fees:>10,.2f}")
        print("=" * 80)


def aggregate_simulation_results(
    results: List['SimulationResult'],
    strategy_name: str,
    chunk_days: int
) -> MonteCarloResult:
    """
    Aggregate results from multiple simulations into summary statistics.

    Args:
        results: List of simulation results
        strategy_name: Name of the strategy
        chunk_days: Chunk size used

    Returns:
        Aggregated Monte Carlo results
    """
    if not results:
        raise ValueError("No simulation results to aggregate")

    # Extract metrics
    returns = [r.total_return_percent for r in results]
    final_values = [r.final_value for r in results]
    sharpe_ratios = [r.sharpe_ratio for r in results]
    max_drawdowns = [r.max_drawdown_percent for r in results]
    num_rebalances = [r.num_rebalances for r in results]
    total_fees = [r.total_fees_paid for r in results]

    return MonteCarloResult(
        strategy_name=strategy_name,
        num_simulations=len(results),
        chunk_days=chunk_days,
        # Returns
        mean_return=float(np.mean(returns)),
        median_return=float(np.median(returns)),
        std_return=float(np.std(returns)),
        percentile_5_return=float(np.percentile(returns, 5)),
        percentile_95_return=float(np.percentile(returns, 95)),
        min_return=float(np.min(returns)),
        max_return=float(np.max(returns)),
        # Final values
        mean_final_value=float(np.mean(final_values)),
        median_final_value=float(np.median(final_values)),
        std_final_value=float(np.std(final_values)),
        percentile_5_final_value=float(np.percentile(final_values, 5)),
        percentile_95_final_value=float(np.percentile(final_values, 95)),
        # Risk metrics
        mean_sharpe=float(np.mean(sharpe_ratios)),
        median_sharpe=float(np.median(sharpe_ratios)),
        mean_max_drawdown=float(np.mean(max_drawdowns)),
        median_max_drawdown=float(np.median(max_drawdowns)),
        worst_max_drawdown=float(np.max(max_drawdowns)),
        # Trading
        mean_num_rebalances=float(np.mean(num_rebalances)),
        mean_total_fees=float(np.mean(total_fees)),
        # Raw data
        all_returns=returns,
        all_final_values=final_values,
        all_sharpe_ratios=sharpe_ratios,
        all_max_drawdowns=max_drawdowns
    )
