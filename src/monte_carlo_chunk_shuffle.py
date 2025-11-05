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
import json
from pathlib import Path
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

        # Generate seed if not provided
        if config.seed is not None:
            self.seed = config.seed
        else:
            self.seed = random.randint(0, 2**32 - 1)
            logger.info(f"Generated random seed: {self.seed}")

        # Set seeds for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)

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
        new_start_date: datetime,
        target_end_date: Optional[datetime] = None
    ) -> Dict[str, List[Tuple[datetime, float]]]:
        """
        Reassemble shuffled chunks into continuous price data using percentage returns.

        This converts each chunk to percentage returns, then applies them sequentially
        to build a new price path. This preserves real volatility while testing
        different sequences of market conditions.

        Args:
            chunks: List of shuffled chunks
            new_start_date: New starting date for the timeline
            target_end_date: Optional target end date - if provided, will forward-fill
                           to reach this date

        Returns:
            Price data with reassembled chunks and sequential timestamps
        """
        # Collect all assets
        all_assets = set()
        for chunk in chunks:
            all_assets.update(chunk.keys())

        # Reassemble each asset's price series using returns
        reassembled_data = {}

        for asset in all_assets:
            asset_prices = []
            current_date = new_start_date

            # Start with initial price from first chunk containing this asset
            current_price = None
            for chunk in chunks:
                if asset in chunk and chunk[asset]:
                    current_price = chunk[asset][0][1]
                    break

            if current_price is None:
                logger.warning(f"No initial price found for {asset}")
                continue

            for chunk_idx, chunk in enumerate(chunks):
                if asset in chunk:
                    chunk_prices = chunk[asset]

                    if len(chunk_prices) > 1:
                        original_start = chunk_prices[0][0]

                        # For first chunk, add the starting price
                        # For subsequent chunks, add a point at current_date before applying chunk returns
                        if len(asset_prices) == 0:
                            # First chunk - add starting price
                            asset_prices.append((current_date, current_price))
                        else:
                            # Subsequent chunks - add current price at current_date as starting point
                            asset_prices.append((current_date, current_price))

                        # Convert chunk to returns and apply sequentially
                        # Start from index 1 to calculate returns between consecutive prices
                        for i in range(1, len(chunk_prices)):
                            prev_price = chunk_prices[i-1][1]
                            curr_price_raw = chunk_prices[i][1]

                            # Calculate percentage return
                            if prev_price > 0:
                                pct_return = (curr_price_raw - prev_price) / prev_price
                            else:
                                pct_return = 0

                            # Apply return to current price
                            current_price = current_price * (1 + pct_return)

                            # Calculate timestamp - this is relative to start of reassembled timeline
                            time_offset = (chunk_prices[i][0] - original_start).total_seconds()
                            new_ts = current_date + timedelta(seconds=time_offset)

                            asset_prices.append((new_ts, current_price))

                        # Move current_date forward by chunk duration PLUS one interval
                        # This accounts for moving past the last data point to where next chunk starts
                        if len(chunk_prices) >= 2:
                            # Calculate typical spacing between data points
                            spacing = (chunk_prices[1][0] - chunk_prices[0][0]).total_seconds()
                        else:
                            spacing = 86400  # Default to 1 day if only one point

                        chunk_duration = (chunk_prices[-1][0] - chunk_prices[0][0]).total_seconds()
                        current_date += timedelta(seconds=chunk_duration + spacing)

                    elif len(chunk_prices) == 1:
                        if len(asset_prices) == 0:
                            asset_prices.append((current_date, current_price))
                        # For single-price chunks, advance by 1 day
                        current_date += timedelta(days=1)

                else:
                    # Asset not in this chunk - need to advance current_date anyway
                    # Use the duration from another asset's chunk
                    for other_asset in chunk.keys():
                        if chunk[other_asset]:
                            other_prices = chunk[other_asset]
                            if len(other_prices) >= 2:
                                spacing = (other_prices[1][0] - other_prices[0][0]).total_seconds()
                            else:
                                spacing = 86400

                            chunk_duration = (other_prices[-1][0] - other_prices[0][0]).total_seconds()
                            current_date += timedelta(seconds=chunk_duration + spacing)
                            break

            reassembled_data[asset] = asset_prices

        # If target_end_date specified, forward-fill to reach it
        if target_end_date is not None:
            for asset in reassembled_data.keys():
                asset_prices = reassembled_data[asset]
                if asset_prices:
                    last_timestamp, last_price = asset_prices[-1]

                    # If we haven't reached the target end date, forward-fill
                    if last_timestamp < target_end_date:
                        current_ts = last_timestamp + timedelta(days=1)
                        while current_ts <= target_end_date:
                            asset_prices.append((current_ts, last_price))
                            current_ts += timedelta(days=1)

                        logger.debug(f"{asset}: Forward-filled from {last_timestamp.date()} to {target_end_date.date()}")

        logger.debug(f"Reassembled {len(reassembled_data)} assets from {len(chunks)} chunks using percentage returns")
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
        # Determine start and end dates from original data
        all_timestamps = [ts for prices in price_data.values() for ts, _ in prices]
        if not all_timestamps:
            logger.warning("No timestamps in price data")
            return price_data

        original_start = min(all_timestamps)
        original_end = max(all_timestamps)

        if new_start_date is None:
            new_start_date = original_start

        # Calculate target end date to maintain same duration
        duration = original_end - original_start
        target_end_date = new_start_date + duration

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

        # Reassemble with target end date to ensure full coverage
        shuffled_data = self.reassemble_chunks(shuffled_chunks, new_start_date, target_end_date)

        return shuffled_data


@dataclass
class MonteCarloResult:
    """Aggregated results from multiple Monte Carlo simulations."""
    strategy_name: str
    num_simulations: int
    chunk_days: int
    seed: int  # Random seed used for reproducibility

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
    best_max_drawdown: float

    # Trading metrics
    mean_num_rebalances: float
    mean_total_fees: float

    # Raw simulation results
    all_returns: List[float]
    all_final_values: List[float]
    all_sharpe_ratios: List[float]
    all_max_drawdowns: List[float]

    def to_dict(self, include_raw_data: bool = False) -> dict:
        """Convert to dictionary for JSON serialization.

        Args:
            include_raw_data: If True, include raw distribution arrays (larger file)
        """
        result = {
            'strategy_name': self.strategy_name,
            'num_simulations': self.num_simulations,
            'chunk_days': self.chunk_days,
            'seed': self.seed,
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
                'worst_max_drawdown': self.worst_max_drawdown,
                'best_max_drawdown': self.best_max_drawdown
            },
            'trading': {
                'mean_num_rebalances': self.mean_num_rebalances,
                'mean_total_fees': self.mean_total_fees
            }
        }

        # Include raw distribution arrays if requested
        if include_raw_data:
            result['raw_distributions'] = {
                'returns': self.all_returns,
                'final_values': self.all_final_values,
                'sharpe_ratios': self.all_sharpe_ratios,
                'max_drawdowns': self.all_max_drawdowns
            }

        return result

    def save_to_json(self, file_path: str, original_result=None):
        """Save Monte Carlo results to JSON file.

        Args:
            file_path: Path to save JSON file
            original_result: Optional SimulationResult from original (unshuffled) run
        """
        output = {
            'monte_carlo': self.to_dict(include_raw_data=True),
            'metadata': {
                'saved_at': datetime.now().isoformat(),
                'version': '1.0'
            }
        }

        # Add original result if provided
        if original_result:
            output['original_result'] = {
                'total_return_percent': original_result.total_return_percent,
                'annualized_return_percent': original_result.annualized_return_percent,
                'sharpe_ratio': original_result.sharpe_ratio,
                'max_drawdown_percent': original_result.max_drawdown_percent,
                'num_rebalances': original_result.num_rebalances,
                'total_fees_paid': original_result.total_fees_paid,
                'initial_value': original_result.initial_value,
                'final_value': original_result.final_value
            }

        # Create parent directory if needed
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Save to file
        with open(file_path, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Saved Monte Carlo results to {file_path}")

    def print_summary(self, original_result=None):
        """Print formatted summary of results with optional original strategy comparison."""
        print("\n" + "=" * 80)
        print(f"MONTE CARLO RESULTS: {self.strategy_name}")
        print("=" * 80)
        print(f"Simulations: {self.num_simulations:,}")
        print(f"Chunk Size: {self.chunk_days} days")
        print(f"Random Seed: {self.seed} (use --seed {self.seed} to reproduce)")
        print()

        # Original strategy comparison if provided
        if original_result:
            print("ORIGINAL (UNSHUFFLED) STRATEGY:")
            print(f"  Return:             {original_result.total_return_percent:>10.2f}%")
            print(f"  Sharpe Ratio:       {original_result.sharpe_ratio:>10.3f}")
            print(f"  Max Drawdown:       {original_result.max_drawdown_percent:>10.2f}%")
            print(f"  Rebalances:         {original_result.num_rebalances:>10.0f}")
            print(f"  Fees Paid:          ${original_result.total_fees_paid:>10,.2f}")
            print()

        print("MONTE CARLO DISTRIBUTIONS:")
        print()
        print("RETURNS (emphasis on median):")
        print(f"  Median:     {self.median_return:>10.2f}%  ⭐")
        print(f"  Mean:       {self.mean_return:>10.2f}%")
        print(f"  Std Dev:    {self.std_return:>10.2f}%")
        print(f"  5th %ile:   {self.percentile_5_return:>10.2f}%")
        print(f"  95th %ile:  {self.percentile_95_return:>10.2f}%")
        print(f"  Range:      {self.min_return:>10.2f}% to {self.max_return:>10.2f}%")
        print()
        print("FINAL VALUES:")
        print(f"  Median:     ${self.median_final_value:>12,.2f}  ⭐")
        print(f"  Mean:       ${self.mean_final_value:>12,.2f}")
        print(f"  5th %ile:   ${self.percentile_5_final_value:>12,.2f}")
        print(f"  95th %ile:  ${self.percentile_95_final_value:>12,.2f}")
        print()
        print("RISK METRICS:")
        print(f"  Median Sharpe:      {self.median_sharpe:>8.3f}  ⭐")
        print(f"  Mean Sharpe:        {self.mean_sharpe:>8.3f}")
        print(f"  Median Max DD:      {self.median_max_drawdown:>8.2f}%  ⭐")
        print(f"  Mean Max DD:        {self.mean_max_drawdown:>8.2f}%")
        print(f"  Best Drawdown:      {self.best_max_drawdown:>8.2f}%")
        print(f"  Worst Drawdown:     {self.worst_max_drawdown:>8.2f}%")
        print()
        print("TRADING:")
        print(f"  Mean Rebalances:    {self.mean_num_rebalances:>8.1f}")
        print(f"  Mean Fees Paid:     ${self.mean_total_fees:>10,.2f}")
        print("=" * 80)


def aggregate_simulation_results(
    results: List['SimulationResult'],
    strategy_name: str,
    chunk_days: int,
    seed: int
) -> MonteCarloResult:
    """
    Aggregate results from multiple simulations into summary statistics.

    Args:
        results: List of simulation results
        strategy_name: Name of the strategy
        chunk_days: Chunk size used
        seed: Random seed used for simulations

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
        seed=seed,
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
        best_max_drawdown=float(np.min(max_drawdowns)),
        # Trading
        mean_num_rebalances=float(np.mean(num_rebalances)),
        mean_total_fees=float(np.mean(total_fees)),
        # Raw data
        all_returns=returns,
        all_final_values=final_values,
        all_sharpe_ratios=sharpe_ratios,
        all_max_drawdowns=max_drawdowns
    )


@dataclass
class OriginalResult:
    """Simplified result from original (unshuffled) strategy run."""
    total_return_percent: float
    annualized_return_percent: float
    sharpe_ratio: float
    max_drawdown_percent: float
    num_rebalances: int
    total_fees_paid: float
    initial_value: float
    final_value: float


def load_monte_carlo_results(file_path: str) -> Tuple[MonteCarloResult, Optional[OriginalResult]]:
    """Load Monte Carlo results from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Tuple of (MonteCarloResult, OriginalResult or None)
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    mc_data = data['monte_carlo']
    raw_dist = mc_data.get('raw_distributions', {})

    # Reconstruct MonteCarloResult
    mc_result = MonteCarloResult(
        strategy_name=mc_data['strategy_name'],
        num_simulations=mc_data['num_simulations'],
        chunk_days=mc_data['chunk_days'],
        seed=mc_data['seed'],
        # Returns
        mean_return=mc_data['returns']['mean'],
        median_return=mc_data['returns']['median'],
        std_return=mc_data['returns']['std'],
        percentile_5_return=mc_data['returns']['percentile_5'],
        percentile_95_return=mc_data['returns']['percentile_95'],
        min_return=mc_data['returns']['min'],
        max_return=mc_data['returns']['max'],
        # Final values
        mean_final_value=mc_data['final_values']['mean'],
        median_final_value=mc_data['final_values']['median'],
        std_final_value=mc_data['final_values']['std'],
        percentile_5_final_value=mc_data['final_values']['percentile_5'],
        percentile_95_final_value=mc_data['final_values']['percentile_95'],
        # Risk metrics
        mean_sharpe=mc_data['risk_metrics']['mean_sharpe'],
        median_sharpe=mc_data['risk_metrics']['median_sharpe'],
        mean_max_drawdown=mc_data['risk_metrics']['mean_max_drawdown'],
        median_max_drawdown=mc_data['risk_metrics']['median_max_drawdown'],
        worst_max_drawdown=mc_data['risk_metrics']['worst_max_drawdown'],
        best_max_drawdown=mc_data['risk_metrics']['best_max_drawdown'],
        # Trading
        mean_num_rebalances=mc_data['trading']['mean_num_rebalances'],
        mean_total_fees=mc_data['trading']['mean_total_fees'],
        # Raw data (if available)
        all_returns=raw_dist.get('returns', []),
        all_final_values=raw_dist.get('final_values', []),
        all_sharpe_ratios=raw_dist.get('sharpe_ratios', []),
        all_max_drawdowns=raw_dist.get('max_drawdowns', [])
    )

    # Load original result if present
    original_result = None
    if 'original_result' in data:
        orig = data['original_result']
        original_result = OriginalResult(
            total_return_percent=orig['total_return_percent'],
            annualized_return_percent=orig['annualized_return_percent'],
            sharpe_ratio=orig['sharpe_ratio'],
            max_drawdown_percent=orig['max_drawdown_percent'],
            num_rebalances=orig['num_rebalances'],
            total_fees_paid=orig['total_fees_paid'],
            initial_value=orig['initial_value'],
            final_value=orig['final_value']
        )

    logger.info(f"Loaded Monte Carlo results from {file_path}")
    logger.info(f"  Simulations: {mc_result.num_simulations:,}")
    logger.info(f"  Strategy: {mc_result.strategy_name}")

    return mc_result, original_result


def aggregate_from_metrics(
    metrics: Dict[str, List[float]],
    strategy_name: str,
    chunk_days: int,
    seed: int
) -> MonteCarloResult:
    """
    Aggregate results from pre-extracted summary metrics (memory-efficient).

    Args:
        metrics: Dict with keys 'returns', 'final_values', 'sharpe_ratios',
                'max_drawdowns', 'num_rebalances', 'total_fees'
        strategy_name: Name of the strategy
        chunk_days: Chunk size used
        seed: Random seed used for simulations

    Returns:
        Aggregated Monte Carlo results
    """
    returns = metrics['returns']
    final_values = metrics['final_values']
    sharpe_ratios = metrics['sharpe_ratios']
    max_drawdowns = metrics['max_drawdowns']
    num_rebalances = metrics['num_rebalances']
    total_fees = metrics['total_fees']

    if not returns:
        raise ValueError("No simulation metrics to aggregate")

    return MonteCarloResult(
        strategy_name=strategy_name,
        num_simulations=len(returns),
        chunk_days=chunk_days,
        seed=seed,
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
        best_max_drawdown=float(np.min(max_drawdowns)),
        # Trading
        mean_num_rebalances=float(np.mean(num_rebalances)),
        mean_total_fees=float(np.mean(total_fees)),
        # Raw data
        all_returns=returns,
        all_final_values=final_values,
        all_sharpe_ratios=sharpe_ratios,
        all_max_drawdowns=max_drawdowns
    )
