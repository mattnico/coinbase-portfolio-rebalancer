"""
Advanced bootstrap methods for Monte Carlo simulations.

Implements statistically rigorous resampling methods including:
- Stationary Bootstrap (Politis & Romano, 1994)
- Block Bootstrap
- Improved chunk shuffling with proper normalization

Reference:
    Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. 
    Journal of the American Statistical Association, 89(428), 1303-1313.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap resampling."""
    method: str  # 'stationary', 'block', 'chunk_normalized'
    avg_block_size: int = 30  # Average block size in days
    preserve_start: bool = True  # Keep first block in place
    seed: Optional[int] = None


class StationaryBootstrap:
    """
    Stationary Bootstrap with geometric block lengths.
    
    The stationary bootstrap preserves temporal dependencies better than
    fixed-size block bootstrap by using geometrically distributed block lengths.
    This makes it more suitable for financial time series where dependencies
    decay over time.
    
    Reference:
        Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. 
        Journal of the American Statistical Association, 89(428), 1303-1313.
    """
    
    def __init__(self, config: BootstrapConfig):
        """
        Initialize stationary bootstrap.
        
        Args:
            config: Bootstrap configuration including average block size
        """
        self.config = config
        
        # Set random seed for reproducibility
        if config.seed is not None:
            self.seed = config.seed
        else:
            self.seed = np.random.randint(0, 2**32 - 1)
            logger.info(f"Generated random seed: {self.seed}")
        
        np.random.seed(self.seed)
    
    def _calculate_returns(
        self,
        price_data: Dict[str, List[Tuple[datetime, float]]]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate returns from price data.
        
        Args:
            price_data: Dictionary of {asset: [(timestamp, price), ...]}
        
        Returns:
            Dictionary of {asset: returns_array}
        """
        returns_data = {}
        
        for asset, prices in price_data.items():
            if len(prices) < 2:
                logger.warning(f"{asset}: Not enough data points for returns")
                returns_data[asset] = np.array([])
                continue
            
            # Extract prices
            price_values = np.array([p[1] for p in prices])
            
            # Calculate log returns (more stable for multiplication)
            log_returns = np.diff(np.log(price_values))
            
            returns_data[asset] = log_returns
        
        return returns_data
    
    def _generate_block_indices(
        self,
        T: int,
        preserve_start: bool = True
    ) -> List[Tuple[int, int]]:
        """
        Generate block start indices and lengths using geometric distribution.
        
        Args:
            T: Total length of time series
            preserve_start: If True, first block starts at 0
        
        Returns:
            List of (start_index, block_length) tuples
        """
        # Probability of block termination
        p = 1.0 / self.config.avg_block_size
        
        blocks = []
        covered = 0
        
        # First block (optionally preserved)
        if preserve_start:
            # First block of average length starting at 0
            first_length = np.random.geometric(p)
            first_length = min(first_length, T)
            blocks.append((0, first_length))
            covered = first_length
        
        # Generate remaining blocks
        while covered < T:
            # Random start point
            start_idx = np.random.randint(0, T)
            
            # Geometric block length
            block_length = np.random.geometric(p)
            
            # Don't exceed total length
            block_length = min(block_length, T - covered)
            
            blocks.append((start_idx, block_length))
            covered += block_length
        
        return blocks
    
    def resample_returns(
        self,
        price_data: Dict[str, List[Tuple[datetime, float]]],
        new_start_date: Optional[datetime] = None
    ) -> Dict[str, List[Tuple[datetime, float]]]:
        """
        Resample price data using stationary bootstrap on returns.
        
        Args:
            price_data: Original price data
            new_start_date: Start date for resampled data (default: original start)
        
        Returns:
            Resampled price data with same format as input
        """
        # Get all timestamps to determine period
        all_timestamps = [ts for prices in price_data.values() for ts, _ in prices]
        if not all_timestamps:
            logger.warning("No price data to resample")
            return price_data
        
        original_start = min(all_timestamps)
        original_end = max(all_timestamps)
        
        if new_start_date is None:
            new_start_date = original_start
        
        # Calculate returns for all assets
        returns_data = self._calculate_returns(price_data)
        
        # Determine time series length (use longest series)
        T = max(len(returns) for returns in returns_data.values() if len(returns) > 0)
        
        if T == 0:
            logger.warning("No returns to resample")
            return price_data
        
        # Generate block structure
        blocks = self._generate_block_indices(T, self.config.preserve_start)
        
        # Resample returns for each asset
        resampled_data = {}
        
        for asset, original_prices in price_data.items():
            returns = returns_data[asset]
            
            if len(returns) == 0:
                # No returns - keep original data
                resampled_data[asset] = original_prices
                continue
            
            # Build resampled returns sequence
            resampled_returns = []
            
            for start_idx, block_length in blocks:
                # Extract block (with wraparound)
                block = []
                for i in range(block_length):
                    idx = (start_idx + i) % len(returns)
                    block.append(returns[idx])
                
                resampled_returns.extend(block)
            
            # Truncate to original length
            resampled_returns = resampled_returns[:T]
            
            # Convert returns back to prices
            # Start with first price from original data
            initial_price = original_prices[0][1]
            
            prices_resampled = [initial_price]
            for log_return in resampled_returns:
                # Apply log return: P_t = P_{t-1} * exp(r_t)
                new_price = prices_resampled[-1] * np.exp(log_return)
                prices_resampled.append(new_price)
            
            # Create timestamps
            timestamps_resampled = []
            current_date = new_start_date
            
            # Use original time spacing
            if len(original_prices) >= 2:
                time_delta = (original_prices[1][0] - original_prices[0][0])
            else:
                time_delta = timedelta(days=1)
            
            for price in prices_resampled:
                timestamps_resampled.append((current_date, price))
                current_date += time_delta
            
            resampled_data[asset] = timestamps_resampled
        
        logger.info(f"Resampled {len(resampled_data)} assets using stationary bootstrap")
        logger.info(f"  Average block size: {self.config.avg_block_size} days")
        logger.info(f"  Generated {len(blocks)} blocks")
        
        return resampled_data


class ImprovedChunkShuffler:
    """
    Improved chunk shuffling with proper price normalization.
    
    This addresses the issues in the original implementation by:
    1. Normalizing each chunk to start at the current price level
    2. Using percentage returns within chunks
    3. Properly handling the transition between chunks
    """
    
    def __init__(self, chunk_days: int, preserve_start: bool = True, seed: Optional[int] = None):
        """
        Initialize chunk shuffler.
        
        Args:
            chunk_days: Size of chunks in days
            preserve_start: Keep first chunk in place
            seed: Random seed for reproducibility
        """
        self.chunk_days = chunk_days
        self.preserve_start = preserve_start
        
        if seed is not None:
            self.seed = seed
        else:
            self.seed = np.random.randint(0, 2**32 - 1)
        
        np.random.seed(self.seed)
    
    def split_into_chunks(
        self,
        price_data: Dict[str, List[Tuple[datetime, float]]]
    ) -> List[Dict[str, List[Tuple[datetime, float]]]]:
        """Split price data into time-based chunks."""
        all_timestamps = [ts for prices in price_data.values() for ts, _ in prices]
        if not all_timestamps:
            return []
        
        start_date = min(all_timestamps)
        end_date = max(all_timestamps)
        
        chunks = []
        current_date = start_date
        
        while current_date < end_date:
            chunk_end = current_date + timedelta(days=self.chunk_days)
            
            chunk_data = {}
            for asset, prices in price_data.items():
                chunk_prices = [
                    (ts, price) for ts, price in prices
                    if current_date <= ts < chunk_end
                ]
                if chunk_prices:
                    chunk_data[asset] = chunk_prices
            
            if chunk_data:
                chunks.append(chunk_data)
            
            current_date = chunk_end
        
        logger.debug(f"Split data into {len(chunks)} chunks of {self.chunk_days} days each")
        return chunks
    
    def shuffle_chunks(
        self,
        chunks: List[Dict[str, List[Tuple[datetime, float]]]]
    ) -> List[Dict[str, List[Tuple[datetime, float]]]]:
        """Shuffle chunks while optionally preserving the first."""
        if len(chunks) <= 1:
            return chunks.copy()
        
        if self.preserve_start:
            # Keep first chunk, shuffle rest
            first = [chunks[0]]
            rest = chunks[1:]
            np.random.shuffle(rest)
            return first + rest
        else:
            # Shuffle all chunks
            shuffled = chunks.copy()
            np.random.shuffle(shuffled)
            return shuffled
    
    def reassemble_normalized(
        self,
        chunks: List[Dict[str, List[Tuple[datetime, float]]]],
        new_start_date: datetime
    ) -> Dict[str, List[Tuple[datetime, float]]]:
        """
        Reassemble chunks with proper price normalization.
        
        Each chunk's returns are applied starting from the current price level,
        preventing unrealistic price jumps between chunks.
        """
        all_assets = set()
        for chunk in chunks:
            all_assets.update(chunk.keys())
        
        reassembled_data = {}
        
        for asset in all_assets:
            asset_prices = []
            current_date = new_start_date
            
            # Initialize with first available price
            current_price = None
            for chunk in chunks:
                if asset in chunk and chunk[asset]:
                    current_price = chunk[asset][0][1]
                    break
            
            if current_price is None:
                logger.warning(f"No initial price for {asset}")
                continue
            
            for chunk in chunks:
                if asset not in chunk or not chunk[asset]:
                    continue
                
                chunk_prices = chunk[asset]
                
                if len(chunk_prices) < 2:
                    # Single price point - just add it
                    if not asset_prices:  # First point
                        asset_prices.append((current_date, current_price))
                    current_date += timedelta(days=1)
                    continue
                
                # Calculate normalized returns for this chunk
                # This preserves the chunk's return pattern while starting from current_price
                chunk_start_price = chunk_prices[0][1]
                
                # Add first point at current price level
                asset_prices.append((current_date, current_price))
                
                # Apply chunk returns starting from current level
                for i in range(1, len(chunk_prices)):
                    # Calculate return from chunk
                    prev_chunk_price = chunk_prices[i-1][1]
                    curr_chunk_price = chunk_prices[i][1]
                    
                    if prev_chunk_price > 0:
                        chunk_return = (curr_chunk_price - prev_chunk_price) / prev_chunk_price
                    else:
                        chunk_return = 0
                    
                    # Apply return to current price level
                    current_price = current_price * (1 + chunk_return)
                    
                    # Calculate time offset
                    time_offset = chunk_prices[i][0] - chunk_prices[0][0]
                    new_timestamp = current_date + time_offset
                    
                    asset_prices.append((new_timestamp, current_price))
                
                # Advance date for next chunk
                if len(chunk_prices) >= 2:
                    spacing = (chunk_prices[1][0] - chunk_prices[0][0])
                else:
                    spacing = timedelta(days=1)
                
                chunk_duration = chunk_prices[-1][0] - chunk_prices[0][0]
                current_date += chunk_duration + spacing
            
            reassembled_data[asset] = asset_prices
        
        logger.info(f"Reassembled {len(reassembled_data)} assets with normalized prices")
        return reassembled_data
    
    def generate_shuffled_timeline(
        self,
        price_data: Dict[str, List[Tuple[datetime, float]]],
        new_start_date: Optional[datetime] = None
    ) -> Dict[str, List[Tuple[datetime, float]]]:
        """
        Generate shuffled timeline with proper normalization.
        
        Args:
            price_data: Original price data
            new_start_date: Start date for shuffled data
        
        Returns:
            Shuffled and normalized price data
        """
        # Determine start date
        if new_start_date is None:
            all_timestamps = [ts for prices in price_data.values() for ts, _ in prices]
            new_start_date = min(all_timestamps) if all_timestamps else datetime.now()
        
        # Split, shuffle, reassemble
        chunks = self.split_into_chunks(price_data)
        shuffled_chunks = self.shuffle_chunks(chunks)
        normalized_data = self.reassemble_normalized(shuffled_chunks, new_start_date)
        
        return normalized_data
