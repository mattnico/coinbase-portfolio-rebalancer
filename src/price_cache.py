"""
Disk-based caching system for historical price data.

Caches price data to avoid repeated API calls when running
multiple simulations or optimizations on the same time period.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional
import hashlib

logger = logging.getLogger(__name__)


class PriceCache:
    """Disk-based cache for historical price data."""

    def __init__(self, cache_dir: str = "data/price_cache"):
        """
        Initialize price cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_cache_key(
        self,
        asset: str,
        start_date: datetime,
        end_date: datetime,
        granularity: str
    ) -> str:
        """
        Generate cache filename for given parameters.

        Args:
            asset: Asset symbol (e.g., 'BTC')
            start_date: Start of price data
            end_date: End of price data
            granularity: Data granularity

        Returns:
            Cache filename
        """
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # Create filename: ASSET_START_END_GRANULARITY.json
        filename = f"{asset}_{start_str}_{end_str}_{granularity}.json"

        return filename

    def get(
        self,
        asset: str,
        start_date: datetime,
        end_date: datetime,
        granularity: str,
        max_age_days: Optional[int] = 7
    ) -> Optional[List[Tuple[datetime, float]]]:
        """
        Get cached price data if available and fresh.

        Args:
            asset: Asset symbol
            start_date: Start date
            end_date: End date
            granularity: Data granularity
            max_age_days: Maximum age of cache in days (None = no limit)

        Returns:
            List of (timestamp, price) tuples if cache hit, None if miss
        """
        cache_file = self.cache_dir / self._generate_cache_key(
            asset, start_date, end_date, granularity
        )

        if not cache_file.exists():
            logger.debug(f"Cache miss: {cache_file.name}")
            return None

        # Check cache age
        if max_age_days is not None:
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age > timedelta(days=max_age_days):
                logger.info(f"Cache expired: {cache_file.name} (age: {cache_age.days} days)")
                return None

        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)

            # Convert back to (datetime, float) tuples
            prices = [
                (datetime.fromisoformat(item['timestamp']), item['price'])
                for item in data['prices']
            ]

            logger.info(f"Cache hit: {asset} ({len(prices)} candles)")
            return prices

        except Exception as e:
            logger.error(f"Error loading cache {cache_file}: {e}")
            return None

    def set(
        self,
        asset: str,
        start_date: datetime,
        end_date: datetime,
        granularity: str,
        prices: List[Tuple[datetime, float]]
    ):
        """
        Save price data to cache.

        Args:
            asset: Asset symbol
            start_date: Start date
            end_date: End date
            granularity: Data granularity
            prices: List of (timestamp, price) tuples
        """
        cache_file = self.cache_dir / self._generate_cache_key(
            asset, start_date, end_date, granularity
        )

        try:
            # Convert to JSON-serializable format
            data = {
                'asset': asset,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'granularity': granularity,
                'cached_at': datetime.now().isoformat(),
                'num_candles': len(prices),
                'prices': [
                    {'timestamp': ts.isoformat(), 'price': price}
                    for ts, price in prices
                ]
            }

            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Cached {asset}: {len(prices)} candles to {cache_file.name}")

        except Exception as e:
            logger.error(f"Error saving cache {cache_file}: {e}")

    def clear(self, asset: Optional[str] = None, older_than_days: Optional[int] = None):
        """
        Clear cache files.

        Args:
            asset: If specified, only clear this asset's cache
            older_than_days: If specified, only clear files older than this
        """
        pattern = f"{asset}_*" if asset else "*"

        deleted_count = 0
        for cache_file in self.cache_dir.glob(pattern):
            if not cache_file.is_file():
                continue

            # Check age if specified
            if older_than_days is not None:
                cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if cache_age.days < older_than_days:
                    continue

            try:
                cache_file.unlink()
                deleted_count += 1
                logger.info(f"Deleted cache file: {cache_file.name}")
            except Exception as e:
                logger.error(f"Error deleting {cache_file}: {e}")

        if deleted_count > 0:
            logger.info(f"Cleared {deleted_count} cache file(s)")
        else:
            logger.info("No cache files to clear")

    def list_cache(self) -> List[dict]:
        """
        List all cached files with metadata.

        Returns:
            List of dicts with cache file info
        """
        cache_files = []

        for cache_file in self.cache_dir.glob("*.json"):
            if not cache_file.is_file():
                continue

            try:
                stat = cache_file.stat()
                cache_age = datetime.now() - datetime.fromtimestamp(stat.st_mtime)

                # Try to load metadata
                with open(cache_file, 'r') as f:
                    data = json.load(f)

                cache_files.append({
                    'filename': cache_file.name,
                    'asset': data.get('asset', '?'),
                    'start_date': data.get('start_date', '?'),
                    'end_date': data.get('end_date', '?'),
                    'granularity': data.get('granularity', '?'),
                    'num_candles': data.get('num_candles', 0),
                    'size_mb': stat.st_size / (1024 * 1024),
                    'age_days': cache_age.days,
                    'cached_at': data.get('cached_at', '?'),
                })

            except Exception as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")
                continue

        return cache_files

    def get_cache_stats(self) -> dict:
        """
        Get overall cache statistics.

        Returns:
            Dict with cache stats
        """
        cache_files = self.list_cache()

        if not cache_files:
            return {
                'num_files': 0,
                'total_size_mb': 0.0,
                'oldest_age_days': 0,
                'newest_age_days': 0,
            }

        total_size = sum(f['size_mb'] for f in cache_files)
        ages = [f['age_days'] for f in cache_files]

        return {
            'num_files': len(cache_files),
            'total_size_mb': round(total_size, 2),
            'oldest_age_days': max(ages) if ages else 0,
            'newest_age_days': min(ages) if ages else 0,
        }
