"""
Tests for the PriceCache system.
"""

import unittest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from src.price_cache import PriceCache


class TestPriceCache(unittest.TestCase):
    """Test price caching functionality."""

    def setUp(self):
        """Create temporary cache directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = PriceCache(cache_dir=self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_cache_key_generation(self):
        """Test cache key filename generation."""
        asset = "BTC"
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        granularity = "ONE_HOUR"

        key = self.cache._generate_cache_key(asset, start, end, granularity)

        self.assertEqual(key, "BTC_2024-01-01_2024-01-31_ONE_HOUR.json")

    def test_cache_miss(self):
        """Test cache returns None on miss."""
        result = self.cache.get(
            "BTC",
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
            "ONE_DAY"
        )

        self.assertIsNone(result)

    def test_cache_set_and_get(self):
        """Test setting and retrieving cached data."""
        asset = "BTC"
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3)
        granularity = "ONE_DAY"

        # Create test price data
        prices = [
            (datetime(2024, 1, 1), 45000.0),
            (datetime(2024, 1, 2), 46000.0),
            (datetime(2024, 1, 3), 47000.0),
        ]

        # Set cache
        self.cache.set(asset, start, end, granularity, prices)

        # Get from cache
        cached_prices = self.cache.get(asset, start, end, granularity)

        self.assertIsNotNone(cached_prices)
        self.assertEqual(len(cached_prices), 3)
        self.assertEqual(cached_prices[0][1], 45000.0)
        self.assertEqual(cached_prices[1][1], 46000.0)
        self.assertEqual(cached_prices[2][1], 47000.0)

    def test_cache_expiration(self):
        """Test cache expiration based on max_age_days."""
        asset = "BTC"
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3)
        granularity = "ONE_DAY"

        prices = [
            (datetime(2024, 1, 1), 45000.0),
            (datetime(2024, 1, 2), 46000.0),
        ]

        # Set cache
        self.cache.set(asset, start, end, granularity, prices)

        # Should be available with max_age of 1 day
        cached = self.cache.get(asset, start, end, granularity, max_age_days=1)
        self.assertIsNotNone(cached)

        # Should be available with no max_age limit
        cached = self.cache.get(asset, start, end, granularity, max_age_days=None)
        self.assertIsNotNone(cached)

    def test_cache_clear_all(self):
        """Test clearing all cache files."""
        # Create multiple cache entries
        for asset in ["BTC", "ETH"]:
            self.cache.set(
                asset,
                datetime(2024, 1, 1),
                datetime(2024, 1, 3),
                "ONE_DAY",
                [(datetime(2024, 1, 1), 1000.0)]
            )

        # Clear all
        self.cache.clear()

        # Verify cleared
        result = self.cache.get("BTC", datetime(2024, 1, 1), datetime(2024, 1, 3), "ONE_DAY")
        self.assertIsNone(result)

    def test_cache_clear_specific_asset(self):
        """Test clearing cache for specific asset."""
        # Create cache for BTC and ETH
        for asset in ["BTC", "ETH"]:
            self.cache.set(
                asset,
                datetime(2024, 1, 1),
                datetime(2024, 1, 3),
                "ONE_DAY",
                [(datetime(2024, 1, 1), 1000.0)]
            )

        # Clear only BTC
        self.cache.clear(asset="BTC")

        # BTC should be cleared
        result = self.cache.get("BTC", datetime(2024, 1, 1), datetime(2024, 1, 3), "ONE_DAY")
        self.assertIsNone(result)

        # ETH should still exist
        result = self.cache.get("ETH", datetime(2024, 1, 1), datetime(2024, 1, 3), "ONE_DAY")
        self.assertIsNotNone(result)

    def test_list_cache(self):
        """Test listing cache files."""
        # Create cache entries
        self.cache.set(
            "BTC",
            datetime(2024, 1, 1),
            datetime(2024, 1, 3),
            "ONE_DAY",
            [(datetime(2024, 1, 1), 45000.0), (datetime(2024, 1, 2), 46000.0)]
        )

        cache_list = self.cache.list_cache()

        self.assertEqual(len(cache_list), 1)
        self.assertEqual(cache_list[0]['asset'], 'BTC')
        self.assertEqual(cache_list[0]['num_candles'], 2)
        self.assertIn('size_mb', cache_list[0])
        self.assertIn('age_days', cache_list[0])

    def test_cache_stats(self):
        """Test cache statistics."""
        # Empty cache
        stats = self.cache.get_cache_stats()
        self.assertEqual(stats['num_files'], 0)
        self.assertEqual(stats['total_size_mb'], 0.0)

        # Add cache entries
        for asset in ["BTC", "ETH"]:
            self.cache.set(
                asset,
                datetime(2024, 1, 1),
                datetime(2024, 1, 3),
                "ONE_DAY",
                [(datetime(2024, 1, 1), 1000.0)]
            )

        stats = self.cache.get_cache_stats()
        self.assertEqual(stats['num_files'], 2)
        self.assertGreaterEqual(stats['total_size_mb'], 0.0)  # Size might be very small


class TestChunkCalculation(unittest.TestCase):
    """Test chunk calculation for batched API requests."""

    def test_chunk_calculation_within_limit(self):
        """Test that small time ranges don't get chunked."""
        from src.monte_carlo_simulator import HistoricalPriceFetcher
        from unittest.mock import Mock

        mock_client = Mock()
        fetcher = HistoricalPriceFetcher(mock_client, use_cache=False)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)  # 10 days = 10 candles for ONE_DAY
        granularity = "ONE_DAY"

        chunks = fetcher._calculate_chunks(start, end, granularity)

        # Should be single chunk (10 candles < 300 limit)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][0], start)
        self.assertEqual(chunks[0][1], end)

    def test_chunk_calculation_exceeds_limit(self):
        """Test that large time ranges get split into chunks."""
        from src.monte_carlo_simulator import HistoricalPriceFetcher
        from unittest.mock import Mock

        mock_client = Mock()
        fetcher = HistoricalPriceFetcher(mock_client, use_cache=False)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)  # 365 days
        granularity = "ONE_DAY"

        chunks = fetcher._calculate_chunks(start, end, granularity)

        # Should be split into multiple chunks (365 > 300)
        self.assertGreater(len(chunks), 1)

        # Verify chunks cover full range
        self.assertEqual(chunks[0][0], start)
        self.assertLessEqual(chunks[-1][1], end)

    def test_chunk_calculation_hourly(self):
        """Test chunk calculation for hourly granularity."""
        from src.monte_carlo_simulator import HistoricalPriceFetcher
        from unittest.mock import Mock

        mock_client = Mock()
        fetcher = HistoricalPriceFetcher(mock_client, use_cache=False)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 2, 1)  # ~744 hours
        granularity = "ONE_HOUR"

        chunks = fetcher._calculate_chunks(start, end, granularity)

        # Should require multiple chunks (744 > 300)
        self.assertGreater(len(chunks), 2)

    def test_candles_needed_calculation(self):
        """Test calculation of candles needed for different granularities."""
        from src.monte_carlo_simulator import HistoricalPriceFetcher
        from unittest.mock import Mock

        mock_client = Mock()
        fetcher = HistoricalPriceFetcher(mock_client, use_cache=False)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)  # 30 days

        # Daily: ~30 candles
        candles_daily = fetcher._calculate_candles_needed(start, end, "ONE_DAY")
        self.assertAlmostEqual(candles_daily, 30, delta=1)

        # Hourly: ~720 candles (30 days * 24 hours)
        candles_hourly = fetcher._calculate_candles_needed(start, end, "ONE_HOUR")
        self.assertAlmostEqual(candles_hourly, 720, delta=24)

        # Six hour: ~120 candles (30 days * 4)
        candles_6h = fetcher._calculate_candles_needed(start, end, "SIX_HOUR")
        self.assertAlmostEqual(candles_6h, 120, delta=4)


if __name__ == '__main__':
    unittest.main()
