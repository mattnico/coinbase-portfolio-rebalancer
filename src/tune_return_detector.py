#!/usr/bin/env python3
"""
Tune ReturnDetector thresholds to optimize performance.

Tests different combinations of bull_threshold and bear_threshold to find
optimal settings that maximize overall accuracy while maintaining perfect
extreme detection (100% bull/bear accuracy).

Usage:
    python -m src.tune_return_detector --days 1825
"""

import argparse
import logging
import sys
import warnings
from datetime import datetime, timedelta
from typing import List, Tuple
from dataclasses import dataclass

# Suppress urllib3 OpenSSL warnings
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')

from src.coinbase_client import CoinbaseClient
from src.monte_carlo_simulator import HistoricalPriceFetcher
from src.regime_detector import ReturnDetector, MarketRegime
from src.compare_regime_detectors import (
    determine_ground_truth_regime,
    calculate_detector_performance,
    DetectorPerformance
)


@dataclass
class ThresholdConfig:
    """Configuration for ReturnDetector thresholds."""
    bull_threshold: float
    bear_threshold: float
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"bull={self.bull_threshold:+.1f}%, bear={self.bear_threshold:+.1f}%"


def setup_logging(quiet: bool = False):
    """Configure logging."""
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_threshold_config(
    config: ThresholdConfig,
    reference_prices: List[Tuple[datetime, float]],
    windows: List[Tuple[datetime, datetime, List[Tuple[datetime, float]]]],
    window_days: int
) -> DetectorPerformance:
    """
    Test a specific threshold configuration.

    Args:
        config: Threshold configuration to test
        reference_prices: Full price history
        windows: Pre-generated windows
        window_days: Window size in days

    Returns:
        DetectorPerformance metrics
    """
    detector = ReturnDetector(
        window_days=window_days,
        bull_threshold=config.bull_threshold,
        bear_threshold=config.bear_threshold
    )

    detections = []

    for window_start, window_end, window_prices in windows:
        # Extract prices and timestamps from tuples
        prices = [p[1] for p in window_prices]
        timestamps = [p[0] for p in window_prices]

        # Determine ground truth
        return_30d = ((prices[-1] / prices[0]) - 1) * 100

        if return_30d > 15:
            ground_truth = MarketRegime.BULL
        elif return_30d < -10:
            ground_truth = MarketRegime.BEAR
        else:
            ground_truth = MarketRegime.NEUTRAL

        # Run detector
        detection = detector.detect(prices, timestamps)
        detections.append((detection, ground_truth))

    return calculate_detector_performance(detections, config.name)


def print_tuning_summary(performances: List[DetectorPerformance]):
    """Print summary of threshold tuning results."""
    print("\n" + "="*120)
    print("RETURN DETECTOR THRESHOLD TUNING RESULTS")
    print("="*120)

    # Sort by overall accuracy first, prioritizing perfect bull/bear detection
    sorted_perf = sorted(
        performances,
        key=lambda p: (p.bull_accuracy + p.bear_accuracy, p.accuracy),
        reverse=True
    )

    print(f"\n{'Rank':<6} {'Configuration':<30} {'Overall':<10} {'BULL':<10} {'BEAR':<10} {'NEUTRAL':<10} {'Confidence':<12}")
    print("-" * 120)

    for i, perf in enumerate(sorted_perf, 1):
        # Highlight perfect extreme detection
        perfect = perf.bull_accuracy == 100 and perf.bear_accuracy == 100
        marker = "🏆" if i == 1 and perfect else "⭐" if perfect else "  "
        print(f"{marker}{i:<5} {perf.detector_name:<30} "
              f"{perf.accuracy:>6.2f}%   "
              f"{perf.bull_accuracy:>6.2f}%   "
              f"{perf.bear_accuracy:>6.2f}%   "
              f"{perf.neutral_accuracy:>6.2f}%   "
              f"{perf.avg_confidence:>6.3f}")

    # Show top 5 in detail
    print("\n" + "="*120)
    print("TOP 5 CONFIGURATIONS (DETAILED)")
    print("="*120)

    for i, perf in enumerate(sorted_perf[:5], 1):
        print(f"\n#{i} - {perf.detector_name}")
        print(f"  Overall: {perf.accuracy:.2f}% ({perf.correct_predictions}/{perf.total_windows})")
        print(f"  BULL:    {perf.bull_accuracy:.2f}% ({perf.bull_correct} correct, {perf.bull_wrong} false positive)")
        print(f"  BEAR:    {perf.bear_accuracy:.2f}% ({perf.bear_correct} correct, {perf.bear_wrong} false positive)")
        print(f"  NEUTRAL: {perf.neutral_accuracy:.2f}% ({perf.neutral_correct} correct, {perf.neutral_wrong} false positive)")
        print(f"  Confidence: {perf.avg_confidence:.3f} (correct: {perf.avg_confidence_correct:.3f}, wrong: {perf.avg_confidence_wrong:.3f})")

    # Find configurations with perfect extreme detection
    print("\n" + "="*120)
    print("PERFECT EXTREME DETECTION (100% BULL + 100% BEAR)")
    print("="*120)

    perfect_configs = [p for p in performances if p.bull_accuracy == 100 and p.bear_accuracy == 100]
    if perfect_configs:
        # Sort by overall accuracy
        perfect_configs = sorted(perfect_configs, key=lambda p: p.accuracy, reverse=True)
        print(f"{'Rank':<6} {'Configuration':<30} {'Overall Accuracy':<18} {'NEUTRAL Accuracy':<18} {'False Bulls':<15}")
        print("-" * 120)

        for i, perf in enumerate(perfect_configs[:10], 1):
            print(f"{i:<6} {perf.detector_name:<30} {perf.accuracy:>10.2f}%       "
                  f"{perf.neutral_accuracy:>10.2f}%       {perf.bull_wrong:<10}")
    else:
        print("No configurations achieved 100% on both BULL and BEAR detection.")

    print("\n" + "="*120)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Tune ReturnDetector thresholds for optimal performance'
    )

    # Time period
    time_group = parser.add_mutually_exclusive_group(required=True)
    time_group.add_argument(
        '--days',
        type=int,
        help='Number of days of historical data to analyze'
    )
    time_group.add_argument(
        '--start',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end',
        type=str,
        help='End date (YYYY-MM-DD), defaults to today'
    )

    parser.add_argument(
        '--window-days',
        type=int,
        default=30,
        help='Rolling window size (default: 30)'
    )

    parser.add_argument(
        '--step-days',
        type=int,
        default=7,
        help='Step size between windows (default: 7)'
    )

    parser.add_argument(
        '--reference-asset',
        type=str,
        default='BTC',
        help='Asset to use for detection (default: BTC)'
    )

    # Tuning ranges
    parser.add_argument(
        '--bull-min',
        type=float,
        default=5.0,
        help='Minimum bull threshold to test (30-day return %%) (default: 5.0)'
    )

    parser.add_argument(
        '--bull-max',
        type=float,
        default=25.0,
        help='Maximum bull threshold to test (30-day return %%) (default: 25.0)'
    )

    parser.add_argument(
        '--bull-step',
        type=float,
        default=1.0,
        help='Bull threshold step size (%%) (default: 1.0)'
    )

    parser.add_argument(
        '--bear-min',
        type=float,
        default=-25.0,
        help='Minimum bear threshold to test (30-day return %%) (default: -25.0)'
    )

    parser.add_argument(
        '--bear-max',
        type=float,
        default=-5.0,
        help='Maximum bear threshold to test (30-day return %%) (default: -5.0)'
    )

    parser.add_argument(
        '--bear-step',
        type=float,
        default=1.0,
        help='Bear threshold step size (%%) (default: 1.0)'
    )

    # Cache options
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Skip cache'
    )

    parser.add_argument(
        '--cache-only',
        action='store_true',
        help='Only use cached data'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed logging'
    )

    return parser.parse_args()


def main():
    """Run threshold tuning."""
    args = parse_arguments()

    setup_logging(args.quiet)
    logger = logging.getLogger(__name__)

    # Parse dates
    if args.days:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
    else:
        start_date = datetime.fromisoformat(args.start)
        if args.end:
            end_date = datetime.fromisoformat(args.end)
        else:
            end_date = datetime.now()

    # Generate threshold configurations to test
    configs = []
    bull_val = args.bull_min
    while bull_val <= args.bull_max + 0.001:
        bear_val = args.bear_min
        while bear_val <= args.bear_max + 0.001:
            configs.append(ThresholdConfig(
                bull_threshold=bull_val,
                bear_threshold=bear_val
            ))
            bear_val += args.bear_step
        bull_val += args.bull_step

    print("\n" + "="*120)
    print("RETURN DETECTOR THRESHOLD TUNING")
    print("="*120)
    print(f"\nAnalysis Period: {start_date.date()} to {end_date.date()}")
    print(f"Window Size: {args.window_days} days")
    print(f"Step Size: {args.step_days} days")
    print(f"Reference Asset: {args.reference_asset}")
    print(f"\nTesting {len(configs)} threshold combinations:")
    print(f"  Bull: {args.bull_min}% to {args.bull_max}% (step {args.bull_step}%)")
    print(f"  Bear: {args.bear_min}% to {args.bear_max}% (step {args.bear_step}%)")
    print("="*120 + "\n")

    # Initialize Coinbase client and fetch data
    logger.info("Initializing Coinbase client...")
    try:
        client = CoinbaseClient()
    except Exception as e:
        logger.error(f"Failed to initialize client: {e}")
        sys.exit(1)

    logger.info(f"Fetching historical price data for {args.reference_asset}...")
    fetcher = HistoricalPriceFetcher(
        client,
        use_cache=not args.no_cache,
        cache_max_age_days=7 if not args.no_cache else None
    )

    try:
        price_data = fetcher.fetch_historical_prices(
            assets=[args.reference_asset],
            start_date=start_date,
            end_date=end_date,
            granularity='ONE_DAY',
            show_progress=not args.quiet,
            cache_only=args.cache_only
        )
    except Exception as e:
        logger.error(f"Failed to fetch price data: {e}")
        sys.exit(1)

    reference_prices = price_data[args.reference_asset]
    logger.info(f"Loaded {len(reference_prices)} data points")

    # Generate windows once
    logger.info("Generating rolling windows...")
    current_date = start_date
    windows = []

    while current_date + timedelta(days=args.window_days) <= end_date:
        window_end = current_date + timedelta(days=args.window_days)

        window_prices = [
            p for p in reference_prices
            if current_date <= p[0] <= window_end
        ]

        if len(window_prices) >= args.window_days:
            windows.append((current_date, window_end, window_prices))

        current_date += timedelta(days=args.step_days)

    logger.info(f"Generated {len(windows)} windows")

    # Test all configurations
    print(f"Testing {len(configs)} configurations...")
    performances = []

    for i, config in enumerate(configs, 1):
        if i % 50 == 0 or i == len(configs):
            print(f"  Progress: {i}/{len(configs)} configurations tested...")

        perf = test_threshold_config(config, reference_prices, windows, args.window_days)
        performances.append(perf)

    print("  Complete!\n")

    # Print results
    print_tuning_summary(performances)

    # Recommend winner (prioritize perfect extreme detection)
    perfect_configs = [p for p in performances if p.bull_accuracy == 100 and p.bear_accuracy == 100]

    if perfect_configs:
        winner = max(perfect_configs, key=lambda p: p.accuracy)
        print(f"\n{'='*120}")
        print(f"🏆 RECOMMENDED CONFIGURATION: {winner.detector_name}")
        print(f"{'='*120}")
        print(f"Overall Accuracy: {winner.accuracy:.2f}%")
        print(f"BULL Accuracy: {winner.bull_accuracy:.2f}% (PERFECT)")
        print(f"BEAR Accuracy: {winner.bear_accuracy:.2f}% (PERFECT)")
        print(f"NEUTRAL Accuracy: {winner.neutral_accuracy:.2f}%")
        print(f"False BULL predictions: {winner.bull_wrong}")
        print(f"\nTo use this configuration:")
        print(f"```python")
        print(f"detector = ReturnDetector(")
        print(f"    bull_threshold={winner.detector_name.split('bull=')[1].split('%')[0]},")
        print(f"    bear_threshold={winner.detector_name.split('bear=')[1].split('%')[0]}")
        print(f")")
        print(f"```")
    else:
        winner = max(performances, key=lambda p: p.accuracy)
        print(f"\n{'='*120}")
        print(f"⚠️  NO CONFIGURATION ACHIEVED PERFECT EXTREME DETECTION")
        print(f"{'='*120}")
        print(f"Best overall: {winner.detector_name} with {winner.accuracy:.2f}% accuracy")
        print(f"BULL: {winner.bull_accuracy:.2f}%, BEAR: {winner.bear_accuracy:.2f}%")

    print(f"{'='*120}\n")


if __name__ == '__main__':
    main()
