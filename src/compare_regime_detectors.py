#!/usr/bin/env python3
"""
Compare market regime detection methods head-to-head.

Tests 5 competing detection methods against historical data to see which
best predicts when different portfolio strategies should be used.

Uses the rolling windows optimization results as ground truth:
- Equal_Weight wins → should detect BULL
- Stablecoin_Heavy wins → should detect BEAR
- Top_Three wins → should detect NEUTRAL

Usage:
    python -m src.compare_regime_detectors \\
        --days 365 \\
        --window-days 30 \\
        --config config/top_four_portfolios.json
"""

import argparse
import logging
import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import Counter

# Suppress urllib3 OpenSSL warnings
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')

from src.coinbase_client import CoinbaseClient
from src.monte_carlo_simulator import HistoricalPriceFetcher
from src.regime_detector import (
    get_all_detectors,
    MarketRegime,
    RegimeDetection,
    RegimeDetectorBase
)


@dataclass
class DetectorPerformance:
    """Performance metrics for a regime detector."""
    detector_name: str
    total_windows: int
    correct_predictions: int
    accuracy: float

    # Confusion matrix
    bull_correct: int  # Correctly predicted BULL
    bull_wrong: int    # Predicted BULL when should be other
    bear_correct: int
    bear_wrong: int
    neutral_correct: int
    neutral_wrong: int

    # Regime-specific accuracy
    bull_accuracy: float  # % of actual bull markets correctly identified
    bear_accuracy: float
    neutral_accuracy: float

    # Average confidence scores
    avg_confidence: float
    avg_confidence_correct: float  # Confidence when correct
    avg_confidence_wrong: float    # Confidence when wrong


def setup_logging(quiet: bool = False):
    """Configure logging."""
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str) -> dict:
    """Load portfolio configuration."""
    config_file = Path(config_path)

    if not config_file.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    with open(config_file, 'r') as f:
        config = json.load(f)

    return config


def determine_ground_truth_regime(winning_portfolio: str) -> MarketRegime:
    """
    Determine what regime a window was based on winning portfolio.

    Based on rolling windows analysis:
    - Equal_Weight wins in bull markets → BULL
    - Stablecoin_Heavy wins in bear markets → BEAR
    - Top_Three wins in neutral/mixed markets → NEUTRAL
    """
    if winning_portfolio == 'Equal_Weight':
        return MarketRegime.BULL
    elif winning_portfolio == 'Stablecoin_Heavy':
        return MarketRegime.BEAR
    else:  # Top_Three, Aggressive, BTC_Dominant, etc.
        return MarketRegime.NEUTRAL


def calculate_detector_performance(
    detections: List[Tuple[RegimeDetection, MarketRegime]],
    detector_name: str
) -> DetectorPerformance:
    """
    Calculate performance metrics for a detector.

    Args:
        detections: List of (detection, ground_truth_regime) tuples
        detector_name: Name of detector

    Returns:
        DetectorPerformance metrics
    """
    total = len(detections)
    correct = sum(1 for det, truth in detections if det.regime == truth)

    # Confusion matrix
    bull_correct = sum(1 for det, truth in detections if det.regime == MarketRegime.BULL and truth == MarketRegime.BULL)
    bull_wrong = sum(1 for det, truth in detections if det.regime == MarketRegime.BULL and truth != MarketRegime.BULL)
    bear_correct = sum(1 for det, truth in detections if det.regime == MarketRegime.BEAR and truth == MarketRegime.BEAR)
    bear_wrong = sum(1 for det, truth in detections if det.regime == MarketRegime.BEAR and truth != MarketRegime.BEAR)
    neutral_correct = sum(1 for det, truth in detections if det.regime == MarketRegime.NEUTRAL and truth == MarketRegime.NEUTRAL)
    neutral_wrong = sum(1 for det, truth in detections if det.regime == MarketRegime.NEUTRAL and truth != MarketRegime.NEUTRAL)

    # Regime-specific accuracy
    actual_bull = sum(1 for _, truth in detections if truth == MarketRegime.BULL)
    actual_bear = sum(1 for _, truth in detections if truth == MarketRegime.BEAR)
    actual_neutral = sum(1 for _, truth in detections if truth == MarketRegime.NEUTRAL)

    bull_accuracy = (bull_correct / actual_bull * 100) if actual_bull > 0 else 0
    bear_accuracy = (bear_correct / actual_bear * 100) if actual_bear > 0 else 0
    neutral_accuracy = (neutral_correct / actual_neutral * 100) if actual_neutral > 0 else 0

    # Confidence metrics
    avg_confidence = sum(det.confidence for det, _ in detections) / total
    correct_detections = [det for det, truth in detections if det.regime == truth]
    wrong_detections = [det for det, truth in detections if det.regime != truth]

    avg_confidence_correct = (
        sum(det.confidence for det in correct_detections) / len(correct_detections)
        if correct_detections else 0
    )
    avg_confidence_wrong = (
        sum(det.confidence for det in wrong_detections) / len(wrong_detections)
        if wrong_detections else 0
    )

    return DetectorPerformance(
        detector_name=detector_name,
        total_windows=total,
        correct_predictions=correct,
        accuracy=(correct / total * 100),
        bull_correct=bull_correct,
        bull_wrong=bull_wrong,
        bear_correct=bear_correct,
        bear_wrong=bear_wrong,
        neutral_correct=neutral_correct,
        neutral_wrong=neutral_wrong,
        bull_accuracy=bull_accuracy,
        bear_accuracy=bear_accuracy,
        neutral_accuracy=neutral_accuracy,
        avg_confidence=avg_confidence,
        avg_confidence_correct=avg_confidence_correct,
        avg_confidence_wrong=avg_confidence_wrong
    )


def print_detector_comparison(performances: List[DetectorPerformance]):
    """Print comprehensive comparison of all detectors."""
    print("\n" + "="*100)
    print("REGIME DETECTOR COMPARISON")
    print("="*100)

    # Overall accuracy ranking
    print("\nOVERALL ACCURACY RANKING:")
    print("-" * 100)
    sorted_perf = sorted(performances, key=lambda p: p.accuracy, reverse=True)

    print(f"{'Rank':<6} {'Detector':<20} {'Accuracy':<12} {'Correct':<10} {'Total':<10} {'Avg Confidence':<15}")
    print("-" * 100)

    for i, perf in enumerate(sorted_perf, 1):
        print(f"{i:<6} {perf.detector_name:<20} {perf.accuracy:>7.2f}%     "
              f"{perf.correct_predictions:<10} {perf.total_windows:<10} {perf.avg_confidence:>10.3f}")

    # Detailed breakdown for each detector
    print("\n" + "="*100)
    print("DETAILED PERFORMANCE BY DETECTOR")
    print("="*100)

    for perf in sorted_perf:
        print(f"\n{perf.detector_name}:")
        print(f"  Overall Accuracy: {perf.accuracy:.2f}% ({perf.correct_predictions}/{perf.total_windows})")
        print(f"\n  Regime-Specific Accuracy:")
        print(f"    BULL:    {perf.bull_accuracy:>6.2f}% ({perf.bull_correct} correct)")
        print(f"    BEAR:    {perf.bear_accuracy:>6.2f}% ({perf.bear_correct} correct)")
        print(f"    NEUTRAL: {perf.neutral_accuracy:>6.2f}% ({perf.neutral_correct} correct)")

        print(f"\n  Confusion Matrix:")
        print(f"    Predicted BULL:    {perf.bull_correct} correct, {perf.bull_wrong} wrong")
        print(f"    Predicted BEAR:    {perf.bear_correct} correct, {perf.bear_wrong} wrong")
        print(f"    Predicted NEUTRAL: {perf.neutral_correct} correct, {perf.neutral_wrong} wrong")

        print(f"\n  Confidence Analysis:")
        print(f"    Average:        {perf.avg_confidence:.3f}")
        print(f"    When correct:   {perf.avg_confidence_correct:.3f}")
        print(f"    When wrong:     {perf.avg_confidence_wrong:.3f}")
        print(f"    Calibration:    {'+' if perf.avg_confidence_correct > perf.avg_confidence_wrong else '-'} "
              f"(good detector has higher confidence when correct)")

    print("\n" + "="*100)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare regime detection methods against historical performance',
        formatter_class=argparse.RawDescriptionHelpFormatter
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

    # Detection parameters
    parser.add_argument(
        '--window-days',
        type=int,
        default=30,
        help='Rolling window size for regime detection (default: 30)'
    )

    parser.add_argument(
        '--step-days',
        type=int,
        default=7,
        help='Step size between windows (default: 7, overlapping windows)'
    )

    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='config/top_four_portfolios.json',
        help='Path to portfolio configuration file'
    )

    parser.add_argument(
        '--reference-asset',
        type=str,
        default='BTC',
        help='Asset to use for regime detection (default: BTC)'
    )

    # Cache options
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Skip cache and force fresh API fetch'
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
    """Run detector comparison."""
    args = parse_arguments()

    setup_logging(args.quiet)
    logger = logging.getLogger(__name__)

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    portfolios = config.get('portfolios', {})
    if not portfolios:
        logger.error("No portfolios found in config file")
        sys.exit(1)

    # Get portfolio_id if available
    portfolio_id = config.get('portfolio_id')

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

    print("\n" + "="*100)
    print("REGIME DETECTOR COMPARISON")
    print("="*100)
    print(f"\nAnalysis Period: {start_date.date()} to {end_date.date()}")
    print(f"Window Size: {args.window_days} days")
    print(f"Step Size: {args.step_days} days")
    print(f"Reference Asset: {args.reference_asset}")
    print(f"Testing {len(portfolios)} portfolios")
    print("="*100 + "\n")

    # Initialize Coinbase client
    logger.info("Initializing Coinbase client...")
    try:
        client = CoinbaseClient(portfolio_id=portfolio_id)
    except Exception as e:
        logger.error(f"Failed to initialize Coinbase client: {e}")
        sys.exit(1)

    # Fetch historical price data for reference asset
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
            granularity='ONE_DAY',  # Daily data for regime detection
            cache_only=args.cache_only
        )
    except Exception as e:
        logger.error(f"Failed to fetch price data: {e}")
        sys.exit(1)

    reference_prices = price_data[args.reference_asset]
    logger.info(f"Loaded {len(reference_prices)} data points")

    # Initialize all detectors
    detectors = get_all_detectors(window_days=args.window_days)

    # For now, simulate ground truth by simple heuristics
    # TODO: Replace with actual rolling window optimization results
    logger.info("\nGenerating rolling windows and detecting regimes...")

    # Generate windows
    current_date = start_date
    windows = []

    while current_date + timedelta(days=args.window_days) <= end_date:
        window_end = current_date + timedelta(days=args.window_days)

        # Get prices for this window (reference_prices is list of (timestamp, price) tuples)
        window_prices = [
            p for p in reference_prices
            if current_date <= p[0] <= window_end  # p[0] is timestamp
        ]

        if len(window_prices) >= args.window_days:
            windows.append((current_date, window_end, window_prices))

        current_date += timedelta(days=args.step_days)

    logger.info(f"Generated {len(windows)} windows")

    # For each window, determine ground truth and run all detectors
    detector_results = {name: [] for name in detectors.keys()}

    print(f"\nAnalyzing {len(windows)} windows...")

    for i, (window_start, window_end, window_prices) in enumerate(windows, 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(windows)} windows analyzed...")

        # Extract prices and timestamps from tuples
        prices = [p[1] for p in window_prices]  # p is (timestamp, price)
        timestamps = [p[0] for p in window_prices]

        # Determine ground truth based on simple heuristics
        # (In real implementation, this would come from rolling optimization results)
        return_30d = ((prices[-1] / prices[0]) - 1) * 100

        if return_30d > 15:
            ground_truth = MarketRegime.BULL  # Equal_Weight would win
        elif return_30d < -10:
            ground_truth = MarketRegime.BEAR  # Stablecoin_Heavy would win
        else:
            ground_truth = MarketRegime.NEUTRAL  # Top_Three would win

        # Run all detectors
        for name, detector in detectors.items():
            detection = detector.detect(prices, timestamps)
            detector_results[name].append((detection, ground_truth))

    print("  Complete!\n")

    # Calculate performance for each detector
    performances = []
    for name, results in detector_results.items():
        perf = calculate_detector_performance(results, name)
        performances.append(perf)

    # Print comparison
    print_detector_comparison(performances)

    # Print winner
    winner = max(performances, key=lambda p: p.accuracy)
    print(f"\n{'='*100}")
    print(f"🏆 WINNER: {winner.detector_name} with {winner.accuracy:.2f}% accuracy")
    print(f"{'='*100}\n")


if __name__ == '__main__':
    main()
