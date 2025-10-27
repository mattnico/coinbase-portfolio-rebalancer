#!/usr/bin/env python3
"""
Market regime detection methods for adaptive portfolio selection.

Implements 5 competing methods to detect bull/bear/neutral markets:
1. Volatility-based (standard deviation)
2. Trend-based (moving average slope)
3. Drawdown-based (distance from highs)
4. Return-based (cumulative returns)
5. Hybrid composite (weighted multi-factor)

Each method classifies current market as: BULL, BEAR, or NEUTRAL
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Literal
from enum import Enum
import numpy as np


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "BULL"      # Strong uptrend - use Equal_Weight
    BEAR = "BEAR"      # Downtrend/correction - use Stablecoin_Heavy
    NEUTRAL = "NEUTRAL"  # Choppy/sideways - use Top_Three


@dataclass
class RegimeDetection:
    """Result of regime detection at a point in time."""
    timestamp: datetime
    regime: MarketRegime
    confidence: float  # 0-1 scale
    raw_score: float   # Method-specific score
    method_name: str

    # Additional metrics for analysis
    volatility: float = 0.0
    trend_slope: float = 0.0
    drawdown_percent: float = 0.0
    return_30d: float = 0.0


class RegimeDetectorBase:
    """Base class for regime detection methods."""

    def __init__(self, window_days: int = 30):
        """
        Initialize detector.

        Args:
            window_days: Rolling window size for calculations
        """
        self.window_days = window_days
        self.name = self.__class__.__name__

    def detect(self, prices: List[float], timestamps: List[datetime]) -> RegimeDetection:
        """
        Detect market regime from price history.

        Args:
            prices: Historical prices (most recent last)
            timestamps: Corresponding timestamps

        Returns:
            RegimeDetection with classification and metrics
        """
        raise NotImplementedError

    def _calculate_returns(self, prices: List[float]) -> np.ndarray:
        """Calculate percentage returns from prices."""
        prices = np.array(prices)
        returns = np.diff(prices) / prices[:-1]
        return returns

    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility."""
        return np.std(returns) * np.sqrt(365) * 100  # Annualized %

    def _calculate_drawdown(self, prices: List[float]) -> float:
        """Calculate current drawdown from peak."""
        prices = np.array(prices)
        peak = np.max(prices)
        current = prices[-1]
        return ((current - peak) / peak) * 100  # Percent


class VolatilityDetector(RegimeDetectorBase):
    """
    Method 1: Volatility-based regime detection.

    Logic:
    - High volatility indicates uncertain/fearful markets → BEAR
    - Low volatility indicates calm, grinding uptrends → BULL
    - Medium volatility → NEUTRAL

    Thresholds (annualized volatility %):
    - < 40%: BULL (calm, stable uptrend)
    - > 80%: BEAR (high uncertainty, fear)
    - 40-80%: NEUTRAL (moderate chop)
    """

    def __init__(self, window_days: int = 30, bull_threshold: float = 40, bear_threshold: float = 80):
        super().__init__(window_days)
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold

    def detect(self, prices: List[float], timestamps: List[datetime]) -> RegimeDetection:
        returns = self._calculate_returns(prices)
        volatility = self._calculate_volatility(returns)

        # Classify based on volatility
        if volatility < self.bull_threshold:
            regime = MarketRegime.BULL
            confidence = 1 - (volatility / self.bull_threshold)  # Higher confidence at lower vol
        elif volatility > self.bear_threshold:
            regime = MarketRegime.BEAR
            confidence = min(1.0, (volatility - self.bear_threshold) / self.bear_threshold)
        else:
            regime = MarketRegime.NEUTRAL
            # Confidence decreases as we move away from thresholds
            dist_from_bull = abs(volatility - self.bull_threshold)
            dist_from_bear = abs(volatility - self.bear_threshold)
            confidence = 1 - (min(dist_from_bull, dist_from_bear) / (self.bear_threshold - self.bull_threshold))

        return RegimeDetection(
            timestamp=timestamps[-1],
            regime=regime,
            confidence=confidence,
            raw_score=volatility,
            method_name=self.name,
            volatility=volatility,
            drawdown_percent=self._calculate_drawdown(prices),
            return_30d=((prices[-1] / prices[0]) - 1) * 100
        )


class TrendDetector(RegimeDetectorBase):
    """
    Method 2: Trend/momentum-based regime detection.

    Logic:
    - Calculate linear regression slope over 30 days
    - Positive slope → BULL
    - Negative slope → BEAR
    - Flat slope → NEUTRAL

    Thresholds (% gain per day):
    - > +0.5%/day: BULL (strong uptrend)
    - < -0.5%/day: BEAR (downtrend)
    - -0.5% to +0.5%/day: NEUTRAL (sideways)
    """

    def __init__(self, window_days: int = 30, bull_threshold: float = 0.5, bear_threshold: float = -0.5):
        super().__init__(window_days)
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold

    def detect(self, prices: List[float], timestamps: List[datetime]) -> RegimeDetection:
        # Calculate linear regression slope
        prices_array = np.array(prices)
        x = np.arange(len(prices_array))

        # Fit linear regression
        coeffs = np.polyfit(x, prices_array, 1)
        slope = coeffs[0]

        # Convert slope to % per day
        avg_price = np.mean(prices_array)
        slope_pct_per_day = (slope / avg_price) * 100

        # Classify based on slope
        if slope_pct_per_day > self.bull_threshold:
            regime = MarketRegime.BULL
            confidence = min(1.0, slope_pct_per_day / (self.bull_threshold * 3))
        elif slope_pct_per_day < self.bear_threshold:
            regime = MarketRegime.BEAR
            confidence = min(1.0, abs(slope_pct_per_day) / (abs(self.bear_threshold) * 3))
        else:
            regime = MarketRegime.NEUTRAL
            # Confidence decreases as we move away from zero
            dist_from_zero = abs(slope_pct_per_day)
            confidence = 1 - (dist_from_zero / max(abs(self.bull_threshold), abs(self.bear_threshold)))

        returns = self._calculate_returns(prices)

        return RegimeDetection(
            timestamp=timestamps[-1],
            regime=regime,
            confidence=confidence,
            raw_score=slope_pct_per_day,
            method_name=self.name,
            volatility=self._calculate_volatility(returns),
            trend_slope=slope_pct_per_day,
            drawdown_percent=self._calculate_drawdown(prices),
            return_30d=((prices[-1] / prices[0]) - 1) * 100
        )


class DrawdownDetector(RegimeDetectorBase):
    """
    Method 3: Drawdown-based regime detection.

    Logic:
    - Near all-time highs → BULL (market making new highs)
    - Large drawdown → BEAR (correction/crash)
    - Medium drawdown → NEUTRAL (recovery/consolidation)

    Thresholds (drawdown from 30-day peak):
    - < 5%: BULL (near highs)
    - > 15%: BEAR (significant correction)
    - 5-15%: NEUTRAL (moderate pullback)
    """

    def __init__(self, window_days: int = 30, bull_threshold: float = -5, bear_threshold: float = -15):
        super().__init__(window_days)
        self.bull_threshold = bull_threshold  # e.g., -5% (near highs)
        self.bear_threshold = bear_threshold  # e.g., -15% (correction)

    def detect(self, prices: List[float], timestamps: List[datetime]) -> RegimeDetection:
        drawdown = self._calculate_drawdown(prices)

        # Classify based on drawdown
        if drawdown > self.bull_threshold:
            regime = MarketRegime.BULL
            confidence = 1 - (drawdown / self.bull_threshold)
        elif drawdown < self.bear_threshold:
            regime = MarketRegime.BEAR
            confidence = min(1.0, abs(drawdown - self.bear_threshold) / abs(self.bear_threshold))
        else:
            regime = MarketRegime.NEUTRAL
            # Confidence decreases as we move toward thresholds
            dist_from_bull = abs(drawdown - self.bull_threshold)
            dist_from_bear = abs(drawdown - self.bear_threshold)
            range_size = abs(self.bear_threshold - self.bull_threshold)
            confidence = 1 - (min(dist_from_bull, dist_from_bear) / range_size)

        returns = self._calculate_returns(prices)

        return RegimeDetection(
            timestamp=timestamps[-1],
            regime=regime,
            confidence=confidence,
            raw_score=drawdown,
            method_name=self.name,
            volatility=self._calculate_volatility(returns),
            drawdown_percent=drawdown,
            return_30d=((prices[-1] / prices[0]) - 1) * 100
        )


class ReturnDetector(RegimeDetectorBase):
    """
    Method 4: Simple return-based regime detection.

    Logic:
    - Strong positive returns → BULL
    - Negative returns → BEAR
    - Flat returns → NEUTRAL

    Thresholds (30-day cumulative return):
    - > +10%: BULL (strong gains)
    - < -10%: BEAR (losses)
    - -10% to +10%: NEUTRAL (flat/choppy)
    """

    def __init__(self, window_days: int = 30, bull_threshold: float = 10, bear_threshold: float = -10):
        super().__init__(window_days)
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold

    def detect(self, prices: List[float], timestamps: List[datetime]) -> RegimeDetection:
        # Calculate 30-day return
        return_30d = ((prices[-1] / prices[0]) - 1) * 100

        # Classify based on return
        if return_30d > self.bull_threshold:
            regime = MarketRegime.BULL
            confidence = min(1.0, return_30d / (self.bull_threshold * 3))
        elif return_30d < self.bear_threshold:
            regime = MarketRegime.BEAR
            confidence = min(1.0, abs(return_30d) / (abs(self.bear_threshold) * 3))
        else:
            regime = MarketRegime.NEUTRAL
            dist_from_zero = abs(return_30d)
            confidence = 1 - (dist_from_zero / max(abs(self.bull_threshold), abs(self.bear_threshold)))

        returns = self._calculate_returns(prices)

        return RegimeDetection(
            timestamp=timestamps[-1],
            regime=regime,
            confidence=confidence,
            raw_score=return_30d,
            method_name=self.name,
            volatility=self._calculate_volatility(returns),
            drawdown_percent=self._calculate_drawdown(prices),
            return_30d=return_30d
        )


class HybridDetector(RegimeDetectorBase):
    """
    Method 5: Hybrid composite detector combining multiple signals.

    Logic:
    - Combines returns, volatility, drawdown, and trend
    - Each factor contributes to a composite score: -100 (extreme bear) to +100 (extreme bull)
    - Weighted scoring allows combining strengths of all methods

    Weights (default):
    - Returns: 30% (what happened)
    - Trend: 30% (momentum/direction)
    - Drawdown: 25% (distance from highs)
    - Volatility: 15% (risk/uncertainty - inverse)

    Thresholds (composite score):
    - > +30: BULL
    - < -30: BEAR
    - -30 to +30: NEUTRAL
    """

    def __init__(
        self,
        window_days: int = 30,
        bull_threshold: float = 30,
        bear_threshold: float = -30,
        weights: Dict[str, float] = None
    ):
        super().__init__(window_days)
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.weights = weights or {
            'return': 0.30,
            'trend': 0.30,
            'drawdown': 0.25,
            'volatility': 0.15
        }

    def _normalize_score(self, value: float, low: float, mid: float, high: float) -> float:
        """
        Normalize value to -100 to +100 scale.

        Args:
            value: Raw value to normalize
            low: Value that maps to -100 (extreme bear)
            mid: Value that maps to 0 (neutral)
            high: Value that maps to +100 (extreme bull)
        """
        if value >= mid:
            # Map mid->high to 0->100
            if high == mid:
                return 0
            return min(100, ((value - mid) / (high - mid)) * 100)
        else:
            # Map low->mid to -100->0
            if mid == low:
                return 0
            return max(-100, ((value - mid) / (mid - low)) * 100)

    def detect(self, prices: List[float], timestamps: List[datetime]) -> RegimeDetection:
        returns = self._calculate_returns(prices)

        # Calculate all metrics
        return_30d = ((prices[-1] / prices[0]) - 1) * 100
        volatility = self._calculate_volatility(returns)
        drawdown = self._calculate_drawdown(prices)

        # Calculate trend slope
        prices_array = np.array(prices)
        x = np.arange(len(prices_array))
        coeffs = np.polyfit(x, prices_array, 1)
        slope = coeffs[0]
        avg_price = np.mean(prices_array)
        slope_pct_per_day = (slope / avg_price) * 100

        # Normalize each factor to -100 to +100 scale
        return_score = self._normalize_score(return_30d, -30, 0, 30)
        trend_score = self._normalize_score(slope_pct_per_day, -2, 0, 2)
        drawdown_score = self._normalize_score(drawdown, -30, -10, 0)
        volatility_score = self._normalize_score(volatility, 120, 60, 30)  # Lower vol = higher score

        # Calculate weighted composite score
        composite = (
            return_score * self.weights['return'] +
            trend_score * self.weights['trend'] +
            drawdown_score * self.weights['drawdown'] +
            volatility_score * self.weights['volatility']
        )

        # Classify based on composite score
        if composite > self.bull_threshold:
            regime = MarketRegime.BULL
            confidence = min(1.0, (composite - self.bull_threshold) / (100 - self.bull_threshold))
        elif composite < self.bear_threshold:
            regime = MarketRegime.BEAR
            confidence = min(1.0, (self.bear_threshold - composite) / (100 - abs(self.bear_threshold)))
        else:
            regime = MarketRegime.NEUTRAL
            dist_from_zero = abs(composite)
            confidence = 1 - (dist_from_zero / max(abs(self.bull_threshold), abs(self.bear_threshold)))

        return RegimeDetection(
            timestamp=timestamps[-1],
            regime=regime,
            confidence=confidence,
            raw_score=composite,
            method_name=self.name,
            volatility=volatility,
            trend_slope=slope_pct_per_day,
            drawdown_percent=drawdown,
            return_30d=return_30d
        )


def get_all_detectors(window_days: int = 30) -> Dict[str, RegimeDetectorBase]:
    """
    Get all regime detector implementations.

    Args:
        window_days: Rolling window size

    Returns:
        Dictionary mapping detector name to detector instance
    """
    return {
        'Volatility': VolatilityDetector(window_days),
        'Trend': TrendDetector(window_days),
        'Drawdown': DrawdownDetector(window_days),
        'Return': ReturnDetector(window_days),
        'Hybrid': HybridDetector(window_days)
    }
