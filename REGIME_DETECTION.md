# Market Regime Detection

This document explains the market regime detection system for adaptive portfolio selection.

## Overview

The regime detection system identifies whether the market is in a **BULL**, **BEAR**, or **NEUTRAL** state, enabling dynamic portfolio selection:

- **BULL** markets → Use **Equal_Weight** portfolio (maximizes gains in uptrends)
- **BEAR** markets → Use **Stablecoin_Heavy** portfolio (capital preservation)
- **NEUTRAL** markets → Use **Top_Three** portfolio (all-weather balanced)

This classification is based on rolling windows analysis that showed different portfolios excel in different market conditions.

## 5 Competing Detection Methods

### 1. Volatility-Based Detector (`VolatilityDetector`)

**Logic**: Market volatility indicates risk and uncertainty level.

- **High volatility (>80% annualized)** → BEAR (fearful, uncertain markets)
- **Low volatility (<40% annualized)** → BULL (calm, grinding uptrends)
- **Medium volatility (40-80%)** → NEUTRAL (moderate chop)

**Strengths**:
- Simple, robust calculation
- Works well in risk-on/risk-off regimes
- Volatility spikes clearly mark corrections

**Weaknesses**:
- Can lag trend changes
- May misclassify low-volatility downtrends as bull markets

### 2. Trend-Based Detector (`TrendDetector`)

**Logic**: Linear regression slope indicates momentum and direction.

- **Positive slope (>+0.5%/day)** → BULL (uptrend)
- **Negative slope (<-0.5%/day)** → BEAR (downtrend)
- **Flat slope (-0.5% to +0.5%/day)** → NEUTRAL (sideways)

**Strengths**:
- Captures momentum clearly
- Works well in trending markets
- Intuitive interpretation

**Weaknesses**:
- Can be noisy in choppy markets
- Linear fit may miss curve changes
- Sensitive to outliers

### 3. Drawdown-Based Detector (`DrawdownDetector`)

**Logic**: Distance from 30-day peak indicates market health.

- **Near highs (<5% drawdown)** → BULL (making new highs)
- **Large drawdown (>15%)** → BEAR (correction/crash)
- **Medium drawdown (5-15%)** → NEUTRAL (recovery/consolidation)

**Strengths**:
- Clear signal of market damage
- Good for identifying corrections early
- Aligns with trader psychology

**Weaknesses**:
- Always bullish at new highs (even if topping)
- Can be slow to recover from neutral to bull
- Doesn't consider volatility

### 4. Return-Based Detector (`ReturnDetector`)

**Logic**: Simple 30-day cumulative return.

- **Strong gains (>+10%)** → BULL
- **Losses (<-10%)** → BEAR
- **Flat (-10% to +10%)** → NEUTRAL

**Strengths**:
- Extremely simple and interpretable
- No complex calculations
- Fast to compute

**Weaknesses**:
- Backward-looking (always late)
- Ignores volatility and drawdown patterns
- Can oscillate rapidly in choppy markets

### 5. Hybrid Composite Detector (`HybridDetector`)

**Logic**: Weighted combination of all factors for robust classification.

**Scoring** (-100 to +100):
- Returns: 30% weight (what happened)
- Trend: 30% weight (momentum/direction)
- Drawdown: 25% weight (distance from highs)
- Volatility: 15% weight (risk/uncertainty - inverse)

- **Composite score >+30** → BULL
- **Composite score <-30** → BEAR
- **Composite score -30 to +30** → NEUTRAL

**Strengths**:
- Combines strengths of all methods
- More robust to false signals
- Configurable weights for different strategies

**Weaknesses**:
- More complex to understand
- Requires tuning weights
- Can be slower to react

## Usage

### Compare All Detectors

Test all 5 methods against historical data to see which performs best:

```bash
# Test detectors on last year of data
python -m src.compare_regime_detectors --days 365

# Test with custom windows
python -m src.compare_regime_detectors \\
    --days 730 \\
    --window-days 30 \\
    --step-days 7 \\
    --reference-asset BTC

# Use specific date range
python -m src.compare_regime_detectors \\
    --start 2024-01-01 \\
    --end 2024-12-31 \\
    --window-days 30
```

### Output Metrics

The comparison tool provides:

1. **Overall Accuracy**: % of windows correctly classified
2. **Regime-Specific Accuracy**: Performance for each regime type
3. **Confusion Matrix**: What did detector predict vs. ground truth
4. **Confidence Calibration**: Is detector confident when correct?
5. **Winner Declaration**: Best performing detector

### Using a Detector in Code

```python
from src.regime_detector import HybridDetector, MarketRegime

# Initialize detector
detector = HybridDetector(window_days=30)

# Detect regime from price history
prices = [45000, 46000, 47000, ...]  # Last 30 days of prices
timestamps = [datetime(...), ...]     # Corresponding timestamps

detection = detector.detect(prices, timestamps)

print(f"Regime: {detection.regime}")           # BULL, BEAR, or NEUTRAL
print(f"Confidence: {detection.confidence}")   # 0-1 scale
print(f"30-day return: {detection.return_30d}%")
print(f"Volatility: {detection.volatility}%")
```

### Customizing Thresholds

All detectors accept custom thresholds:

```python
# Conservative volatility detector (easier to trigger bear)
detector = VolatilityDetector(
    window_days=30,
    bull_threshold=35,   # Lower threshold for bull
    bear_threshold=70    # Lower threshold for bear
)

# Aggressive trend detector (requires stronger signals)
detector = TrendDetector(
    window_days=30,
    bull_threshold=1.0,   # Need +1%/day for bull
    bear_threshold=-1.0   # Need -1%/day for bear
)

# Custom hybrid weights
detector = HybridDetector(
    window_days=30,
    weights={
        'return': 0.40,      # Emphasize returns more
        'trend': 0.30,
        'drawdown': 0.20,
        'volatility': 0.10   # De-emphasize volatility
    }
)
```

## Ground Truth for Testing

The comparison tool uses a simplified ground truth based on 30-day returns:

- **>+15% return** → BULL (Equal_Weight would win)
- **<-10% return** → BEAR (Stablecoin_Heavy would win)
- **-10% to +15%** → NEUTRAL (Top_Three would win)

**Note**: For production use, ground truth should come from actual rolling window optimization results showing which portfolio performed best in each window.

## Interpreting Results

### What Makes a Good Detector?

1. **High Overall Accuracy**: >70% is good, >80% is excellent
2. **Balanced Regime Accuracy**: Should perform well across all regimes, not just one
3. **High Confidence When Correct**: Good calibration means confident predictions are accurate
4. **Low False Positives on BEAR**: Missing a bear market is costly

### Example Output Interpretation

```
OVERALL ACCURACY RANKING:
Rank   Detector             Accuracy    Correct    Total
1      Hybrid               78.26%      72         92
2      Drawdown             73.91%      68         92
3      Trend                69.57%      64         92
4      Volatility           65.22%      60         92
5      Return               60.87%      56         92
```

**Analysis**:
- Hybrid wins with 78% accuracy - combining signals is most robust
- Drawdown is strong second - distance from highs is powerful signal
- Return is worst - backward-looking nature hurts performance

### Regime-Specific Performance

```
Hybrid:
  BULL:    85.29% (29 correct)  ← Strong at identifying bull markets
  BEAR:    80.00% (16 correct)  ← Strong at identifying bear markets
  NEUTRAL: 70.83% (27 correct)  ← Decent at neutral classification
```

**Analysis**: This is a well-balanced detector. Some detectors may be 95%+ on one regime but 40% on others - that's less useful.

## Choosing a Detector

### For Conservative Risk Management
→ Use **DrawdownDetector** - Quick to identify corrections, errs on side of caution

### For Aggressive Growth
→ Use **TrendDetector** - Rides trends longer, maximizes bull market exposure

### For Balanced Approach
→ Use **HybridDetector** - Best overall accuracy, robust across conditions

### For Simplicity
→ Use **ReturnDetector** - Dead simple, easy to explain, "good enough"

## Implementation Roadmap

### Phase 1: Testing (Current)
- Compare all 5 detectors on historical data
- Identify best performer
- Tune thresholds and parameters

### Phase 2: Integration
- Add regime detection to `src/main.py`
- Select portfolio based on detected regime
- Log regime with each rebalancing run

### Phase 3: Validation
- Monitor live performance
- Compare detected regime vs. actual portfolio performance
- Iterate on detector choice/thresholds

### Phase 4: Advanced Features
- Ensemble methods (vote across multiple detectors)
- Regime transition detection (bull→bear is more important than static regime)
- Confidence-weighted allocation (blend portfolios based on uncertainty)

## Files

- `src/regime_detector.py` - All 5 detector implementations
- `src/compare_regime_detectors.py` - Head-to-head comparison tool
- `REGIME_DETECTION.md` - This documentation

## References

Based on rolling windows analysis showing:
- Equal_Weight won 22/92 windows (23.9%) - dominated bull periods
- Stablecoin_Heavy won 22/92 windows (23.9%) - dominated corrections
- Top_Three won 19/92 windows (20.7%) - consistent all-weather performer

Most common winning parameters across all windows:
- **Threshold**: 5.0% (tight rebalancing)
- **Interval**: 60 minutes (frequent checks)

This contradicts the single-period optimization (15.5% threshold, 360min interval), proving the importance of regime-aware adaptation.
