# Monte Carlo & Sensitivity Analysis Implementation Roadmap

## Summary of Completed Improvements

### ✅ Completed (Committed)

1. **Statistical Utilities Module** (`src/statistical_utils.py`)
   - Sample size calculations with confidence interval specifications
   - Geweke convergence diagnostics for assessing Monte Carlo stability
   - Comprehensive risk metrics:
     - Value at Risk (VaR) at 95% and 99% confidence
     - Conditional VaR (CVaR/Expected Shortfall)
     - Sortino Ratio (downside risk-adjusted returns)
     - Calmar Ratio (return/max drawdown)
   - Parameter stability analysis for optimization robustness
   - Bootstrap confidence intervals for any statistic

2. **Advanced Bootstrap Methods** (`src/bootstrap_methods.py`)
   - Stationary Bootstrap (Politis & Romano, 1994) with geometric block lengths
   - Improved chunk shuffling with proper price normalization
   - Fixes for artificial path dependencies in original implementation
   - Better preservation of temporal dependencies

3. **Documentation & Citations**
   - All functions include peer-reviewed academic references
   - Mathematical specifications in docstrings
   - Usage examples and guidance

## Remaining High-Priority Items

### 🔴 Priority 1: Fee Structure & Realism

**File**: `src/fee_models.py` (new file)

**Implementation**:
```python
class RealisticFeeModel:
    """
    Tiered fee structure matching Coinbase Advanced Trade pricing.
    
    Includes:
    - Volume-based tier calculation (30-day rolling)
    - Maker/taker distinctions
    - Bid-ask spread modeling
    - Slippage for large orders
    
    Reference: Hasbrouck, J. (2009). Trading costs and returns.
    """
    
    def calculate_tiered_fee(self, trade_value_usd, monthly_volume):
        # Tiers: <$10k: 0.60%, $10k-$50k: 0.40%, etc.
        pass
    
    def apply_spread_cost(self, asset, value, liquidity_tier):
        # BTC/ETH: ~0.05-0.10%, Altcoins: ~0.20-0.50%
        pass
```

**Integration Points**:
- Update `monte_carlo_simulator.py` `HybridStrategy.calculate_trades()` to use tiered fees
- Add spread costs to trade execution
- Track 30-day volume for tier calculations

### 🔴 Priority 2: Walk-Forward Optimization

**File**: `src/walk_forward.py` (new file)

**Implementation**:
```python
class WalkForwardOptimizer:
    """
    Walk-forward analysis with train/test splits.
    
    Prevents overfitting by:
    1. Optimizing on in-sample window
    2. Testing on out-of-sample forward window
    3. Rolling forward through entire dataset
    
    Reference: Pardo, R. (2008). Evaluation and Optimization of Trading Strategies.
    """
    
    def optimize_and_test(self, 
                         optimization_window_days=90,
                         test_window_days=30,
                         step_days=30):
        # Implement sliding window logic
        pass
```

**Integration**:
- Add `--walk-forward` flag to `optimize_strategy.py`
- Report: in-sample vs out-of-sample performance comparison
- Generate stability metrics across windows

### 🔴 Priority 3: Regime Detection Enhancement

**File**: `src/hmm_regime_detector.py` (new file)

**Requirements**:
```
hmmlearn>=0.3.0
```

**Implementation**:
```python
class HMMRegimeDetector(RegimeDetector):
    """
    Hidden Markov Model for regime detection.
    
    Advantages over threshold-based:
    - Learns regimes from data automatically
    - Provides transition probabilities
    - Accounts for regime persistence
    
    Reference: Hamilton, J. D. (1989). Econometrica, 357-384.
    """
    
    def fit_detect(self, returns):
        # Fit 3-state Gaussian HMM
        # Map states to bull/bear/neutral based on mean returns
        pass
    
    def get_transition_matrix(self):
        # Return learned P[regime_t+1 | regime_t]
        pass
```

**Integration**:
- Update `adaptive_strategy.py` to support HMM detector
- Add `--regime-method hmm` CLI option
- Compare HMM vs threshold performance

### 🟡 Priority 4: Bayesian Optimization

**File**: `src/bayesian_optimizer.py` (new file)

**Requirements**:
```
bayesian-optimization>=1.4.0
```

**Implementation**:
```python
class BayesianStrategyOptimizer:
    """
    Efficient parameter search using Bayesian optimization.
    
    Advantages over grid search:
    - ~10-20x fewer evaluations needed
    - Automatic exploration-exploitation balance
    - Works well for expensive objectives (Monte Carlo sims)
    
    Reference: Snoek, J., et al. (2012). NeurIPS.
    """
    
    def optimize(self, n_iter=50):
        # Define objective (Sharpe ratio, Calmar, etc.)
        # Set parameter bounds
        # Run Bayesian optimization
        pass
```

**Integration**:
- Add to `optimize_strategy.py` as alternative method
- `--method bayesian` or `--method grid`

### 🟡 Priority 5: Code Quality Improvements

**Changes Across Multiple Files**:

1. **Constants Module** (`src/constants.py`):
```python
# Remove magic numbers
DEFAULT_FEE_RATE = 0.006  # 0.6% (Coinbase Advanced Trade taker fee)
TRADING_DAYS_PER_YEAR = 365.25  # Crypto trades 24/7
RISK_FREE_RATE = 0.0  # Assume 0% for crypto

COINBASE_FEE_TIERS = [
    (10_000, 0.006),
    (50_000, 0.004),
    (100_000, 0.0025),
    (1_000_000, 0.0015),
    (float('inf'), 0.001)
]
```

2. **Validation Enhancements**:
   - Add `SimulationConfig` validation for positive capital, valid dates, sensible fees
   - Type hints with `from __future__ import annotations` for Python 3.9+ compatibility
   - Input validation in all public functions

3. **Logging Improvements**:
   - Log extreme outcomes (returns > 3σ)
   - Convergence warnings
   - Data quality issues (missing prices, gaps)

## Integration Examples

### Using New Features Together

**Example 1: Rigorous Monte Carlo with Convergence Checks**
```python
from src.statistical_utils import calculate_required_simulations, check_convergence
from src.bootstrap_methods import StationaryBootstrap, BootstrapConfig

# Step 1: Determine sample size
pilot_sims = 100
pilot_results = run_pilot(pilot_sims)
estimated_std = np.std([r.total_return_percent for r in pilot_results])

required_n = calculate_required_simulations(
    desired_ci_width=0.05,  # ±2.5%
    estimated_std=estimated_std
)
print(f"Pilot std={estimated_std:.2f}%, need {required_n} sims for ±2.5% CI")

# Step 2: Run full Monte Carlo with stationary bootstrap
config = BootstrapConfig(method='stationary', avg_block_size=30, seed=42)
bootstrap = StationaryBootstrap(config)

results = []
for i in range(required_n):
    resampled_data = bootstrap.resample_returns(price_data)
    result = simulate(resampled_data)
    results.append(result)
    
    # Check convergence every 100 sims
    if (i + 1) % 100 == 0:
        returns = [r.total_return_percent for r in results]
        diag = check_convergence(returns)
        if not diag.converged:
            print(f"After {i+1} sims: Not converged (z={diag.z_score:.2f})")
        else:
            print(f"After {i+1} sims: Converged!")

# Step 3: Calculate comprehensive risk metrics
from src.statistical_utils import calculate_comprehensive_risk_metrics

portfolio_values = [r.final_value for r in results]
returns = [r.total_return_percent for r in results]
risk_metrics = calculate_comprehensive_risk_metrics(returns, portfolio_values, years=1.0)

print(f"Sharpe: {risk_metrics.sharpe_ratio:.2f}")
print(f"Sortino: {risk_metrics.sortino_ratio:.2f}")
print(f"95% VaR: {risk_metrics.var_95:.2f}%")
print(f"95% CVaR: {risk_metrics.cvar_95:.2f}%")
```

**Example 2: Walk-Forward with Stability Analysis**
```python
from src.walk_forward import WalkForwardOptimizer
from src.statistical_utils import parameter_stability_analysis

wf_optimizer = WalkForwardOptimizer(
    price_data=price_data,
    optimization_window_days=90,
    test_window_days=30,
    step_days=30
)

# Run walk-forward optimization
wf_results = wf_optimizer.run()

# Analyze parameter stability
stability = parameter_stability_analysis(wf_results)

for param_name, metrics in stability.items():
    print(f"{param_name}:")
    print(f"  Mean: {metrics['mean']:.2f}")
    print(f"  Std: {metrics['std']:.2f}")
    print(f"  CV: {metrics['cv']:.3f} {'✓ Stable' if metrics['cv'] < 0.3 else '✗ Unstable'}")
    print(f"  Stability Score: {metrics['stability_score']:.2f}/1.00")
```

## Testing Requirements

### Unit Tests to Add

**File**: `tests/test_statistical_utils.py`
```python
class TestStatisticalUtils(unittest.TestCase):
    def test_sample_size_calculation(self):
        n = calculate_required_simulations(desired_ci_width=0.05, estimated_std=0.15)
        assert n > 100, "Should need >100 sims for reasonable CI"
    
    def test_convergence_diagnostic(self):
        # Test with converged and non-converged sequences
        pass
    
    def test_var_cvar_calculation(self):
        # Test against known distributions
        pass
    
    def test_sortino_ratio(self):
        # Verify only downside risk is penalized
        pass
```

**File**: `tests/test_bootstrap_methods.py`
```python
class TestStationaryBootstrap(unittest.TestCase):
    def test_stationary_bootstrap_preserves_mean(self):
        # Resampled data should have similar mean to original
        pass
    
    def test_block_length_distribution(self):
        # Should follow geometric distribution
        pass
    
    def test_improved_chunk_normalization(self):
        # Verify no unrealistic price jumps
        pass
```

## Documentation Updates Needed

### 1. SIMULATOR_README.md Additions

Add new section:

```markdown
## Advanced Monte Carlo Features

### Stationary Bootstrap

The stationary bootstrap (Politis & Romano, 1994) improves upon fixed-chunk shuffling
by using geometrically distributed block lengths. This better preserves temporal
dependencies while still testing strategy robustness to different market sequences.

```bash
python -m src.run_monte_carlo --days 365 --simulations 1000 --bootstrap-method stationary
```

### Convergence Diagnostics

Monitor whether your Monte Carlo has run long enough:

```bash
python -m src.run_monte_carlo --days 365 --simulations 5000 --check-convergence
```

Output includes Geweke z-score and recommended minimum simulations.

### Walk-Forward Optimization

Prevent overfitting with train/test splits:

```bash
python -m src.optimize_strategy --days 365 --walk-forward \
    --opt-window 90 --test-window 30 --step 30
```

This reports both in-sample and out-of-sample performance for each parameter combination.
```

### 2. Mathematical Specifications Document

**File**: `MATHEMATICAL_SPECIFICATIONS.md`

```markdown
# Mathematical Specifications

## Risk Metrics

### Sharpe Ratio
$$\text{Sharpe} = \frac{\mathbb{E}[R_p - R_f]}{\sigma(R_p)} \times \sqrt{252}$$

Where:
- $R_p$ = Portfolio return
- $R_f$ = Risk-free rate (0% for crypto)
- $\sigma(R_p)$ = Standard deviation of returns
- $252$ = Annualization factor (365.25 for crypto)

### Sortino Ratio
$$\text{Sortino} = \frac{\mathbb{E}[R_p - R_f]}{\sigma_d(R_p)} \times \sqrt{252}$$

Where $\sigma_d$ is the downside semi-deviation (only negative returns).

### Value at Risk (VaR)
$$\text{VaR}_\alpha = \inf\{x : P(R \leq x) \geq 1 - \alpha\}$$

For $\alpha = 0.95$, VaR is the 5th percentile of the return distribution.

### Conditional VaR (Expected Shortfall)
$$\text{CVaR}_\alpha = \mathbb{E}[R | R \leq \text{VaR}_\alpha]$$

Average of all returns worse than VaR.

### Calmar Ratio
$$\text{Calmar} = \frac{R_{\text{annualized}}}{\text{MaxDrawdown}}$$

## Bootstrap Methods

### Stationary Bootstrap

Block start: $i_0 \sim \text{Uniform}(1, T)$

Block length: $L \sim \text{Geometric}(p)$ where $p = 1/\bar{L}$

Preserves stationarity while allowing variable block lengths.

**Reference**: Politis, D. N., & Romano, J. P. (1994). JASA, 89(428), 1303-1313.
```

## Priority Summary

| Priority | Task | Estimated Effort | Impact |
|----------|------|------------------|---------|
| 🔴 High | Fee model improvements | 2-3 hours | High - makes results realistic |
| 🔴 High | Walk-forward optimization | 3-4 hours | High - prevents overfitting |
| 🔴 High | Improved chunk normalization integration | 1-2 hours | High - fixes statistical validity |
| 🟡 Medium | HMM regime detection | 4-5 hours | Medium - better regime ID |
| 🟡 Medium | Bayesian optimization | 2-3 hours | Medium - efficiency gain |
| 🟡 Medium | Code quality improvements | 2-3 hours | Medium - maintainability |
| 🟢 Low | Comprehensive testing | 3-4 hours | Medium - confidence |
| 🟢 Low | Documentation updates | 2-3 hours | Low - usability |

**Total Estimated Effort**: 19-27 hours

## Next Steps

1. **Immediate** (next session):
   - Implement realistic fee model
   - Integrate stationary bootstrap into `run_monte_carlo.py`
   - Add convergence checks with auto-stop

2. **Short-term** (next few sessions):
   - Walk-forward optimization
   - HMM regime detection
   - Bayesian optimization

3. **Medium-term** (ongoing):
   - Code quality improvements
   - Comprehensive test suite
   - Documentation updates
   - Performance benchmarking

## References

All methods implemented are based on peer-reviewed academic literature:

1. **Politis & Romano (1994)** - Stationary bootstrap
2. **Geweke (1992)** - Convergence diagnostics
3. **Rockafellar & Uryasev (2000)** - CVaR optimization
4. **Hamilton (1989)** - Markov regime-switching models
5. **Snoek et al. (2012)** - Bayesian optimization
6. **Pardo (2008)** - Walk-forward analysis
7. **Hasbrouck (2009)** - Trading costs and market microstructure
