"""
Statistical utilities for Monte Carlo simulations and risk analysis.

This module provides academically rigorous statistical methods including:
- Sample size calculations
- Convergence diagnostics
- Risk metrics (VaR, CVaR, Sortino, Calmar)
- Bootstrap confidence intervals

References:
    - Geweke, J. (1992). Evaluating the accuracy of sampling-based approaches 
      to calculating posterior moments. Bayesian Statistics, 4, 641-649.
    - Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. 
      Journal of the American Statistical Association, 89(428), 1303-1313.
    - Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional 
      value-at-risk. Journal of risk, 2, 21-42.
    - Sortino, F. A., & Price, L. N. (1994). Performance measurement in a 
      downside risk framework. The Journal of Investing, 3(3), 59-64.
"""

import logging
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Constants
DEFAULT_CONFIDENCE_LEVEL = 0.95
RISK_FREE_RATE = 0.0  # Assume 0% risk-free rate for crypto
TRADING_DAYS_PER_YEAR = 365.25  # Crypto trades 24/7


@dataclass
class ConvergenceDiagnostics:
    """Results from Monte Carlo convergence analysis."""
    converged: bool
    z_score: float
    recommended_min_sims: int
    first_half_mean: float
    second_half_mean: float
    std_error: float


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for portfolio analysis."""
    # Standard metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Value at Risk
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    cvar_95: float  # 95% CVaR (Expected Shortfall)
    cvar_99: float  # 99% CVaR
    
    # Drawdown metrics
    max_drawdown_pct: float
    avg_drawdown_pct: float
    
    # Volatility metrics
    total_volatility: float
    downside_volatility: float
    upside_volatility: float


def calculate_required_simulations(
    desired_ci_width: float = 0.05,  # 5% CI width
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    estimated_std: float = None,  # From pilot run
) -> int:
    """
    Calculate minimum simulations for reliable percentile estimates.
    
    Uses normal approximation for sample size calculation. For Monte Carlo
    simulations, this provides the number of runs needed to achieve a 
    desired confidence interval width for mean estimates.
    
    Args:
        desired_ci_width: Desired half-width of confidence interval (e.g., 0.05 for ±2.5%)
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        estimated_std: Estimated standard deviation from pilot run (if None, uses conservative 15%)
    
    Returns:
        Minimum number of simulations required
    
    Reference:
        Jordà, Ò., & Taylor, A. M. (2013). The time for austerity: estimating 
        the average treatment effect of fiscal policy. The Economic Journal, 
        123(566), F2-F27.
    
    Example:
        >>> n = calculate_required_simulations(desired_ci_width=0.05, estimated_std=0.12)
        >>> print(f"Need {n} simulations for ±5% CI with 12% volatility")
    """
    z = stats.norm.ppf((1 + confidence_level) / 2)
    
    if estimated_std is None:
        # Conservative estimate: assume normalized returns std = 15%
        estimated_std = 0.15
    
    # Calculate sample size: n = (z * σ / E)²
    # where E is the desired margin of error (half-width)
    n = (z * estimated_std / desired_ci_width) ** 2
    
    return int(np.ceil(n))


def check_convergence(
    returns: np.ndarray,
    split_point: Optional[float] = 0.5
) -> ConvergenceDiagnostics:
    """
    Check Monte Carlo convergence using Geweke diagnostic.
    
    Compares first and second portions of the simulation sequence to test
    if the distribution has stabilized. A z-score < 2 indicates convergence
    at the 95% confidence level.
    
    Args:
        returns: Array of simulation returns
        split_point: Where to split sequence (default: 0.5 for equal halves)
    
    Returns:
        ConvergenceDiagnostics object with convergence test results
    
    Reference:
        Geweke, J. (1992). Evaluating the accuracy of sampling-based 
        approaches to calculating posterior moments. In Bayesian Statistics 4 
        (pp. 641-649). Oxford University Press.
    
    Example:
        >>> results = [simulate() for _ in range(1000)]
        >>> returns = np.array([r.total_return_percent for r in results])
        >>> diag = check_convergence(returns)
        >>> if not diag.converged:
        >>>     print(f"Need {diag.recommended_min_sims} more simulations")
    """
    n = len(returns)
    split_idx = int(n * split_point)
    
    # Calculate means for first and second portions
    first_half = returns[:split_idx]
    second_half = returns[split_idx:]
    
    first_mean = np.mean(first_half)
    second_mean = np.mean(second_half)
    
    # Standard error for difference of means
    std_error = np.std(returns) / np.sqrt(n)
    
    # Geweke z-score
    z_score = abs(first_mean - second_mean) / (std_error * np.sqrt(2))
    
    # Converged if z < 2 (approximately 95% confidence)
    converged = z_score < 2.0
    
    # Estimate required samples if not converged
    # Scale by (z_observed / z_target)²
    if not converged:
        scale_factor = (z_score / 2.0) ** 2
        recommended_min_sims = int(n * scale_factor)
    else:
        recommended_min_sims = n
    
    return ConvergenceDiagnostics(
        converged=converged,
        z_score=z_score,
        recommended_min_sims=recommended_min_sims,
        first_half_mean=first_mean,
        second_half_mean=second_mean,
        std_error=std_error
    )


def calculate_var_cvar(
    returns: np.ndarray,
    confidence_levels: List[float] = [0.95, 0.99]
) -> Dict[str, float]:
    """
    Calculate Value at Risk (VaR) and Conditional VaR (CVaR/Expected Shortfall).
    
    VaR is the maximum expected loss at a given confidence level.
    CVaR is the expected loss given that VaR has been exceeded - it's
    more conservative and captures tail risk better than VaR alone.
    
    Args:
        returns: Array of portfolio returns (as percentages)
        confidence_levels: List of confidence levels to calculate (e.g., [0.95, 0.99])
    
    Returns:
        Dictionary with VaR and CVaR at each confidence level
    
    Reference:
        Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional 
        value-at-risk. Journal of risk, 2, 21-42.
    
    Example:
        >>> returns = np.array([2.5, -1.2, 3.1, -5.4, ...])
        >>> risk = calculate_var_cvar(returns, [0.95, 0.99])
        >>> print(f"95% VaR: {risk['var_95']:.2f}%")  # "At 95% confidence, worst loss is X%"
        >>> print(f"95% CVaR: {risk['cvar_95']:.2f}%")  # "If we exceed VaR, expected loss is Y%"
    """
    sorted_returns = np.sort(returns)
    n = len(sorted_returns)
    
    result = {}
    
    for conf_level in confidence_levels:
        # VaR: the return at the (1-confidence_level) percentile
        var_idx = int((1 - conf_level) * n)
        var = sorted_returns[var_idx] if var_idx < n else sorted_returns[-1]
        
        # CVaR: average of all returns worse than VaR
        cvar = sorted_returns[:var_idx].mean() if var_idx > 0 else var
        
        # Store with clear labels
        conf_pct = int(conf_level * 100)
        result[f'var_{conf_pct}'] = float(var)
        result[f'cvar_{conf_pct}'] = float(cvar)
    
    return result


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: float = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Calculate Sortino ratio - like Sharpe but only penalizes downside volatility.
    
    The Sortino ratio is a modification of the Sharpe ratio that only considers
    downside risk (negative returns) rather than total volatility. This is more
    appropriate for investors who care more about losses than gains.
    
    Args:
        returns: Array of portfolio returns (as percentages or decimals)
        risk_free_rate: Risk-free rate (default: 0 for crypto)
        periods_per_year: Annualization factor (default: 365.25 for crypto)
    
    Returns:
        Sortino ratio (annualized)
    
    Reference:
        Sortino, F. A., & Price, L. N. (1994). Performance measurement in a 
        downside risk framework. The Journal of Investing, 3(3), 59-64.
    
    Example:
        >>> daily_returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> sortino = calculate_sortino_ratio(daily_returns)
        >>> print(f"Sortino Ratio: {sortino:.2f}")
    """
    excess_returns = returns - risk_free_rate
    
    # Only consider downside deviations (negative excess returns)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        # No downside risk - return infinity (or a very large number)
        return float('inf')
    
    # Downside deviation (semi-deviation)
    downside_std = np.std(downside_returns)
    
    if downside_std == 0:
        return float('inf')
    
    # Annualize
    mean_return = np.mean(excess_returns)
    sortino = (mean_return * periods_per_year) / (downside_std * np.sqrt(periods_per_year))
    
    return float(sortino)


def calculate_calmar_ratio(
    total_return_pct: float,
    max_drawdown_pct: float,
    years: float
) -> float:
    """
    Calculate Calmar ratio = Annualized Return / Maximum Drawdown.
    
    The Calmar ratio measures return per unit of drawdown risk. Higher is better.
    A ratio > 1 indicates returns exceeded the maximum loss.
    
    Args:
        total_return_pct: Total return over period (as percentage, e.g., 25.5)
        max_drawdown_pct: Maximum drawdown over period (as positive percentage, e.g., 15.0)
        years: Duration of period in years
    
    Returns:
        Calmar ratio (dimensionless)
    
    Reference:
        Young, T. W. (1991). Calmar ratio: A smoother tool. Futures, 20(1), 40.
    
    Example:
        >>> calmar = calculate_calmar_ratio(total_return_pct=50.0, max_drawdown_pct=20.0, years=1.0)
        >>> print(f"Calmar Ratio: {calmar:.2f}")  # 50% / 20% = 2.5
    """
    if years <= 0:
        raise ValueError("years must be positive")
    
    if max_drawdown_pct <= 0:
        # No drawdown - return infinity
        return float('inf')
    
    # Annualize return: (1 + R)^(1/T) - 1
    annualized_return_pct = ((1 + total_return_pct / 100) ** (1 / years) - 1) * 100
    
    # Calmar = Annualized Return / Max Drawdown
    return float(annualized_return_pct / max_drawdown_pct)


def calculate_comprehensive_risk_metrics(
    returns: np.ndarray,
    portfolio_values: np.ndarray,
    years: float
) -> RiskMetrics:
    """
    Calculate comprehensive risk metrics for a simulation or backtest.
    
    Args:
        returns: Array of period returns (as percentages or decimals)
        portfolio_values: Array of portfolio values over time
        years: Duration in years
    
    Returns:
        RiskMetrics object with all calculated metrics
    
    Example:
        >>> metrics = calculate_comprehensive_risk_metrics(
        ...     returns=daily_returns,
        ...     portfolio_values=values,
        ...     years=1.0
        ... )
        >>> print(f"Sharpe: {metrics.sharpe_ratio:.2f}, Sortino: {metrics.sortino_ratio:.2f}")
    """
    # Calculate returns if not provided
    if len(returns) == 0:
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Standard Sharpe ratio
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe = (mean_return / std_return * np.sqrt(TRADING_DAYS_PER_YEAR)) if std_return > 0 else 0
    
    # Sortino ratio
    sortino = calculate_sortino_ratio(returns)
    
    # VaR and CVaR
    var_cvar = calculate_var_cvar(returns)
    
    # Drawdown analysis
    peak_value = portfolio_values[0]
    drawdowns = []
    
    for value in portfolio_values:
        if value > peak_value:
            peak_value = value
        drawdown_pct = ((peak_value - value) / peak_value) * 100
        drawdowns.append(drawdown_pct)
    
    max_drawdown = max(drawdowns) if drawdowns else 0
    avg_drawdown = np.mean(drawdowns) if drawdowns else 0
    
    # Calmar ratio
    total_return_pct = ((portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]) * 100
    calmar = calculate_calmar_ratio(total_return_pct, max_drawdown, years)
    
    # Volatility decomposition
    downside_returns = returns[returns < 0]
    upside_returns = returns[returns >= 0]
    
    downside_vol = np.std(downside_returns) * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside_returns) > 0 else 0
    upside_vol = np.std(upside_returns) * np.sqrt(TRADING_DAYS_PER_YEAR) if len(upside_returns) > 0 else 0
    total_vol = std_return * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return RiskMetrics(
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        calmar_ratio=float(calmar),
        var_95=float(var_cvar.get('var_95', 0)),
        var_99=float(var_cvar.get('var_99', 0)),
        cvar_95=float(var_cvar.get('cvar_95', 0)),
        cvar_99=float(var_cvar.get('cvar_99', 0)),
        max_drawdown_pct=float(max_drawdown),
        avg_drawdown_pct=float(avg_drawdown),
        total_volatility=float(total_vol),
        downside_volatility=float(downside_vol),
        upside_volatility=float(upside_vol)
    )


def parameter_stability_analysis(
    optimization_results: List[Dict]
) -> Dict[str, float]:
    """
    Analyze how optimal parameters vary across different conditions.
    
    Calculates stability metrics for optimized parameters. Low coefficient
    of variation (< 0.3) indicates stable, generalizable parameters.
    
    Args:
        optimization_results: List of dicts containing 'best_params' from walk-forward tests
    
    Returns:
        Dictionary with stability metrics for each parameter
    
    Example:
        >>> results = walk_forward_optimization(...)
        >>> stability = parameter_stability_analysis(results)
        >>> if stability['threshold_cv'] > 0.3:
        >>>     print("Warning: Threshold parameter is unstable across periods")
    """
    if not optimization_results:
        return {}
    
    # Extract parameter values across all optimization windows
    params_by_name = {}
    for result in optimization_results:
        for param_name, param_value in result.get('best_params', {}).items():
            if param_name not in params_by_name:
                params_by_name[param_name] = []
            params_by_name[param_name].append(param_value)
    
    # Calculate stability metrics for each parameter
    stability_metrics = {}
    
    for param_name, values in params_by_name.items():
        values_array = np.array(values)
        
        mean_val = np.mean(values_array)
        std_val = np.std(values_array)
        
        # Coefficient of variation (std/mean) - lower is more stable
        cv = std_val / mean_val if mean_val != 0 else float('inf')
        
        stability_metrics[param_name] = {
            'mean': float(mean_val),
            'std': float(std_val),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'range': float(np.max(values_array) - np.min(values_array)),
            'cv': float(cv),  # Key metric: < 0.3 is stable
            'stability_score': float(1.0 / (1.0 + cv))  # 0-1 scale, higher is better
        }
    
    return stability_metrics


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_func: callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate bootstrap confidence interval for any statistic.
    
    Uses percentile method for bootstrap CI. Useful for metrics where
    analytical CI formulas don't exist.
    
    Args:
        data: Original data sample
        statistic_func: Function to calculate statistic (e.g., np.mean, np.median)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95)
    
    Returns:
        Tuple of (point_estimate, (lower_bound, upper_bound))
    
    Reference:
        Efron, B., & Tibshirani, R. J. (1994). An introduction to the bootstrap.
        CRC press.
    
    Example:
        >>> returns = np.array([0.01, 0.02, -0.01, 0.03, ...])
        >>> sharpe, (lower, upper) = bootstrap_confidence_interval(
        ...     returns,
        ...     lambda x: np.mean(x) / np.std(x),
        ...     n_bootstrap=1000
        ... )
        >>> print(f"Sharpe: {sharpe:.2f}, 95% CI: [{lower:.2f}, {upper:.2f}]")
    """
    n = len(data)
    
    # Calculate point estimate on original data
    point_estimate = statistic_func(data)
    
    # Bootstrap resampling
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stat = statistic_func(bootstrap_sample)
        bootstrap_estimates.append(bootstrap_stat)
    
    # Calculate percentile CI
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_estimates, lower_percentile)
    upper_bound = np.percentile(bootstrap_estimates, upper_percentile)
    
    return float(point_estimate), (float(lower_bound), float(upper_bound))
