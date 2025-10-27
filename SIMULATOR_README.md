# Portfolio Rebalancing Strategy Simulator

A Monte Carlo simulation system for backtesting and comparing cryptocurrency portfolio rebalancing strategies using historical Coinbase price data.

## Overview

This simulator helps you answer critical questions about your rebalancing strategy:

- **Should I rebalance?** Compare returns vs. buy-and-hold baseline
- **How often?** Test different time intervals (daily, weekly, monthly)
- **What threshold?** Optimize deviation triggers (±1%, ±2.5%, ±5%)
- **What's the cost?** Calculate total fees paid and their impact on returns
- **Is it worth it?** Measure if rebalancing benefits exceed trading costs

## Features

### Strategies

1. **Buy and Hold** - Never rebalances; serves as baseline for comparison
2. **Hybrid Strategy** - Rebalances at fixed intervals only if threshold exceeded
   - Time-based: Check every N days
   - Threshold-based: Only rebalance if deviation exceeds ±X%
   - Example: "Check weekly, rebalance if any asset is ±2.5% off target"

### Performance Metrics

- **Total Return**: Percentage gain/loss over simulation period
- **Annualized Return**: Return normalized to yearly basis
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Total Fees**: Cumulative trading fees paid
- **Rebalance Count**: Number of times portfolio was rebalanced

### Historical Data

- Fetches real price data from Coinbase Advanced Trade API
- Supports multiple granularities: daily, hourly, etc.
- Handles any assets tradeable on Coinbase
- Automatically aligns data across multiple assets

## Quick Start

### Basic Usage

```bash
# Simulate last 90 days with current portfolio config (default: 7-day intervals)
python -m src.simulate_strategies --days 90
```

This will:
1. Load your portfolio configuration from `config/portfolio.json`
2. Fetch 90 days of historical prices from Coinbase
3. Run simulations for both strategies (buy-and-hold vs hybrid)
4. Display comprehensive comparison report

### For High-Frequency Bots (5-minute intervals)

If your bot runs every 5 minutes like yours, test with:

```bash
# Test your 5-minute strategy over last 2 days with hourly data
python -m src.simulate_strategies \
  --days 2 \
  --interval-minutes 5 \
  --threshold 2.5 \
  --granularity ONE_HOUR

# Test last week with 5-minute intervals (requires hourly data)
python -m src.simulate_strategies \
  --days 7 \
  --interval-minutes 5 \
  --threshold 2.5 \
  --granularity ONE_HOUR
```

**Note**: Coinbase API limits mean fetching long periods with hourly granularity may be slow. Start with shorter periods (2-7 days) for 5-minute intervals.

### Example Output

```
================================================================================
REBALANCING STRATEGY SIMULATION RESULTS
================================================================================

Buy and Hold
--------------------------------------------------------------------------------
Period: 2024-07-01 to 2024-10-01
Initial Value: $10,000.00
Final Value: $11,234.56
Total Return: +12.35% ⭐ BEST
Annualized Return: +52.14%
Sharpe Ratio: 1.845
Max Drawdown: 8.34%
Total Fees Paid: $0.00 ⭐ LOWEST
Number of Rebalances: 0
Total Trades: 0

Hybrid (every 7d, ±2.5%)
--------------------------------------------------------------------------------
Period: 2024-07-01 to 2024-10-01
Initial Value: $10,000.00
Final Value: $11,156.78
Total Return: +11.57%
Annualized Return: +48.93%
Sharpe Ratio: 1.923 ⭐ BEST
Max Drawdown: 6.12% ⭐ BEST
Total Fees Paid: $78.56
Number of Rebalances: 4
Total Trades: 32
Avg Fee per Rebalance: $19.64

================================================================================
KEY INSIGHTS
================================================================================
✗ Hybrid strategy UNDERPERFORMED by 0.78%
✓ Hybrid had better risk-adjusted returns (Sharpe: 1.923 vs 1.845)
✓ Hybrid reduced max drawdown from 8.34% to 6.12%
```

## Advanced Usage

### Custom Time Period

```bash
# Specific date range
python -m src.simulate_strategies --start 2024-01-01 --end 2024-06-30
```

### Test Different Parameters

The simulator now supports minute, hour, and day-level intervals:

```bash
# MINUTE-LEVEL (for high-frequency rebalancing like 5-minute bots)
python -m src.simulate_strategies --days 2 --interval-minutes 5 --threshold 2.5 --granularity ONE_HOUR

# HOURLY INTERVALS (intra-day rebalancing)
python -m src.simulate_strategies --days 7 --interval-hours 1 --threshold 2.5 --granularity ONE_HOUR   # Hourly
python -m src.simulate_strategies --days 7 --interval-hours 6 --threshold 2.5 --granularity ONE_HOUR   # Every 6 hours

# DAILY INTERVALS (traditional rebalancing)
python -m src.simulate_strategies --days 90 --interval-days 1 --threshold 1.0   # Daily with tight threshold
python -m src.simulate_strategies --days 180 --interval-days 7 --threshold 2.5  # Weekly (common default)
python -m src.simulate_strategies --days 365 --interval-days 30 --threshold 5.0 # Monthly with loose threshold

# Legacy syntax (interval treated as days)
python -m src.simulate_strategies --days 180 --interval 7 --threshold 2.5  # Same as --interval-days 7
```

**Important**: When using minute or hourly intervals, always specify `--granularity ONE_HOUR` to fetch hourly price data, otherwise you'll be limited to daily granularity.

### Custom Configuration

```bash
# Use different portfolio allocation
python -m src.simulate_strategies --config my_test_config.json --days 90

# Adjust initial capital
python -m src.simulate_strategies --days 90 --initial-capital 50000

# Test different fee rates (0.5% instead of 0.6%)
python -m src.simulate_strategies --days 90 --fee-rate 0.005
```

### Save Results

```bash
# Export to JSON for further analysis
python -m src.simulate_strategies --days 90 --output simulation_results.json
```

## Data Caching & Long Time Periods

The simulator includes built-in disk caching to speed up repeated runs and enable long time periods like a full year of hourly data.

### How Caching Works

- **Automatic**: Cache is enabled by default and stores price data in `data/price_cache/`
- **Persistent**: Cache survives across runs - fetch once, reuse many times
- **Smart**: Cache keys include asset, date range, and granularity
- **Expiration**: Cache expires after 7 days by default (configurable)

### Fetching Long Time Periods

For long simulations (e.g., 1 year of hourly data), the simulator automatically:
1. **Batches API requests**: Splits into chunks of 300 candles per request
2. **Shows progress**: Displays progress bars for each asset
3. **Rate limits**: Adds delays to avoid hitting Coinbase API limits
4. **Caches results**: Saves to disk so subsequent runs are instant

**Example: Full year of hourly data**

```bash
# First run: fetches ~30 API requests per asset, takes 3-4 minutes
python -m src.simulate_strategies \
  --days 365 \
  --interval-hours 6 \
  --threshold 2.5 \
  --granularity ONE_HOUR

# Subsequent runs: loads from cache, completes in seconds
python -m src.simulate_strategies \
  --days 365 \
  --interval-hours 12 \
  --threshold 3.0 \
  --granularity ONE_HOUR
```

### Cache Management Options

```bash
# Skip cache and force fresh fetch
python -m src.simulate_strategies --days 90 --no-cache

# Only use cached data (fail if not available)
python -m src.simulate_strategies --days 90 --cache-only

# Use older cache (default: 7 days)
python -m src.simulate_strategies --days 90 --cache-max-age 30

# Adjust API request delay (default: 0.3s)
python -m src.simulate_strategies --days 365 --request-delay 0.5 --granularity ONE_HOUR
```

### Cache Storage

Cache files are stored in `data/price_cache/` with descriptive names:

```
data/price_cache/
├── BTC_2024-01-01_2024-12-31_ONE_HOUR.json
├── ETH_2024-01-01_2024-12-31_ONE_HOUR.json
└── SOL_2024-01-01_2024-12-31_ONE_HOUR.json
```

Each file contains:
- Asset and time period metadata
- All price data (timestamp + price pairs)
- Cache creation timestamp
- Number of candles

### Clearing Cache

To manually clear cache:

```python
from src.price_cache import PriceCache

cache = PriceCache()

# Clear all cache
cache.clear()

# Clear specific asset
cache.clear(asset="BTC")

# Clear old cache (older than 30 days)
cache.clear(older_than_days=30)

# View cache statistics
stats = cache.get_cache_stats()
print(f"Cache: {stats['num_files']} files, {stats['total_size_mb']:.2f} MB")

# List all cached files
for entry in cache.list_cache():
    print(f"{entry['asset']}: {entry['num_candles']} candles, {entry['age_days']} days old")
```

### Performance Comparison

| Time Period | Granularity | API Requests | First Run | Cached Run |
|-------------|-------------|--------------|-----------|------------|
| 30 days | ONE_DAY | 1 per asset | ~1 sec | <1 sec |
| 30 days | ONE_HOUR | 3 per asset | ~3 sec | <1 sec |
| 365 days | ONE_DAY | 2 per asset | ~2 sec | <1 sec |
| 365 days | ONE_HOUR | ~30 per asset | ~3-4 min | ~2 sec |

**Tip**: When running parameter optimization over 1 year of hourly data, data is fetched once before optimization starts, then reused for all parameter combinations.

## Programmatic Usage

You can also use the simulator directly in Python code:

```python
from datetime import datetime, timedelta
from src.coinbase_client import CoinbaseClient
from src.monte_carlo_simulator import (
    SimulationConfig,
    HistoricalPriceFetcher,
    PortfolioSimulator,
    BuyAndHoldStrategy,
    HybridStrategy,
)

# Initialize
client = CoinbaseClient()
fetcher = HistoricalPriceFetcher(client)

# Fetch data
price_data = fetcher.fetch_historical_prices(
    assets=['BTC', 'ETH'],
    start_date=datetime(2024, 1, 1),
    end_date=datetime.now(),
    granularity="ONE_DAY"
)

# Configure simulation
config = SimulationConfig(
    start_date=datetime(2024, 1, 1),
    end_date=datetime.now(),
    initial_capital_usd=10000.0,
    target_allocation={'BTC': 50.0, 'ETH': 50.0},
    fee_rate=0.006
)

# Run simulation
strategy = HybridStrategy(rebalance_interval_days=7, threshold_percent=2.5)
simulator = PortfolioSimulator(config, strategy, price_data)
result = simulator.run()

# Analyze results
print(f"Total Return: {result.total_return_percent:.2f}%")
print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
print(f"Total Fees: ${result.total_fees_paid:.2f}")
```

See `examples/run_simulation_example.py` for a complete working example.

## Understanding the Results

### When Rebalancing Helps

Rebalancing typically benefits portfolios when:

1. **Assets have similar long-term returns** but different volatility
2. **Mean reversion** - assets that deviate tend to return to average
3. **Uncorrelated assets** - diversification benefits are maintained
4. **Volatile markets** - more opportunities to buy low, sell high

### When Buy-and-Hold Wins

Buy-and-hold often outperforms when:

1. **Strong trends** - one asset consistently outperforms others
2. **Low volatility** - prices move steadily without deviation
3. **High fees** - trading costs exceed rebalancing benefits
4. **Short time periods** - insufficient time for mean reversion

### Interpreting Sharpe Ratio

- **< 0**: Strategy lost money (negative returns)
- **0 to 1**: Positive returns but high volatility
- **1 to 2**: Good risk-adjusted returns
- **> 2**: Excellent risk-adjusted returns

### Maximum Drawdown

- Measures largest peak-to-trough decline
- Lower is better (indicates less downside risk)
- Important for understanding worst-case scenarios
- A strategy with lower drawdown may be preferable even with slightly lower returns

## Configuration Options

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--days` | int | Required* | Number of days to simulate (backward from today) |
| `--start` | date | - | Start date (YYYY-MM-DD), alternative to --days |
| `--end` | date | today | End date (YYYY-MM-DD) |
| `--interval-days` | int | - | Rebalance check interval in days (e.g., 7 for weekly) |
| `--interval-hours` | int | - | Rebalance check interval in hours (e.g., 6) |
| `--interval-minutes` | int | - | Rebalance check interval in minutes (e.g., 5) |
| `--interval` | int | 7 | Legacy: Rebalance interval in days (use --interval-days) |
| `--threshold` | float | 2.5 | Deviation threshold percentage |
| `--fee-rate` | float | 0.006 | Trading fee rate (0.006 = 0.6%) |
| `--config` | path | config/portfolio.json | Portfolio configuration file |
| `--initial-capital` | float | 10000.0 | Starting portfolio value (USD) |
| `--granularity` | string | ONE_DAY | Price data granularity (ONE_DAY, ONE_HOUR, SIX_HOUR) |
| `--check-interval-hours` | float | auto | How often to check for rebalancing (auto-set from interval) |
| `--output` | path | - | Save results to JSON file |
| `--quiet` | flag | false | Suppress detailed logging |

**Note**: Only one of `--interval-days`, `--interval-hours`, `--interval-minutes`, or `--interval` should be specified.

## Technical Details

### Architecture

- **HistoricalPriceFetcher** - Retrieves historical candle data from Coinbase API
- **SimulationConfig** - Defines simulation parameters and constraints
- **PortfolioState** - Tracks holdings, prices, and allocations at a point in time
- **RebalancingStrategy** - Abstract base class for strategy implementations
- **PortfolioSimulator** - Orchestrates simulation and calculates performance metrics

### Data Flow

1. Fetch historical prices for all portfolio assets
2. Initialize portfolio with target allocation
3. Step through time at configured intervals
4. At each step:
   - Update portfolio value based on current prices
   - Check if strategy wants to rebalance
   - If yes, calculate and execute trades
   - Deduct fees from portfolio
   - Record state and trades
5. Calculate final performance metrics

### Trade Execution

For simplicity, the simulator uses a **flat fee rate** applied to all trades, rather than modeling the complex direct pair routing used by the actual bot. This makes comparisons clearer and focuses on rebalancing strategy effectiveness.

### Limitations

- **Market orders only** - Does not simulate limit orders or slippage
- **Perfect fills** - Assumes orders execute at exact prices
- **No liquidity constraints** - Can trade any amount
- **Historical data only** - Cannot predict future performance
- **Simplified fees** - Single flat rate, not tiered or pair-specific

## Testing

Run the comprehensive test suite:

```bash
# All simulator tests
python -m unittest tests.test_monte_carlo_simulator -v

# Specific test class
python -m unittest tests.test_monte_carlo_simulator.TestHybridStrategy -v
```

## Use Cases

### 1. Optimize Rebalancing Frequency

Test different intervals to find the sweet spot:

```bash
# High-frequency (for bots running every few minutes)
python -m src.simulate_strategies --days 2 --interval-minutes 5 --threshold 2.5 --granularity ONE_HOUR
python -m src.simulate_strategies --days 2 --interval-minutes 15 --threshold 2.5 --granularity ONE_HOUR
python -m src.simulate_strategies --days 2 --interval-minutes 60 --threshold 2.5 --granularity ONE_HOUR

# Intra-day (for active management)
python -m src.simulate_strategies --days 30 --interval-hours 1 --threshold 2.5 --granularity ONE_HOUR
python -m src.simulate_strategies --days 30 --interval-hours 6 --threshold 2.5 --granularity ONE_HOUR
python -m src.simulate_strategies --days 30 --interval-hours 12 --threshold 2.5 --granularity ONE_HOUR

# Traditional (daily/weekly/monthly)
python -m src.simulate_strategies --days 365 --interval-days 1 --threshold 2.5  # Daily
python -m src.simulate_strategies --days 365 --interval-days 7 --threshold 2.5  # Weekly
python -m src.simulate_strategies --days 365 --interval-days 30 --threshold 2.5 # Monthly
```

### 2. Find Optimal Threshold

Determine how much deviation to tolerate:

```bash
python -m src.simulate_strategies --days 180 --interval 7 --threshold 1.0  # Tight
python -m src.simulate_strategies --days 180 --interval 7 --threshold 2.5  # Medium
python -m src.simulate_strategies --days 180 --interval 7 --threshold 5.0  # Loose
```

### 3. Validate Current Settings

Check if your current bot configuration is optimal:

```bash
# Use exact settings from your config/portfolio.json
python -m src.simulate_strategies --days 90
```

### 4. Test New Assets

Before adding a new asset to your portfolio, simulate historical performance:

```bash
# Create test config with new asset allocation
python -m src.simulate_strategies --config test_allocation.json --days 180
```

## Best Practices

1. **Test multiple time periods** - Results vary by market conditions
2. **Compare strategies** - Always benchmark against buy-and-hold
3. **Consider fees** - Higher rebalancing frequency = higher costs
4. **Check risk metrics** - Don't only focus on returns; consider drawdown and Sharpe
5. **Use realistic parameters** - Match your actual trading constraints
6. **Validate assumptions** - Past performance ≠ future results

## Troubleshooting

### "No price data retrieved for [ASSET]"

- Check asset symbol spelling (use ticker, e.g., "BTC" not "Bitcoin")
- Verify asset is tradeable on Coinbase Advanced Trade
- Ensure date range doesn't exceed available data

### Rate Limit Errors

- Add delays between API calls (implemented automatically)
- Reduce granularity or shorten time period
- Use cached data when re-running similar simulations

### Very High Sharpe Ratios

- Normal for consistent uptrends with low volatility
- Indicates excellent risk-adjusted returns
- Verify results make sense given price movements

## Parameter Optimization

The simulator includes a powerful grid search optimizer that tests all combinations of threshold and interval parameters to find the optimal settings for your specific market conditions and risk tolerance.

### Quick Start

```bash
# Optimize over threshold 0.5-2.5% and interval 5-60 minutes
python -m src.optimize_strategy \
  --days 7 \
  --threshold-min 0.5 --threshold-max 2.5 --threshold-step 0.1 \
  --interval-min 5 --interval-max 60 --interval-step 5 \
  --granularity ONE_HOUR \
  --output results.csv \
  --heatmap heatmaps/
```

This tests **21 thresholds × 12 intervals = 252 combinations** in parallel.

### Understanding the Output

The optimizer provides multiple views of the results:

**1. Optimization Summary**
- Total combinations tested
- Execution time and performance metrics
- Buy-and-hold baseline for comparison
- Best strategy for each metric (return, Sharpe, fees, etc.)

**2. Top 10 Tables**
- **By Total Return**: Which parameters maximized gains
- **By Sharpe Ratio**: Which had best risk-adjusted returns
- **By Net Return**: Which performed best after accounting for fees

**3. Baseline Comparison**
- Shows which strategies beat buy-and-hold
- Highlights outperformance in percentage points
- Helps identify if rebalancing added value

**4. CSV Export** (`results.csv`)
- All combinations with complete metrics
- Sort and filter in Excel or Python
- Columns: threshold, interval, returns, Sharpe, fees, etc.

**5. Heatmap Visualizations** (requires matplotlib)
- `return_heatmap.png` - Visual map of returns by parameter
- `sharpe_heatmap.png` - Risk-adjusted performance landscape
- `fees_heatmap.png` - Fee costs across parameter space
- `net_return_heatmap.png` - Returns after fees

### Advanced Usage

**Wider Parameter Ranges**
```bash
# Test thresholds from 1-10% in 1% increments
# Test intervals from 15 minutes to 6 hours
python -m src.optimize_strategy \
  --days 30 \
  --threshold-min 1.0 --threshold-max 10.0 --threshold-step 1.0 \
  --interval-min 15 --interval-max 360 --interval-step 15 \
  --granularity ONE_HOUR
```

**Control Parallel Processing**
```bash
# Use 8 CPU cores for faster optimization
python -m src.optimize_strategy \
  --days 14 \
  --threshold-min 0.5 --threshold-max 2.5 --threshold-step 0.1 \
  --interval-min 5 --interval-max 60 --interval-step 5 \
  --workers 8 \
  --granularity ONE_HOUR
```

**Suppress Progress Bar**
```bash
# Useful for running in scripts or cron jobs
python -m src.optimize_strategy \
  --days 7 \
  --threshold-min 0.5 --threshold-max 2.5 --threshold-step 0.1 \
  --interval-min 5 --interval-max 60 --interval-step 5 \
  --no-progress \
  --quiet
```

**Long Time Periods with Caching**
```bash
# Full year optimization (data fetched once, then cached)
# First run: ~3-4 minutes for data fetch, then optimization runs
python -m src.optimize_strategy \
  --days 365 \
  --threshold-min 1.0 --threshold-max 5.0 --threshold-step 0.5 \
  --interval-min 15 --interval-max 120 --interval-step 15 \
  --granularity ONE_HOUR \
  --output year_optimization.csv

# Subsequent runs with different parameters: uses cached data (instant)
python -m src.optimize_strategy \
  --days 365 \
  --threshold-min 0.5 --threshold-max 3.0 --threshold-step 0.25 \
  --interval-min 30 --interval-max 180 --interval-step 30 \
  --granularity ONE_HOUR \
  --cache-only  # Fail if cache not available (ensures we're using same data)
```

**Cache Management in Optimization**
```bash
# Force fresh data fetch (ignore cache)
python -m src.optimize_strategy --days 90 --no-cache ...

# Use cached data only (fail if not available)
python -m src.optimize_strategy --days 90 --cache-only ...

# Custom cache expiration (use cache up to 30 days old)
python -m src.optimize_strategy --days 90 --cache-max-age 30 ...
```

**Pro Tip**: When optimizing over long periods (1 year of hourly data), the data is fetched once at the start, cached, and then reused for all 200+ parameter combinations. This means the expensive API fetching only happens once, not once per combination.

### Interpreting Results

**Example Output:**

```
TOP 10 STRATEGIES BY TOTAL RETURN
================================================================================
Rank   Threshold    Interval     Total Return       Sharpe     Fees        Rebal.
       (%)          (min)                           Ratio      ($)         Count
--------------------------------------------------------------------------------
1      1.5%         30min        +5.23%             1.842      $12.45      8
2      1.0%         15min        +5.18%             1.756      $24.30      15
3      2.0%         45min        +5.12%             1.923      $8.20       5
...
```

**Key Insights:**
- **Rank 1**: 1.5% threshold, 30-minute intervals achieved highest return
- **Tradeoff**: Rank 2 had similar returns but higher fees (more frequent rebalancing)
- **Risk-adjusted**: Rank 3 had best Sharpe ratio despite slightly lower return

**Heatmap Interpretation:**
- **Red zones**: Poor performance for those parameters
- **Green zones**: Strong performance
- **Patterns**: Often see a "sweet spot" in the middle (not too tight, not too loose)

### Optimization Strategies

**1. Find Your Threshold Sweet Spot**

Your portfolio's optimal threshold depends on volatility:
- **High volatility assets** (DOGE, memecoins): May need wider thresholds (2-5%)
- **Stable assets** (BTC, ETH): Can use tighter thresholds (0.5-2%)
- **Mixed portfolios**: Test ranges spanning both (0.5-5%)

```bash
# For volatile portfolios
python -m src.optimize_strategy --days 30 \
  --threshold-min 2.0 --threshold-max 10.0 --threshold-step 1.0 \
  --interval-min 30 --interval-max 240 --interval-step 30 \
  --granularity ONE_HOUR
```

**2. Optimize for Your Time Commitment**

Match intervals to how often you can monitor:
- **High-frequency bot** (every 5-15 min): Test 5-60 minute intervals
- **Hourly monitoring**: Test 1-6 hour intervals
- **Daily checks**: Test 6-24 hour intervals

```bash
# For hourly monitoring
python -m src.optimize_strategy --days 14 \
  --threshold-min 1.0 --threshold-max 5.0 --threshold-step 0.5 \
  --interval-min 60 --interval-max 360 --interval-step 60 \
  --granularity ONE_HOUR
```

**3. Balance Returns vs. Fees**

Look at **Net Return** column to see real profitability:
- High-frequency strategies may have high gross returns but eat into profits with fees
- Lower frequency may have lower returns but keep more after fees

```bash
# Focus on net returns by examining fee impact
python -m src.optimize_strategy --days 30 \
  --threshold-min 0.5 --threshold-max 5.0 --threshold-step 0.5 \
  --interval-min 5 --interval-max 120 --interval-step 15 \
  --top-n 20  # Show top 20 to see fee patterns
```

### When Optimization Is Most Useful

**Best Use Cases:**
- **Initial setup**: Finding good starting parameters before going live
- **After market regime changes**: Re-optimize when volatility shifts
- **Performance review**: Check if current settings are still optimal
- **Multiple portfolios**: Different allocations may need different parameters

**Limitations:**
- **Past ≠ Future**: Optimal historical parameters may not work going forward
- **Market dependent**: What works in trending markets fails in ranging markets
- **Overfitting risk**: Don't over-optimize to specific historical periods

### Tips for Effective Optimization

1. **Test multiple time periods** - Don't rely on just one week's data
2. **Consider multiple metrics** - Don't only look at returns; check Sharpe and drawdown
3. **Start broad, then narrow** - Use wide ranges first, then zoom into promising areas
4. **Check baseline** - If no strategy beats buy-and-hold, maybe don't rebalance
5. **Run periodic re-optimizations** - Market conditions change over time

### Example Workflow

```bash
# Step 1: Broad initial scan (low resolution)
python -m src.optimize_strategy --days 30 \
  --threshold-min 0.5 --threshold-max 10.0 --threshold-step 1.0 \
  --interval-min 15 --interval-max 240 --interval-step 30 \
  --output scan_broad.csv

# Step 2: Narrow down based on results (higher resolution)
# Suppose best was around 2-3% threshold, 60-90 min interval
python -m src.optimize_strategy --days 30 \
  --threshold-min 1.5 --threshold-max 4.0 --threshold-step 0.25 \
  --interval-min 45 --interval-max 120 --interval-step 5 \
  --output scan_narrow.csv \
  --heatmap heatmaps/

# Step 3: Validate on different time period
python -m src.optimize_strategy --days 60 --start 2024-08-01 --end 2024-09-30 \
  --threshold-min 2.0 --threshold-max 3.0 --threshold-step 0.1 \
  --interval-min 60 --interval-max 90 --interval-step 5 \
  --output validation.csv
```

## Future Enhancements

Potential additions (not currently implemented):

- Monte Carlo price generation (random walks)
- More strategy types (threshold-only, calendar-based)
- Transaction cost optimization (direct pairs vs USD routing)
- Portfolio correlation analysis
- Multiple portfolio comparison
- Walk-forward optimization
- Out-of-sample validation
- Sensitivity analysis

## Support

For issues or questions:
1. Check existing tests in `tests/test_monte_carlo_simulator.py`
2. Review example in `examples/run_simulation_example.py`
3. Consult main documentation in `CLAUDE.md`

---

**Remember**: Past performance is not indicative of future results. Use simulations to understand strategies, not to predict outcomes. Always start with `dry_run: true` when applying optimized parameters to your live bot.
