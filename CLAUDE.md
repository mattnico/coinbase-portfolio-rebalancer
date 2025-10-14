# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an automated cryptocurrency portfolio rebalancing bot that uses the Coinbase Advanced Trade API (CDP platform). The bot's primary goal is to maintain a target asset allocation over time while **minimizing trading costs through capital-efficient routing**.

The bot implements a sophisticated trade optimization algorithm that routes trades through direct trading pairs (e.g., BTC-ETH) instead of always going through USD, which cuts trading fees by approximately 50% on matched trades.

## Development Commands

### Running the Bot

```bash
# View current portfolio status (no trades)
python -m src.main --mode status

# Run rebalancing once and exit (respects dry_run setting)
python -m src.main --mode once

# Start continuous scheduled rebalancing
python -m src.main --mode schedule

# Use custom config file
python -m src.main --config /path/to/config.json
```

### Testing

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_trade_optimizer

# Run with verbose output
python -m unittest tests.test_trade_optimizer -v
```

### Reporting

```bash
# Generate comprehensive trading report (last 30 days)
python -m src.reporting

# Generate report for specific time period
python -m src.reporting --days 7

# View asset-specific trading summary
python -m src.reporting --asset BTC --days 30

# Export all transaction data to CSV
python -m src.reporting --export-csv
```

## Core Architecture

### Capital-Efficient Trade Routing (Critical Feature)

The bot's most important architectural feature is the **TradeOptimizer** system (`src/trade_optimizer.py`), which minimizes trading costs:

1. **Trade Matching Algorithm**: When rebalancing requires selling asset A and buying asset B, the optimizer first checks if a direct A-B trading pair exists on Coinbase
2. **Direct Pair Execution**: If A-B pair exists, executes a single direct swap (1 fee instead of 2)
3. **USD Fallback**: Only routes through USD (A→USD→B) when direct pairs don't exist
4. **Optimization Logging**: Logs estimated fee savings and trade count reduction

**Example**: Rebalancing from 35% ETH to 40% BTC
- **Naive approach**: Sell ETH→USD (fee #1), Buy USD→BTC (fee #2) = 2 fees
- **Optimized approach**: Direct swap ETH→BTC (fee #1) = 1 fee, **50% savings**

### Unknown Asset Handling

The bot automatically detects and handles assets in the portfolio that aren't part of the target allocation (e.g., airdrops, Coinbase Earn rewards):

- **Detection**: Identifies any assets not in `target_allocation` config
- **Action**: Automatically sells unknown assets (configurable via `handle_unknown_assets: "sell"` or `"ignore"`)
- **Distribution**: Proceeds from sold assets are distributed to target assets proportionally according to target allocation percentages

### Data Flow Through Components

1. **Entry Point** (`src/main.py`):
   - Handles CLI arguments and run modes (once/schedule/status)
   - Manages APScheduler for periodic execution
   - Instantiates `PortfolioManager`

2. **Portfolio Manager** (`src/portfolio_manager.py`):
   - **Core rebalancing logic**
   - Fetches current portfolio state via `CoinbaseClient`
   - Calculates deviations from target allocation
   - Identifies unknown assets and queues them for sale
   - Calls `TradeOptimizer.calculate_optimal_trades()` to find best routes
   - Executes trades and logs comprehensive transaction data
   - Validates configuration on each run (hot-reloadable)

3. **Trade Optimizer** (`src/trade_optimizer.py`):
   - **Accepts**: `to_sell` dict, `to_buy` dict, `available_pairs` from Coinbase
   - **Step 1**: Matches sellers with buyers for direct swaps using greedy algorithm
   - **Step 2**: Routes remaining amounts through USD
   - **Returns**: Optimized trade list with `is_direct` flags
   - Logs optimization summary with estimated savings

4. **Coinbase Client** (`src/coinbase_client.py`):
   - Wraps Coinbase Advanced Trade API (CDP format)
   - **Caches trading pairs** for 1 hour to minimize API calls
   - Provides helper methods: `get_available_trading_pairs()`, `is_trading_pair_available()`, `get_trading_pair_format()`
   - Handles both dict and object response formats from API
   - Extracts comprehensive order details including fees and fills

5. **Transaction Logger** (`src/transaction_logger.py`):
   - Persists all trades to `data/transactions.json`
   - Records: fees, slippage (%), fill details, pre-trade vs. filled prices
   - Creates automatic backups every 10 transactions
   - Provides summary statistics and filtering methods

6. **Reporting** (`src/reporting.py`):
   - Analyzes transaction history
   - Calculates: total fees paid, average slippage, fee rates, trading volume
   - Per-asset summaries: buy/sell totals, average prices, net cash flow
   - CSV export for external analysis

### Configuration System

**Location**: `config/portfolio.json` (hot-reloadable before each rebalancing run)

**Critical Settings**:
- `target_allocation`: Must sum to exactly 100%
- `threshold_percent`: Deviation percentage to trigger rebalancing (e.g., 2.5 means ±2.5% tolerance)
- `min_trade_value_usd`: Minimum trade size to prevent dust trades
- `dry_run`: **SAFETY**: Must be explicitly set to `false` to enable live trading
- `prefer_direct_routes`: Enable trade optimization (should always be `true`)
- `handle_unknown_assets`: `"sell"` (auto-liquidate) or `"ignore"` (leave untouched)

### API Authentication

**Location**: `config/.env`

The bot uses **CDP API keys** (not legacy Coinbase keys):
- Format: `organizations/{org_id}/apiKeys/{key_id}`
- Secret: Multi-line PEM format (EC private key)
- Get keys from: https://portal.cdp.coinbase.com/projects
- Keys must have **trading permissions** enabled

**Important**: `.env` loading is explicitly configured to `config/.env` directory via:
```python
env_path = Path(__file__).parent.parent / 'config' / '.env'
load_dotenv(env_path)
```

## Code Modification Guidelines

### Adding New Assets

1. Add to `target_allocation` in `config/portfolio.json`
2. Ensure percentages sum to 100%
3. Asset must be tradeable on Coinbase Advanced Trade
4. Use correct ticker symbol (e.g., "BTC" not "BITCOIN")

### Modifying Trade Logic

**DO NOT** bypass the `TradeOptimizer` - capital efficiency is a core feature. If modifying rebalancing logic:

1. Changes should go through `PortfolioManager.calculate_rebalancing_trades()`
2. Always pass trades through `TradeOptimizer.calculate_optimal_trades()`
3. Maintain the `is_direct` flag for logging/analytics
4. Test with `dry_run: true` first

### Transaction Logging Changes

If adding new fields to transaction logs:

1. Update `PortfolioManager.execute_trade()` to capture data
2. Add field to transaction dict passed to `TransactionLogger.log_transaction()`
3. Update `TransactionReporter.export_to_csv()` to include new column
4. Consider adding analysis method to `TransactionReporter`

### API Response Format Handling

The Coinbase API returns data in inconsistent formats (sometimes objects, sometimes dicts). When accessing API responses:

```python
# Use this pattern for defensive parsing
if hasattr(response, 'field_name'):
    value = response.field_name
elif isinstance(response, dict):
    value = response.get('field_name', default)
else:
    value = default
```

See `CoinbaseClient.get_accounts()` and `CoinbaseClient.get_order()` for examples.

## Testing Strategy

### Unit Tests
- `test_trade_optimizer.py`: Tests direct pair matching algorithm
- `test_unknown_assets.py`: Tests unknown asset detection and distribution
- `test_optimizer_standalone.py`: Standalone optimizer logic verification

### Integration Tests
- `test_integration.py`: Mocks Coinbase API responses and tests full rebalancing flow

**Test Data Strategy**: Tests use mocked portfolios with known deviations to verify:
1. Correct deviation calculations
2. Direct pair matching finds optimal routes
3. Fee savings are calculated correctly
4. Unknown assets are detected and handled
5. Proceeds distribution matches target allocation percentages

### Manual Testing

Always test with `dry_run: true` first:
```bash
python -m src.main --mode once
```

Review logs in `logs/rebalance_bot.log` for:
- Trade optimization summary
- Direct pair matches found
- Estimated fee savings
- Unknown asset warnings

## Data Persistence

- **Transactions**: `data/transactions.json` (includes fees, slippage, fills)
- **Backups**: `data/transactions_backup_*.json` (created every 10 transactions)
- **Logs**: `logs/rebalance_bot.log` (application logs)
- **CSV Export**: `data/transactions.csv` (generated on demand)

**Transaction Schema** includes:
- Basic: timestamp, asset, action, product_id, order_id, success
- Pricing: pre_trade_price, average_filled_price, filled_value
- Fees: total_fees, fee_currency, per-fill fees
- Slippage: slippage_percent, slippage_usd
- Fills: Individual fill array with trade_id, price, size, fee, trade_time

## Common Pitfalls

1. **Wrong API Keys**: Must use CDP format (`organizations/.../apiKeys/...`), not legacy keys
2. **Configuration Sum**: Target allocation must sum to exactly 100.0%
3. **Minimum Trade Values**: Set `min_trade_value_usd` appropriately for Coinbase minimums (typically $10+)
4. **Rate Limiting**: 0.1s delay between price fetches, 1s between trades (see `time.sleep()` calls)
5. **Direct Pair Availability**: Not all asset pairs have direct trading pairs; optimizer handles this automatically
6. **Unknown Assets**: Assets like USDC and USD are excluded from unknown asset detection
7. **Dry Run Default**: Bot starts in `dry_run: true` mode for safety - must explicitly disable

## Rate Limits and Performance

- **Trading Pairs Cache**: Refreshed every 1 hour to minimize API calls
- **Price Fetching**: 0.1s delay between requests
- **Trade Execution**: 1s delay between trades
- **Order Status**: 0.5s wait after placing order before fetching details

If encountering rate limit errors, increase delays in `src/coinbase_client.py` and `src/portfolio_manager.py`.
