# Coinbase Portfolio Rebalancing Bot

An automated cryptocurrency portfolio rebalancing bot that uses the Coinbase Advanced Trade API to maintain your desired asset allocation over time.

## Features

- Automatic portfolio rebalancing at configurable intervals
- **Capital-efficient trade routing** - Uses direct trading pairs (BTC-ETH) to minimize fees
- JSON-based portfolio configuration (hot-reloadable)
- Complete transaction history logging with **fees and slippage tracking**
- Dry-run mode for safe testing
- Configurable rebalancing thresholds
- Support for multiple cryptocurrencies
- Scheduled execution with APScheduler
- Comprehensive error handling and logging
- **Advanced reporting and analytics** (fee analysis, slippage tracking, performance metrics)
- CSV export for custom analysis
- **Intelligent trade optimization** - Automatically matches buyers/sellers for direct swaps

## Project Structure

```
rebalance-bot/
├── config/
│   ├── portfolio.json          # Target portfolio allocation
│   ├── .env.example           # Example environment variables
│   └── .env                   # Your API credentials (create this)
├── src/
│   ├── __init__.py
│   ├── main.py                # Entry point with scheduler
│   ├── coinbase_client.py     # Coinbase API wrapper
│   ├── portfolio_manager.py   # Rebalancing logic
│   ├── trade_optimizer.py     # Capital-efficient routing (NEW)
│   ├── transaction_logger.py  # Transaction record keeper
│   ├── reporting.py           # Analysis and reporting tools
│   └── utils.py               # Helper functions
├── data/
│   ├── transactions.json      # Transaction history (with fees & slippage)
│   └── transactions.csv       # Exported CSV for analysis
├── logs/
│   └── rebalance_bot.log      # Application logs
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Coinbase account with Advanced Trade API access
- API keys from Coinbase Developer Platform

### Setup Steps

1. **Install Python dependencies:**

```bash
pip install -r requirements.txt
```

2. **Configure API credentials:**

Copy the example environment file and add your credentials:

```bash
cp config/.env.example config/.env
```

Edit `config/.env` and add your Coinbase API credentials:

```
COINBASE_API_KEY=your_api_key_here
COINBASE_API_SECRET=your_api_secret_here
```

To get API keys:
- Go to https://portal.cdp.coinbase.com/projects
- Create a new API key with trading permissions
- Save the key and secret securely

3. **Configure your portfolio:**

Copy the example configuration file:

```bash
cp config/portfolio.json.example config/portfolio.json
```

Edit `config/portfolio.json` and add your portfolio ID and desired allocation:

```json
{
  "portfolio_id": "your-portfolio-id-here",
  "target_allocation": {
    "BTC": 40.0,
    "ETH": 30.0,
    "SOL": 20.0,
    "USDC": 10.0
  },
  "rebalancing": {
    "threshold_percent": 5.0,
    "min_trade_value_usd": 10.0,
    "dry_run": true,
    "prefer_direct_routes": true,
    "handle_unknown_assets": "sell"
  },
  "schedule": {
    "interval_hours": 24,
    "enabled": true
  }
}
```

To get your portfolio ID:
- Go to Coinbase and navigate to your portfolio
- The portfolio ID is in the URL: `https://www.coinbase.com/portfolio/{PORTFOLIO_ID}`
- Or leave blank/null to use your default portfolio

**Configuration Options:**

- `portfolio_id`: Coinbase portfolio ID (optional - uses default portfolio if not specified)
- `target_allocation`: Desired percentage allocation for each asset (must sum to 100%)
- `threshold_percent`: Minimum deviation percentage to trigger rebalancing
- `min_trade_value_usd`: Minimum trade value in USD (prevents tiny trades)
- `dry_run`: If true, simulates trades without executing them
- `prefer_direct_routes`: Use direct trading pairs (BTC-ETH) instead of routing through USD (default: true)
- `handle_unknown_assets`: What to do with assets not in target allocation: `"sell"` (default) or `"ignore"`
- `interval_hours`: Hours between rebalancing attempts
- `enabled`: Enable/disable scheduled rebalancing

### Unknown Asset Handling

The bot can automatically handle assets in your portfolio that aren't part of your target allocation:

- **`"sell"`** (default): Automatically sells unknown assets and distributes proceeds according to your target allocation
  - Example: If you have LINK but it's not in your config, it will be sold and the value distributed 20% to BTC, 40% to ETH, etc.
- **`"ignore"`**: Leaves unknown assets untouched (not recommended - they'll throw off your allocation percentages)

**When would unknown assets appear?**
- Airdrops or rewards from Coinbase
- Assets you manually bought outside the bot
- Leftover positions from a previous allocation strategy
- Coinbase Earn rewards or staking yields

## Usage

### Run Modes

The bot supports three run modes:

#### 1. Status Mode (View Portfolio)

Check your current portfolio allocation without making any trades:

```bash
python -m src.main --mode status
```

This displays:
- Total portfolio value
- Current allocation percentages
- Target allocation percentages
- Deviation from target
- USD value per asset

#### 2. Once Mode (Single Rebalance)

Run rebalancing once and exit:

```bash
python -m src.main --mode once
```

Useful for:
- Testing the bot
- Manual rebalancing
- Running from cron or other schedulers

#### 3. Schedule Mode (Continuous Operation)

Start the scheduler for automatic periodic rebalancing:

```bash
python -m src.main --mode schedule
```

The bot will:
- Run continuously
- Execute rebalancing at configured intervals
- Reload configuration before each run
- Log all operations

Press Ctrl+C to stop gracefully.

### Custom Configuration File

Specify a different configuration file:

```bash
python -m src.main --config /path/to/config.json
```

## Safety Features

### Dry Run Mode

**IMPORTANT:** The bot starts in dry-run mode by default. This means it will:
- Calculate what trades would be executed
- Log all operations
- NOT execute actual trades

To enable live trading, set `"dry_run": false` in `config/portfolio.json`.

### Testing Recommendations

Before enabling live trading:

1. Run in dry-run mode several times
2. Verify the calculated trades make sense
3. Start with a small portfolio
4. Set higher `min_trade_value_usd` initially
5. Monitor the first few live runs closely

## Transaction Logging

All transactions are logged to `data/transactions.json` with comprehensive data for analysis:

```json
[
  {
    "timestamp": "2025-10-13T14:30:00",
    "asset": "BTC",
    "action": "BUY",
    "product_id": "BTC-USD",
    "size": null,
    "quote_size": 100.0,
    "value_usd": 100.0,
    "reason": "Increase from 35.00% to 40.00%",
    "order_id": "abc123",
    "success": true,
    "pre_trade_price": 67500.00,
    "filled_size": 0.00148148,
    "average_filled_price": 67520.00,
    "filled_value": 100.02,
    "total_fees": 0.60,
    "fee_currency": "USD",
    "number_of_fills": 1,
    "order_status": "FILLED",
    "slippage_percent": 0.03,
    "slippage_usd": 0.02,
    "fills": [
      {
        "trade_id": "xyz789",
        "price": 67520.00,
        "size": 0.00148148,
        "fee": 0.60,
        "trade_time": "2025-10-13T14:30:01"
      }
    ]
  }
]
```

**Captured Data:**
- Basic trade info (timestamp, asset, action, amounts)
- Order details (order ID, status, number of fills)
- Pricing data (pre-trade price, filled price, filled value)
- **Fees** (total fees, fee currency, per-fill fees)
- **Slippage** (percentage and USD impact)
- Individual fill details for partial fills

Rebalancing sessions are also logged with complete session metadata including portfolio values before/after.

The logger automatically creates backups every 10 transactions.

### Export to CSV

Export transaction data to CSV for analysis in Excel or other tools:

```bash
python -m src.reporting --export-csv
```

This creates `data/transactions.csv` with all trade data.

## Logging

Application logs are written to:
- `logs/rebalance_bot.log` (file)
- Console output (stdout)

Logs include:
- Rebalancing decisions
- Trade executions
- Errors and warnings
- Configuration changes

## How It Works

### Rebalancing Algorithm

The bot uses a **capital-efficient** rebalancing strategy to minimize trading fees and slippage:

1. **Fetch Current State:**
   - Get all account balances from Coinbase
   - Calculate current USD value of each asset
   - Calculate current allocation percentages

2. **Compare to Target:**
   - Load target allocation from configuration
   - Calculate deviation for each asset
   - Identify assets above/below threshold

3. **Optimize Trade Routes (NEW):**
   - Fetch available trading pairs from Coinbase
   - **Match sellers with buyers** for direct swaps (e.g., BTC→ETH instead of BTC→USD→ETH)
   - Calculate minimum number of trades needed
   - Route remaining trades through USD only when necessary
   - Log estimated fee savings

4. **Execute Trades:**
   - Execute direct pair trades first (most efficient)
   - Execute USD-routed trades for remaining amounts
   - Log all transactions with fee and slippage data
   - Rate limit between trades

5. **Verify Results:**
   - Fetch new balances
   - Calculate new allocation
   - Log session summary with optimization metrics

### Example Scenario

**Current Portfolio:**
- BTC: 35% ($3,500)
- ETH: 35% ($3,500)
- SOL: 20% ($2,000)
- USDC: 10% ($1,000)
- Total: $10,000

**Target Allocation:**
- BTC: 40%
- ETH: 30%
- SOL: 20%
- USDC: 10%

**Rebalancing Analysis (threshold: 5%):**
- BTC: -5% → Need to buy $500 worth
- ETH: +5% → Need to sell $500 worth
- SOL: 0% → No action
- USDC: 0% → No action

**Capital-Efficient Solution:**

**OLD APPROACH** (Naive):
1. Sell $500 ETH → USD (pays ~$3 fee)
2. Buy $500 BTC with USD (pays ~$3 fee)
- **Total fees: ~$6**
- **Total trades: 2**

**NEW APPROACH** (Optimized):
1. Direct swap: Sell $500 ETH → BTC using ETH-BTC pair (pays ~$3 fee)
- **Total fees: ~$3**
- **Total trades: 1**
- **Savings: 50% reduction in fees!**

The optimizer automatically detects that ETH-BTC trading pair exists and executes a single direct swap instead of two separate trades through USD.

## Troubleshooting

### Common Issues

**"API credentials not found"**
- Ensure `.env` file exists in `config/` directory
- Check that `COINBASE_API_KEY` and `COINBASE_API_SECRET` are set

**"Invalid configuration file"**
- Verify target allocation sums to 100%
- Check all percentages are positive
- Ensure JSON syntax is valid

**"Could not get price for [asset]"**
- Asset may not be tradeable on Coinbase
- Check asset symbol is correct (e.g., "BTC" not "BITCOIN")
- Verify your account has access to the asset

**Trades not executing**
- Check if `dry_run` is set to `true`
- Verify API keys have trading permissions
- Check minimum trade value requirements

### Rate Limiting

The bot implements rate limiting:
- 0.1 second delay between price fetches
- 1 second delay between trade executions

If you encounter rate limit errors, the bot will log them. Consider:
- Increasing delays in `coinbase_client.py`
- Reducing rebalancing frequency

## Security Best Practices

1. **API Keys:**
   - Never commit `.env` file to version control
   - Use read-only keys for testing
   - Rotate keys periodically
   - Limit API key permissions to what's needed

2. **Portfolio Configuration:**
   - Start with small amounts
   - Test thoroughly in dry-run mode
   - Set reasonable minimum trade values
   - Monitor transaction logs regularly

3. **Server Security:**
   - Run on a secure, private server
   - Keep dependencies updated
   - Use strong passwords
   - Enable firewall rules

## Advanced Configuration

### Multiple Portfolios

To manage multiple portfolios, create separate configuration files with different `portfolio_id` values:

**Example: Conservative Portfolio** (`config/portfolio_conservative.json`):
```json
{
  "portfolio_id": "portfolio-1-id-here",
  "target_allocation": {
    "BTC": 30.0,
    "ETH": 30.0,
    "USDC": 40.0
  },
  ...
}
```

**Example: Aggressive Portfolio** (`config/portfolio_aggressive.json`):
```json
{
  "portfolio_id": "portfolio-2-id-here",
  "target_allocation": {
    "BTC": 50.0,
    "ETH": 40.0,
    "SOL": 10.0
  },
  ...
}
```

Then run them separately:
```bash
python -m src.main --config config/portfolio_conservative.json --mode once
python -m src.main --config config/portfolio_aggressive.json --mode once
```

### Custom Scheduling

For more complex scheduling needs, use cron instead of the built-in scheduler:

```cron
# Run every day at 2 AM
0 2 * * * cd /path/to/bot && python -m src.main --mode once
```

Set `"enabled": false` in the schedule configuration to disable the built-in scheduler.

## Transaction Analysis & Reporting

The bot includes a comprehensive reporting tool to analyze your trading performance.

### Generate Reports

Run a comprehensive report of your trading activity:

```bash
# Last 30 days (default)
python -m src.reporting

# Last 7 days
python -m src.reporting --days 7

# All time
python -m src.reporting --days 0
```

**Report includes:**
- Overall statistics (total trades, fees, volume)
- Fee analysis (total fees, average fee rate, fees per trade)
- Slippage analysis (average, min, max slippage, slippage costs)
- Rebalancing performance (sessions, success rate, portfolio changes)
- Breakdown by asset

### Asset-Specific Analysis

Get detailed trading history for a specific asset:

```bash
python -m src.reporting --asset BTC --days 30
```

**Shows:**
- Total buy/sell trades
- Total amount bought/sold
- Net position change
- Average buy/sell prices
- Total fees paid
- Net cash flow

### Export to CSV

Export all transaction data to CSV for custom analysis:

```bash
python -m src.reporting --export-csv
```

Creates `data/transactions.csv` with columns:
- timestamp, asset, action, product_id
- size, quote_size, value_usd
- pre_trade_price, average_filled_price
- filled_size, filled_value
- total_fees, fee_currency
- slippage_percent, slippage_usd
- number_of_fills, order_id, order_status
- reason, success

### Programmatic Access

You can also access transaction data programmatically:

```python
from src.transaction_logger import TransactionLogger
from src.reporting import TransactionReporter

# Get transaction logger
logger = TransactionLogger()

# Get summary statistics
stats = logger.get_summary_stats()
print(f"Total fees: ${stats['total_fees_usd']:.2f}")

# Get recent transactions
recent = logger.get_transactions(limit=10)

# Generate reports
reporter = TransactionReporter()

# Fee analysis for last 30 days
fee_analysis = reporter.get_fee_analysis(days=30)

# Slippage analysis
slippage = reporter.get_slippage_analysis(days=30)

# Asset-specific summary
btc_summary = reporter.get_asset_trading_summary('BTC', days=30)
```

## Contributing

This bot is designed to be simple and extensible. Potential improvements:

- Add support for limit orders
- Implement more sophisticated rebalancing strategies
- Add email/SMS notifications
- Support for multiple exchanges
- Web dashboard for monitoring
- Backtesting capabilities

## Disclaimer

This software is for educational purposes only. Use at your own risk.

- Cryptocurrency trading involves substantial risk
- Past performance does not guarantee future results
- Always test with small amounts first
- The authors are not responsible for any financial losses
- Ensure you understand the tax implications of frequent trading

## License

MIT License - feel free to modify and use for your own purposes.

## Support

For issues or questions:
- Check the logs in `logs/rebalance_bot.log`
- Review transaction history in `data/transactions.json`
- Consult Coinbase Advanced Trade API documentation

## Version

Current version: 1.0.0
