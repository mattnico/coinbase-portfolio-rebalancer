# Agent Guidelines for Rebalance Bot

## Test Commands
```bash
python -m unittest discover tests                    # Run all tests
python -m unittest tests.test_trade_optimizer        # Run single test file
python -m unittest tests.test_trade_optimizer -v     # Verbose output
python -m unittest tests.test_trade_optimizer.TestTradeOptimizer.test_direct_trade_single_pair  # Single test
```

## Code Style

**Imports**: Standard library → third-party → local modules. Use `from typing import Dict, List, Optional, Tuple` for type hints.

**Types**: Always use type hints for function parameters and returns. Example: `def get_accounts(self) -> Dict[str, float]:`

**Naming**: snake_case for functions/variables, PascalCase for classes. Prefix private methods with `_` (e.g., `_trading_pairs_cache`).

**Docstrings**: Required for all public methods. Use Google style with Args/Returns sections.

**Error Handling**: Use try/except blocks with logging. Always log errors with `self.logger.error()` before raising or returning.

**Config**: Use `load_json()` and `save_json()` from utils.py. Never hardcode paths - use Path from pathlib.

**Logging**: Use module-level logger: `self.logger = logging.getLogger(__name__)`. Never use print() statements.
