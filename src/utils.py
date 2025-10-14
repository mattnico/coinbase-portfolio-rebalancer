"""Utility functions for the rebalancing bot."""
import json
import logging
from pathlib import Path
from typing import Dict, Any


def setup_logging(log_file: str = "logs/rebalance_bot.log") -> logging.Logger:
    """Configure logging for the application."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON file and return as dictionary."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save dictionary to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def validate_portfolio_config(config: Dict[str, Any]) -> bool:
    """Validate portfolio configuration structure and values."""
    required_keys = ['target_allocation', 'rebalancing', 'schedule']

    if not all(key in config for key in required_keys):
        return False

    # Check allocation sums to 100%
    total_allocation = sum(config['target_allocation'].values())
    if not (99.9 <= total_allocation <= 100.1):  # Allow for floating point precision
        return False

    # Check all allocations are positive
    if any(v <= 0 for v in config['target_allocation'].values()):
        return False

    return True


def format_currency(amount: float, decimals: int = 2) -> str:
    """Format amount as currency string."""
    return f"${amount:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage string."""
    return f"{value:.{decimals}f}%"
