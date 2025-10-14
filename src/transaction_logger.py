"""Transaction logging for rebalancing operations."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from shutil import copy2


class TransactionLogger:
    """Logs all rebalancing transactions to a JSON file."""

    def __init__(self, log_file: str = "data/transactions.json"):
        """Initialize transaction logger."""
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Create file if it doesn't exist
        if not self.log_file.exists():
            self._write_transactions([])

    def _read_transactions(self) -> List[Dict[str, Any]]:
        """Read all transactions from file."""
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error reading transactions: {e}")
            return []

    def _write_transactions(self, transactions: List[Dict[str, Any]]) -> None:
        """Write transactions to file."""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(transactions, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error writing transactions: {e}")
            raise

    def _create_backup(self) -> None:
        """Create a backup of the transaction file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.log_file.parent / f"transactions_backup_{timestamp}.json"
            copy2(self.log_file, backup_file)
            self.logger.info(f"Created backup: {backup_file}")
        except Exception as e:
            self.logger.warning(f"Could not create backup: {e}")

    def log_transaction(self, transaction: Dict[str, Any]) -> None:
        """
        Log a single transaction.

        Args:
            transaction: Dictionary containing transaction details
        """
        # Add timestamp if not present
        if 'timestamp' not in transaction:
            transaction['timestamp'] = datetime.now().isoformat()

        # Read existing transactions
        transactions = self._read_transactions()

        # Create backup every 10 transactions
        if len(transactions) % 10 == 0 and len(transactions) > 0:
            self._create_backup()

        # Append new transaction
        transactions.append(transaction)

        # Write back to file
        self._write_transactions(transactions)

        self.logger.info(f"Logged transaction: {transaction.get('order_id', 'N/A')}")

    def log_rebalance_session(self, session_data: Dict[str, Any]) -> None:
        """
        Log a complete rebalancing session with multiple transactions.

        Args:
            session_data: Dictionary containing session metadata and transactions
        """
        session_entry = {
            'type': 'rebalance_session',
            'timestamp': datetime.now().isoformat(),
            'session_id': session_data.get('session_id'),
            'dry_run': session_data.get('dry_run', False),
            'total_trades': len(session_data.get('trades', [])),
            'portfolio_value_before': session_data.get('portfolio_value_before'),
            'portfolio_value_after': session_data.get('portfolio_value_after'),
            'trades': session_data.get('trades', [])
        }

        transactions = self._read_transactions()

        if len(transactions) % 10 == 0 and len(transactions) > 0:
            self._create_backup()

        transactions.append(session_entry)
        self._write_transactions(transactions)

        self.logger.info(f"Logged rebalance session: {session_entry['session_id']}")

    def get_transactions(self, limit: int = None,
                        start_date: str = None,
                        end_date: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve transactions with optional filtering.

        Args:
            limit: Maximum number of transactions to return
            start_date: ISO format date string
            end_date: ISO format date string

        Returns:
            List of transaction dictionaries
        """
        transactions = self._read_transactions()

        # Filter by date range
        if start_date:
            transactions = [t for t in transactions
                          if t.get('timestamp', '') >= start_date]
        if end_date:
            transactions = [t for t in transactions
                          if t.get('timestamp', '') <= end_date]

        # Apply limit
        if limit:
            transactions = transactions[-limit:]

        return transactions

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of all transactions.

        Returns:
            Dictionary with summary statistics including fees and slippage
        """
        transactions = self._read_transactions()

        total_transactions = len(transactions)
        rebalance_sessions = [t for t in transactions if t.get('type') == 'rebalance_session']
        individual_trades = [t for t in transactions if t.get('type') != 'rebalance_session']

        # Calculate total fees
        total_fees = sum(t.get('total_fees', 0) for t in individual_trades)

        # Calculate total trade value
        total_trade_value = sum(t.get('filled_value', 0) for t in individual_trades)

        # Calculate average slippage
        slippages = [t.get('slippage_percent', 0) for t in individual_trades
                    if t.get('slippage_percent') is not None]
        avg_slippage = sum(slippages) / len(slippages) if slippages else 0

        # Count successful trades
        successful_trades = sum(1 for t in individual_trades if t.get('success', False))

        return {
            'total_transactions': total_transactions,
            'total_rebalance_sessions': len(rebalance_sessions),
            'individual_trades': len(individual_trades),
            'successful_trades': successful_trades,
            'total_fees_usd': total_fees,
            'total_trade_value_usd': total_trade_value,
            'average_slippage_percent': avg_slippage,
            'first_transaction': transactions[0].get('timestamp') if transactions else None,
            'last_transaction': transactions[-1].get('timestamp') if transactions else None,
        }

    def export_to_csv(self, output_file: str = "data/transactions.csv") -> None:
        """
        Export transactions to CSV format for analysis.

        Args:
            output_file: Path to output CSV file
        """
        import csv

        transactions = self._read_transactions()
        individual_trades = [t for t in transactions if t.get('type') != 'rebalance_session']

        if not individual_trades:
            self.logger.warning("No individual trades to export")
            return

        # Define CSV columns
        fieldnames = [
            'timestamp', 'asset', 'action', 'product_id', 'size', 'quote_size',
            'value_usd', 'pre_trade_price', 'average_filled_price', 'filled_size',
            'filled_value', 'total_fees', 'fee_currency', 'slippage_percent',
            'slippage_usd', 'number_of_fills', 'order_id', 'order_status',
            'reason', 'success'
        ]

        try:
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()

                for trade in individual_trades:
                    writer.writerow(trade)

            self.logger.info(f"Exported {len(individual_trades)} trades to {output_file}")

        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")
            raise
