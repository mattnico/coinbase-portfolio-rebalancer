"""Reporting and analysis utilities for transaction data."""
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
from src.transaction_logger import TransactionLogger
from src.utils import format_currency, format_percentage


class TransactionReporter:
    """Generate reports and analysis from transaction history."""

    def __init__(self, transaction_logger: Optional[TransactionLogger] = None):
        """Initialize transaction reporter."""
        self.tx_logger = transaction_logger or TransactionLogger()
        self.logger = logging.getLogger(__name__)

    def get_fee_analysis(self, days: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze fees paid over a period.

        Args:
            days: Number of days to analyze (None for all time)

        Returns:
            Dictionary with fee analysis
        """
        start_date = None
        if days:
            start_date = (datetime.now() - timedelta(days=days)).isoformat()

        transactions = self.tx_logger.get_transactions(start_date=start_date)
        trades = [t for t in transactions if t.get('type') != 'rebalance_session']

        total_fees = 0
        total_volume = 0
        fees_by_asset = {}

        for trade in trades:
            fee = trade.get('total_fees', 0)
            volume = trade.get('filled_value', 0)
            asset = trade.get('asset')

            total_fees += fee
            total_volume += volume

            if asset:
                if asset not in fees_by_asset:
                    fees_by_asset[asset] = {'fees': 0, 'volume': 0, 'count': 0}
                fees_by_asset[asset]['fees'] += fee
                fees_by_asset[asset]['volume'] += volume
                fees_by_asset[asset]['count'] += 1

        # Calculate fee percentages
        avg_fee_pct = (total_fees / total_volume * 100) if total_volume > 0 else 0

        return {
            'period_days': days,
            'total_fees_usd': total_fees,
            'total_volume_usd': total_volume,
            'average_fee_percent': avg_fee_pct,
            'total_trades': len(trades),
            'fees_per_trade': total_fees / len(trades) if trades else 0,
            'fees_by_asset': fees_by_asset
        }

    def get_slippage_analysis(self, days: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze slippage over a period.

        Args:
            days: Number of days to analyze (None for all time)

        Returns:
            Dictionary with slippage analysis
        """
        start_date = None
        if days:
            start_date = (datetime.now() - timedelta(days=days)).isoformat()

        transactions = self.tx_logger.get_transactions(start_date=start_date)
        trades = [t for t in transactions
                 if t.get('type') != 'rebalance_session' and t.get('slippage_percent') is not None]

        if not trades:
            return {
                'period_days': days,
                'total_trades_analyzed': 0,
                'message': 'No trades with slippage data'
            }

        slippages = [t['slippage_percent'] for t in trades]
        slippage_usd = [t.get('slippage_usd', 0) for t in trades]

        # Separate by buy/sell
        buy_slippages = [t['slippage_percent'] for t in trades if t['action'] == 'BUY']
        sell_slippages = [t['slippage_percent'] for t in trades if t['action'] == 'SELL']

        return {
            'period_days': days,
            'total_trades_analyzed': len(trades),
            'average_slippage_percent': sum(slippages) / len(slippages),
            'max_slippage_percent': max(slippages),
            'min_slippage_percent': min(slippages),
            'total_slippage_cost_usd': sum(slippage_usd),
            'buy_slippage_avg': sum(buy_slippages) / len(buy_slippages) if buy_slippages else 0,
            'sell_slippage_avg': sum(sell_slippages) / len(sell_slippages) if sell_slippages else 0,
        }

    def get_rebalancing_performance(self, days: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze rebalancing session performance.

        Args:
            days: Number of days to analyze (None for all time)

        Returns:
            Dictionary with performance metrics
        """
        start_date = None
        if days:
            start_date = (datetime.now() - timedelta(days=days)).isoformat()

        transactions = self.tx_logger.get_transactions(start_date=start_date)
        sessions = [t for t in transactions if t.get('type') == 'rebalance_session']

        if not sessions:
            return {
                'period_days': days,
                'total_sessions': 0,
                'message': 'No rebalancing sessions found'
            }

        total_sessions = len(sessions)
        successful_sessions = sum(1 for s in sessions
                                 if all(t.get('success') for t in s.get('trades', [])))

        # Calculate portfolio value changes
        value_changes = []
        for session in sessions:
            before = session.get('portfolio_value_before', 0)
            after = session.get('portfolio_value_after', 0)
            if before > 0:
                change_pct = ((after - before) / before) * 100
                value_changes.append(change_pct)

        avg_value_change = sum(value_changes) / len(value_changes) if value_changes else 0

        # Count total trades across all sessions
        total_trades = sum(s.get('total_trades', 0) for s in sessions)
        dry_run_sessions = sum(1 for s in sessions if s.get('dry_run', False))

        return {
            'period_days': days,
            'total_sessions': total_sessions,
            'successful_sessions': successful_sessions,
            'dry_run_sessions': dry_run_sessions,
            'live_sessions': total_sessions - dry_run_sessions,
            'total_trades': total_trades,
            'average_trades_per_session': total_trades / total_sessions if total_sessions else 0,
            'average_portfolio_change_percent': avg_value_change,
        }

    def get_asset_trading_summary(self, asset: str, days: Optional[int] = None) -> Dict[str, Any]:
        """
        Get trading summary for a specific asset.

        Args:
            asset: Asset symbol (e.g., 'BTC')
            days: Number of days to analyze (None for all time)

        Returns:
            Dictionary with asset trading summary
        """
        start_date = None
        if days:
            start_date = (datetime.now() - timedelta(days=days)).isoformat()

        transactions = self.tx_logger.get_transactions(start_date=start_date)
        trades = [t for t in transactions
                 if t.get('type') != 'rebalance_session' and t.get('asset') == asset]

        if not trades:
            return {
                'asset': asset,
                'period_days': days,
                'total_trades': 0,
                'message': f'No trades found for {asset}'
            }

        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']

        total_bought = sum(t.get('filled_size', 0) for t in buy_trades)
        total_sold = sum(t.get('filled_size', 0) for t in sell_trades)

        total_spent = sum(t.get('filled_value', 0) for t in buy_trades)
        total_received = sum(t.get('filled_value', 0) for t in sell_trades)

        total_fees = sum(t.get('total_fees', 0) for t in trades)

        avg_buy_price = total_spent / total_bought if total_bought > 0 else 0
        avg_sell_price = total_received / total_sold if total_sold > 0 else 0

        return {
            'asset': asset,
            'period_days': days,
            'total_trades': len(trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'total_bought': total_bought,
            'total_sold': total_sold,
            'net_position_change': total_bought - total_sold,
            'total_spent_usd': total_spent,
            'total_received_usd': total_received,
            'net_cash_flow_usd': total_received - total_spent,
            'total_fees_usd': total_fees,
            'average_buy_price': avg_buy_price,
            'average_sell_price': avg_sell_price,
        }

    def print_comprehensive_report(self, days: Optional[int] = 30) -> None:
        """
        Print a comprehensive report to console.

        Args:
            days: Number of days to analyze (None for all time)
        """
        period_str = f"Last {days} days" if days else "All time"

        print("\n" + "=" * 80)
        print(f"COMPREHENSIVE TRADING REPORT - {period_str}")
        print("=" * 80)

        # Summary stats
        print("\n📊 OVERALL STATISTICS")
        print("-" * 80)
        stats = self.tx_logger.get_summary_stats()
        print(f"Total Transactions: {stats['total_transactions']}")
        print(f"Rebalancing Sessions: {stats['total_rebalance_sessions']}")
        print(f"Individual Trades: {stats['individual_trades']}")
        print(f"Successful Trades: {stats['successful_trades']}")
        print(f"Total Fees: {format_currency(stats['total_fees_usd'])}")
        print(f"Total Trade Volume: {format_currency(stats['total_trade_value_usd'])}")
        if stats['total_trade_value_usd'] > 0:
            fee_pct = (stats['total_fees_usd'] / stats['total_trade_value_usd']) * 100
            print(f"Average Fee Rate: {format_percentage(fee_pct)}")
        print(f"Average Slippage: {format_percentage(stats['average_slippage_percent'])}")

        # Fee analysis
        print("\n💰 FEE ANALYSIS")
        print("-" * 80)
        fee_analysis = self.get_fee_analysis(days)
        print(f"Total Fees: {format_currency(fee_analysis['total_fees_usd'])}")
        print(f"Total Volume: {format_currency(fee_analysis['total_volume_usd'])}")
        print(f"Average Fee: {format_percentage(fee_analysis['average_fee_percent'])}")
        print(f"Fees per Trade: {format_currency(fee_analysis['fees_per_trade'])}")

        # Slippage analysis
        print("\n📉 SLIPPAGE ANALYSIS")
        print("-" * 80)
        slippage = self.get_slippage_analysis(days)
        if slippage.get('total_trades_analyzed', 0) > 0:
            print(f"Trades Analyzed: {slippage['total_trades_analyzed']}")
            print(f"Average Slippage: {format_percentage(slippage['average_slippage_percent'])}")
            print(f"Max Slippage: {format_percentage(slippage['max_slippage_percent'])}")
            print(f"Min Slippage: {format_percentage(slippage['min_slippage_percent'])}")
            print(f"Total Slippage Cost: {format_currency(slippage['total_slippage_cost_usd'])}")
            print(f"Buy Slippage Avg: {format_percentage(slippage['buy_slippage_avg'])}")
            print(f"Sell Slippage Avg: {format_percentage(slippage['sell_slippage_avg'])}")
        else:
            print("No slippage data available")

        # Rebalancing performance
        print("\n🔄 REBALANCING PERFORMANCE")
        print("-" * 80)
        performance = self.get_rebalancing_performance(days)
        if performance.get('total_sessions', 0) > 0:
            print(f"Total Sessions: {performance['total_sessions']}")
            print(f"Successful Sessions: {performance['successful_sessions']}")
            print(f"Live Sessions: {performance['live_sessions']}")
            print(f"Dry Run Sessions: {performance['dry_run_sessions']}")
            print(f"Total Trades: {performance['total_trades']}")
            print(f"Avg Trades/Session: {performance['average_trades_per_session']:.1f}")
            print(f"Avg Portfolio Change: {format_percentage(performance['average_portfolio_change_percent'])}")
        else:
            print("No rebalancing sessions found")

        print("\n" + "=" * 80 + "\n")


def main():
    """Main function for running reports from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Transaction Reporting Tool")
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days to analyze (default: 30, 0 for all time)')
    parser.add_argument('--export-csv', action='store_true',
                       help='Export transactions to CSV')
    parser.add_argument('--asset', type=str,
                       help='Get detailed report for specific asset')

    args = parser.parse_args()

    reporter = TransactionReporter()
    days = None if args.days == 0 else args.days

    if args.asset:
        # Asset-specific report
        print(f"\nAsset Trading Summary: {args.asset}")
        print("=" * 80)
        summary = reporter.get_asset_trading_summary(args.asset, days)
        for key, value in summary.items():
            if isinstance(value, float):
                if 'usd' in key.lower() or 'price' in key.lower():
                    print(f"{key}: {format_currency(value)}")
                elif 'percent' in key.lower():
                    print(f"{key}: {format_percentage(value)}")
                else:
                    print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    else:
        # Comprehensive report
        reporter.print_comprehensive_report(days)

    if args.export_csv:
        logger = TransactionLogger()
        logger.export_to_csv()
        print("\n✅ Transactions exported to data/transactions.csv")


if __name__ == "__main__":
    main()
