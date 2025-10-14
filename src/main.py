"""Main entry point for the portfolio rebalancing bot."""
import logging
import signal
import sys
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from src.portfolio_manager import PortfolioManager
from src.utils import setup_logging, load_json, validate_portfolio_config


class RebalanceBot:
    """Main bot class that handles scheduling and execution."""

    def __init__(self, config_path: str = "config/portfolio.json"):
        """Initialize the rebalancing bot."""
        self.config_path = config_path
        self.config = load_json(config_path)
        self.logger = setup_logging()
        self.scheduler = BlockingScheduler()
        self.portfolio_manager = PortfolioManager(config_path)

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info("Rebalancing bot initialized")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)

    def validate_configuration(self) -> bool:
        """Validate the configuration file."""
        if not validate_portfolio_config(self.config):
            self.logger.error("Invalid configuration file")
            return False

        self.logger.info("Configuration validated successfully")
        return True

    def run_rebalance(self):
        """Execute a rebalancing operation."""
        try:
            self.logger.info("=" * 60)
            self.logger.info("Starting scheduled rebalancing")
            self.logger.info("=" * 60)

            result = self.portfolio_manager.rebalance()

            self.logger.info("=" * 60)
            self.logger.info("Rebalancing completed")
            self.logger.info(f"Session ID: {result['session_id']}")
            self.logger.info(f"Trades executed: {result['trades_executed']}/{result.get('total_trades', 0)}")
            self.logger.info(f"Dry run: {result.get('dry_run', False)}")
            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"Error during scheduled rebalancing: {e}", exc_info=True)

    def run_once(self):
        """Run rebalancing once and exit."""
        self.logger.info("Running rebalancing once")

        if not self.validate_configuration():
            return

        self.run_rebalance()
        self.logger.info("One-time rebalancing complete")

    def start_scheduler(self):
        """Start the scheduler for periodic rebalancing."""
        if not self.validate_configuration():
            self.logger.error("Cannot start scheduler due to invalid configuration")
            return

        schedule_config = self.config.get('schedule', {})
        interval_hours = schedule_config.get('interval_hours', 24)
        enabled = schedule_config.get('enabled', True)

        if not enabled:
            self.logger.warning("Scheduler is disabled in configuration")
            return

        self.logger.info(f"Scheduling rebalancing every {interval_hours} hours")

        # Add the rebalancing job
        self.scheduler.add_job(
            self.run_rebalance,
            trigger=IntervalTrigger(hours=interval_hours),
            id='rebalance_job',
            name='Portfolio Rebalancing',
            replace_existing=True
        )

        try:
            self.logger.info("Starting scheduler...")
            self.logger.info("Press Ctrl+C to stop")
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            self.logger.info("Scheduler stopped")

    def stop(self):
        """Stop the scheduler."""
        if self.scheduler.running:
            self.logger.info("Stopping scheduler...")
            self.scheduler.shutdown()
            self.logger.info("Scheduler stopped")

    def status(self):
        """Display current portfolio status."""
        try:
            status = self.portfolio_manager.get_portfolio_status()

            print("\n" + "=" * 60)
            print("PORTFOLIO STATUS")
            print("=" * 60)
            print(f"Total Value: ${status['total_value']:,.2f}")
            print("\nAsset Allocation:")
            print("-" * 60)
            print(f"{'Asset':<10} {'Current %':<12} {'Target %':<12} {'Deviation':<12} {'Value':<15}")
            print("-" * 60)

            for asset_info in sorted(status['assets'], key=lambda x: x['current_value'], reverse=True):
                asset = asset_info['asset']
                current = asset_info['current_allocation']
                target = asset_info['target_allocation']
                deviation = asset_info['deviation']
                value = asset_info['current_value']

                deviation_str = f"{deviation:+.2f}%"
                print(f"{asset:<10} {current:>10.2f}% {target:>10.2f}% {deviation_str:>10} ${value:>12,.2f}")

            print("=" * 60 + "\n")

        except Exception as e:
            self.logger.error(f"Error getting portfolio status: {e}", exc_info=True)


def main():
    """Main function to run the bot."""
    import argparse

    parser = argparse.ArgumentParser(description="Coinbase Portfolio Rebalancing Bot")
    parser.add_argument(
        '--mode',
        choices=['once', 'schedule', 'status'],
        default='schedule',
        help='Run mode: once (single run), schedule (continuous), or status (view portfolio)'
    )
    parser.add_argument(
        '--config',
        default='config/portfolio.json',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    bot = RebalanceBot(config_path=args.config)

    if args.mode == 'once':
        bot.run_once()
    elif args.mode == 'schedule':
        bot.start_scheduler()
    elif args.mode == 'status':
        bot.status()


if __name__ == "__main__":
    main()
