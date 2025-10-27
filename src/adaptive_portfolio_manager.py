"""Adaptive portfolio manager with regime-based allocation."""
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Optional
from pathlib import Path

from src.portfolio_manager import PortfolioManager
from src.regime_detector import ReturnDetector, MarketRegime
from src.monte_carlo_simulator import HistoricalPriceFetcher
from src.coinbase_client import CoinbaseClient
from src.transaction_logger import TransactionLogger


class AdaptivePortfolioManager(PortfolioManager):
    """Portfolio manager with regime-based adaptive allocation."""

    def __init__(self, config_path: str = "config/portfolio.json",
                 coinbase_client: Optional[CoinbaseClient] = None,
                 transaction_logger: Optional[TransactionLogger] = None):
        """
        Initialize adaptive portfolio manager.

        Args:
            config_path: Path to portfolio configuration file
            coinbase_client: Optional CoinbaseClient instance
            transaction_logger: Optional TransactionLogger instance
        """
        super().__init__(config_path, coinbase_client, transaction_logger)

        # Load adaptive configuration
        self.adaptive_config = self.config.get('adaptive', {})

        if not self.adaptive_config.get('enabled', False):
            self.logger.warning("Adaptive mode is not enabled in configuration")

        # Initialize regime detector
        detector_config = self.adaptive_config.get('detector', {})
        self.detector = ReturnDetector(
            window_days=detector_config.get('window_days', 30),
            bull_threshold=detector_config.get('bull_threshold', 15.0),
            bear_threshold=detector_config.get('bear_threshold', -10.0)
        )

        # Load portfolio mappings
        portfolios_path = self.adaptive_config.get('portfolios_path', 'config/adaptive_portfolios.json')
        self.regime_portfolios_config = self._load_portfolio_configs(portfolios_path)

        # Regime portfolio mapping
        regime_map = self.adaptive_config.get('regime_portfolios', {})
        self.regime_portfolios = {
            MarketRegime.BULL: regime_map.get('bull', 'Top_Three'),
            MarketRegime.BEAR: regime_map.get('bear', 'Stablecoin_Heavy'),
            MarketRegime.NEUTRAL: regime_map.get('neutral', 'Top_Three')
        }

        # Persistence tracking
        self.persistence_days = self.adaptive_config.get('persistence_days', 14)
        self.check_frequency_days = self.adaptive_config.get('check_frequency_days', 7)

        # State tracking
        self.regime_buffer = []
        self.last_check_date = None
        self.current_regime = MarketRegime.NEUTRAL
        self.current_portfolio_name = None

        # Initialize historical price fetcher
        self.price_fetcher = HistoricalPriceFetcher(
            self.client,
            use_cache=True,
            cache_max_age_days=7
        )

        self.logger.info("Adaptive Portfolio Manager initialized")
        self.logger.info(f"  Detector: ReturnDetector(bull={detector_config.get('bull_threshold', 15.0)}%, "
                        f"bear={detector_config.get('bear_threshold', -10.0)}%)")
        self.logger.info(f"  Persistence: {self.persistence_days} days")
        self.logger.info(f"  Check Frequency: {self.check_frequency_days} days")

        # Perform initial regime detection to set correct portfolio immediately
        self.logger.info("Performing initial regime detection...")
        self._initialize_regime()

    def _load_portfolio_configs(self, path: str) -> Dict[str, Dict[str, float]]:
        """
        Load portfolio configurations from JSON file.

        Args:
            path: Path to portfolios JSON file

        Returns:
            Dictionary of portfolio name -> allocation dict
        """
        file_path = Path(path)
        if not file_path.exists():
            self.logger.error(f"Portfolio configs file not found: {path}")
            return {}

        with open(file_path, 'r') as f:
            portfolios = json.load(f)

        self.logger.info(f"Loaded {len(portfolios)} portfolio configurations from {path}")
        return portfolios

    def _fetch_btc_history(self, days: int = 30) -> tuple[list[float], list[datetime]]:
        """
        Fetch recent BTC price history for regime detection.

        Args:
            days: Number of days of history to fetch

        Returns:
            Tuple of (prices, timestamps)
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        try:
            # Fetch BTC prices
            price_data = self.price_fetcher.fetch_historical_prices(
                assets=['BTC'],
                start_date=start_date,
                end_date=end_date,
                granularity='ONE_DAY',
                show_progress=False
            )

            btc_prices = price_data.get('BTC', [])

            if not btc_prices:
                self.logger.warning("No BTC price history available")
                return [], []

            # Extract prices and timestamps
            prices = [p[1] for p in btc_prices]
            timestamps = [p[0] for p in btc_prices]

            self.logger.debug(f"Fetched {len(prices)} BTC price points for regime detection")
            return prices, timestamps

        except Exception as e:
            self.logger.error(f"Error fetching BTC price history: {e}")
            return [], []

    def _initialize_regime(self) -> None:
        """Initialize regime and portfolio on first startup."""
        detected_regime = self.detect_regime()

        if detected_regime:
            self.current_regime = detected_regime
            portfolio_name = self.regime_portfolios[self.current_regime]

            if portfolio_name in self.regime_portfolios_config:
                self.config['target_allocation'] = self.regime_portfolios_config[portfolio_name]
                self.current_portfolio_name = portfolio_name
                self.logger.info(f"✅ Initialized with portfolio: {portfolio_name} (regime: {self.current_regime.value})")
            else:
                self.logger.error(f"Portfolio {portfolio_name} not found in configurations")

    def detect_regime(self) -> Optional[MarketRegime]:
        """
        Detect current market regime.

        Returns:
            Detected regime or None if detection fails
        """
        # Fetch 30 days of BTC price history
        prices, timestamps = self._fetch_btc_history(days=30)

        if not prices or len(prices) < 2:
            self.logger.warning("Insufficient price data for regime detection")
            return None

        # Detect regime
        try:
            detection = self.detector.detect(prices, timestamps)

            self.logger.info(f"Regime detected: {detection.regime.value}")
            self.logger.info(f"  Confidence: {detection.confidence:.3f}")
            self.logger.info(f"  30-day return: {detection.return_30d:.2f}%")

            return detection.regime

        except Exception as e:
            self.logger.error(f"Error detecting regime: {e}")
            return None

    def update_target_allocation(self) -> bool:
        """
        Update target allocation based on detected regime.

        Returns:
            True if allocation was updated, False otherwise
        """
        # Check if enough time has passed since last check
        if self.last_check_date is not None:
            days_since_check = (datetime.now() - self.last_check_date).days
            if days_since_check < self.check_frequency_days:
                self.logger.debug(f"Skipping regime check ({days_since_check} < {self.check_frequency_days} days)")
                return False

        # Detect current regime
        detected_regime = self.detect_regime()

        if detected_regime is None:
            self.logger.warning("Could not detect regime, keeping current allocation")
            return False

        # Update regime buffer for persistence check
        self.regime_buffer.append(detected_regime)
        buffer_size = max(2, self.persistence_days // self.check_frequency_days)
        if len(self.regime_buffer) > buffer_size:
            self.regime_buffer.pop(0)

        self.last_check_date = datetime.now()

        # Check if regime change is confirmed (most common regime in buffer)
        if len(self.regime_buffer) >= 2:
            from collections import Counter
            regime_counts = Counter(self.regime_buffer)
            most_common_regime = regime_counts.most_common(1)[0][0]

            # Check if regime has changed
            if most_common_regime != self.current_regime:
                self.logger.warning(f"🔄 REGIME CHANGE DETECTED: {self.current_regime.value} → {most_common_regime.value}")
                self.logger.warning(f"   Regime buffer: {[r.value for r in self.regime_buffer]}")

                # Update current regime
                old_regime = self.current_regime
                self.current_regime = most_common_regime

                # Get portfolio name for new regime
                new_portfolio_name = self.regime_portfolios[self.current_regime]

                # Load new target allocation
                if new_portfolio_name in self.regime_portfolios_config:
                    new_allocation = self.regime_portfolios_config[new_portfolio_name]

                    # Validate allocation sums to 100%
                    total = sum(new_allocation.values())
                    if abs(total - 100.0) > 0.01:
                        self.logger.error(f"Portfolio {new_portfolio_name} allocation sums to {total}%, not 100%")
                        return False

                    # Update config with new allocation
                    self.config['target_allocation'] = new_allocation
                    self.current_portfolio_name = new_portfolio_name

                    self.logger.warning(f"✅ Switched to portfolio: {new_portfolio_name}")
                    self.logger.warning(f"   New allocation: {new_allocation}")

                    return True
                else:
                    self.logger.error(f"Portfolio {new_portfolio_name} not found in configurations")
                    self.current_regime = old_regime  # Revert
                    return False

        return False

    def reload_config(self) -> None:
        """
        Reload configuration from file, but preserve dynamic target allocation.
        """
        # Save current target allocation
        saved_target_allocation = self.config.get('target_allocation')

        # Reload config from parent
        super().reload_config()

        # Restore dynamic target allocation (overrides static config)
        if saved_target_allocation is not None:
            self.config['target_allocation'] = saved_target_allocation
            self.logger.debug("Preserved dynamic target allocation after config reload")

    def rebalance(self) -> Dict:
        """
        Execute portfolio rebalancing with regime-based adaptation.

        Returns:
            Dictionary with rebalancing results
        """
        # Check if adaptive mode is enabled
        if not self.adaptive_config.get('enabled', False):
            self.logger.info("Adaptive mode disabled, using static allocation")
            return super().rebalance()

        # Update target allocation based on detected regime
        allocation_updated = self.update_target_allocation()

        if allocation_updated:
            self.logger.warning(f"📊 Portfolio allocation updated due to regime change")

        # Log current regime status
        self.logger.info(f"Current Regime: {self.current_regime.value}")
        self.logger.info(f"Current Portfolio: {self.current_portfolio_name or self.regime_portfolios[self.current_regime]}")

        # Call parent rebalance method with updated allocation
        return super().rebalance()

    def get_portfolio_status(self) -> Dict:
        """
        Get current portfolio status with regime information.

        Returns:
            Dictionary with portfolio and regime information
        """
        # Update regime if needed (but don't change allocation)
        detected_regime = self.detect_regime()

        # Get base status from parent
        status = super().get_portfolio_status()

        # Add regime information
        status['regime'] = {
            'current': self.current_regime.value if self.current_regime else 'unknown',
            'detected': detected_regime.value if detected_regime else 'unknown',
            'portfolio': self.current_portfolio_name or self.regime_portfolios.get(self.current_regime, 'unknown'),
            'last_check': self.last_check_date.isoformat() if self.last_check_date else None,
            'regime_buffer': [r.value for r in self.regime_buffer]
        }

        return status
