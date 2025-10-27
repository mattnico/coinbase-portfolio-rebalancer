"""Coinbase API client wrapper for portfolio rebalancing."""
import logging
import os
import time
from typing import Dict, List, Optional, Tuple
from coinbase.rest import RESTClient
from dotenv import load_dotenv


class CoinbaseClient:
    """Wrapper for Coinbase Advanced Trade API."""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, portfolio_id: Optional[str] = None):
        """Initialize Coinbase client with API credentials."""
        # Load .env from config directory
        from pathlib import Path
        env_path = Path(__file__).parent.parent / 'config' / '.env'
        load_dotenv(env_path)

        self.api_key = api_key or os.getenv('COINBASE_API_KEY')
        self.api_secret = api_secret or os.getenv('COINBASE_API_SECRET')
        # Portfolio ID now comes from config parameter (not .env)
        self.portfolio_id = portfolio_id

        if not self.api_key or not self.api_secret:
            raise ValueError("API credentials not found. Set COINBASE_API_KEY and COINBASE_API_SECRET")

        self.client = RESTClient(api_key=self.api_key, api_secret=self.api_secret)
        self.logger = logging.getLogger(__name__)

        # Cache for available trading pairs
        self._trading_pairs_cache: Optional[Dict[str, bool]] = None
        self._cache_timestamp: Optional[float] = None

        # Cache for product details (including precision info)
        self._product_details_cache: Dict[str, Dict] = {}
        self._product_cache_timestamp: Optional[float] = None

    def get_accounts(self) -> Dict[str, float]:
        """
        Get all account balances.

        Returns:
            Dictionary mapping asset symbols to available balances
        """
        try:
            accounts_response = self.client.get_accounts()
            balances = {}

            if hasattr(accounts_response, 'accounts'):
                for account in accounts_response.accounts:
                    # Handle both object and dict response formats
                    if hasattr(account, 'currency'):
                        currency = account.currency
                    else:
                        currency = account.get('currency')

                    # Handle available_balance as object or dict
                    if hasattr(account, 'available_balance'):
                        avail_bal = account.available_balance
                        if hasattr(avail_bal, 'value'):
                            available_balance = float(avail_bal.value)
                        elif isinstance(avail_bal, dict):
                            available_balance = float(avail_bal.get('value', 0))
                        else:
                            available_balance = float(avail_bal)
                    else:
                        avail_bal = account.get('available_balance', {})
                        if isinstance(avail_bal, dict):
                            available_balance = float(avail_bal.get('value', 0))
                        else:
                            available_balance = float(avail_bal)

                    if available_balance > 0:
                        balances[currency] = available_balance

            self.logger.info(f"Retrieved {len(balances)} account balances")
            return balances

        except Exception as e:
            self.logger.error(f"Error fetching accounts: {e}")
            raise

    def get_product_price(self, product_id: str) -> float:
        """
        Get current price for a trading pair.

        Args:
            product_id: Trading pair (e.g., 'BTC-USD')

        Returns:
            Current price as float
        """
        try:
            product = self.client.get_product(product_id)

            if hasattr(product, 'price'):
                price = float(product.price)
                self.logger.debug(f"Price for {product_id}: ${price:,.2f}")
                return price
            else:
                raise ValueError(f"Could not retrieve price for {product_id}")

        except Exception as e:
            self.logger.error(f"Error fetching price for {product_id}: {e}")
            raise

    def get_portfolio_value_usd(self, balances: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total portfolio value in USD.

        Args:
            balances: Optional pre-fetched balances

        Returns:
            Tuple of (total_value_usd, individual_values_usd)
        """
        if balances is None:
            balances = self.get_accounts()

        values = {}
        total_value = 0.0

        for asset, balance in balances.items():
            if asset == 'USD' or asset == 'USDC':
                value = balance
            else:
                try:
                    price = self.get_product_price(f"{asset}-USD")
                    value = balance * price
                    time.sleep(0.1)  # Rate limiting
                except Exception as e:
                    self.logger.warning(f"Could not get price for {asset}: {e}")
                    continue

            values[asset] = value
            total_value += value

        return total_value, values

    def place_market_order(self, product_id: str, side: str, size: Optional[float] = None,
                          quote_size: Optional[float] = None) -> Dict:
        """
        Place a market order.

        Args:
            product_id: Trading pair (e.g., 'BTC-USD')
            side: 'BUY' or 'SELL'
            size: Amount in base currency (e.g., 0.01 BTC)
            quote_size: Amount in quote currency (e.g., 100 USD)

        Returns:
            Order response dictionary
        """
        try:
            import uuid

            # Generate unique client order ID
            client_order_id = str(uuid.uuid4())

            self.logger.info(f"Placing {side} order for {product_id}: "
                           f"size={size}, quote_size={quote_size}")

            # Get product details for precision
            product_details = self.get_product_details(product_id)

            # Build parameters for market_order call
            order_params = {
                "client_order_id": client_order_id,
                "product_id": product_id,
                "side": side.upper(),
            }

            # Add retail_portfolio_id if available
            if self.portfolio_id:
                order_params["retail_portfolio_id"] = self.portfolio_id

            # Add size parameters with proper precision rounding
            if size:
                # Round base_size according to product's base_increment
                base_increment = product_details.get('base_increment', '0.00000001')
                rounded_size = self.round_size_to_increment(size, base_increment)
                order_params["base_size"] = rounded_size
                self.logger.debug(f"Rounded base_size from {size} to {rounded_size} (increment: {base_increment})")
            elif quote_size:
                # Round quote_size according to product's quote_increment
                quote_increment = product_details.get('quote_increment', '0.01')
                rounded_quote_size = self.round_size_to_increment(quote_size, quote_increment)
                order_params["quote_size"] = rounded_quote_size
                self.logger.debug(f"Rounded quote_size from {quote_size} to {rounded_quote_size} (increment: {quote_increment})")
            else:
                raise ValueError("Must specify either size or quote_size")

            # Call market_order with correct signature
            response = self.client.market_order(**order_params)

            # Log full response for debugging
            self.logger.info(f"API Response - success: {response.success if hasattr(response, 'success') else 'N/A'}")

            # Check for error response
            if hasattr(response, 'error_response') and response.error_response:
                self.logger.error(f"Order failed with error: {response.error_response}")
                raise Exception(f"Order placement failed: {response.error_response}")

            # Extract order_id from success_response
            order_id = None
            if hasattr(response, 'success_response') and response.success_response:
                if isinstance(response.success_response, dict):
                    order_id = response.success_response.get('order_id')
                elif hasattr(response.success_response, 'order_id'):
                    order_id = response.success_response.order_id

            if not order_id:
                self.logger.warning(f"No order_id in response. Response dict: {response.to_dict() if hasattr(response, 'to_dict') else 'N/A'}")

            self.logger.info(f"Order placed. Order ID: {order_id}")

            return {
                "success": response.success if hasattr(response, 'success') else True,
                "order_id": order_id,
                "product_id": product_id,
                "side": side,
                "size": size,
                "quote_size": quote_size,
            }

        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            raise

    def get_available_trading_pairs(self, force_refresh: bool = False) -> Dict[str, bool]:
        """
        Get all available trading pairs on Coinbase.
        Results are cached for 1 hour to reduce API calls.

        Args:
            force_refresh: Force refresh of cache

        Returns:
            Dictionary mapping product_id to True (e.g., {'BTC-ETH': True})
        """
        # Check cache validity (1 hour = 3600 seconds)
        current_time = time.time()
        if (not force_refresh and self._trading_pairs_cache is not None and
            self._cache_timestamp is not None and
            current_time - self._cache_timestamp < 3600):
            return self._trading_pairs_cache

        try:
            self.logger.info("Fetching available trading pairs from Coinbase")
            products = self.client.get_products()

            trading_pairs = {}
            if hasattr(products, 'products'):
                for product in products.products:
                    if hasattr(product, 'product_id'):
                        # Only include active trading pairs
                        is_active = True
                        if hasattr(product, 'status'):
                            is_active = product.status == 'online'

                        if is_active:
                            trading_pairs[product.product_id] = True

            self._trading_pairs_cache = trading_pairs
            self._cache_timestamp = current_time
            self.logger.info(f"Cached {len(trading_pairs)} trading pairs")

            return trading_pairs

        except Exception as e:
            self.logger.error(f"Error fetching trading pairs: {e}")
            # Return cached data if available, otherwise empty dict
            return self._trading_pairs_cache or {}

    def is_trading_pair_available(self, base: str, quote: str) -> bool:
        """
        Check if a direct trading pair exists.

        Args:
            base: Base currency (e.g., 'BTC')
            quote: Quote currency (e.g., 'ETH')

        Returns:
            True if the pair exists
        """
        pairs = self.get_available_trading_pairs()

        # Check both directions
        pair_1 = f"{base}-{quote}"
        pair_2 = f"{quote}-{base}"

        return pair_1 in pairs or pair_2 in pairs

    def get_trading_pair_format(self, base: str, quote: str) -> Optional[str]:
        """
        Get the correct format for a trading pair if it exists.

        Args:
            base: Base currency
            quote: Quote currency

        Returns:
            Product ID in correct format, or None if pair doesn't exist
        """
        pairs = self.get_available_trading_pairs()

        pair_1 = f"{base}-{quote}"
        pair_2 = f"{quote}-{base}"

        if pair_1 in pairs:
            return pair_1
        elif pair_2 in pairs:
            return pair_2
        return None

    def get_product_details(self, product_id: str, force_refresh: bool = False) -> Dict:
        """
        Get detailed product information including size precision.

        Args:
            product_id: Trading pair (e.g., 'BTC-USD')
            force_refresh: Force refresh of cache

        Returns:
            Dictionary with product details including base_increment and quote_increment
        """
        # Check cache validity (1 hour = 3600 seconds)
        current_time = time.time()
        if (not force_refresh and product_id in self._product_details_cache and
            self._product_cache_timestamp is not None and
            current_time - self._product_cache_timestamp < 3600):
            return self._product_details_cache[product_id]

        try:
            product = self.client.get_product(product_id)

            details = {
                'product_id': product_id,
                'base_increment': '0.00000001',  # Default to 8 decimals
                'quote_increment': '0.01',  # Default to 2 decimals
            }

            # Extract base_increment if available
            if hasattr(product, 'base_increment'):
                details['base_increment'] = str(product.base_increment)
            elif isinstance(product, dict) and 'base_increment' in product:
                details['base_increment'] = str(product['base_increment'])

            # Extract quote_increment if available
            if hasattr(product, 'quote_increment'):
                details['quote_increment'] = str(product.quote_increment)
            elif isinstance(product, dict) and 'quote_increment' in product:
                details['quote_increment'] = str(product['quote_increment'])

            # Cache the result
            self._product_details_cache[product_id] = details
            if self._product_cache_timestamp is None:
                self._product_cache_timestamp = current_time

            self.logger.debug(f"Product details for {product_id}: base_increment={details['base_increment']}, quote_increment={details['quote_increment']}")

            return details

        except Exception as e:
            self.logger.warning(f"Error fetching product details for {product_id}: {e}. Using defaults.")
            # Return defaults if API call fails
            return {
                'product_id': product_id,
                'base_increment': '0.00000001',
                'quote_increment': '0.01',
            }

    def round_size_to_increment(self, size: float, increment: str) -> str:
        """
        Round a size value to match the product's increment precision.

        Args:
            size: The size to round
            increment: The increment string (e.g., '0.00000001')

        Returns:
            Rounded size as string with correct precision
        """
        import decimal
        from decimal import Decimal, ROUND_DOWN

        # Convert increment to Decimal for precision
        inc = Decimal(increment)

        # Get number of decimal places from increment
        decimal_places = abs(inc.as_tuple().exponent)

        # Round down to the increment
        size_decimal = Decimal(str(size))
        rounded = (size_decimal / inc).quantize(Decimal('1'), rounding=ROUND_DOWN) * inc

        # Format to string with correct decimal places
        return f"{rounded:.{decimal_places}f}"

    def get_order(self, order_id: str) -> Dict:
        """
        Get order details by ID with complete fill information.

        Args:
            order_id: Order ID to query

        Returns:
            Order details dictionary with fees, fills, and prices
        """
        try:
            order = self.client.get_order(order_id)

            # Extract basic order info
            order_details = {
                "order_id": order_id,
                "status": order.status if hasattr(order, 'status') else 'UNKNOWN',
                "product_id": order.product_id if hasattr(order, 'product_id') else None,
                "side": order.side if hasattr(order, 'side') else None,
                "created_time": order.created_time if hasattr(order, 'created_time') else None,
                "completion_time": order.completion_percentage if hasattr(order, 'completion_percentage') else None,
            }

            # Extract fill information
            if hasattr(order, 'order_configuration'):
                order_details['order_type'] = 'market'

            # Filled size and price
            if hasattr(order, 'filled_size'):
                order_details['filled_size'] = float(order.filled_size)
            else:
                order_details['filled_size'] = 0.0

            if hasattr(order, 'average_filled_price'):
                order_details['average_filled_price'] = float(order.average_filled_price)
            else:
                order_details['average_filled_price'] = 0.0

            # Calculate filled value
            order_details['filled_value'] = (
                order_details['filled_size'] * order_details['average_filled_price']
            )

            # Extract fee information
            if hasattr(order, 'total_fees'):
                order_details['total_fees'] = float(order.total_fees)
            else:
                order_details['total_fees'] = 0.0

            if hasattr(order, 'fee_currency'):
                order_details['fee_currency'] = order.fee_currency
            else:
                order_details['fee_currency'] = 'USD'

            # Extract number of fills
            if hasattr(order, 'number_of_fills'):
                order_details['number_of_fills'] = int(order.number_of_fills)
            else:
                order_details['number_of_fills'] = 0

            # Extract fill details if available
            fills = []
            if hasattr(order, 'fills') and order.fills:
                for fill in order.fills:
                    fill_data = {
                        'trade_id': fill.trade_id if hasattr(fill, 'trade_id') else None,
                        'price': float(fill.price) if hasattr(fill, 'price') else 0.0,
                        'size': float(fill.size) if hasattr(fill, 'size') else 0.0,
                        'fee': float(fill.commission) if hasattr(fill, 'commission') else 0.0,
                        'trade_time': fill.trade_time if hasattr(fill, 'trade_time') else None,
                    }
                    fills.append(fill_data)

            order_details['fills'] = fills

            return order_details

        except Exception as e:
            self.logger.error(f"Error fetching order {order_id}: {e}")
            raise

    def get_candles(
        self,
        product_id: str,
        start: int,
        end: int,
        granularity: str = "ONE_DAY"
    ) -> Dict:
        """
        Get historical candle data for a product.

        Args:
            product_id: Trading pair (e.g., 'BTC-USD')
            start: Start time as Unix timestamp
            end: End time as Unix timestamp
            granularity: Candle granularity (ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE,
                        THIRTY_MINUTE, ONE_HOUR, TWO_HOUR, SIX_HOUR, ONE_DAY)

        Returns:
            Dictionary with candles data
        """
        try:
            self.logger.debug(f"Fetching candles for {product_id} from {start} to {end}")

            response = self.client.get_candles(
                product_id=product_id,
                start=str(start),
                end=str(end),
                granularity=granularity
            )

            # Convert response to dict format
            result = {'candles': []}

            if hasattr(response, 'candles'):
                for candle in response.candles:
                    candle_data = {
                        'start': candle.start if hasattr(candle, 'start') else None,
                        'low': candle.low if hasattr(candle, 'low') else None,
                        'high': candle.high if hasattr(candle, 'high') else None,
                        'open': candle.open if hasattr(candle, 'open') else None,
                        'close': candle.close if hasattr(candle, 'close') else None,
                        'volume': candle.volume if hasattr(candle, 'volume') else None,
                    }
                    result['candles'].append(candle_data)

            return result

        except Exception as e:
            self.logger.error(f"Error fetching candles for {product_id}: {e}")
            raise
