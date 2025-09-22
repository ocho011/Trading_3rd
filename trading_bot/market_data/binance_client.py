"""
Binance API client wrapper for trading bot application.

Provides a secure, testable wrapper around the python-binance library
following SOLID principles and dependency injection patterns.
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Dict, Optional, Union

from binance.client import Client
from binance.exceptions import BinanceAPIException

from trading_bot.core.config_manager import ConfigManager
from trading_bot.core.logger import get_module_logger


class BinanceError(Exception):
    """Base exception for Binance-related errors."""

    pass


class BinanceConnectionError(BinanceError):
    """Exception raised for Binance connection failures."""

    pass


class BinanceAuthenticationError(BinanceError):
    """Exception raised for Binance authentication failures."""

    pass


class BinanceRateLimitError(BinanceError):
    """Exception raised when Binance rate limits are exceeded."""

    pass


class BinanceOrderError(BinanceError):
    """Exception raised for Binance order execution failures."""

    pass


class IExchangeClient(ABC):
    """Interface for cryptocurrency exchange clients."""

    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.

        Returns:
            Dict[str, Any]: Account information

        Raises:
            BinanceError: If account info retrieval fails
        """
        pass

    @abstractmethod
    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: Union[str, Decimal],
    ) -> Dict[str, Any]:
        """
        Place a market order.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity

        Returns:
            Dict[str, Any]: Order execution result

        Raises:
            BinanceOrderError: If order placement fails
        """
        pass

    @abstractmethod
    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: Union[str, Decimal],
        price: Union[str, Decimal],
    ) -> Dict[str, Any]:
        """
        Place a limit order.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity
            price: Order price

        Returns:
            Dict[str, Any]: Order execution result

        Raises:
            BinanceOrderError: If order placement fails
        """
        pass

    @abstractmethod
    def get_symbol_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get symbol ticker information.

        Args:
            symbol: Trading pair symbol

        Returns:
            Dict[str, Any]: Ticker information

        Raises:
            BinanceError: If ticker retrieval fails
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if client is connected and authenticated.

        Returns:
            bool: True if connected, False otherwise
        """
        pass


class BinanceClient(IExchangeClient):
    """
    Binance exchange client wrapper.

    Wraps the python-binance Client with additional error handling,
    logging, and configuration management following SOLID principles.
    """

    def __init__(self, config_manager: ConfigManager) -> None:
        """
        Initialize Binance client with dependency injection.

        Args:
            config_manager: Configuration manager for credentials
        """
        self._config_manager = config_manager
        self._client: Optional[Client] = None
        self._logger = get_module_logger("binance_client")
        self._is_testnet = False
        self._is_connected = False

    def initialize(self) -> None:
        """
        Initialize Binance client with API credentials.

        Raises:
            BinanceAuthenticationError: If credentials are invalid
            BinanceConnectionError: If connection setup fails
        """
        try:
            credentials = self._config_manager.get_api_credentials()
            trading_config = self._config_manager.get_trading_config()

            self._is_testnet = trading_config.get("trading_mode") == "paper"

            self._client = Client(
                api_key=credentials["api_key"],
                api_secret=credentials["secret_key"],
                testnet=self._is_testnet,
            )

            self._validate_connection()
            self._is_connected = True

            mode = "testnet" if self._is_testnet else "mainnet"
            self._logger.info(f"Binance client initialized in {mode} mode")

        except Exception as e:
            self._handle_connection_error(e)

    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information from Binance.

        Returns:
            Dict[str, Any]: Account information including balances

        Raises:
            BinanceError: If account info retrieval fails
        """
        self._ensure_connected()

        try:
            account_info = self._client.get_account()
            self._logger.debug("Account information retrieved successfully")
            return account_info

        except BinanceAPIException as e:
            self._handle_api_exception(e, "get account info")
        except Exception as e:
            self._handle_general_error(e, "get account info")

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: Union[str, Decimal],
    ) -> Dict[str, Any]:
        """
        Place a market order on Binance.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity

        Returns:
            Dict[str, Any]: Order execution result

        Raises:
            BinanceOrderError: If order placement fails
        """
        self._ensure_connected()
        self._validate_order_params(symbol, side, quantity)

        try:
            order_result = self._client.order_market(
                symbol=symbol, side=side, quantity=str(quantity)
            )

            self._logger.info(
                f"Market order placed: {side} {quantity} {symbol}"
            )
            return order_result

        except BinanceAPIException as e:
            self._handle_order_error(e, "market", symbol, side, quantity)
        except Exception as e:
            self._handle_general_error(e, "place market order")

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: Union[str, Decimal],
        price: Union[str, Decimal],
    ) -> Dict[str, Any]:
        """
        Place a limit order on Binance.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity
            price: Order price

        Returns:
            Dict[str, Any]: Order execution result

        Raises:
            BinanceOrderError: If order placement fails
        """
        self._ensure_connected()
        self._validate_order_params(symbol, side, quantity, price)

        try:
            order_result = self._client.order_limit(
                symbol=symbol,
                side=side,
                quantity=str(quantity),
                price=str(price),
            )

            self._logger.info(
                f"Limit order placed: {side} {quantity} {symbol} @ {price}"
            )
            return order_result

        except BinanceAPIException as e:
            self._handle_order_error(e, "limit", symbol, side, quantity, price)
        except Exception as e:
            self._handle_general_error(e, "place limit order")

    def get_symbol_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get symbol ticker information from Binance.

        Args:
            symbol: Trading pair symbol

        Returns:
            Dict[str, Any]: Ticker information including price

        Raises:
            BinanceError: If ticker retrieval fails
        """
        self._ensure_connected()
        self._validate_symbol(symbol)

        try:
            ticker = self._client.get_symbol_ticker(symbol=symbol)
            self._logger.debug(f"Ticker retrieved for {symbol}")
            return ticker

        except BinanceAPIException as e:
            self._handle_api_exception(e, f"get ticker for {symbol}")
        except Exception as e:
            self._handle_general_error(e, f"get ticker for {symbol}")

    def is_connected(self) -> bool:
        """
        Check if client is connected and authenticated.

        Returns:
            bool: True if connected, False otherwise
        """
        return self._is_connected and self._client is not None

    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get exchange information for symbols.

        Args:
            symbol: Optional specific symbol to get info for

        Returns:
            Dict[str, Any]: Exchange information

        Raises:
            BinanceError: If exchange info retrieval fails
        """
        self._ensure_connected()

        try:
            if symbol:
                exchange_info = self._client.get_symbol_info(symbol)
            else:
                exchange_info = self._client.get_exchange_info()

            self._logger.debug("Exchange information retrieved")
            return exchange_info

        except BinanceAPIException as e:
            self._handle_api_exception(e, "get exchange info")
        except Exception as e:
            self._handle_general_error(e, "get exchange info")

    def _validate_connection(self) -> None:
        """
        Validate connection by making a test API call.

        Raises:
            BinanceAuthenticationError: If authentication fails
        """
        try:
            self._client.get_account()
        except BinanceAPIException as e:
            if e.code in [-2014, -1021]:  # Authentication errors
                raise BinanceAuthenticationError(
                    f"Authentication failed: {e.message}"
                )
            raise

    def _ensure_connected(self) -> None:
        """
        Ensure client is connected before operations.

        Raises:
            BinanceConnectionError: If not connected
        """
        if not self.is_connected():
            raise BinanceConnectionError(
                "Client not connected. Call initialize() first."
            )

    def _validate_symbol(self, symbol: str) -> None:
        """
        Validate trading symbol format.

        Args:
            symbol: Trading symbol to validate

        Raises:
            ValueError: If symbol format is invalid
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")

        if len(symbol) < 6:
            raise ValueError("Symbol must be at least 6 characters")

    def _validate_order_params(
        self,
        symbol: str,
        side: str,
        quantity: Union[str, Decimal],
        price: Optional[Union[str, Decimal]] = None,
    ) -> None:
        """
        Validate order parameters.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            price: Optional order price

        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_symbol(symbol)

        if side not in ["BUY", "SELL"]:
            raise ValueError("Side must be 'BUY' or 'SELL'")

        try:
            qty = Decimal(str(quantity))
            if qty <= 0:
                raise ValueError("Quantity must be positive")
        except (ValueError, TypeError, Exception) as e:
            if "positive" in str(e):
                raise e
            raise ValueError("Quantity must be a valid number")

        if price is not None:
            try:
                p = Decimal(str(price))
                if p <= 0:
                    raise ValueError("Price must be positive")
            except (ValueError, TypeError, Exception) as e:
                if "positive" in str(e):
                    raise e
                raise ValueError("Price must be a valid number")

    def _handle_connection_error(self, error: Exception) -> None:
        """
        Handle connection-related errors.

        Args:
            error: The original exception

        Raises:
            BinanceAuthenticationError: For auth failures
            BinanceConnectionError: For connection failures
        """
        if isinstance(error, BinanceAuthenticationError):
            # Re-raise authentication errors as-is
            raise error

        if isinstance(error, BinanceAPIException):
            if error.code in [-2014, -1021]:
                self._logger.error(f"Authentication failed: {error.message}")
                raise BinanceAuthenticationError(
                    f"Invalid API credentials: {error.message}"
                )

        self._logger.error(f"Connection failed: {error}")
        raise BinanceConnectionError(f"Failed to connect to Binance: {error}")

    def _handle_api_exception(
        self, error: BinanceAPIException, operation: str
    ) -> None:
        """
        Handle Binance API exceptions.

        Args:
            error: Binance API exception
            operation: Description of the operation

        Raises:
            BinanceRateLimitError: For rate limit errors
            BinanceError: For other API errors
        """
        self._logger.error(f"API error during {operation}: {error.message}")

        if error.code == -1003:  # Rate limit
            raise BinanceRateLimitError(f"Rate limit exceeded: {error.message}")

        raise BinanceError(f"API error during {operation}: {error.message}")

    def _handle_order_error(
        self,
        error: BinanceAPIException,
        order_type: str,
        symbol: str,
        side: str,
        quantity: Union[str, Decimal],
        price: Optional[Union[str, Decimal]] = None,
    ) -> None:
        """
        Handle order-specific errors.

        Args:
            error: Binance API exception
            order_type: Type of order (market/limit)
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            price: Optional order price

        Raises:
            BinanceOrderError: For order execution failures
        """
        order_details = f"{order_type} {side} {quantity} {symbol}"
        if price:
            order_details += f" @ {price}"

        self._logger.error(f"Order failed - {order_details}: {error.message}")
        raise BinanceOrderError(
            f"Failed to place {order_details}: {error.message}"
        )

    def _handle_general_error(self, error: Exception, operation: str) -> None:
        """
        Handle general exceptions.

        Args:
            error: The exception
            operation: Description of the operation

        Raises:
            BinanceError: For general errors
        """
        self._logger.error(f"Error during {operation}: {error}")
        raise BinanceError(f"Error during {operation}: {error}")


def create_binance_client(config_manager: ConfigManager) -> BinanceClient:
    """
    Factory function to create and initialize BinanceClient.

    Args:
        config_manager: Configuration manager with API credentials

    Returns:
        BinanceClient: Initialized Binance client

    Raises:
        BinanceError: If client creation or initialization fails
    """
    client = BinanceClient(config_manager)
    client.initialize()
    return client
