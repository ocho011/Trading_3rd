"""
Unit tests for BinanceClient wrapper class.

Tests cover initialization, error handling, order placement,
and integration with ConfigManager following dependency injection patterns.
"""

import unittest
from decimal import Decimal
from unittest.mock import Mock, patch

from binance.exceptions import BinanceAPIException

from trading_bot.core.config_manager import ConfigManager
from trading_bot.market_data.binance_client import (
    BinanceAuthenticationError,
    BinanceClient,
    BinanceConnectionError,
    BinanceError,
    BinanceOrderError,
    BinanceRateLimitError,
    IExchangeClient,
    create_binance_client,
)


def create_binance_api_exception(message: str, code: int) -> BinanceAPIException:
    """
    Helper function to create BinanceAPIException with proper format.

    Args:
        message: Error message
        code: Error code

    Returns:
        BinanceAPIException: Properly formatted exception
    """
    mock_response = Mock()
    mock_response.json.return_value = {"code": code, "msg": message}
    mock_response.text = f'{{"code": {code}, "msg": "{message}"}}'

    exception = BinanceAPIException(mock_response, 400, message)
    exception.code = code
    exception.message = message

    return exception


class TestBinanceClientInterface(unittest.TestCase):
    """Test that BinanceClient properly implements IExchangeClient interface."""

    def test_interface_implementation(self) -> None:
        """Test that BinanceClient implements all required interface methods."""
        # Verify BinanceClient is a subclass of IExchangeClient
        self.assertTrue(issubclass(BinanceClient, IExchangeClient))

        # Verify all abstract methods are implemented
        config_mock = Mock(spec=ConfigManager)
        client = BinanceClient(config_mock)

        # Check that all abstract methods exist and are callable
        self.assertTrue(hasattr(client, "get_account_info"))
        self.assertTrue(callable(getattr(client, "get_account_info")))

        self.assertTrue(hasattr(client, "place_market_order"))
        self.assertTrue(callable(getattr(client, "place_market_order")))

        self.assertTrue(hasattr(client, "place_limit_order"))
        self.assertTrue(callable(getattr(client, "place_limit_order")))

        self.assertTrue(hasattr(client, "get_symbol_ticker"))
        self.assertTrue(callable(getattr(client, "get_symbol_ticker")))

        self.assertTrue(hasattr(client, "is_connected"))
        self.assertTrue(callable(getattr(client, "is_connected")))


class TestBinanceClientInitialization(unittest.TestCase):
    """Test BinanceClient initialization and dependency injection."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config_mock = Mock(spec=ConfigManager)
        self.config_mock.get_api_credentials.return_value = {
            "api_key": "test_key",
            "secret_key": "test_secret",
        }
        self.config_mock.get_trading_config.return_value = {"trading_mode": "paper"}

    def test_init_with_config_manager(self) -> None:
        """Test initialization with ConfigManager dependency injection."""
        client = BinanceClient(self.config_mock)

        self.assertEqual(client._config_manager, self.config_mock)
        self.assertIsNone(client._client)
        self.assertFalse(client._is_connected)
        self.assertFalse(client._is_testnet)

    @patch("trading_bot.market_data.binance_client.Client")
    def test_initialize_success_testnet(self, mock_client_class: Mock) -> None:
        """Test successful initialization in testnet mode."""
        mock_client = Mock()
        mock_client.get_account.return_value = {"status": "TRADING"}
        mock_client_class.return_value = mock_client

        client = BinanceClient(self.config_mock)
        client.initialize()

        # Verify client creation with correct parameters
        mock_client_class.assert_called_once_with(
            api_key="test_key", api_secret="test_secret", testnet=True
        )

        # Verify connection validation
        mock_client.get_account.assert_called_once()

        # Verify state updates
        self.assertTrue(client._is_connected)
        self.assertTrue(client._is_testnet)
        self.assertEqual(client._client, mock_client)

    @patch("trading_bot.market_data.binance_client.Client")
    def test_initialize_success_mainnet(self, mock_client_class: Mock) -> None:
        """Test successful initialization in mainnet mode."""
        self.config_mock.get_trading_config.return_value = {"trading_mode": "live"}

        mock_client = Mock()
        mock_client.get_account.return_value = {"status": "TRADING"}
        mock_client_class.return_value = mock_client

        client = BinanceClient(self.config_mock)
        client.initialize()

        # Verify client creation with mainnet settings
        mock_client_class.assert_called_once_with(
            api_key="test_key", api_secret="test_secret", testnet=False
        )

        self.assertTrue(client._is_connected)
        self.assertFalse(client._is_testnet)

    @patch("trading_bot.market_data.binance_client.Client")
    def test_initialize_authentication_error(self, mock_client_class: Mock) -> None:
        """Test initialization with authentication error."""
        mock_client = Mock()
        mock_client.get_account.side_effect = create_binance_api_exception(
            "Invalid API-key", -2014
        )
        mock_client_class.return_value = mock_client

        client = BinanceClient(self.config_mock)

        with self.assertRaises(BinanceAuthenticationError) as context:
            client.initialize()

        self.assertIn("Authentication failed", str(context.exception))
        self.assertFalse(client._is_connected)

    @patch("trading_bot.market_data.binance_client.Client")
    def test_initialize_connection_error(self, mock_client_class: Mock) -> None:
        """Test initialization with general connection error."""
        mock_client_class.side_effect = Exception("Network error")

        client = BinanceClient(self.config_mock)

        with self.assertRaises(BinanceConnectionError) as context:
            client.initialize()

        self.assertIn("Failed to connect to Binance", str(context.exception))
        self.assertFalse(client._is_connected)


class TestBinanceClientOperations(unittest.TestCase):
    """Test BinanceClient trading operations."""

    def setUp(self) -> None:
        """Set up test fixtures with initialized client."""
        self.config_mock = Mock(spec=ConfigManager)
        self.config_mock.get_api_credentials.return_value = {
            "api_key": "test_key",
            "secret_key": "test_secret",
        }
        self.config_mock.get_trading_config.return_value = {"trading_mode": "paper"}

        self.client = BinanceClient(self.config_mock)
        self.mock_binance_client = Mock()
        self.client._client = self.mock_binance_client
        self.client._is_connected = True

    def test_get_account_info_success(self) -> None:
        """Test successful account info retrieval."""
        expected_account = {"status": "TRADING", "balances": []}
        self.mock_binance_client.get_account.return_value = expected_account

        result = self.client.get_account_info()

        self.assertEqual(result, expected_account)
        self.mock_binance_client.get_account.assert_called_once()

    def test_get_account_info_not_connected(self) -> None:
        """Test account info retrieval when not connected."""
        self.client._is_connected = False

        with self.assertRaises(BinanceConnectionError) as context:
            self.client.get_account_info()

        self.assertIn("not connected", str(context.exception))

    def test_get_account_info_api_error(self) -> None:
        """Test account info retrieval with API error."""
        self.mock_binance_client.get_account.side_effect = create_binance_api_exception(
            "API error", -1000
        )

        with self.assertRaises(BinanceError) as context:
            self.client.get_account_info()

        self.assertIn("API error", str(context.exception))

    def test_place_market_order_success(self) -> None:
        """Test successful market order placement."""
        expected_order = {"orderId": 123, "status": "FILLED"}
        self.mock_binance_client.order_market.return_value = expected_order

        result = self.client.place_market_order("BTCUSDT", "BUY", "0.001")

        self.assertEqual(result, expected_order)
        self.mock_binance_client.order_market.assert_called_once_with(
            symbol="BTCUSDT", side="BUY", quantity="0.001"
        )

    def test_place_market_order_with_decimal(self) -> None:
        """Test market order placement with Decimal quantity."""
        expected_order = {"orderId": 123, "status": "FILLED"}
        self.mock_binance_client.order_market.return_value = expected_order

        quantity = Decimal("0.001")
        result = self.client.place_market_order("BTCUSDT", "BUY", quantity)

        self.assertEqual(result, expected_order)
        self.mock_binance_client.order_market.assert_called_once_with(
            symbol="BTCUSDT", side="BUY", quantity="0.001"
        )

    def test_place_market_order_invalid_side(self) -> None:
        """Test market order placement with invalid side."""
        with self.assertRaises(ValueError) as context:
            self.client.place_market_order("BTCUSDT", "INVALID", "0.001")

        self.assertIn("Side must be", str(context.exception))

    def test_place_market_order_invalid_quantity(self) -> None:
        """Test market order placement with invalid quantity."""
        with self.assertRaises(ValueError) as context:
            self.client.place_market_order("BTCUSDT", "BUY", "-0.001")

        self.assertIn("Quantity must be positive", str(context.exception))

    def test_place_market_order_api_error(self) -> None:
        """Test market order placement with API error."""
        self.mock_binance_client.order_market.side_effect = (
            create_binance_api_exception("Insufficient balance", -2010)
        )

        with self.assertRaises(BinanceOrderError) as context:
            self.client.place_market_order("BTCUSDT", "BUY", "0.001")

        self.assertIn("Failed to place market", str(context.exception))

    def test_place_limit_order_success(self) -> None:
        """Test successful limit order placement."""
        expected_order = {"orderId": 456, "status": "NEW"}
        self.mock_binance_client.order_limit.return_value = expected_order

        result = self.client.place_limit_order("BTCUSDT", "SELL", "0.001", "50000")

        self.assertEqual(result, expected_order)
        self.mock_binance_client.order_limit.assert_called_once_with(
            symbol="BTCUSDT", side="SELL", quantity="0.001", price="50000"
        )

    def test_place_limit_order_with_decimals(self) -> None:
        """Test limit order placement with Decimal values."""
        expected_order = {"orderId": 456, "status": "NEW"}
        self.mock_binance_client.order_limit.return_value = expected_order

        quantity = Decimal("0.001")
        price = Decimal("50000.50")

        result = self.client.place_limit_order("BTCUSDT", "SELL", quantity, price)

        self.assertEqual(result, expected_order)
        self.mock_binance_client.order_limit.assert_called_once_with(
            symbol="BTCUSDT", side="SELL", quantity="0.001", price="50000.50"
        )

    def test_place_limit_order_invalid_price(self) -> None:
        """Test limit order placement with invalid price."""
        with self.assertRaises(ValueError) as context:
            self.client.place_limit_order("BTCUSDT", "BUY", "0.001", "0")

        self.assertIn("Price must be positive", str(context.exception))

    def test_get_symbol_ticker_success(self) -> None:
        """Test successful symbol ticker retrieval."""
        expected_ticker = {"symbol": "BTCUSDT", "price": "50000.00"}
        self.mock_binance_client.get_symbol_ticker.return_value = expected_ticker

        result = self.client.get_symbol_ticker("BTCUSDT")

        self.assertEqual(result, expected_ticker)
        self.mock_binance_client.get_symbol_ticker.assert_called_once_with(
            symbol="BTCUSDT"
        )

    def test_get_symbol_ticker_invalid_symbol(self) -> None:
        """Test ticker retrieval with invalid symbol."""
        with self.assertRaises(ValueError) as context:
            self.client.get_symbol_ticker("")

        self.assertIn("Symbol must be", str(context.exception))

    def test_get_exchange_info_success(self) -> None:
        """Test successful exchange info retrieval."""
        expected_info = {"timezone": "UTC", "symbols": []}
        self.mock_binance_client.get_exchange_info.return_value = expected_info

        result = self.client.get_exchange_info()

        self.assertEqual(result, expected_info)
        self.mock_binance_client.get_exchange_info.assert_called_once()

    def test_get_exchange_info_specific_symbol(self) -> None:
        """Test exchange info retrieval for specific symbol."""
        expected_info = {"symbol": "BTCUSDT", "status": "TRADING"}
        self.mock_binance_client.get_symbol_info.return_value = expected_info

        result = self.client.get_exchange_info("BTCUSDT")

        self.assertEqual(result, expected_info)
        self.mock_binance_client.get_symbol_info.assert_called_once_with("BTCUSDT")


class TestBinanceClientErrorHandling(unittest.TestCase):
    """Test BinanceClient error handling and custom exceptions."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config_mock = Mock(spec=ConfigManager)
        self.client = BinanceClient(self.config_mock)
        self.mock_binance_client = Mock()
        self.client._client = self.mock_binance_client
        self.client._is_connected = True

    def test_rate_limit_error_handling(self) -> None:
        """Test rate limit error handling."""
        self.mock_binance_client.get_account.side_effect = create_binance_api_exception(
            "Rate limit exceeded", -1003
        )

        with self.assertRaises(BinanceRateLimitError) as context:
            self.client.get_account_info()

        self.assertIn("Rate limit exceeded", str(context.exception))

    def test_general_error_handling(self) -> None:
        """Test general error handling."""
        self.mock_binance_client.get_account.side_effect = Exception("Network timeout")

        with self.assertRaises(BinanceError) as context:
            self.client.get_account_info()

        self.assertIn("Error during get account info", str(context.exception))

    def test_is_connected_property(self) -> None:
        """Test is_connected property behavior."""
        # Test connected state
        self.assertTrue(self.client.is_connected())

        # Test disconnected state
        self.client._is_connected = False
        self.assertFalse(self.client.is_connected())

        # Test no client state
        self.client._is_connected = True
        self.client._client = None
        self.assertFalse(self.client.is_connected())


class TestBinanceClientValidation(unittest.TestCase):
    """Test BinanceClient input validation methods."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config_mock = Mock(spec=ConfigManager)
        self.client = BinanceClient(self.config_mock)

    def test_validate_symbol_valid(self) -> None:
        """Test symbol validation with valid symbols."""
        # Should not raise any exceptions
        self.client._validate_symbol("BTCUSDT")
        self.client._validate_symbol("ETHBTC")
        self.client._validate_symbol("ADAUSDT")

    def test_validate_symbol_invalid(self) -> None:
        """Test symbol validation with invalid symbols."""
        with self.assertRaises(ValueError):
            self.client._validate_symbol("")

        with self.assertRaises(ValueError):
            self.client._validate_symbol("BTC")  # Too short

        with self.assertRaises(ValueError):
            self.client._validate_symbol(None)

        with self.assertRaises(ValueError):
            self.client._validate_symbol(123)

    def test_validate_order_params_valid(self) -> None:
        """Test order parameter validation with valid inputs."""
        # Should not raise any exceptions
        self.client._validate_order_params("BTCUSDT", "BUY", "0.001")
        self.client._validate_order_params("ETHUSDT", "SELL", Decimal("1.5"))
        self.client._validate_order_params("BTCUSDT", "BUY", "0.001", "50000")

    def test_validate_order_params_invalid(self) -> None:
        """Test order parameter validation with invalid inputs."""
        # Invalid side
        with self.assertRaises(ValueError):
            self.client._validate_order_params("BTCUSDT", "HOLD", "0.001")

        # Invalid quantity
        with self.assertRaises(ValueError):
            self.client._validate_order_params("BTCUSDT", "BUY", "0")

        with self.assertRaises(ValueError):
            self.client._validate_order_params("BTCUSDT", "BUY", "-1")

        with self.assertRaises(ValueError):
            self.client._validate_order_params("BTCUSDT", "BUY", "invalid")

        # Invalid price
        with self.assertRaises(ValueError):
            self.client._validate_order_params("BTCUSDT", "BUY", "0.001", "0")

        with self.assertRaises(ValueError):
            self.client._validate_order_params("BTCUSDT", "BUY", "0.001", "invalid")


class TestBinanceClientFactory(unittest.TestCase):
    """Test BinanceClient factory function."""

    @patch("trading_bot.market_data.binance_client.BinanceClient")
    def test_create_binance_client_success(self, mock_client_class: Mock) -> None:
        """Test successful client creation via factory function."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        config_mock = Mock(spec=ConfigManager)
        result = create_binance_client(config_mock)

        # Verify client creation and initialization
        mock_client_class.assert_called_once_with(config_mock)
        mock_client.initialize.assert_called_once()
        self.assertEqual(result, mock_client)

    @patch("trading_bot.market_data.binance_client.BinanceClient")
    def test_create_binance_client_initialization_error(
        self, mock_client_class: Mock
    ) -> None:
        """Test client creation with initialization error."""
        mock_client = Mock()
        mock_client.initialize.side_effect = BinanceAuthenticationError(
            "Invalid credentials"
        )
        mock_client_class.return_value = mock_client

        config_mock = Mock(spec=ConfigManager)

        with self.assertRaises(BinanceAuthenticationError):
            create_binance_client(config_mock)


if __name__ == "__main__":
    unittest.main()
