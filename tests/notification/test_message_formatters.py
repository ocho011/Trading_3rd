"""
Unit tests for message formatters module.

Tests all message formatter implementations and the factory pattern
for Discord notification formatting.
"""

import time
import unittest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from trading_bot.core.event_hub import EventType
from trading_bot.execution.execution_engine import (ExecutionResult,
                                                    FillDetail, OrderStatus)
from trading_bot.notification.message_formatters import (
    ConnectionEventMessageFormatter, DiscordColor,
    ErrorOccurredMessageFormatter, FormatterNotFoundError, FormatterUtils,
    InvalidEventDataError, MessageFormatterFactory,
    OrderFilledMessageFormatter, RiskLimitExceededMessageFormatter,
    TradingSignalMessageFormatter)
from trading_bot.risk_management.risk_manager import OrderRequest, OrderType
from trading_bot.strategies.base_strategy import (SignalStrength, SignalType,
                                                  TradingSignal)


class TestFormatterUtils(unittest.TestCase):
    """Test cases for FormatterUtils class."""

    def test_format_price(self):
        """Test price formatting with different decimal places."""
        self.assertEqual(FormatterUtils.format_price(123.456789), "$123.4568")
        self.assertEqual(FormatterUtils.format_price(123.456789, decimals=2), "$123.46")
        self.assertEqual(FormatterUtils.format_price(1000.5), "$1,000.5000")

    def test_format_quantity(self):
        """Test quantity formatting with trailing zero removal."""
        self.assertEqual(FormatterUtils.format_quantity(123.456789), "123.456789")
        self.assertEqual(FormatterUtils.format_quantity(123.0), "123")
        self.assertEqual(
            FormatterUtils.format_quantity(Decimal("123.45000000")), "123.45"
        )

    def test_format_percentage(self):
        """Test percentage formatting with sign."""
        self.assertEqual(FormatterUtils.format_percentage(5.678), "+5.68%")
        self.assertEqual(FormatterUtils.format_percentage(-2.345), "-2.35%")
        self.assertEqual(FormatterUtils.format_percentage(0), "+0.00%")

    def test_format_timestamp(self):
        """Test timestamp formatting for Discord."""
        # Test with None (current time)
        result = FormatterUtils.format_timestamp(None)
        self.assertIsInstance(result, str)
        self.assertIn("T", result)  # ISO format

        # Test with millisecond timestamp
        ms_timestamp = 1640995200000  # 2022-01-01 00:00:00 UTC in milliseconds
        result = FormatterUtils.format_timestamp(ms_timestamp)
        self.assertIn("2022", result)

        # Test with second timestamp
        s_timestamp = 1640995200  # 2022-01-01 00:00:00 UTC in seconds
        result = FormatterUtils.format_timestamp(s_timestamp)
        self.assertIn("2022", result)

    def test_get_signal_color(self):
        """Test signal color mapping."""
        self.assertEqual(
            FormatterUtils.get_signal_color(SignalType.BUY), DiscordColor.GREEN
        )
        self.assertEqual(
            FormatterUtils.get_signal_color(SignalType.STRONG_BUY), DiscordColor.GREEN
        )
        self.assertEqual(
            FormatterUtils.get_signal_color(SignalType.SELL), DiscordColor.RED
        )
        self.assertEqual(
            FormatterUtils.get_signal_color(SignalType.STRONG_SELL), DiscordColor.RED
        )
        self.assertEqual(
            FormatterUtils.get_signal_color(SignalType.HOLD), DiscordColor.YELLOW
        )

    def test_get_status_color(self):
        """Test order status color mapping."""
        self.assertEqual(
            FormatterUtils.get_status_color(OrderStatus.EXECUTED), DiscordColor.GREEN
        )
        self.assertEqual(
            FormatterUtils.get_status_color(OrderStatus.CANCELLED), DiscordColor.RED
        )
        self.assertEqual(
            FormatterUtils.get_status_color(OrderStatus.PENDING_EXECUTION),
            DiscordColor.BLUE,
        )
        self.assertEqual(
            FormatterUtils.get_status_color(OrderStatus.EXECUTING), DiscordColor.ORANGE
        )


class TestOrderFilledMessageFormatter(unittest.TestCase):
    """Test cases for OrderFilledMessageFormatter."""

    def setUp(self):
        """Set up test fixtures."""
        self.formatter = OrderFilledMessageFormatter()

        # Create mock TradingSignal
        self.mock_signal = TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            price=45000.0,
            timestamp=int(time.time()),
            strategy_name="TestStrategy",
            confidence=0.85,
            reasoning="Strong buy signal detected",
        )

        # Create mock OrderRequest
        self.mock_order_request = MagicMock(spec=OrderRequest)
        self.mock_order_request.signal = self.mock_signal
        self.mock_order_request.symbol = "BTCUSDT"
        self.mock_order_request.order_type = OrderType.MARKET
        self.mock_order_request.quantity = Decimal("0.1")

        # Create mock ExecutionResult
        self.mock_execution_result = ExecutionResult(
            order_request=self.mock_order_request,
            execution_status=OrderStatus.EXECUTED,
            filled_quantity=Decimal("0.1"),
            filled_price=45000.0,
            average_fill_price=45000.0,
            commission_amount=0.001,
            commission_asset="BNB",
            slippage_percentage=0.05,
            exchange_order_id="12345",
        )

    def test_format_message_success(self):
        """Test successful message formatting for ORDER_FILLED event."""
        event_data = {"execution_result": self.mock_execution_result}

        result = self.formatter.format_message(event_data)

        # Verify message structure
        self.assertIn("embeds", result)
        self.assertEqual(len(result["embeds"]), 1)

        embed = result["embeds"][0]
        self.assertIn("title", embed)
        self.assertIn("description", embed)
        self.assertIn("color", embed)
        self.assertIn("fields", embed)
        self.assertIn("timestamp", embed)

        # Verify content
        self.assertIn("Order Executed", embed["title"])
        self.assertIn("BTCUSDT", embed["description"])
        self.assertEqual(embed["color"], DiscordColor.GREEN)

        # Check specific fields
        field_names = [field["name"] for field in embed["fields"]]
        self.assertIn("📊 Trading Pair", field_names)
        self.assertIn("📈 Signal Type", field_names)
        self.assertIn("💰 Filled Quantity", field_names)
        self.assertIn("🎯 Confidence", field_names)

    def test_validate_event_data_missing_execution_result(self):
        """Test validation with missing execution result."""
        event_data = {}

        with self.assertRaises(InvalidEventDataError) as context:
            self.formatter.validate_event_data(event_data)

        self.assertIn("execution_result", str(context.exception))

    def test_validate_event_data_invalid_type(self):
        """Test validation with invalid execution result type."""
        event_data = {"execution_result": "invalid"}

        with self.assertRaises(InvalidEventDataError) as context:
            self.formatter.validate_event_data(event_data)

        self.assertIn("Invalid ExecutionResult type", str(context.exception))


class TestErrorOccurredMessageFormatter(unittest.TestCase):
    """Test cases for ErrorOccurredMessageFormatter."""

    def setUp(self):
        """Set up test fixtures."""
        self.formatter = ErrorOccurredMessageFormatter()

    def test_format_message_with_error_dict(self):
        """Test message formatting with error dictionary."""
        event_data = {
            "error": {
                "message": "Connection timeout",
                "type": "NetworkError",
                "severity": "high",
                "recovery_suggestion": "Check network connection",
            },
            "component": "ExchangeClient",
            "timestamp": int(time.time() * 1000),
        }

        result = self.formatter.format_message(event_data)

        embed = result["embeds"][0]
        self.assertIn("System Error", embed["title"])
        self.assertEqual(embed["color"], DiscordColor.RED)

        field_names = [field["name"] for field in embed["fields"]]
        self.assertIn("⚠️ Severity", field_names)
        self.assertIn("🚨 Error Type", field_names)
        self.assertIn("🔧 Component", field_names)
        self.assertIn("💡 Suggested Action", field_names)

    def test_format_message_minimal_data(self):
        """Test message formatting with minimal error data."""
        event_data = {"error": {"message": "Unknown error"}}

        result = self.formatter.format_message(event_data)

        embed = result["embeds"][0]
        self.assertIn("System Error", embed["title"])
        self.assertEqual(embed["color"], DiscordColor.RED)

    def test_validate_event_data_missing_error(self):
        """Test validation with missing error information."""
        event_data = {}

        with self.assertRaises(InvalidEventDataError) as context:
            self.formatter.validate_event_data(event_data)

        self.assertIn("Missing error information", str(context.exception))


class TestConnectionEventMessageFormatter(unittest.TestCase):
    """Test cases for ConnectionEventMessageFormatter."""

    def setUp(self):
        """Set up test fixtures."""
        self.formatter = ConnectionEventMessageFormatter()

    def test_format_connection_lost_message(self):
        """Test formatting CONNECTION_LOST event."""
        event_data = {
            "event_type": EventType.CONNECTION_LOST,
            "service": "Binance",
            "timestamp": int(time.time() * 1000),
            "details": "WebSocket connection timeout",
        }

        result = self.formatter.format_message(event_data)

        embed = result["embeds"][0]
        self.assertIn("Connection Lost", embed["title"])
        self.assertEqual(embed["color"], DiscordColor.RED)
        self.assertIn("Lost connection to Binance", embed["description"])

    def test_format_connection_restored_message(self):
        """Test formatting CONNECTION_RESTORED event."""
        event_data = {
            "event_type": EventType.CONNECTION_RESTORED,
            "service": "Binance",
            "timestamp": int(time.time() * 1000),
        }

        result = self.formatter.format_message(event_data)

        embed = result["embeds"][0]
        self.assertIn("Connection Restored", embed["title"])
        self.assertEqual(embed["color"], DiscordColor.GREEN)
        self.assertIn("Connection to Binance restored", embed["description"])

    def test_validate_event_data_missing_type(self):
        """Test validation with missing event type."""
        event_data = {"service": "Binance"}

        with self.assertRaises(InvalidEventDataError) as context:
            self.formatter.validate_event_data(event_data)

        self.assertIn("event_type", str(context.exception))

    def test_validate_event_data_invalid_type(self):
        """Test validation with invalid event type."""
        event_data = {"event_type": "INVALID_EVENT"}

        with self.assertRaises(InvalidEventDataError) as context:
            self.formatter.validate_event_data(event_data)

        self.assertIn("Invalid event type", str(context.exception))


class TestTradingSignalMessageFormatter(unittest.TestCase):
    """Test cases for TradingSignalMessageFormatter."""

    def setUp(self):
        """Set up test fixtures."""
        self.formatter = TradingSignalMessageFormatter()

        self.mock_signal = TradingSignal(
            symbol="ETHUSDT",
            signal_type=SignalType.STRONG_BUY,
            strength=SignalStrength.VERY_STRONG,
            price=3000.0,
            timestamp=int(time.time()),
            strategy_name="ICTStrategy",
            confidence=0.92,
            reasoning="Strong bullish momentum detected with high volume",
            target_price=3200.0,
            stop_loss=2800.0,
            take_profit=3500.0,
        )

    def test_format_message_complete_signal(self):
        """Test message formatting with complete trading signal."""
        event_data = {"trading_signal": self.mock_signal}

        result = self.formatter.format_message(event_data)

        embed = result["embeds"][0]
        self.assertIn("Trading Signal Generated", embed["title"])
        self.assertEqual(embed["color"], DiscordColor.GREEN)

        field_names = [field["name"] for field in embed["fields"]]
        self.assertIn("📊 Trading Pair", field_names)
        self.assertIn("🎯 Target Price", field_names)
        self.assertIn("🛑 Stop Loss", field_names)
        self.assertIn("💰 Take Profit", field_names)
        self.assertIn("🧠 Analysis", field_names)

    def test_format_message_minimal_signal(self):
        """Test message formatting with minimal trading signal."""
        minimal_signal = TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalType.HOLD,
            strength=SignalStrength.WEAK,
            price=45000.0,
            timestamp=int(time.time()),
            strategy_name="TestStrategy",
            confidence=0.5,
        )

        event_data = {"trading_signal": minimal_signal}
        result = self.formatter.format_message(event_data)

        embed = result["embeds"][0]
        self.assertEqual(embed["color"], DiscordColor.YELLOW)

        # Should not have optional fields
        field_names = [field["name"] for field in embed["fields"]]
        self.assertNotIn("🎯 Target Price", field_names)
        self.assertNotIn("🛑 Stop Loss", field_names)

    def test_validate_event_data_missing_signal(self):
        """Test validation with missing trading signal."""
        event_data = {}

        with self.assertRaises(InvalidEventDataError) as context:
            self.formatter.validate_event_data(event_data)

        self.assertIn("trading_signal", str(context.exception))


class TestRiskLimitExceededMessageFormatter(unittest.TestCase):
    """Test cases for RiskLimitExceededMessageFormatter."""

    def setUp(self):
        """Set up test fixtures."""
        self.formatter = RiskLimitExceededMessageFormatter()

    def test_format_message_complete_data(self):
        """Test message formatting with complete risk limit data."""
        event_data = {
            "risk_info": {
                "limit_type": "Position Size",
                "current_value": 15000.0,
                "limit_value": 10000.0,
            },
            "symbol": "BTCUSDT",
            "action_taken": "Order rejected",
            "details": "Position size exceeds maximum allowed",
            "timestamp": int(time.time() * 1000),
        }

        result = self.formatter.format_message(event_data)

        embed = result["embeds"][0]
        self.assertIn("Risk Limit Exceeded", embed["title"])
        self.assertEqual(embed["color"], DiscordColor.ORANGE)

        field_names = [field["name"] for field in embed["fields"]]
        self.assertIn("⚠️ Limit Type", field_names)
        self.assertIn("📊 Percentage", field_names)
        self.assertIn("⚡ Action Taken", field_names)

    def test_format_message_minimal_data(self):
        """Test message formatting with minimal risk data."""
        event_data = {"risk_info": {"limit_type": "Unknown"}}

        result = self.formatter.format_message(event_data)

        embed = result["embeds"][0]
        self.assertIn("Risk Limit Exceeded", embed["title"])
        self.assertEqual(embed["color"], DiscordColor.ORANGE)

    def test_validate_event_data_missing_info(self):
        """Test validation with missing risk information."""
        event_data = {}

        with self.assertRaises(InvalidEventDataError) as context:
            self.formatter.validate_event_data(event_data)

        self.assertIn("Missing risk information", str(context.exception))


class TestMessageFormatterFactory(unittest.TestCase):
    """Test cases for MessageFormatterFactory."""

    def test_get_formatter_valid_types(self):
        """Test getting formatters for valid event types."""
        # Test all supported types
        supported_types = [
            EventType.ORDER_FILLED,
            EventType.ERROR_OCCURRED,
            EventType.CONNECTION_LOST,
            EventType.CONNECTION_RESTORED,
            EventType.TRADING_SIGNAL_GENERATED,
            EventType.RISK_LIMIT_EXCEEDED,
        ]

        for event_type in supported_types:
            formatter = MessageFormatterFactory.get_formatter(event_type)
            self.assertIsNotNone(formatter)

    def test_get_formatter_invalid_type(self):
        """Test getting formatter for invalid event type."""
        with self.assertRaises(FormatterNotFoundError) as context:
            MessageFormatterFactory.get_formatter("INVALID_EVENT_TYPE")

        self.assertIn("No formatter found", str(context.exception))

    def test_get_supported_event_types(self):
        """Test getting list of supported event types."""
        supported_types = MessageFormatterFactory.get_supported_event_types()

        expected_types = [
            EventType.ORDER_FILLED,
            EventType.ERROR_OCCURRED,
            EventType.CONNECTION_LOST,
            EventType.CONNECTION_RESTORED,
            EventType.TRADING_SIGNAL_GENERATED,
            EventType.RISK_LIMIT_EXCEEDED,
        ]

        for event_type in expected_types:
            self.assertIn(event_type, supported_types)

    def test_format_event_message_convenience_method(self):
        """Test convenience method for formatting messages."""
        # Create mock signal and event data
        mock_signal = TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.MODERATE,
            price=45000.0,
            timestamp=int(time.time()),
            strategy_name="TestStrategy",
            confidence=0.75,
        )

        event_data = {"trading_signal": mock_signal}

        result = MessageFormatterFactory.format_event_message(
            EventType.TRADING_SIGNAL_GENERATED, event_data
        )

        self.assertIn("embeds", result)
        self.assertEqual(len(result["embeds"]), 1)

    def test_format_event_message_invalid_data(self):
        """Test convenience method with invalid data."""
        event_data = {"invalid": "data"}

        with self.assertRaises(InvalidEventDataError):
            MessageFormatterFactory.format_event_message(
                EventType.TRADING_SIGNAL_GENERATED, event_data
            )


class TestDiscordEmbedDataClass(unittest.TestCase):
    """Test cases for DiscordEmbed data class."""

    def test_embed_to_dict(self):
        """Test converting embed to Discord API format."""
        from trading_bot.notification.message_formatters import DiscordEmbed

        embed = DiscordEmbed(
            title="Test Title",
            description="Test Description",
            color=DiscordColor.BLUE,
            fields=[{"name": "Field", "value": "Value", "inline": True}],
            timestamp="2022-01-01T00:00:00",
            footer={"text": "Footer"},
            thumbnail={"url": "https://example.com/image.png"},
        )

        result = embed.to_dict()

        expected_keys = [
            "title",
            "description",
            "color",
            "fields",
            "timestamp",
            "footer",
            "thumbnail",
        ]
        for key in expected_keys:
            self.assertIn(key, result)

        self.assertEqual(result["title"], "Test Title")
        self.assertEqual(result["color"], DiscordColor.BLUE)
        self.assertEqual(len(result["fields"]), 1)

    def test_embed_to_dict_minimal(self):
        """Test converting minimal embed to dict."""
        from trading_bot.notification.message_formatters import DiscordEmbed

        embed = DiscordEmbed(
            title="Test", description="Test", color=DiscordColor.GREEN, fields=[]
        )

        result = embed.to_dict()

        # Should only have required fields
        self.assertIn("title", result)
        self.assertIn("description", result)
        self.assertIn("color", result)
        self.assertIn("fields", result)
        self.assertNotIn("timestamp", result)
        self.assertNotIn("footer", result)
        self.assertNotIn("thumbnail", result)


if __name__ == "__main__":
    unittest.main()
