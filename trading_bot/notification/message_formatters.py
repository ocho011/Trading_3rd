"""
Event message formatting system for Discord notifications.

Provides a comprehensive set of message formatters for different trading bot events
following SOLID principles with dependency injection and factory patterns.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

from trading_bot.core.event_hub import EventType
from trading_bot.core.logger import get_module_logger
from trading_bot.execution.execution_engine import ExecutionResult, OrderStatus
from trading_bot.strategies.base_strategy import SignalType, TradingSignal


class MessageFormatterError(Exception):
    """Base exception for message formatter errors."""
    pass


class InvalidEventDataError(MessageFormatterError):
    """Exception raised when event data is malformed or missing required fields."""
    pass


class FormatterNotFoundError(MessageFormatterError):
    """Exception raised when no formatter is found for event type."""
    pass


@dataclass
class DiscordEmbed:
    """Discord embed data structure for rich message formatting."""

    title: str
    description: str
    color: int
    fields: List[Dict[str, Union[str, bool]]]
    timestamp: Optional[str] = None
    footer: Optional[Dict[str, str]] = None
    thumbnail: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert embed to Discord API format."""
        embed = {
            "title": self.title,
            "description": self.description,
            "color": self.color,
            "fields": self.fields,
        }

        if self.timestamp:
            embed["timestamp"] = self.timestamp
        if self.footer:
            embed["footer"] = self.footer
        if self.thumbnail:
            embed["thumbnail"] = self.thumbnail

        return embed


class DiscordColor:
    """Discord embed color constants."""

    GREEN = 0x00ff00      # Success, buy signals
    RED = 0xff0000        # Error, sell signals, cancellations
    BLUE = 0x0066cc       # System, connection events
    ORANGE = 0xff6600     # Warnings, risk events
    YELLOW = 0xffff00     # Hold signals, neutral events
    PURPLE = 0x9932cc     # Analysis, insights
    GRAY = 0x808080       # Information, logs


class IMessageFormatter(ABC):
    """Interface for event message formatters."""

    @abstractmethod
    def format_message(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format event data into Discord webhook message.

        Args:
            event_data: Event payload containing relevant information

        Returns:
            Dict[str, Any]: Discord webhook message with embeds

        Raises:
            InvalidEventDataError: If event data is malformed
        """
        pass

    @abstractmethod
    def validate_event_data(self, event_data: Dict[str, Any]) -> bool:
        """
        Validate that event data contains required fields.

        Args:
            event_data: Event payload to validate

        Returns:
            bool: True if data is valid

        Raises:
            InvalidEventDataError: If required fields are missing
        """
        pass


class FormatterUtils:
    """Utility functions for message formatting."""

    @staticmethod
    def format_price(price: float, decimals: int = 4) -> str:
        """Format price with appropriate decimal places."""
        return f"${price:,.{decimals}f}"

    @staticmethod
    def format_quantity(quantity: Union[float, Decimal], decimals: int = 8) -> str:
        """Format quantity with appropriate decimal places."""
        if isinstance(quantity, Decimal):
            quantity = float(quantity)
        return f"{quantity:,.{decimals}f}".rstrip('0').rstrip('.')

    @staticmethod
    def format_percentage(value: float, decimals: int = 2) -> str:
        """Format percentage value."""
        return f"{value:+.{decimals}f}%"

    @staticmethod
    def format_timestamp(timestamp: Optional[int] = None) -> str:
        """Format timestamp for Discord embed."""
        if timestamp is None:
            timestamp = int(time.time())
        elif timestamp > 1e12:  # Millisecond timestamp
            timestamp = int(timestamp / 1000)

        # Convert to ISO format for Discord
        import datetime
        return datetime.datetime.fromtimestamp(timestamp).isoformat()

    @staticmethod
    def get_signal_color(signal_type: SignalType) -> int:
        """Get Discord color based on signal type."""
        color_map = {
            SignalType.BUY: DiscordColor.GREEN,
            SignalType.STRONG_BUY: DiscordColor.GREEN,
            SignalType.SELL: DiscordColor.RED,
            SignalType.STRONG_SELL: DiscordColor.RED,
            SignalType.HOLD: DiscordColor.YELLOW,
        }
        return color_map.get(signal_type, DiscordColor.GRAY)

    @staticmethod
    def get_status_color(status: OrderStatus) -> int:
        """Get Discord color based on order status."""
        color_map = {
            OrderStatus.EXECUTED: DiscordColor.GREEN,
            OrderStatus.CANCELLED: DiscordColor.RED,
            OrderStatus.REJECTED: DiscordColor.RED,
            OrderStatus.FAILED: DiscordColor.RED,
            OrderStatus.PENDING_VALIDATION: DiscordColor.BLUE,
            OrderStatus.VALIDATED: DiscordColor.BLUE,
            OrderStatus.PENDING_EXECUTION: DiscordColor.BLUE,
            OrderStatus.EXECUTING: DiscordColor.ORANGE,
        }
        return color_map.get(status, DiscordColor.GRAY)


class OrderFilledMessageFormatter(IMessageFormatter):
    """Formatter for ORDER_FILLED events."""

    def __init__(self) -> None:
        """Initialize ORDER_FILLED message formatter."""
        self._logger = get_module_logger(__name__)

    def format_message(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format ORDER_FILLED event into Discord message."""
        self.validate_event_data(event_data)

        execution_result: ExecutionResult = event_data["execution_result"]
        order_request = execution_result.order_request
        signal = order_request.signal

        # Determine embed color based on signal type and status
        if execution_result.execution_status == OrderStatus.EXECUTED:
            color = FormatterUtils.get_signal_color(signal.signal_type)
        else:
            color = FormatterUtils.get_status_color(execution_result.execution_status)

        # Create embed fields
        fields = [
            {
                "name": "📊 Trading Pair",
                "value": signal.symbol,
                "inline": True
            },
            {
                "name": "📈 Signal Type",
                "value": signal.signal_type.value.upper(),
                "inline": True
            },
            {
                "name": "🎯 Status",
                "value": execution_result.execution_status.value.upper(),
                "inline": True
            },
            {
                "name": "💰 Filled Quantity",
                "value": FormatterUtils.format_quantity(
                    execution_result.filled_quantity
                ),
                "inline": True
            },
            {
                "name": "💵 Average Price",
                "value": FormatterUtils.format_price(
                    execution_result.average_fill_price or
                    execution_result.filled_price or 0
                ),
                "inline": True
            },
            {
                "name": "💸 Total Value",
                "value": FormatterUtils.format_price(
                    float(execution_result.filled_quantity) *
                    (execution_result.average_fill_price or
                     execution_result.filled_price or 0)
                ),
                "inline": True
            }
        ]

        # Add commission information if available
        if execution_result.commission_amount > 0:
            fields.append({
                "name": "🏷️ Commission",
                "value": (f"{execution_result.commission_amount:.6f} "
                          f"{execution_result.commission_asset}"),
                "inline": True
            })

        # Add slippage if available
        if execution_result.slippage_percentage is not None:
            fields.append({
                "name": "📉 Slippage",
                "value": FormatterUtils.format_percentage(
                    execution_result.slippage_percentage
                ),
                "inline": True
            })

        # Add strategy and confidence
        fields.extend([
            {
                "name": "🤖 Strategy",
                "value": signal.strategy_name,
                "inline": True
            },
            {
                "name": "🎯 Confidence",
                "value": f"{signal.confidence:.1%}",
                "inline": True
            }
        ])

        # Add order IDs if available
        if execution_result.exchange_order_id:
            fields.append({
                "name": "🔗 Order ID",
                "value": execution_result.exchange_order_id,
                "inline": False
            })

        embed = DiscordEmbed(
            title=f"🎉 Order {execution_result.execution_status.value.replace('_', ' ').title()}",
            description=f"**{signal.signal_type.value.upper()}** order for {signal.symbol}",
            color=color,
            fields=fields,
            timestamp=FormatterUtils.format_timestamp(execution_result.execution_timestamp),
            footer={"text": "Trading Bot Execution Engine"}
        )

        return {"embeds": [embed.to_dict()]}

    def validate_event_data(self, event_data: Dict[str, Any]) -> bool:
        """Validate ORDER_FILLED event data."""
        if "execution_result" not in event_data:
            raise InvalidEventDataError("Missing 'execution_result' in ORDER_FILLED event")

        execution_result = event_data["execution_result"]
        if not isinstance(execution_result, ExecutionResult):
            raise InvalidEventDataError("Invalid ExecutionResult type in ORDER_FILLED event")

        return True


class ErrorOccurredMessageFormatter(IMessageFormatter):
    """Formatter for ERROR_OCCURRED events."""

    def __init__(self) -> None:
        """Initialize ERROR_OCCURRED message formatter."""
        self._logger = get_module_logger(__name__)

    def format_message(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format ERROR_OCCURRED event into Discord message."""
        self.validate_event_data(event_data)

        error_info = event_data.get("error", {})
        error_message = error_info.get("message", "Unknown error occurred")
        error_type = error_info.get("type", "Unknown")
        component = event_data.get("component", "System")
        timestamp = event_data.get("timestamp", int(time.time() * 1000))

        fields = [
            {
                "name": "🚨 Error Type",
                "value": error_type,
                "inline": True
            },
            {
                "name": "🔧 Component",
                "value": component,
                "inline": True
            },
            {
                "name": "📝 Details",
                "value": (error_message[:1000] + "..." if len(error_message) > 1000
                          else error_message),
                "inline": False
            }
        ]

        # Add severity if available
        severity = error_info.get("severity")
        if severity:
            fields.insert(0, {
                "name": "⚠️ Severity",
                "value": severity.upper(),
                "inline": True
            })

        # Add recovery suggestion if available
        recovery = error_info.get("recovery_suggestion")
        if recovery:
            fields.append({
                "name": "💡 Suggested Action",
                "value": recovery,
                "inline": False
            })

        embed = DiscordEmbed(
            title="🚨 System Error Occurred",
            description=f"Error in {component}",
            color=DiscordColor.RED,
            fields=fields,
            timestamp=FormatterUtils.format_timestamp(timestamp),
            footer={"text": "Trading Bot Error Handler"}
        )

        return {"embeds": [embed.to_dict()]}

    def validate_event_data(self, event_data: Dict[str, Any]) -> bool:
        """Validate ERROR_OCCURRED event data."""
        if "error" not in event_data and "message" not in event_data:
            raise InvalidEventDataError("Missing error information in ERROR_OCCURRED event")

        return True


class ConnectionEventMessageFormatter(IMessageFormatter):
    """Formatter for CONNECTION_LOST and CONNECTION_RESTORED events."""

    def __init__(self) -> None:
        """Initialize connection event message formatter."""
        self._logger = get_module_logger(__name__)

    def format_message(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format connection event into Discord message."""
        self.validate_event_data(event_data)

        event_type = event_data["event_type"]
        service = event_data.get("service", "Exchange")
        timestamp = event_data.get("timestamp", int(time.time() * 1000))

        if event_type == EventType.CONNECTION_LOST:
            title = "🔌 Connection Lost"
            description = f"Lost connection to {service}"
            color = DiscordColor.RED
            icon = "❌"
        else:  # CONNECTION_RESTORED
            title = "✅ Connection Restored"
            description = f"Connection to {service} restored"
            color = DiscordColor.GREEN
            icon = "✅"

        fields = [
            {
                "name": "🌐 Service",
                "value": service,
                "inline": True
            },
            {
                "name": "📊 Status",
                "value": icon + " " + (
                    "Offline" if event_type == EventType.CONNECTION_LOST
                    else "Online"
                ),
                "inline": True
            }
        ]

        # Add additional connection details if available
        details = event_data.get("details")
        if details:
            fields.append({
                "name": "📋 Details",
                "value": str(details)[:500],
                "inline": False
            })

        embed = DiscordEmbed(
            title=title,
            description=description,
            color=color,
            fields=fields,
            timestamp=FormatterUtils.format_timestamp(timestamp),
            footer={"text": "Trading Bot Connection Monitor"}
        )

        return {"embeds": [embed.to_dict()]}

    def validate_event_data(self, event_data: Dict[str, Any]) -> bool:
        """Validate connection event data."""
        if "event_type" not in event_data:
            raise InvalidEventDataError("Missing 'event_type' in connection event")

        event_type = event_data["event_type"]
        valid_types = [EventType.CONNECTION_LOST, EventType.CONNECTION_RESTORED]
        if event_type not in valid_types:
            raise InvalidEventDataError(
                f"Invalid event type for connection formatter: {event_type}"
            )

        return True


class TradingSignalMessageFormatter(IMessageFormatter):
    """Formatter for TRADING_SIGNAL_GENERATED events."""

    def __init__(self) -> None:
        """Initialize TRADING_SIGNAL_GENERATED message formatter."""
        self._logger = get_module_logger(__name__)

    def format_message(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format TRADING_SIGNAL_GENERATED event into Discord message."""
        self.validate_event_data(event_data)

        trading_signal: TradingSignal = event_data["trading_signal"]

        # Get color based on signal type
        color = FormatterUtils.get_signal_color(trading_signal.signal_type)

        # Create signal type emoji
        signal_emoji = {
            SignalType.BUY: "📈",
            SignalType.STRONG_BUY: "🚀",
            SignalType.SELL: "📉",
            SignalType.STRONG_SELL: "💥",
            SignalType.HOLD: "⏸️"
        }.get(trading_signal.signal_type, "📊")

        fields = [
            {
                "name": "📊 Trading Pair",
                "value": trading_signal.symbol,
                "inline": True
            },
            {
                "name": "📈 Signal Type",
                "value": f"{signal_emoji} {trading_signal.signal_type.value.upper()}",
                "inline": True
            },
            {
                "name": "💪 Strength",
                "value": trading_signal.strength.value.upper(),
                "inline": True
            },
            {
                "name": "💰 Price",
                "value": FormatterUtils.format_price(trading_signal.price),
                "inline": True
            },
            {
                "name": "🎯 Confidence",
                "value": f"{trading_signal.confidence:.1%}",
                "inline": True
            },
            {
                "name": "🤖 Strategy",
                "value": trading_signal.strategy_name,
                "inline": True
            }
        ]

        # Add target price if available
        if trading_signal.target_price:
            fields.append({
                "name": "🎯 Target Price",
                "value": FormatterUtils.format_price(trading_signal.target_price),
                "inline": True
            })

        # Add stop loss if available
        if trading_signal.stop_loss:
            fields.append({
                "name": "🛑 Stop Loss",
                "value": FormatterUtils.format_price(trading_signal.stop_loss),
                "inline": True
            })

        # Add take profit if available
        if trading_signal.take_profit:
            fields.append({
                "name": "💰 Take Profit",
                "value": FormatterUtils.format_price(trading_signal.take_profit),
                "inline": True
            })

        # Add reasoning if available
        if trading_signal.reasoning:
            reasoning_text = (trading_signal.reasoning[:500] + "..."
                              if len(trading_signal.reasoning) > 500
                              else trading_signal.reasoning)
            fields.append({
                "name": "🧠 Analysis",
                "value": reasoning_text,
                "inline": False
            })

        embed = DiscordEmbed(
            title=f"{signal_emoji} Trading Signal Generated",
            description=(f"**{trading_signal.signal_type.value.upper()}** "
                         f"signal for {trading_signal.symbol}"),
            color=color,
            fields=fields,
            timestamp=FormatterUtils.format_timestamp(trading_signal.timestamp),
            footer={"text": f"Generated by {trading_signal.strategy_name}"}
        )

        return {"embeds": [embed.to_dict()]}

    def validate_event_data(self, event_data: Dict[str, Any]) -> bool:
        """Validate TRADING_SIGNAL_GENERATED event data."""
        if "trading_signal" not in event_data:
            raise InvalidEventDataError(
                "Missing 'trading_signal' in TRADING_SIGNAL_GENERATED event"
            )

        trading_signal = event_data["trading_signal"]
        if not isinstance(trading_signal, TradingSignal):
            raise InvalidEventDataError(
                "Invalid TradingSignal type in TRADING_SIGNAL_GENERATED event"
            )

        return True


class RiskLimitExceededMessageFormatter(IMessageFormatter):
    """Formatter for RISK_LIMIT_EXCEEDED events."""

    def __init__(self) -> None:
        """Initialize RISK_LIMIT_EXCEEDED message formatter."""
        self._logger = get_module_logger(__name__)

    def format_message(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format RISK_LIMIT_EXCEEDED event into Discord message."""
        self.validate_event_data(event_data)

        risk_info = event_data.get("risk_info", {})
        limit_type = risk_info.get("limit_type", "Unknown")
        current_value = risk_info.get("current_value", "N/A")
        limit_value = risk_info.get("limit_value", "N/A")
        symbol = event_data.get("symbol", "N/A")
        timestamp = event_data.get("timestamp", int(time.time() * 1000))

        fields = [
            {
                "name": "⚠️ Limit Type",
                "value": limit_type,
                "inline": True
            },
            {
                "name": "📊 Symbol",
                "value": symbol,
                "inline": True
            },
            {
                "name": "📈 Current Value",
                "value": str(current_value),
                "inline": True
            },
            {
                "name": "🚫 Limit Value",
                "value": str(limit_value),
                "inline": True
            }
        ]

        # Add percentage if both values are numeric
        try:
            current_num = float(current_value)
            limit_num = float(limit_value)
            if limit_num != 0:
                percentage = (current_num / limit_num) * 100
                fields.append({
                    "name": "📊 Percentage",
                    "value": f"{percentage:.1f}%",
                    "inline": True
                })
        except (ValueError, TypeError):
            pass

        # Add action taken if available
        action = event_data.get("action_taken")
        if action:
            fields.append({
                "name": "⚡ Action Taken",
                "value": action,
                "inline": False
            })

        # Add risk details if available
        details = event_data.get("details")
        if details:
            fields.append({
                "name": "📋 Details",
                "value": str(details)[:500],
                "inline": False
            })

        embed = DiscordEmbed(
            title="⚠️ Risk Limit Exceeded",
            description=f"Risk limit exceeded for {symbol}",
            color=DiscordColor.ORANGE,
            fields=fields,
            timestamp=FormatterUtils.format_timestamp(timestamp),
            footer={"text": "Trading Bot Risk Manager"}
        )

        return {"embeds": [embed.to_dict()]}

    def validate_event_data(self, event_data: Dict[str, Any]) -> bool:
        """Validate RISK_LIMIT_EXCEEDED event data."""
        if "risk_info" not in event_data and "limit_type" not in event_data:
            raise InvalidEventDataError(
                "Missing risk information in RISK_LIMIT_EXCEEDED event"
            )

        return True


class MessageFormatterFactory:
    """Factory for creating appropriate message formatters based on event type."""

    _formatters: Dict[str, IMessageFormatter] = {}

    @classmethod
    def get_formatter(cls, event_type: str) -> IMessageFormatter:
        """
        Get appropriate formatter for event type.

        Args:
            event_type: Type of event to format

        Returns:
            IMessageFormatter: Appropriate formatter instance

        Raises:
            FormatterNotFoundError: If no formatter exists for event type
        """
        if event_type not in cls._formatters:
            cls._initialize_formatters()

        formatter = cls._formatters.get(event_type)
        if formatter is None:
            raise FormatterNotFoundError(f"No formatter found for event type: {event_type}")

        return formatter

    @classmethod
    def _initialize_formatters(cls) -> None:
        """Initialize formatter instances."""
        cls._formatters = {
            EventType.ORDER_FILLED: OrderFilledMessageFormatter(),
            EventType.ERROR_OCCURRED: ErrorOccurredMessageFormatter(),
            EventType.CONNECTION_LOST: ConnectionEventMessageFormatter(),
            EventType.CONNECTION_RESTORED: ConnectionEventMessageFormatter(),
            EventType.TRADING_SIGNAL_GENERATED: TradingSignalMessageFormatter(),
            EventType.RISK_LIMIT_EXCEEDED: RiskLimitExceededMessageFormatter(),
        }

    @classmethod
    def get_supported_event_types(cls) -> List[str]:
        """
        Get list of supported event types.

        Returns:
            List[str]: List of supported event type constants
        """
        cls._initialize_formatters()
        return list(cls._formatters.keys())

    @classmethod
    def format_event_message(cls, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convenience method to format event message.

        Args:
            event_type: Type of event to format
            event_data: Event payload data

        Returns:
            Dict[str, Any]: Formatted Discord webhook message

        Raises:
            FormatterNotFoundError: If no formatter exists for event type
            InvalidEventDataError: If event data is invalid
        """
        formatter = cls.get_formatter(event_type)
        return formatter.format_message(event_data)


def create_message_formatter_factory() -> MessageFormatterFactory:
    """
    Factory function to create MessageFormatterFactory instance.

    Returns:
        MessageFormatterFactory: Configured factory instance
    """
    return MessageFormatterFactory()
