"""
Comprehensive tests for EventHub integration functionality in DiscordNotifier.

Tests cover event subscription management, event handler methods, lifecycle management,
integration scenarios, error handling, and end-to-end EventHub communication.
"""

import asyncio
import pytest
import time
from decimal import Decimal
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch, MagicMock, call

from trading_bot.core.config_manager import ConfigManager, IConfigLoader
from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.execution.execution_engine import ExecutionResult, OrderStatus, FillDetail
from trading_bot.notification.discord_notifier import (
    DiscordNotifier,
    DiscordNotificationError,
    IHttpClient,
)
from trading_bot.notification.message_formatters import (
    MessageFormatterFactory,
    MessageFormatterError,
    InvalidEventDataError,
)
from trading_bot.risk_management.risk_manager import OrderRequest, OrderType
from trading_bot.strategies.base_strategy import TradingSignal, SignalType, SignalStrength


class MockConfigLoader(IConfigLoader):
    """Mock config loader for testing."""

    def __init__(self, webhook_url: str = "https://discord.com/api/webhooks/test") -> None:
        self.webhook_url = webhook_url

    def load_config(self) -> Dict[str, Any]:
        """Load mock configuration."""
        return {
            "discord_webhook_url": self.webhook_url,
            "log_level": "INFO",
        }


class MockHttpClient(IHttpClient):
    """Mock HTTP client for testing."""

    def __init__(self, should_fail: bool = False, fail_with: Exception = None) -> None:
        self.should_fail = should_fail
        self.fail_with = fail_with or DiscordNotificationError("Mock error")
        self.post_async_calls = []
        self.post_sync_calls = []

    async def post_async(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock async POST request."""
        self.post_async_calls.append({"url": url, "data": data})
        if self.should_fail:
            raise self.fail_with
        return {"status": "success", "status_code": 204}

    def post_sync(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock sync POST request."""
        self.post_sync_calls.append({"url": url, "data": data})
        if self.should_fail:
            raise self.fail_with
        return {"status": "success", "status_code": 204}


class MockMessageFormatterFactory(MessageFormatterFactory):
    """Mock message formatter factory for testing."""

    def __init__(self, should_fail: bool = False) -> None:
        super().__init__()
        self.should_fail = should_fail
        self.format_calls = []

    def get_formatter(self, event_type: str):
        """Get mock formatter."""
        mock_formatter = Mock()
        if self.should_fail:
            mock_formatter.format_message.side_effect = MessageFormatterError("Mock formatter error")
        else:
            mock_formatter.format_message.return_value = {
                "content": f"Formatted message for {event_type}",
                "embeds": [{
                    "title": f"Event: {event_type}",
                    "description": "Mock formatted event",
                    "color": 0x00ff00
                }]
            }
        self.format_calls.append(event_type)
        return mock_formatter


@pytest.fixture
def config_manager():
    """Create config manager for testing."""
    loader = MockConfigLoader()
    manager = ConfigManager(loader)
    manager.load_configuration()
    return manager


@pytest.fixture
def config_manager_no_webhook():
    """Create config manager without webhook URL."""
    loader = MockConfigLoader(webhook_url="")
    manager = ConfigManager(loader)
    manager.load_configuration()
    return manager


@pytest.fixture
def mock_http_client():
    """Create mock HTTP client."""
    return MockHttpClient()


@pytest.fixture
def failing_http_client():
    """Create failing HTTP client."""
    return MockHttpClient(should_fail=True)


@pytest.fixture
def mock_event_hub():
    """Create mock EventHub."""
    return Mock(spec=EventHub)


@pytest.fixture
def real_event_hub():
    """Create real EventHub for integration tests."""
    return EventHub()


@pytest.fixture
def mock_message_formatter():
    """Create mock message formatter factory."""
    return MockMessageFormatterFactory()


@pytest.fixture
def failing_message_formatter():
    """Create failing message formatter factory."""
    return MockMessageFormatterFactory(should_fail=True)


@pytest.fixture
def sample_trading_signal():
    """Create sample trading signal for testing."""
    return TradingSignal(
        symbol="BTCUSDT",
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        price=45000.50,
        timestamp=int(time.time() * 1000),
        strategy_name="TestStrategy",
        confidence=0.85,
        reasoning="Strong bullish momentum detected",
        target_price=47000.00,
        stop_loss=43000.00,
        take_profit=46500.00
    )


@pytest.fixture
def sample_order_request(sample_trading_signal):
    """Create sample order request for testing."""
    # Create mock objects for complex dependencies
    mock_position_sizing = Mock()
    mock_risk_assessment = Mock()
    mock_stop_loss = Mock()

    return OrderRequest(
        signal=sample_trading_signal,
        symbol="BTCUSDT",
        order_type=OrderType.MARKET,
        quantity=Decimal("0.001"),
        price=45000.50,
        position_size_result=mock_position_sizing,
        risk_assessment_result=mock_risk_assessment,
        stop_loss_result=mock_stop_loss,
        account_risk_result=None,
        entry_price=45000.50,
        stop_loss_price=43000.00,
        take_profit_price=46500.00
    )


@pytest.fixture
def sample_execution_result(sample_order_request):
    """Create sample execution result for testing."""
    return ExecutionResult(
        order_request=sample_order_request,
        execution_status=OrderStatus.EXECUTED,
        exchange_order_id="12345",
        client_order_id="client_123",
        filled_quantity=Decimal("0.001"),
        filled_price=45100.25,
        average_fill_price=45100.25,
        commission_amount=0.45,
        commission_asset="USDT",
        market_price_at_execution=45100.00,
        slippage_percentage=0.22,
        execution_timestamp=int(time.time() * 1000)
    )


class TestEventHubSubscription:
    """Test cases for EventHub subscription functionality."""

    def test_subscribe_to_events_with_valid_eventhub(
        self, config_manager, mock_http_client, mock_event_hub, mock_message_formatter
    ):
        """Test subscribing to events with valid EventHub."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, mock_event_hub, mock_message_formatter
        )

        notifier.subscribe_to_events()

        # Verify all expected event types were subscribed
        expected_events = [
            EventType.ORDER_FILLED,
            EventType.ERROR_OCCURRED,
            EventType.CONNECTION_LOST,
            EventType.CONNECTION_RESTORED,
            EventType.TRADING_SIGNAL_GENERATED,
            EventType.RISK_LIMIT_EXCEEDED,
        ]

        assert mock_event_hub.subscribe.call_count == len(expected_events)

        # Verify each event type was subscribed with correct handler
        subscribed_events = [call.args[0] for call in mock_event_hub.subscribe.call_args_list]
        assert set(subscribed_events) == set(expected_events)

        # Verify subscription state
        assert notifier._subscriptions_active is True
        assert len(notifier._event_handlers) == len(expected_events)

    def test_subscribe_to_events_without_eventhub_raises_error(
        self, config_manager, mock_http_client
    ):
        """Test subscribing to events without EventHub raises error."""
        notifier = DiscordNotifier(config_manager, mock_http_client)

        with pytest.raises(DiscordNotificationError) as excinfo:
            notifier.subscribe_to_events()

        assert "EventHub not configured" in str(excinfo.value)

    def test_subscribe_to_events_already_subscribed(
        self, config_manager, mock_http_client, mock_event_hub, mock_message_formatter
    ):
        """Test subscribing when already subscribed shows warning."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, mock_event_hub, mock_message_formatter
        )

        # First subscription
        notifier.subscribe_to_events()
        initial_call_count = mock_event_hub.subscribe.call_count

        # Second subscription attempt
        with patch.object(notifier._logger, 'warning') as mock_warning:
            notifier.subscribe_to_events()
            mock_warning.assert_called_once_with("EventHub subscriptions already active")

        # Verify no additional subscriptions were made
        assert mock_event_hub.subscribe.call_count == initial_call_count

    def test_subscribe_to_events_with_exception(
        self, config_manager, mock_http_client, mock_event_hub, mock_message_formatter
    ):
        """Test subscribing to events handles exceptions properly."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, mock_event_hub, mock_message_formatter
        )

        mock_event_hub.subscribe.side_effect = Exception("Subscription failed")

        with pytest.raises(DiscordNotificationError) as excinfo:
            notifier.subscribe_to_events()

        assert "Failed to subscribe to EventHub events" in str(excinfo.value)
        assert "Subscription failed" in str(excinfo.value)

    def test_unsubscribe_from_events_successful(
        self, config_manager, mock_http_client, mock_event_hub, mock_message_formatter
    ):
        """Test successful unsubscription from events."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, mock_event_hub, mock_message_formatter
        )

        # First subscribe
        notifier.subscribe_to_events()
        initial_handlers = notifier._event_handlers.copy()

        # Then unsubscribe
        notifier.unsubscribe_from_events()

        # Verify unsubscription calls
        assert mock_event_hub.unsubscribe.call_count == len(initial_handlers)

        # Verify subscription state cleared
        assert notifier._subscriptions_active is False
        assert len(notifier._event_handlers) == 0

    def test_unsubscribe_from_events_without_active_subscriptions(
        self, config_manager, mock_http_client, mock_event_hub, mock_message_formatter
    ):
        """Test unsubscribing without active subscriptions."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, mock_event_hub, mock_message_formatter
        )

        with patch.object(notifier._logger, 'debug') as mock_debug:
            notifier.unsubscribe_from_events()
            mock_debug.assert_called_once_with("No active EventHub subscriptions to remove")

    def test_unsubscribe_from_events_without_eventhub(
        self, config_manager, mock_http_client
    ):
        """Test unsubscribing without EventHub."""
        notifier = DiscordNotifier(config_manager, mock_http_client)

        with patch.object(notifier._logger, 'debug') as mock_debug:
            notifier.unsubscribe_from_events()
            mock_debug.assert_called_once_with("No active EventHub subscriptions to remove")

    def test_unsubscribe_handles_keyerror(
        self, config_manager, mock_http_client, mock_event_hub, mock_message_formatter
    ):
        """Test unsubscribing handles KeyError for missing handlers."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, mock_event_hub, mock_message_formatter
        )

        # Subscribe first
        notifier.subscribe_to_events()

        # Mock unsubscribe to raise KeyError for one handler
        def side_effect(event_type, handler):
            if event_type == EventType.ORDER_FILLED:
                raise KeyError("Handler not found")

        mock_event_hub.unsubscribe.side_effect = side_effect

        with patch.object(notifier._logger, 'warning') as mock_warning:
            notifier.unsubscribe_from_events()
            mock_warning.assert_called()

    def test_unsubscribe_handles_general_exception(
        self, config_manager, mock_http_client, mock_event_hub, mock_message_formatter
    ):
        """Test unsubscribing handles general exceptions."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, mock_event_hub, mock_message_formatter
        )

        # Subscribe first
        notifier.subscribe_to_events()

        # Mock unsubscribe to raise general exception
        mock_event_hub.unsubscribe.side_effect = Exception("General error")

        with patch.object(notifier._logger, 'error') as mock_error:
            notifier.unsubscribe_from_events()
            mock_error.assert_called()


class TestEventHandlers:
    """Test cases for individual event handler methods."""

    @pytest.mark.asyncio
    async def test_handle_order_filled_async_success(
        self, config_manager, mock_http_client, mock_message_formatter, sample_execution_result
    ):
        """Test successful ORDER_FILLED event handling."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, None, mock_message_formatter
        )

        event_data = {"execution_result": sample_execution_result}
        await notifier._handle_order_filled_async(event_data)

        # Verify message formatter was called
        assert EventType.ORDER_FILLED in mock_message_formatter.format_calls

        # Verify Discord message was sent
        assert len(mock_http_client.post_async_calls) == 1
        call_data = mock_http_client.post_async_calls[0]
        assert call_data["data"]["content"] == f"Formatted message for {EventType.ORDER_FILLED}"
        assert len(call_data["data"]["embeds"]) == 1

    @pytest.mark.asyncio
    async def test_handle_order_filled_async_formatter_error(
        self, config_manager, mock_http_client, failing_message_formatter, sample_execution_result
    ):
        """Test ORDER_FILLED event handling with formatter error."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, None, failing_message_formatter
        )

        event_data = {"execution_result": sample_execution_result}

        with patch.object(notifier._logger, 'error') as mock_error:
            await notifier._handle_order_filled_async(event_data)
            mock_error.assert_called()

    @pytest.mark.asyncio
    async def test_handle_error_occurred_async_success(
        self, config_manager, mock_http_client, mock_message_formatter
    ):
        """Test successful ERROR_OCCURRED event handling."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, None, mock_message_formatter
        )

        event_data = {
            "error": {
                "message": "Database connection failed",
                "type": "ConnectionError",
                "severity": "high"
            },
            "component": "DatabaseManager",
            "timestamp": int(time.time() * 1000)
        }

        await notifier._handle_error_occurred_async(event_data)

        # Verify message formatter was called
        assert EventType.ERROR_OCCURRED in mock_message_formatter.format_calls

        # Verify Discord message was sent
        assert len(mock_http_client.post_async_calls) == 1

    @pytest.mark.asyncio
    async def test_handle_connection_event_async_success(
        self, config_manager, mock_http_client, mock_message_formatter
    ):
        """Test successful connection event handling."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, None, mock_message_formatter
        )

        event_data = {
            "event_type": EventType.CONNECTION_LOST,
            "service": "Binance",
            "timestamp": int(time.time() * 1000),
            "details": "WebSocket connection timeout"
        }

        await notifier._handle_connection_event_async(event_data)

        # Verify message formatter was called
        assert EventType.CONNECTION_LOST in mock_message_formatter.format_calls

        # Verify Discord message was sent
        assert len(mock_http_client.post_async_calls) == 1

    @pytest.mark.asyncio
    async def test_handle_connection_event_async_missing_event_type(
        self, config_manager, mock_http_client, mock_message_formatter
    ):
        """Test connection event handling with missing event_type."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, None, mock_message_formatter
        )

        event_data = {
            "service": "Binance",
            "timestamp": int(time.time() * 1000)
        }

        with patch.object(notifier._logger, 'warning') as mock_warning:
            await notifier._handle_connection_event_async(event_data)
            mock_warning.assert_called_once_with("Connection event missing event_type field")

    @pytest.mark.asyncio
    async def test_handle_trading_signal_async_success(
        self, config_manager, mock_http_client, mock_message_formatter, sample_trading_signal
    ):
        """Test successful TRADING_SIGNAL_GENERATED event handling."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, None, mock_message_formatter
        )

        event_data = {"trading_signal": sample_trading_signal}
        await notifier._handle_trading_signal_async(event_data)

        # Verify message formatter was called
        assert EventType.TRADING_SIGNAL_GENERATED in mock_message_formatter.format_calls

        # Verify Discord message was sent
        assert len(mock_http_client.post_async_calls) == 1

    @pytest.mark.asyncio
    async def test_handle_risk_limit_exceeded_async_success(
        self, config_manager, mock_http_client, mock_message_formatter
    ):
        """Test successful RISK_LIMIT_EXCEEDED event handling."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, None, mock_message_formatter
        )

        event_data = {
            "risk_info": {
                "limit_type": "Max Position Size",
                "current_value": 1.5,
                "limit_value": 1.0
            },
            "symbol": "BTCUSDT",
            "timestamp": int(time.time() * 1000),
            "action_taken": "Order rejected"
        }

        await notifier._handle_risk_limit_exceeded_async(event_data)

        # Verify message formatter was called
        assert EventType.RISK_LIMIT_EXCEEDED in mock_message_formatter.format_calls

        # Verify Discord message was sent
        assert len(mock_http_client.post_async_calls) == 1

    @pytest.mark.asyncio
    async def test_send_formatted_message_async_with_content(
        self, config_manager, mock_http_client
    ):
        """Test sending formatted message with content."""
        notifier = DiscordNotifier(config_manager, mock_http_client)

        embeds = [{"title": "Test Embed", "description": "Test", "color": 0x00ff00}]
        result = await notifier._send_formatted_message_async(
            content="Test content",
            embeds=embeds,
            username="Test Bot"
        )

        assert result is True
        assert len(mock_http_client.post_async_calls) == 1
        call_data = mock_http_client.post_async_calls[0]
        assert call_data["data"]["content"] == "Test content"
        assert call_data["data"]["username"] == "Test Bot"
        assert call_data["data"]["embeds"] == embeds

    @pytest.mark.asyncio
    async def test_send_formatted_message_async_with_embeds_only(
        self, config_manager, mock_http_client
    ):
        """Test sending formatted message with embeds only."""
        notifier = DiscordNotifier(config_manager, mock_http_client)

        embeds = [{"title": "Test Embed", "description": "Test", "color": 0x00ff00}]
        result = await notifier._send_formatted_message_async(embeds=embeds)

        assert result is True
        call_data = mock_http_client.post_async_calls[0]
        assert call_data["data"]["content"] == "📢 Test Embed"

    @pytest.mark.asyncio
    async def test_send_formatted_message_async_fallback_content(
        self, config_manager, mock_http_client
    ):
        """Test sending formatted message with fallback content."""
        notifier = DiscordNotifier(config_manager, mock_http_client)

        result = await notifier._send_formatted_message_async()

        assert result is True
        call_data = mock_http_client.post_async_calls[0]
        assert call_data["data"]["content"] == "📢 Trading Bot Notification"

    @pytest.mark.asyncio
    async def test_send_formatted_message_async_error(
        self, config_manager, failing_http_client
    ):
        """Test sending formatted message handles errors."""
        notifier = DiscordNotifier(config_manager, failing_http_client)

        with patch.object(notifier._logger, 'error') as mock_error:
            result = await notifier._send_formatted_message_async(content="Test")
            assert result is False
            mock_error.assert_called()


class TestLifecycleManagement:
    """Test cases for lifecycle management methods."""

    def test_initialize_notifications_success(
        self, config_manager, mock_http_client, mock_event_hub, mock_message_formatter
    ):
        """Test successful notification initialization."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, mock_event_hub, mock_message_formatter
        )

        notifier.initialize_notifications()

        # Verify subscription was called
        assert mock_event_hub.subscribe.called
        assert notifier._subscriptions_active is True

    def test_initialize_notifications_without_eventhub(
        self, config_manager, mock_http_client
    ):
        """Test notification initialization without EventHub."""
        notifier = DiscordNotifier(config_manager, mock_http_client)

        with pytest.raises(DiscordNotificationError) as excinfo:
            notifier.initialize_notifications()

        assert "EventHub not configured" in str(excinfo.value)

    def test_initialize_notifications_subscription_failure(
        self, config_manager, mock_http_client, mock_event_hub, mock_message_formatter
    ):
        """Test notification initialization handles subscription failure."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, mock_event_hub, mock_message_formatter
        )

        mock_event_hub.subscribe.side_effect = Exception("Subscription failed")

        with pytest.raises(DiscordNotificationError) as excinfo:
            notifier.initialize_notifications()

        assert "Failed to initialize Discord notifications" in str(excinfo.value)

    def test_shutdown_notifications_success(
        self, config_manager, mock_http_client, mock_event_hub, mock_message_formatter
    ):
        """Test successful notification shutdown."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, mock_event_hub, mock_message_formatter
        )

        # Initialize first
        notifier.initialize_notifications()

        # Then shutdown
        notifier.shutdown_notifications()

        # Verify unsubscription was called
        assert mock_event_hub.unsubscribe.called
        assert notifier._subscriptions_active is False

    def test_shutdown_notifications_handles_errors(
        self, config_manager, mock_http_client, mock_event_hub, mock_message_formatter
    ):
        """Test notification shutdown handles errors gracefully."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, mock_event_hub, mock_message_formatter
        )

        # Mock unsubscribe to raise exception
        with patch.object(notifier, 'unsubscribe_from_events', side_effect=Exception("Unsubscribe failed")):
            with patch.object(notifier._logger, 'error') as mock_error:
                notifier.shutdown_notifications()
                mock_error.assert_called()

    @pytest.mark.asyncio
    async def test_close_async_cleanup(
        self, config_manager, mock_http_client, mock_event_hub, mock_message_formatter
    ):
        """Test async close performs proper cleanup."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, mock_event_hub, mock_message_formatter
        )

        # Initialize first
        notifier.initialize_notifications()

        # Mock http client close method
        mock_http_client.close = AsyncMock()

        # Close
        await notifier.close_async()

        # Verify subscriptions were cleaned up
        assert notifier._subscriptions_active is False
        # Verify http client was closed
        mock_http_client.close.assert_called_once()

    def test_close_sync_cleanup(
        self, config_manager, mock_http_client, mock_event_hub, mock_message_formatter
    ):
        """Test sync close performs proper cleanup."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, mock_event_hub, mock_message_formatter
        )

        # Mock http client close_sync method
        mock_http_client.close_sync = Mock()

        # Close
        notifier.close_sync()

        # Verify http client was closed
        mock_http_client.close_sync.assert_called_once()


class TestIntegrationScenarios:
    """Test cases for integration scenarios with real EventHub."""

    @pytest.mark.asyncio
    async def test_eventhub_publishes_order_filled_notifier_handles(
        self, config_manager, mock_http_client, real_event_hub,
        mock_message_formatter, sample_execution_result
    ):
        """Test EventHub publishing ORDER_FILLED event and DiscordNotifier handling it."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, real_event_hub, mock_message_formatter
        )

        # Subscribe to events
        notifier.subscribe_to_events()

        # Publish event through EventHub
        event_data = {"execution_result": sample_execution_result}
        real_event_hub.publish(EventType.ORDER_FILLED, event_data)

        # Allow async handlers to complete
        await asyncio.sleep(0.1)

        # Verify message was formatted and sent
        assert EventType.ORDER_FILLED in mock_message_formatter.format_calls
        assert len(mock_http_client.post_async_calls) == 1

    @pytest.mark.asyncio
    async def test_eventhub_publishes_error_notifier_handles(
        self, config_manager, mock_http_client, real_event_hub, mock_message_formatter
    ):
        """Test EventHub publishing ERROR_OCCURRED event and DiscordNotifier handling it."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, real_event_hub, mock_message_formatter
        )

        # Subscribe to events
        notifier.subscribe_to_events()

        # Publish error event
        error_data = {
            "error": {"message": "Test error", "type": "TestError"},
            "component": "TestComponent"
        }
        real_event_hub.publish(EventType.ERROR_OCCURRED, error_data)

        # Allow async handlers to complete
        await asyncio.sleep(0.1)

        # Verify message was formatted and sent
        assert EventType.ERROR_OCCURRED in mock_message_formatter.format_calls
        assert len(mock_http_client.post_async_calls) == 1

    @pytest.mark.asyncio
    async def test_eventhub_publishes_trading_signal_notifier_handles(
        self, config_manager, mock_http_client, real_event_hub,
        mock_message_formatter, sample_trading_signal
    ):
        """Test EventHub publishing TRADING_SIGNAL_GENERATED event and DiscordNotifier handling it."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, real_event_hub, mock_message_formatter
        )

        # Subscribe to events
        notifier.subscribe_to_events()

        # Publish trading signal event
        signal_data = {"trading_signal": sample_trading_signal}
        real_event_hub.publish(EventType.TRADING_SIGNAL_GENERATED, signal_data)

        # Allow async handlers to complete
        await asyncio.sleep(0.1)

        # Verify message was formatted and sent
        assert EventType.TRADING_SIGNAL_GENERATED in mock_message_formatter.format_calls
        assert len(mock_http_client.post_async_calls) == 1

    @pytest.mark.asyncio
    async def test_multiple_event_types_handling(
        self, config_manager, mock_http_client, real_event_hub,
        mock_message_formatter, sample_execution_result, sample_trading_signal
    ):
        """Test handling multiple different event types."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, real_event_hub, mock_message_formatter
        )

        # Subscribe to events
        notifier.subscribe_to_events()

        # Publish multiple events
        events = [
            (EventType.ORDER_FILLED, {"execution_result": sample_execution_result}),
            (EventType.TRADING_SIGNAL_GENERATED, {"trading_signal": sample_trading_signal}),
            (EventType.CONNECTION_LOST, {"event_type": EventType.CONNECTION_LOST, "service": "Binance"}),
            (EventType.RISK_LIMIT_EXCEEDED, {"risk_info": {"limit_type": "test", "current_value": 1, "limit_value": 0.5}})
        ]

        for event_type, event_data in events:
            real_event_hub.publish(event_type, event_data)

        # Allow async handlers to complete
        await asyncio.sleep(0.1)

        # Verify all messages were handled
        expected_event_types = [event[0] for event in events]
        for event_type in expected_event_types:
            assert event_type in mock_message_formatter.format_calls

        assert len(mock_http_client.post_async_calls) == len(events)

    def test_unsubscribe_stops_event_handling(
        self, config_manager, mock_http_client, real_event_hub,
        mock_message_formatter, sample_execution_result
    ):
        """Test unsubscribing stops event handling."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, real_event_hub, mock_message_formatter
        )

        # Subscribe and then unsubscribe
        notifier.subscribe_to_events()
        notifier.unsubscribe_from_events()

        # Publish event after unsubscribing
        event_data = {"execution_result": sample_execution_result}
        real_event_hub.publish(EventType.ORDER_FILLED, event_data)

        # Verify no messages were sent
        assert len(mock_http_client.post_async_calls) == 0
        assert len(mock_message_formatter.format_calls) == 0


class TestErrorHandling:
    """Test cases for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_malformed_execution_result_data(
        self, config_manager, mock_http_client
    ):
        """Test handling malformed ExecutionResult data."""
        # Create a formatter that will fail with the malformed data
        failing_formatter = MockMessageFormatterFactory(should_fail=True)
        notifier = DiscordNotifier(
            config_manager, mock_http_client, None, failing_formatter
        )

        # Create malformed event data
        event_data = {"execution_result": "not_an_execution_result"}

        # This should not raise an exception but should handle the error gracefully
        with patch.object(notifier._logger, 'error') as mock_error:
            await notifier._handle_order_filled_async(event_data)
            # The event handler catches all exceptions, so error should be logged
            assert mock_error.call_count > 0

    @pytest.mark.asyncio
    async def test_malformed_trading_signal_data(
        self, config_manager, mock_http_client, mock_message_formatter
    ):
        """Test handling malformed TradingSignal data."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, None, mock_message_formatter
        )

        # Create malformed event data
        event_data = {"trading_signal": {"symbol": "INVALID"}}

        with patch.object(notifier._logger, 'error') as mock_error:
            await notifier._handle_trading_signal_async(event_data)
            # The event handler catches all exceptions, so error should be logged
            assert mock_error.call_count > 0

    @pytest.mark.asyncio
    async def test_discord_api_failure_during_event_handling(
        self, config_manager, failing_http_client, mock_message_formatter, sample_execution_result
    ):
        """Test handling Discord API failures during event processing."""
        notifier = DiscordNotifier(
            config_manager, failing_http_client, None, mock_message_formatter
        )

        event_data = {"execution_result": sample_execution_result}

        with patch.object(notifier._logger, 'error') as mock_error:
            await notifier._handle_order_filled_async(event_data)
            mock_error.assert_called()

    @pytest.mark.asyncio
    async def test_message_formatter_exception(
        self, config_manager, mock_http_client, failing_message_formatter, sample_execution_result
    ):
        """Test handling message formatter exceptions."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, None, failing_message_formatter
        )

        event_data = {"execution_result": sample_execution_result}

        with patch.object(notifier._logger, 'error') as mock_error:
            await notifier._handle_order_filled_async(event_data)
            mock_error.assert_called()

    @pytest.mark.asyncio
    async def test_missing_event_data_fields(
        self, config_manager, mock_http_client, mock_message_formatter
    ):
        """Test handling missing required event data fields."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, None, mock_message_formatter
        )

        # Test missing execution_result
        event_data = {}

        with patch.object(notifier._logger, 'error') as mock_error:
            await notifier._handle_order_filled_async(event_data)
            assert mock_error.call_count > 0

        # Test missing trading_signal
        with patch.object(notifier._logger, 'error') as mock_error:
            await notifier._handle_trading_signal_async(event_data)
            assert mock_error.call_count > 0


class TestEdgeCasesAndBoundaryConditions:
    """Test cases for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_event_data(
        self, config_manager, mock_http_client, mock_message_formatter
    ):
        """Test handling completely empty event data."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, None, mock_message_formatter
        )

        with patch.object(notifier._logger, 'error') as mock_error:
            await notifier._handle_order_filled_async({})
            assert mock_error.call_count > 0

    @pytest.mark.asyncio
    async def test_none_event_data(
        self, config_manager, mock_http_client, mock_message_formatter
    ):
        """Test handling None event data."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, None, mock_message_formatter
        )

        with patch.object(notifier._logger, 'error') as mock_error:
            await notifier._handle_order_filled_async(None)
            assert mock_error.call_count > 0

    def test_subscription_state_consistency(
        self, config_manager, mock_http_client, mock_event_hub, mock_message_formatter
    ):
        """Test subscription state remains consistent across operations."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, mock_event_hub, mock_message_formatter
        )

        # Initial state
        assert notifier._subscriptions_active is False
        assert len(notifier._event_handlers) == 0

        # After subscribing
        notifier.subscribe_to_events()
        assert notifier._subscriptions_active is True
        assert len(notifier._event_handlers) > 0

        # After unsubscribing
        notifier.unsubscribe_from_events()
        assert notifier._subscriptions_active is False
        assert len(notifier._event_handlers) == 0

    def test_event_handler_mapping_completeness(
        self, config_manager, mock_http_client, mock_event_hub, mock_message_formatter
    ):
        """Test that all expected event types have handlers mapped."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, mock_event_hub, mock_message_formatter
        )

        notifier.subscribe_to_events()

        expected_handlers = {
            EventType.ORDER_FILLED: notifier._handle_order_filled_async,
            EventType.ERROR_OCCURRED: notifier._handle_error_occurred_async,
            EventType.CONNECTION_LOST: notifier._handle_connection_event_async,
            EventType.CONNECTION_RESTORED: notifier._handle_connection_event_async,
            EventType.TRADING_SIGNAL_GENERATED: notifier._handle_trading_signal_async,
            EventType.RISK_LIMIT_EXCEEDED: notifier._handle_risk_limit_exceeded_async,
        }

        for event_type, expected_handler in expected_handlers.items():
            assert event_type in notifier._event_handlers
            assert notifier._event_handlers[event_type] == expected_handler

    @pytest.mark.asyncio
    async def test_concurrent_event_handling(
        self, config_manager, mock_http_client, real_event_hub,
        mock_message_formatter, sample_execution_result
    ):
        """Test handling multiple concurrent events."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, real_event_hub, mock_message_formatter
        )

        notifier.subscribe_to_events()

        # Publish multiple events simultaneously
        events = []
        for i in range(5):
            event_data = {"execution_result": sample_execution_result}
            events.append(real_event_hub.publish(EventType.ORDER_FILLED, event_data))

        # Allow all async handlers to complete
        await asyncio.sleep(0.2)

        # Verify all events were handled
        assert len(mock_http_client.post_async_calls) == 5
        assert mock_message_formatter.format_calls.count(EventType.ORDER_FILLED) == 5

    def test_initialization_and_shutdown_lifecycle(
        self, config_manager, mock_http_client, mock_event_hub, mock_message_formatter
    ):
        """Test complete initialization and shutdown lifecycle."""
        notifier = DiscordNotifier(
            config_manager, mock_http_client, mock_event_hub, mock_message_formatter
        )

        # Test initialization
        notifier.initialize_notifications()
        assert notifier._subscriptions_active is True

        # Test shutdown
        notifier.shutdown_notifications()
        assert notifier._subscriptions_active is False

        # Test re-initialization after shutdown
        notifier.initialize_notifications()
        assert notifier._subscriptions_active is True

    def test_factory_function_with_eventhub_integration(self, config_manager, real_event_hub):
        """Test factory function creates DiscordNotifier with EventHub integration."""
        from trading_bot.notification.discord_notifier import create_discord_notifier

        notifier = create_discord_notifier(
            config_manager=config_manager,
            event_hub=real_event_hub
        )

        assert isinstance(notifier, DiscordNotifier)
        assert notifier._event_hub == real_event_hub
        assert isinstance(notifier._message_formatter_factory, MessageFormatterFactory)

        # Test that initialization works
        notifier.initialize_notifications()
        assert notifier._subscriptions_active is True