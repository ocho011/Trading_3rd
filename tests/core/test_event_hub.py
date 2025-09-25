"""Comprehensive unit tests for EventHub class.

This module contains comprehensive unit tests for the EventHub class,
ensuring proper functionality, thread safety, async/await support,
and adherence to the event-driven architecture requirements.
"""

import asyncio
import threading
from typing import Any

import pytest

from trading_bot.core.event_hub import EventHub, EventHubInterface, EventType


class TestEventHubInitialization:
    """Test cases for EventHub initialization and interface compliance."""

    def test_implements_interface(self) -> None:
        """Test that EventHub implements EventHubInterface."""
        event_hub = EventHub()
        assert isinstance(event_hub, EventHubInterface)

    def test_initialization(self) -> None:
        """Test EventHub initialization."""
        event_hub = EventHub()

        # Test that EventHub can be instantiated
        assert event_hub is not None

        # Test that internal structures are initialized
        assert hasattr(event_hub, "_subscribers")
        assert hasattr(event_hub, "_lock")
        assert isinstance(event_hub._subscribers, dict)
        assert isinstance(event_hub._lock, type(threading.RLock()))

    def test_initial_state(self) -> None:
        """Test that EventHub starts with empty subscriber registry."""
        event_hub = EventHub()
        assert len(event_hub._subscribers) == 0


class TestEventHubSingleSubscriber:
    """Test cases for single subscriber functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.event_hub = EventHub()
        self.callback_data = []

        def test_callback(data: Any) -> None:
            self.callback_data.append(data)

        self.test_callback = test_callback

    def test_single_subscriber_receives_event(self) -> None:
        """Test that a single subscriber receives published events."""
        # Subscribe to event
        self.event_hub.subscribe(EventType.PRICE_UPDATE, self.test_callback)

        # Publish event
        test_data = {"price": 100.50, "symbol": "BTC"}
        self.event_hub.publish(EventType.PRICE_UPDATE, test_data)

        # Verify callback was called with correct data
        assert len(self.callback_data) == 1
        assert self.callback_data[0] == test_data

    def test_single_subscriber_multiple_events(self) -> None:
        """Test that a single subscriber receives multiple events."""
        self.event_hub.subscribe(EventType.PRICE_UPDATE, self.test_callback)

        # Publish multiple events
        test_data_1 = {"price": 100.50, "symbol": "BTC"}
        test_data_2 = {"price": 3000.75, "symbol": "ETH"}

        self.event_hub.publish(EventType.PRICE_UPDATE, test_data_1)
        self.event_hub.publish(EventType.PRICE_UPDATE, test_data_2)

        # Verify both events were received
        assert len(self.callback_data) == 2
        assert self.callback_data[0] == test_data_1
        assert self.callback_data[1] == test_data_2

    def test_subscriber_only_receives_subscribed_events(self) -> None:
        """Test that subscriber only receives events for subscribed event types."""
        self.event_hub.subscribe(EventType.PRICE_UPDATE, self.test_callback)

        # Publish to subscribed event type
        self.event_hub.publish(EventType.PRICE_UPDATE, {"price": 100})

        # Publish to non-subscribed event type
        self.event_hub.publish(EventType.ORDER_PLACED, {"order_id": "123"})

        # Verify only subscribed event was received
        assert len(self.callback_data) == 1
        assert self.callback_data[0] == {"price": 100}


class TestEventHubMultipleSubscribers:
    """Test cases for multiple subscribers functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.event_hub = EventHub()
        self.callback_data_1 = []
        self.callback_data_2 = []
        self.callback_data_3 = []

        def callback_1(data: Any) -> None:
            self.callback_data_1.append(data)

        def callback_2(data: Any) -> None:
            self.callback_data_2.append(data)

        def callback_3(data: Any) -> None:
            self.callback_data_3.append(data)

        self.callback_1 = callback_1
        self.callback_2 = callback_2
        self.callback_3 = callback_3

    def test_multiple_subscribers_receive_same_event(self) -> None:
        """Test that multiple subscribers all receive the same event."""
        # Subscribe multiple callbacks to same event type
        self.event_hub.subscribe(EventType.PRICE_UPDATE, self.callback_1)
        self.event_hub.subscribe(EventType.PRICE_UPDATE, self.callback_2)
        self.event_hub.subscribe(EventType.PRICE_UPDATE, self.callback_3)

        # Publish event
        test_data = {"price": 100.50, "symbol": "BTC"}
        self.event_hub.publish(EventType.PRICE_UPDATE, test_data)

        # Verify all callbacks received the event
        assert len(self.callback_data_1) == 1
        assert len(self.callback_data_2) == 1
        assert len(self.callback_data_3) == 1

        assert self.callback_data_1[0] == test_data
        assert self.callback_data_2[0] == test_data
        assert self.callback_data_3[0] == test_data

    def test_different_subscribers_for_different_events(self) -> None:
        """Test subscribers for different event types receive appropriate events."""
        # Subscribe different callbacks to different event types
        self.event_hub.subscribe(EventType.PRICE_UPDATE, self.callback_1)
        self.event_hub.subscribe(EventType.ORDER_PLACED, self.callback_2)
        self.event_hub.subscribe(EventType.SYSTEM_STARTUP, self.callback_3)

        # Publish different events
        price_data = {"price": 100.50}
        order_data = {"order_id": "123"}
        system_data = {"timestamp": "2024-01-01"}

        self.event_hub.publish(EventType.PRICE_UPDATE, price_data)
        self.event_hub.publish(EventType.ORDER_PLACED, order_data)
        self.event_hub.publish(EventType.SYSTEM_STARTUP, system_data)

        # Verify each callback received only its event
        assert len(self.callback_data_1) == 1
        assert len(self.callback_data_2) == 1
        assert len(self.callback_data_3) == 1

        assert self.callback_data_1[0] == price_data
        assert self.callback_data_2[0] == order_data
        assert self.callback_data_3[0] == system_data

    def test_duplicate_subscription_prevention(self) -> None:
        """Test that duplicate subscriptions are prevented."""
        # Subscribe same callback multiple times
        self.event_hub.subscribe(EventType.PRICE_UPDATE, self.callback_1)
        self.event_hub.subscribe(EventType.PRICE_UPDATE, self.callback_1)
        self.event_hub.subscribe(EventType.PRICE_UPDATE, self.callback_1)

        # Verify only one subscription exists
        assert self.event_hub.get_subscriber_count(EventType.PRICE_UPDATE) == 1

        # Publish event and verify callback called only once
        self.event_hub.publish(EventType.PRICE_UPDATE, {"test": "data"})
        assert len(self.callback_data_1) == 1


class TestEventHubNoSubscribers:
    """Test cases for publishing events with no subscribers."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.event_hub = EventHub()

    def test_publish_with_no_subscribers_does_not_error(self) -> None:
        """Test publishing to event type with no subscribers doesn't raise errors."""
        # Should not raise any exceptions
        self.event_hub.publish(EventType.PRICE_UPDATE, {"price": 100})
        self.event_hub.publish(EventType.ORDER_PLACED, {"order_id": "123"})
        self.event_hub.publish(EventType.SYSTEM_STARTUP, {"timestamp": "now"})

    def test_publish_to_nonexistent_event_type(self) -> None:
        """Test publishing to event types that have never had subscribers."""
        # Should not raise any exceptions
        self.event_hub.publish("nonexistent_event", {"data": "test"})


class TestEventHubAsyncHandlers:
    """Test cases for asynchronous event handlers."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.event_hub = EventHub()
        self.async_callback_data = []
        self.sync_callback_data = []

    async def async_callback(self, data: Any) -> None:
        """Async callback for testing."""
        await asyncio.sleep(0.01)  # Simulate async work
        self.async_callback_data.append(data)

    def sync_callback(self, data: Any) -> None:
        """Sync callback for testing."""
        self.sync_callback_data.append(data)

    @pytest.mark.asyncio
    async def test_async_handler_execution(self) -> None:
        """Test that async handlers are executed properly."""
        self.event_hub.subscribe(EventType.PRICE_UPDATE, self.async_callback)

        # Publish event
        test_data = {"price": 100.50}
        self.event_hub.publish(EventType.PRICE_UPDATE, test_data)

        # Wait for async execution
        await asyncio.sleep(0.1)

        # Verify async callback was called
        assert len(self.async_callback_data) == 1
        assert self.async_callback_data[0] == test_data

    @pytest.mark.asyncio
    async def test_mixed_sync_async_handlers(self) -> None:
        """Test that both sync and async handlers work together."""
        # Subscribe both sync and async callbacks
        self.event_hub.subscribe(EventType.PRICE_UPDATE, self.sync_callback)
        self.event_hub.subscribe(EventType.PRICE_UPDATE, self.async_callback)

        # Publish event
        test_data = {"price": 100.50}
        self.event_hub.publish(EventType.PRICE_UPDATE, test_data)

        # Wait for async execution
        await asyncio.sleep(0.1)

        # Verify both callbacks were called
        assert len(self.sync_callback_data) == 1
        assert len(self.async_callback_data) == 1
        assert self.sync_callback_data[0] == test_data
        assert self.async_callback_data[0] == test_data

    @pytest.mark.asyncio
    async def test_async_handler_with_exception(self) -> None:
        """Test that exceptions in async handlers don't break the system."""

        async def failing_async_callback(data: Any) -> None:
            await asyncio.sleep(0.01)
            raise RuntimeError("Test async error")

        # Subscribe both failing and working callbacks
        self.event_hub.subscribe(EventType.PRICE_UPDATE, failing_async_callback)
        self.event_hub.subscribe(EventType.PRICE_UPDATE, self.async_callback)

        # Publish event
        test_data = {"price": 100.50}
        self.event_hub.publish(EventType.PRICE_UPDATE, test_data)

        # Wait for async execution
        await asyncio.sleep(0.1)

        # Verify working callback still executed despite failing one
        assert len(self.async_callback_data) == 1
        assert self.async_callback_data[0] == test_data


class TestEventHubSubscriptionManagement:
    """Test cases for subscription and unsubscription functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.event_hub = EventHub()
        self.callback_data = []

        def test_callback(data: Any) -> None:
            self.callback_data.append(data)

        self.test_callback = test_callback

    def test_unsubscribe_removes_callback(self) -> None:
        """Test that unsubscribing removes the callback."""
        # Subscribe and verify subscription
        self.event_hub.subscribe(EventType.PRICE_UPDATE, self.test_callback)
        assert self.event_hub.get_subscriber_count(EventType.PRICE_UPDATE) == 1

        # Unsubscribe and verify removal
        self.event_hub.unsubscribe(EventType.PRICE_UPDATE, self.test_callback)
        assert self.event_hub.get_subscriber_count(EventType.PRICE_UPDATE) == 0

        # Publish event and verify callback not called
        self.event_hub.publish(EventType.PRICE_UPDATE, {"price": 100})
        assert len(self.callback_data) == 0

    def test_unsubscribe_specific_callback(self) -> None:
        """Test that unsubscribing removes only the specific callback."""
        callback_data_2 = []

        def callback_2(data: Any) -> None:
            callback_data_2.append(data)

        # Subscribe both callbacks
        self.event_hub.subscribe(EventType.PRICE_UPDATE, self.test_callback)
        self.event_hub.subscribe(EventType.PRICE_UPDATE, callback_2)
        assert self.event_hub.get_subscriber_count(EventType.PRICE_UPDATE) == 2

        # Unsubscribe one callback
        self.event_hub.unsubscribe(EventType.PRICE_UPDATE, self.test_callback)
        assert self.event_hub.get_subscriber_count(EventType.PRICE_UPDATE) == 1

        # Publish event and verify only remaining callback is called
        self.event_hub.publish(EventType.PRICE_UPDATE, {"price": 100})
        assert len(self.callback_data) == 0
        assert len(callback_data_2) == 1

    def test_unsubscribe_nonexistent_callback_raises_error(self) -> None:
        """Test that unsubscribing a non-existent callback raises KeyError."""
        # Try to unsubscribe without subscribing first
        with pytest.raises(KeyError):
            self.event_hub.unsubscribe(EventType.PRICE_UPDATE, self.test_callback)

    def test_unsubscribe_from_nonexistent_event_type_raises_error(self) -> None:
        """Test that unsubscribing from non-existent event type raises KeyError."""
        with pytest.raises(KeyError):
            self.event_hub.unsubscribe("nonexistent_event", self.test_callback)


class TestEventHubThreadSafety:
    """Test cases for thread safety."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.event_hub = EventHub()
        self.callback_data = []
        self.lock = threading.Lock()

        def thread_safe_callback(data: Any) -> None:
            with self.lock:
                self.callback_data.append(data)

        self.thread_safe_callback = thread_safe_callback

    def test_concurrent_subscriptions(self) -> None:
        """Test that concurrent subscriptions from multiple threads work correctly."""
        results = []

        def subscribe_worker(worker_id: int) -> None:
            def callback(data: Any) -> None:
                results.append(f"worker_{worker_id}_{data}")

            self.event_hub.subscribe(EventType.PRICE_UPDATE, callback)

        # Create multiple threads subscribing concurrently
        threads = []
        for i in range(10):
            thread = threading.Thread(target=subscribe_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all subscriptions were successful
        assert self.event_hub.get_subscriber_count(EventType.PRICE_UPDATE) == 10

        # Publish event and verify all callbacks were called
        self.event_hub.publish(EventType.PRICE_UPDATE, "test")
        assert len(results) == 10

    def test_concurrent_publications(self) -> None:
        """Test that concurrent publications from multiple threads work correctly."""
        self.event_hub.subscribe(EventType.PRICE_UPDATE, self.thread_safe_callback)

        def publish_worker(worker_id: int) -> None:
            for i in range(5):
                self.event_hub.publish(
                    EventType.PRICE_UPDATE, f"worker_{worker_id}_event_{i}"
                )

        # Create multiple threads publishing concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=publish_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all events were received
        assert len(self.callback_data) == 25

    def test_concurrent_subscribe_unsubscribe(self) -> None:
        """Test concurrent subscription and unsubscription operations."""

        def subscribe_unsubscribe_worker(worker_id: int) -> None:
            def callback(data: Any) -> None:
                pass

            # Subscribe and immediately unsubscribe
            self.event_hub.subscribe(EventType.PRICE_UPDATE, callback)
            self.event_hub.unsubscribe(EventType.PRICE_UPDATE, callback)

        # Create multiple threads performing subscribe/unsubscribe operations
        threads = []
        for i in range(10):
            thread = threading.Thread(target=subscribe_unsubscribe_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no subscribers remain
        assert self.event_hub.get_subscriber_count(EventType.PRICE_UPDATE) == 0


class TestEventHubErrorHandling:
    """Test cases for error handling."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.event_hub = EventHub()

    def test_subscribe_with_empty_event_type_raises_error(self) -> None:
        """Test that subscribing with empty event type raises ValueError."""

        def callback(data: Any) -> None:
            pass

        with pytest.raises(ValueError, match="Event type cannot be empty or None"):
            self.event_hub.subscribe("", callback)

        with pytest.raises(ValueError, match="Event type cannot be empty or None"):
            self.event_hub.subscribe(None, callback)

    def test_subscribe_with_non_callable_raises_error(self) -> None:
        """Test that subscribing with non-callable raises TypeError."""
        with pytest.raises(TypeError, match="Callback must be callable"):
            self.event_hub.subscribe(EventType.PRICE_UPDATE, "not_callable")

        with pytest.raises(TypeError, match="Callback must be callable"):
            self.event_hub.subscribe(EventType.PRICE_UPDATE, 123)

    def test_publish_with_empty_event_type_raises_error(self) -> None:
        """Test that publishing with empty event type raises ValueError."""
        with pytest.raises(ValueError, match="Event type cannot be empty or None"):
            self.event_hub.publish("", {"data": "test"})

        with pytest.raises(ValueError, match="Event type cannot be empty or None"):
            self.event_hub.publish(None, {"data": "test"})

    def test_unsubscribe_with_empty_event_type_raises_error(self) -> None:
        """Test that unsubscribing with empty event type raises ValueError."""

        def callback(data: Any) -> None:
            pass

        with pytest.raises(ValueError, match="Event type cannot be empty or None"):
            self.event_hub.unsubscribe("", callback)

        with pytest.raises(ValueError, match="Event type cannot be empty or None"):
            self.event_hub.unsubscribe(None, callback)

    def test_callback_exception_handling(self) -> None:
        """Test that exceptions in callbacks don't break the event system."""
        working_callback_data = []

        def failing_callback(data: Any) -> None:
            raise RuntimeError("Test error")

        def working_callback(data: Any) -> None:
            working_callback_data.append(data)

        # Subscribe both callbacks
        self.event_hub.subscribe(EventType.PRICE_UPDATE, failing_callback)
        self.event_hub.subscribe(EventType.PRICE_UPDATE, working_callback)

        # Publish event - should not raise exception
        test_data = {"price": 100}
        self.event_hub.publish(EventType.PRICE_UPDATE, test_data)

        # Verify working callback still executed
        assert len(working_callback_data) == 1
        assert working_callback_data[0] == test_data


class TestEventHubGetSubscriberCount:
    """Test cases for get_subscriber_count method."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.event_hub = EventHub()

    def test_get_subscriber_count_no_subscribers(self) -> None:
        """Test get_subscriber_count returns 0 when no subscribers."""
        count = self.event_hub.get_subscriber_count(EventType.PRICE_UPDATE)
        assert count == 0

    def test_get_subscriber_count_with_subscribers(self) -> None:
        """Test get_subscriber_count returns correct count with subscribers."""

        def callback_1(data: Any) -> None:
            pass

        def callback_2(data: Any) -> None:
            pass

        # Initially no subscribers
        assert self.event_hub.get_subscriber_count(EventType.PRICE_UPDATE) == 0

        # Add first subscriber
        self.event_hub.subscribe(EventType.PRICE_UPDATE, callback_1)
        assert self.event_hub.get_subscriber_count(EventType.PRICE_UPDATE) == 1

        # Add second subscriber
        self.event_hub.subscribe(EventType.PRICE_UPDATE, callback_2)
        assert self.event_hub.get_subscriber_count(EventType.PRICE_UPDATE) == 2

        # Remove one subscriber
        self.event_hub.unsubscribe(EventType.PRICE_UPDATE, callback_1)
        assert self.event_hub.get_subscriber_count(EventType.PRICE_UPDATE) == 1

        # Remove last subscriber
        self.event_hub.unsubscribe(EventType.PRICE_UPDATE, callback_2)
        assert self.event_hub.get_subscriber_count(EventType.PRICE_UPDATE) == 0

    def test_get_subscriber_count_different_event_types(self) -> None:
        """Test get_subscriber_count for different event types."""

        def callback(data: Any) -> None:
            pass

        # Subscribe to different event types
        self.event_hub.subscribe(EventType.PRICE_UPDATE, callback)
        self.event_hub.subscribe(EventType.ORDER_PLACED, callback)

        # Verify counts are independent
        assert self.event_hub.get_subscriber_count(EventType.PRICE_UPDATE) == 1
        assert self.event_hub.get_subscriber_count(EventType.ORDER_PLACED) == 1
        assert self.event_hub.get_subscriber_count(EventType.SYSTEM_STARTUP) == 0

    def test_get_subscriber_count_with_empty_event_type_raises_error(self) -> None:
        """Test that get_subscriber_count with empty event type raises ValueError."""
        with pytest.raises(ValueError, match="Event type cannot be empty or None"):
            self.event_hub.get_subscriber_count("")

        with pytest.raises(ValueError, match="Event type cannot be empty or None"):
            self.event_hub.get_subscriber_count(None)


class TestEventHubClearSubscribers:
    """Test cases for clear_subscribers method."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.event_hub = EventHub()

        def callback_1(data: Any) -> None:
            pass

        def callback_2(data: Any) -> None:
            pass

        self.callback_1 = callback_1
        self.callback_2 = callback_2

    def test_clear_subscribers_specific_event_type(self) -> None:
        """Test clearing subscribers for a specific event type."""
        # Subscribe to multiple event types
        self.event_hub.subscribe(EventType.PRICE_UPDATE, self.callback_1)
        self.event_hub.subscribe(EventType.ORDER_PLACED, self.callback_1)
        self.event_hub.subscribe(EventType.PRICE_UPDATE, self.callback_2)

        # Verify initial state
        assert self.event_hub.get_subscriber_count(EventType.PRICE_UPDATE) == 2
        assert self.event_hub.get_subscriber_count(EventType.ORDER_PLACED) == 1

        # Clear subscribers for specific event type
        self.event_hub.clear_subscribers(EventType.PRICE_UPDATE)

        # Verify only specified event type was cleared
        assert self.event_hub.get_subscriber_count(EventType.PRICE_UPDATE) == 0
        assert self.event_hub.get_subscriber_count(EventType.ORDER_PLACED) == 1

    def test_clear_subscribers_all_event_types(self) -> None:
        """Test clearing all subscribers for all event types."""
        # Subscribe to multiple event types
        self.event_hub.subscribe(EventType.PRICE_UPDATE, self.callback_1)
        self.event_hub.subscribe(EventType.ORDER_PLACED, self.callback_1)
        self.event_hub.subscribe(EventType.SYSTEM_STARTUP, self.callback_2)

        # Verify initial state
        assert self.event_hub.get_subscriber_count(EventType.PRICE_UPDATE) == 1
        assert self.event_hub.get_subscriber_count(EventType.ORDER_PLACED) == 1
        assert self.event_hub.get_subscriber_count(EventType.SYSTEM_STARTUP) == 1

        # Clear all subscribers
        self.event_hub.clear_subscribers()

        # Verify all subscribers were cleared
        assert self.event_hub.get_subscriber_count(EventType.PRICE_UPDATE) == 0
        assert self.event_hub.get_subscriber_count(EventType.ORDER_PLACED) == 0
        assert self.event_hub.get_subscriber_count(EventType.SYSTEM_STARTUP) == 0

    def test_clear_subscribers_nonexistent_event_type(self) -> None:
        """Test clearing subscribers for non-existent event type doesn't raise error."""
        # Should not raise any exception
        self.event_hub.clear_subscribers("nonexistent_event")
        self.event_hub.clear_subscribers(EventType.PRICE_UPDATE)

    def test_clear_subscribers_with_empty_event_type_raises_error(self) -> None:
        """Test that clear_subscribers with empty event type raises ValueError."""
        with pytest.raises(ValueError, match="Event type cannot be empty"):
            self.event_hub.clear_subscribers("")

        # None should be allowed (clears all)
        self.event_hub.clear_subscribers(None)


class TestEventHubIntegration:
    """Integration tests combining multiple features."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.event_hub = EventHub()

    def test_complex_event_flow(self) -> None:
        """Test a complex event flow with multiple subscribers and event types."""
        price_data = []
        order_data = []
        system_data = []

        def price_callback(data: Any) -> None:
            price_data.append(data)

        def order_callback(data: Any) -> None:
            order_data.append(data)

        def system_callback(data: Any) -> None:
            system_data.append(data)

        # Subscribe to different event types
        self.event_hub.subscribe(EventType.PRICE_UPDATE, price_callback)
        self.event_hub.subscribe(EventType.ORDER_PLACED, order_callback)
        self.event_hub.subscribe(EventType.SYSTEM_STARTUP, system_callback)

        # Also subscribe system callback to price updates
        self.event_hub.subscribe(EventType.PRICE_UPDATE, system_callback)

        # Publish various events
        self.event_hub.publish(EventType.PRICE_UPDATE, {"price": 100, "symbol": "BTC"})
        self.event_hub.publish(
            EventType.ORDER_PLACED, {"order_id": "123", "amount": 0.1}
        )
        self.event_hub.publish(EventType.SYSTEM_STARTUP, {"timestamp": "2024-01-01"})
        self.event_hub.publish(EventType.PRICE_UPDATE, {"price": 101, "symbol": "BTC"})

        # Verify all callbacks received appropriate events
        assert len(price_data) == 2
        assert len(order_data) == 1
        assert len(system_data) == 3  # 2 price updates + 1 system startup

        # Verify correct data
        assert price_data[0]["price"] == 100
        assert price_data[1]["price"] == 101
        assert order_data[0]["order_id"] == "123"
        # system_data contains: [price_update_1, price_update_2, system_startup]
        # So the system startup event is at index 2
        # But we need to check if it has the timestamp field from system_startup event
        # Let's verify the actual system startup event
        system_startup_events = [event for event in system_data if "timestamp" in event]
        assert len(system_startup_events) == 1
        assert system_startup_events[0]["timestamp"] == "2024-01-01"

    @pytest.mark.asyncio
    async def test_mixed_sync_async_with_error_handling(self) -> None:
        """Test complex scenario with sync/async callbacks and error handling."""
        sync_data = []
        async_data = []

        def sync_callback(data: Any) -> None:
            sync_data.append(data)

        def failing_sync_callback(data: Any) -> None:
            raise RuntimeError("Sync error")

        async def async_callback(data: Any) -> None:
            await asyncio.sleep(0.01)
            async_data.append(data)

        async def failing_async_callback(data: Any) -> None:
            await asyncio.sleep(0.01)
            raise RuntimeError("Async error")

        # Subscribe all callbacks
        self.event_hub.subscribe(EventType.PRICE_UPDATE, sync_callback)
        self.event_hub.subscribe(EventType.PRICE_UPDATE, failing_sync_callback)
        self.event_hub.subscribe(EventType.PRICE_UPDATE, async_callback)
        self.event_hub.subscribe(EventType.PRICE_UPDATE, failing_async_callback)

        # Publish event
        test_data = {"price": 100}
        self.event_hub.publish(EventType.PRICE_UPDATE, test_data)

        # Wait for async execution
        await asyncio.sleep(0.1)

        # Verify working callbacks executed despite failing ones
        assert len(sync_data) == 1
        assert len(async_data) == 1
        assert sync_data[0] == test_data
        assert async_data[0] == test_data


class TestEventType:
    """Test cases for EventType constants."""

    def test_event_type_constants_exist(self) -> None:
        """Test that all required event type constants are defined."""
        # Market data events
        assert EventType.MARKET_DATA_RECEIVED == "market_data_received"
        assert EventType.PRICE_UPDATE == "price_update"
        assert EventType.VOLUME_UPDATE == "volume_update"
        assert EventType.ORDER_BOOK_UPDATE == "order_book_update"

        # Trading signal events
        assert EventType.TRADING_SIGNAL_GENERATED == "trading_signal_generated"
        assert EventType.BUY_SIGNAL == "buy_signal"
        assert EventType.SELL_SIGNAL == "sell_signal"
        assert EventType.HOLD_SIGNAL == "hold_signal"

        # Order management events
        assert EventType.ORDER_PLACED == "order_placed"
        assert EventType.ORDER_FILLED == "order_filled"
        assert EventType.ORDER_CANCELLED == "order_cancelled"
        assert EventType.ORDER_REJECTED == "order_rejected"

        # Risk management events
        assert EventType.RISK_LIMIT_EXCEEDED == "risk_limit_exceeded"
        assert EventType.POSITION_SIZE_WARNING == "position_size_warning"
        assert EventType.PORTFOLIO_REBALANCE == "portfolio_rebalance"

        # System events
        assert EventType.SYSTEM_STARTUP == "system_startup"
        assert EventType.SYSTEM_SHUTDOWN == "system_shutdown"
        assert EventType.ERROR_OCCURRED == "error_occurred"
        assert EventType.CONNECTION_LOST == "connection_lost"
        assert EventType.CONNECTION_RESTORED == "connection_restored"

    def test_event_type_constants_are_strings(self) -> None:
        """Test that all event type constants are strings."""
        event_attributes = [attr for attr in dir(EventType) if not attr.startswith("_")]

        for attr_name in event_attributes:
            attr_value = getattr(EventType, attr_name)
            assert isinstance(
                attr_value, str
            ), f"EventType.{attr_name} should be a string"
