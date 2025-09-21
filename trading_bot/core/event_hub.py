"""Event Hub module for managing event-driven communication in the trading bot system.

This module provides a centralized event hub for managing event subscriptions
and publications across different components of the trading bot application.
"""

import asyncio
import inspect
import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List


class EventType:
    """Event type constants for the trading bot system.

    This class defines all event types that can be published and subscribed to
    within the trading bot application. Using constants ensures type safety
    and prevents typos in event type names.
    """

    # Market data events
    MARKET_DATA_RECEIVED: str = "market_data_received"
    PRICE_UPDATE: str = "price_update"
    VOLUME_UPDATE: str = "volume_update"
    ORDER_BOOK_UPDATE: str = "order_book_update"

    # Trading signal events
    TRADING_SIGNAL_GENERATED: str = "trading_signal_generated"
    BUY_SIGNAL: str = "buy_signal"
    SELL_SIGNAL: str = "sell_signal"
    HOLD_SIGNAL: str = "hold_signal"

    # Order management events
    ORDER_PLACED: str = "order_placed"
    ORDER_FILLED: str = "order_filled"
    ORDER_CANCELLED: str = "order_cancelled"
    ORDER_REJECTED: str = "order_rejected"

    # Risk management events
    RISK_LIMIT_EXCEEDED: str = "risk_limit_exceeded"
    POSITION_SIZE_WARNING: str = "position_size_warning"
    PORTFOLIO_REBALANCE: str = "portfolio_rebalance"

    # System events
    SYSTEM_STARTUP: str = "system_startup"
    SYSTEM_SHUTDOWN: str = "system_shutdown"
    ERROR_OCCURRED: str = "error_occurred"
    CONNECTION_LOST: str = "connection_lost"
    CONNECTION_RESTORED: str = "connection_restored"


class EventHubInterface(ABC):
    """Abstract interface for event hub implementations.

    This interface defines the contract for event hub implementations,
    following the Interface Segregation Principle.
    """

    @abstractmethod
    def subscribe(self, event_type: str, callback: Callable[[Any], None]) -> None:
        """Subscribe to an event type with a callback function."""
        pass

    @abstractmethod
    def unsubscribe(self, event_type: str, callback: Callable[[Any], None]) -> None:
        """Unsubscribe from an event type."""
        pass

    @abstractmethod
    def publish(self, event_type: str, data: Any) -> None:
        """Publish an event to all subscribers."""
        pass


class EventHub(EventHubInterface):
    """Centralized event hub for managing event-driven communication.

    The EventHub class provides a thread-safe implementation for managing
    event subscriptions and publications across different components of the
    trading bot system. It follows the Observer pattern and ensures loose
    coupling between components.

    This class adheres to the Single Responsibility Principle by focusing
    solely on event management and distribution.

    Attributes:
        _subscribers: Dictionary mapping event types to lists of callback functions
        _lock: Threading lock for ensuring thread-safe operations
    """

    def __init__(self) -> None:
        """Initialize the EventHub with empty subscriber registry and thread lock.

        The constructor initializes the internal data structures required for
        event management:
        - A dictionary to store subscribers for each event type
        - A threading lock to ensure thread-safe operations

        The design follows the Dependency Inversion Principle by not depending
        on concrete implementations of subscribers.
        """
        self._subscribers: Dict[str, List[Callable[[Any], None]]] = {}
        self._lock: threading.RLock = threading.RLock()
        self._logger = logging.getLogger(__name__)

    def subscribe(self, event_type: str, callback: Callable[[Any], None]) -> None:
        """Subscribe to an event type with a callback function.

        Args:
            event_type: The type of event to subscribe to (use EventType constants)
            callback: The callback function to be called when the event is published

        Raises:
            ValueError: If event_type is empty or None
            TypeError: If callback is not callable

        Note:
            This method is thread-safe and can be called from multiple threads.
        """
        if not event_type:
            raise ValueError("Event type cannot be empty or None")

        if not callable(callback):
            raise TypeError("Callback must be callable")

        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []

            # Check if callback is already subscribed to avoid duplicates
            if callback not in self._subscribers[event_type]:
                self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable[[Any], None]) -> None:
        """Unsubscribe from an event type.

        Args:
            event_type: The type of event to unsubscribe from
            callback: The callback function to remove from subscribers

        Raises:
            ValueError: If event_type is empty or None
            KeyError: If the callback is not found in the subscribers list

        Note:
            This method is thread-safe and can be called from multiple threads.
        """
        if not event_type:
            raise ValueError("Event type cannot be empty or None")

        with self._lock:
            if event_type not in self._subscribers:
                raise KeyError(f"No subscribers found for event type: {event_type}")

            if callback not in self._subscribers[event_type]:
                raise KeyError(
                    f"Callback not found in subscribers for event type: {event_type}"
                )

            self._subscribers[event_type].remove(callback)

            # Clean up empty event type entries
            if not self._subscribers[event_type]:
                del self._subscribers[event_type]

    def publish(self, event_type: str, data: Any) -> None:
        """Publish an event to all subscribers.

        Args:
            event_type: The type of event to publish (use EventType constants)
            data: The event data to be passed to all subscribers

        Raises:
            ValueError: If event_type is empty or None

        Note:
            This method is thread-safe and can be called from multiple threads.
            Both synchronous and asynchronous callbacks are supported.
            Asynchronous callbacks are executed using asyncio.create_task().
        """
        if not event_type:
            raise ValueError("Event type cannot be empty or None")

        with self._lock:
            # Get a copy of subscribers to avoid issues if list is modified during iteration
            subscribers = self._subscribers.get(event_type, []).copy()

        # Execute callbacks outside of the lock to avoid blocking other operations
        for callback in subscribers:
            try:
                self._execute_callback_safely(callback, data, event_type)
            except Exception as e:
                # Log error but continue processing other callbacks
                self._logger.error(
                    f"Unexpected error executing callback for event {event_type}: {e}"
                )
                continue

    def _execute_callback_safely(
        self, callback: Callable[[Any], None], data: Any, event_type: str
    ) -> None:
        """Safely execute a callback with proper error handling.

        Args:
            callback: The callback function to execute
            data: The event data to pass to the callback
            event_type: The event type for logging purposes

        Note:
            This method handles both synchronous and asynchronous callbacks.
        """
        try:
            if inspect.iscoroutinefunction(callback):
                # Handle async callbacks
                self._execute_async_callback(callback, data)
            else:
                # Handle sync callbacks
                callback(data)
        except Exception as e:
            self._logger.error(f"Error executing callback for event {event_type}: {e}")

    def _execute_async_callback(
        self, callback: Callable[[Any], None], data: Any
    ) -> None:
        """Execute an asynchronous callback with proper event loop handling.

        Args:
            callback: The async callback function to execute
            data: The event data to pass to the callback
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # Schedule the callback to run in the current loop
            task = loop.create_task(callback(data))
            # Add error handling for the task
            task.add_done_callback(self._handle_async_callback_completion)
        except RuntimeError:
            # No event loop is running, run the callback in a new event loop
            # This is less efficient but necessary for compatibility
            try:
                asyncio.run(callback(data))
            except Exception as e:
                self._logger.error(
                    f"Error running async callback in new event loop: {e}"
                )

    def _handle_async_callback_completion(self, task: asyncio.Task) -> None:
        """Handle completion of an async callback task.

        Args:
            task: The completed asyncio Task
        """
        try:
            # Check if the task completed with an exception
            if task.exception() is not None:
                self._logger.error(f"Async callback failed: {task.exception()}")
        except asyncio.CancelledError:
            self._logger.warning("Async callback was cancelled")

    def get_subscriber_count(self, event_type: str) -> int:
        """Get the number of subscribers for a specific event type.

        Args:
            event_type: The event type to check

        Returns:
            The number of subscribers for the given event type

        Raises:
            ValueError: If event_type is empty or None

        Note:
            This method is thread-safe and useful for monitoring and debugging.
        """
        if not event_type:
            raise ValueError("Event type cannot be empty or None")

        with self._lock:
            return len(self._subscribers.get(event_type, []))

    def clear_subscribers(self, event_type: str = None) -> None:
        """Clear subscribers for a specific event type or all event types.

        Args:
            event_type: The event type to clear subscribers for.
                       If None, clears all subscribers for all event types.

        Raises:
            ValueError: If event_type is empty (but not None)

        Note:
            This method is thread-safe and useful for cleanup operations.
        """
        if event_type is not None and not event_type:
            raise ValueError("Event type cannot be empty")

        with self._lock:
            if event_type is None:
                # Clear all subscribers
                self._subscribers.clear()
            else:
                # Clear subscribers for specific event type
                if event_type in self._subscribers:
                    del self._subscribers[event_type]
