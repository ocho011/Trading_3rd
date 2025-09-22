"""
WebSocket manager for real-time market data streaming from cryptocurrency exchanges.

Provides async WebSocket connection management with proper error handling,
connection state monitoring, and event-driven data publishing following
SOLID principles and dependency injection patterns.
"""

import asyncio
import json
import random
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from trading_bot.core.config_manager import ConfigManager
from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.core.logger import get_module_logger


class WebSocketError(Exception):
    """Base exception for WebSocket-related errors."""

    pass


class WebSocketConnectionError(WebSocketError):
    """Exception raised for WebSocket connection failures."""

    pass


class WebSocketDataError(WebSocketError):
    """Exception raised for WebSocket data parsing failures."""

    pass


class ConnectionState(Enum):
    """WebSocket connection state enumeration."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class ReconnectionConfig:
    """Configuration for WebSocket auto-reconnection behavior.

    This class encapsulates all reconnection-related settings following
    the Single Responsibility Principle.
    """

    def __init__(
        self,
        enabled: bool = True,
        max_retries: int = 10,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_multiplier: float = 2.0,
        jitter_factor: float = 0.25,
    ) -> None:
        """Initialize reconnection configuration.

        Args:
            enabled: Whether auto-reconnection is enabled
            max_retries: Maximum number of reconnection attempts
            initial_delay: Initial delay in seconds before reconnection
            max_delay: Maximum delay in seconds between reconnection attempts
            backoff_multiplier: Multiplier for exponential backoff
            jitter_factor: Random jitter factor (±percentage of delay)
        """
        self.enabled = enabled
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter_factor = jitter_factor


class IWebSocketManager(ABC):
    """Interface for WebSocket manager implementations.

    This interface defines the contract for WebSocket managers,
    following the Interface Segregation Principle.
    """

    @abstractmethod
    async def start(self) -> None:
        """Start WebSocket connection and begin streaming data."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop WebSocket connection and cleanup resources."""
        pass

    @abstractmethod
    def get_connection_state(self) -> ConnectionState:
        """Get current connection state."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if WebSocket is currently connected."""
        pass


class WebSocketDataValidator:
    """Validates WebSocket message data format and content.

    This class follows the Single Responsibility Principle by focusing
    solely on data validation logic.
    """

    def __init__(self) -> None:
        """Initialize WebSocket data validator."""
        self._logger = get_module_logger("websocket_data_validator")

    def validate_kline_data(self, data: Dict[str, Any]) -> bool:
        """Validate kline (candlestick) data format.

        Args:
            data: Raw kline data from WebSocket

        Returns:
            True if data is valid, False otherwise
        """
        try:
            required_fields = ["s", "k"]  # symbol, kline data
            if not all(field in data for field in required_fields):
                return False

            kline = data.get("k", {})
            kline_fields = ["t", "T", "o", "c", "h", "l", "v"]
            return all(field in kline for field in kline_fields)

        except Exception as e:
            self._logger.error(f"Error validating kline data: {e}")
            return False

    def validate_ticker_data(self, data: Dict[str, Any]) -> bool:
        """Validate ticker data format.

        Args:
            data: Raw ticker data from WebSocket

        Returns:
            True if data is valid, False otherwise
        """
        try:
            # symbol, price, high, low, volume
            required_fields = ["s", "c", "h", "l", "v"]
            return all(field in data for field in required_fields)

        except Exception as e:
            self._logger.error(f"Error validating ticker data: {e}")
            return False


class BinanceWebSocketManager(IWebSocketManager):
    """Binance WebSocket manager for real-time market data streaming.

    This class manages WebSocket connections to Binance streams and publishes
    received data through the EventHub following the Observer pattern.

    Attributes:
        _config_manager: Configuration manager for settings
        _event_hub: Event hub for publishing market data
        _logger: Module-specific logger
        _connection_state: Current WebSocket connection state
        _websocket: WebSocket connection instance
        _subscribed_streams: Set of currently subscribed streams
        _data_validator: Data validation helper
    """

    # Binance WebSocket URLs
    MAINNET_WS_URL = "wss://stream.binance.com:9443/ws/"
    TESTNET_WS_URL = "wss://testnet.binance.vision/ws/"

    def __init__(
        self,
        config_manager: ConfigManager,
        event_hub: EventHub,
        symbol: str = "btcusdt",
        reconnection_config: Optional[ReconnectionConfig] = None,
    ) -> None:
        """Initialize Binance WebSocket manager.

        Args:
            config_manager: Configuration manager instance
            event_hub: Event hub for publishing data
            symbol: Trading symbol to subscribe to (default: btcusdt)
            reconnection_config: Auto-reconnection configuration
        """
        self._config_manager = config_manager
        self._event_hub = event_hub
        self._symbol = symbol.lower()
        self._logger = get_module_logger("binance_websocket_manager")

        self._connection_state = ConnectionState.DISCONNECTED
        self._websocket: Optional[Any] = None
        self._subscribed_streams: Set[str] = set()
        self._data_validator = WebSocketDataValidator()
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Reconnection configuration and state
        self._reconnection_config = (
            reconnection_config or self._load_reconnection_config()
        )
        self._retry_count = 0
        self._last_connection_time = 0.0
        self._manual_disconnect = False

    def get_connection_state(self) -> ConnectionState:
        """Get current WebSocket connection state.

        Returns:
            Current connection state
        """
        return self._connection_state

    def is_connected(self) -> bool:
        """Check if WebSocket is currently connected.

        Returns:
            True if connected, False otherwise
        """
        return self._connection_state == ConnectionState.CONNECTED

    async def start(self) -> None:
        """Start WebSocket connection and begin streaming data.

        Raises:
            WebSocketConnectionError: If connection fails
        """
        if self._running:
            self._logger.warning("WebSocket manager already running")
            return

        self._running = True
        self._manual_disconnect = False
        self._retry_count = 0
        self._logger.info("Starting Binance WebSocket manager")

        try:
            await self._connect_and_stream()
        except Exception as e:
            self._connection_state = ConnectionState.ERROR
            if not self._should_reconnect():
                raise WebSocketConnectionError(f"Failed to start WebSocket: {e}")
            # Let reconnection logic handle the error
            await self._handle_connection_error(e)

    async def stop(self) -> None:
        """Stop WebSocket connection and cleanup resources."""
        if not self._running:
            return

        self._logger.info("Stopping Binance WebSocket manager")
        self._running = False
        self._manual_disconnect = True

        # Cancel all running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._tasks.clear()

        # Close WebSocket connection
        await self._disconnect()
        self._connection_state = ConnectionState.DISCONNECTED
        self._logger.info("WebSocket manager stopped")

    def _get_websocket_url(self) -> str:
        """Get WebSocket URL based on trading mode.

        Returns:
            WebSocket URL for current trading mode
        """
        trading_config = self._config_manager.get_trading_config()
        trading_mode = trading_config.get("trading_mode", "paper")

        if trading_mode == "live":
            return self.MAINNET_WS_URL
        else:
            return self.TESTNET_WS_URL

    def _get_stream_names(self) -> List[str]:
        """Get list of stream names to subscribe to.

        Returns:
            List of Binance stream names
        """
        return [
            f"{self._symbol}@kline_1m",  # 1-minute kline data
            f"{self._symbol}@ticker",  # 24hr ticker statistics
        ]

    def _load_reconnection_config(self) -> ReconnectionConfig:
        """Load reconnection configuration from config manager.

        Returns:
            ReconnectionConfig instance with loaded settings
        """
        try:
            # Load configuration values using ConfigManager's interface
            enabled = self._config_manager.get_config_value(
                "websocket_reconnection_enabled", True
            )
            max_retries = self._config_manager.get_config_value(
                "websocket_max_retries", 10
            )
            initial_delay = self._config_manager.get_config_value(
                "websocket_initial_delay", 1.0
            )
            max_delay = self._config_manager.get_config_value(
                "websocket_max_delay", 60.0
            )
            backoff_multiplier = self._config_manager.get_config_value(
                "websocket_backoff_multiplier", 2.0
            )
            jitter_factor = self._config_manager.get_config_value(
                "websocket_jitter_factor", 0.25
            )

            return ReconnectionConfig(
                enabled=bool(enabled),
                max_retries=int(max_retries),
                initial_delay=float(initial_delay),
                max_delay=float(max_delay),
                backoff_multiplier=float(backoff_multiplier),
                jitter_factor=float(jitter_factor),
            )
        except Exception as e:
            self._logger.warning(f"Failed to load reconnection config: {e}")
            return ReconnectionConfig()

    def _should_reconnect(self) -> bool:
        """Determine if reconnection should be attempted.

        Returns:
            True if reconnection should be attempted
        """
        if not self._reconnection_config.enabled:
            return False
        if self._manual_disconnect:
            return False
        if not self._running:
            return False
        if self._retry_count >= self._reconnection_config.max_retries:
            self._logger.error(
                f"Max reconnection attempts "
                f"({self._reconnection_config.max_retries}) reached"
            )
            return False
        return True

    def _calculate_backoff_delay(self) -> float:
        """Calculate delay with exponential backoff and jitter.

        Returns:
            Delay in seconds before next reconnection attempt
        """
        exponential_delay = self._reconnection_config.initial_delay * (
            self._reconnection_config.backoff_multiplier**self._retry_count
        )
        delay = min(exponential_delay, self._reconnection_config.max_delay)

        # Add jitter to prevent thundering herd
        jitter_range = delay * self._reconnection_config.jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)

        return max(0.1, delay + jitter)

    async def _attempt_reconnection(self) -> bool:
        """Attempt to reconnect with exponential backoff.

        Returns:
            True if reconnection successful, False otherwise
        """
        if not self._should_reconnect():
            return False

        self._retry_count += 1
        delay = self._calculate_backoff_delay()

        self._logger.info(
            f"Attempting reconnection {self._retry_count}/"
            f"{self._reconnection_config.max_retries} in {delay:.1f}s"
        )

        self._connection_state = ConnectionState.RECONNECTING
        await asyncio.sleep(delay)

        try:
            await self._connect_without_reconnection()
            self._retry_count = 0  # Reset on successful connection
            self._last_connection_time = time.time()
            return True

        except Exception as e:
            self._logger.warning(
                f"Reconnection attempt {self._retry_count} failed: {e}"
            )
            return False

    async def _handle_connection_error(self, error: Exception) -> None:
        """Handle connection errors with potential reconnection.

        Args:
            error: The connection error that occurred
        """
        self._logger.error(f"WebSocket connection error: {error}")
        self._connection_state = ConnectionState.ERROR

        if self._should_reconnect():
            self._logger.info("Initiating automatic reconnection...")
            reconnection_successful = await self._attempt_reconnection()

            if not reconnection_successful and self._should_reconnect():
                # Schedule next reconnection attempt
                reconnect_task = asyncio.create_task(self._attempt_reconnection())
                self._tasks.append(reconnect_task)
        else:
            self._logger.error("Auto-reconnection disabled or max retries reached")

    async def _connect_without_reconnection(self) -> None:
        """Connect to WebSocket without triggering reconnection logic.

        This method is used during reconnection attempts to avoid infinite recursion.
        """
        self._connection_state = ConnectionState.CONNECTING

        base_url = self._get_websocket_url()
        stream_names = self._get_stream_names()
        stream_path = "/".join(stream_names)
        ws_url = f"{base_url}{stream_path}"

        self._logger.info(f"Connecting to WebSocket: {ws_url}")

        async with websockets.connect(
            ws_url, ping_interval=20, ping_timeout=10, close_timeout=10
        ) as websocket:
            self._websocket = websocket
            self._connection_state = ConnectionState.CONNECTED
            self._subscribed_streams.update(stream_names)

            # Reset reconnection state on successful connection
            self._retry_count = 0
            self._last_connection_time = time.time()

            self._logger.info("WebSocket connected successfully")

            # Start data streaming task
            stream_task = asyncio.create_task(self._stream_data())
            self._tasks.append(stream_task)

            await stream_task

    async def _connect_and_stream(self) -> None:
        """Connect to WebSocket and start streaming data."""
        self._connection_state = ConnectionState.CONNECTING

        base_url = self._get_websocket_url()
        stream_names = self._get_stream_names()
        stream_path = "/".join(stream_names)
        ws_url = f"{base_url}{stream_path}"

        self._logger.info(f"Connecting to WebSocket: {ws_url}")

        try:
            async with websockets.connect(
                ws_url, ping_interval=20, ping_timeout=10, close_timeout=10
            ) as websocket:
                self._websocket = websocket
                self._connection_state = ConnectionState.CONNECTED
                self._subscribed_streams.update(stream_names)

                # Reset reconnection state on successful connection
                self._retry_count = 0
                self._last_connection_time = time.time()

                self._logger.info("WebSocket connected successfully")

                # Start data streaming task
                stream_task = asyncio.create_task(self._stream_data())
                self._tasks.append(stream_task)

                await stream_task

        except ConnectionClosed as e:
            await self._handle_connection_error(e)
        except WebSocketException as e:
            await self._handle_connection_error(e)
        except Exception as e:
            await self._handle_connection_error(e)

    async def _stream_data(self) -> None:
        """Stream data from WebSocket connection."""
        if not self._websocket:
            raise WebSocketConnectionError("WebSocket not connected")

        self._logger.info("Starting data streaming")

        try:
            while self._running and self._websocket:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(
                        self._websocket.recv(), timeout=30.0
                    )
                    await self._process_message(message)

                except asyncio.TimeoutError:
                    self._logger.warning("WebSocket message timeout")
                    if self._should_reconnect():
                        raise WebSocketConnectionError(
                            "Message timeout - triggering reconnection"
                        )
                    continue

                except ConnectionClosed as e:
                    self._logger.warning("WebSocket connection closed during streaming")
                    if self._should_reconnect():
                        raise WebSocketConnectionError(
                            f"Connection closed during streaming: {e}"
                        )
                    break

        except Exception as e:
            self._logger.error(f"Error in data streaming: {e}")
            raise

    async def _process_message(self, message: str) -> None:
        """Process incoming WebSocket message.

        Args:
            message: Raw message from WebSocket
        """
        try:
            data = json.loads(message)

            # Determine message type and validate
            if self._is_kline_message(data):
                if self._data_validator.validate_kline_data(data):
                    await self._publish_market_data(data, "kline")
                else:
                    self._logger.warning("Invalid kline data received")

            elif self._is_ticker_message(data):
                if self._data_validator.validate_ticker_data(data):
                    await self._publish_market_data(data, "ticker")
                else:
                    self._logger.warning("Invalid ticker data received")

            else:
                self._logger.debug(f"Unknown message type: {data}")

        except json.JSONDecodeError as e:
            self._logger.error(f"Failed to parse JSON message: {e}")
            raise WebSocketDataError(f"JSON parse error: {e}")
        except Exception as e:
            self._logger.error(f"Error processing message: {e}")
            raise WebSocketDataError(f"Message processing error: {e}")

    def _is_kline_message(self, data: Dict[str, Any]) -> bool:
        """Check if message is kline data.

        Args:
            data: Parsed message data

        Returns:
            True if message contains kline data
        """
        return "k" in data and data.get("e") == "kline"

    def _is_ticker_message(self, data: Dict[str, Any]) -> bool:
        """Check if message is ticker data.

        Args:
            data: Parsed message data

        Returns:
            True if message contains ticker data
        """
        return data.get("e") == "24hrTicker"

    async def _publish_market_data(self, data: Dict[str, Any], data_type: str) -> None:
        """Publish market data through EventHub.

        Args:
            data: Market data to publish
            data_type: Type of market data (kline, ticker)
        """
        try:
            market_data = {
                "source": "binance",
                "symbol": self._symbol.upper(),
                "type": data_type,
                "data": data,
                "timestamp": data.get("E", 0),  # Event time
            }

            self._event_hub.publish(EventType.MARKET_DATA_RECEIVED, market_data)
            self._logger.debug(f"Published {data_type} data for {self._symbol.upper()}")

        except Exception as e:
            self._logger.error(f"Error publishing market data: {e}")

    async def _disconnect(self) -> None:
        """Disconnect from WebSocket."""
        if self._websocket:
            try:
                await self._websocket.close()
                self._logger.info("WebSocket disconnected")
            except Exception as e:
                self._logger.error(f"Error closing WebSocket: {e}")
            finally:
                self._websocket = None
                self._subscribed_streams.clear()


@asynccontextmanager
async def create_websocket_manager(
    config_manager: ConfigManager,
    event_hub: EventHub,
    symbol: str = "btcusdt",
    reconnection_config: Optional[ReconnectionConfig] = None,
):
    """Async context manager for WebSocket manager lifecycle.

    Args:
        config_manager: Configuration manager instance
        event_hub: Event hub instance
        symbol: Trading symbol to monitor
        reconnection_config: Auto-reconnection configuration

    Yields:
        BinanceWebSocketManager: Configured and started WebSocket manager

    Example:
        async with create_websocket_manager(config, hub) as ws_manager:
            # WebSocket manager is running
            await asyncio.sleep(10)
        # WebSocket manager is automatically stopped
    """
    ws_manager = BinanceWebSocketManager(
        config_manager, event_hub, symbol, reconnection_config
    )

    try:
        await ws_manager.start()
        yield ws_manager
    finally:
        await ws_manager.stop()


def create_binance_websocket_manager(
    config_manager: ConfigManager,
    event_hub: EventHub,
    symbol: str = "btcusdt",
    reconnection_config: Optional[ReconnectionConfig] = None,
) -> BinanceWebSocketManager:
    """Factory function to create BinanceWebSocketManager instance.

    Args:
        config_manager: Configuration manager instance
        event_hub: Event hub instance
        symbol: Trading symbol to monitor
        reconnection_config: Auto-reconnection configuration

    Returns:
        BinanceWebSocketManager: Configured WebSocket manager instance
    """
    return BinanceWebSocketManager(
        config_manager, event_hub, symbol, reconnection_config
    )
