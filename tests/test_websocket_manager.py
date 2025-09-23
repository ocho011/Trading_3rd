"""
Unit tests for WebSocket manager module.

Tests WebSocket connection management, data validation, and event publishing
following test-driven development principles.
"""

import asyncio
import json
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest

from trading_bot.core.config_manager import ConfigManager
from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.market_data.websocket_manager import (
    BinanceWebSocketManager,
    ConnectionState,
    WebSocketConnectionError,
    WebSocketDataError,
    WebSocketDataValidator,
    create_binance_websocket_manager,
    create_websocket_manager,
)


class TestWebSocketDataValidator:
    """Test cases for WebSocketDataValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = WebSocketDataValidator()

    def test_validate_kline_data_valid(self):
        """Test validation of valid kline data."""
        valid_kline_data = {
            "e": "kline",
            "E": 1234567890,
            "s": "BTCUSDT",
            "k": {
                "t": 1234567800,
                "T": 1234567859,
                "o": "50000.00",
                "c": "50100.00",
                "h": "50200.00",
                "l": "49900.00",
                "v": "10.5",
            },
        }

        assert self.validator.validate_kline_data(valid_kline_data) is True

    def test_validate_kline_data_missing_fields(self):
        """Test validation of kline data with missing fields."""
        invalid_kline_data = {
            "e": "kline",
            "s": "BTCUSDT",
            "k": {
                "t": 1234567800,
                "o": "50000.00",
                # Missing required fields
            },
        }

        assert self.validator.validate_kline_data(invalid_kline_data) is False

    def test_validate_kline_data_no_kline_section(self):
        """Test validation of data without kline section."""
        invalid_data = {
            "e": "kline",
            "s": "BTCUSDT",
            # Missing 'k' section
        }

        assert self.validator.validate_kline_data(invalid_data) is False

    def test_validate_ticker_data_valid(self):
        """Test validation of valid ticker data."""
        valid_ticker_data = {
            "e": "24hrTicker",
            "E": 1234567890,
            "s": "BTCUSDT",
            "c": "50000.00",
            "h": "51000.00",
            "l": "49000.00",
            "v": "1000.0",
        }

        assert self.validator.validate_ticker_data(valid_ticker_data) is True

    def test_validate_ticker_data_missing_fields(self):
        """Test validation of ticker data with missing fields."""
        invalid_ticker_data = {
            "e": "24hrTicker",
            "s": "BTCUSDT",
            "c": "50000.00",
            # Missing required fields
        }

        assert self.validator.validate_ticker_data(invalid_ticker_data) is False

    def test_validate_ticker_data_exception_handling(self):
        """Test validation with malformed data causing exceptions."""
        malformed_data = None

        assert self.validator.validate_ticker_data(malformed_data) is False


class TestBinanceWebSocketManager:
    """Test cases for BinanceWebSocketManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = Mock(spec=ConfigManager)
        self.config_manager.get_trading_config.return_value = {"trading_mode": "paper"}

        self.event_hub = Mock(spec=EventHub)
        self.ws_manager = BinanceWebSocketManager(
            self.config_manager, self.event_hub, symbol="btcusdt"
        )

    def test_initialization(self):
        """Test WebSocket manager initialization."""
        assert self.ws_manager.get_connection_state() == ConnectionState.DISCONNECTED
        assert not self.ws_manager.is_connected()
        assert self.ws_manager._symbol == "btcusdt"

    def test_get_websocket_url_testnet(self):
        """Test WebSocket URL selection for testnet mode."""
        self.config_manager.get_trading_config.return_value = {"trading_mode": "paper"}

        url = self.ws_manager._get_websocket_url()
        assert url == BinanceWebSocketManager.TESTNET_WS_URL

    def test_get_websocket_url_mainnet(self):
        """Test WebSocket URL selection for mainnet mode."""
        self.config_manager.get_trading_config.return_value = {"trading_mode": "live"}

        url = self.ws_manager._get_websocket_url()
        assert url == BinanceWebSocketManager.MAINNET_WS_URL

    def test_get_stream_names(self):
        """Test generation of stream names."""
        stream_names = self.ws_manager._get_stream_names()

        expected_streams = ["btcusdt@kline_1m", "btcusdt@ticker"]
        assert stream_names == expected_streams

    def test_is_kline_message(self):
        """Test kline message detection."""
        kline_message = {"e": "kline", "k": {"t": 1234567800}}

        ticker_message = {"e": "24hrTicker"}

        assert self.ws_manager._is_kline_message(kline_message) is True
        assert self.ws_manager._is_kline_message(ticker_message) is False

    def test_is_ticker_message(self):
        """Test ticker message detection."""
        ticker_message = {"e": "24hrTicker"}

        kline_message = {"e": "kline"}

        assert self.ws_manager._is_ticker_message(ticker_message) is True
        assert self.ws_manager._is_ticker_message(kline_message) is False

    @pytest.mark.asyncio
    async def test_publish_market_data(self):
        """Test market data publishing through EventHub."""
        test_data = {"e": "kline", "E": 1234567890, "s": "BTCUSDT"}

        await self.ws_manager._publish_market_data(test_data, "kline")

        # Verify EventHub.publish was called
        self.event_hub.publish.assert_called_once()
        call_args = self.event_hub.publish.call_args

        assert call_args[0][0] == EventType.MARKET_DATA_RECEIVED
        published_data = call_args[0][1]

        assert published_data["source"] == "binance"
        assert published_data["symbol"] == "BTCUSDT"
        assert published_data["type"] == "kline"
        assert published_data["data"] == test_data

    @pytest.mark.asyncio
    async def test_process_message_valid_kline(self):
        """Test processing of valid kline message."""
        kline_message = json.dumps(
            {
                "e": "kline",
                "E": 1234567890,
                "s": "BTCUSDT",
                "k": {
                    "t": 1234567800,
                    "T": 1234567859,
                    "o": "50000.00",
                    "c": "50100.00",
                    "h": "50200.00",
                    "l": "49900.00",
                    "v": "10.5",
                },
            }
        )

        await self.ws_manager._process_message(kline_message)

        # Verify data was published
        self.event_hub.publish.assert_called_once_with(
            EventType.MARKET_DATA_RECEIVED, self.event_hub.publish.call_args[0][1]
        )

    @pytest.mark.asyncio
    async def test_process_message_valid_ticker(self):
        """Test processing of valid ticker message."""
        ticker_message = json.dumps(
            {
                "e": "24hrTicker",
                "E": 1234567890,
                "s": "BTCUSDT",
                "c": "50000.00",
                "h": "51000.00",
                "l": "49000.00",
                "v": "1000.0",
            }
        )

        await self.ws_manager._process_message(ticker_message)

        # Verify data was published
        self.event_hub.publish.assert_called_once_with(
            EventType.MARKET_DATA_RECEIVED, self.event_hub.publish.call_args[0][1]
        )

    @pytest.mark.asyncio
    async def test_process_message_invalid_json(self):
        """Test processing of invalid JSON message."""
        invalid_message = "invalid json {"

        with pytest.raises(WebSocketDataError):
            await self.ws_manager._process_message(invalid_message)

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self):
        """Test stopping WebSocket manager when not running."""
        # Should not raise exception
        await self.ws_manager.stop()

        assert self.ws_manager.get_connection_state() == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    @patch("websockets.connect")
    async def test_start_connection_failure(self, mock_connect):
        """Test WebSocket start with connection failure."""
        mock_connect.side_effect = Exception("Connection failed")

        with pytest.raises(WebSocketConnectionError):
            await self.ws_manager.start()

        assert self.ws_manager.get_connection_state() == ConnectionState.ERROR

    @pytest.mark.asyncio
    async def test_start_already_running(self):
        """Test starting WebSocket manager when already running."""
        self.ws_manager._running = True

        # Should not raise exception and should log warning
        await self.ws_manager.start()

    @pytest.mark.asyncio
    @patch("websockets.connect")
    async def test_websocket_lifecycle(self, mock_connect):
        """Test complete WebSocket lifecycle."""
        # Mock WebSocket connection
        mock_websocket = AsyncMock()
        mock_websocket.recv.side_effect = [
            json.dumps(
                {
                    "e": "kline",
                    "E": 1234567890,
                    "s": "BTCUSDT",
                    "k": {
                        "t": 1234567800,
                        "T": 1234567859,
                        "o": "50000.00",
                        "c": "50100.00",
                        "h": "50200.00",
                        "l": "49900.00",
                        "v": "10.5",
                    },
                }
            ),
            asyncio.CancelledError(),  # Simulate task cancellation
        ]

        mock_connect.return_value.__aenter__.return_value = mock_websocket

        # Start WebSocket in background task
        start_task = asyncio.create_task(self.ws_manager.start())

        # Let it run for a short time
        await asyncio.sleep(0.1)

        # Stop the WebSocket
        await self.ws_manager.stop()

        # Wait for start task to complete
        try:
            await start_task
        except (asyncio.CancelledError, WebSocketConnectionError):
            pass

        # Verify final state
        assert self.ws_manager.get_connection_state() == ConnectionState.DISCONNECTED
        assert not self.ws_manager.is_connected()


class TestWebSocketManagerFactory:
    """Test cases for WebSocket manager factory functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = Mock(spec=ConfigManager)
        self.event_hub = Mock(spec=EventHub)

    def test_create_binance_websocket_manager(self):
        """Test factory function for creating WebSocket manager."""
        ws_manager = create_binance_websocket_manager(
            self.config_manager, self.event_hub, symbol="ethusdt"
        )

        assert isinstance(ws_manager, BinanceWebSocketManager)
        assert ws_manager._symbol == "ethusdt"

    @pytest.mark.asyncio
    @patch("trading_bot.market_data.websocket_manager.BinanceWebSocketManager.start")
    @patch("trading_bot.market_data.websocket_manager.BinanceWebSocketManager.stop")
    async def test_create_websocket_manager_context(self, mock_stop, mock_start):
        """Test async context manager for WebSocket manager."""
        mock_start.return_value = None
        mock_stop.return_value = None

        async with create_websocket_manager(
            self.config_manager, self.event_hub, symbol="ethusdt"
        ) as ws_manager:
            assert isinstance(ws_manager, BinanceWebSocketManager)
            assert ws_manager._symbol == "ethusdt"

        # Verify start and stop were called
        mock_start.assert_called_once()
        mock_stop.assert_called_once()


class TestWebSocketManagerIntegration:
    """Integration tests for WebSocket manager with real dependencies."""

    def setup_method(self):
        """Set up integration test fixtures."""
        # Create real instances for integration testing
        config_loader = Mock()
        config_loader.load_config.return_value = {
            "trading_mode": "paper",
            "log_level": "INFO",
        }

        self.config_manager = ConfigManager(config_loader)
        self.config_manager.load_configuration()

        self.event_hub = EventHub()
        self.received_events = []

        # Subscribe to market data events
        self.event_hub.subscribe(
            EventType.MARKET_DATA_RECEIVED, self._on_market_data_received
        )

    def _on_market_data_received(self, data: Dict[str, Any]) -> None:
        """Handle received market data events."""
        self.received_events.append(data)

    @pytest.mark.asyncio
    async def test_websocket_manager_with_real_event_hub(self):
        """Test WebSocket manager with real EventHub instance."""
        ws_manager = BinanceWebSocketManager(
            self.config_manager, self.event_hub, symbol="btcusdt"
        )

        # Simulate processing a market data message
        test_data = {
            "e": "kline",
            "E": 1234567890,
            "s": "BTCUSDT",
            "k": {
                "t": 1234567800,
                "T": 1234567859,
                "o": "50000.00",
                "c": "50100.00",
                "h": "50200.00",
                "l": "49900.00",
                "v": "10.5",
            },
        }

        await ws_manager._publish_market_data(test_data, "kline")

        # Verify event was received
        assert len(self.received_events) == 1
        received_data = self.received_events[0]

        assert received_data["source"] == "binance"
        assert received_data["symbol"] == "BTCUSDT"
        assert received_data["type"] == "kline"
        assert received_data["data"] == test_data

    def test_websocket_manager_configuration_dependency(self):
        """Test WebSocket manager dependency on configuration."""
        ws_manager = BinanceWebSocketManager(self.config_manager, self.event_hub)

        # Test URL selection based on configuration
        url = ws_manager._get_websocket_url()
        assert url == BinanceWebSocketManager.TESTNET_WS_URL

        # Change configuration and test again
        self.config_manager._config["trading_mode"] = "live"
        url = ws_manager._get_websocket_url()
        assert url == BinanceWebSocketManager.MAINNET_WS_URL
