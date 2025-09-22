"""
Unit tests for WebSocket auto-reconnection functionality.

Tests the auto-reconnection logic, exponential backoff, configuration loading,
and error handling following comprehensive testing practices.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from websockets.exceptions import ConnectionClosed, WebSocketException

from trading_bot.core.config_manager import ConfigManager
from trading_bot.core.event_hub import EventHub
from trading_bot.market_data.websocket_manager import (
    BinanceWebSocketManager,
    ReconnectionConfig,
    ConnectionState,
    WebSocketConnectionError,
    create_binance_websocket_manager
)


class TestReconnectionConfig:
    """Test ReconnectionConfig class functionality."""

    def test_default_configuration(self):
        """Test default reconnection configuration values."""
        config = ReconnectionConfig()

        assert config.enabled is True
        assert config.max_retries == 10
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_multiplier == 2.0
        assert config.jitter_factor == 0.25

    def test_custom_configuration(self):
        """Test custom reconnection configuration values."""
        config = ReconnectionConfig(
            enabled=False,
            max_retries=5,
            initial_delay=2.0,
            max_delay=30.0,
            backoff_multiplier=1.5,
            jitter_factor=0.1
        )

        assert config.enabled is False
        assert config.max_retries == 5
        assert config.initial_delay == 2.0
        assert config.max_delay == 30.0
        assert config.backoff_multiplier == 1.5
        assert config.jitter_factor == 0.1


class TestBinanceWebSocketManagerReconnection:
    """Test BinanceWebSocketManager auto-reconnection functionality."""

    @pytest.fixture
    def mock_config_manager(self):
        """Create mock configuration manager."""
        config_manager = MagicMock(spec=ConfigManager)

        # Mock the get_config_value method for different configuration keys
        def mock_get_config_value(key, default=None):
            config_values = {
                "websocket_reconnection_enabled": True,
                "websocket_max_retries": 3,
                "websocket_initial_delay": 0.1,
                "websocket_max_delay": 1.0,
                "websocket_backoff_multiplier": 2.0,
                "websocket_jitter_factor": 0.1
            }
            return config_values.get(key, default)

        config_manager.get_config_value.side_effect = mock_get_config_value
        config_manager.get_trading_config.return_value = {"trading_mode": "paper"}
        return config_manager

    @pytest.fixture
    def mock_event_hub(self):
        """Create mock event hub."""
        return MagicMock(spec=EventHub)

    @pytest.fixture
    def ws_manager(self, mock_config_manager, mock_event_hub):
        """Create WebSocket manager instance."""
        return BinanceWebSocketManager(
            mock_config_manager,
            mock_event_hub,
            "btcusdt"
        )

    def test_reconnection_config_loading(self, ws_manager):
        """Test reconnection configuration is loaded correctly."""
        config = ws_manager._reconnection_config

        assert config.enabled is True
        assert config.max_retries == 3
        assert config.initial_delay == 0.1
        assert config.max_delay == 1.0
        assert config.backoff_multiplier == 2.0
        assert config.jitter_factor == 0.1

    def test_reconnection_config_loading_failure(self, mock_event_hub):
        """Test fallback to default config on loading failure."""
        config_manager = MagicMock(spec=ConfigManager)
        config_manager.get_config_value.side_effect = Exception("Config error")
        config_manager.get_trading_config.return_value = {"trading_mode": "paper"}

        ws_manager = BinanceWebSocketManager(
            config_manager,
            mock_event_hub,
            "btcusdt"
        )

        # Should use default configuration
        config = ws_manager._reconnection_config
        assert config.enabled is True
        assert config.max_retries == 10

    def test_should_reconnect_conditions(self, ws_manager):
        """Test _should_reconnect logic under various conditions."""
        # Should reconnect when enabled and running
        ws_manager._running = True
        ws_manager._manual_disconnect = False
        ws_manager._retry_count = 0
        assert ws_manager._should_reconnect() is True

        # Should not reconnect when disabled
        ws_manager._reconnection_config.enabled = False
        assert ws_manager._should_reconnect() is False
        ws_manager._reconnection_config.enabled = True

        # Should not reconnect when manually disconnected
        ws_manager._manual_disconnect = True
        assert ws_manager._should_reconnect() is False
        ws_manager._manual_disconnect = False

        # Should not reconnect when not running
        ws_manager._running = False
        assert ws_manager._should_reconnect() is False
        ws_manager._running = True

        # Should not reconnect when max retries exceeded
        ws_manager._retry_count = 5
        assert ws_manager._should_reconnect() is False

    def test_calculate_backoff_delay(self, ws_manager):
        """Test exponential backoff delay calculation."""
        ws_manager._retry_count = 0
        delay1 = ws_manager._calculate_backoff_delay()
        assert 0.09 <= delay1 <= 0.11  # 0.1 ± 10% jitter

        ws_manager._retry_count = 1
        delay2 = ws_manager._calculate_backoff_delay()
        assert 0.18 <= delay2 <= 0.22  # 0.2 ± 10% jitter

        ws_manager._retry_count = 2
        delay3 = ws_manager._calculate_backoff_delay()
        assert 0.36 <= delay3 <= 0.44  # 0.4 ± 10% jitter

        # Test max delay cap
        ws_manager._retry_count = 10
        delay_max = ws_manager._calculate_backoff_delay()
        assert delay_max <= 1.1  # Should be capped at max_delay + jitter

    @pytest.mark.asyncio
    async def test_manual_disconnect_stops_reconnection(self, ws_manager):
        """Test that manual disconnect prevents reconnection."""
        ws_manager._running = True
        ws_manager._manual_disconnect = False

        # Stop should set manual_disconnect flag
        await ws_manager.stop()

        assert ws_manager._manual_disconnect is True
        assert ws_manager._should_reconnect() is False

    @pytest.mark.asyncio
    async def test_start_initializes_reconnection_state(self, ws_manager):
        """Test that start() properly initializes reconnection state."""
        with patch.object(ws_manager, '_connect_and_stream') as mock_connect:
            mock_connect.side_effect = ConnectionClosed(None, None)

            with patch.object(ws_manager, '_should_reconnect', return_value=False):
                try:
                    await ws_manager.start()
                except:
                    pass

            assert ws_manager._manual_disconnect is False
            assert ws_manager._retry_count == 0
            assert ws_manager._running is True

    @pytest.mark.asyncio
    async def test_connection_error_triggers_reconnection(self, ws_manager):
        """Test that connection errors trigger reconnection logic."""
        ws_manager._running = True
        ws_manager._manual_disconnect = False

        error = ConnectionClosed(None, None)

        with patch.object(ws_manager, '_should_reconnect', return_value=True):
            with patch.object(ws_manager, '_attempt_reconnection') as mock_reconnect:
                mock_reconnect.return_value = True

                await ws_manager._handle_connection_error(error)

                mock_reconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_error_no_reconnection_when_disabled(self, ws_manager):
        """Test no reconnection when auto-reconnection is disabled."""
        ws_manager._running = True
        ws_manager._reconnection_config.enabled = False

        error = ConnectionClosed(None, None)

        with patch.object(ws_manager, '_attempt_reconnection') as mock_reconnect:
            await ws_manager._handle_connection_error(error)

            mock_reconnect.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_reconnection_resets_retry_count(self, ws_manager):
        """Test that successful reconnection resets retry count."""
        ws_manager._retry_count = 2
        ws_manager._running = True

        with patch.object(ws_manager, '_connect_without_reconnection') as mock_connect:
            mock_connect.return_value = None

            result = await ws_manager._attempt_reconnection()

            assert result is True
            assert ws_manager._retry_count == 0

    @pytest.mark.asyncio
    async def test_failed_reconnection_increments_retry_count(self, ws_manager):
        """Test that failed reconnection increments retry count."""
        initial_retry_count = ws_manager._retry_count
        ws_manager._running = True

        with patch.object(ws_manager, '_connect_without_reconnection') as mock_connect:
            mock_connect.side_effect = ConnectionClosed(None, None)

            result = await ws_manager._attempt_reconnection()

            assert result is False
            assert ws_manager._retry_count == initial_retry_count + 1

    @pytest.mark.asyncio
    async def test_stream_data_timeout_triggers_reconnection(self, ws_manager):
        """Test that streaming timeout triggers reconnection when enabled."""
        ws_manager._running = True
        ws_manager._websocket = MagicMock()

        with patch.object(ws_manager, '_should_reconnect', return_value=True):
            with patch('asyncio.wait_for') as mock_wait_for:
                mock_wait_for.side_effect = asyncio.TimeoutError()

                with pytest.raises(WebSocketConnectionError):
                    await ws_manager._stream_data()

    @pytest.mark.asyncio
    async def test_stream_data_connection_closed_triggers_reconnection(self, ws_manager):
        """Test that connection closed during streaming triggers reconnection."""
        ws_manager._running = True
        ws_manager._websocket = MagicMock()

        with patch.object(ws_manager, '_should_reconnect', return_value=True):
            with patch('asyncio.wait_for') as mock_wait_for:
                mock_wait_for.side_effect = ConnectionClosed(None, None)

                with pytest.raises(WebSocketConnectionError):
                    await ws_manager._stream_data()

    def test_custom_reconnection_config_via_constructor(self, mock_config_manager, mock_event_hub):
        """Test custom reconnection config can be passed via constructor."""
        custom_config = ReconnectionConfig(
            enabled=False,
            max_retries=5,
            initial_delay=2.0
        )

        ws_manager = BinanceWebSocketManager(
            mock_config_manager,
            mock_event_hub,
            "ethusdt",
            custom_config
        )

        assert ws_manager._reconnection_config.enabled is False
        assert ws_manager._reconnection_config.max_retries == 5
        assert ws_manager._reconnection_config.initial_delay == 2.0

    def test_factory_function_with_reconnection_config(self, mock_config_manager, mock_event_hub):
        """Test factory function supports reconnection configuration."""
        custom_config = ReconnectionConfig(max_retries=7)

        ws_manager = create_binance_websocket_manager(
            mock_config_manager,
            mock_event_hub,
            "adausdt",
            custom_config
        )

        assert isinstance(ws_manager, BinanceWebSocketManager)
        assert ws_manager._reconnection_config.max_retries == 7

    @pytest.mark.asyncio
    async def test_reconnection_maintains_subscribed_streams(self, ws_manager):
        """Test that reconnection maintains previously subscribed streams."""
        original_streams = {"btcusdt@kline_1m", "btcusdt@ticker"}
        ws_manager._subscribed_streams = original_streams.copy()

        with patch.object(ws_manager, '_get_stream_names', return_value=list(original_streams)):
            with patch('websockets.connect') as mock_connect:
                mock_websocket = AsyncMock()
                mock_connect.return_value.__aenter__.return_value = mock_websocket
                mock_connect.return_value.__aexit__.return_value = None

                with patch.object(ws_manager, '_stream_data') as mock_stream:
                    mock_stream.return_value = None

                    await ws_manager._connect_without_reconnection()

                    # Should maintain subscribed streams
                    assert ws_manager._subscribed_streams == original_streams


if __name__ == "__main__":
    pytest.main([__file__])