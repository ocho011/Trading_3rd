"""
Market data module for trading bot.

Handles real-time and historical market data collection, processing,
and management from various cryptocurrency exchanges.
"""

from .binance_client import BinanceClient, create_binance_client
from .websocket_manager import (BinanceWebSocketManager, ConnectionState,
                                IWebSocketManager, WebSocketConnectionError,
                                WebSocketDataError, WebSocketDataValidator,
                                create_binance_websocket_manager,
                                create_websocket_manager)

__all__ = [
    # Binance client
    "BinanceClient",
    "create_binance_client",
    # WebSocket manager
    "BinanceWebSocketManager",
    "ConnectionState",
    "IWebSocketManager",
    "WebSocketConnectionError",
    "WebSocketDataError",
    "WebSocketDataValidator",
    "create_binance_websocket_manager",
    "create_websocket_manager",
]
