"""
Market data module for trading bot.

Handles real-time and historical market data collection, processing,
and management from various cryptocurrency exchanges.
"""

from .binance_client import BinanceClient, create_binance_client
from .data_processor import (BinanceDataValidator, CandleAggregationError,
                           CandleData, DataProcessingError,
                           ICandleAggregator, IMarketDataProcessor,
                           IMarketDataValidator, InvalidDataError,
                           MarketData, MarketDataProcessor, Timeframe,
                           TimeframeCandleAggregator, TimeframeError,
                           create_market_data_processor)
from .websocket_manager import (BinanceWebSocketManager, ConnectionState,
                                IWebSocketManager, WebSocketConnectionError,
                                WebSocketDataError, WebSocketDataValidator,
                                create_binance_websocket_manager,
                                create_websocket_manager)

__all__ = [
    # Binance client
    "BinanceClient",
    "create_binance_client",
    # Data processor
    "BinanceDataValidator",
    "CandleAggregationError",
    "CandleData",
    "DataProcessingError",
    "ICandleAggregator",
    "IMarketDataProcessor",
    "IMarketDataValidator",
    "InvalidDataError",
    "MarketData",
    "MarketDataProcessor",
    "Timeframe",
    "TimeframeCandleAggregator",
    "TimeframeError",
    "create_market_data_processor",
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
