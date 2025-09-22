"""
Market Data Processor for real-time trading data aggregation and processing.

Provides comprehensive market data processing with candle aggregation,
data validation, and event-driven publishing following SOLID principles
and dependency injection patterns.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from trading_bot.core.config_manager import ConfigManager
from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.core.logger import get_module_logger


class DataProcessingError(Exception):
    """Base exception for data processing errors."""

    pass


class InvalidDataError(DataProcessingError):
    """Exception raised for invalid or malformed data."""

    pass


class TimeframeError(DataProcessingError):
    """Exception raised for timeframe-related errors."""

    pass


class CandleAggregationError(DataProcessingError):
    """Exception raised for candle aggregation errors."""

    pass


class Timeframe(Enum):
    """Supported timeframe intervals for candle aggregation."""

    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"

    @property
    def seconds(self) -> int:
        """Get timeframe duration in seconds."""
        timeframe_seconds = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
        }
        return timeframe_seconds[self.value]

    @classmethod
    def from_string(cls, timeframe_str: str) -> "Timeframe":
        """Create Timeframe from string value.

        Args:
            timeframe_str: String representation of timeframe

        Returns:
            Timeframe: Corresponding timeframe enum

        Raises:
            TimeframeError: If timeframe string is invalid
        """
        for tf in cls:
            if tf.value == timeframe_str:
                return tf
        raise TimeframeError(f"Invalid timeframe: {timeframe_str}")


@dataclass
class MarketData:
    """Market data structure for standardized data representation.

    This dataclass provides a standardized format for market data
    following the Data Transfer Object pattern.
    """

    symbol: str
    timestamp: int
    price: float
    volume: float
    source: str
    data_type: str
    raw_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate data after initialization."""
        if self.price <= 0:
            raise InvalidDataError("Price must be positive")
        if self.volume < 0:
            raise InvalidDataError("Volume cannot be negative")
        if not self.symbol:
            raise InvalidDataError("Symbol cannot be empty")


@dataclass
class CandleData:
    """OHLCV candle data structure for aggregated market data.

    This dataclass represents Open, High, Low, Close, Volume data
    for a specific timeframe and symbol.
    """

    symbol: str
    timeframe: Timeframe
    open_time: int
    close_time: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    trade_count: int = 0
    is_closed: bool = False
    last_update: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        """Validate candle data after initialization."""
        if any(
            p <= 0
            for p in [
                self.open_price,
                self.high_price,
                self.low_price,
                self.close_price,
            ]
        ):
            raise InvalidDataError("All prices must be positive")
        if self.volume < 0:
            raise InvalidDataError("Volume cannot be negative")
        if self.high_price < max(self.open_price, self.close_price):
            raise InvalidDataError("High price cannot be less than open/close")
        if self.low_price > min(self.open_price, self.close_price):
            raise InvalidDataError(
                "Low price cannot be greater than open/close"
            )


class IMarketDataValidator(ABC):
    """Interface for market data validation strategies."""

    @abstractmethod
    def validate_raw_data(self, data: Dict[str, Any]) -> bool:
        """Validate raw market data format."""
        pass

    @abstractmethod
    def validate_candle_data(self, candle: CandleData) -> bool:
        """Validate candle data integrity."""
        pass


class BinanceDataValidator(IMarketDataValidator):
    """Binance-specific data validator implementation."""

    def __init__(self) -> None:
        """Initialize Binance data validator."""
        self._logger = get_module_logger("binance_data_validator")

    def validate_raw_data(self, data: Dict[str, Any]) -> bool:
        """Validate raw Binance market data format.

        Args:
            data: Raw data from Binance WebSocket

        Returns:
            True if data is valid, False otherwise
        """
        try:
            if data.get("type") == "kline":
                return self._validate_kline_data(data.get("data", {}))
            elif data.get("type") == "ticker":
                return self._validate_ticker_data(data.get("data", {}))
            return False
        except Exception as e:
            self._logger.error(f"Error validating raw data: {e}")
            return False

    def validate_candle_data(self, candle: CandleData) -> bool:
        """Validate candle data integrity.

        Args:
            candle: CandleData instance to validate

        Returns:
            True if candle is valid, False otherwise
        """
        try:
            # Basic validation is done in __post_init__
            # Additional business logic validation here
            current_time = int(time.time() * 1000)
            if candle.open_time > current_time:
                return False
            if candle.close_time <= candle.open_time:
                return False
            return True
        except Exception as e:
            self._logger.error(f"Error validating candle data: {e}")
            return False

    def _validate_kline_data(self, data: Dict[str, Any]) -> bool:
        """Validate kline data structure."""
        if "k" not in data:
            return False
        kline = data["k"]
        required_fields = ["t", "T", "o", "c", "h", "l", "v", "n", "x"]
        return all(field in kline for field in required_fields)

    def _validate_ticker_data(self, data: Dict[str, Any]) -> bool:
        """Validate ticker data structure."""
        required_fields = ["s", "c", "h", "l", "v"]
        return all(field in data for field in required_fields)


class ICandleAggregator(ABC):
    """Interface for candle aggregation strategies."""

    @abstractmethod
    def update_candle(self, market_data: MarketData) -> Optional[CandleData]:
        """Update candle with new market data."""
        pass

    @abstractmethod
    def get_current_candle(self, timeframe: Timeframe) -> Optional[CandleData]:
        """Get current incomplete candle for timeframe."""
        pass

    @abstractmethod
    def get_completed_candles(self) -> List[CandleData]:
        """Get all completed candles since last call."""
        pass


class TimeframeCandleAggregator(ICandleAggregator):
    """Multi-timeframe candle aggregator implementation."""

    def __init__(
        self, symbol: str, supported_timeframes: List[Timeframe]
    ) -> None:
        """Initialize timeframe candle aggregator.

        Args:
            symbol: Trading symbol to aggregate data for
            supported_timeframes: List of timeframes to support
        """
        self._symbol = symbol
        self._supported_timeframes = supported_timeframes
        self._logger = get_module_logger("candle_aggregator")

        # Current incomplete candles by timeframe
        self._current_candles: Dict[Timeframe, CandleData] = {}

        # Completed candles ready for emission
        self._completed_candles: List[CandleData] = []

    def update_candle(self, market_data: MarketData) -> Optional[CandleData]:
        """Update candles with new market data.

        Args:
            market_data: New market data to process

        Returns:
            Optional CandleData that was just completed, if any
        """
        completed_candle = None

        try:
            for timeframe in self._supported_timeframes:
                candle_timestamp = self._get_candle_timestamp(
                    market_data.timestamp, timeframe
                )

                current_candle = self._current_candles.get(timeframe)

                # Check if we need to start a new candle
                if (
                    current_candle is None
                    or current_candle.open_time != candle_timestamp
                ):

                    # Complete previous candle if exists
                    if current_candle is not None:
                        current_candle.is_closed = True
                        self._completed_candles.append(current_candle)
                        completed_candle = current_candle

                    # Start new candle
                    self._current_candles[timeframe] = self._create_new_candle(
                        market_data, timeframe, candle_timestamp
                    )
                else:
                    # Update existing candle
                    self._update_existing_candle(current_candle, market_data)

        except Exception as e:
            self._logger.error(f"Error updating candle: {e}")
            raise CandleAggregationError(f"Failed to update candle: {e}")

        return completed_candle

    def get_current_candle(self, timeframe: Timeframe) -> Optional[CandleData]:
        """Get current incomplete candle for timeframe.

        Args:
            timeframe: Timeframe to get candle for

        Returns:
            Current candle if exists, None otherwise
        """
        return self._current_candles.get(timeframe)

    def get_completed_candles(self) -> List[CandleData]:
        """Get all completed candles since last call.

        Returns:
            List of completed candles, cleared after call
        """
        completed = self._completed_candles.copy()
        self._completed_candles.clear()
        return completed

    def _get_candle_timestamp(
        self, timestamp: int, timeframe: Timeframe
    ) -> int:
        """Calculate candle open timestamp for given timeframe.

        Args:
            timestamp: Raw timestamp in milliseconds
            timeframe: Target timeframe

        Returns:
            Candle open timestamp aligned to timeframe
        """
        interval_ms = timeframe.seconds * 1000
        return (timestamp // interval_ms) * interval_ms

    def _create_new_candle(
        self,
        market_data: MarketData,
        timeframe: Timeframe,
        candle_timestamp: int,
    ) -> CandleData:
        """Create new candle from market data.

        Args:
            market_data: Initial market data
            timeframe: Candle timeframe
            candle_timestamp: Candle open timestamp

        Returns:
            New CandleData instance
        """
        close_timestamp = candle_timestamp + (timeframe.seconds * 1000) - 1

        return CandleData(
            symbol=self._symbol,
            timeframe=timeframe,
            open_time=candle_timestamp,
            close_time=close_timestamp,
            open_price=market_data.price,
            high_price=market_data.price,
            low_price=market_data.price,
            close_price=market_data.price,
            volume=market_data.volume,
            trade_count=1,
            is_closed=False,
            last_update=market_data.timestamp,
        )

    def _update_existing_candle(
        self, candle: CandleData, market_data: MarketData
    ) -> None:
        """Update existing candle with new market data.

        Args:
            candle: Existing candle to update
            market_data: New market data
        """
        candle.high_price = max(candle.high_price, market_data.price)
        candle.low_price = min(candle.low_price, market_data.price)
        candle.close_price = market_data.price
        candle.volume += market_data.volume
        candle.trade_count += 1
        candle.last_update = market_data.timestamp


class IMarketDataProcessor(ABC):
    """Interface for market data processor implementations."""

    @abstractmethod
    async def start(self) -> None:
        """Start data processing."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop data processing."""
        pass

    @abstractmethod
    def process_market_data(self, event_data: Dict[str, Any]) -> None:
        """Process incoming market data event."""
        pass


class MarketDataProcessor(IMarketDataProcessor):
    """Comprehensive market data processor implementation.

    This class handles real-time market data processing, candle aggregation,
    and event publishing following SOLID principles and dependency injection.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        event_hub: EventHub,
        data_validator: Optional[IMarketDataValidator] = None,
        supported_timeframes: Optional[List[Timeframe]] = None,
    ) -> None:
        """Initialize market data processor.

        Args:
            config_manager: Configuration manager instance
            event_hub: Event hub for subscribing/publishing
            data_validator: Data validation strategy
            supported_timeframes: List of timeframes to process
        """
        self._config_manager = config_manager
        self._event_hub = event_hub
        self._logger = get_module_logger("market_data_processor")

        # Initialize validator
        self._data_validator = data_validator or BinanceDataValidator()

        # Initialize supported timeframes
        self._supported_timeframes = supported_timeframes or [
            Timeframe.ONE_MINUTE,
            Timeframe.FIVE_MINUTES,
            Timeframe.FIFTEEN_MINUTES,
            Timeframe.ONE_HOUR,
        ]

        # Symbol-specific aggregators
        self._aggregators: Dict[str, ICandleAggregator] = {}

        # Processing statistics
        self._processed_count = 0
        self._error_count = 0
        self._last_processed_time = 0

        # Configuration
        self._max_data_age_seconds = self._load_config_value(
            "max_data_age_seconds", 300
        )
        self._enable_volume_updates = self._load_config_value(
            "enable_volume_updates", True
        )
        self._enable_price_updates = self._load_config_value(
            "enable_price_updates", True
        )

    async def start(self) -> None:
        """Start market data processing."""
        self._logger.info("Starting market data processor")

        try:
            # Subscribe to market data events
            self._event_hub.subscribe(
                EventType.MARKET_DATA_RECEIVED, self.process_market_data
            )

            self._logger.info("Market data processor started successfully")

        except Exception as e:
            self._logger.error(f"Failed to start market data processor: {e}")
            raise DataProcessingError(f"Failed to start processor: {e}")

    async def stop(self) -> None:
        """Stop market data processing."""
        self._logger.info("Stopping market data processor")

        try:
            # Unsubscribe from events
            self._event_hub.unsubscribe(
                EventType.MARKET_DATA_RECEIVED, self.process_market_data
            )

            # Log final statistics
            self._log_processing_statistics()

            self._logger.info("Market data processor stopped successfully")

        except Exception as e:
            self._logger.error(f"Error stopping market data processor: {e}")

    def process_market_data(self, event_data: Dict[str, Any]) -> None:
        """Process incoming market data event.

        Args:
            event_data: Market data event from WebSocketManager
        """
        try:
            # Validate raw data
            if not self._data_validator.validate_raw_data(event_data):
                self._logger.warning("Invalid market data received")
                self._error_count += 1
                return

            # Parse market data
            market_data = self._parse_market_data(event_data)
            if not market_data:
                return

            # Check data freshness
            if not self._is_data_fresh(market_data):
                self._logger.warning(
                    f"Stale data received: {market_data.symbol}"
                )
                return

            # Get or create aggregator for symbol
            aggregator = self._get_aggregator(market_data.symbol)

            # Update candles
            completed_candle = aggregator.update_candle(market_data)

            # Publish events
            if completed_candle:
                self._publish_candle_data(completed_candle)

            if self._enable_price_updates:
                self._publish_price_update(market_data)

            if self._enable_volume_updates:
                self._publish_volume_update(market_data)

            # Update statistics
            self._processed_count += 1
            self._last_processed_time = int(time.time())

        except Exception as e:
            self._logger.error(f"Error processing market data: {e}")
            self._error_count += 1

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics.

        Returns:
            Dictionary containing processing statistics
        """
        return {
            "processed_count": self._processed_count,
            "error_count": self._error_count,
            "last_processed_time": self._last_processed_time,
            "active_symbols": list(self._aggregators.keys()),
            "supported_timeframes": [
                tf.value for tf in self._supported_timeframes
            ],
        }

    def _parse_market_data(
        self, event_data: Dict[str, Any]
    ) -> Optional[MarketData]:
        """Parse event data into MarketData structure.

        Args:
            event_data: Raw event data

        Returns:
            MarketData instance if parsing successful, None otherwise
        """
        try:
            if event_data.get("type") == "kline":
                return self._parse_kline_data(event_data)
            elif event_data.get("type") == "ticker":
                return self._parse_ticker_data(event_data)

            self._logger.warning(
                f"Unknown data type: {event_data.get('type')}"
            )
            return None

        except Exception as e:
            self._logger.error(f"Error parsing market data: {e}")
            return None

    def _parse_kline_data(self, event_data: Dict[str, Any]) -> MarketData:
        """Parse kline data into MarketData structure."""
        raw_data = event_data.get("data", {})
        kline = raw_data.get("k", {})

        return MarketData(
            symbol=event_data.get("symbol", "").upper(),
            timestamp=int(kline.get("t", 0)),  # Kline start time
            price=float(kline.get("c", 0)),  # Close price
            volume=float(kline.get("v", 0)),  # Volume
            source=event_data.get("source", "unknown"),
            data_type="kline",
            raw_data=raw_data,
            metadata={
                "open_price": float(kline.get("o", 0)),
                "high_price": float(kline.get("h", 0)),
                "low_price": float(kline.get("l", 0)),
                "trade_count": int(kline.get("n", 0)),
                "is_closed": bool(kline.get("x", False)),
                "interval": kline.get("i", "1m"),
            },
        )

    def _parse_ticker_data(self, event_data: Dict[str, Any]) -> MarketData:
        """Parse ticker data into MarketData structure."""
        raw_data = event_data.get("data", {})

        return MarketData(
            symbol=event_data.get("symbol", "").upper(),
            timestamp=event_data.get("timestamp", int(time.time() * 1000)),
            price=float(raw_data.get("c", 0)),  # Current close price
            volume=float(raw_data.get("v", 0)),  # 24h volume
            source=event_data.get("source", "unknown"),
            data_type="ticker",
            raw_data=raw_data,
            metadata={
                "high_24h": float(raw_data.get("h", 0)),
                "low_24h": float(raw_data.get("l", 0)),
                "price_change": float(raw_data.get("p", 0)),
                "price_change_percent": float(raw_data.get("P", 0)),
            },
        )

    def _is_data_fresh(self, market_data: MarketData) -> bool:
        """Check if market data is fresh enough to process.

        Args:
            market_data: Market data to check

        Returns:
            True if data is fresh, False otherwise
        """
        # Handle case where max_data_age_seconds is None
        if self._max_data_age_seconds is None:
            return True

        current_time = int(time.time() * 1000)
        age_seconds = (current_time - market_data.timestamp) / 1000
        return age_seconds <= self._max_data_age_seconds

    def _get_aggregator(self, symbol: str) -> ICandleAggregator:
        """Get or create candle aggregator for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Candle aggregator instance
        """
        if symbol not in self._aggregators:
            self._aggregators[symbol] = TimeframeCandleAggregator(
                symbol, self._supported_timeframes
            )
            self._logger.info(f"Created new aggregator for symbol: {symbol}")

        return self._aggregators[symbol]

    def _publish_candle_data(self, candle: CandleData) -> None:
        """Publish completed candle data event.

        Args:
            candle: Completed candle data
        """
        try:
            event_data = {
                "symbol": candle.symbol,
                "timeframe": candle.timeframe.value,
                "candle": candle,
                "timestamp": candle.last_update,
            }

            self._event_hub.publish(
                EventType.CANDLE_DATA_PROCESSED, event_data
            )

            self._logger.debug(
                f"Published candle data: {candle.symbol} "
                f"{candle.timeframe.value}"
            )

        except Exception as e:
            self._logger.error(f"Error publishing candle data: {e}")

    def _publish_price_update(self, market_data: MarketData) -> None:
        """Publish price update event.

        Args:
            market_data: Market data with price information
        """
        try:
            event_data = {
                "symbol": market_data.symbol,
                "price": market_data.price,
                "timestamp": market_data.timestamp,
                "source": market_data.source,
            }

            self._event_hub.publish(EventType.PRICE_UPDATE, event_data)

        except Exception as e:
            self._logger.error(f"Error publishing price update: {e}")

    def _publish_volume_update(self, market_data: MarketData) -> None:
        """Publish volume update event.

        Args:
            market_data: Market data with volume information
        """
        try:
            event_data = {
                "symbol": market_data.symbol,
                "volume": market_data.volume,
                "timestamp": market_data.timestamp,
                "source": market_data.source,
            }

            self._event_hub.publish(EventType.VOLUME_UPDATE, event_data)

        except Exception as e:
            self._logger.error(f"Error publishing volume update: {e}")

    def _load_config_value(self, key: str, default: Any) -> Any:
        """Load configuration value with fallback to default.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        try:
            return self._config_manager.get_config_value(key, default)
        except Exception as e:
            self._logger.warning(f"Error loading config value {key}: {e}")
            return default

    def _log_processing_statistics(self) -> None:
        """Log processing statistics."""
        stats = self.get_processing_statistics()
        self._logger.info(f"Processing statistics: {stats}")


def create_market_data_processor(
    config_manager: ConfigManager,
    event_hub: EventHub,
    supported_timeframes: Optional[List[str]] = None,
) -> MarketDataProcessor:
    """Factory function to create MarketDataProcessor instance.

    Args:
        config_manager: Configuration manager instance
        event_hub: Event hub instance
        supported_timeframes: List of timeframe strings to support

    Returns:
        MarketDataProcessor: Configured processor instance

    Raises:
        TimeframeError: If invalid timeframe strings provided
    """
    # Convert timeframe strings to Timeframe enums
    if supported_timeframes:
        timeframes = [Timeframe.from_string(tf) for tf in supported_timeframes]
    else:
        timeframes = None

    # Create validator
    validator = BinanceDataValidator()

    return MarketDataProcessor(
        config_manager=config_manager,
        event_hub=event_hub,
        data_validator=validator,
        supported_timeframes=timeframes,
    )
