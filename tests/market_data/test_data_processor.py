"""
Unit tests for Market Data Processor module.

Tests all components of the market data processing system including
data validation, candle aggregation, and event publishing.
"""

import time
import unittest
from unittest.mock import Mock, patch

import pytest

from trading_bot.core.config_manager import ConfigManager
from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.market_data.data_processor import (
    BinanceDataValidator,
    CandleData,
    InvalidDataError,
    MarketData,
    MarketDataProcessor,
    Timeframe,
    TimeframeCandleAggregator,
    TimeframeError,
    create_market_data_processor,
)


class TestTimeframe(unittest.TestCase):
    """Test cases for Timeframe enum."""

    def test_timeframe_seconds(self) -> None:
        """Test timeframe seconds property."""
        self.assertEqual(Timeframe.ONE_MINUTE.seconds, 60)
        self.assertEqual(Timeframe.FIVE_MINUTES.seconds, 300)
        self.assertEqual(Timeframe.FIFTEEN_MINUTES.seconds, 900)
        self.assertEqual(Timeframe.ONE_HOUR.seconds, 3600)
        self.assertEqual(Timeframe.FOUR_HOURS.seconds, 14400)
        self.assertEqual(Timeframe.ONE_DAY.seconds, 86400)

    def test_from_string_valid(self) -> None:
        """Test creating timeframe from valid string."""
        self.assertEqual(Timeframe.from_string("1m"), Timeframe.ONE_MINUTE)
        self.assertEqual(Timeframe.from_string("5m"), Timeframe.FIVE_MINUTES)
        self.assertEqual(Timeframe.from_string("15m"), Timeframe.FIFTEEN_MINUTES)
        self.assertEqual(Timeframe.from_string("1h"), Timeframe.ONE_HOUR)

    def test_from_string_invalid(self) -> None:
        """Test creating timeframe from invalid string."""
        with self.assertRaises(TimeframeError):
            Timeframe.from_string("invalid")
        with self.assertRaises(TimeframeError):
            Timeframe.from_string("2m")


class TestMarketData(unittest.TestCase):
    """Test cases for MarketData dataclass."""

    def test_valid_market_data(self) -> None:
        """Test creating valid market data."""
        data = MarketData(
            symbol="BTCUSDT",
            timestamp=1640995200000,
            price=50000.0,
            volume=1.5,
            source="binance",
            data_type="kline",
        )
        self.assertEqual(data.symbol, "BTCUSDT")
        self.assertEqual(data.price, 50000.0)
        self.assertEqual(data.volume, 1.5)

    def test_invalid_price(self) -> None:
        """Test validation of invalid price."""
        with self.assertRaises(InvalidDataError):
            MarketData(
                symbol="BTCUSDT",
                timestamp=1640995200000,
                price=0.0,  # Invalid price
                volume=1.5,
                source="binance",
                data_type="kline",
            )

    def test_invalid_volume(self) -> None:
        """Test validation of invalid volume."""
        with self.assertRaises(InvalidDataError):
            MarketData(
                symbol="BTCUSDT",
                timestamp=1640995200000,
                price=50000.0,
                volume=-1.0,  # Invalid volume
                source="binance",
                data_type="kline",
            )

    def test_invalid_symbol(self) -> None:
        """Test validation of invalid symbol."""
        with self.assertRaises(InvalidDataError):
            MarketData(
                symbol="",  # Empty symbol
                timestamp=1640995200000,
                price=50000.0,
                volume=1.5,
                source="binance",
                data_type="kline",
            )


class TestCandleData(unittest.TestCase):
    """Test cases for CandleData dataclass."""

    def test_valid_candle_data(self) -> None:
        """Test creating valid candle data."""
        candle = CandleData(
            symbol="BTCUSDT",
            timeframe=Timeframe.ONE_MINUTE,
            open_time=1640995200000,
            close_time=1640995259999,
            open_price=50000.0,
            high_price=50100.0,
            low_price=49900.0,
            close_price=50050.0,
            volume=1.5,
        )
        self.assertEqual(candle.symbol, "BTCUSDT")
        self.assertEqual(candle.open_price, 50000.0)
        self.assertEqual(candle.high_price, 50100.0)

    def test_invalid_prices(self) -> None:
        """Test validation of invalid prices."""
        with self.assertRaises(InvalidDataError):
            CandleData(
                symbol="BTCUSDT",
                timeframe=Timeframe.ONE_MINUTE,
                open_time=1640995200000,
                close_time=1640995259999,
                open_price=0.0,  # Invalid price
                high_price=50100.0,
                low_price=49900.0,
                close_price=50050.0,
                volume=1.5,
            )

    def test_invalid_high_price(self) -> None:
        """Test validation of invalid high price."""
        with self.assertRaises(InvalidDataError):
            CandleData(
                symbol="BTCUSDT",
                timeframe=Timeframe.ONE_MINUTE,
                open_time=1640995200000,
                close_time=1640995259999,
                open_price=50000.0,
                high_price=49000.0,  # High < Open
                low_price=49900.0,
                close_price=50050.0,
                volume=1.5,
            )

    def test_invalid_low_price(self) -> None:
        """Test validation of invalid low price."""
        with self.assertRaises(InvalidDataError):
            CandleData(
                symbol="BTCUSDT",
                timeframe=Timeframe.ONE_MINUTE,
                open_time=1640995200000,
                close_time=1640995259999,
                open_price=50000.0,
                high_price=50100.0,
                low_price=51000.0,  # Low > Open
                close_price=50050.0,
                volume=1.5,
            )


class TestBinanceDataValidator(unittest.TestCase):
    """Test cases for BinanceDataValidator."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.validator = BinanceDataValidator()

    def test_validate_kline_data_valid(self) -> None:
        """Test validation of valid kline data."""
        data = {
            "type": "kline",
            "data": {
                "k": {
                    "t": 1640995200000,
                    "T": 1640995259999,
                    "o": "50000.0",
                    "c": "50050.0",
                    "h": "50100.0",
                    "l": "49900.0",
                    "v": "1.5",
                    "n": 100,
                    "x": True,
                }
            },
        }
        self.assertTrue(self.validator.validate_raw_data(data))

    def test_validate_kline_data_invalid(self) -> None:
        """Test validation of invalid kline data."""
        data = {
            "type": "kline",
            "data": {
                "k": {
                    "t": 1640995200000,
                    # Missing required fields
                }
            },
        }
        self.assertFalse(self.validator.validate_raw_data(data))

    def test_validate_ticker_data_valid(self) -> None:
        """Test validation of valid ticker data."""
        data = {
            "type": "ticker",
            "data": {
                "s": "BTCUSDT",
                "c": "50000.0",
                "h": "50100.0",
                "l": "49900.0",
                "v": "1.5",
            },
        }
        self.assertTrue(self.validator.validate_raw_data(data))

    def test_validate_ticker_data_invalid(self) -> None:
        """Test validation of invalid ticker data."""
        data = {
            "type": "ticker",
            "data": {
                "s": "BTCUSDT",
                # Missing required fields
            },
        }
        self.assertFalse(self.validator.validate_raw_data(data))

    def test_validate_candle_data_valid(self) -> None:
        """Test validation of valid candle data."""
        candle = CandleData(
            symbol="BTCUSDT",
            timeframe=Timeframe.ONE_MINUTE,
            open_time=1640995200000,
            close_time=1640995259999,
            open_price=50000.0,
            high_price=50100.0,
            low_price=49900.0,
            close_price=50050.0,
            volume=1.5,
        )
        self.assertTrue(self.validator.validate_candle_data(candle))

    def test_validate_candle_data_future_time(self) -> None:
        """Test validation of candle with future timestamp."""
        future_time = int(time.time() * 1000) + 86400000  # +1 day
        candle = CandleData(
            symbol="BTCUSDT",
            timeframe=Timeframe.ONE_MINUTE,
            open_time=future_time,
            close_time=future_time + 60000,
            open_price=50000.0,
            high_price=50100.0,
            low_price=49900.0,
            close_price=50050.0,
            volume=1.5,
        )
        self.assertFalse(self.validator.validate_candle_data(candle))


class TestTimeframeCandleAggregator(unittest.TestCase):
    """Test cases for TimeframeCandleAggregator."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.aggregator = TimeframeCandleAggregator(
            symbol="BTCUSDT",
            supported_timeframes=[Timeframe.ONE_MINUTE, Timeframe.FIVE_MINUTES],
        )

    def test_create_first_candle(self) -> None:
        """Test creating first candle from market data."""
        market_data = MarketData(
            symbol="BTCUSDT",
            timestamp=1640995200000,  # Aligned to minute
            price=50000.0,
            volume=1.5,
            source="binance",
            data_type="kline",
        )

        completed_candle = self.aggregator.update_candle(market_data)
        self.assertIsNone(completed_candle)  # First candle, no completion

        # Check current candle created
        current_1m = self.aggregator.get_current_candle(Timeframe.ONE_MINUTE)
        self.assertIsNotNone(current_1m)
        self.assertEqual(current_1m.open_price, 50000.0)
        self.assertEqual(current_1m.close_price, 50000.0)

    def test_update_existing_candle(self) -> None:
        """Test updating existing candle with new data."""
        # First trade
        market_data1 = MarketData(
            symbol="BTCUSDT",
            timestamp=1640995200000,
            price=50000.0,
            volume=1.0,
            source="binance",
            data_type="kline",
        )
        self.aggregator.update_candle(market_data1)

        # Second trade in same minute
        market_data2 = MarketData(
            symbol="BTCUSDT",
            timestamp=1640995230000,  # Same minute
            price=50100.0,
            volume=0.5,
            source="binance",
            data_type="kline",
        )
        self.aggregator.update_candle(market_data2)

        current_candle = self.aggregator.get_current_candle(Timeframe.ONE_MINUTE)
        self.assertEqual(current_candle.open_price, 50000.0)
        self.assertEqual(current_candle.close_price, 50100.0)
        self.assertEqual(current_candle.high_price, 50100.0)
        self.assertEqual(current_candle.volume, 1.5)

    def test_complete_candle(self) -> None:
        """Test completing candle when new timeframe starts."""
        # First minute
        market_data1 = MarketData(
            symbol="BTCUSDT",
            timestamp=1640995200000,
            price=50000.0,
            volume=1.0,
            source="binance",
            data_type="kline",
        )
        self.aggregator.update_candle(market_data1)

        # Next minute (should complete previous)
        market_data2 = MarketData(
            symbol="BTCUSDT",
            timestamp=1640995260000,  # Next minute
            price=50100.0,
            volume=0.5,
            source="binance",
            data_type="kline",
        )
        completed_candle = self.aggregator.update_candle(market_data2)

        self.assertIsNotNone(completed_candle)
        self.assertTrue(completed_candle.is_closed)
        self.assertEqual(completed_candle.open_price, 50000.0)

    def test_get_completed_candles(self) -> None:
        """Test getting completed candles."""
        # Create multiple candles
        for i in range(3):
            market_data = MarketData(
                symbol="BTCUSDT",
                timestamp=1640995200000 + (i * 60000),  # Different minutes
                price=50000.0 + i,
                volume=1.0,
                source="binance",
                data_type="kline",
            )
            self.aggregator.update_candle(market_data)

        completed_candles = self.aggregator.get_completed_candles()
        self.assertEqual(len(completed_candles), 2)  # 2 completed, 1 current

        # Second call should return empty list
        completed_candles2 = self.aggregator.get_completed_candles()
        self.assertEqual(len(completed_candles2), 0)


class TestMarketDataProcessor(unittest.TestCase):
    """Test cases for MarketDataProcessor."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config_manager = Mock(spec=ConfigManager)
        self.config_manager.get_config_value.side_effect = lambda key, default: default

        self.event_hub = Mock(spec=EventHub)

        self.processor = MarketDataProcessor(
            config_manager=self.config_manager, event_hub=self.event_hub
        )

    @pytest.mark.asyncio
    async def test_start_processor(self) -> None:
        """Test starting the processor."""
        await self.processor.start()

        # Verify subscription to market data events
        self.event_hub.subscribe.assert_called_once_with(
            EventType.MARKET_DATA_RECEIVED, self.processor.process_market_data
        )

    @pytest.mark.asyncio
    async def test_stop_processor(self) -> None:
        """Test stopping the processor."""
        await self.processor.start()
        await self.processor.stop()

        # Verify unsubscription from events
        self.event_hub.unsubscribe.assert_called_once_with(
            EventType.MARKET_DATA_RECEIVED, self.processor.process_market_data
        )

    def test_process_valid_kline_data(self) -> None:
        """Test processing valid kline data."""
        current_time = int(time.time() * 1000)

        event_data = {
            "symbol": "BTCUSDT",
            "type": "kline",
            "source": "binance",
            "timestamp": current_time,
            "data": {
                "k": {
                    "t": current_time,
                    "T": current_time + 59999,
                    "o": "50000.0",
                    "c": "50050.0",
                    "h": "50100.0",
                    "l": "49900.0",
                    "v": "1.5",
                    "n": 100,
                    "x": True,
                    "i": "1m",
                }
            },
        }

        self.processor.process_market_data(event_data)

        # Verify events were published
        self.assertGreater(self.event_hub.publish.call_count, 0)

    def test_process_invalid_data(self) -> None:
        """Test processing invalid data."""
        event_data = {"symbol": "BTCUSDT", "type": "invalid", "source": "binance"}

        # Should not raise exception
        self.processor.process_market_data(event_data)

    def test_get_processing_statistics(self) -> None:
        """Test getting processing statistics."""
        stats = self.processor.get_processing_statistics()

        self.assertIsInstance(stats, dict)
        self.assertIn("processed_count", stats)
        self.assertIn("error_count", stats)
        self.assertIn("active_symbols", stats)

    @patch("time.time")
    def test_stale_data_rejection(self, mock_time) -> None:
        """Test rejection of stale data."""
        mock_time.return_value = 1640995500  # Current time

        # Configure max age to 60 seconds
        self.config_manager.get_config_value.side_effect = lambda key, default: (
            60 if key == "max_data_age_seconds" else default
        )

        # Create processor with config
        processor = MarketDataProcessor(
            config_manager=self.config_manager, event_hub=self.event_hub
        )

        # Old data (more than 60 seconds old)
        event_data = {
            "symbol": "BTCUSDT",
            "type": "kline",
            "source": "binance",
            "timestamp": 1640995200000,  # 5 minutes old in seconds
            "data": {
                "k": {
                    "t": 1640995200000,
                    "T": 1640995259999,
                    "o": "50000.0",
                    "c": "50050.0",
                    "h": "50100.0",
                    "l": "49900.0",
                    "v": "1.5",
                    "n": 100,
                    "x": True,
                    "i": "1m",
                }
            },
        }

        processor.process_market_data(event_data)

        # Should not publish any events for stale data
        self.event_hub.publish.assert_not_called()


class TestFactoryFunction(unittest.TestCase):
    """Test cases for factory function."""

    def test_create_market_data_processor(self) -> None:
        """Test creating processor with factory function."""
        config_manager = Mock(spec=ConfigManager)
        event_hub = Mock(spec=EventHub)

        processor = create_market_data_processor(
            config_manager=config_manager,
            event_hub=event_hub,
            supported_timeframes=["1m", "5m"],
        )

        self.assertIsInstance(processor, MarketDataProcessor)

    def test_create_processor_invalid_timeframe(self) -> None:
        """Test creating processor with invalid timeframe."""
        config_manager = Mock(spec=ConfigManager)
        event_hub = Mock(spec=EventHub)

        with self.assertRaises(TimeframeError):
            create_market_data_processor(
                config_manager=config_manager,
                event_hub=event_hub,
                supported_timeframes=["invalid"],
            )


if __name__ == "__main__":
    unittest.main()
