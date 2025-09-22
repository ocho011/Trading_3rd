"""
Example usage of Market Data Processor with WebSocket integration.

This script demonstrates how to use the Market Data Processor with the
existing WebSocket Manager and Event Hub to process real-time market data.
"""

import asyncio
import logging
import signal
import sys
from typing import Any, Dict

from trading_bot.core.config_manager import ConfigManager, EnvConfigLoader
from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.core.logger import create_trading_logger
from trading_bot.market_data.data_processor import (
    CandleData,
    MarketDataProcessor,
    Timeframe,
    create_market_data_processor,
)
from trading_bot.market_data.websocket_manager import (
    BinanceWebSocketManager,
    ReconnectionConfig,
)


class MarketDataExample:
    """Example application for market data processing."""

    def __init__(self) -> None:
        """Initialize the example application."""
        # Set up logging
        self.logger = create_trading_logger("market_data_example", "INFO")

        # Initialize core components
        self.event_hub = EventHub()
        self.config_manager = self._create_config_manager()

        # Initialize WebSocket manager
        reconnection_config = ReconnectionConfig(
            enabled=True,
            max_retries=5,
            initial_delay=1.0,
            max_delay=30.0
        )

        self.websocket_manager = BinanceWebSocketManager(
            config_manager=self.config_manager,
            event_hub=self.event_hub,
            symbol="btcusdt",
            reconnection_config=reconnection_config
        )

        # Initialize data processor
        self.data_processor = create_market_data_processor(
            config_manager=self.config_manager,
            event_hub=self.event_hub,
            supported_timeframes=["1m", "5m", "15m", "1h"]
        )

        # Set up event handlers
        self._setup_event_handlers()

        # Shutdown flag
        self.shutdown_requested = False

    def _create_config_manager(self) -> ConfigManager:
        """Create and configure the config manager."""
        try:
            config_loader = EnvConfigLoader()
            config_manager = ConfigManager(config_loader)
            config_manager.load_configuration()
            return config_manager
        except Exception as e:
            self.logger.error(f"Failed to create config manager: {e}")
            # Create minimal config for demo
            return self._create_demo_config()

    def _create_demo_config(self) -> ConfigManager:
        """Create demo config manager for testing."""
        from trading_bot.core.config_manager import IConfigLoader

        class DemoConfigLoader(IConfigLoader):
            def load_config(self) -> Dict[str, Any]:
                return {
                    "trading_mode": "paper",
                    "log_level": "INFO",
                    "max_data_age_seconds": 300,
                    "enable_volume_updates": True,
                    "enable_price_updates": True,
                }

        config_manager = ConfigManager(DemoConfigLoader())
        config_manager.load_configuration()
        return config_manager

    def _setup_event_handlers(self) -> None:
        """Set up event handlers for market data events."""
        # Subscribe to candle data processed events
        self.event_hub.subscribe(
            EventType.CANDLE_DATA_PROCESSED,
            self._handle_candle_data
        )

        # Subscribe to price update events
        self.event_hub.subscribe(
            EventType.PRICE_UPDATE,
            self._handle_price_update
        )

        # Subscribe to volume update events
        self.event_hub.subscribe(
            EventType.VOLUME_UPDATE,
            self._handle_volume_update
        )

        # Subscribe to system events
        self.event_hub.subscribe(
            EventType.CONNECTION_LOST,
            self._handle_connection_lost
        )

        self.event_hub.subscribe(
            EventType.CONNECTION_RESTORED,
            self._handle_connection_restored
        )

    def _handle_candle_data(self, event_data: Dict[str, Any]) -> None:
        """Handle completed candle data.

        Args:
            event_data: Candle data event
        """
        try:
            symbol = event_data.get("symbol")
            timeframe = event_data.get("timeframe")
            candle = event_data.get("candle")

            if isinstance(candle, CandleData):
                self.logger.info(
                    f"📊 Candle completed [{symbol} {timeframe}]: "
                    f"O:{candle.open_price:.2f} H:{candle.high_price:.2f} "
                    f"L:{candle.low_price:.2f} C:{candle.close_price:.2f} "
                    f"V:{candle.volume:.4f}"
                )

                # Example: Perform technical analysis on completed candle
                self._analyze_candle(candle)

        except Exception as e:
            self.logger.error(f"Error handling candle data: {e}")

    def _handle_price_update(self, event_data: Dict[str, Any]) -> None:
        """Handle price update events.

        Args:
            event_data: Price update event
        """
        try:
            symbol = event_data.get("symbol")
            price = event_data.get("price")

            # Log every 10th price update to avoid spam
            if hasattr(self, '_price_update_count'):
                self._price_update_count += 1
            else:
                self._price_update_count = 1

            if self._price_update_count % 10 == 0:
                self.logger.info(f"💰 Price update [{symbol}]: ${price:.2f}")

        except Exception as e:
            self.logger.error(f"Error handling price update: {e}")

    def _handle_volume_update(self, event_data: Dict[str, Any]) -> None:
        """Handle volume update events.

        Args:
            event_data: Volume update event
        """
        try:
            symbol = event_data.get("symbol")
            volume = event_data.get("volume")

            # Log significant volume updates only
            if volume > 1.0:  # Example threshold
                self.logger.info(f"📈 Volume update [{symbol}]: {volume:.4f}")

        except Exception as e:
            self.logger.error(f"Error handling volume update: {e}")

    def _handle_connection_lost(self, event_data: Dict[str, Any]) -> None:
        """Handle connection lost events.

        Args:
            event_data: Connection lost event
        """
        self.logger.warning("🔴 WebSocket connection lost")

    def _handle_connection_restored(self, event_data: Dict[str, Any]) -> None:
        """Handle connection restored events.

        Args:
            event_data: Connection restored event
        """
        self.logger.info("🟢 WebSocket connection restored")

    def _analyze_candle(self, candle: CandleData) -> None:
        """Perform basic technical analysis on candle.

        Args:
            candle: Completed candle data
        """
        try:
            # Calculate simple metrics
            price_change = candle.close_price - candle.open_price
            price_change_percent = (price_change / candle.open_price) * 100
            price_range = candle.high_price - candle.low_price

            # Example simple trend detection
            if price_change_percent > 0.5:
                trend = "🟢 Strong Up"
            elif price_change_percent > 0:
                trend = "🔵 Up"
            elif price_change_percent < -0.5:
                trend = "🔴 Strong Down"
            elif price_change_percent < 0:
                trend = "🟠 Down"
            else:
                trend = "⚫ Flat"

            self.logger.info(
                f"📈 Analysis [{candle.symbol} {candle.timeframe.value}]: "
                f"Change: {price_change_percent:.2f}% "
                f"Range: {price_range:.2f} "
                f"Trend: {trend}"
            )

        except Exception as e:
            self.logger.error(f"Error analyzing candle: {e}")

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def run(self) -> None:
        """Run the market data processing example."""
        self.logger.info("🚀 Starting Market Data Processor Example")

        try:
            # Set up signal handlers
            self._setup_signal_handlers()

            # Start data processor
            await self.data_processor.start()
            self.logger.info("✅ Data processor started")

            # Start WebSocket manager
            websocket_task = asyncio.create_task(self.websocket_manager.start())
            self.logger.info("✅ WebSocket manager started")

            # Main event loop
            self.logger.info("📡 Processing market data... (Press Ctrl+C to stop)")

            # Monitor processing statistics
            stats_task = asyncio.create_task(self._monitor_statistics())

            # Wait for shutdown signal
            while not self.shutdown_requested:
                await asyncio.sleep(1)

            self.logger.info("🛑 Shutdown requested, stopping services...")

            # Cancel tasks
            websocket_task.cancel()
            stats_task.cancel()

            # Stop services
            await self.websocket_manager.stop()
            await self.data_processor.stop()

            self.logger.info("✅ All services stopped successfully")

        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            raise

    async def _monitor_statistics(self) -> None:
        """Monitor and log processing statistics."""
        while not self.shutdown_requested:
            try:
                await asyncio.sleep(30)  # Log stats every 30 seconds

                stats = self.data_processor.get_processing_statistics()
                self.logger.info(
                    f"📊 Stats: Processed: {stats['processed_count']}, "
                    f"Errors: {stats['error_count']}, "
                    f"Active symbols: {len(stats['active_symbols'])}"
                )

            except Exception as e:
                self.logger.error(f"Error monitoring statistics: {e}")


async def main() -> None:
    """Main entry point."""
    try:
        example = MarketDataExample()
        await example.run()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())