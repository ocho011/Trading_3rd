"""
Example usage of the WebSocket manager for real-time market data streaming.

This example demonstrates how to use the BinanceWebSocketManager to connect
to Binance WebSocket streams and receive real-time market data through the
event-driven architecture.
"""

import asyncio
import signal
import sys
from typing import Any, Dict

from trading_bot.core.config_manager import ConfigManager, EnvConfigLoader
from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.core.logger import create_trading_logger
from trading_bot.market_data.websocket_manager import (
    create_websocket_manager,
    BinanceWebSocketManager,
)


class MarketDataHandler:
    """Example handler for processing market data events."""

    def __init__(self) -> None:
        """Initialize market data handler."""
        self.logger = create_trading_logger("market_data_handler")
        self.kline_count = 0
        self.ticker_count = 0

    async def handle_market_data(self, data: Dict[str, Any]) -> None:
        """Handle incoming market data.

        Args:
            data: Market data event payload
        """
        try:
            symbol = data.get("symbol", "unknown")
            data_type = data.get("type", "unknown")
            timestamp = data.get("timestamp", 0)

            if data_type == "kline":
                await self._handle_kline_data(data, symbol, timestamp)
            elif data_type == "ticker":
                await self._handle_ticker_data(data, symbol, timestamp)
            else:
                self.logger.warning(f"Unknown data type: {data_type}")

        except Exception as e:
            self.logger.error(f"Error handling market data: {e}")

    async def _handle_kline_data(
        self, data: Dict[str, Any], symbol: str, timestamp: int
    ) -> None:
        """Handle kline (candlestick) data.

        Args:
            data: Market data event payload
            symbol: Trading symbol
            timestamp: Event timestamp
        """
        self.kline_count += 1
        kline_data = data.get("data", {}).get("k", {})

        if kline_data:
            open_price = kline_data.get("o", "0")
            close_price = kline_data.get("c", "0")
            high_price = kline_data.get("h", "0")
            low_price = kline_data.get("l", "0")
            volume = kline_data.get("v", "0")

            self.logger.info(
                f"Kline #{self.kline_count} - {symbol}: "
                f"O:{open_price} H:{high_price} L:{low_price} "
                f"C:{close_price} V:{volume}"
            )

    async def _handle_ticker_data(
        self, data: Dict[str, Any], symbol: str, timestamp: int
    ) -> None:
        """Handle ticker data.

        Args:
            data: Market data event payload
            symbol: Trading symbol
            timestamp: Event timestamp
        """
        self.ticker_count += 1
        ticker_data = data.get("data", {})

        if ticker_data:
            current_price = ticker_data.get("c", "0")
            high_price = ticker_data.get("h", "0")
            low_price = ticker_data.get("l", "0")
            volume = ticker_data.get("v", "0")

            self.logger.info(
                f"Ticker #{self.ticker_count} - {symbol}: "
                f"Price:{current_price} H:{high_price} L:{low_price} "
                f"V:{volume}"
            )

    def get_statistics(self) -> Dict[str, int]:
        """Get processing statistics.

        Returns:
            Dictionary with processing counts
        """
        return {
            "kline_messages": self.kline_count,
            "ticker_messages": self.ticker_count,
            "total_messages": self.kline_count + self.ticker_count,
        }


class WebSocketExample:
    """Example application demonstrating WebSocket manager usage."""

    def __init__(self) -> None:
        """Initialize the WebSocket example application."""
        self.logger = create_trading_logger("websocket_example")
        self.running = True

        # Initialize core components
        self.config_manager = ConfigManager(EnvConfigLoader())
        self.event_hub = EventHub()
        self.market_data_handler = MarketDataHandler()

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False

    async def setup(self) -> None:
        """Set up the application components."""
        try:
            # Load configuration
            self.config_manager.load_configuration()
            self.logger.info("Configuration loaded successfully")

            # Subscribe to market data events
            self.event_hub.subscribe(
                EventType.MARKET_DATA_RECEIVED,
                self.market_data_handler.handle_market_data
            )
            self.logger.info("Subscribed to market data events")

        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            raise

    async def run_basic_example(self) -> None:
        """Run basic WebSocket manager example."""
        self.logger.info("Starting basic WebSocket example")

        try:
            # Create and start WebSocket manager manually
            ws_manager = BinanceWebSocketManager(
                self.config_manager,
                self.event_hub,
                symbol="btcusdt"
            )

            await ws_manager.start()
            self.logger.info("WebSocket manager started")

            # Run for specified duration or until interrupted
            runtime = 30  # seconds
            for i in range(runtime):
                if not self.running:
                    break

                await asyncio.sleep(1)

                if i % 10 == 0:  # Log status every 10 seconds
                    stats = self.market_data_handler.get_statistics()
                    self.logger.info(f"Statistics: {stats}")

            await ws_manager.stop()
            self.logger.info("WebSocket manager stopped")

        except Exception as e:
            self.logger.error(f"Error in basic example: {e}")
            raise

    async def run_context_manager_example(self) -> None:
        """Run WebSocket manager example using async context manager."""
        self.logger.info("Starting context manager WebSocket example")

        try:
            # Use context manager for automatic lifecycle management
            async with create_websocket_manager(
                self.config_manager,
                self.event_hub,
                symbol="ethusdt"
            ) as ws_manager:
                self.logger.info("WebSocket manager started (context manager)")

                # Monitor connection and collect data
                runtime = 20  # seconds
                for i in range(runtime):
                    if not self.running:
                        break

                    # Log connection status
                    if i % 5 == 0:
                        state = ws_manager.get_connection_state()
                        connected = ws_manager.is_connected()
                        self.logger.info(
                            f"Connection state: {state.value}, "
                            f"Connected: {connected}"
                        )

                    await asyncio.sleep(1)

            # WebSocket manager is automatically stopped here
            self.logger.info("WebSocket manager stopped (context manager)")

        except Exception as e:
            self.logger.error(f"Error in context manager example: {e}")
            raise

    async def run_multi_symbol_example(self) -> None:
        """Run example with multiple symbols."""
        self.logger.info("Starting multi-symbol WebSocket example")

        try:
            # Create WebSocket managers for multiple symbols
            symbols = ["btcusdt", "ethusdt", "adausdt"]
            managers = []

            for symbol in symbols:
                ws_manager = BinanceWebSocketManager(
                    self.config_manager,
                    self.event_hub,
                    symbol=symbol
                )
                managers.append(ws_manager)

            # Start all managers
            start_tasks = [manager.start() for manager in managers]
            await asyncio.gather(*start_tasks, return_exceptions=True)
            self.logger.info(f"Started WebSocket managers for {len(symbols)} symbols")

            # Run for specified duration
            runtime = 15  # seconds
            for i in range(runtime):
                if not self.running:
                    break

                if i % 5 == 0:
                    # Check connection status for all managers
                    for idx, manager in enumerate(managers):
                        state = manager.get_connection_state()
                        self.logger.info(
                            f"{symbols[idx]}: {state.value}"
                        )

                await asyncio.sleep(1)

            # Stop all managers
            stop_tasks = [manager.stop() for manager in managers]
            await asyncio.gather(*stop_tasks, return_exceptions=True)
            self.logger.info("All WebSocket managers stopped")

        except Exception as e:
            self.logger.error(f"Error in multi-symbol example: {e}")
            raise

    async def run(self) -> None:
        """Run the complete example application."""
        try:
            await self.setup()

            self.logger.info("=== Running Basic Example ===")
            await self.run_basic_example()

            if self.running:
                self.logger.info("=== Running Context Manager Example ===")
                await self.run_context_manager_example()

            if self.running:
                self.logger.info("=== Running Multi-Symbol Example ===")
                await self.run_multi_symbol_example()

            # Final statistics
            final_stats = self.market_data_handler.get_statistics()
            self.logger.info(f"Final statistics: {final_stats}")

        except KeyboardInterrupt:
            self.logger.info("Application interrupted by user")
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            raise
        finally:
            self.logger.info("WebSocket example completed")


async def main() -> None:
    """Main entry point for the WebSocket example."""
    example = WebSocketExample()
    await example.run()


if __name__ == "__main__":
    # Check if required environment variables are set
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Note: API keys are not required for WebSocket connections
    # Only trading mode needs to be set
    if not os.getenv("TRADING_MODE"):
        print("Setting default TRADING_MODE=paper")
        os.environ["TRADING_MODE"] = "paper"

    print("Starting WebSocket Manager Example")
    print("This example will connect to Binance WebSocket streams")
    print("and demonstrate real-time market data processing.")
    print()
    print("Press Ctrl+C to stop the example gracefully")
    print("=" * 50)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExample stopped by user")
    except Exception as e:
        print(f"Example failed: {e}")
        sys.exit(1)
