#!/usr/bin/env python3
"""
Example demonstrating WebSocket auto-reconnection functionality.

This script shows how to use the enhanced BinanceWebSocketManager with
auto-reconnection capabilities including exponential backoff and configuration.
"""

import asyncio
import logging
from trading_bot.core.config_manager import ConfigManager, EnvConfigLoader
from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.market_data.websocket_manager import (
    BinanceWebSocketManager,
    ReconnectionConfig,
    create_websocket_manager
)


async def handle_market_data(event_type: EventType, data: dict) -> None:
    """Handle incoming market data events.

    Args:
        event_type: Type of event received
        data: Market data dictionary
    """
    if event_type == EventType.MARKET_DATA_RECEIVED:
        print(f"Received {data['type']} data for {data['symbol']}")
        if data['type'] == 'ticker':
            ticker_data = data['data']
            print(f"  Price: {ticker_data.get('c', 'N/A')}")
            print(f"  24h Volume: {ticker_data.get('v', 'N/A')}")
        elif data['type'] == 'kline':
            kline_data = data['data']['k']
            print(f"  Open: {kline_data.get('o', 'N/A')}")
            print(f"  Close: {kline_data.get('c', 'N/A')}")
            print(f"  High: {kline_data.get('h', 'N/A')}")
            print(f"  Low: {kline_data.get('l', 'N/A')}")


async def demonstrate_basic_reconnection():
    """Demonstrate basic auto-reconnection with default settings."""
    print("=== Basic Auto-Reconnection Demo ===")

    # Setup configuration manager
    config_manager = ConfigManager(EnvConfigLoader())
    config_manager.load_configuration()

    # Setup event hub
    event_hub = EventHub()
    event_hub.subscribe(EventType.MARKET_DATA_RECEIVED, handle_market_data)

    # Create WebSocket manager with default reconnection settings
    try:
        async with create_websocket_manager(
            config_manager,
            event_hub,
            symbol="btcusdt"
        ) as ws_manager:
            print(f"WebSocket connected. State: {ws_manager.get_connection_state()}")
            print("Listening for market data (Ctrl+C to stop)...")

            # Run for 30 seconds
            await asyncio.sleep(30)

    except KeyboardInterrupt:
        print("\nStopping WebSocket manager...")
    except Exception as e:
        print(f"Error: {e}")


async def demonstrate_custom_reconnection():
    """Demonstrate custom reconnection configuration."""
    print("=== Custom Auto-Reconnection Demo ===")

    # Setup configuration manager
    config_manager = ConfigManager(EnvConfigLoader())
    config_manager.load_configuration()

    # Setup event hub
    event_hub = EventHub()
    event_hub.subscribe(EventType.MARKET_DATA_RECEIVED, handle_market_data)

    # Create custom reconnection configuration
    custom_reconnection = ReconnectionConfig(
        enabled=True,
        max_retries=5,        # Lower retry count for demo
        initial_delay=0.5,    # Faster initial retry
        max_delay=10.0,       # Lower max delay for demo
        backoff_multiplier=1.5,  # Gentler backoff
        jitter_factor=0.1     # Less jitter
    )

    # Create WebSocket manager with custom reconnection settings
    ws_manager = BinanceWebSocketManager(
        config_manager,
        event_hub,
        symbol="ethusdt",
        reconnection_config=custom_reconnection
    )

    try:
        await ws_manager.start()
        print(f"WebSocket connected. State: {ws_manager.get_connection_state()}")
        print("Reconnection settings:")
        print(f"  Max retries: {custom_reconnection.max_retries}")
        print(f"  Initial delay: {custom_reconnection.initial_delay}s")
        print(f"  Max delay: {custom_reconnection.max_delay}s")
        print(f"  Backoff multiplier: {custom_reconnection.backoff_multiplier}")
        print("Listening for market data (Ctrl+C to stop)...")

        # Run for 30 seconds
        await asyncio.sleep(30)

    except KeyboardInterrupt:
        print("\nStopping WebSocket manager...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await ws_manager.stop()


async def demonstrate_reconnection_scenarios():
    """Demonstrate different reconnection scenarios."""
    print("=== Reconnection Scenarios Demo ===")

    # Setup configuration manager
    config_manager = ConfigManager(EnvConfigLoader())
    config_manager.load_configuration()

    # Setup event hub
    event_hub = EventHub()
    event_hub.subscribe(EventType.MARKET_DATA_RECEIVED, handle_market_data)

    # Configuration for quick testing
    test_reconnection = ReconnectionConfig(
        enabled=True,
        max_retries=3,
        initial_delay=1.0,
        max_delay=5.0,
        backoff_multiplier=2.0,
        jitter_factor=0.2
    )

    ws_manager = BinanceWebSocketManager(
        config_manager,
        event_hub,
        symbol="adausdt",
        reconnection_config=test_reconnection
    )

    try:
        print("1. Starting WebSocket connection...")
        await ws_manager.start()
        print(f"   Connected. State: {ws_manager.get_connection_state()}")

        print("\n2. Running for 10 seconds...")
        await asyncio.sleep(10)

        print("\n3. Simulating network interruption by stopping...")
        await ws_manager.stop()
        print(f"   Stopped. State: {ws_manager.get_connection_state()}")

        print("\n4. Starting again (reconnection will be triggered on failures)...")
        await ws_manager.start()
        print(f"   Restarted. State: {ws_manager.get_connection_state()}")

        print("\n5. Running for another 10 seconds...")
        await asyncio.sleep(10)

    except KeyboardInterrupt:
        print("\nStopping due to keyboard interrupt...")
    except Exception as e:
        print(f"Error during demo: {e}")
    finally:
        await ws_manager.stop()
        print(f"Final state: {ws_manager.get_connection_state()}")


def print_usage():
    """Print usage instructions."""
    print("WebSocket Auto-Reconnection Demo")
    print("===============================")
    print("This demo shows the auto-reconnection capabilities of the")
    print("enhanced BinanceWebSocketManager.")
    print("")
    print("Features demonstrated:")
    print("- Automatic reconnection on connection failures")
    print("- Exponential backoff with jitter")
    print("- Configurable retry limits and delays")
    print("- Graceful handling of manual disconnections")
    print("")
    print("Make sure you have a .env file with BINANCE_API_KEY and")
    print("BINANCE_SECRET_KEY for live mode, or set TRADING_MODE=paper")
    print("for testnet mode.")
    print("")


async def main():
    """Main demonstration function."""
    print_usage()

    # Configure logging to see reconnection messages
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run different demonstration scenarios
    print("Choose a demo:")
    print("1. Basic auto-reconnection with default settings")
    print("2. Custom reconnection configuration")
    print("3. Reconnection scenarios simulation")

    choice = input("Enter choice (1-3): ").strip()

    try:
        if choice == "1":
            await demonstrate_basic_reconnection()
        elif choice == "2":
            await demonstrate_custom_reconnection()
        elif choice == "3":
            await demonstrate_reconnection_scenarios()
        else:
            print("Invalid choice. Running basic demo...")
            await demonstrate_basic_reconnection()
    except Exception as e:
        print(f"Demo error: {e}")


if __name__ == "__main__":
    asyncio.run(main())