"""Example usage of EventHub class.

This example demonstrates how to use the EventHub class for event-driven
communication in the trading bot system.
"""

from typing import Any
from trading_bot.core.event_hub import EventHub, EventType


def market_data_handler(data: Any) -> None:
    """Handle market data events."""
    print(f"Market data received: {data}")


def trading_signal_handler(data: Any) -> None:
    """Handle trading signal events."""
    print(f"Trading signal generated: {data}")


def order_handler(data: Any) -> None:
    """Handle order events."""
    print(f"Order event: {data}")


def main() -> None:
    """Demonstrate EventHub usage."""
    # Create EventHub instance
    event_hub = EventHub()

    # Subscribe to various events
    print("Setting up event subscribers...")
    event_hub.subscribe(EventType.MARKET_DATA_RECEIVED, market_data_handler)
    event_hub.subscribe(EventType.TRADING_SIGNAL_GENERATED, trading_signal_handler)
    event_hub.subscribe(EventType.ORDER_PLACED, order_handler)
    event_hub.subscribe(EventType.ORDER_FILLED, order_handler)

    # Simulate publishing events
    print("\nPublishing sample events...")

    # Market data event
    market_data = {
        "symbol": "BTCUSD",
        "price": 50000.0,
        "volume": 1.5,
        "timestamp": "2024-01-01T10:00:00Z"
    }
    event_hub.publish(EventType.MARKET_DATA_RECEIVED, market_data)

    # Trading signal event
    signal_data = {
        "signal_type": "BUY",
        "symbol": "BTCUSD",
        "strength": 0.85,
        "timestamp": "2024-01-01T10:00:01Z"
    }
    event_hub.publish(EventType.TRADING_SIGNAL_GENERATED, signal_data)

    # Order events
    order_data = {
        "order_id": "12345",
        "symbol": "BTCUSD",
        "side": "BUY",
        "quantity": 0.1,
        "price": 50000.0
    }
    event_hub.publish(EventType.ORDER_PLACED, order_data)
    event_hub.publish(EventType.ORDER_FILLED, {**order_data, "filled_price": 49999.5})

    print("\nEvent publishing completed.")
    print("Note: Since subscribe/publish methods have pass statements,")
    print("no actual event handling will occur until implementation is added.")


if __name__ == "__main__":
    main()
