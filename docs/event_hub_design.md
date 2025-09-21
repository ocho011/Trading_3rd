# EventHub Design Documentation

## Overview

The EventHub is a centralized event management system for the trading bot application. It provides a thread-safe, event-driven communication mechanism that follows the Observer pattern and adheres to SOLID principles.

## Architecture Design

### Core Components

1. **EventHubInterface**: Abstract interface defining the contract for event hub implementations
2. **EventHub**: Concrete implementation of the event hub
3. **EventType**: Constants class defining all available event types

### Design Principles

#### SOLID Principles Compliance

- **Single Responsibility Principle (SRP)**: EventHub focuses solely on event management and distribution
- **Open-Closed Principle (OCP)**: The system is open for extension (new event types) but closed for modification
- **Liskov Substitution Principle (LSP)**: EventHub implements EventHubInterface and can be substituted
- **Interface Segregation Principle (ISP)**: EventHubInterface provides only essential event management methods
- **Dependency Inversion Principle (DIP)**: Components depend on the EventHubInterface abstraction

#### Additional Design Principles

- **Thread Safety**: All operations are protected by threading locks
- **Type Safety**: Comprehensive type hints for all parameters and return values
- **Error Handling**: Proper exception handling with descriptive error messages
- **Documentation**: Google-style docstrings for all public methods

## Event Types

The system defines event types across several categories:

### Market Data Events
- `MARKET_DATA_RECEIVED`: General market data updates
- `PRICE_UPDATE`: Price changes
- `VOLUME_UPDATE`: Volume changes
- `ORDER_BOOK_UPDATE`: Order book modifications

### Trading Signal Events
- `TRADING_SIGNAL_GENERATED`: General trading signals
- `BUY_SIGNAL`: Buy recommendations
- `SELL_SIGNAL`: Sell recommendations
- `HOLD_SIGNAL`: Hold recommendations

### Order Management Events
- `ORDER_PLACED`: New order submissions
- `ORDER_FILLED`: Order executions
- `ORDER_CANCELLED`: Order cancellations
- `ORDER_REJECTED`: Order rejections

### Risk Management Events
- `RISK_LIMIT_EXCEEDED`: Risk threshold breaches
- `POSITION_SIZE_WARNING`: Position size alerts
- `PORTFOLIO_REBALANCE`: Portfolio rebalancing events

### System Events
- `SYSTEM_STARTUP`: System initialization
- `SYSTEM_SHUTDOWN`: System shutdown
- `ERROR_OCCURRED`: Error notifications
- `CONNECTION_LOST`: Connection failures
- `CONNECTION_RESTORED`: Connection recovery

## Class Structure

### EventHub Class

```python
class EventHub(EventHubInterface):
    def __init__(self) -> None
    def subscribe(self, event_type: str, callback: Callable[[Any], None]) -> None
    def unsubscribe(self, event_type: str, callback: Callable[[Any], None]) -> None
    def publish(self, event_type: str, data: Any) -> None
    def get_subscriber_count(self, event_type: str) -> int
    def clear_subscribers(self, event_type: str = None) -> None
```

### Key Attributes

- `_subscribers`: Dictionary mapping event types to callback function lists
- `_lock`: Threading lock for thread-safe operations

## Thread Safety

The EventHub implementation ensures thread safety through:

1. **Threading Locks**: All subscriber modifications are protected by locks
2. **Atomic Operations**: Individual operations are atomic and consistent
3. **Concurrent Access**: Multiple threads can safely subscribe, publish, and unsubscribe

## Usage Examples

### Basic Usage

```python
from trading_bot.core.event_hub import EventHub, EventType

# Create event hub
hub = EventHub()

# Define event handler
def price_handler(data):
    print(f"Price update: {data}")

# Subscribe to events
hub.subscribe(EventType.PRICE_UPDATE, price_handler)

# Publish events
hub.publish(EventType.PRICE_UPDATE, {"symbol": "BTCUSD", "price": 50000})
```

### Multiple Subscribers

```python
# Multiple handlers for the same event
def handler1(data): pass
def handler2(data): pass

hub.subscribe(EventType.MARKET_DATA_RECEIVED, handler1)
hub.subscribe(EventType.MARKET_DATA_RECEIVED, handler2)

# Both handlers will be called
hub.publish(EventType.MARKET_DATA_RECEIVED, market_data)
```

## Implementation Status

### Completed Features

- ✅ Basic class structure and initialization
- ✅ Event type constants definition
- ✅ Abstract interface definition
- ✅ Thread-safe initialization
- ✅ Method signatures with type hints
- ✅ Comprehensive documentation
- ✅ Unit test suite

### Pending Implementation

- ⏳ `subscribe()` method implementation
- ⏳ `unsubscribe()` method implementation
- ⏳ `publish()` method implementation
- ⏳ `get_subscriber_count()` method implementation
- ⏳ `clear_subscribers()` method implementation

## Testing

The EventHub includes comprehensive unit tests covering:

- Event type constant validation
- Class initialization and thread safety
- Method signature verification
- Interface implementation compliance
- Integration scenarios

Run tests with:
```bash
python3 -m pytest tests/test_event_hub.py -v
```

## Integration Points

The EventHub is designed to integrate with:

1. **Market Data Providers**: Publish price and volume updates
2. **Trading Strategies**: Subscribe to market data, publish signals
3. **Order Management**: Subscribe to signals, publish order events
4. **Risk Management**: Subscribe to all events, publish risk alerts
5. **Notification Systems**: Subscribe to important events for alerts

## Future Enhancements

Potential future improvements include:

1. **Event Filtering**: Allow subscribers to filter events based on criteria
2. **Event History**: Optional event storage for replay and analysis
3. **Priority Handling**: Support for high-priority events
4. **Async Support**: Asynchronous event publishing and handling
5. **Event Serialization**: Support for persistent event storage

## Performance Considerations

- **Memory Efficiency**: Subscriber lists are cleaned up when empty
- **Execution Speed**: Synchronous event delivery for low latency
- **Scalability**: Designed to handle hundreds of subscribers per event type
- **Thread Overhead**: Minimal locking for optimal performance