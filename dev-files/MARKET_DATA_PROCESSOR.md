# Market Data Processor

A comprehensive market data processing system for real-time trading applications, built following SOLID principles and dependency injection patterns.

## Overview

The Market Data Processor is a production-ready component that handles:

- Real-time market data processing from cryptocurrency exchanges
- Multi-timeframe candle aggregation (1m, 5m, 15m, 1h, 4h, 1d)
- Data validation and integrity checking
- Event-driven publishing of processed data
- Comprehensive error handling and logging

## Architecture

The system follows SOLID principles and implements:

- **Single Responsibility**: Each class has one clear purpose
- **Open-Closed**: Extensible through interfaces and dependency injection
- **Liskov Substitution**: Proper inheritance hierarchies
- **Interface Segregation**: Focused interfaces for specific needs
- **Dependency Inversion**: Depends on abstractions, not implementations

## Core Components

### Data Structures

#### `MarketData`
Standardized market data format with validation:
```python
@dataclass
class MarketData:
    symbol: str
    timestamp: int
    price: float
    volume: float
    source: str
    data_type: str
    raw_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### `CandleData`
OHLCV candle data for aggregated market data:
```python
@dataclass
class CandleData:
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
```

#### `Timeframe`
Enum defining supported timeframe intervals:
```python
class Timeframe(Enum):
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
```

### Validation Components

#### `IMarketDataValidator`
Interface for data validation strategies with Binance implementation:
- Validates raw WebSocket data format
- Checks candle data integrity
- Handles malformed data gracefully

#### `BinanceDataValidator`
Binance-specific validator implementation:
- Validates kline (candlestick) data
- Validates ticker data
- Performs business logic validation

### Aggregation Components

#### `ICandleAggregator`
Interface for candle aggregation strategies:
- Updates candles with new market data
- Manages multiple timeframes simultaneously
- Handles candle completion events

#### `TimeframeCandleAggregator`
Multi-timeframe candle aggregator:
- Real-time OHLCV calculation
- Automatic candle completion detection
- Support for multiple symbols and timeframes

### Main Processor

#### `MarketDataProcessor`
Core processing engine that:
- Subscribes to `MARKET_DATA_RECEIVED` events
- Validates and parses incoming data
- Aggregates data into candles
- Publishes processed events:
  - `CANDLE_DATA_PROCESSED`
  - `PRICE_UPDATE`
  - `VOLUME_UPDATE`

## Configuration

The processor supports configuration through `ConfigManager`:

```python
# Data freshness validation
max_data_age_seconds: int = 300

# Event publishing controls
enable_volume_updates: bool = True
enable_price_updates: bool = True

# Supported timeframes
supported_timeframes: List[str] = ["1m", "5m", "15m", "1h"]
```

## Usage

### Basic Setup

```python
from trading_bot.core.config_manager import ConfigManager, EnvConfigLoader
from trading_bot.core.event_hub import EventHub
from trading_bot.market_data.data_processor import create_market_data_processor

# Initialize dependencies
config_manager = ConfigManager(EnvConfigLoader())
config_manager.load_configuration()
event_hub = EventHub()

# Create processor
processor = create_market_data_processor(
    config_manager=config_manager,
    event_hub=event_hub,
    supported_timeframes=["1m", "5m", "15m", "1h"]
)

# Start processing
await processor.start()
```

### Event Handling

```python
# Subscribe to processed candle data
def handle_candle_data(event_data):
    candle = event_data["candle"]
    symbol = event_data["symbol"]
    timeframe = event_data["timeframe"]

    print(f"Candle completed: {symbol} {timeframe}")
    print(f"OHLCV: {candle.open_price}, {candle.high_price}, "
          f"{candle.low_price}, {candle.close_price}, {candle.volume}")

event_hub.subscribe(EventType.CANDLE_DATA_PROCESSED, handle_candle_data)

# Subscribe to price updates
def handle_price_update(event_data):
    symbol = event_data["symbol"]
    price = event_data["price"]
    print(f"Price update: {symbol} = ${price}")

event_hub.subscribe(EventType.PRICE_UPDATE, handle_price_update)
```

### Integration with WebSocket Manager

```python
from trading_bot.market_data.websocket_manager import BinanceWebSocketManager

# Create WebSocket manager
ws_manager = BinanceWebSocketManager(
    config_manager=config_manager,
    event_hub=event_hub,
    symbol="btcusdt"
)

# Start both services
await ws_manager.start()  # Publishes MARKET_DATA_RECEIVED events
await processor.start()   # Processes and publishes structured events
```

## Data Flow

1. **WebSocket Manager** receives raw data from Binance
2. **WebSocket Manager** publishes `MARKET_DATA_RECEIVED` event
3. **Market Data Processor** subscribes and receives raw data
4. **Data Validator** validates data format and integrity
5. **Parser** converts raw data to `MarketData` structure
6. **Aggregator** updates candles for multiple timeframes
7. **Publisher** emits events for completed candles and updates

## Event Types

### Input Events
- `MARKET_DATA_RECEIVED`: Raw market data from WebSocket

### Output Events
- `CANDLE_DATA_PROCESSED`: Completed candle data
- `PRICE_UPDATE`: Real-time price changes
- `VOLUME_UPDATE`: Volume information

## Error Handling

The system implements comprehensive error handling:

### Custom Exceptions
- `DataProcessingError`: Base processing error
- `InvalidDataError`: Data validation failures
- `TimeframeError`: Timeframe-related issues
- `CandleAggregationError`: Aggregation failures

### Graceful Degradation
- Invalid data is logged and skipped
- Processing continues despite individual errors
- Statistics tracking for monitoring
- Configurable data freshness validation

## Testing

Comprehensive unit tests cover:

- Data structure validation
- Validator functionality
- Aggregator behavior
- Processor event flow
- Error handling scenarios
- Configuration edge cases

Run tests:
```bash
python3 -m pytest tests/test_data_processor.py -v
```

## Performance Considerations

- Efficient timestamp-based candle alignment
- Minimal memory usage for current candles
- Configurable data retention and freshness
- Thread-safe event publishing
- Optimized for high-frequency data

## Monitoring

The processor provides statistics:

```python
stats = processor.get_processing_statistics()
# Returns:
# {
#     "processed_count": 1000,
#     "error_count": 5,
#     "last_processed_time": 1640995200,
#     "active_symbols": ["BTCUSDT", "ETHUSDT"],
#     "supported_timeframes": ["1m", "5m", "15m", "1h"]
# }
```

## Extension Points

The system is designed for extensibility:

1. **New Exchange Support**: Implement `IMarketDataValidator`
2. **Custom Aggregation**: Implement `ICandleAggregator`
3. **Additional Validation**: Extend validator classes
4. **New Event Types**: Add events to `EventType` class
5. **Custom Processing**: Extend `MarketDataProcessor`

## Examples

See `example_data_processor.py` for a complete working example that demonstrates:

- System initialization
- Event handling
- Error recovery
- Graceful shutdown
- Statistics monitoring
- Integration with WebSocket Manager

## Dependencies

- `trading_bot.core.event_hub`: Event-driven communication
- `trading_bot.core.config_manager`: Configuration management
- `trading_bot.core.logger`: Structured logging
- `trading_bot.market_data.websocket_manager`: Data source

## Code Quality

The implementation follows project standards:

- ✅ PEP 8 compliance with type hints
- ✅ Google-style docstrings
- ✅ Functions under 20 lines
- ✅ SOLID principles adherence
- ✅ Comprehensive error handling
- ✅ Dependency injection patterns
- ✅ Unit test coverage
- ✅ Production-ready logging