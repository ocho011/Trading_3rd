#!/usr/bin/env python3
"""
ICT Strategy Usage Example

This example demonstrates how to create and use the ICTStrategy class
for institutional trading pattern analysis with order block detection.
"""

import time
from typing import Dict, Any

from trading_bot.core.event_hub import EventHub
from trading_bot.market_data.data_processor import MarketData
from trading_bot.strategies import (
    ICTStrategy,
    IctConfiguration,
    IctStrategyConfiguration,
    create_ict_strategy,
)


def create_sample_market_data(
    symbol: str, price: float, volume: float, **metadata: Any
) -> MarketData:
    """Create sample market data for testing.

    Args:
        symbol: Trading symbol
        price: Current price
        volume: Trading volume
        **metadata: Additional metadata

    Returns:
        MarketData instance
    """
    return MarketData(
        symbol=symbol,
        timestamp=int(time.time() * 1000),
        price=price,
        volume=volume,
        source="example",
        data_type="tick",
        raw_data={},
        metadata=metadata,
    )


def main() -> None:
    """Main example function demonstrating ICT strategy usage."""
    print("ICT Strategy Example")
    print("=" * 50)

    # 1. Create Event Hub
    event_hub = EventHub()

    # 2. Create ICT Configuration
    ict_config = IctConfiguration(
        min_candles_for_structure=20,
        volume_confirmation_factor=1.5,
        min_confidence_threshold=0.7,
    )

    # 3. Create ICT Strategy Configuration
    ict_strategy_config = IctStrategyConfiguration(
        min_order_block_confidence=0.75,
        max_order_blocks_for_signals=3,
        confidence_based_sizing=True,
        target_multiplier=2.0,
    )

    # 4. Create ICT Strategy using factory function
    strategy = create_ict_strategy(
        name="ICT_EURUSD_5M",
        symbol="EURUSD",
        timeframe="5m",
        event_hub=event_hub,
        ict_config=ict_config,
        ict_strategy_config=ict_strategy_config,
        min_confidence=0.7,
        risk_tolerance=0.02,
    )

    print(f"Created ICT Strategy: {strategy._config.name}")
    print(f"Symbol: {strategy._config.symbol}")
    print(f"Timeframe: {strategy._config.timeframe}")

    # 5. Initialize the strategy
    try:
        strategy.initialize()
        print("✓ Strategy initialized successfully")
    except Exception as e:
        print(f"✗ Strategy initialization failed: {e}")
        return

    # 6. Generate sample market data and test signal generation
    print("\nTesting signal generation with sample data:")
    print("-" * 40)

    sample_prices = [1.1050, 1.1055, 1.1060, 1.1058, 1.1062, 1.1065]

    for i, price in enumerate(sample_prices):
        # Create market data with OHLC information
        market_data = create_sample_market_data(
            symbol="EURUSD",
            price=price,
            volume=1000.0 + (i * 100),
            open_price=price - 0.0002,
            high_price=price + 0.0001,
            low_price=price - 0.0003,
            is_closed=True,
        )

        try:
            # Generate signal
            signal = strategy.generate_signal(market_data)

            if signal:
                print(f"Signal generated: {signal.signal_type.value}")
                print(f"  Price: {signal.price:.6f}")
                print(f"  Confidence: {signal.confidence:.3f}")
                print(f"  Strength: {signal.strength.value}")
                print(f"  Stop Loss: {signal.stop_loss:.6f}")
                print(f"  Target: {signal.target_price:.6f}")
                print(f"  Reasoning: {signal.reasoning}")
                print()
            else:
                print(f"No signal at price {price:.6f}")

        except Exception as e:
            print(f"Error generating signal at price {price}: {e}")

        # Small delay to simulate real-time data
        time.sleep(0.1)

    # 7. Get strategy performance metrics
    print("\nStrategy Performance Metrics:")
    print("-" * 30)

    try:
        metrics = strategy.get_ict_performance_metrics()

        print(f"Total Signals: {metrics.get('total_signals', 0)}")
        print(f"ICT Order Blocks Detected: {metrics.get('ict_order_blocks_detected', 0)}")
        print(f"Active Order Blocks: {metrics.get('ict_active_order_blocks', 0)}")
        print(f"Average Signal Confidence: {metrics.get('ict_average_signal_confidence', 0):.3f}")
        print(f"Mitigation Success Rate: {metrics.get('ict_mitigation_success_rate', 0):.3f}")

        # Print configuration
        config = metrics.get('ict_configuration', {})
        print(f"\nICT Configuration:")
        print(f"  Min Confidence: {config.get('min_confidence', 'N/A')}")
        print(f"  Max Order Blocks: {config.get('max_order_blocks', 'N/A')}")
        print(f"  Target Multiplier: {config.get('target_multiplier', 'N/A')}")

    except Exception as e:
        print(f"Error getting metrics: {e}")

    # 8. Get current order blocks
    print("\nCurrent Order Blocks:")
    print("-" * 20)

    try:
        order_blocks = strategy.get_current_order_blocks()

        if order_blocks:
            for i, ob in enumerate(order_blocks, 1):
                print(f"Order Block {i}:")
                print(f"  Type: {ob['type']}")
                print(f"  Direction: {ob['direction']}")
                print(f"  Price Range: {ob['low_price']:.6f} - {ob['high_price']:.6f}")
                print(f"  Confidence: {ob['confidence']:.3f}")
                print(f"  Tested: {ob['tested']}")
                print(f"  Mitigations: {ob['mitigation_count']}")
                print()
        else:
            print("No active order blocks")

    except Exception as e:
        print(f"Error getting order blocks: {e}")

    # 9. Cleanup strategy
    try:
        strategy.cleanup()
        print("✓ Strategy cleanup completed")
    except Exception as e:
        print(f"✗ Strategy cleanup failed: {e}")

    print("\nICT Strategy example completed!")


if __name__ == "__main__":
    main()