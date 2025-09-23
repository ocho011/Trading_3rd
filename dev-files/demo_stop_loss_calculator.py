#!/usr/bin/env python3
"""
Demonstration script for the stop-loss calculator module.

This script shows how to use the stop-loss calculator with different methods
and configurations to calculate stop-loss and take-profit levels.
"""

import sys
import os

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from trading_bot.risk_management.stop_loss_calculator import (
    create_stop_loss_calculator,
    StopLossMethod,
    PositionType,
)
from trading_bot.core.event_hub import EventHub


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_result(result, title: str) -> None:
    """Print formatted stop-loss calculation result."""
    print(f"\n{title}:")
    print(f"  Entry Price: ${result.entry_price:.2f}")
    print(f"  Position Type: {result.position_type.value}")
    print(f"  Method: {result.method_used.value}")

    if result.stop_loss_level:
        print(f"  Stop Loss: ${result.stop_loss_level.price:.2f} ({result.stop_loss_level.percentage_from_entry:.2f}%)")
    else:
        print("  Stop Loss: None")

    if result.take_profit_level:
        print(f"  Take Profit: ${result.take_profit_level.price:.2f} ({result.take_profit_level.percentage_from_entry:.2f}%)")
    else:
        print("  Take Profit: None")

    print(f"  Risk/Reward Ratio: {result.risk_reward_ratio:.2f}")
    print(f"  Confidence: {result.overall_confidence:.2f}")

    if result.warnings:
        print(f"  Warnings: {', '.join(result.warnings)}")

    if result.recommendations:
        print(f"  Recommendations:")
        for rec in result.recommendations:
            print(f"    - {rec}")


def demo_fixed_percentage():
    """Demonstrate fixed percentage stop-loss calculation."""
    print_section("Fixed Percentage Method")

    # Create calculator with 2% stop-loss and 2:1 risk-reward
    calculator = create_stop_loss_calculator(
        method=StopLossMethod.FIXED_PERCENTAGE,
        stop_loss_percentage=2.0,
        risk_reward_ratio=2.0,
    )

    # Long position
    long_result = calculator.calculate_levels(100.0, PositionType.LONG)
    print_result(long_result, "Long Position (Buy)")

    # Short position
    short_result = calculator.calculate_levels(100.0, PositionType.SHORT)
    print_result(short_result, "Short Position (Sell)")


def demo_atr_based():
    """Demonstrate ATR-based stop-loss calculation."""
    print_section("ATR-Based Method")

    # Create calculator with ATR-based method
    calculator = create_stop_loss_calculator(
        method=StopLossMethod.ATR_BASED,
        atr_multiplier=2.0,
        risk_reward_ratio=3.0,
    )

    # Provide market data with ATR value
    market_data = {
        "atr": 1.5,  # $1.50 ATR
        "volatility": 0.03,
    }

    # Long position with market data
    result = calculator.calculate_levels(100.0, PositionType.LONG, market_data)
    print_result(result, "Long Position with ATR = $1.50")

    # Calculate risk for 1000 shares
    position_size = 1000
    risk_amount = result.get_risk_amount(position_size)
    profit_potential = result.get_profit_potential(position_size)

    print(f"\n  For {position_size} shares:")
    print(f"    Risk Amount: ${risk_amount:.2f}")
    print(f"    Profit Potential: ${profit_potential:.2f}")


def demo_support_resistance():
    """Demonstrate support/resistance-based stop-loss calculation."""
    print_section("Support/Resistance Method")

    # Create calculator with support/resistance levels
    calculator = create_stop_loss_calculator(
        method=StopLossMethod.SUPPORT_RESISTANCE,
        support_level=95.0,
        resistance_level=110.0,
        buffer_percentage=0.5,  # 0.5% buffer
        risk_reward_ratio=2.5,
    )

    # Long position using support level
    long_result = calculator.calculate_levels(100.0, PositionType.LONG)
    print_result(long_result, "Long Position (using support at $95)")

    # Short position using resistance level
    short_result = calculator.calculate_levels(100.0, PositionType.SHORT)
    print_result(short_result, "Short Position (using resistance at $110)")


def demo_volatility_adjusted():
    """Demonstrate volatility-adjusted stop-loss calculation."""
    print_section("Volatility-Adjusted Method")

    # Create calculator with volatility adjustment
    calculator = create_stop_loss_calculator(
        method=StopLossMethod.VOLATILITY_ADJUSTED,
        base_stop_percentage=2.0,
        volatility_multiplier=1.5,
        risk_reward_ratio=2.0,
    )

    # Low volatility scenario
    low_vol_data = {"volatility": 0.01}  # 1% volatility
    low_vol_result = calculator.calculate_levels(100.0, PositionType.LONG, low_vol_data)
    print_result(low_vol_result, "Low Volatility (1%)")

    # High volatility scenario
    high_vol_data = {"volatility": 0.05}  # 5% volatility
    high_vol_result = calculator.calculate_levels(100.0, PositionType.LONG, high_vol_data)
    print_result(high_vol_result, "High Volatility (5%)")


def demo_with_event_hub():
    """Demonstrate stop-loss calculator with event publishing."""
    print_section("Event-Driven Integration")

    # Create event hub
    event_hub = EventHub()

    # Subscribe to events
    events_received = []

    def on_calculation_event(event_data):
        events_received.append(event_data)
        print(f"  Event received: {event_data['entry_price']} -> {event_data['stop_loss_price']}")

    event_hub.subscribe("position_size_warning", on_calculation_event)

    # Create calculator with event hub
    calculator = create_stop_loss_calculator(
        method=StopLossMethod.FIXED_PERCENTAGE,
        stop_loss_percentage=3.0,
        event_hub=event_hub,
    )

    print("Calculating levels with event publishing...")
    result = calculator.calculate_levels(100.0, PositionType.LONG)
    print_result(result, "Calculation with Events")

    print(f"\nEvents received: {len(events_received)}")


def demo_safety_limits():
    """Demonstrate safety limits and validation."""
    print_section("Safety Limits and Validation")

    # Create calculator with tight safety limits
    calculator = create_stop_loss_calculator(
        method=StopLossMethod.FIXED_PERCENTAGE,
        stop_loss_percentage=8.0,  # Request 8%
        max_stop_loss_percentage=5.0,  # But limit to 5%
        min_stop_loss_percentage=1.0,
        risk_reward_ratio=2.0,
    )

    result = calculator.calculate_levels(100.0, PositionType.LONG)
    print_result(result, "Stop-Loss Capped by Safety Limits")

    # Validate the result
    is_valid, validation_errors = result.validate_levels()
    print(f"\nValidation Result: {'Valid' if is_valid else 'Invalid'}")
    if validation_errors:
        for error in validation_errors:
            print(f"  Error: {error}")


def main():
    """Run all demonstration scenarios."""
    print("Stop-Loss Calculator Demonstration")
    print("This demo shows various calculation methods and features.")

    try:
        demo_fixed_percentage()
        demo_atr_based()
        demo_support_resistance()
        demo_volatility_adjusted()
        demo_with_event_hub()
        demo_safety_limits()

        print_section("Demo Complete")
        print("All stop-loss calculation methods demonstrated successfully!")
        print("\nThe stop-loss calculator provides:")
        print("  • Multiple calculation methods (Fixed %, ATR, S/R, Volatility)")
        print("  • Automatic take-profit calculation with risk-reward ratios")
        print("  • Safety limits and validation")
        print("  • Event-driven integration")
        print("  • Comprehensive error handling")
        print("  • Support for both long and short positions")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()