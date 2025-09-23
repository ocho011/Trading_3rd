#!/usr/bin/env python3
"""
Position Sizer Demo Script

This script demonstrates the usage of the position sizing algorithm
with different methods and configurations.
"""

import sys
from pathlib import Path

# Add the project root to sys.path for importing modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trading_bot.risk_management.position_sizer import (
    PositionSizer,
    PositionSizingConfig,
    PositionSizingMethod,
    create_position_sizer,
)
from trading_bot.core.event_hub import EventHub


def demonstrate_fixed_percentage():
    """Demonstrate fixed percentage position sizing."""
    print("=" * 60)
    print("FIXED PERCENTAGE POSITION SIZING")
    print("=" * 60)

    # Create a position sizer with fixed percentage method
    sizer = create_position_sizer(
        account_balance=10000.0,
        risk_percentage=2.0,  # Risk 2% per trade
        method=PositionSizingMethod.FIXED_PERCENTAGE,
        max_position_size=100.0,
        min_position_size=0.001,
    )

    # Test scenario 1: With stop loss
    print("\nScenario 1: BTC/USDT with stop loss")
    print("Account Balance: $10,000")
    print("Risk Percentage: 2% ($200)")
    print("Entry Price: $50,000")
    print("Stop Loss: $48,000")

    result = sizer.calculate_position_size(
        entry_price=50000.0,
        stop_loss_price=48000.0,
    )

    print(f"Position Size: {result.position_size:.6f} BTC")
    print(f"Risk Amount: ${result.risk_amount:.2f}")
    print(f"Total Position Value: ${result.position_size * result.entry_price:.2f}")
    print(f"Risk per Share: ${result.entry_price - (result.stop_loss_price or 0):.2f}")

    # Test scenario 2: Without stop loss
    print("\nScenario 2: ETH/USDT without stop loss")
    print("Entry Price: $3,000")

    result = sizer.calculate_position_size(entry_price=3000.0)

    print(f"Position Size: {result.position_size:.6f} ETH")
    print(f"Risk Amount: ${result.risk_amount:.2f}")
    print(f"Total Position Value: ${result.position_size * result.entry_price:.2f}")


def demonstrate_kelly_criterion():
    """Demonstrate Kelly criterion position sizing."""
    print("\n" + "=" * 60)
    print("KELLY CRITERION POSITION SIZING")
    print("=" * 60)

    # Create Kelly criterion sizer with historical performance data
    sizer = create_position_sizer(
        account_balance=10000.0,
        risk_percentage=5.0,  # Higher base risk for Kelly
        method=PositionSizingMethod.KELLY_CRITERION,
        kelly_win_rate=0.65,  # 65% win rate
        kelly_avg_win=120.0,  # Average gain of $120
        kelly_avg_loss=80.0,  # Average loss of $80
        max_position_size=100.0,
    )

    print("Historical Performance:")
    print("Win Rate: 65%")
    print("Average Win: $120")
    print("Average Loss: $80")
    print("Risk Budget: 5% ($500)")

    result = sizer.calculate_position_size(entry_price=100.0)

    print(f"\nKelly Calculation:")
    b = 120.0 / 80.0  # odds
    p = 0.65  # win rate
    q = 0.35  # loss rate
    kelly_fraction = (b * p - q) / b
    kelly_capped = min(kelly_fraction, 0.25)  # Safety cap

    print(f"Odds (b): {b:.2f}")
    print(f"Kelly Fraction: {kelly_fraction:.3f} ({kelly_fraction*100:.1f}%)")
    print(f"Capped Fraction: {kelly_capped:.3f} ({kelly_capped*100:.1f}%)")
    print(f"Position Size: {result.position_size:.3f}")
    print(f"Risk Amount: ${result.risk_amount:.2f}")


def demonstrate_volatility_adjusted():
    """Demonstrate volatility-adjusted position sizing."""
    print("\n" + "=" * 60)
    print("VOLATILITY-ADJUSTED POSITION SIZING")
    print("=" * 60)

    # High volatility scenario
    high_vol_sizer = create_position_sizer(
        account_balance=10000.0,
        risk_percentage=2.0,
        method=PositionSizingMethod.VOLATILITY_ADJUSTED,
        volatility_factor=0.5,  # High volatility
        max_position_size=100.0,
    )

    # Low volatility scenario
    low_vol_sizer = create_position_sizer(
        account_balance=10000.0,
        risk_percentage=2.0,
        method=PositionSizingMethod.VOLATILITY_ADJUSTED,
        volatility_factor=0.1,  # Low volatility
        max_position_size=100.0,
    )

    entry_price = 100.0
    stop_loss_price = 95.0

    print(f"Entry Price: ${entry_price}")
    print(f"Stop Loss: ${stop_loss_price}")
    print(f"Risk per Share: ${entry_price - stop_loss_price}")

    # High volatility result
    high_vol_result = high_vol_sizer.calculate_position_size(
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
    )

    # Low volatility result
    low_vol_result = low_vol_sizer.calculate_position_size(
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
    )

    print(f"\nHigh Volatility (factor: 0.5):")
    print(f"Position Size: {high_vol_result.position_size:.3f}")
    print(f"Volatility Adjustment: {1.0/(1.0+0.5):.3f}")

    print(f"\nLow Volatility (factor: 0.1):")
    print(f"Position Size: {low_vol_result.position_size:.3f}")
    print(f"Volatility Adjustment: {1.0/(1.0+0.1):.3f}")


def demonstrate_with_event_hub():
    """Demonstrate position sizing with event hub integration."""
    print("\n" + "=" * 60)
    print("POSITION SIZING WITH EVENT HUB")
    print("=" * 60)

    # Create event hub
    event_hub = EventHub()

    # Subscribe to position sizing events
    def on_position_sizing_event(event_data):
        result = event_data["result"]
        print(f"Event received: Position size {result.position_size:.6f} calculated")
        if result.warnings:
            print(f"Warnings: {', '.join(result.warnings)}")

    event_hub.subscribe("position_size_warning", on_position_sizing_event)

    # Create sizer with event hub
    sizer = create_position_sizer(
        account_balance=1000.0,  # Small account
        risk_percentage=10.0,    # High risk
        method=PositionSizingMethod.FIXED_PERCENTAGE,
        max_position_size=0.1,   # Small max position to trigger warning
        event_hub=event_hub,
    )

    print("Creating position size that will trigger max limit warning...")
    result = sizer.calculate_position_size(entry_price=50.0)

    print(f"\nResult:")
    print(f"Position Size: {result.position_size:.6f}")
    print(f"Warnings: {result.warnings}")


def main():
    """Run all demonstration scenarios."""
    print("Position Sizer Algorithm Demonstration")
    print("=====================================")

    try:
        demonstrate_fixed_percentage()
        demonstrate_kelly_criterion()
        demonstrate_volatility_adjusted()
        demonstrate_with_event_hub()

        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✓ Fixed percentage position sizing")
        print("✓ Kelly criterion with safety caps")
        print("✓ Volatility-adjusted sizing")
        print("✓ Position size limits enforcement")
        print("✓ Event hub integration")
        print("✓ Comprehensive input validation")
        print("✓ Error handling and warnings")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()