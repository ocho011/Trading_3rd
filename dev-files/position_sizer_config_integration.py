#!/usr/bin/env python3
"""
Position Sizer Config Integration Demo

This script demonstrates how the position sizing algorithm integrates
with the ConfigManager for risk parameters.
"""

import sys
from pathlib import Path

# Add the project root to sys.path for importing modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trading_bot.core.config_manager import EnvConfigLoader, ConfigManager
from trading_bot.risk_management.position_sizer import (
    create_position_sizer,
    PositionSizingMethod
)


def demonstrate_config_integration():
    """Demonstrate position sizing with ConfigManager integration."""
    print("=" * 60)
    print("POSITION SIZING WITH CONFIG MANAGER INTEGRATION")
    print("=" * 60)

    # Create a mock environment config loader with trading parameters
    env_loader = EnvConfigLoader()
    config_manager = ConfigManager(env_loader)

    # Load configuration
    config_manager.load_configuration()

    # Get trading configuration
    trading_config = config_manager.get_trading_config()

    print("Configuration loaded:")
    print(f"Trading Mode: {trading_config.get('trading_mode', 'paper')}")
    print(f"Max Position Size: {trading_config.get('max_position_size', 0.1)}")
    print(f"Risk Percentage: {trading_config.get('risk_percentage', 2.0)}%")

    # Create position sizer using config values
    sizer = create_position_sizer(
        account_balance=10000.0,  # This would come from account info in real implementation
        risk_percentage=trading_config.get('risk_percentage', 2.0),
        method=PositionSizingMethod.FIXED_PERCENTAGE,
        max_position_size=trading_config.get('max_position_size', 0.1),
        min_position_size=0.001,
    )

    # Calculate position size for a sample trade
    print("\nSample Trade Calculation:")
    print("Entry Price: $45,000 (BTC)")
    print("Stop Loss: $43,000")

    result = sizer.calculate_position_size(
        entry_price=45000.0,
        stop_loss_price=43000.0,
    )

    print(f"\nPosition Sizing Result:")
    print(f"Position Size: {result.position_size:.6f} BTC")
    print(f"Risk Amount: ${result.risk_amount:.2f}")
    print(f"Total Position Value: ${result.position_size * result.entry_price:.2f}")
    print(f"Max Loss if Stop Hit: ${(result.entry_price - result.stop_loss_price) * result.position_size:.2f}")
    print(f"Risk as % of Account: {(result.risk_amount / 10000.0) * 100:.2f}%")

    if result.warnings:
        print(f"Warnings: {', '.join(result.warnings)}")


def main():
    """Run configuration integration demonstration."""
    try:
        demonstrate_config_integration()

        print("\n" + "=" * 60)
        print("CONFIG INTEGRATION DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("\nIntegration Features:")
        print("✓ ConfigManager integration for risk parameters")
        print("✓ Environment variable support")
        print("✓ Consistent configuration across components")
        print("✓ Production-ready parameter handling")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()