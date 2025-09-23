"""
Example usage of the RiskManager orchestrator.

This example demonstrates how to use the RiskManager to process trading signals
and generate comprehensive order requests with risk management.
"""

import asyncio
import time
from decimal import Decimal

from trading_bot.core.config_manager import ConfigManager
from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.market_data.data_processor import MarketData
from trading_bot.risk_management.account_risk_evaluator import AccountState, PositionInfo
from trading_bot.risk_management.risk_manager import create_risk_manager
from trading_bot.strategies.base_strategy import (
    SignalStrength,
    SignalType,
    TradingSignal,
)


async def main():
    """Demonstrate RiskManager usage."""
    print("=== RiskManager Orchestrator Example ===\n")

    # Initialize core components
    event_hub = EventHub()
    config_manager = ConfigManager()

    # Create RiskManager with default components
    risk_manager = create_risk_manager(
        event_hub=event_hub,
        config_manager=config_manager,
        account_balance=50000.0,  # $50K account
        max_position_risk=2.0,    # 2% risk per position
        min_confidence=0.7,       # 70% minimum confidence
        require_stop_loss=True,
        min_risk_reward_ratio=2.0,
    )

    # Create sample trading signal
    signal = TradingSignal(
        symbol="BTCUSD",
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        price=45000.0,
        timestamp=int(time.time() * 1000),
        strategy_name="momentum_strategy",
        confidence=0.85,
        reasoning="Strong momentum breakout with high volume",
        target_price=48000.0,
        stop_loss=43000.0,
        metadata={"volume_spike": True, "rsi": 65},
    )

    # Create sample market data
    market_data = MarketData(
        symbol="BTCUSD",
        timestamp=int(time.time() * 1000),
        price=45000.0,
        volume=1250000,
        metadata={
            "atr": 1200.0,
            "volatility": 0.035,
            "support": 44000.0,
            "resistance": 47000.0,
        },
    )

    # Create sample account state
    account_state = AccountState(
        account_id="demo_account",
        total_equity=Decimal("50000"),
        available_cash=Decimal("30000"),
        used_margin=Decimal("15000"),
        available_margin=Decimal("20000"),
        total_portfolio_value=Decimal("50000"),
        unrealized_pnl=Decimal("2500"),
        realized_pnl_today=Decimal("500"),
        positions={
            "ETHUSD": PositionInfo(
                symbol="ETHUSD",
                quantity=Decimal("10"),
                average_price=Decimal("3000"),
                current_price=Decimal("3100"),
                market_value=Decimal("31000"),
                unrealized_pnl=Decimal("1000"),
                position_type="long",
                entry_timestamp=int(time.time() * 1000) - 3600000,  # 1 hour ago
                stop_loss_price=Decimal("2850"),
            )
        },
    )

    print("1. Processing Trading Signal...")
    print(f"   Symbol: {signal.symbol}")
    print(f"   Signal: {signal.signal_type.value.upper()}")
    print(f"   Price: ${signal.price:,.2f}")
    print(f"   Confidence: {signal.confidence:.1%}")
    print(f"   Strategy: {signal.strategy_name}")

    # Process the signal through risk management
    try:
        order_request = await risk_manager.process_trading_signal(
            signal=signal,
            market_data=market_data,
            account_state=account_state,
        )

        if order_request:
            print("\n2. Order Request Generated Successfully!")
            print(f"   Symbol: {order_request.symbol}")
            print(f"   Quantity: {order_request.quantity}")
            print(f"   Entry Price: ${order_request.entry_price:,.2f}")
            print(f"   Order Type: {order_request.order_type.value}")

            if order_request.stop_loss_price:
                print(f"   Stop Loss: ${order_request.stop_loss_price:,.2f}")

            if order_request.take_profit_price:
                print(f"   Take Profit: ${order_request.take_profit_price:,.2f}")

            print(f"   Risk Amount: ${order_request.total_risk_amount:,.2f}")
            print(f"   Risk %: {order_request.risk_percentage:.2f}%")
            print(f"   Confidence: {order_request.confidence_score:.1%}")
            print(f"   Risk Multiplier: {order_request.risk_multiplier:.2f}x")

            # Calculate risk-reward ratio
            rr_ratio = order_request.calculate_risk_reward_ratio()
            if rr_ratio:
                print(f"   Risk:Reward Ratio: 1:{rr_ratio:.2f}")

            print(f"\n   Validation Checks: {', '.join(order_request.validation_checks)}")

            if order_request.warnings:
                print(f"   Warnings: {len(order_request.warnings)}")
                for warning in order_request.warnings[:3]:  # Show first 3
                    print(f"     - {warning}")

            if order_request.recommendations:
                print(f"   Recommendations: {len(order_request.recommendations)}")
                for rec in order_request.recommendations[:3]:  # Show first 3
                    print(f"     - {rec}")

            print(f"\n   Order Summary:")
            summary = order_request.get_order_summary()
            for key, value in summary.items():
                print(f"     {key}: {value}")

        else:
            print("\n2. Signal Rejected!")
            print("   Signal did not pass risk management checks.")

    except Exception as e:
        print(f"\n2. Error Processing Signal: {e}")

    # Show risk manager statistics
    print("\n3. Risk Manager Statistics:")
    stats = risk_manager.get_risk_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"     {sub_key}: {sub_value}")
        else:
            print(f"   {key}: {value}")

    print("\n=== Example Complete ===")


# Set up event handler to monitor ORDER_REQUEST_GENERATED events
async def order_request_handler(event_data):
    """Handle order request events."""
    order_request = event_data.get("order_request")
    if order_request:
        print(f"\n📈 ORDER_REQUEST_GENERATED Event:")
        print(f"   Symbol: {order_request.symbol}")
        print(f"   Quantity: {order_request.quantity}")
        print(f"   Price: ${order_request.price:,.2f}")
        print(f"   Risk: ${order_request.total_risk_amount:,.2f}")


if __name__ == "__main__":
    # Subscribe to order events before running example
    event_hub = EventHub()
    event_hub.subscribe(EventType.ORDER_REQUEST_GENERATED, order_request_handler)

    # Run the example
    asyncio.run(main())