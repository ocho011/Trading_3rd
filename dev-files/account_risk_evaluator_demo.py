#!/usr/bin/env python3
"""
Demonstration script for AccountRiskEvaluator module.

This script shows how to use the comprehensive account balance-based
risk assessment system in a practical trading scenario.
"""

import sys
import time
from decimal import Decimal
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trading_bot.core.event_hub import EventHub
from trading_bot.risk_management.account_risk_evaluator import (
    AccountRiskEvaluator,
    AccountState,
    PositionInfo,
    RiskProfile,
    AccountRiskLevel,
    create_account_risk_evaluator,
)
from trading_bot.strategies.base_strategy import TradingSignal, SignalType, SignalStrength


def create_sample_account_state() -> AccountState:
    """Create a sample account state for demonstration."""
    # Sample existing positions
    positions = {
        "AAPL": PositionInfo(
            symbol="AAPL",
            quantity=Decimal('100'),
            average_price=Decimal('150.00'),
            current_price=Decimal('155.00'),
            market_value=Decimal('15500.00'),
            unrealized_pnl=Decimal('500.00'),
            position_type="long",
            entry_timestamp=int(time.time() * 1000) - 86400000,  # 1 day ago
            stop_loss_price=Decimal('145.00'),
            correlation_group="tech",
            volatility=0.025,
        ),
        "GOOGL": PositionInfo(
            symbol="GOOGL",
            quantity=Decimal('20'),
            average_price=Decimal('2500.00'),
            current_price=Decimal('2600.00'),
            market_value=Decimal('52000.00'),
            unrealized_pnl=Decimal('2000.00'),
            position_type="long",
            entry_timestamp=int(time.time() * 1000) - 172800000,  # 2 days ago
            stop_loss_price=Decimal('2400.00'),
            correlation_group="tech",
            volatility=0.030,
        ),
        "SPY": PositionInfo(
            symbol="SPY",
            quantity=Decimal('50'),
            average_price=Decimal('400.00'),
            current_price=Decimal('405.00'),
            market_value=Decimal('20250.00'),
            unrealized_pnl=Decimal('250.00'),
            position_type="long",
            entry_timestamp=int(time.time() * 1000) - 259200000,  # 3 days ago
            stop_loss_price=Decimal('390.00'),
            correlation_group="index",
            volatility=0.015,
        ),
    }

    return AccountState(
        account_id="demo_account_001",
        total_equity=Decimal('150000.00'),
        available_cash=Decimal('60000.00'),
        used_margin=Decimal('25000.00'),
        available_margin=Decimal('35000.00'),
        total_portfolio_value=Decimal('150000.00'),
        unrealized_pnl=Decimal('2750.00'),
        realized_pnl_today=Decimal('500.00'),
        positions=positions,
        buying_power=Decimal('95000.00'),
        max_drawdown_today=Decimal('-1200.00'),
        max_drawdown_period=Decimal('-5000.00'),
    )


def create_sample_trading_signal() -> TradingSignal:
    """Create a sample trading signal."""
    return TradingSignal(
        symbol="TSLA",
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        confidence=0.82,
        price=250.0,
        timestamp=int(time.time() * 1000),
        strategy_name="momentum_strategy",
        reasoning="Strong upward momentum with high volume confirmation",
        target_price=280.0,
        stop_loss=230.0,
        metadata={
            "volatility": 0.045,
            "volume_ratio": 1.8,
            "technical_score": 0.85,
        },
    )


def demonstrate_risk_profiles():
    """Demonstrate different risk profiles."""
    print("=== RISK PROFILE COMPARISON ===\n")

    account_state = create_sample_account_state()
    signal = create_sample_trading_signal()

    profiles = [
        (RiskProfile.CONSERVATIVE, "Conservative Investor"),
        (RiskProfile.MODERATE, "Moderate Investor"),
        (RiskProfile.AGGRESSIVE, "Aggressive Trader"),
    ]

    for profile, description in profiles:
        print(f"--- {description} ---")
        evaluator = create_account_risk_evaluator(risk_profile=profile)

        # Get maximum position size
        max_qty, max_value = evaluator.get_max_position_size(
            signal.symbol, Decimal(str(signal.price)), account_state
        )

        print(f"Max Position Size: {max_qty} shares (${max_value:,.2f})")
        print(f"Max Leverage: {profile.max_leverage}x")
        print(f"Max Position %: {profile.max_position_size:.1%}")
        print(f"Max Portfolio Risk: {profile.max_portfolio_risk:.1%}")

        # Evaluate a moderate position
        test_quantity = min(max_qty, Decimal('40'))  # Test with 40 shares or max allowed
        result = evaluator.evaluate_new_position(signal, account_state, test_quantity)

        print(f"Risk Level: {result.risk_level.value.title()}")
        print(f"Can Add Position: {'Yes' if result.can_add_position else 'No'}")
        print(f"Overall Risk Score: {result.overall_risk_score:.2%}")
        print()


def demonstrate_portfolio_health_check():
    """Demonstrate portfolio health assessment."""
    print("=== PORTFOLIO HEALTH ASSESSMENT ===\n")

    account_state = create_sample_account_state()
    evaluator = create_account_risk_evaluator()

    # Perform health check
    health_result = evaluator.check_portfolio_health(account_state)

    print(f"Account ID: {health_result.account_id}")
    print(f"Overall Risk Level: {health_result.risk_level.value.title()}")
    print(f"Risk Score: {health_result.overall_risk_score:.2%}")
    print(f"Available Buying Power: ${health_result.available_buying_power:,.2f}")
    print()

    print("Risk Factor Breakdown:")
    for factor, score in health_result.risk_factors.items():
        risk_level = "🔴 Critical" if score >= 0.9 else "🟡 High" if score >= 0.7 else "🟢 Normal"
        print(f"  {factor.replace('_', ' ').title()}: {score:.1%} {risk_level}")
    print()

    print("Portfolio Concentration:")
    for symbol, concentration in health_result.concentration_analysis.items():
        print(f"  {symbol}: {concentration:.1%}")
    print()

    if health_result.warnings:
        print("⚠️  Warnings:")
        for warning in health_result.warnings:
            print(f"  • {warning}")
        print()

    if health_result.recommendations:
        print("💡 Recommendations:")
        for recommendation in health_result.recommendations:
            print(f"  • {recommendation}")
        print()


def demonstrate_position_evaluation():
    """Demonstrate new position evaluation."""
    print("=== NEW POSITION EVALUATION ===\n")

    account_state = create_sample_account_state()
    signal = create_sample_trading_signal()
    evaluator = create_account_risk_evaluator()

    # Test different position sizes
    test_quantities = [Decimal('25'), Decimal('50'), Decimal('100'), Decimal('200')]

    for quantity in test_quantities:
        position_value = quantity * Decimal(str(signal.price))
        print(f"--- Testing {quantity} shares (${position_value:,.2f}) ---")

        try:
            result = evaluator.evaluate_new_position(signal, account_state, quantity)

            print(f"Can Add Position: {'✅ Yes' if result.can_add_position else '❌ No'}")
            print(f"Risk Level: {result.risk_level.value.title()}")
            print(f"Risk Score: {result.overall_risk_score:.1%}")
            print(f"Margin Required: ${result.margin_requirement:,.2f}")

            # Show top risk factors
            top_risks = sorted(
                result.risk_factors.items(), key=lambda x: x[1], reverse=True
            )[:3]
            print("Top Risk Factors:")
            for factor, score in top_risks:
                print(f"  • {factor.replace('_', ' ').title()}: {score:.1%}")

            if result.warnings:
                print("Warnings:")
                for warning in result.warnings[:2]:  # Show first 2 warnings
                    print(f"  ⚠️  {warning}")

            print()

        except Exception as e:
            print(f"❌ Error: {e}\n")


def demonstrate_margin_validation():
    """Demonstrate margin requirement validation."""
    print("=== MARGIN REQUIREMENT VALIDATION ===\n")

    account_state = create_sample_account_state()
    signal = create_sample_trading_signal()
    evaluator = create_account_risk_evaluator()

    print(f"Account Available Margin: ${account_state.available_margin:,.2f}")
    print(f"Account Used Margin: ${account_state.used_margin:,.2f}")
    print(f"Margin Utilization: {account_state.margin_utilization:.1%}")
    print()

    # Test different position sizes for margin requirements
    test_quantities = [Decimal('50'), Decimal('100'), Decimal('200'), Decimal('400')]

    for quantity in test_quantities:
        position_value = quantity * Decimal(str(signal.price))
        print(f"Position: {quantity} shares @ ${signal.price} = ${position_value:,.2f}")

        try:
            # Validate margin
            has_sufficient = evaluator.validate_margin_requirements(
                signal, quantity, account_state
            )

            # Calculate margin requirement
            margin_req = evaluator._calculate_margin_requirement(
                signal, quantity, account_state
            )

            print(f"Margin Required: ${margin_req:,.2f}")
            print(f"Sufficient Margin: {'✅ Yes' if has_sufficient else '❌ No'}")

        except Exception as e:
            print(f"❌ Insufficient Margin: {e}")

        print()


def demonstrate_event_integration():
    """Demonstrate event hub integration."""
    print("=== EVENT HUB INTEGRATION ===\n")

    # Create event hub and subscribe to risk events
    event_hub = EventHub()
    events_received = []

    def risk_event_handler(event_data):
        events_received.append(event_data)
        print(f"🔔 Risk Event: {event_data.get('risk_level', 'unknown')} risk detected")

    event_hub.subscribe("position_size_warning", risk_event_handler)
    event_hub.subscribe("risk_limit_exceeded", risk_event_handler)

    # Create evaluator with event hub
    account_state = create_sample_account_state()
    signal = create_sample_trading_signal()
    evaluator = create_account_risk_evaluator(event_hub=event_hub)

    print("Evaluating high-risk position to trigger events...")

    # Test with a large position that should trigger warnings
    large_quantity = Decimal('500')  # Very large position
    result = evaluator.evaluate_new_position(signal, account_state, large_quantity)

    print(f"Position evaluated. Risk level: {result.risk_level.value}")
    print(f"Events received: {len(events_received)}")
    print()


def main():
    """Run all demonstrations."""
    print("🚀 ACCOUNT RISK EVALUATOR DEMONSTRATION 🚀\n")
    print("This demo shows the comprehensive account balance-based")
    print("risk assessment system for trading portfolio management.\n")
    print("=" * 60)
    print()

    try:
        demonstrate_risk_profiles()
        demonstrate_portfolio_health_check()
        demonstrate_position_evaluation()
        demonstrate_margin_validation()
        demonstrate_event_integration()

        print("✅ All demonstrations completed successfully!")
        print("\nThe AccountRiskEvaluator provides:")
        print("• Portfolio-level risk assessment")
        print("• Position concentration monitoring")
        print("• Margin requirement validation")
        print("• Leverage and drawdown protection")
        print("• Configurable risk profiles")
        print("• Real-time event notifications")
        print("\nIntegrate into your trading system for comprehensive risk management!")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()