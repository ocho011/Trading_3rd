#!/usr/bin/env python3
"""
Demo script for the Risk Assessment System.

This script demonstrates the comprehensive risk assessment capabilities
including signal quality analysis, volatility assessment, and position
concentration risk evaluation.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trading_bot.core.event_hub import EventHub
from trading_bot.market_data.data_processor import MarketData
from trading_bot.strategies.base_strategy import (
    TradingSignal,
    SignalType,
    SignalStrength,
)
from trading_bot.risk_management.risk_assessor import (
    RiskAssessor,
    RiskAssessmentConfig,
    RiskFactor,
    RiskLevel,
    create_risk_assessor,
)


def create_sample_signals():
    """Create sample trading signals for demonstration."""
    current_time = int(time.time() * 1000)

    signals = [
        # High quality signal
        TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.VERY_STRONG,
            price=45000.0,
            timestamp=current_time,
            strategy_name="momentum_strategy",
            confidence=0.85,
            metadata={"volatility": 0.02, "volume_surge": True},
            reasoning="Strong bullish momentum with high volume",
            target_price=47000.0,
            stop_loss=43500.0,
        ),

        # Low quality signal
        TradingSignal(
            symbol="ETHUSDT",
            signal_type=SignalType.SELL,
            strength=SignalStrength.WEAK,
            price=3200.0,
            timestamp=current_time - 3600000,  # 1 hour old
            strategy_name="experimental_strategy",
            confidence=0.45,
            metadata={"volatility": 0.08, "trend_uncertain": True},
            reasoning="Weak bearish signal with uncertain trend",
            target_price=3000.0,
            stop_loss=3300.0,
        ),

        # Moderate quality signal
        TradingSignal(
            symbol="ADAUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.MODERATE,
            price=1.25,
            timestamp=current_time - 600000,  # 10 minutes old
            strategy_name="trend_following",
            confidence=0.72,
            metadata={"volatility": 0.03, "support_level": True},
            reasoning="Moderate buy signal at support level",
            target_price=1.35,
            stop_loss=1.20,
        ),
    ]

    return signals


def create_sample_market_data():
    """Create sample market data for demonstration."""
    current_time = int(time.time() * 1000)

    market_data_samples = [
        MarketData(
            symbol="BTCUSDT",
            timestamp=current_time,
            price=45000.0,
            volume=1500.0,
            source="binance",
            data_type="ticker",
            metadata={"volatility": 0.02, "bid_ask_spread": 0.01},
        ),

        MarketData(
            symbol="ETHUSDT",
            timestamp=current_time,
            price=3200.0,
            volume=800.0,
            source="binance",
            data_type="ticker",
            metadata={"volatility": 0.08, "bid_ask_spread": 0.05},
        ),

        MarketData(
            symbol="ADAUSDT",
            timestamp=current_time,
            price=1.25,
            volume=50000.0,
            source="binance",
            data_type="ticker",
            metadata={"volatility": 0.03, "bid_ask_spread": 0.001},
        ),
    ]

    return market_data_samples


def create_sample_portfolio_context():
    """Create sample portfolio context for demonstration."""
    return {
        "total_value": 100000.0,  # $100k portfolio
        "positions": {
            "BTCUSDT": {"value": 25000.0, "quantity": 0.5556},  # 25% allocation
            "ETHUSDT": {"value": 15000.0, "quantity": 4.6875},  # 15% allocation
            "ADAUSDT": {"value": 5000.0, "quantity": 4000.0},   # 5% allocation
        },
        "available_cash": 55000.0,
        "margin_used": 0.0,
    }


def demonstrate_basic_risk_assessment():
    """Demonstrate basic risk assessment functionality."""
    print("=" * 60)
    print("BASIC RISK ASSESSMENT DEMONSTRATION")
    print("=" * 60)

    # Create risk assessor with default configuration
    event_hub = EventHub()
    risk_assessor = create_risk_assessor(
        min_confidence_threshold=0.6,
        volatility_threshold_high=0.05,
        max_position_concentration=0.3,
        event_hub=event_hub,
    )

    # Get sample data
    signals = create_sample_signals()
    market_data_samples = create_sample_market_data()
    portfolio_context = create_sample_portfolio_context()

    # Assess risk for each signal
    for i, signal in enumerate(signals):
        market_data = market_data_samples[i]

        print(f"\n--- Risk Assessment for {signal.symbol} ---")
        print(f"Signal Type: {signal.signal_type.value}")
        print(f"Signal Strength: {signal.strength.value}")
        print(f"Confidence: {signal.confidence:.3f}")

        # Perform risk assessment
        result = risk_assessor.assess_risk(
            signal=signal,
            market_data=market_data,
            portfolio_context=portfolio_context,
        )

        # Display results
        print(f"\nRisk Assessment Results:")
        print(f"  Overall Risk Level: {result.overall_risk_level.value}")
        print(f"  Risk Multiplier: {result.overall_risk_multiplier:.3f}")
        print(f"  Position Size Adjustment: {result.position_size_adjustment:.3f}")
        print(f"  Assessment Confidence: {result.confidence:.3f}")

        # Show factor-specific results
        print(f"\nRisk Factor Breakdown:")
        for factor_result in result.factor_results:
            print(f"  {factor_result.factor.value}: "
                  f"score={factor_result.risk_score:.3f}, "
                  f"multiplier={factor_result.risk_multiplier:.3f}")

        # Show warnings and recommendations
        if result.warnings:
            print(f"\nWarnings:")
            for warning in result.warnings:
                print(f"  ⚠️  {warning}")

        if result.recommendations:
            print(f"\nRecommendations:")
            for rec in result.recommendations:
                print(f"  💡 {rec}")

        print("-" * 50)


def demonstrate_custom_configuration():
    """Demonstrate risk assessment with custom configuration."""
    print("\n" + "=" * 60)
    print("CUSTOM CONFIGURATION DEMONSTRATION")
    print("=" * 60)

    # Create custom configuration
    custom_config = RiskAssessmentConfig(
        min_confidence_threshold=0.7,  # Higher threshold
        high_confidence_threshold=0.9,
        weak_signal_penalty=0.3,  # Higher penalty
        strong_signal_bonus=0.15,  # Higher bonus
        volatility_threshold_low=0.015,
        volatility_threshold_high=0.04,  # Lower threshold
        volatility_risk_multiplier=2.0,  # Higher multiplier
        max_position_concentration=0.25,  # Stricter limit
        concentration_penalty_factor=0.2,
        min_risk_multiplier=0.2,  # Wider range
        max_risk_multiplier=3.0,
        enable_signal_quality_check=True,
        enable_volatility_check=True,
        enable_concentration_check=True,
        enable_time_context_check=True,
        enable_strategy_check=False,  # Disable strategy check
    )

    # Create risk assessor with custom config
    event_hub = EventHub()
    risk_assessor = RiskAssessor(custom_config, event_hub)

    # Test with high-volatility signal
    signal = TradingSignal(
        symbol="DOGEUSDT",
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        price=0.08,
        timestamp=int(time.time() * 1000),
        strategy_name="volatility_strategy",
        confidence=0.75,
        metadata={"volatility": 0.12},  # Very high volatility
        reasoning="High volatility breakout signal",
    )

    market_data = MarketData(
        symbol="DOGEUSDT",
        timestamp=int(time.time() * 1000),
        price=0.08,
        volume=10000000.0,
        source="binance",
        data_type="ticker",
        metadata={"volatility": 0.12, "bid_ask_spread": 0.0001},
    )

    portfolio_context = {
        "total_value": 50000.0,
        "positions": {"DOGEUSDT": {"value": 15000.0}},  # 30% concentration
        "available_cash": 35000.0,
    }

    print(f"\nAssessing high-volatility signal for {signal.symbol}")
    print(f"Volatility: {market_data.metadata['volatility']:.1%}")
    print(f"Position Concentration: 30%")

    result = risk_assessor.assess_risk(
        signal=signal,
        market_data=market_data,
        portfolio_context=portfolio_context,
    )

    print(f"\nResults with Strict Configuration:")
    print(f"  Risk Level: {result.overall_risk_level.value}")
    print(f"  Risk Multiplier: {result.overall_risk_multiplier:.3f}")
    print(f"  Position Adjustment: {result.position_size_adjustment:.3f}")

    print(f"\nFactor Details:")
    factors = result.get_risk_factors_summary()
    for factor, score in factors.items():
        print(f"  {factor}: {score:.3f}")

    print(f"\nHigh Risk Factors: {result.has_high_risk_factors()}")


def demonstrate_performance_tracking():
    """Demonstrate strategy performance tracking."""
    print("\n" + "=" * 60)
    print("STRATEGY PERFORMANCE TRACKING")
    print("=" * 60)

    risk_assessor = create_risk_assessor()

    # Simulate multiple signals from different strategies
    strategies = [
        ("momentum_strategy", 0.85, SignalStrength.VERY_STRONG),
        ("momentum_strategy", 0.78, SignalStrength.STRONG),
        ("momentum_strategy", 0.82, SignalStrength.STRONG),
        ("experimental_strategy", 0.45, SignalStrength.WEAK),
        ("experimental_strategy", 0.52, SignalStrength.MODERATE),
        ("trend_following", 0.68, SignalStrength.MODERATE),
        ("trend_following", 0.71, SignalStrength.MODERATE),
        ("trend_following", 0.65, SignalStrength.MODERATE),
    ]

    print("Processing signals from different strategies...")

    for strategy_name, confidence, strength in strategies:
        signal = TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=strength,
            price=45000.0,
            timestamp=int(time.time() * 1000),
            strategy_name=strategy_name,
            confidence=confidence,
            reasoning=f"Signal from {strategy_name}",
        )

        # Simple assessment without market data
        result = risk_assessor.assess_risk(signal)

        print(f"  {strategy_name}: confidence={confidence:.3f}, "
              f"risk_level={result.overall_risk_level.value}")

    # Show statistics
    stats = risk_assessor.get_assessment_statistics()
    print(f"\nAssessment Statistics:")
    print(f"  Total Assessments: {stats['assessments_count']}")
    print(f"  Strategies Tracked: {len(stats['strategy_performance'])}")

    for strategy, perf in stats['strategy_performance'].items():
        print(f"  {strategy}: {perf['total_trades']} signals processed")


def demonstrate_event_integration():
    """Demonstrate event hub integration."""
    print("\n" + "=" * 60)
    print("EVENT HUB INTEGRATION")
    print("=" * 60)

    # Create event hub and set up subscriber
    event_hub = EventHub()

    def risk_event_handler(event_data):
        """Handle risk assessment events."""
        result = event_data["result"]
        print(f"📡 Risk Event: {event_data['symbol']} "
              f"-> {event_data['risk_level']} "
              f"(multiplier: {event_data['risk_multiplier']:.3f})")

    # Subscribe to risk events
    event_hub.subscribe("position_size_warning", risk_event_handler)
    event_hub.subscribe("risk_limit_exceeded", risk_event_handler)

    # Create risk assessor with event hub
    risk_assessor = create_risk_assessor(event_hub=event_hub)

    # Test with various risk levels
    test_signals = [
        ("Low Risk", 0.90, SignalStrength.VERY_STRONG, 0.01),
        ("High Risk", 0.40, SignalStrength.WEAK, 0.10),
    ]

    print("Generating risk assessment events...")

    for description, confidence, strength, volatility in test_signals:
        signal = TradingSignal(
            symbol="TESTUSDT",
            signal_type=SignalType.BUY,
            strength=strength,
            price=100.0,
            timestamp=int(time.time() * 1000),
            strategy_name="test_strategy",
            confidence=confidence,
            reasoning=f"{description} test signal",
        )

        market_data = MarketData(
            symbol="TESTUSDT",
            timestamp=int(time.time() * 1000),
            price=100.0,
            volume=1000.0,
            source="test",
            data_type="ticker",
            metadata={"volatility": volatility},
        )

        print(f"\nAssessing {description} signal...")
        result = risk_assessor.assess_risk(signal, market_data)

        # Events are published automatically and handled by our subscriber


def main():
    """Run all demonstrations."""
    print("🔍 Risk Assessment System Demonstration")
    print("========================================")

    try:
        demonstrate_basic_risk_assessment()
        demonstrate_custom_configuration()
        demonstrate_performance_tracking()
        demonstrate_event_integration()

        print("\n" + "=" * 60)
        print("✅ All demonstrations completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())