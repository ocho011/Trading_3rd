# Risk Assessment System Usage Guide

## Overview

The Risk Assessment System provides comprehensive risk evaluation for trading signals, analyzing multiple risk factors to determine appropriate position sizing adjustments and risk management recommendations.

## Key Features

- **Multi-Factor Risk Analysis**: Evaluates signal quality, market volatility, position concentration, time context, and strategy performance
- **Configurable Risk Parameters**: Customizable thresholds and penalties for different risk factors
- **Event-Driven Integration**: Publishes risk events to the EventHub for system-wide coordination
- **Actionable Recommendations**: Provides specific recommendations based on risk assessment results
- **Performance Tracking**: Tracks strategy performance over time for better risk evaluation

## Quick Start

### Basic Usage

```python
from trading_bot.risk_management import create_risk_assessor
from trading_bot.strategies.base_strategy import TradingSignal, SignalType, SignalStrength

# Create risk assessor with default configuration
risk_assessor = create_risk_assessor(
    min_confidence_threshold=0.6,
    volatility_threshold_high=0.05,
    max_position_concentration=0.3
)

# Create a trading signal
signal = TradingSignal(
    symbol="BTCUSDT",
    signal_type=SignalType.BUY,
    strength=SignalStrength.STRONG,
    price=45000.0,
    timestamp=int(time.time() * 1000),
    strategy_name="momentum_strategy",
    confidence=0.8,
    reasoning="Strong momentum breakout"
)

# Assess risk
result = risk_assessor.assess_risk(signal)

print(f"Risk Level: {result.overall_risk_level.value}")
print(f"Risk Multiplier: {result.overall_risk_multiplier:.3f}")
print(f"Position Adjustment: {result.position_size_adjustment:.3f}")
```

### Advanced Usage with Market Data and Portfolio Context

```python
from trading_bot.market_data.data_processor import MarketData

# Include market data for volatility assessment
market_data = MarketData(
    symbol="BTCUSDT",
    timestamp=int(time.time() * 1000),
    price=45000.0,
    volume=1500.0,
    source="binance",
    data_type="ticker",
    metadata={"volatility": 0.03}  # 3% volatility
)

# Include portfolio context for concentration assessment
portfolio_context = {
    "total_value": 100000.0,
    "positions": {
        "BTCUSDT": {"value": 20000.0}  # 20% allocation
    },
    "available_cash": 80000.0
}

# Comprehensive risk assessment
result = risk_assessor.assess_risk(
    signal=signal,
    market_data=market_data,
    portfolio_context=portfolio_context
)

# Review detailed factor results
for factor_result in result.factor_results:
    print(f"{factor_result.factor.value}: score={factor_result.risk_score:.3f}")

# Check recommendations
for recommendation in result.recommendations:
    print(f"💡 {recommendation}")
```

## Risk Factors Explained

### 1. Signal Quality
- **Evaluates**: Signal confidence and strength
- **Impact**: Low confidence or weak signals increase risk
- **Adjustments**: Penalties for weak signals, bonuses for very strong signals

### 2. Market Volatility
- **Evaluates**: Market volatility from metadata
- **Impact**: High volatility increases risk multiplier
- **Thresholds**: Low (<2%), Moderate (2-5%), High (>5%)

### 3. Position Concentration
- **Evaluates**: Current position size relative to portfolio
- **Impact**: High concentration increases risk
- **Default Limit**: 30% maximum concentration

### 4. Time Context
- **Evaluates**: Signal age and market timing
- **Impact**: Stale signals increase risk
- **Considerations**: Signal freshness, market hours (future enhancement)

### 5. Strategy Track Record
- **Evaluates**: Historical strategy performance
- **Impact**: Poor performing strategies increase risk
- **Tracking**: Win rate and trade count

## Configuration Options

### Custom Configuration

```python
from trading_bot.risk_management import RiskAssessmentConfig, RiskAssessor

config = RiskAssessmentConfig(
    # Signal quality settings
    min_confidence_threshold=0.7,        # Higher minimum confidence
    high_confidence_threshold=0.9,       # Higher threshold for bonus
    weak_signal_penalty=0.3,             # Increased penalty for weak signals
    strong_signal_bonus=0.15,            # Increased bonus for strong signals

    # Volatility settings
    volatility_threshold_low=0.015,      # 1.5% low threshold
    volatility_threshold_high=0.04,      # 4% high threshold
    volatility_risk_multiplier=2.0,      # Higher volatility impact

    # Concentration settings
    max_position_concentration=0.25,     # Stricter 25% limit
    concentration_penalty_factor=0.2,    # Higher concentration penalty

    # Risk adjustment limits
    min_risk_multiplier=0.2,             # Wider range
    max_risk_multiplier=3.0,

    # Enable/disable specific factors
    enable_signal_quality_check=True,
    enable_volatility_check=True,
    enable_concentration_check=True,
    enable_time_context_check=True,
    enable_strategy_check=False,         # Disable strategy tracking
)

risk_assessor = RiskAssessor(config)
```

## Risk Levels and Multipliers

| Risk Level | Multiplier | Recommendation |
|------------|------------|----------------|
| Very Low   | 0.5x       | Consider increasing position size |
| Low        | 0.75x      | Normal position sizing |
| Moderate   | 1.0x       | Standard risk management |
| High       | 1.25x      | Reduce position size, tighter stops |
| Very High  | 1.5x       | Minimal position or avoid trade |

## Event Integration

```python
from trading_bot.core.event_hub import EventHub

# Set up event handling
event_hub = EventHub()

def handle_risk_event(event_data):
    result = event_data["result"]
    if result.overall_risk_level in ["high", "very_high"]:
        print(f"⚠️ High risk detected for {event_data['symbol']}")

# Subscribe to risk events
event_hub.subscribe("position_size_warning", handle_risk_event)
event_hub.subscribe("risk_limit_exceeded", handle_risk_event)

# Create risk assessor with event publishing
risk_assessor = create_risk_assessor(event_hub=event_hub)
```

## Integration with Position Sizing

```python
from trading_bot.risk_management import create_position_sizer, PositionSizingMethod

# Create position sizer
position_sizer = create_position_sizer(
    account_balance=100000.0,
    risk_percentage=2.0,
    method=PositionSizingMethod.FIXED_PERCENTAGE
)

# Assess risk first
risk_result = risk_assessor.assess_risk(signal, market_data, portfolio_context)

# Calculate base position size
base_position = position_sizer.calculate_position_size(
    entry_price=signal.price,
    stop_loss_price=signal.stop_loss
)

# Apply risk adjustment
adjusted_position_size = base_position.position_size * risk_result.position_size_adjustment

print(f"Base Position: {base_position.position_size:.6f}")
print(f"Risk Adjustment: {risk_result.position_size_adjustment:.3f}")
print(f"Adjusted Position: {adjusted_position_size:.6f}")
```

## Best Practices

1. **Always Include Market Data**: Volatility assessment significantly improves risk evaluation
2. **Track Portfolio Context**: Concentration risk is crucial for diversified portfolios
3. **Monitor Strategy Performance**: Enable strategy tracking for better long-term assessment
4. **Use Event Integration**: Subscribe to risk events for system-wide risk management
5. **Customize Thresholds**: Adjust configuration based on your risk tolerance and trading style
6. **Review Recommendations**: Use the generated recommendations for better risk management
7. **Combine with Position Sizing**: Use risk assessments to adjust position sizes dynamically

## Error Handling

```python
from trading_bot.risk_management import RiskCalculationError, InvalidRiskConfigError

try:
    result = risk_assessor.assess_risk(signal, market_data, portfolio_context)
except RiskCalculationError as e:
    print(f"Risk calculation failed: {e}")
except InvalidRiskConfigError as e:
    print(f"Invalid configuration: {e}")
```

## Performance Monitoring

```python
# Get assessment statistics
stats = risk_assessor.get_assessment_statistics()
print(f"Total Assessments: {stats['assessments_count']}")
print(f"Strategies Tracked: {len(stats['strategy_performance'])}")

# Review strategy performance
for strategy, performance in stats['strategy_performance'].items():
    print(f"{strategy}: {performance['total_trades']} trades, "
          f"win rate: {performance['win_rate']:.2%}")
```

This comprehensive risk assessment system provides the foundation for sophisticated risk management in your trading bot, enabling dynamic position sizing and intelligent risk-based decision making.