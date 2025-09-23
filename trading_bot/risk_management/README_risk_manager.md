# RiskManager Orchestrator

The `RiskManager` is the central orchestrator that integrates all risk management components and handles the complete workflow from `TRADING_SIGNAL_GENERATED` to `ORDER_REQUEST_GENERATED` events.

## Overview

The RiskManager serves as the main hub that coordinates all risk management decisions and ensures that every trading signal is properly validated and enriched with all necessary risk parameters before being passed to the execution engine.

## Key Features

### 1. Event-Driven Orchestration
- Subscribes to `TRADING_SIGNAL_GENERATED` events from strategies
- Orchestrates all risk management calculations in sequence
- Publishes `ORDER_REQUEST_GENERATED` events with complete risk parameters

### 2. Component Integration
- **PositionSizer**: Calculates optimal position sizes based on risk parameters
- **RiskAssessor**: Performs signal-level risk assessment and confidence evaluation
- **StopLossCalculator**: Calculates stop-loss and take-profit levels
- **AccountRiskEvaluator**: Validates portfolio-level risk constraints

### 3. Comprehensive Data Flow
```
Trading Signal → Validation → Risk Assessment → Account Evaluation →
Position Sizing → Stop-Loss Calculation → Order Request Generation → Event Publishing
```

### 4. Order Request Structure
The `OrderRequest` dataclass contains:
- Original trading signal information
- Calculated position size and risk amounts
- Entry, stop-loss, and take-profit prices and order types
- Risk metadata including confidence scores and multipliers
- Validation timestamps and audit trail
- Warnings and recommendations for manual review

## Usage

### Basic Usage with Factory Function

```python
from trading_bot.risk_management.risk_manager import create_risk_manager
from trading_bot.core.event_hub import EventHub
from trading_bot.core.config_manager import ConfigManager

# Initialize core components
event_hub = EventHub()
config_manager = ConfigManager()

# Create RiskManager with sensible defaults
risk_manager = create_risk_manager(
    event_hub=event_hub,
    config_manager=config_manager,
    account_balance=50000.0,     # $50K account
    max_position_risk=2.0,       # 2% risk per position
    min_confidence=0.7,          # 70% minimum confidence
    require_stop_loss=True,      # Require stop-loss orders
    min_risk_reward_ratio=2.0,   # Minimum 2:1 risk-reward
)
```

### Processing Signals Manually

```python
# Process individual signals
order_request = await risk_manager.process_trading_signal(
    signal=trading_signal,
    market_data=current_market_data,
    account_state=account_state,
)

if order_request:
    print(f"Order generated: {order_request.get_order_summary()}")
else:
    print("Signal rejected by risk management")
```

### Event-Driven Processing

The RiskManager automatically subscribes to `TRADING_SIGNAL_GENERATED` events and processes them asynchronously:

```python
# RiskManager automatically handles these events
event_hub.publish(EventType.TRADING_SIGNAL_GENERATED, {
    "signal": trading_signal,
    "market_data": market_data,
    "account_state": account_state,
})
```

## Configuration

### RiskManagerConfig Options

```python
from trading_bot.risk_management.risk_manager import RiskManagerConfig, OrderType

config = RiskManagerConfig(
    # Component control
    enable_position_sizing=True,
    enable_risk_assessment=True,
    enable_stop_loss_calculation=True,
    enable_account_risk_evaluation=True,

    # Risk limits
    max_position_risk_percentage=2.0,      # Max 2% risk per position
    max_portfolio_risk_percentage=10.0,    # Max 10% total portfolio risk
    min_confidence_threshold=0.6,          # Min 60% signal confidence
    max_correlation_exposure=0.4,          # Max 40% correlated exposure

    # Order requirements
    default_order_type=OrderType.MARKET,   # Default to market orders
    require_stop_loss=True,                # Require stop-loss orders
    require_take_profit=False,             # Optional take-profit orders
    min_risk_reward_ratio=1.5,             # Min 1.5:1 risk-reward

    # Safety controls
    enable_emergency_stops=True,           # Enable emergency stop mechanisms
    max_daily_trades=50,                   # Max trades per day
    cooldown_after_loss_minutes=30,       # Cooldown after losses

    # Data requirements
    require_market_data=True,              # Require market data for processing
    require_account_state=True,            # Require account state for processing
    max_signal_age_minutes=5,              # Max signal age before rejection
)
```

## Risk Management Workflow

### 1. Signal Validation
- Checks signal age (max 5 minutes by default)
- Validates signal confidence against threshold
- Rejects HOLD signals (no action needed)
- Verifies required data availability

### 2. Risk Assessment
- Analyzes signal quality and strength
- Evaluates market volatility and conditions
- Assesses portfolio concentration risk
- Considers strategy track record

### 3. Account Risk Evaluation
- Checks account balance and margin requirements
- Validates position concentration limits
- Evaluates portfolio correlation exposure
- Assesses leverage and drawdown limits

### 4. Position Sizing
- Calculates optimal position size based on risk parameters
- Applies risk adjustments from assessments
- Enforces account-level position limits
- Considers portfolio balance and diversification

### 5. Stop-Loss Calculation
- Determines appropriate stop-loss levels
- Calculates take-profit targets based on risk-reward ratio
- Validates price levels for logical consistency
- Applies safety limits and buffers

### 6. Order Request Generation
- Combines all components into comprehensive order request
- Includes complete risk metadata and audit trail
- Provides warnings and recommendations
- Validates final order parameters

### 7. Event Publishing
- Publishes `ORDER_REQUEST_GENERATED` event
- Includes order summary and risk metrics
- Provides complete audit trail for compliance

## Error Handling and Safety

### Risk Limit Violations
- Signals are rejected if risk limits are exceeded
- Emergency stops activate during excessive drawdown
- Position limits prevent over-concentration
- Margin requirements are strictly enforced

### Data Quality Checks
- Stale signals are automatically rejected
- Missing required data causes processing failure
- Invalid price levels trigger validation errors
- Component failures are gracefully handled

### Audit Trail
- All decisions are logged with timestamps
- Risk calculations are preserved for review
- Component results are included in metadata
- Warnings and recommendations are tracked

## Monitoring and Statistics

### Risk Statistics
```python
stats = risk_manager.get_risk_statistics()
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Signals processed: {stats['processed_signals']}")
print(f"Orders generated: {stats['generated_orders']}")
print(f"Signals rejected: {stats['rejected_signals']}")
```

### Component Status
The statistics include the status of all risk management components:
- Position Sizer availability
- Risk Assessor functionality
- Stop-Loss Calculator status
- Account Risk Evaluator readiness

## Integration with Execution Engine

The RiskManager generates `ORDER_REQUEST_GENERATED` events that should be consumed by the execution engine:

```python
def handle_order_request(event_data):
    order_request = event_data["order_request"]

    # Execute entry order
    entry_order = execute_market_order(
        symbol=order_request.symbol,
        quantity=order_request.quantity,
        side="buy" if order_request.signal.signal_type == SignalType.BUY else "sell"
    )

    # Set stop-loss if specified
    if order_request.stop_loss_price:
        stop_order = execute_stop_order(
            symbol=order_request.symbol,
            quantity=order_request.quantity,
            stop_price=order_request.stop_loss_price
        )

    # Set take-profit if specified
    if order_request.take_profit_price:
        limit_order = execute_limit_order(
            symbol=order_request.symbol,
            quantity=order_request.quantity,
            limit_price=order_request.take_profit_price
        )

# Subscribe to order requests
event_hub.subscribe(EventType.ORDER_REQUEST_GENERATED, handle_order_request)
```

## Dependencies

The RiskManager requires the following components:
- `EventHub` for event communication
- `ConfigManager` for configuration management
- `PositionSizer` for position size calculations
- `RiskAssessor` for signal risk assessment
- `StopLossCalculator` for stop-loss level calculations
- `AccountRiskEvaluator` for account risk evaluation

All components are created automatically by the factory function with sensible defaults.

## Testing

Comprehensive unit tests are available in `tests/risk_management/test_risk_manager.py`. Run tests with:

```bash
python -m pytest tests/risk_management/test_risk_manager.py -v
```

## Examples

See `dev-files/risk_manager_example.py` for a complete working example demonstrating all RiskManager functionality.