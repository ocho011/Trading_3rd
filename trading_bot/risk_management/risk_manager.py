"""
Risk Manager orchestrator for comprehensive trading risk management.

This module serves as the central orchestrator that integrates all risk management
components and handles the complete workflow from TRADING_SIGNAL_GENERATED to
ORDER_REQUEST_GENERATED events. Follows SOLID principles and provides a unified
interface for all risk management operations.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from trading_bot.core.config_manager import ConfigManager
from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.core.logger import get_module_logger
from trading_bot.market_data.data_processor import MarketData
from trading_bot.risk_management.account_risk_evaluator import (
    AccountRiskEvaluator,
    AccountState,
    create_account_risk_evaluator,
)
from trading_bot.risk_management.position_sizer import (
    PositionSizer,
    PositionSizingResult,
    create_position_sizer,
)
from trading_bot.risk_management.risk_assessor import (
    RiskAssessmentResult,
    RiskAssessor,
    create_risk_assessor,
)
from trading_bot.risk_management.stop_loss_calculator import (
    PositionType,
    StopLossCalculator,
    StopLossResult,
    create_stop_loss_calculator,
)
from trading_bot.strategies.base_strategy import SignalType, TradingSignal


class RiskManagerError(Exception):
    """Base exception for risk manager errors."""

    pass


class OrderGenerationError(RiskManagerError):
    """Exception raised for order generation errors."""

    pass


class InsufficientRiskDataError(RiskManagerError):
    """Exception raised when insufficient risk data is available."""

    pass


class RiskLimitExceededError(RiskManagerError):
    """Exception raised when risk limits are exceeded."""

    pass


class OrderType(Enum):
    """Order types for order requests."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class OrderRequest:
    """Comprehensive order request with risk parameters.

    Contains all information needed for order execution including
    original signal, calculated risk parameters, and validation metadata.
    """

    # Original signal information
    signal: TradingSignal
    symbol: str
    order_type: OrderType
    quantity: Decimal
    price: float

    # Risk management parameters
    position_size_result: PositionSizingResult
    risk_assessment_result: RiskAssessmentResult
    stop_loss_result: Optional[StopLossResult]
    account_risk_result: Optional[Any]

    # Order execution details
    entry_price: float
    entry_order_type: OrderType = OrderType.MARKET
    stop_loss_price: Optional[float] = None
    stop_loss_order_type: Optional[OrderType] = None
    take_profit_price: Optional[float] = None
    take_profit_order_type: Optional[OrderType] = None

    # Risk metadata
    total_risk_amount: Decimal = Decimal("0")
    risk_percentage: float = 0.0
    confidence_score: float = 0.0
    risk_multiplier: float = 1.0

    # Validation and audit
    validation_timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    validation_checks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate order request after initialization."""
        if self.quantity <= 0:
            raise OrderGenerationError("Order quantity must be positive")
        if self.price <= 0:
            raise OrderGenerationError("Order price must be positive")
        if not 0.0 <= self.confidence_score <= 1.0:
            raise OrderGenerationError("Confidence score must be between 0.0 and 1.0")

    def get_order_summary(self) -> Dict[str, Any]:
        """Get summary of order request for logging and monitoring."""
        return {
            "symbol": self.symbol,
            "signal_type": self.signal.signal_type.value,
            "quantity": float(self.quantity),
            "price": self.price,
            "risk_amount": float(self.total_risk_amount),
            "risk_percentage": self.risk_percentage,
            "confidence": self.confidence_score,
            "stop_loss": self.stop_loss_price,
            "take_profit": self.take_profit_price,
            "warnings_count": len(self.warnings),
        }

    def has_stop_loss(self) -> bool:
        """Check if order has stop-loss configured."""
        return self.stop_loss_price is not None

    def has_take_profit(self) -> bool:
        """Check if order has take-profit configured."""
        return self.take_profit_price is not None

    def calculate_risk_reward_ratio(self) -> Optional[float]:
        """Calculate risk-reward ratio if both levels are set."""
        if not (self.stop_loss_price and self.take_profit_price):
            return None

        risk_distance = abs(self.price - self.stop_loss_price)
        reward_distance = abs(self.take_profit_price - self.price)

        if risk_distance <= 0:
            return None

        return reward_distance / risk_distance


@dataclass
class RiskManagerConfig:
    """Configuration for risk manager operations."""

    # Component settings
    enable_position_sizing: bool = True
    enable_risk_assessment: bool = True
    enable_stop_loss_calculation: bool = True
    enable_account_risk_evaluation: bool = True

    # Risk limits
    max_position_risk_percentage: float = 2.0
    max_portfolio_risk_percentage: float = 10.0
    min_confidence_threshold: float = 0.6
    max_correlation_exposure: float = 0.4

    # Order generation settings
    default_order_type: OrderType = OrderType.MARKET
    require_stop_loss: bool = True
    require_take_profit: bool = False
    min_risk_reward_ratio: float = 1.5

    # Emergency controls
    enable_emergency_stops: bool = True
    max_daily_trades: int = 50
    cooldown_after_loss_minutes: int = 30

    # Data requirements
    require_market_data: bool = True
    require_account_state: bool = True
    max_signal_age_minutes: int = 5

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not 0.1 <= self.max_position_risk_percentage <= 50.0:
            raise RiskManagerError("Position risk percentage must be 0.1-50%")
        if not 0.1 <= self.max_portfolio_risk_percentage <= 100.0:
            raise RiskManagerError("Portfolio risk percentage must be 0.1-100%")
        if not 0.0 <= self.min_confidence_threshold <= 1.0:
            raise RiskManagerError("Confidence threshold must be 0.0-1.0")
        if self.min_risk_reward_ratio < 0.5:
            raise RiskManagerError("Min risk-reward ratio must be >= 0.5")


class IRiskManager(ABC):
    """Abstract interface for risk manager implementations."""

    @abstractmethod
    async def process_trading_signal(
        self,
        signal: TradingSignal,
        market_data: Optional[MarketData] = None,
        account_state: Optional[AccountState] = None,
    ) -> Optional[OrderRequest]:
        """Process trading signal and generate order request.

        Args:
            signal: Trading signal to process
            market_data: Optional current market data
            account_state: Optional current account state

        Returns:
            OrderRequest: Generated order request or None if rejected

        Raises:
            RiskManagerError: If processing fails
        """
        pass

    @abstractmethod
    def update_config(self, config: RiskManagerConfig) -> None:
        """Update risk manager configuration.

        Args:
            config: New risk manager configuration

        Raises:
            RiskManagerError: If configuration is invalid
        """
        pass

    @abstractmethod
    def get_risk_statistics(self) -> Dict[str, Any]:
        """Get risk management statistics.

        Returns:
            Dictionary containing risk management statistics
        """
        pass


class RiskManager(IRiskManager):
    """Comprehensive risk management orchestrator.

    This class serves as the central hub that coordinates all risk management
    decisions and ensures every trading signal is properly validated and
    enriched with necessary risk parameters before order generation.

    Attributes:
        _config: Risk manager configuration
        _event_hub: Event hub for publishing events
        _config_manager: Configuration manager for settings
        _position_sizer: Position sizing component
        _risk_assessor: Risk assessment component
        _stop_loss_calculator: Stop-loss calculation component
        _account_risk_evaluator: Account risk evaluation component
        _logger: Logger instance for risk manager logging
        _processed_signals_count: Counter for processed signals
        _generated_orders_count: Counter for generated orders
        _rejected_signals_count: Counter for rejected signals
    """

    def __init__(
        self,
        config: RiskManagerConfig,
        event_hub: EventHub,
        config_manager: ConfigManager,
        position_sizer: Optional[PositionSizer] = None,
        risk_assessor: Optional[RiskAssessor] = None,
        stop_loss_calculator: Optional[StopLossCalculator] = None,
        account_risk_evaluator: Optional[AccountRiskEvaluator] = None,
    ) -> None:
        """Initialize risk manager with configuration and dependencies.

        Args:
            config: Risk manager configuration
            event_hub: Event hub for publishing events
            config_manager: Configuration manager for settings
            position_sizer: Optional position sizer component
            risk_assessor: Optional risk assessor component
            stop_loss_calculator: Optional stop-loss calculator component
            account_risk_evaluator: Optional account risk evaluator component

        Raises:
            RiskManagerError: If initialization fails
        """
        if not isinstance(config, RiskManagerConfig):
            raise RiskManagerError("Config must be RiskManagerConfig instance")

        self._config = config
        self._event_hub = event_hub
        self._config_manager = config_manager
        self._logger = get_module_logger("risk_manager")

        # Initialize components (dependency injection)
        self._position_sizer = position_sizer
        self._risk_assessor = risk_assessor
        self._stop_loss_calculator = stop_loss_calculator
        self._account_risk_evaluator = account_risk_evaluator

        # Statistics tracking
        self._processed_signals_count = 0
        self._generated_orders_count = 0
        self._rejected_signals_count = 0
        self._risk_history: List[Dict[str, Any]] = []

        # Subscribe to trading signals
        self._event_hub.subscribe(
            EventType.TRADING_SIGNAL_GENERATED, self._handle_trading_signal
        )

        self._logger.info("Initialized risk manager with comprehensive orchestration")

    async def process_trading_signal(
        self,
        signal: TradingSignal,
        market_data: Optional[MarketData] = None,
        account_state: Optional[AccountState] = None,
    ) -> Optional[OrderRequest]:
        """Process trading signal through complete risk management workflow.

        Args:
            signal: Trading signal to process
            market_data: Optional current market data
            account_state: Optional current account state

        Returns:
            OrderRequest: Generated order request or None if rejected

        Raises:
            RiskManagerError: If processing fails
        """
        try:
            self._processed_signals_count += 1
            start_time = time.time()

            self._logger.info(
                f"Processing trading signal: {signal.symbol} "
                f"{signal.signal_type.value} @ {signal.price}"
            )

            # Step 1: Validate signal and inputs
            validation_result = await self._validate_signal_inputs(
                signal, market_data, account_state
            )
            if not validation_result["valid"]:
                self._rejected_signals_count += 1
                self._logger.warning(
                    f"Signal validation failed: {validation_result['reason']}"
                )
                return None

            # Step 2: Assess signal-level risk
            risk_assessment = await self._assess_signal_risk(
                signal, market_data, account_state
            )

            # Step 3: Evaluate account/portfolio constraints
            account_risk_result = await self._evaluate_account_risk(
                signal, account_state, risk_assessment
            )

            # Step 4: Calculate position size with risk adjustments
            position_sizing_result = await self._calculate_position_size(
                signal, risk_assessment, account_risk_result, account_state
            )

            # Step 5: Calculate stop-loss and take-profit levels
            stop_loss_result = await self._calculate_stop_loss_levels(
                signal, position_sizing_result, market_data
            )

            # Step 6: Create comprehensive order request
            order_request = await self._create_order_request(
                signal,
                position_sizing_result,
                risk_assessment,
                stop_loss_result,
                account_risk_result,
            )

            # Step 7: Final validation and safety checks
            final_validation = await self._perform_final_validation(order_request)
            if not final_validation["valid"]:
                self._rejected_signals_count += 1
                self._logger.warning(
                    f"Final validation failed: {final_validation['reason']}"
                )
                return None

            # Step 8: Publish ORDER_REQUEST_GENERATED event
            await self._publish_order_request_event(order_request)

            # Update statistics
            self._generated_orders_count += 1
            processing_time = time.time() - start_time

            self._logger.info(
                f"Order request generated successfully: {order_request.symbol} "
                f"qty={order_request.quantity} risk=${order_request.total_risk_amount} "
                f"processing_time={processing_time:.3f}s"
            )

            return order_request

        except Exception as e:
            self._rejected_signals_count += 1
            self._logger.error(f"Error processing trading signal: {e}")
            raise RiskManagerError(f"Signal processing failed: {e}")

    def update_config(self, config: RiskManagerConfig) -> None:
        """Update risk manager configuration.

        Args:
            config: New risk manager configuration

        Raises:
            RiskManagerError: If configuration is invalid
        """
        if not isinstance(config, RiskManagerConfig):
            raise RiskManagerError("Config must be RiskManagerConfig instance")

        self._config = config
        self._logger.info("Updated risk manager configuration")

    def get_risk_statistics(self) -> Dict[str, Any]:
        """Get comprehensive risk management statistics.

        Returns:
            Dictionary containing risk management statistics
        """
        return {
            "processed_signals": self._processed_signals_count,
            "generated_orders": self._generated_orders_count,
            "rejected_signals": self._rejected_signals_count,
            "success_rate": (
                self._generated_orders_count / max(1, self._processed_signals_count)
            ),
            "components_status": {
                "position_sizer": self._position_sizer is not None,
                "risk_assessor": self._risk_assessor is not None,
                "stop_loss_calculator": self._stop_loss_calculator is not None,
                "account_risk_evaluator": self._account_risk_evaluator is not None,
            },
            "config_summary": {
                "max_position_risk": self._config.max_position_risk_percentage,
                "min_confidence": self._config.min_confidence_threshold,
                "require_stop_loss": self._config.require_stop_loss,
            },
        }

    async def _handle_trading_signal(self, event_data: Dict[str, Any]) -> None:
        """Handle TRADING_SIGNAL_GENERATED event.

        Args:
            event_data: Event data containing trading signal
        """
        try:
            signal = event_data.get("signal")
            market_data = event_data.get("market_data")
            account_state = event_data.get("account_state")

            if not signal:
                self._logger.warning("Received signal event without signal data")
                return

            # Process signal asynchronously
            order_request = await self.process_trading_signal(
                signal, market_data, account_state
            )

            if order_request:
                self._logger.debug(
                    f"Successfully processed signal event: {signal.symbol}"
                )
            else:
                self._logger.debug(
                    f"Signal rejected during processing: {signal.symbol}"
                )

        except Exception as e:
            self._logger.error(f"Error handling trading signal event: {e}")

    async def _validate_signal_inputs(
        self,
        signal: TradingSignal,
        market_data: Optional[MarketData],
        account_state: Optional[AccountState],
    ) -> Dict[str, Any]:
        """Validate signal and input data before processing.

        Args:
            signal: Trading signal to validate
            market_data: Optional market data
            account_state: Optional account state

        Returns:
            Dictionary with validation result
        """
        # Check signal age
        current_time = int(time.time() * 1000)
        signal_age_minutes = (current_time - signal.timestamp) / (1000 * 60)

        if signal_age_minutes > self._config.max_signal_age_minutes:
            return {
                "valid": False,
                "reason": f"Signal too old: {signal_age_minutes:.1f} minutes",
            }

        # Check signal confidence
        if signal.confidence < self._config.min_confidence_threshold:
            return {
                "valid": False,
                "reason": f"Signal confidence too low: {signal.confidence:.3f}",
            }

        # Check data requirements
        if self._config.require_market_data and not market_data:
            return {"valid": False, "reason": "Market data required but not provided"}

        if self._config.require_account_state and not account_state:
            return {"valid": False, "reason": "Account state required but not provided"}

        # Check signal type
        if signal.signal_type == SignalType.HOLD:
            return {"valid": False, "reason": "HOLD signals do not generate orders"}

        return {"valid": True, "reason": "Validation passed"}

    async def _assess_signal_risk(
        self,
        signal: TradingSignal,
        market_data: Optional[MarketData],
        account_state: Optional[AccountState],
    ) -> Optional[RiskAssessmentResult]:
        """Assess signal-level risk using risk assessor.

        Args:
            signal: Trading signal to assess
            market_data: Optional market data
            account_state: Optional account state

        Returns:
            RiskAssessmentResult or None if assessment fails
        """
        if not self._config.enable_risk_assessment or not self._risk_assessor:
            self._logger.debug("Risk assessment disabled or unavailable")
            return None

        try:
            # Build portfolio context from account state
            portfolio_context = None
            if account_state:
                portfolio_context = {
                    "positions": {
                        symbol: {
                            "value": float(pos.market_value),
                            "quantity": float(pos.quantity),
                        }
                        for symbol, pos in account_state.positions.items()
                    },
                    "total_value": float(account_state.total_portfolio_value),
                    "available_cash": float(account_state.available_cash),
                }

            result = self._risk_assessor.assess_risk(
                signal, market_data, portfolio_context
            )

            self._logger.debug(
                f"Risk assessment completed: {signal.symbol} "
                f"level={result.overall_risk_level.value}"
            )

            return result

        except Exception as e:
            self._logger.error(f"Risk assessment failed: {e}")
            return None

    async def _evaluate_account_risk(
        self,
        signal: TradingSignal,
        account_state: Optional[AccountState],
        risk_assessment: Optional[RiskAssessmentResult],
    ) -> Optional[Any]:
        """Evaluate account-level risk constraints.

        Args:
            signal: Trading signal
            account_state: Optional account state
            risk_assessment: Optional risk assessment result

        Returns:
            Account risk result or None if evaluation fails
        """
        if (
            not self._config.enable_account_risk_evaluation
            or not self._account_risk_evaluator
            or not account_state
        ):
            self._logger.debug("Account risk evaluation disabled or unavailable")
            return None

        try:
            result = self._account_risk_evaluator.evaluate_new_position(
                signal, account_state
            )

            if not result.can_add_position:
                raise RiskLimitExceededError(
                    f"Account risk limits prevent new position: {result.risk_level.value}"
                )

            self._logger.debug(
                f"Account risk evaluation passed: {signal.symbol} "
                f"level={result.risk_level.value}"
            )

            return result

        except RiskLimitExceededError:
            raise
        except Exception as e:
            self._logger.error(f"Account risk evaluation failed: {e}")
            return None

    async def _calculate_position_size(
        self,
        signal: TradingSignal,
        risk_assessment: Optional[RiskAssessmentResult],
        account_risk_result: Optional[Any],
        account_state: Optional[AccountState],
    ) -> Optional[PositionSizingResult]:
        """Calculate position size with risk adjustments.

        Args:
            signal: Trading signal
            risk_assessment: Optional risk assessment result
            account_risk_result: Optional account risk result
            account_state: Optional account state

        Returns:
            PositionSizingResult or None if calculation fails
        """
        if not self._config.enable_position_sizing or not self._position_sizer:
            self._logger.debug("Position sizing disabled or unavailable")
            return None

        try:
            # Use stop loss from signal if available
            stop_loss_price = signal.stop_loss

            # Apply risk adjustments from assessments
            if risk_assessment:
                # Adjust position size based on risk level
                position_adjustment = risk_assessment.position_size_adjustment
                self._logger.debug(
                    f"Applying risk adjustment: {position_adjustment:.3f}"
                )

            result = self._position_sizer.calculate_position_size(
                signal.price, stop_loss_price
            )

            # Apply account-level constraints
            if account_risk_result and hasattr(
                account_risk_result, "max_new_position_value"
            ):
                max_value = float(account_risk_result.max_new_position_value)
                current_value = float(result.position_size * signal.price)

                if current_value > max_value:
                    # Reduce position size to fit account constraints
                    adjusted_size = max_value / signal.price
                    result.position_size = adjusted_size
                    result.warnings.append(
                        "Position size reduced due to account limits"
                    )

            self._logger.debug(
                f"Position size calculated: {signal.symbol} "
                f"size={result.position_size:.6f}"
            )

            return result

        except Exception as e:
            self._logger.error(f"Position size calculation failed: {e}")
            return None

    async def _calculate_stop_loss_levels(
        self,
        signal: TradingSignal,
        position_sizing_result: Optional[PositionSizingResult],
        market_data: Optional[MarketData],
    ) -> Optional[StopLossResult]:
        """Calculate stop-loss and take-profit levels.

        Args:
            signal: Trading signal
            position_sizing_result: Position sizing result
            market_data: Optional market data

        Returns:
            StopLossResult or None if calculation fails
        """
        if (
            not self._config.enable_stop_loss_calculation
            or not self._stop_loss_calculator
        ):
            self._logger.debug("Stop-loss calculation disabled or unavailable")
            return None

        try:
            # Determine position type from signal
            position_type = (
                PositionType.LONG
                if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]
                else PositionType.SHORT
            )

            # Prepare market data for calculation
            market_data_dict = None
            if market_data:
                market_data_dict = {
                    "atr": market_data.metadata.get("atr"),
                    "volatility": market_data.metadata.get("volatility"),
                    "support": market_data.metadata.get("support"),
                    "resistance": market_data.metadata.get("resistance"),
                }

            result = self._stop_loss_calculator.calculate_levels(
                signal.price, position_type, market_data_dict
            )

            self._logger.debug(
                f"Stop-loss levels calculated: {signal.symbol} "
                f"stop={result.stop_loss_level.price if result.stop_loss_level else 'None'} "
                f"target={result.take_profit_level.price if result.take_profit_level else 'None'}"
            )

            return result

        except Exception as e:
            self._logger.error(f"Stop-loss calculation failed: {e}")
            return None

    async def _create_order_request(
        self,
        signal: TradingSignal,
        position_sizing_result: Optional[PositionSizingResult],
        risk_assessment: Optional[RiskAssessmentResult],
        stop_loss_result: Optional[StopLossResult],
        account_risk_result: Optional[Any],
    ) -> OrderRequest:
        """Create comprehensive order request with all risk parameters.

        Args:
            signal: Original trading signal
            position_sizing_result: Position sizing result
            risk_assessment: Risk assessment result
            stop_loss_result: Stop-loss calculation result
            account_risk_result: Account risk evaluation result

        Returns:
            OrderRequest: Comprehensive order request
        """
        # Determine position size (fallback to default if not calculated)
        quantity = Decimal("0.01")  # Default minimal position
        risk_amount = Decimal("0")
        risk_percentage = 0.0

        if position_sizing_result:
            quantity = Decimal(str(position_sizing_result.position_size))
            risk_amount = Decimal(str(position_sizing_result.risk_amount))
            risk_percentage = position_sizing_result.risk_percentage

        # Determine stop-loss and take-profit levels
        stop_loss_price = None
        take_profit_price = None

        if stop_loss_result:
            if stop_loss_result.stop_loss_level:
                stop_loss_price = stop_loss_result.stop_loss_level.price
            if stop_loss_result.take_profit_level:
                take_profit_price = stop_loss_result.take_profit_level.price

        # Calculate confidence score
        confidence_score = signal.confidence
        if risk_assessment:
            confidence_score = min(confidence_score, risk_assessment.confidence)

        # Calculate risk multiplier
        risk_multiplier = 1.0
        if risk_assessment:
            risk_multiplier = risk_assessment.overall_risk_multiplier

        # Collect warnings and recommendations
        warnings = []
        recommendations = []

        if position_sizing_result:
            warnings.extend(position_sizing_result.warnings)

        if risk_assessment:
            warnings.extend(risk_assessment.warnings)
            recommendations.extend(risk_assessment.recommendations)

        if stop_loss_result:
            warnings.extend(stop_loss_result.warnings)
            recommendations.extend(stop_loss_result.recommendations)

        # Add validation checks
        validation_checks = ["signal_validated", "risk_assessed"]

        if position_sizing_result:
            validation_checks.append("position_sized")
        if stop_loss_result:
            validation_checks.append("stop_loss_calculated")
        if account_risk_result:
            validation_checks.append("account_risk_evaluated")

        # Create order request
        order_request = OrderRequest(
            signal=signal,
            symbol=signal.symbol,
            order_type=self._config.default_order_type,
            quantity=quantity,
            price=signal.price,
            position_size_result=position_sizing_result,
            risk_assessment_result=risk_assessment,
            stop_loss_result=stop_loss_result,
            account_risk_result=account_risk_result,
            entry_price=signal.price,
            entry_order_type=self._config.default_order_type,
            stop_loss_price=stop_loss_price,
            stop_loss_order_type=OrderType.STOP if stop_loss_price else None,
            take_profit_price=take_profit_price,
            take_profit_order_type=OrderType.LIMIT if take_profit_price else None,
            total_risk_amount=risk_amount,
            risk_percentage=risk_percentage,
            confidence_score=confidence_score,
            risk_multiplier=risk_multiplier,
            validation_checks=validation_checks,
            warnings=warnings,
            recommendations=recommendations,
            metadata={
                "processing_timestamp": int(time.time() * 1000),
                "signal_strength": signal.strength.value,
                "risk_components_used": {
                    "position_sizer": position_sizing_result is not None,
                    "risk_assessor": risk_assessment is not None,
                    "stop_loss_calculator": stop_loss_result is not None,
                    "account_risk_evaluator": account_risk_result is not None,
                },
            },
        )

        return order_request

    async def _perform_final_validation(
        self, order_request: OrderRequest
    ) -> Dict[str, Any]:
        """Perform final validation and safety checks on order request.

        Args:
            order_request: Order request to validate

        Returns:
            Dictionary with validation result
        """
        # Check minimum confidence requirements
        if order_request.confidence_score < self._config.min_confidence_threshold:
            return {
                "valid": False,
                "reason": f"Final confidence too low: {order_request.confidence_score:.3f}",
            }

        # Check stop-loss requirements
        if self._config.require_stop_loss and not order_request.has_stop_loss():
            return {"valid": False, "reason": "Stop-loss required but not set"}

        # Check take-profit requirements
        if self._config.require_take_profit and not order_request.has_take_profit():
            return {"valid": False, "reason": "Take-profit required but not set"}

        # Check risk-reward ratio
        risk_reward = order_request.calculate_risk_reward_ratio()
        if risk_reward and risk_reward < self._config.min_risk_reward_ratio:
            return {
                "valid": False,
                "reason": f"Risk-reward ratio too low: {risk_reward:.2f}",
            }

        # Check position size limits
        if order_request.quantity <= 0:
            return {"valid": False, "reason": "Invalid position size"}

        # Check emergency stops
        if self._config.enable_emergency_stops:
            # Check daily trade limits
            if self._generated_orders_count >= self._config.max_daily_trades:
                return {"valid": False, "reason": "Daily trade limit exceeded"}

        return {"valid": True, "reason": "Final validation passed"}

    async def _publish_order_request_event(self, order_request: OrderRequest) -> None:
        """Publish ORDER_REQUEST_GENERATED event.

        Args:
            order_request: Order request to publish
        """
        try:
            event_data = {
                "order_request": order_request,
                "symbol": order_request.symbol,
                "order_type": order_request.order_type.value,
                "quantity": float(order_request.quantity),
                "price": order_request.price,
                "risk_amount": float(order_request.total_risk_amount),
                "confidence": order_request.confidence_score,
                "timestamp": order_request.validation_timestamp,
                "summary": order_request.get_order_summary(),
            }

            self._event_hub.publish(EventType.ORDER_REQUEST_GENERATED, event_data)

            self._logger.debug(
                f"Published ORDER_REQUEST_GENERATED event: {order_request.symbol}"
            )

        except Exception as e:
            self._logger.error(f"Error publishing order request event: {e}")


def create_risk_manager(
    event_hub: EventHub,
    config_manager: ConfigManager,
    account_balance: float = 10000.0,
    max_position_risk: float = 2.0,
    min_confidence: float = 0.6,
    **kwargs: Any,
) -> RiskManager:
    """Factory function to create RiskManager with default components.

    Args:
        event_hub: Event hub for publishing events
        config_manager: Configuration manager for settings
        account_balance: Account balance for position sizing
        max_position_risk: Maximum position risk percentage
        min_confidence: Minimum signal confidence threshold
        **kwargs: Additional configuration parameters

    Returns:
        RiskManager: Configured risk manager instance

    Raises:
        RiskManagerError: If creation fails
    """
    try:
        # Create risk manager configuration
        risk_config = RiskManagerConfig(
            max_position_risk_percentage=max_position_risk,
            min_confidence_threshold=min_confidence,
            require_stop_loss=kwargs.get("require_stop_loss", True),
            require_take_profit=kwargs.get("require_take_profit", False),
            min_risk_reward_ratio=kwargs.get("min_risk_reward_ratio", 1.5),
            enable_emergency_stops=kwargs.get("enable_emergency_stops", True),
            max_daily_trades=kwargs.get("max_daily_trades", 50),
        )

        # Create position sizer
        position_sizer = create_position_sizer(
            account_balance=account_balance,
            risk_percentage=max_position_risk,
            event_hub=event_hub,
        )

        # Create risk assessor
        risk_assessor = create_risk_assessor(
            min_confidence_threshold=min_confidence,
            event_hub=event_hub,
        )

        # Create stop-loss calculator
        stop_loss_calculator = create_stop_loss_calculator(
            stop_loss_percentage=max_position_risk,
            event_hub=event_hub,
        )

        # Create account risk evaluator
        account_risk_evaluator = create_account_risk_evaluator(
            event_hub=event_hub,
        )

        # Create risk manager
        risk_manager = RiskManager(
            config=risk_config,
            event_hub=event_hub,
            config_manager=config_manager,
            position_sizer=position_sizer,
            risk_assessor=risk_assessor,
            stop_loss_calculator=stop_loss_calculator,
            account_risk_evaluator=account_risk_evaluator,
        )

        return risk_manager

    except Exception as e:
        raise RiskManagerError(f"Failed to create risk manager: {e}")
