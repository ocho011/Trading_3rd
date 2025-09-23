"""
Position sizing module for calculating optimal position sizes based on risk management rules.

This module provides position sizing algorithms that calculate the appropriate position size
based on account balance, risk percentage, and various risk management methodologies.
Follows SOLID principles and integrates with the event-driven architecture.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.core.logger import get_module_logger


class PositionSizingError(Exception):
    """Base exception for position sizing related errors."""

    pass


class InvalidPositionSizingConfigError(PositionSizingError):
    """Exception raised for invalid position sizing configuration."""

    pass


class PositionSizingCalculationError(PositionSizingError):
    """Exception raised for position sizing calculation errors."""

    pass


class PositionSizingMethod(Enum):
    """Position sizing calculation methods."""

    FIXED_PERCENTAGE = "fixed_percentage"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    EQUAL_RISK = "equal_risk"


@dataclass
class PositionSizingConfig:
    """Configuration data structure for position sizing calculations.

    This dataclass defines the parameters required for position sizing
    calculations across different methodologies.
    """

    account_balance: float
    risk_percentage: float
    method: PositionSizingMethod = PositionSizingMethod.FIXED_PERCENTAGE
    max_position_size: float = 1.0
    min_position_size: float = 0.001
    use_compounding: bool = True
    kelly_win_rate: Optional[float] = None
    kelly_avg_win: Optional[float] = None
    kelly_avg_loss: Optional[float] = None
    volatility_factor: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration data after initialization."""
        if self.account_balance <= 0:
            raise InvalidPositionSizingConfigError("Account balance must be positive")
        if not 0.0 < self.risk_percentage <= 100.0:
            raise InvalidPositionSizingConfigError(
                "Risk percentage must be between 0.0 and 100.0"
            )
        if self.max_position_size <= 0:
            raise InvalidPositionSizingConfigError("Max position size must be positive")
        if self.min_position_size <= 0:
            raise InvalidPositionSizingConfigError("Min position size must be positive")
        if self.min_position_size > self.max_position_size:
            raise InvalidPositionSizingConfigError(
                "Min position size cannot exceed max position size"
            )

        # Kelly criterion specific validations
        if self.method == PositionSizingMethod.KELLY_CRITERION:
            if any(
                param is None
                for param in [self.kelly_win_rate, self.kelly_avg_win, self.kelly_avg_loss]
            ):
                raise InvalidPositionSizingConfigError(
                    "Kelly criterion requires win_rate, avg_win, and avg_loss parameters"
                )

        # Volatility adjusted specific validations
        if self.method == PositionSizingMethod.VOLATILITY_ADJUSTED:
            if self.volatility_factor is None:
                raise InvalidPositionSizingConfigError(
                    "Volatility adjusted method requires volatility_factor parameter"
                )


@dataclass
class PositionSizingResult:
    """Result data structure for position sizing calculations.

    Contains the calculated position size and related metadata for
    risk management and audit purposes.
    """

    position_size: float
    risk_amount: float
    entry_price: float
    stop_loss_price: Optional[float]
    method_used: PositionSizingMethod
    account_balance: float
    risk_percentage: float
    calculation_timestamp: int
    confidence: float = 1.0
    warnings: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result data after initialization."""
        if self.position_size < 0:
            raise PositionSizingCalculationError("Position size cannot be negative")
        if self.risk_amount < 0:
            raise PositionSizingCalculationError("Risk amount cannot be negative")
        if self.entry_price <= 0:
            raise PositionSizingCalculationError("Entry price must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise PositionSizingCalculationError("Confidence must be between 0.0 and 1.0")


class IPositionSizer(ABC):
    """Abstract interface for position sizing implementations.

    This interface defines the contract for position sizing implementations,
    following the Interface Segregation Principle.
    """

    @abstractmethod
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
    ) -> PositionSizingResult:
        """Calculate position size based on risk parameters.

        Args:
            entry_price: Entry price for the position
            stop_loss_price: Optional stop loss price for risk calculation

        Returns:
            PositionSizingResult: Calculated position size and metadata

        Raises:
            PositionSizingCalculationError: If calculation fails
        """
        pass

    @abstractmethod
    def update_config(self, config: PositionSizingConfig) -> None:
        """Update position sizing configuration.

        Args:
            config: New position sizing configuration

        Raises:
            InvalidPositionSizingConfigError: If configuration is invalid
        """
        pass


class PositionSizer(IPositionSizer):
    """Position sizing implementation with multiple calculation methods.

    This class provides various position sizing algorithms including fixed percentage,
    Kelly criterion, and volatility-adjusted methods. It follows SOLID principles
    and integrates with the event-driven architecture.

    Attributes:
        _config: Position sizing configuration
        _event_hub: Event hub for publishing sizing events
        _logger: Logger instance for position sizing logging
        _calculations_count: Counter for performed calculations
    """

    def __init__(
        self,
        config: PositionSizingConfig,
        event_hub: Optional[EventHub] = None,
    ) -> None:
        """Initialize position sizer with configuration and dependencies.

        Args:
            config: Position sizing configuration
            event_hub: Optional event hub for event publishing

        Raises:
            InvalidPositionSizingConfigError: If configuration is invalid
        """
        if not isinstance(config, PositionSizingConfig):
            raise InvalidPositionSizingConfigError(
                "Config must be PositionSizingConfig instance"
            )

        self._config = config
        self._event_hub = event_hub
        self._logger = get_module_logger("position_sizer")
        self._calculations_count = 0

        self._logger.info(f"Initialized position sizer with method: {config.method.value}")

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
    ) -> PositionSizingResult:
        """Calculate position size based on configured risk parameters.

        Args:
            entry_price: Entry price for the position
            stop_loss_price: Optional stop loss price for risk calculation

        Returns:
            PositionSizingResult: Calculated position size and metadata

        Raises:
            PositionSizingCalculationError: If calculation fails
        """
        try:
            self._validate_calculation_inputs(entry_price, stop_loss_price)

            # Calculate risk amount
            risk_amount = self._calculate_risk_amount()

            # Calculate position size based on method
            position_size = self._calculate_position_size_by_method(
                entry_price, stop_loss_price, risk_amount
            )

            # Apply position size limits
            position_size = self._apply_position_limits(position_size)

            # Create result
            result = self._create_result(
                position_size, risk_amount, entry_price, stop_loss_price
            )

            # Publish event if event hub is available
            if self._event_hub:
                self._publish_sizing_event(result)

            # Update statistics
            self._calculations_count += 1

            self._logger.info(
                f"Calculated position size: {position_size:.6f} for entry price: {entry_price}"
            )

            return result

        except Exception as e:
            self._logger.error(f"Error calculating position size: {e}")
            raise PositionSizingCalculationError(f"Position size calculation failed: {e}")

    def update_config(self, config: PositionSizingConfig) -> None:
        """Update position sizing configuration.

        Args:
            config: New position sizing configuration

        Raises:
            InvalidPositionSizingConfigError: If configuration is invalid
        """
        if not isinstance(config, PositionSizingConfig):
            raise InvalidPositionSizingConfigError(
                "Config must be PositionSizingConfig instance"
            )

        old_method = self._config.method.value
        self._config = config

        self._logger.info(
            f"Updated position sizing config: method changed from {old_method} "
            f"to {config.method.value}"
        )

    def _validate_calculation_inputs(
        self, entry_price: float, stop_loss_price: Optional[float]
    ) -> None:
        """Validate inputs for position size calculation.

        Args:
            entry_price: Entry price to validate
            stop_loss_price: Stop loss price to validate

        Raises:
            PositionSizingCalculationError: If inputs are invalid
        """
        if entry_price <= 0:
            raise PositionSizingCalculationError("Entry price must be positive")

        if stop_loss_price is not None:
            if stop_loss_price <= 0:
                raise PositionSizingCalculationError("Stop loss price must be positive")
            if stop_loss_price >= entry_price:
                raise PositionSizingCalculationError(
                    "Stop loss price must be less than entry price"
                )

    def _calculate_risk_amount(self) -> float:
        """Calculate the amount to risk based on account balance and risk percentage.

        Returns:
            Risk amount in base currency
        """
        return self._config.account_balance * (self._config.risk_percentage / 100.0)

    def _calculate_position_size_by_method(
        self, entry_price: float, stop_loss_price: Optional[float], risk_amount: float
    ) -> float:
        """Calculate position size using the configured method.

        Args:
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price for risk calculation
            risk_amount: Amount to risk

        Returns:
            Calculated position size
        """
        method = self._config.method

        if method == PositionSizingMethod.FIXED_PERCENTAGE:
            return self._calculate_fixed_percentage_size(
                entry_price, stop_loss_price, risk_amount
            )
        elif method == PositionSizingMethod.KELLY_CRITERION:
            return self._calculate_kelly_criterion_size(risk_amount)
        elif method == PositionSizingMethod.VOLATILITY_ADJUSTED:
            return self._calculate_volatility_adjusted_size(
                entry_price, stop_loss_price, risk_amount
            )
        elif method == PositionSizingMethod.EQUAL_RISK:
            return self._calculate_equal_risk_size(
                entry_price, stop_loss_price, risk_amount
            )
        else:
            raise PositionSizingCalculationError(f"Unsupported method: {method}")

    def _calculate_fixed_percentage_size(
        self, entry_price: float, stop_loss_price: Optional[float], risk_amount: float
    ) -> float:
        """Calculate position size using fixed percentage method.

        Args:
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price for risk calculation
            risk_amount: Amount to risk

        Returns:
            Position size based on fixed percentage
        """
        if stop_loss_price is None:
            # Simple percentage of account balance
            return risk_amount / entry_price

        # Calculate based on risk per share
        risk_per_share = entry_price - stop_loss_price
        return risk_amount / risk_per_share

    def _calculate_kelly_criterion_size(self, risk_amount: float) -> float:
        """Calculate position size using Kelly criterion method.

        Args:
            risk_amount: Amount to risk

        Returns:
            Position size based on Kelly criterion
        """
        win_rate = self._config.kelly_win_rate
        avg_win = self._config.kelly_avg_win
        avg_loss = self._config.kelly_avg_loss

        # Kelly formula: f = (bp - q) / b
        # where b = odds (avg_win/avg_loss), p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate

        kelly_fraction = (b * p - q) / b

        # Apply Kelly fraction to risk amount
        # Cap at reasonable maximum to avoid over-leverage
        kelly_fraction = max(0, min(kelly_fraction, 0.25))
        return risk_amount * kelly_fraction

    def _calculate_volatility_adjusted_size(
        self, entry_price: float, stop_loss_price: Optional[float], risk_amount: float
    ) -> float:
        """Calculate position size using volatility-adjusted method.

        Args:
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price for risk calculation
            risk_amount: Amount to risk

        Returns:
            Position size adjusted for volatility
        """
        base_size = self._calculate_fixed_percentage_size(
            entry_price, stop_loss_price, risk_amount
        )

        # Adjust by volatility factor (higher volatility = smaller position)
        volatility_factor = self._config.volatility_factor
        volatility_adjustment = 1.0 / (1.0 + volatility_factor)

        return base_size * volatility_adjustment

    def _calculate_equal_risk_size(
        self, entry_price: float, stop_loss_price: Optional[float], risk_amount: float
    ) -> float:
        """Calculate position size using equal risk method.

        Args:
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price for risk calculation
            risk_amount: Amount to risk

        Returns:
            Position size for equal risk exposure
        """
        # Equal risk method is similar to fixed percentage but normalizes risk
        return self._calculate_fixed_percentage_size(
            entry_price, stop_loss_price, risk_amount
        )

    def _apply_position_limits(self, position_size: float) -> float:
        """Apply minimum and maximum position size limits.

        Args:
            position_size: Calculated position size

        Returns:
            Position size after applying limits
        """
        # Apply minimum limit
        if position_size < self._config.min_position_size:
            position_size = self._config.min_position_size

        # Apply maximum limit
        if position_size > self._config.max_position_size:
            position_size = self._config.max_position_size

        return position_size

    def _create_result(
        self,
        position_size: float,
        risk_amount: float,
        entry_price: float,
        stop_loss_price: Optional[float],
    ) -> PositionSizingResult:
        """Create position sizing result with metadata.

        Args:
            position_size: Calculated position size
            risk_amount: Risk amount used
            entry_price: Entry price
            stop_loss_price: Stop loss price

        Returns:
            PositionSizingResult instance
        """
        warnings = []

        # Check for potential warnings
        if position_size == self._config.min_position_size:
            warnings.append("Position size capped at minimum limit")
        elif position_size == self._config.max_position_size:
            warnings.append("Position size capped at maximum limit")

        return PositionSizingResult(
            position_size=position_size,
            risk_amount=risk_amount,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            method_used=self._config.method,
            account_balance=self._config.account_balance,
            risk_percentage=self._config.risk_percentage,
            calculation_timestamp=int(time.time() * 1000),
            warnings=warnings,
            metadata={
                "calculations_count": self._calculations_count,
                "config_metadata": self._config.metadata,
            },
        )

    def _publish_sizing_event(self, result: PositionSizingResult) -> None:
        """Publish position sizing event to event hub.

        Args:
            result: Position sizing result to publish
        """
        try:
            event_data = {
                "result": result,
                "method": result.method_used.value,
                "position_size": result.position_size,
                "risk_amount": result.risk_amount,
                "timestamp": result.calculation_timestamp,
            }

            self._event_hub.publish(EventType.POSITION_SIZE_WARNING, event_data)

        except Exception as e:
            self._logger.error(f"Error publishing sizing event: {e}")


def create_position_sizer(
    account_balance: float,
    risk_percentage: float,
    method: PositionSizingMethod = PositionSizingMethod.FIXED_PERCENTAGE,
    event_hub: Optional[EventHub] = None,
    **kwargs: Any,
) -> PositionSizer:
    """Factory function to create PositionSizer instance.

    Args:
        account_balance: Account balance for calculations
        risk_percentage: Risk percentage per trade
        method: Position sizing method to use
        event_hub: Optional event hub for event publishing
        **kwargs: Additional configuration parameters

    Returns:
        PositionSizer: Configured position sizer instance

    Raises:
        InvalidPositionSizingConfigError: If configuration is invalid
    """
    config = PositionSizingConfig(
        account_balance=account_balance,
        risk_percentage=risk_percentage,
        method=method,
        max_position_size=kwargs.get("max_position_size", 1.0),
        min_position_size=kwargs.get("min_position_size", 0.001),
        use_compounding=kwargs.get("use_compounding", True),
        kelly_win_rate=kwargs.get("kelly_win_rate"),
        kelly_avg_win=kwargs.get("kelly_avg_win"),
        kelly_avg_loss=kwargs.get("kelly_avg_loss"),
        volatility_factor=kwargs.get("volatility_factor"),
        metadata=kwargs.get("metadata", {}),
    )

    return PositionSizer(config, event_hub)
