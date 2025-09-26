"""
Stop-loss and take-profit level calculation module for trading bot risk management.

This module provides comprehensive stop-loss and take-profit level calculations
using multiple methodologies including fixed percentage, ATR-based, and
volatility-adjusted
approaches. Follows SOLID principles and integrates with the event-driven architecture.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.core.logger import get_module_logger


class StopLossCalculationError(Exception):
    """Base exception for stop-loss calculation related errors."""


class InvalidStopLossConfigError(StopLossCalculationError):
    """Exception raised for invalid stop-loss configuration."""


class InvalidPriceLevelError(StopLossCalculationError):
    """Exception raised for invalid price level calculations."""


class StopLossMethod(Enum):
    """Stop-loss calculation methods."""

    FIXED_PERCENTAGE = "fixed_percentage"
    ATR_BASED = "atr_based"
    SUPPORT_RESISTANCE = "support_resistance"
    VOLATILITY_ADJUSTED = "volatility_adjusted"


class PositionType(Enum):
    """Position types for stop-loss calculations."""

    LONG = "long"
    SHORT = "short"


@dataclass
class StopLossConfig:
    """Configuration data structure for stop-loss calculations.

    This dataclass defines the parameters required for stop-loss and take-profit
    calculations across different methodologies.
    """

    # Basic configuration
    method: StopLossMethod = StopLossMethod.FIXED_PERCENTAGE
    risk_reward_ratio: float = 2.0

    # Fixed percentage parameters
    stop_loss_percentage: float = 2.0
    take_profit_percentage: Optional[float] = None

    # ATR-based parameters
    atr_multiplier: float = 2.0
    atr_period: int = 14
    current_atr: Optional[float] = None

    # Support/resistance parameters
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    buffer_percentage: float = 0.1

    # Volatility adjustment parameters
    volatility_multiplier: float = 1.5
    current_volatility: Optional[float] = None
    base_stop_percentage: float = 1.5

    # Position limits
    max_stop_loss_percentage: float = 5.0
    min_stop_loss_percentage: float = 0.5
    max_take_profit_percentage: float = 20.0
    min_take_profit_percentage: float = 1.0

    # Validation and safety
    allow_negative_stops: bool = False
    validate_price_levels: bool = True

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration data after initialization."""
        if self.risk_reward_ratio <= 0:
            raise InvalidStopLossConfigError("Risk reward ratio must be positive")

        if not 0.1 <= self.stop_loss_percentage <= 50.0:
            raise InvalidStopLossConfigError(
                "Stop loss percentage must be between 0.1% and 50%"
            )

        if self.take_profit_percentage is not None:
            if not 0.1 <= self.take_profit_percentage <= 100.0:
                raise InvalidStopLossConfigError(
                    "Take profit percentage must be between 0.1% and 100%"
                )

        if self.atr_multiplier <= 0:
            raise InvalidStopLossConfigError("ATR multiplier must be positive")

        if self.atr_period <= 0:
            raise InvalidStopLossConfigError("ATR period must be positive")

        if self.buffer_percentage < 0:
            raise InvalidStopLossConfigError("Buffer percentage cannot be negative")

        if self.max_stop_loss_percentage <= self.min_stop_loss_percentage:
            raise InvalidStopLossConfigError(
                "Max stop loss percentage must exceed min stop loss percentage"
            )

        # Method-specific validations
        # Note: ATR method can work with either config ATR or market data ATR
        # Validation happens during calculation, not configuration

        if self.method == StopLossMethod.SUPPORT_RESISTANCE:
            if self.support_level is None and self.resistance_level is None:
                raise InvalidStopLossConfigError(
                    "Support/resistance method requires at least one level"
                )

        # Note: Volatility method can work with either config volatility or
        # market data volatility
        # Validation happens during calculation, not configuration


@dataclass
class StopLossLevel:
    """Individual stop-loss or take-profit level calculation result."""

    price: float
    percentage_from_entry: float
    distance_from_entry: float
    level_type: str  # "stop_loss" or "take_profit"
    calculation_method: str
    confidence: float = 1.0
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate level data after initialization."""
        if self.price <= 0:
            raise InvalidPriceLevelError("Price must be positive")
        if self.distance_from_entry < 0:
            raise InvalidPriceLevelError("Distance from entry cannot be negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise InvalidPriceLevelError("Confidence must be between 0.0 and 1.0")


@dataclass
class StopLossResult:
    """Comprehensive stop-loss and take-profit calculation result.

    Contains calculated levels, validation status, and metadata for
    risk management and audit purposes.
    """

    entry_price: float
    position_type: PositionType
    stop_loss_level: Optional[StopLossLevel]
    take_profit_level: Optional[StopLossLevel]
    method_used: StopLossMethod
    risk_reward_ratio: float
    calculation_timestamp: int
    is_valid: bool = True
    overall_confidence: float = 1.0
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result data after initialization."""
        if self.entry_price <= 0:
            raise StopLossCalculationError("Entry price must be positive")
        if self.risk_reward_ratio <= 0:
            raise StopLossCalculationError("Risk reward ratio must be positive")
        if not 0.0 <= self.overall_confidence <= 1.0:
            raise StopLossCalculationError(
                "Overall confidence must be between 0.0 and 1.0"
            )

    def get_risk_amount(self, position_size: float) -> float:
        """Calculate risk amount for the position.

        Args:
            position_size: Size of the position

        Returns:
            Risk amount in base currency
        """
        if not self.stop_loss_level:
            return 0.0

        risk_per_unit = abs(self.entry_price - self.stop_loss_level.price)
        return position_size * risk_per_unit

    def get_profit_potential(self, position_size: float) -> float:
        """Calculate profit potential for the position.

        Args:
            position_size: Size of the position

        Returns:
            Profit potential in base currency
        """
        if not self.take_profit_level:
            return 0.0

        profit_per_unit = abs(self.take_profit_level.price - self.entry_price)
        return position_size * profit_per_unit

    def validate_levels(self) -> Tuple[bool, List[str]]:
        """Validate calculated levels for logical consistency.

        Returns:
            Tuple of (is_valid, validation_errors)
        """
        errors = []

        if self.stop_loss_level:
            if self.position_type == PositionType.LONG:
                if self.stop_loss_level.price >= self.entry_price:
                    errors.append("Long position stop-loss must be below entry price")
            else:  # SHORT
                if self.stop_loss_level.price <= self.entry_price:
                    errors.append("Short position stop-loss must be above entry price")

        if self.take_profit_level:
            if self.position_type == PositionType.LONG:
                if self.take_profit_level.price <= self.entry_price:
                    errors.append("Long position take-profit must be above entry price")
            else:  # SHORT
                if self.take_profit_level.price >= self.entry_price:
                    errors.append(
                        "Short position take-profit must be below entry price"
                    )

        if self.stop_loss_level and self.take_profit_level:
            actual_ratio = abs(self.take_profit_level.price - self.entry_price) / abs(
                self.entry_price - self.stop_loss_level.price
            )
            expected_ratio = self.risk_reward_ratio

            if abs(actual_ratio - expected_ratio) > 0.1:  # 10% tolerance
                errors.append(
                    f"Risk-reward ratio mismatch: expected {expected_ratio:.2f}, "
                    f"calculated {actual_ratio:.2f}"
                )

        return len(errors) == 0, errors


class IStopLossCalculator(ABC):
    """Abstract interface for stop-loss calculation implementations.

    This interface defines the contract for stop-loss calculation implementations,
    following the Interface Segregation Principle.
    """

    @abstractmethod
    def calculate_levels(
        self,
        entry_price: float,
        position_type: PositionType,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> StopLossResult:
        """Calculate stop-loss and take-profit levels.

        Args:
            entry_price: Entry price for the position
            position_type: Whether this is a long or short position
            market_data: Optional market data for calculations

        Returns:
            StopLossResult: Calculated levels and metadata

        Raises:
            StopLossCalculationError: If calculation fails
        """

    @abstractmethod
    def update_config(self, config: StopLossConfig) -> None:
        """Update stop-loss calculation configuration.

        Args:
            config: New stop-loss configuration

        Raises:
            InvalidStopLossConfigError: If configuration is invalid
        """


class StopLossCalculator(IStopLossCalculator):
    """Comprehensive stop-loss and take-profit level calculator.

    This class provides various stop-loss calculation methods including
    fixed percentage,
    ATR-based, support/resistance, and volatility-adjusted approaches. It follows SOLID
    principles and integrates with the event-driven architecture.

    Attributes:
        _config: Stop-loss calculation configuration
        _event_hub: Event hub for publishing calculation events
        _logger: Logger instance for stop-loss calculation logging
        _calculations_count: Counter for performed calculations
    """

    def __init__(
        self,
        config: StopLossConfig,
        event_hub: Optional[EventHub] = None,
    ) -> None:
        """Initialize stop-loss calculator with configuration and dependencies.

        Args:
            config: Stop-loss calculation configuration
            event_hub: Optional event hub for event publishing

        Raises:
            InvalidStopLossConfigError: If configuration is invalid
        """
        if not isinstance(config, StopLossConfig):
            raise InvalidStopLossConfigError("Config must be StopLossConfig instance")

        self._config = config
        self._event_hub = event_hub
        self._logger = get_module_logger("stop_loss_calculator")
        self._calculations_count = 0

        self._logger.info(
            f"Initialized stop-loss calculator with method: {config.method.value}"
        )

    def calculate_levels(
        self,
        entry_price: float,
        position_type: PositionType,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> StopLossResult:
        """Calculate stop-loss and take-profit levels using configured method.

        Args:
            entry_price: Entry price for the position
            position_type: Whether this is a long or short position
            market_data: Optional market data for calculations

        Returns:
            StopLossResult: Calculated levels and metadata

        Raises:
            StopLossCalculationError: If calculation fails
        """
        try:
            self._validate_calculation_inputs(entry_price, position_type, market_data)

            # Calculate stop-loss level
            stop_loss_level = self._calculate_stop_loss_level(
                entry_price, position_type, market_data
            )

            # Calculate take-profit level
            take_profit_level = self._calculate_take_profit_level(
                entry_price, position_type, stop_loss_level, market_data
            )

            # Create result
            result = self._create_result(
                entry_price, position_type, stop_loss_level, take_profit_level
            )

            # Validate result if enabled
            if self._config.validate_price_levels:
                is_valid, validation_errors = result.validate_levels()
                result.is_valid = is_valid
                if validation_errors:
                    result.warnings.extend(validation_errors)

            # Apply safety limits
            result = self._apply_safety_limits(result)

            # Publish event if event hub is available
            if self._event_hub:
                self._publish_calculation_event(result)

            # Update statistics
            self._calculations_count += 1

            self._logger.info(
                f"Calculated levels for {position_type.value} position: "
                f"entry={entry_price:.6f}, "
                f"stop={stop_loss_level.price if stop_loss_level else 'None'}, "
                f"target={take_profit_level.price if take_profit_level else 'None'}"
            )

            return result

        except Exception as e:
            self._logger.error(f"Error calculating stop-loss levels: {e}")
            raise StopLossCalculationError(f"Stop-loss calculation failed: {e}")

    def update_config(self, config: StopLossConfig) -> None:
        """Update stop-loss calculation configuration.

        Args:
            config: New stop-loss configuration

        Raises:
            InvalidStopLossConfigError: If configuration is invalid
        """
        if not isinstance(config, StopLossConfig):
            raise InvalidStopLossConfigError("Config must be StopLossConfig instance")

        old_method = self._config.method.value
        self._config = config

        self._logger.info(
            f"Updated stop-loss config: method changed from {old_method} "
            f"to {config.method.value}"
        )

    def get_calculation_statistics(self) -> Dict[str, Any]:
        """Get calculation statistics.

        Returns:
            Dictionary containing calculation statistics
        """
        return {
            "calculations_count": self._calculations_count,
            "method": self._config.method.value,
            "config_summary": {
                "risk_reward_ratio": self._config.risk_reward_ratio,
                "stop_loss_percentage": self._config.stop_loss_percentage,
                "max_stop_loss_percentage": self._config.max_stop_loss_percentage,
            },
        }

    def _validate_calculation_inputs(
        self,
        entry_price: float,
        position_type: PositionType,
        market_data: Optional[Dict[str, Any]],
    ) -> None:
        """Validate inputs for stop-loss calculation.

        Args:
            entry_price: Entry price to validate
            position_type: Position type to validate
            market_data: Market data to validate

        Raises:
            StopLossCalculationError: If inputs are invalid
        """
        if entry_price <= 0:
            raise StopLossCalculationError("Entry price must be positive")

        if not isinstance(position_type, PositionType):
            raise StopLossCalculationError("Position type must be PositionType enum")

        if market_data and not isinstance(market_data, dict):
            raise StopLossCalculationError("Market data must be dictionary")

        # Method-specific validations
        if self._config.method == StopLossMethod.ATR_BASED:
            has_atr_in_config = self._config.current_atr is not None
            has_atr_in_market_data = market_data and "atr" in market_data
            if not has_atr_in_config and not has_atr_in_market_data:
                raise StopLossCalculationError(
                    "ATR-based method requires ATR data in config or market_data"
                )

        if self._config.method == StopLossMethod.VOLATILITY_ADJUSTED:
            has_vol_in_config = self._config.current_volatility is not None
            has_vol_in_market_data = market_data and "volatility" in market_data
            if not has_vol_in_config and not has_vol_in_market_data:
                raise StopLossCalculationError(
                    "Volatility-adjusted method requires volatility data in config "
                    "or market_data"
                )

    def _calculate_stop_loss_level(
        self,
        entry_price: float,
        position_type: PositionType,
        market_data: Optional[Dict[str, Any]],
    ) -> Optional[StopLossLevel]:
        """Calculate stop-loss level using configured method.

        Args:
            entry_price: Entry price for the position
            position_type: Position type (long/short)
            market_data: Optional market data

        Returns:
            StopLossLevel or None if calculation fails
        """
        method = self._config.method

        try:
            if method == StopLossMethod.FIXED_PERCENTAGE:
                return self._calculate_fixed_percentage_stop(entry_price, position_type)
            elif method == StopLossMethod.ATR_BASED:
                return self._calculate_atr_based_stop(
                    entry_price, position_type, market_data
                )
            elif method == StopLossMethod.SUPPORT_RESISTANCE:
                return self._calculate_support_resistance_stop(
                    entry_price, position_type
                )
            elif method == StopLossMethod.VOLATILITY_ADJUSTED:
                return self._calculate_volatility_adjusted_stop(
                    entry_price, position_type, market_data
                )
            else:
                raise StopLossCalculationError(f"Unsupported method: {method}")

        except Exception as e:
            self._logger.warning(f"Stop-loss calculation failed: {e}")
            return None

    def _calculate_take_profit_level(
        self,
        entry_price: float,
        position_type: PositionType,
        stop_loss_level: Optional[StopLossLevel],
        market_data: Optional[Dict[str, Any]],
    ) -> Optional[StopLossLevel]:
        """Calculate take-profit level based on risk-reward ratio.

        Args:
            entry_price: Entry price for the position
            position_type: Position type (long/short)
            stop_loss_level: Calculated stop-loss level
            market_data: Optional market data

        Returns:
            StopLossLevel or None if calculation fails
        """
        try:
            if self._config.take_profit_percentage is not None:
                # Use explicit take-profit percentage
                return self._calculate_fixed_percentage_target(
                    entry_price, position_type, self._config.take_profit_percentage
                )

            if stop_loss_level is None:
                # Can't calculate risk-reward based target without stop-loss
                return None

            # Calculate based on risk-reward ratio
            stop_distance = abs(entry_price - stop_loss_level.price)
            target_distance = stop_distance * self._config.risk_reward_ratio

            if position_type == PositionType.LONG:
                target_price = entry_price + target_distance
            else:  # SHORT
                target_price = entry_price - target_distance

            # Ensure target price is positive
            if target_price <= 0:
                return None

            percentage_from_entry = abs(target_price - entry_price) / entry_price * 100

            return StopLossLevel(
                price=target_price,
                percentage_from_entry=percentage_from_entry,
                distance_from_entry=target_distance,
                level_type="take_profit",
                calculation_method=(
                    f"risk_reward_ratio_{self._config.risk_reward_ratio}"
                ),
                confidence=0.9,
                metadata={
                    "risk_reward_ratio": self._config.risk_reward_ratio,
                    "stop_distance": stop_distance,
                },
            )

        except Exception as e:
            self._logger.warning(f"Take-profit calculation failed: {e}")
            return None

    def _calculate_fixed_percentage_stop(
        self, entry_price: float, position_type: PositionType
    ) -> StopLossLevel:
        """Calculate stop-loss using fixed percentage method.

        Args:
            entry_price: Entry price for the position
            position_type: Position type (long/short)

        Returns:
            StopLossLevel with calculated stop-loss
        """
        percentage = self._config.stop_loss_percentage / 100.0

        if position_type == PositionType.LONG:
            stop_price = entry_price * (1.0 - percentage)
        else:  # SHORT
            stop_price = entry_price * (1.0 + percentage)

        distance = abs(entry_price - stop_price)

        return StopLossLevel(
            price=stop_price,
            percentage_from_entry=self._config.stop_loss_percentage,
            distance_from_entry=distance,
            level_type="stop_loss",
            calculation_method="fixed_percentage",
            confidence=1.0,
            metadata={"percentage_used": self._config.stop_loss_percentage},
        )

    def _calculate_atr_based_stop(
        self,
        entry_price: float,
        position_type: PositionType,
        market_data: Optional[Dict[str, Any]],
    ) -> StopLossLevel:
        """Calculate stop-loss using ATR-based method.

        Args:
            entry_price: Entry price for the position
            position_type: Position type (long/short)
            market_data: Market data containing ATR

        Returns:
            StopLossLevel with calculated stop-loss
        """
        # Get ATR value - prioritize market data over config
        atr = None
        if market_data and "atr" in market_data:
            atr = market_data["atr"]
        elif self._config.current_atr is not None:
            atr = self._config.current_atr

        if atr is None:
            raise StopLossCalculationError("ATR value not available")

        stop_distance = atr * self._config.atr_multiplier

        if position_type == PositionType.LONG:
            stop_price = entry_price - stop_distance
        else:  # SHORT
            stop_price = entry_price + stop_distance

        # Ensure stop price is positive
        if stop_price <= 0:
            stop_price = entry_price * 0.1  # Fallback to 90% stop

        percentage_from_entry = abs(stop_price - entry_price) / entry_price * 100

        warnings = []
        if percentage_from_entry > self._config.max_stop_loss_percentage:
            warnings.append(
                f"ATR-based stop exceeds maximum: {percentage_from_entry:.2f}%"
            )

        return StopLossLevel(
            price=stop_price,
            percentage_from_entry=percentage_from_entry,
            distance_from_entry=stop_distance,
            level_type="stop_loss",
            calculation_method="atr_based",
            confidence=0.85,
            warnings=warnings,
            metadata={
                "atr_value": atr,
                "atr_multiplier": self._config.atr_multiplier,
                "atr_period": self._config.atr_period,
            },
        )

    def _calculate_support_resistance_stop(
        self, entry_price: float, position_type: PositionType
    ) -> StopLossLevel:
        """Calculate stop-loss using support/resistance levels.

        Args:
            entry_price: Entry price for the position
            position_type: Position type (long/short)

        Returns:
            StopLossLevel with calculated stop-loss
        """
        buffer = self._config.buffer_percentage / 100.0

        if position_type == PositionType.LONG:
            # Use support level for long positions
            base_level = self._config.support_level
            if base_level is None:
                raise StopLossCalculationError(
                    "Support level required for long position"
                )

            stop_price = base_level * (1.0 - buffer)
        else:  # SHORT
            # Use resistance level for short positions
            base_level = self._config.resistance_level
            if base_level is None:
                raise StopLossCalculationError(
                    "Resistance level required for short position"
                )

            stop_price = base_level * (1.0 + buffer)

        distance = abs(entry_price - stop_price)
        percentage_from_entry = distance / entry_price * 100

        warnings = []
        confidence = 0.8

        # Validate logical placement
        if position_type == PositionType.LONG and stop_price >= entry_price:
            warnings.append("Support-based stop above entry price")
            confidence = 0.5
        elif position_type == PositionType.SHORT and stop_price <= entry_price:
            warnings.append("Resistance-based stop below entry price")
            confidence = 0.5

        return StopLossLevel(
            price=stop_price,
            percentage_from_entry=percentage_from_entry,
            distance_from_entry=distance,
            level_type="stop_loss",
            calculation_method="support_resistance",
            confidence=confidence,
            warnings=warnings,
            metadata={
                "base_level": base_level,
                "buffer_percentage": self._config.buffer_percentage,
                "level_type": (
                    "support" if position_type == PositionType.LONG else "resistance"
                ),
            },
        )

    def _calculate_volatility_adjusted_stop(
        self,
        entry_price: float,
        position_type: PositionType,
        market_data: Optional[Dict[str, Any]],
    ) -> StopLossLevel:
        """Calculate stop-loss using volatility-adjusted method.

        Args:
            entry_price: Entry price for the position
            position_type: Position type (long/short)
            market_data: Market data containing volatility

        Returns:
            StopLossLevel with calculated stop-loss
        """
        # Get volatility value - prioritize market data over config
        volatility = None
        if market_data and "volatility" in market_data:
            volatility = market_data["volatility"]
        elif self._config.current_volatility is not None:
            volatility = self._config.current_volatility

        if volatility is None:
            raise StopLossCalculationError("Volatility value not available")

        # Adjust base stop percentage by volatility
        base_percentage = self._config.base_stop_percentage / 100.0
        volatility_adjustment = volatility * self._config.volatility_multiplier
        adjusted_percentage = base_percentage * (1.0 + volatility_adjustment)

        # Apply limits
        adjusted_percentage = max(
            self._config.min_stop_loss_percentage / 100.0,
            min(self._config.max_stop_loss_percentage / 100.0, adjusted_percentage),
        )

        if position_type == PositionType.LONG:
            stop_price = entry_price * (1.0 - adjusted_percentage)
        else:  # SHORT
            stop_price = entry_price * (1.0 + adjusted_percentage)

        distance = abs(entry_price - stop_price)
        percentage_from_entry = adjusted_percentage * 100

        warnings = []
        if volatility > 0.1:  # 10% volatility
            warnings.append(f"High volatility detected: {volatility:.2%}")

        return StopLossLevel(
            price=stop_price,
            percentage_from_entry=percentage_from_entry,
            distance_from_entry=distance,
            level_type="stop_loss",
            calculation_method="volatility_adjusted",
            confidence=0.75,
            warnings=warnings,
            metadata={
                "volatility": volatility,
                "base_percentage": self._config.base_stop_percentage,
                "volatility_multiplier": self._config.volatility_multiplier,
                "adjustment_factor": volatility_adjustment,
            },
        )

    def _calculate_fixed_percentage_target(
        self, entry_price: float, position_type: PositionType, percentage: float
    ) -> StopLossLevel:
        """Calculate take-profit using fixed percentage method.

        Args:
            entry_price: Entry price for the position
            position_type: Position type (long/short)
            percentage: Target percentage

        Returns:
            StopLossLevel with calculated take-profit
        """
        percentage_decimal = percentage / 100.0

        if position_type == PositionType.LONG:
            target_price = entry_price * (1.0 + percentage_decimal)
        else:  # SHORT
            target_price = entry_price * (1.0 - percentage_decimal)

        distance = abs(target_price - entry_price)

        return StopLossLevel(
            price=target_price,
            percentage_from_entry=percentage,
            distance_from_entry=distance,
            level_type="take_profit",
            calculation_method="fixed_percentage",
            confidence=1.0,
            metadata={"percentage_used": percentage},
        )

    def _create_result(
        self,
        entry_price: float,
        position_type: PositionType,
        stop_loss_level: Optional[StopLossLevel],
        take_profit_level: Optional[StopLossLevel],
    ) -> StopLossResult:
        """Create comprehensive stop-loss result with metadata.

        Args:
            entry_price: Entry price for the position
            position_type: Position type (long/short)
            stop_loss_level: Calculated stop-loss level
            take_profit_level: Calculated take-profit level

        Returns:
            StopLossResult instance
        """
        warnings = []
        recommendations = []

        # Calculate overall confidence
        confidences = []
        if stop_loss_level:
            confidences.append(stop_loss_level.confidence)
        if take_profit_level:
            confidences.append(take_profit_level.confidence)

        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # Collect warnings
        if stop_loss_level:
            warnings.extend(stop_loss_level.warnings)
        if take_profit_level:
            warnings.extend(take_profit_level.warnings)

        # Generate recommendations
        if stop_loss_level and stop_loss_level.percentage_from_entry > 3.0:
            recommendations.append("Consider tighter stop-loss to reduce risk")

        if stop_loss_level and stop_loss_level.percentage_from_entry < 1.0:
            recommendations.append(
                "Stop-loss may be too tight, consider market volatility"
            )

        if not take_profit_level:
            recommendations.append("Consider setting a take-profit target")

        # Calculate actual risk-reward ratio
        actual_ratio = self._config.risk_reward_ratio
        if stop_loss_level and take_profit_level:
            stop_distance = abs(entry_price - stop_loss_level.price)
            profit_distance = abs(take_profit_level.price - entry_price)
            if stop_distance > 0:
                actual_ratio = profit_distance / stop_distance

        return StopLossResult(
            entry_price=entry_price,
            position_type=position_type,
            stop_loss_level=stop_loss_level,
            take_profit_level=take_profit_level,
            method_used=self._config.method,
            risk_reward_ratio=actual_ratio,
            calculation_timestamp=int(time.time() * 1000),
            overall_confidence=overall_confidence,
            warnings=warnings,
            recommendations=recommendations,
            metadata={
                "calculations_count": self._calculations_count,
                "config_method": self._config.method.value,
                "target_risk_reward": self._config.risk_reward_ratio,
                "actual_risk_reward": actual_ratio,
            },
        )

    def _apply_safety_limits(self, result: StopLossResult) -> StopLossResult:
        """Apply safety limits to calculated levels.

        Args:
            result: Initial calculation result

        Returns:
            Result with safety limits applied
        """
        modified = False

        # Apply stop-loss limits
        if result.stop_loss_level:
            percentage = result.stop_loss_level.percentage_from_entry

            if percentage > self._config.max_stop_loss_percentage:
                # Adjust stop-loss to maximum allowed
                if result.position_type == PositionType.LONG:
                    new_price = result.entry_price * (
                        1.0 - self._config.max_stop_loss_percentage / 100.0
                    )
                else:
                    new_price = result.entry_price * (
                        1.0 + self._config.max_stop_loss_percentage / 100.0
                    )

                result.stop_loss_level.price = new_price
                result.stop_loss_level.percentage_from_entry = (
                    self._config.max_stop_loss_percentage
                )
                result.stop_loss_level.distance_from_entry = abs(
                    result.entry_price - new_price
                )
                result.stop_loss_level.warnings.append(
                    "Stop-loss capped at maximum limit"
                )
                modified = True

            elif percentage < self._config.min_stop_loss_percentage:
                # Adjust stop-loss to minimum allowed
                if result.position_type == PositionType.LONG:
                    new_price = result.entry_price * (
                        1.0 - self._config.min_stop_loss_percentage / 100.0
                    )
                else:
                    new_price = result.entry_price * (
                        1.0 + self._config.min_stop_loss_percentage / 100.0
                    )

                result.stop_loss_level.price = new_price
                result.stop_loss_level.percentage_from_entry = (
                    self._config.min_stop_loss_percentage
                )
                result.stop_loss_level.distance_from_entry = abs(
                    result.entry_price - new_price
                )
                result.stop_loss_level.warnings.append("Stop-loss set to minimum limit")
                modified = True

        # Apply take-profit limits
        if result.take_profit_level:
            percentage = result.take_profit_level.percentage_from_entry

            if percentage > self._config.max_take_profit_percentage:
                result.take_profit_level.warnings.append(
                    "Take-profit exceeds maximum recommended"
                )
            elif percentage < self._config.min_take_profit_percentage:
                result.take_profit_level.warnings.append(
                    "Take-profit below minimum recommended"
                )

        if modified:
            result.warnings.append("Safety limits applied to calculated levels")
            result.overall_confidence *= (
                0.9  # Reduce confidence when limits are applied
            )

        return result

    def _publish_calculation_event(self, result: StopLossResult) -> None:
        """Publish stop-loss calculation event to event hub.

        Args:
            result: Stop-loss calculation result to publish
        """
        try:
            event_data = {
                "result": result,
                "entry_price": result.entry_price,
                "position_type": result.position_type.value,
                "method": result.method_used.value,
                "timestamp": result.calculation_timestamp,
                "stop_loss_price": (
                    result.stop_loss_level.price if result.stop_loss_level else None
                ),
                "take_profit_price": (
                    result.take_profit_level.price if result.take_profit_level else None
                ),
            }

            self._event_hub.publish(EventType.POSITION_SIZE_WARNING, event_data)

            # Publish warning if validation failed
            if not result.is_valid:
                self._event_hub.publish(EventType.RISK_LIMIT_EXCEEDED, event_data)

        except Exception as e:
            self._logger.error(f"Error publishing calculation event: {e}")


def create_stop_loss_calculator(
    method: StopLossMethod = StopLossMethod.FIXED_PERCENTAGE,
    stop_loss_percentage: float = 2.0,
    risk_reward_ratio: float = 2.0,
    event_hub: Optional[EventHub] = None,
    **kwargs: Any,
) -> StopLossCalculator:
    """Factory function to create StopLossCalculator instance.

    Args:
        method: Stop-loss calculation method to use
        stop_loss_percentage: Default stop-loss percentage
        risk_reward_ratio: Target risk-reward ratio
        event_hub: Optional event hub for event publishing
        **kwargs: Additional configuration parameters

    Returns:
        StopLossCalculator: Configured stop-loss calculator instance

    Raises:
        InvalidStopLossConfigError: If configuration is invalid
    """
    config = StopLossConfig(
        method=method,
        risk_reward_ratio=risk_reward_ratio,
        stop_loss_percentage=stop_loss_percentage,
        take_profit_percentage=kwargs.get("take_profit_percentage"),
        atr_multiplier=kwargs.get("atr_multiplier", 2.0),
        atr_period=kwargs.get("atr_period", 14),
        current_atr=kwargs.get("current_atr"),
        support_level=kwargs.get("support_level"),
        resistance_level=kwargs.get("resistance_level"),
        buffer_percentage=kwargs.get("buffer_percentage", 0.1),
        volatility_multiplier=kwargs.get("volatility_multiplier", 1.5),
        current_volatility=kwargs.get("current_volatility"),
        base_stop_percentage=kwargs.get("base_stop_percentage", 1.5),
        max_stop_loss_percentage=kwargs.get("max_stop_loss_percentage", 5.0),
        min_stop_loss_percentage=kwargs.get("min_stop_loss_percentage", 0.5),
        max_take_profit_percentage=kwargs.get("max_take_profit_percentage", 20.0),
        min_take_profit_percentage=kwargs.get("min_take_profit_percentage", 1.0),
        allow_negative_stops=kwargs.get("allow_negative_stops", False),
        validate_price_levels=kwargs.get("validate_price_levels", True),
        metadata=kwargs.get("metadata", {}),
    )

    return StopLossCalculator(config, event_hub)
