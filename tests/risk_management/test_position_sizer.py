"""
Unit tests for position sizing module.

Tests all position sizing methods, error handling, and edge cases
to ensure reliable risk management functionality.
"""

import time
from unittest.mock import Mock

import pytest

from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.risk_management.position_sizer import (
    InvalidPositionSizingConfigError,
    PositionSizer,
    PositionSizingCalculationError,
    PositionSizingConfig,
    PositionSizingMethod,
    PositionSizingResult,
    create_position_sizer,
)


class TestPositionSizingConfig:
    """Test cases for PositionSizingConfig validation."""

    def test_valid_config_creation(self):
        """Test creation of valid configuration."""
        config = PositionSizingConfig(
            account_balance=10000.0,
            risk_percentage=2.0,
            method=PositionSizingMethod.FIXED_PERCENTAGE,
        )

        assert config.account_balance == 10000.0
        assert config.risk_percentage == 2.0
        assert config.method == PositionSizingMethod.FIXED_PERCENTAGE
        assert config.max_position_size == 1.0
        assert config.min_position_size == 0.001

    def test_invalid_account_balance(self):
        """Test validation with invalid account balance."""
        with pytest.raises(
            InvalidPositionSizingConfigError, match="Account balance must be positive"
        ):
            PositionSizingConfig(
                account_balance=-1000.0,
                risk_percentage=2.0,
            )

    def test_invalid_risk_percentage_negative(self):
        """Test validation with negative risk percentage."""
        with pytest.raises(
            InvalidPositionSizingConfigError, match="Risk percentage must be between"
        ):
            PositionSizingConfig(
                account_balance=10000.0,
                risk_percentage=-1.0,
            )

    def test_invalid_risk_percentage_too_high(self):
        """Test validation with risk percentage over 100."""
        with pytest.raises(
            InvalidPositionSizingConfigError, match="Risk percentage must be between"
        ):
            PositionSizingConfig(
                account_balance=10000.0,
                risk_percentage=101.0,
            )

    def test_invalid_position_size_limits(self):
        """Test validation with invalid position size limits."""
        with pytest.raises(
            InvalidPositionSizingConfigError,
            match="Min position size cannot exceed max",
        ):
            PositionSizingConfig(
                account_balance=10000.0,
                risk_percentage=2.0,
                min_position_size=1.0,
                max_position_size=0.5,
            )

    def test_kelly_criterion_missing_parameters(self):
        """Test Kelly criterion validation with missing parameters."""
        with pytest.raises(
            InvalidPositionSizingConfigError, match="Kelly criterion requires"
        ):
            PositionSizingConfig(
                account_balance=10000.0,
                risk_percentage=2.0,
                method=PositionSizingMethod.KELLY_CRITERION,
            )

    def test_volatility_adjusted_missing_factor(self):
        """Test volatility adjusted validation with missing factor."""
        with pytest.raises(
            InvalidPositionSizingConfigError,
            match="Volatility adjusted method requires",
        ):
            PositionSizingConfig(
                account_balance=10000.0,
                risk_percentage=2.0,
                method=PositionSizingMethod.VOLATILITY_ADJUSTED,
            )


class TestPositionSizingResult:
    """Test cases for PositionSizingResult validation."""

    def test_valid_result_creation(self):
        """Test creation of valid result."""
        result = PositionSizingResult(
            position_size=0.5,
            risk_amount=200.0,
            entry_price=50.0,
            stop_loss_price=45.0,
            method_used=PositionSizingMethod.FIXED_PERCENTAGE,
            account_balance=10000.0,
            risk_percentage=2.0,
            calculation_timestamp=int(time.time() * 1000),
        )

        assert result.position_size == 0.5
        assert result.risk_amount == 200.0
        assert result.confidence == 1.0

    def test_invalid_negative_position_size(self):
        """Test validation with negative position size."""
        with pytest.raises(
            PositionSizingCalculationError, match="Position size cannot be negative"
        ):
            PositionSizingResult(
                position_size=-0.5,
                risk_amount=200.0,
                entry_price=50.0,
                stop_loss_price=45.0,
                method_used=PositionSizingMethod.FIXED_PERCENTAGE,
                account_balance=10000.0,
                risk_percentage=2.0,
                calculation_timestamp=int(time.time() * 1000),
            )

    def test_invalid_confidence_range(self):
        """Test validation with invalid confidence."""
        with pytest.raises(
            PositionSizingCalculationError, match="Confidence must be between"
        ):
            PositionSizingResult(
                position_size=0.5,
                risk_amount=200.0,
                entry_price=50.0,
                stop_loss_price=45.0,
                method_used=PositionSizingMethod.FIXED_PERCENTAGE,
                account_balance=10000.0,
                risk_percentage=2.0,
                calculation_timestamp=int(time.time() * 1000),
                confidence=1.5,
            )


class TestPositionSizer:
    """Test cases for PositionSizer implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = PositionSizingConfig(
            account_balance=10000.0,
            risk_percentage=2.0,
            method=PositionSizingMethod.FIXED_PERCENTAGE,
            max_position_size=100.0,  # Increase max to avoid capping in tests
            min_position_size=0.001,
        )
        self.event_hub = Mock(spec=EventHub)
        self.position_sizer = PositionSizer(self.config, self.event_hub)

    def test_initialization(self):
        """Test position sizer initialization."""
        assert self.position_sizer._config == self.config
        assert self.position_sizer._event_hub == self.event_hub
        assert self.position_sizer._calculations_count == 0

    def test_invalid_config_type(self):
        """Test initialization with invalid config type."""
        with pytest.raises(
            InvalidPositionSizingConfigError,
            match="Config must be PositionSizingConfig instance",
        ):
            PositionSizer({"invalid": "config"})

    def test_fixed_percentage_calculation_with_stop_loss(self):
        """Test fixed percentage calculation with stop loss."""
        result = self.position_sizer.calculate_position_size(
            entry_price=100.0,
            stop_loss_price=95.0,
        )

        expected_risk_amount = 10000.0 * 0.02  # 200.0
        expected_position_size = 200.0 / (100.0 - 95.0)  # 40.0

        assert result.position_size == expected_position_size
        assert result.risk_amount == expected_risk_amount
        assert result.entry_price == 100.0
        assert result.stop_loss_price == 95.0
        assert result.method_used == PositionSizingMethod.FIXED_PERCENTAGE

    def test_fixed_percentage_calculation_without_stop_loss(self):
        """Test fixed percentage calculation without stop loss."""
        result = self.position_sizer.calculate_position_size(entry_price=100.0)

        expected_risk_amount = 10000.0 * 0.02  # 200.0
        expected_position_size = 200.0 / 100.0  # 2.0

        assert result.position_size == expected_position_size
        assert result.risk_amount == expected_risk_amount
        assert result.stop_loss_price is None

    def test_kelly_criterion_calculation(self):
        """Test Kelly criterion calculation."""
        kelly_config = PositionSizingConfig(
            account_balance=10000.0,
            risk_percentage=2.0,
            method=PositionSizingMethod.KELLY_CRITERION,
            kelly_win_rate=0.6,
            kelly_avg_win=150.0,
            kelly_avg_loss=100.0,
            max_position_size=100.0,  # Increase max to avoid capping in tests
        )
        kelly_sizer = PositionSizer(kelly_config)

        result = kelly_sizer.calculate_position_size(entry_price=100.0)

        # Kelly formula: f = (bp - q) / b
        # b = 150/100 = 1.5, p = 0.6, q = 0.4
        # f = (1.5 * 0.6 - 0.4) / 1.5 = (0.9 - 0.4) / 1.5 = 0.333...
        # But Kelly fraction is capped at 0.25 for safety
        expected_kelly_fraction = min((1.5 * 0.6 - 0.4) / 1.5, 0.25)
        expected_position_size = 200.0 * expected_kelly_fraction

        assert abs(result.position_size - expected_position_size) < 0.001
        assert result.method_used == PositionSizingMethod.KELLY_CRITERION

    def test_volatility_adjusted_calculation(self):
        """Test volatility adjusted calculation."""
        volatility_config = PositionSizingConfig(
            account_balance=10000.0,
            risk_percentage=2.0,
            method=PositionSizingMethod.VOLATILITY_ADJUSTED,
            volatility_factor=0.2,
            max_position_size=100.0,  # Increase max to avoid capping in tests
        )
        volatility_sizer = PositionSizer(volatility_config)

        result = volatility_sizer.calculate_position_size(
            entry_price=100.0,
            stop_loss_price=95.0,
        )

        # Base calculation: 200 / (100 - 95) = 40
        # Volatility adjustment: 1 / (1 + 0.2) = 0.833...
        # Expected: 40 * 0.833... = 33.333...
        base_size = 200.0 / 5.0  # 40.0
        volatility_adjustment = 1.0 / (1.0 + 0.2)  # 0.8333...
        expected_position_size = base_size * volatility_adjustment

        assert abs(result.position_size - expected_position_size) < 0.001
        assert result.method_used == PositionSizingMethod.VOLATILITY_ADJUSTED

    def test_position_size_limits_minimum(self):
        """Test position size minimum limit enforcement."""
        # Create config with high minimum that will be triggered
        config = PositionSizingConfig(
            account_balance=10000.0,
            risk_percentage=0.001,  # Very small risk
            min_position_size=1.0,
        )
        sizer = PositionSizer(config)

        result = sizer.calculate_position_size(entry_price=1000.0)

        assert result.position_size == 1.0  # Should be capped at minimum
        assert "Position size capped at minimum limit" in result.warnings

    def test_position_size_limits_maximum(self):
        """Test position size maximum limit enforcement."""
        # Create config with low maximum that will be triggered
        config = PositionSizingConfig(
            account_balance=10000.0,
            risk_percentage=50.0,  # Very high risk
            max_position_size=10.0,
        )
        sizer = PositionSizer(config)

        result = sizer.calculate_position_size(entry_price=1.0)

        assert result.position_size == 10.0  # Should be capped at maximum
        assert "Position size capped at maximum limit" in result.warnings

    def test_invalid_entry_price(self):
        """Test calculation with invalid entry price."""
        with pytest.raises(
            PositionSizingCalculationError, match="Entry price must be positive"
        ):
            self.position_sizer.calculate_position_size(entry_price=-100.0)

    def test_invalid_stop_loss_price(self):
        """Test calculation with invalid stop loss price."""
        with pytest.raises(
            PositionSizingCalculationError, match="Stop loss price must be positive"
        ):
            self.position_sizer.calculate_position_size(
                entry_price=100.0,
                stop_loss_price=-50.0,
            )

    def test_stop_loss_above_entry_price(self):
        """Test calculation with stop loss above entry price."""
        with pytest.raises(
            PositionSizingCalculationError,
            match="Stop loss price must be less than entry price",
        ):
            self.position_sizer.calculate_position_size(
                entry_price=100.0,
                stop_loss_price=105.0,
            )

    def test_event_publishing(self):
        """Test event publishing functionality."""
        result = self.position_sizer.calculate_position_size(entry_price=100.0)

        # Verify event was published
        self.event_hub.publish.assert_called_once()
        call_args = self.event_hub.publish.call_args

        assert call_args[0][0] == EventType.POSITION_SIZE_WARNING
        event_data = call_args[0][1]
        assert event_data["result"] == result
        assert event_data["position_size"] == result.position_size

    def test_calculations_counter(self):
        """Test calculations counter increment."""
        initial_count = self.position_sizer._calculations_count

        self.position_sizer.calculate_position_size(entry_price=100.0)
        assert self.position_sizer._calculations_count == initial_count + 1

        self.position_sizer.calculate_position_size(entry_price=110.0)
        assert self.position_sizer._calculations_count == initial_count + 2

    def test_config_update(self):
        """Test configuration update."""
        new_config = PositionSizingConfig(
            account_balance=20000.0,
            risk_percentage=1.5,
            method=PositionSizingMethod.KELLY_CRITERION,
            kelly_win_rate=0.7,
            kelly_avg_win=120.0,
            kelly_avg_loss=80.0,
        )

        self.position_sizer.update_config(new_config)
        assert self.position_sizer._config == new_config

    def test_config_update_invalid_type(self):
        """Test configuration update with invalid type."""
        with pytest.raises(
            InvalidPositionSizingConfigError,
            match="Config must be PositionSizingConfig instance",
        ):
            self.position_sizer.update_config({"invalid": "config"})


class TestFactoryFunction:
    """Test cases for create_position_sizer factory function."""

    def test_factory_function_basic(self):
        """Test basic factory function usage."""
        sizer = create_position_sizer(
            account_balance=10000.0,
            risk_percentage=2.0,
        )

        assert isinstance(sizer, PositionSizer)
        assert sizer._config.account_balance == 10000.0
        assert sizer._config.risk_percentage == 2.0
        assert sizer._config.method == PositionSizingMethod.FIXED_PERCENTAGE

    def test_factory_function_with_event_hub(self):
        """Test factory function with event hub."""
        event_hub = Mock(spec=EventHub)
        sizer = create_position_sizer(
            account_balance=10000.0,
            risk_percentage=2.0,
            event_hub=event_hub,
        )

        assert sizer._event_hub == event_hub

    def test_factory_function_with_kelly_method(self):
        """Test factory function with Kelly criterion method."""
        sizer = create_position_sizer(
            account_balance=10000.0,
            risk_percentage=2.0,
            method=PositionSizingMethod.KELLY_CRITERION,
            kelly_win_rate=0.6,
            kelly_avg_win=150.0,
            kelly_avg_loss=100.0,
        )

        assert sizer._config.method == PositionSizingMethod.KELLY_CRITERION
        assert sizer._config.kelly_win_rate == 0.6

    def test_factory_function_with_kwargs(self):
        """Test factory function with additional kwargs."""
        sizer = create_position_sizer(
            account_balance=10000.0,
            risk_percentage=2.0,
            max_position_size=2.0,
            min_position_size=0.1,
            metadata={"test": "data"},
        )

        assert sizer._config.max_position_size == 2.0
        assert sizer._config.min_position_size == 0.1
        assert sizer._config.metadata == {"test": "data"}


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_account_balance(self):
        """Test with very small account balance."""
        config = PositionSizingConfig(
            account_balance=1.0,
            risk_percentage=1.0,
        )
        sizer = PositionSizer(config)

        result = sizer.calculate_position_size(entry_price=100.0)

        expected_position_size = max(0.01 / 100.0, config.min_position_size)

        assert result.position_size == expected_position_size

    def test_very_high_entry_price(self):
        """Test with very high entry price."""
        config = PositionSizingConfig(
            account_balance=10000.0,
            risk_percentage=2.0,
        )
        sizer = PositionSizer(config)

        result = sizer.calculate_position_size(entry_price=1000000.0)

        expected_position_size = max(200.0 / 1000000.0, config.min_position_size)

        assert result.position_size == expected_position_size

    def test_zero_kelly_fraction(self):
        """Test Kelly criterion with parameters yielding zero fraction."""
        kelly_config = PositionSizingConfig(
            account_balance=10000.0,
            risk_percentage=2.0,
            method=PositionSizingMethod.KELLY_CRITERION,
            kelly_win_rate=0.3,  # Low win rate
            kelly_avg_win=100.0,
            kelly_avg_loss=200.0,  # High average loss
        )
        sizer = PositionSizer(kelly_config)

        result = sizer.calculate_position_size(entry_price=100.0)

        # Should result in very small or zero position size
        assert result.position_size >= 0
        assert result.position_size <= kelly_config.max_position_size
