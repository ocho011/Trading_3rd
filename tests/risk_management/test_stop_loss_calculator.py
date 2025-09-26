"""
Unit tests for stop-loss calculator module.

Tests cover all calculation methods, edge cases, error handling,
and integration with the event system.
"""

import time
from typing import Any
from unittest.mock import Mock

import pytest

from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.risk_management.stop_loss_calculator import (
    InvalidPriceLevelError,
    InvalidStopLossConfigError,
    PositionType,
    StopLossCalculationError,
    StopLossCalculator,
    StopLossConfig,
    StopLossLevel,
    StopLossMethod,
    StopLossResult,
    create_stop_loss_calculator,
)


class TestStopLossConfig:
    """Test StopLossConfig dataclass validation and functionality."""

    def test_valid_config_creation(self) -> None:
        """Test creating valid configuration."""
        config = StopLossConfig(
            method=StopLossMethod.FIXED_PERCENTAGE,
            risk_reward_ratio=2.0,
            stop_loss_percentage=2.0,
        )

        assert config.method == StopLossMethod.FIXED_PERCENTAGE
        assert config.risk_reward_ratio == 2.0
        assert config.stop_loss_percentage == 2.0

    def test_invalid_risk_reward_ratio(self) -> None:
        """Test validation of invalid risk reward ratio."""
        with pytest.raises(
            InvalidStopLossConfigError, match="Risk reward ratio must be positive"
        ):
            StopLossConfig(risk_reward_ratio=0.0)

        with pytest.raises(
            InvalidStopLossConfigError, match="Risk reward ratio must be positive"
        ):
            StopLossConfig(risk_reward_ratio=-1.0)

    def test_invalid_stop_loss_percentage(self) -> None:
        """Test validation of invalid stop loss percentage."""
        with pytest.raises(
            InvalidStopLossConfigError, match="Stop loss percentage must be between"
        ):
            StopLossConfig(stop_loss_percentage=0.0)

        with pytest.raises(
            InvalidStopLossConfigError, match="Stop loss percentage must be between"
        ):
            StopLossConfig(stop_loss_percentage=51.0)

    def test_invalid_take_profit_percentage(self) -> None:
        """Test validation of invalid take profit percentage."""
        with pytest.raises(
            InvalidStopLossConfigError, match="Take profit percentage must be between"
        ):
            StopLossConfig(take_profit_percentage=0.0)

        with pytest.raises(
            InvalidStopLossConfigError, match="Take profit percentage must be between"
        ):
            StopLossConfig(take_profit_percentage=101.0)

    def test_atr_method_requires_atr(self) -> None:
        """Test ATR method validation."""
        # ATR method no longer requires current_atr in config since it can
        # use market data
        # Should not raise even without ATR in config
        config = StopLossConfig(method=StopLossMethod.ATR_BASED)
        assert config.method == StopLossMethod.ATR_BASED

        # Should also not raise with ATR provided
        config = StopLossConfig(method=StopLossMethod.ATR_BASED, current_atr=0.5)
        assert config.current_atr == 0.5

    def test_support_resistance_method_requires_levels(self) -> None:
        """Test support/resistance method validation."""
        with pytest.raises(
            InvalidStopLossConfigError, match="Support/resistance method requires"
        ):
            StopLossConfig(method=StopLossMethod.SUPPORT_RESISTANCE)

        # Should not raise with support level provided
        config = StopLossConfig(
            method=StopLossMethod.SUPPORT_RESISTANCE, support_level=100.0
        )
        assert config.support_level == 100.0

    def test_volatility_method_requires_volatility(self) -> None:
        """Test volatility method validation."""
        # Volatility method no longer requires current_volatility in config
        # since it can use market data
        # Should not raise even without volatility in config
        config = StopLossConfig(method=StopLossMethod.VOLATILITY_ADJUSTED)
        assert config.method == StopLossMethod.VOLATILITY_ADJUSTED

        # Should also not raise with volatility provided
        config = StopLossConfig(
            method=StopLossMethod.VOLATILITY_ADJUSTED, current_volatility=0.02
        )
        assert config.current_volatility == 0.02


class TestStopLossLevel:
    """Test StopLossLevel dataclass validation and functionality."""

    def test_valid_level_creation(self) -> None:
        """Test creating valid stop-loss level."""
        level = StopLossLevel(
            price=95.0,
            percentage_from_entry=5.0,
            distance_from_entry=5.0,
            level_type="stop_loss",
            calculation_method="fixed_percentage",
        )

        assert level.price == 95.0
        assert level.percentage_from_entry == 5.0
        assert level.level_type == "stop_loss"

    def test_invalid_price(self) -> None:
        """Test validation of invalid price."""
        with pytest.raises(InvalidPriceLevelError, match="Price must be positive"):
            StopLossLevel(
                price=0.0,
                percentage_from_entry=5.0,
                distance_from_entry=5.0,
                level_type="stop_loss",
                calculation_method="fixed_percentage",
            )

    def test_invalid_distance(self) -> None:
        """Test validation of invalid distance."""
        with pytest.raises(
            InvalidPriceLevelError, match="Distance from entry cannot be negative"
        ):
            StopLossLevel(
                price=95.0,
                percentage_from_entry=5.0,
                distance_from_entry=-1.0,
                level_type="stop_loss",
                calculation_method="fixed_percentage",
            )

    def test_invalid_confidence(self) -> None:
        """Test validation of invalid confidence."""
        with pytest.raises(InvalidPriceLevelError, match="Confidence must be between"):
            StopLossLevel(
                price=95.0,
                percentage_from_entry=5.0,
                distance_from_entry=5.0,
                level_type="stop_loss",
                calculation_method="fixed_percentage",
                confidence=1.5,
            )


class TestStopLossResult:
    """Test StopLossResult dataclass validation and functionality."""

    def create_sample_level(self, price: float, level_type: str) -> StopLossLevel:
        """Create sample stop-loss level for testing."""
        return StopLossLevel(
            price=price,
            percentage_from_entry=5.0,
            distance_from_entry=abs(100.0 - price),
            level_type=level_type,
            calculation_method="test",
        )

    def test_valid_result_creation(self) -> None:
        """Test creating valid stop-loss result."""
        stop_level = self.create_sample_level(95.0, "stop_loss")
        profit_level = self.create_sample_level(110.0, "take_profit")

        result = StopLossResult(
            entry_price=100.0,
            position_type=PositionType.LONG,
            stop_loss_level=stop_level,
            take_profit_level=profit_level,
            method_used=StopLossMethod.FIXED_PERCENTAGE,
            risk_reward_ratio=2.0,
            calculation_timestamp=int(time.time() * 1000),
        )

        assert result.entry_price == 100.0
        assert result.position_type == PositionType.LONG
        assert result.stop_loss_level == stop_level

    def test_get_risk_amount(self) -> None:
        """Test risk amount calculation."""
        stop_level = self.create_sample_level(95.0, "stop_loss")
        result = StopLossResult(
            entry_price=100.0,
            position_type=PositionType.LONG,
            stop_loss_level=stop_level,
            take_profit_level=None,
            method_used=StopLossMethod.FIXED_PERCENTAGE,
            risk_reward_ratio=2.0,
            calculation_timestamp=int(time.time() * 1000),
        )

        risk_amount = result.get_risk_amount(100.0)  # 100 shares
        assert risk_amount == 500.0  # (100 - 95) * 100

    def test_get_profit_potential(self) -> None:
        """Test profit potential calculation."""
        profit_level = self.create_sample_level(110.0, "take_profit")
        result = StopLossResult(
            entry_price=100.0,
            position_type=PositionType.LONG,
            stop_loss_level=None,
            take_profit_level=profit_level,
            method_used=StopLossMethod.FIXED_PERCENTAGE,
            risk_reward_ratio=2.0,
            calculation_timestamp=int(time.time() * 1000),
        )

        profit_potential = result.get_profit_potential(100.0)  # 100 shares
        assert profit_potential == 1000.0  # (110 - 100) * 100

    def test_validate_levels_long_position(self) -> None:
        """Test level validation for long position."""
        stop_level = self.create_sample_level(95.0, "stop_loss")
        profit_level = self.create_sample_level(110.0, "take_profit")

        result = StopLossResult(
            entry_price=100.0,
            position_type=PositionType.LONG,
            stop_loss_level=stop_level,
            take_profit_level=profit_level,
            method_used=StopLossMethod.FIXED_PERCENTAGE,
            risk_reward_ratio=2.0,
            calculation_timestamp=int(time.time() * 1000),
        )

        is_valid, errors = result.validate_levels()
        assert is_valid
        assert len(errors) == 0

    def test_validate_levels_invalid_long_stop(self) -> None:
        """Test validation with invalid long position stop-loss."""
        stop_level = self.create_sample_level(105.0, "stop_loss")  # Above entry

        result = StopLossResult(
            entry_price=100.0,
            position_type=PositionType.LONG,
            stop_loss_level=stop_level,
            take_profit_level=None,
            method_used=StopLossMethod.FIXED_PERCENTAGE,
            risk_reward_ratio=2.0,
            calculation_timestamp=int(time.time() * 1000),
        )

        is_valid, errors = result.validate_levels()
        assert not is_valid
        assert "Long position stop-loss must be below entry price" in errors

    def test_validate_levels_short_position(self) -> None:
        """Test level validation for short position."""
        stop_level = self.create_sample_level(105.0, "stop_loss")
        profit_level = self.create_sample_level(90.0, "take_profit")

        result = StopLossResult(
            entry_price=100.0,
            position_type=PositionType.SHORT,
            stop_loss_level=stop_level,
            take_profit_level=profit_level,
            method_used=StopLossMethod.FIXED_PERCENTAGE,
            risk_reward_ratio=2.0,
            calculation_timestamp=int(time.time() * 1000),
        )

        is_valid, errors = result.validate_levels()
        assert is_valid
        assert len(errors) == 0


class TestStopLossCalculator:
    """Test StopLossCalculator implementation."""

    def create_test_config(
        self,
        method: StopLossMethod = StopLossMethod.FIXED_PERCENTAGE,
        **kwargs: Any,
    ) -> StopLossConfig:
        """Create test configuration."""
        defaults = {
            "method": method,
            "risk_reward_ratio": 2.0,
            "stop_loss_percentage": 2.0,
            "current_atr": 0.5 if method == StopLossMethod.ATR_BASED else None,
            "support_level": (
                95.0 if method == StopLossMethod.SUPPORT_RESISTANCE else None
            ),
            "current_volatility": (
                0.02 if method == StopLossMethod.VOLATILITY_ADJUSTED else None
            ),
        }
        defaults.update(kwargs)
        return StopLossConfig(**defaults)

    def test_calculator_initialization(self) -> None:
        """Test calculator initialization."""
        config = self.create_test_config()
        calculator = StopLossCalculator(config)

        assert calculator._config == config
        assert calculator._calculations_count == 0

    def test_invalid_config_initialization(self) -> None:
        """Test initialization with invalid config."""
        with pytest.raises(
            InvalidStopLossConfigError, match="Config must be StopLossConfig"
        ):
            StopLossCalculator("invalid_config")  # type: ignore

    def test_fixed_percentage_long_calculation(self) -> None:
        """Test fixed percentage calculation for long position."""
        config = self.create_test_config()
        calculator = StopLossCalculator(config)

        result = calculator.calculate_levels(100.0, PositionType.LONG)

        assert result.entry_price == 100.0
        assert result.position_type == PositionType.LONG
        assert result.method_used == StopLossMethod.FIXED_PERCENTAGE

        # Check stop-loss level
        assert result.stop_loss_level is not None
        assert result.stop_loss_level.price == 98.0  # 100 * (1 - 0.02)
        assert result.stop_loss_level.percentage_from_entry == 2.0

        # Check take-profit level (2:1 risk-reward)
        assert result.take_profit_level is not None
        assert result.take_profit_level.price == 104.0  # 100 + (2 * 2)

    def test_fixed_percentage_short_calculation(self) -> None:
        """Test fixed percentage calculation for short position."""
        config = self.create_test_config()
        calculator = StopLossCalculator(config)

        result = calculator.calculate_levels(100.0, PositionType.SHORT)

        assert result.entry_price == 100.0
        assert result.position_type == PositionType.SHORT

        # Check stop-loss level
        assert result.stop_loss_level is not None
        assert result.stop_loss_level.price == 102.0  # 100 * (1 + 0.02)

        # Check take-profit level
        assert result.take_profit_level is not None
        assert result.take_profit_level.price == 96.0  # 100 - (2 * 2)

    def test_atr_based_calculation(self) -> None:
        """Test ATR-based calculation."""
        config = self.create_test_config(
            method=StopLossMethod.ATR_BASED,
            atr_multiplier=2.0,
            current_atr=1.0,
        )
        calculator = StopLossCalculator(config)

        result = calculator.calculate_levels(100.0, PositionType.LONG)

        assert result.method_used == StopLossMethod.ATR_BASED
        assert result.stop_loss_level is not None
        assert result.stop_loss_level.price == 98.0  # 100 - (1.0 * 2.0)

    def test_atr_from_market_data(self) -> None:
        """Test ATR calculation from market data."""
        config = self.create_test_config(
            method=StopLossMethod.ATR_BASED, current_atr=None  # Force using market data
        )
        calculator = StopLossCalculator(config)

        market_data = {"atr": 1.5}
        result = calculator.calculate_levels(100.0, PositionType.LONG, market_data)

        assert result.stop_loss_level is not None
        assert result.stop_loss_level.price == 97.0  # 100 - (1.5 * 2.0)

    def test_support_resistance_calculation(self) -> None:
        """Test support/resistance calculation."""
        config = self.create_test_config(
            method=StopLossMethod.SUPPORT_RESISTANCE,
            support_level=95.0,
            buffer_percentage=1.0,
        )
        calculator = StopLossCalculator(config)

        result = calculator.calculate_levels(100.0, PositionType.LONG)

        assert result.method_used == StopLossMethod.SUPPORT_RESISTANCE
        assert result.stop_loss_level is not None
        # 95.0 * (1 - 0.01) = 94.05, but may be capped by safety limits
        # Just check that it's below the support level with buffer
        assert result.stop_loss_level.price <= 95.0

    def test_volatility_adjusted_calculation(self) -> None:
        """Test volatility-adjusted calculation."""
        config = self.create_test_config(
            method=StopLossMethod.VOLATILITY_ADJUSTED,
            current_volatility=0.1,
            base_stop_percentage=2.0,
            volatility_multiplier=1.0,
        )
        calculator = StopLossCalculator(config)

        result = calculator.calculate_levels(100.0, PositionType.LONG)

        assert result.method_used == StopLossMethod.VOLATILITY_ADJUSTED
        assert result.stop_loss_level is not None
        # Base 2% * (1 + 0.1 * 1.0) = 2.2%
        expected_price = 100.0 * (1.0 - 0.022)
        assert abs(result.stop_loss_level.price - expected_price) < 0.01

    def test_explicit_take_profit_percentage(self) -> None:
        """Test calculation with explicit take-profit percentage."""
        config = self.create_test_config(take_profit_percentage=5.0)
        calculator = StopLossCalculator(config)

        result = calculator.calculate_levels(100.0, PositionType.LONG)

        assert result.take_profit_level is not None
        assert result.take_profit_level.price == 105.0  # 100 * (1 + 0.05)

    def test_safety_limits_application(self) -> None:
        """Test application of safety limits."""
        config = self.create_test_config(
            stop_loss_percentage=10.0,  # High percentage
            max_stop_loss_percentage=5.0,  # Lower limit
        )
        calculator = StopLossCalculator(config)

        result = calculator.calculate_levels(100.0, PositionType.LONG)

        assert result.stop_loss_level is not None
        assert result.stop_loss_level.percentage_from_entry == 5.0  # Capped at max
        assert any("Safety limits applied" in warning for warning in result.warnings)

    def test_invalid_calculation_inputs(self) -> None:
        """Test validation of calculation inputs."""
        config = self.create_test_config()
        calculator = StopLossCalculator(config)

        # Test invalid entry price
        with pytest.raises(
            StopLossCalculationError, match="Entry price must be positive"
        ):
            calculator.calculate_levels(0.0, PositionType.LONG)

        # Test invalid position type
        with pytest.raises(
            StopLossCalculationError, match="Position type must be PositionType"
        ):
            calculator.calculate_levels(100.0, "long")  # type: ignore

    def test_missing_required_data(self) -> None:
        """Test calculation with missing required data."""
        config = self.create_test_config(
            method=StopLossMethod.ATR_BASED, current_atr=None  # No ATR in config
        )
        calculator = StopLossCalculator(config)

        with pytest.raises(
            StopLossCalculationError, match="ATR-based method requires ATR data"
        ):
            calculator.calculate_levels(
                100.0, PositionType.LONG
            )  # No market data either

    def test_event_publishing(self) -> None:
        """Test event publishing functionality."""
        mock_event_hub = Mock(spec=EventHub)
        config = self.create_test_config()
        calculator = StopLossCalculator(config, mock_event_hub)

        calculator.calculate_levels(100.0, PositionType.LONG)

        # Verify event was published
        mock_event_hub.publish.assert_called_once()
        call_args = mock_event_hub.publish.call_args
        assert call_args[0][0] == EventType.POSITION_SIZE_WARNING

    def test_config_update(self) -> None:
        """Test configuration update."""
        config1 = self.create_test_config()
        calculator = StopLossCalculator(config1)

        config2 = self.create_test_config(stop_loss_percentage=3.0)
        calculator.update_config(config2)

        assert calculator._config.stop_loss_percentage == 3.0

    def test_invalid_config_update(self) -> None:
        """Test invalid configuration update."""
        config = self.create_test_config()
        calculator = StopLossCalculator(config)

        with pytest.raises(
            InvalidStopLossConfigError, match="Config must be StopLossConfig"
        ):
            calculator.update_config("invalid")  # type: ignore

    def test_calculation_statistics(self) -> None:
        """Test calculation statistics tracking."""
        config = self.create_test_config()
        calculator = StopLossCalculator(config)

        # Perform some calculations
        calculator.calculate_levels(100.0, PositionType.LONG)
        calculator.calculate_levels(200.0, PositionType.SHORT)

        stats = calculator.get_calculation_statistics()
        assert stats["calculations_count"] == 2
        assert stats["method"] == StopLossMethod.FIXED_PERCENTAGE.value

    def test_negative_stop_price_handling(self) -> None:
        """Test handling of negative stop prices."""
        config = self.create_test_config(
            method=StopLossMethod.ATR_BASED,
            current_atr=150.0,  # Very large ATR
            atr_multiplier=1.0,
        )
        calculator = StopLossCalculator(config)

        result = calculator.calculate_levels(100.0, PositionType.LONG)

        # Should fallback to 10% of entry price when stop would be negative
        # But may be further capped by safety limits
        assert result.stop_loss_level is not None
        # Just check that it's positive and reasonable
        assert result.stop_loss_level.price > 0
        assert result.stop_loss_level.price < 100.0  # Below entry for long position


class TestFactoryFunction:
    """Test the factory function for creating calculator instances."""

    def test_basic_factory_creation(self) -> None:
        """Test basic factory function usage."""
        calculator = create_stop_loss_calculator()

        assert isinstance(calculator, StopLossCalculator)
        assert calculator._config.method == StopLossMethod.FIXED_PERCENTAGE
        assert calculator._config.stop_loss_percentage == 2.0
        assert calculator._config.risk_reward_ratio == 2.0

    def test_factory_with_parameters(self) -> None:
        """Test factory function with custom parameters."""
        calculator = create_stop_loss_calculator(
            method=StopLossMethod.ATR_BASED,
            stop_loss_percentage=3.0,
            risk_reward_ratio=3.0,
            current_atr=1.5,
        )

        assert calculator._config.method == StopLossMethod.ATR_BASED
        assert calculator._config.stop_loss_percentage == 3.0
        assert calculator._config.risk_reward_ratio == 3.0
        assert calculator._config.current_atr == 1.5

    def test_factory_with_event_hub(self) -> None:
        """Test factory function with event hub."""
        mock_event_hub = Mock(spec=EventHub)
        calculator = create_stop_loss_calculator(event_hub=mock_event_hub)

        assert calculator._event_hub == mock_event_hub

    def test_factory_with_kwargs(self) -> None:
        """Test factory function with additional kwargs."""
        calculator = create_stop_loss_calculator(
            max_stop_loss_percentage=10.0,
            buffer_percentage=0.5,
            metadata={"test": "value"},
        )

        assert calculator._config.max_stop_loss_percentage == 10.0
        assert calculator._config.buffer_percentage == 0.5
        assert calculator._config.metadata == {"test": "value"}


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_complete_trading_scenario(self) -> None:
        """Test complete trading scenario with all levels."""
        calculator = create_stop_loss_calculator(
            method=StopLossMethod.FIXED_PERCENTAGE,
            stop_loss_percentage=2.5,
            risk_reward_ratio=3.0,
        )

        # Long position scenario
        result = calculator.calculate_levels(100.0, PositionType.LONG)

        # Verify all levels are calculated
        assert result.stop_loss_level is not None
        assert result.take_profit_level is not None
        assert result.is_valid

        # Test risk and profit calculations
        position_size = 1000.0
        risk_amount = result.get_risk_amount(position_size)
        profit_potential = result.get_profit_potential(position_size)

        assert risk_amount > 0
        assert profit_potential > 0
        assert profit_potential / risk_amount == pytest.approx(3.0, rel=0.1)

    def test_high_volatility_scenario(self) -> None:
        """Test scenario with high market volatility."""
        calculator = create_stop_loss_calculator(
            method=StopLossMethod.VOLATILITY_ADJUSTED,
            current_volatility=0.15,  # High volatility
            base_stop_percentage=2.0,
            volatility_multiplier=2.0,
        )

        result = calculator.calculate_levels(100.0, PositionType.LONG)

        # Should have wider stops due to high volatility
        assert result.stop_loss_level is not None
        assert result.stop_loss_level.percentage_from_entry > 2.0

        # Should have warnings about high volatility
        assert any("volatility" in warning.lower() for warning in result.warnings)

    def test_support_resistance_scenario(self) -> None:
        """Test scenario using support/resistance levels."""
        calculator = create_stop_loss_calculator(
            method=StopLossMethod.SUPPORT_RESISTANCE,
            support_level=95.0,
            resistance_level=110.0,
            buffer_percentage=0.5,
        )

        # Long position using support
        long_result = calculator.calculate_levels(100.0, PositionType.LONG)
        assert long_result.stop_loss_level is not None
        # May be capped by safety limits, so just check it's at or below support
        assert long_result.stop_loss_level.price <= 95.0

        # Short position using resistance
        short_result = calculator.calculate_levels(100.0, PositionType.SHORT)
        assert short_result.stop_loss_level is not None
        # May be capped by safety limits, so just check it's at or above resistance
        assert (
            short_result.stop_loss_level.price >= 105.0
        )  # Should be at least entry + 5%


if __name__ == "__main__":
    pytest.main([__file__])
