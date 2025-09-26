"""
Unit tests for risk assessment system.

Tests all risk assessment methods, configuration validation,
risk factor calculations, and edge cases to ensure reliable
risk management functionality.
"""

import time
from typing import Any, Dict
from unittest.mock import Mock

import pytest

from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.market_data.data_processor import MarketData
from trading_bot.risk_management.risk_assessor import (
    InvalidRiskConfigError,
    RiskAssessmentConfig,
    RiskAssessmentResult,
    RiskAssessor,
    RiskCalculationError,
    RiskFactor,
    RiskFactorResult,
    RiskLevel,
    create_risk_assessor,
)
from trading_bot.strategies.base_strategy import (
    SignalStrength,
    SignalType,
    TradingSignal,
)


class TestRiskAssessmentConfig:
    """Test cases for RiskAssessmentConfig validation."""

    def test_valid_config_creation(self):
        """Test creation of valid configuration."""
        config = RiskAssessmentConfig(
            min_confidence_threshold=0.6,
            high_confidence_threshold=0.8,
            weak_signal_penalty=0.2,
            strong_signal_bonus=0.1,
        )

        assert config.min_confidence_threshold == 0.6
        assert config.high_confidence_threshold == 0.8
        assert config.weak_signal_penalty == 0.2
        assert config.strong_signal_bonus == 0.1
        assert config.volatility_threshold_low == 0.02
        assert config.volatility_threshold_high == 0.05

    def test_default_values(self):
        """Test default configuration values."""
        config = RiskAssessmentConfig()

        assert config.min_confidence_threshold == 0.6
        assert config.high_confidence_threshold == 0.8
        assert config.weak_signal_penalty == 0.2
        assert config.strong_signal_bonus == 0.1
        assert config.volatility_threshold_low == 0.02
        assert config.volatility_threshold_high == 0.05
        assert config.max_position_concentration == 0.3
        assert config.min_risk_multiplier == 0.3
        assert config.max_risk_multiplier == 2.0

    def test_invalid_min_confidence_threshold_negative(self):
        """Test validation with negative min confidence threshold."""
        with pytest.raises(
            InvalidRiskConfigError, match="Min confidence threshold must be 0.0-1.0"
        ):
            RiskAssessmentConfig(min_confidence_threshold=-0.1)

    def test_invalid_min_confidence_threshold_too_high(self):
        """Test validation with min confidence threshold over 1.0."""
        with pytest.raises(
            InvalidRiskConfigError, match="Min confidence threshold must be 0.0-1.0"
        ):
            RiskAssessmentConfig(min_confidence_threshold=1.1)

    def test_invalid_high_confidence_threshold_negative(self):
        """Test validation with negative high confidence threshold."""
        with pytest.raises(
            InvalidRiskConfigError, match="High confidence threshold must be 0.0-1.0"
        ):
            RiskAssessmentConfig(high_confidence_threshold=-0.1)

    def test_invalid_high_confidence_threshold_too_high(self):
        """Test validation with high confidence threshold over 1.0."""
        with pytest.raises(
            InvalidRiskConfigError, match="High confidence threshold must be 0.0-1.0"
        ):
            RiskAssessmentConfig(high_confidence_threshold=1.1)

    def test_invalid_confidence_threshold_order(self):
        """Test validation when min confidence exceeds high confidence."""
        with pytest.raises(
            InvalidRiskConfigError,
            match="Min confidence cannot exceed high confidence threshold",
        ):
            RiskAssessmentConfig(
                min_confidence_threshold=0.8,
                high_confidence_threshold=0.6,
            )

    def test_invalid_min_risk_multiplier_negative(self):
        """Test validation with negative min risk multiplier."""
        with pytest.raises(
            InvalidRiskConfigError, match="Min risk multiplier must be positive"
        ):
            RiskAssessmentConfig(min_risk_multiplier=-0.1)

    def test_invalid_min_risk_multiplier_zero(self):
        """Test validation with zero min risk multiplier."""
        with pytest.raises(
            InvalidRiskConfigError, match="Min risk multiplier must be positive"
        ):
            RiskAssessmentConfig(min_risk_multiplier=0.0)

    def test_invalid_max_risk_multiplier_order(self):
        """Test validation when max risk multiplier is not greater than min."""
        with pytest.raises(
            InvalidRiskConfigError,
            match="Max risk multiplier must exceed min risk multiplier",
        ):
            RiskAssessmentConfig(
                min_risk_multiplier=2.0,
                max_risk_multiplier=1.0,
            )

    def test_valid_edge_case_values(self):
        """Test valid edge case values."""
        config = RiskAssessmentConfig(
            min_confidence_threshold=0.0,
            high_confidence_threshold=1.0,
            min_risk_multiplier=0.1,
            max_risk_multiplier=10.0,
        )

        assert config.min_confidence_threshold == 0.0
        assert config.high_confidence_threshold == 1.0
        assert config.min_risk_multiplier == 0.1
        assert config.max_risk_multiplier == 10.0


class TestRiskFactorResult:
    """Test cases for RiskFactorResult validation."""

    def test_valid_result_creation(self):
        """Test creation of valid risk factor result."""
        result = RiskFactorResult(
            factor=RiskFactor.SIGNAL_QUALITY,
            risk_score=0.5,
            risk_multiplier=1.2,
            confidence=0.8,
            warnings=["Test warning"],
            metadata={"test": "data"},
        )

        assert result.factor == RiskFactor.SIGNAL_QUALITY
        assert result.risk_score == 0.5
        assert result.risk_multiplier == 1.2
        assert result.confidence == 0.8
        assert result.warnings == ["Test warning"]
        assert result.metadata == {"test": "data"}

    def test_default_values(self):
        """Test default values for optional fields."""
        result = RiskFactorResult(
            factor=RiskFactor.MARKET_VOLATILITY,
            risk_score=0.3,
            risk_multiplier=1.0,
            confidence=0.9,
        )

        assert result.warnings == []
        assert result.metadata == {}

    def test_invalid_risk_score_negative(self):
        """Test validation with negative risk score."""
        with pytest.raises(
            RiskCalculationError, match="Risk score must be between 0.0 and 1.0"
        ):
            RiskFactorResult(
                factor=RiskFactor.SIGNAL_QUALITY,
                risk_score=-0.1,
                risk_multiplier=1.0,
                confidence=0.8,
            )

    def test_invalid_risk_score_too_high(self):
        """Test validation with risk score over 1.0."""
        with pytest.raises(
            RiskCalculationError, match="Risk score must be between 0.0 and 1.0"
        ):
            RiskFactorResult(
                factor=RiskFactor.SIGNAL_QUALITY,
                risk_score=1.1,
                risk_multiplier=1.0,
                confidence=0.8,
            )

    def test_invalid_risk_multiplier_negative(self):
        """Test validation with negative risk multiplier."""
        with pytest.raises(
            RiskCalculationError, match="Risk multiplier must be positive"
        ):
            RiskFactorResult(
                factor=RiskFactor.SIGNAL_QUALITY,
                risk_score=0.5,
                risk_multiplier=-0.1,
                confidence=0.8,
            )

    def test_invalid_risk_multiplier_zero(self):
        """Test validation with zero risk multiplier."""
        with pytest.raises(
            RiskCalculationError, match="Risk multiplier must be positive"
        ):
            RiskFactorResult(
                factor=RiskFactor.SIGNAL_QUALITY,
                risk_score=0.5,
                risk_multiplier=0.0,
                confidence=0.8,
            )

    def test_invalid_confidence_negative(self):
        """Test validation with negative confidence."""
        with pytest.raises(
            RiskCalculationError, match="Confidence must be between 0.0 and 1.0"
        ):
            RiskFactorResult(
                factor=RiskFactor.SIGNAL_QUALITY,
                risk_score=0.5,
                risk_multiplier=1.0,
                confidence=-0.1,
            )

    def test_invalid_confidence_too_high(self):
        """Test validation with confidence over 1.0."""
        with pytest.raises(
            RiskCalculationError, match="Confidence must be between 0.0 and 1.0"
        ):
            RiskFactorResult(
                factor=RiskFactor.SIGNAL_QUALITY,
                risk_score=0.5,
                risk_multiplier=1.0,
                confidence=1.1,
            )

    def test_valid_edge_case_values(self):
        """Test valid edge case values."""
        result = RiskFactorResult(
            factor=RiskFactor.SIGNAL_QUALITY,
            risk_score=0.0,
            risk_multiplier=0.1,
            confidence=1.0,
        )

        assert result.risk_score == 0.0
        assert result.risk_multiplier == 0.1
        assert result.confidence == 1.0


class TestRiskAssessmentResult:
    """Test cases for RiskAssessmentResult validation."""

    def create_sample_signal(self) -> TradingSignal:
        """Create a sample trading signal for testing."""
        return TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            price=50000.0,
            timestamp=int(time.time() * 1000),
            strategy_name="test_strategy",
            confidence=0.8,
        )

    def test_valid_result_creation(self):
        """Test creation of valid assessment result."""
        signal = self.create_sample_signal()
        result = RiskAssessmentResult(
            signal=signal,
            overall_risk_level=RiskLevel.MODERATE,
            overall_risk_multiplier=1.2,
            position_size_adjustment=0.8,
            stop_loss_adjustment=None,
            confidence=0.85,
            assessment_timestamp=int(time.time() * 1000),
        )

        assert result.signal == signal
        assert result.overall_risk_level == RiskLevel.MODERATE
        assert result.overall_risk_multiplier == 1.2
        assert result.position_size_adjustment == 0.8
        assert result.stop_loss_adjustment is None
        assert result.confidence == 0.85

    def test_invalid_overall_risk_multiplier_negative(self):
        """Test validation with negative overall risk multiplier."""
        signal = self.create_sample_signal()
        with pytest.raises(
            RiskCalculationError, match="Overall risk multiplier must be positive"
        ):
            RiskAssessmentResult(
                signal=signal,
                overall_risk_level=RiskLevel.MODERATE,
                overall_risk_multiplier=-0.1,
                position_size_adjustment=0.8,
                stop_loss_adjustment=None,
                confidence=0.85,
                assessment_timestamp=int(time.time() * 1000),
            )

    def test_invalid_position_size_adjustment_negative(self):
        """Test validation with negative position size adjustment."""
        signal = self.create_sample_signal()
        with pytest.raises(
            RiskCalculationError, match="Position size adjustment must be positive"
        ):
            RiskAssessmentResult(
                signal=signal,
                overall_risk_level=RiskLevel.MODERATE,
                overall_risk_multiplier=1.2,
                position_size_adjustment=-0.1,
                stop_loss_adjustment=None,
                confidence=0.85,
                assessment_timestamp=int(time.time() * 1000),
            )

    def test_invalid_confidence_range(self):
        """Test validation with invalid confidence."""
        signal = self.create_sample_signal()
        with pytest.raises(
            RiskCalculationError, match="Confidence must be between 0.0 and 1.0"
        ):
            RiskAssessmentResult(
                signal=signal,
                overall_risk_level=RiskLevel.MODERATE,
                overall_risk_multiplier=1.2,
                position_size_adjustment=0.8,
                stop_loss_adjustment=None,
                confidence=1.5,
                assessment_timestamp=int(time.time() * 1000),
            )

    def test_get_risk_factors_summary(self):
        """Test risk factors summary method."""
        signal = self.create_sample_signal()
        factor_results = [
            RiskFactorResult(
                factor=RiskFactor.SIGNAL_QUALITY,
                risk_score=0.3,
                risk_multiplier=1.1,
                confidence=0.9,
            ),
            RiskFactorResult(
                factor=RiskFactor.MARKET_VOLATILITY,
                risk_score=0.6,
                risk_multiplier=1.3,
                confidence=0.8,
            ),
        ]

        result = RiskAssessmentResult(
            signal=signal,
            overall_risk_level=RiskLevel.MODERATE,
            overall_risk_multiplier=1.2,
            position_size_adjustment=0.8,
            stop_loss_adjustment=None,
            confidence=0.85,
            assessment_timestamp=int(time.time() * 1000),
            factor_results=factor_results,
        )

        summary = result.get_risk_factors_summary()
        expected = {
            "signal_quality": 0.3,
            "market_volatility": 0.6,
        }

        assert summary == expected

    def test_has_high_risk_factors_true(self):
        """Test has_high_risk_factors method when true."""
        signal = self.create_sample_signal()
        factor_results = [
            RiskFactorResult(
                factor=RiskFactor.SIGNAL_QUALITY,
                risk_score=0.8,  # High risk
                risk_multiplier=1.5,
                confidence=0.9,
            ),
        ]

        result = RiskAssessmentResult(
            signal=signal,
            overall_risk_level=RiskLevel.HIGH,
            overall_risk_multiplier=1.5,
            position_size_adjustment=0.6,
            stop_loss_adjustment=None,
            confidence=0.85,
            assessment_timestamp=int(time.time() * 1000),
            factor_results=factor_results,
        )

        assert result.has_high_risk_factors() is True

    def test_has_high_risk_factors_false(self):
        """Test has_high_risk_factors method when false."""
        signal = self.create_sample_signal()
        factor_results = [
            RiskFactorResult(
                factor=RiskFactor.SIGNAL_QUALITY,
                risk_score=0.3,  # Low risk
                risk_multiplier=1.1,
                confidence=0.9,
            ),
            RiskFactorResult(
                factor=RiskFactor.MARKET_VOLATILITY,
                risk_score=0.5,  # Moderate risk
                risk_multiplier=1.2,
                confidence=0.8,
            ),
        ]

        result = RiskAssessmentResult(
            signal=signal,
            overall_risk_level=RiskLevel.MODERATE,
            overall_risk_multiplier=1.2,
            position_size_adjustment=0.8,
            stop_loss_adjustment=None,
            confidence=0.85,
            assessment_timestamp=int(time.time() * 1000),
            factor_results=factor_results,
        )

        assert result.has_high_risk_factors() is False


class TestRiskLevel:
    """Test cases for RiskLevel enum functionality."""

    def test_risk_level_multipliers(self):
        """Test risk level multiplier property."""
        assert RiskLevel.VERY_LOW.multiplier == 0.5
        assert RiskLevel.LOW.multiplier == 0.75
        assert RiskLevel.MODERATE.multiplier == 1.0
        assert RiskLevel.HIGH.multiplier == 1.25
        assert RiskLevel.VERY_HIGH.multiplier == 1.5

    def test_risk_level_values(self):
        """Test risk level enum values."""
        assert RiskLevel.VERY_LOW.value == "very_low"
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MODERATE.value == "moderate"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.VERY_HIGH.value == "very_high"


class TestRiskAssessor:
    """Test cases for RiskAssessor implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = RiskAssessmentConfig(
            min_confidence_threshold=0.6,
            high_confidence_threshold=0.8,
            volatility_threshold_low=0.02,
            volatility_threshold_high=0.05,
            max_position_concentration=0.3,
        )
        self.event_hub = Mock(spec=EventHub)
        self.risk_assessor = RiskAssessor(self.config, self.event_hub)

    def create_sample_signal(
        self,
        confidence: float = 0.8,
        strength: SignalStrength = SignalStrength.STRONG,
        strategy_name: str = "test_strategy",
    ) -> TradingSignal:
        """Create a sample trading signal for testing."""
        return TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=strength,
            price=50000.0,
            timestamp=int(time.time() * 1000),
            strategy_name=strategy_name,
            confidence=confidence,
        )

    def create_sample_market_data(self, volatility: float = 0.03) -> MarketData:
        """Create sample market data for testing."""
        return MarketData(
            symbol="BTCUSDT",
            timestamp=int(time.time() * 1000),
            price=50000.0,
            volume=1000.0,
            source="test",
            data_type="ticker",
            metadata={"volatility": volatility},
        )

    def create_sample_portfolio_context(
        self,
        symbol: str = "BTCUSDT",
        current_value: float = 1000.0,
        total_value: float = 10000.0,
    ) -> Dict[str, Any]:
        """Create sample portfolio context for testing."""
        return {
            "positions": {symbol: {"value": current_value}},
            "total_value": total_value,
        }

    def test_initialization(self):
        """Test risk assessor initialization."""
        assert self.risk_assessor._config == self.config
        assert self.risk_assessor._event_hub == self.event_hub
        assert self.risk_assessor._assessments_count == 0
        assert self.risk_assessor._strategy_performance == {}

    def test_invalid_config_type(self):
        """Test initialization with invalid config type."""
        with pytest.raises(
            InvalidRiskConfigError, match="Config must be RiskAssessmentConfig instance"
        ):
            RiskAssessor({"invalid": "config"})

    def test_assess_risk_basic_signal_only(self):
        """Test basic risk assessment with signal only."""
        signal = self.create_sample_signal()

        result = self.risk_assessor.assess_risk(signal)

        assert isinstance(result, RiskAssessmentResult)
        assert result.signal == signal
        assert isinstance(result.overall_risk_level, RiskLevel)
        assert result.overall_risk_multiplier > 0
        assert result.position_size_adjustment > 0
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.factor_results) >= 1  # At least signal quality

    def test_assess_risk_with_market_data(self):
        """Test risk assessment with market data."""
        signal = self.create_sample_signal()
        market_data = self.create_sample_market_data(volatility=0.06)  # High volatility

        result = self.risk_assessor.assess_risk(signal, market_data)

        assert len(result.factor_results) >= 2  # Signal quality + volatility
        # Check for volatility factor
        volatility_factor = next(
            (
                f
                for f in result.factor_results
                if f.factor == RiskFactor.MARKET_VOLATILITY
            ),
            None,
        )
        assert volatility_factor is not None
        assert volatility_factor.metadata["detected_volatility"] == 0.06

    def test_assess_risk_with_portfolio_context(self):
        """Test risk assessment with portfolio context."""
        signal = self.create_sample_signal()
        portfolio_context = self.create_sample_portfolio_context(
            current_value=4000.0, total_value=10000.0  # 40% concentration - high risk
        )

        result = self.risk_assessor.assess_risk(
            signal, portfolio_context=portfolio_context
        )

        # Check for concentration factor
        concentration_factor = next(
            (
                f
                for f in result.factor_results
                if f.factor == RiskFactor.POSITION_CONCENTRATION
            ),
            None,
        )
        assert concentration_factor is not None
        assert concentration_factor.metadata["current_concentration"] == 0.4
        assert any(
            "High position concentration" in warning
            for warning in concentration_factor.warnings
        )

    def test_assess_risk_complete_context(self):
        """Test risk assessment with all context provided."""
        signal = self.create_sample_signal()
        market_data = self.create_sample_market_data()
        portfolio_context = self.create_sample_portfolio_context()

        result = self.risk_assessor.assess_risk(
            signal, market_data=market_data, portfolio_context=portfolio_context
        )

        # Should have all enabled risk factors
        factor_types = {f.factor for f in result.factor_results}
        expected_factors = {
            RiskFactor.SIGNAL_QUALITY,
            RiskFactor.MARKET_VOLATILITY,
            RiskFactor.POSITION_CONCENTRATION,
            RiskFactor.TIME_CONTEXT,
            RiskFactor.STRATEGY_TRACK_RECORD,
        }
        assert factor_types == expected_factors

    def test_signal_quality_assessment_weak_signal(self):
        """Test signal quality assessment with weak signal."""
        signal = self.create_sample_signal(
            confidence=0.5, strength=SignalStrength.WEAK  # Below min threshold
        )

        result = self.risk_assessor.assess_risk(signal)

        signal_quality_factor = next(
            f for f in result.factor_results if f.factor == RiskFactor.SIGNAL_QUALITY
        )
        warnings_text = " ".join(signal_quality_factor.warnings)
        assert (
            "Signal confidence" in warnings_text
            or "Weak signal strength detected" in warnings_text
        )
        assert "Weak signal strength detected" in signal_quality_factor.warnings
        assert signal_quality_factor.risk_score > 0.5  # High risk due to low confidence

    def test_signal_quality_assessment_strong_signal(self):
        """Test signal quality assessment with strong signal."""
        signal = self.create_sample_signal(
            confidence=0.9, strength=SignalStrength.VERY_STRONG  # High confidence
        )

        result = self.risk_assessor.assess_risk(signal)

        signal_quality_factor = next(
            f for f in result.factor_results if f.factor == RiskFactor.SIGNAL_QUALITY
        )
        assert signal_quality_factor.metadata.get("high_confidence") is True
        assert signal_quality_factor.metadata.get("strength_bonus") is not None
        assert signal_quality_factor.risk_score < 0.3  # Low risk due to high confidence

    def test_market_volatility_assessment_low(self):
        """Test market volatility assessment with low volatility."""
        signal = self.create_sample_signal()
        market_data = self.create_sample_market_data(volatility=0.01)  # Low volatility

        result = self.risk_assessor.assess_risk(signal, market_data)

        volatility_factor = next(
            f for f in result.factor_results if f.factor == RiskFactor.MARKET_VOLATILITY
        )
        assert volatility_factor.metadata["volatility_level"] == "low"
        assert volatility_factor.risk_score == 0.2

    def test_market_volatility_assessment_high(self):
        """Test market volatility assessment with high volatility."""
        signal = self.create_sample_signal()
        market_data = self.create_sample_market_data(volatility=0.08)  # High volatility

        result = self.risk_assessor.assess_risk(signal, market_data)

        volatility_factor = next(
            f for f in result.factor_results if f.factor == RiskFactor.MARKET_VOLATILITY
        )
        assert volatility_factor.metadata["volatility_level"] == "high"
        assert volatility_factor.risk_score == 0.8
        assert "High volatility detected" in volatility_factor.warnings[0]

    def test_position_concentration_assessment_safe(self):
        """Test position concentration assessment with safe levels."""
        signal = self.create_sample_signal()
        portfolio_context = self.create_sample_portfolio_context(
            current_value=1000.0, total_value=10000.0  # 10% concentration - safe
        )

        result = self.risk_assessor.assess_risk(
            signal, portfolio_context=portfolio_context
        )

        concentration_factor = next(
            f
            for f in result.factor_results
            if f.factor == RiskFactor.POSITION_CONCENTRATION
        )
        assert concentration_factor.metadata["current_concentration"] == 0.1
        assert len(concentration_factor.warnings) == 0

    def test_time_context_assessment_fresh_signal(self):
        """Test time context assessment with fresh signal."""
        signal = self.create_sample_signal()
        # Signal timestamp is already recent from create_sample_signal

        result = self.risk_assessor.assess_risk(signal)

        time_factor = next(
            f for f in result.factor_results if f.factor == RiskFactor.TIME_CONTEXT
        )
        assert time_factor.metadata["signal_age_minutes"] < 1
        assert time_factor.risk_score == 0.2  # Fresh signal = low risk

    def test_time_context_assessment_stale_signal(self):
        """Test time context assessment with stale signal."""
        # Create signal with old timestamp (2 hours ago)
        old_timestamp = int((time.time() - 7200) * 1000)
        signal = TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            price=50000.0,
            timestamp=old_timestamp,
            strategy_name="test_strategy",
            confidence=0.8,
        )

        result = self.risk_assessor.assess_risk(signal)

        time_factor = next(
            f for f in result.factor_results if f.factor == RiskFactor.TIME_CONTEXT
        )
        assert time_factor.metadata["signal_age_minutes"] > 60
        assert time_factor.risk_score == 0.7  # Stale signal = high risk
        assert "Stale signal" in time_factor.warnings[0]

    def test_strategy_track_record_assessment_new_strategy(self):
        """Test strategy track record assessment for new strategy."""
        signal = self.create_sample_signal(strategy_name="new_strategy")

        result = self.risk_assessor.assess_risk(signal)

        strategy_factor = next(
            f
            for f in result.factor_results
            if f.factor == RiskFactor.STRATEGY_TRACK_RECORD
        )
        assert strategy_factor.metadata["total_trades"] == 0
        assert strategy_factor.risk_score == 0.5  # Neutral for new strategy
        assert "Limited track record" in strategy_factor.warnings[0]

    def test_event_publishing(self):
        """Test event publishing functionality."""
        signal = self.create_sample_signal()

        result = self.risk_assessor.assess_risk(signal)

        # Verify event was published
        self.event_hub.publish.assert_called()
        call_args = self.event_hub.publish.call_args_list[0]

        assert call_args[0][0] == EventType.POSITION_SIZE_WARNING
        event_data = call_args[0][1]
        assert event_data["result"] == result
        assert event_data["symbol"] == signal.symbol
        assert event_data["risk_level"] == result.overall_risk_level.value

    def test_high_risk_event_publishing(self):
        """Test high risk warning event publishing."""
        # Create conditions that lead to high risk
        signal = self.create_sample_signal(
            confidence=0.3, strength=SignalStrength.WEAK  # Very low confidence
        )
        market_data = self.create_sample_market_data(
            volatility=0.10
        )  # Very high volatility

        result = self.risk_assessor.assess_risk(signal, market_data)

        # Should publish both normal and high risk events
        assert self.event_hub.publish.call_count >= 1

        # Check if high risk event was published
        call_args_list = self.event_hub.publish.call_args_list
        event_types = [call[0][0] for call in call_args_list]

        if result.overall_risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
            assert EventType.RISK_LIMIT_EXCEEDED in event_types

    def test_assessments_counter(self):
        """Test assessments counter increment."""
        signal = self.create_sample_signal()
        initial_count = self.risk_assessor._assessments_count

        self.risk_assessor.assess_risk(signal)
        assert self.risk_assessor._assessments_count == initial_count + 1

        self.risk_assessor.assess_risk(signal)
        assert self.risk_assessor._assessments_count == initial_count + 2

    def test_config_update(self):
        """Test configuration update."""
        new_config = RiskAssessmentConfig(
            min_confidence_threshold=0.7,
            high_confidence_threshold=0.9,
            volatility_threshold_high=0.08,
        )

        self.risk_assessor.update_config(new_config)
        assert self.risk_assessor._config == new_config

    def test_config_update_invalid_type(self):
        """Test configuration update with invalid type."""
        with pytest.raises(
            InvalidRiskConfigError, match="Config must be RiskAssessmentConfig instance"
        ):
            self.risk_assessor.update_config({"invalid": "config"})

    def test_get_assessment_statistics(self):
        """Test assessment statistics retrieval."""
        signal = self.create_sample_signal()

        # Perform some assessments
        self.risk_assessor.assess_risk(signal)
        self.risk_assessor.assess_risk(signal)

        stats = self.risk_assessor.get_assessment_statistics()

        assert stats["assessments_count"] == 2
        assert "strategy_performance" in stats
        assert "config_summary" in stats
        assert (
            stats["config_summary"]["min_confidence_threshold"]
            == self.config.min_confidence_threshold
        )

    def test_input_validation_invalid_signal_type(self):
        """Test input validation with invalid signal type."""
        with pytest.raises(
            RiskCalculationError, match="Signal must be TradingSignal instance"
        ):
            self.risk_assessor.assess_risk("invalid_signal")

    def test_input_validation_invalid_market_data_type(self):
        """Test input validation with invalid market data type."""
        signal = self.create_sample_signal()

        with pytest.raises(
            RiskCalculationError, match="Market data must be MarketData instance"
        ):
            self.risk_assessor.assess_risk(signal, market_data="invalid_data")

    def test_input_validation_invalid_portfolio_context_type(self):
        """Test input validation with invalid portfolio context type."""
        signal = self.create_sample_signal()

        with pytest.raises(
            RiskCalculationError, match="Portfolio context must be dictionary"
        ):
            self.risk_assessor.assess_risk(signal, portfolio_context="invalid_context")

    def test_no_risk_factors_fallback(self):
        """Test fallback behavior when no risk factors are assessed."""
        # Create config with all factors disabled
        config = RiskAssessmentConfig(
            enable_signal_quality_check=False,
            enable_volatility_check=False,
            enable_concentration_check=False,
            enable_time_context_check=False,
            enable_strategy_check=False,
        )
        assessor = RiskAssessor(config)
        signal = self.create_sample_signal()

        result = assessor.assess_risk(signal)

        assert result.overall_risk_level == RiskLevel.MODERATE
        assert result.overall_risk_multiplier == config.default_risk_multiplier
        assert "No risk factors assessed" in result.warnings
        assert result.metadata.get("fallback_used") is True


class TestFactoryFunction:
    """Test cases for create_risk_assessor factory function."""

    def test_factory_function_basic(self):
        """Test basic factory function usage."""
        assessor = create_risk_assessor(
            min_confidence_threshold=0.7,
            volatility_threshold_high=0.06,
            max_position_concentration=0.25,
        )

        assert isinstance(assessor, RiskAssessor)
        assert assessor._config.min_confidence_threshold == 0.7
        assert assessor._config.volatility_threshold_high == 0.06
        assert assessor._config.max_position_concentration == 0.25

    def test_factory_function_with_event_hub(self):
        """Test factory function with event hub."""
        event_hub = Mock(spec=EventHub)
        assessor = create_risk_assessor(
            min_confidence_threshold=0.6,
            event_hub=event_hub,
        )

        assert assessor._event_hub == event_hub

    def test_factory_function_with_kwargs(self):
        """Test factory function with additional kwargs."""
        assessor = create_risk_assessor(
            min_confidence_threshold=0.6,
            weak_signal_penalty=0.3,
            strong_signal_bonus=0.15,
            enable_volatility_check=False,
            metadata={"test": "data"},
        )

        assert assessor._config.weak_signal_penalty == 0.3
        assert assessor._config.strong_signal_bonus == 0.15
        assert assessor._config.enable_volatility_check is False
        assert assessor._config.metadata == {"test": "data"}

    def test_factory_function_default_values(self):
        """Test factory function with default values."""
        assessor = create_risk_assessor()

        assert assessor._config.min_confidence_threshold == 0.6
        assert assessor._config.volatility_threshold_high == 0.05
        assert assessor._config.max_position_concentration == 0.3
        assert assessor._event_hub is None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = RiskAssessmentConfig()
        self.risk_assessor = RiskAssessor(self.config)

    def create_sample_signal(
        self,
        confidence: float = 0.8,
        strength: SignalStrength = SignalStrength.STRONG,
        price: float = 50000.0,
    ) -> TradingSignal:
        """Create a sample trading signal for testing."""
        return TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=strength,
            price=price,
            timestamp=int(time.time() * 1000),
            strategy_name="test_strategy",
            confidence=confidence,
        )

    def test_extreme_confidence_values(self):
        """Test with extreme but valid confidence values."""
        # Test minimum confidence
        signal_min = self.create_sample_signal(confidence=0.0)
        result_min = self.risk_assessor.assess_risk(signal_min)
        assert result_min.overall_risk_multiplier >= self.config.min_risk_multiplier

        # Test maximum confidence
        signal_max = self.create_sample_signal(confidence=1.0)
        result_max = self.risk_assessor.assess_risk(signal_max)
        assert result_max.overall_risk_multiplier <= self.config.max_risk_multiplier

    def test_very_small_price_values(self):
        """Test with very small price values."""
        signal = self.create_sample_signal(price=0.001)
        result = self.risk_assessor.assess_risk(signal)

        assert isinstance(result, RiskAssessmentResult)
        assert result.overall_risk_multiplier > 0

    def test_very_large_price_values(self):
        """Test with very large price values."""
        signal = self.create_sample_signal(price=1000000.0)
        result = self.risk_assessor.assess_risk(signal)

        assert isinstance(result, RiskAssessmentResult)
        assert result.overall_risk_multiplier > 0

    def test_zero_portfolio_value_edge_case(self):
        """Test position concentration with zero portfolio value."""
        signal = self.create_sample_signal()
        portfolio_context = {
            "positions": {"BTCUSDT": {"value": 0.0}},
            "total_value": 0.0,  # Edge case
        }

        result = self.risk_assessor.assess_risk(
            signal, portfolio_context=portfolio_context
        )

        concentration_factor = next(
            f
            for f in result.factor_results
            if f.factor == RiskFactor.POSITION_CONCENTRATION
        )
        assert concentration_factor.metadata["current_concentration"] == 0.0

    def test_missing_portfolio_position(self):
        """Test with portfolio context missing the signal's symbol."""
        signal = self.create_sample_signal()
        portfolio_context = {
            "positions": {"ETHUSDT": {"value": 1000.0}},  # Different symbol
            "total_value": 10000.0,
        }

        result = self.risk_assessor.assess_risk(
            signal, portfolio_context=portfolio_context
        )

        concentration_factor = next(
            f
            for f in result.factor_results
            if f.factor == RiskFactor.POSITION_CONCENTRATION
        )
        assert concentration_factor.metadata["current_concentration"] == 0.0

    def test_missing_volatility_in_market_data(self):
        """Test market volatility assessment with missing volatility data."""
        signal = self.create_sample_signal()
        market_data = MarketData(
            symbol="BTCUSDT",
            timestamp=int(time.time() * 1000),
            price=50000.0,
            volume=1000.0,
            source="test",
            data_type="ticker",
            metadata={},  # No volatility data
        )

        result = self.risk_assessor.assess_risk(signal, market_data)

        volatility_factor = next(
            f for f in result.factor_results if f.factor == RiskFactor.MARKET_VOLATILITY
        )
        # Should use default volatility value
        assert volatility_factor.metadata["detected_volatility"] == 0.02

    def test_risk_multiplier_limits_enforcement(self):
        """Test that risk multiplier limits are properly enforced."""
        # Create conditions that would normally exceed limits
        config = RiskAssessmentConfig(
            min_risk_multiplier=0.5,
            max_risk_multiplier=1.5,
        )
        assessor = RiskAssessor(config)

        # Test with high-risk signal
        signal = self.create_sample_signal(
            confidence=0.1, strength=SignalStrength.WEAK  # Very low confidence
        )
        market_data = MarketData(
            symbol="BTCUSDT",
            timestamp=int(time.time() * 1000),
            price=50000.0,
            volume=1000.0,
            source="test",
            data_type="ticker",
            metadata={"volatility": 0.15},  # Very high volatility
        )

        result = assessor.assess_risk(signal, market_data)

        # Risk multiplier should be capped at max limit
        assert (
            config.min_risk_multiplier
            <= result.overall_risk_multiplier
            <= config.max_risk_multiplier
        )

    def test_empty_factor_results_handling(self):
        """Test handling when individual factor assessments return empty results."""
        # Mock a scenario where factor assessment might fail
        signal = self.create_sample_signal()

        # This should still work with the base implementation
        result = self.risk_assessor.assess_risk(signal)

        assert isinstance(result, RiskAssessmentResult)
        assert len(result.factor_results) > 0  # Should have at least signal quality
