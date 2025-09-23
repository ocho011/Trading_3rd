#!/usr/bin/env python3
"""
Unit tests for the Risk Assessment System.

This test suite validates the comprehensive risk assessment functionality
including configuration validation, risk factor calculations, and event integration.
"""

import pytest
import time
import sys
from pathlib import Path
from unittest.mock import Mock, patch

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
    RiskAssessmentResult,
    RiskFactorResult,
    RiskFactor,
    RiskLevel,
    RiskAssessmentError,
    InvalidRiskConfigError,
    RiskCalculationError,
    create_risk_assessor,
)


class TestRiskAssessmentConfig:
    """Test risk assessment configuration validation."""

    def test_valid_config_creation(self):
        """Test creating valid configuration."""
        config = RiskAssessmentConfig(
            min_confidence_threshold=0.6,
            high_confidence_threshold=0.8,
            volatility_threshold_high=0.05,
            max_position_concentration=0.3,
        )

        assert config.min_confidence_threshold == 0.6
        assert config.high_confidence_threshold == 0.8
        assert config.volatility_threshold_high == 0.05
        assert config.max_position_concentration == 0.3

    def test_invalid_confidence_threshold(self):
        """Test invalid confidence threshold validation."""
        with pytest.raises(InvalidRiskConfigError):
            RiskAssessmentConfig(min_confidence_threshold=1.5)

        with pytest.raises(InvalidRiskConfigError):
            RiskAssessmentConfig(min_confidence_threshold=-0.1)

        with pytest.raises(InvalidRiskConfigError):
            RiskAssessmentConfig(
                min_confidence_threshold=0.8,
                high_confidence_threshold=0.6  # min > high
            )

    def test_invalid_risk_multipliers(self):
        """Test invalid risk multiplier validation."""
        with pytest.raises(InvalidRiskConfigError):
            RiskAssessmentConfig(min_risk_multiplier=0.0)

        with pytest.raises(InvalidRiskConfigError):
            RiskAssessmentConfig(
                min_risk_multiplier=2.0,
                max_risk_multiplier=1.0  # min > max
            )


class TestRiskFactorResult:
    """Test risk factor result validation."""

    def test_valid_factor_result(self):
        """Test creating valid factor result."""
        result = RiskFactorResult(
            factor=RiskFactor.SIGNAL_QUALITY,
            risk_score=0.5,
            risk_multiplier=1.2,
            confidence=0.8,
        )

        assert result.factor == RiskFactor.SIGNAL_QUALITY
        assert result.risk_score == 0.5
        assert result.risk_multiplier == 1.2
        assert result.confidence == 0.8

    def test_invalid_risk_score(self):
        """Test invalid risk score validation."""
        with pytest.raises(RiskCalculationError):
            RiskFactorResult(
                factor=RiskFactor.SIGNAL_QUALITY,
                risk_score=1.5,  # > 1.0
                risk_multiplier=1.0,
                confidence=0.8,
            )

        with pytest.raises(RiskCalculationError):
            RiskFactorResult(
                factor=RiskFactor.SIGNAL_QUALITY,
                risk_score=-0.1,  # < 0.0
                risk_multiplier=1.0,
                confidence=0.8,
            )

    def test_invalid_risk_multiplier(self):
        """Test invalid risk multiplier validation."""
        with pytest.raises(RiskCalculationError):
            RiskFactorResult(
                factor=RiskFactor.SIGNAL_QUALITY,
                risk_score=0.5,
                risk_multiplier=0.0,  # <= 0
                confidence=0.8,
            )


class TestRiskAssessmentResult:
    """Test risk assessment result validation."""

    def create_sample_signal(self):
        """Create sample trading signal for testing."""
        return TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            price=45000.0,
            timestamp=int(time.time() * 1000),
            strategy_name="test_strategy",
            confidence=0.8,
            reasoning="Test signal",
        )

    def test_valid_assessment_result(self):
        """Test creating valid assessment result."""
        signal = self.create_sample_signal()

        result = RiskAssessmentResult(
            signal=signal,
            overall_risk_level=RiskLevel.MODERATE,
            overall_risk_multiplier=1.2,
            position_size_adjustment=0.83,
            stop_loss_adjustment=None,
            confidence=0.8,
            assessment_timestamp=int(time.time() * 1000),
        )

        assert result.signal == signal
        assert result.overall_risk_level == RiskLevel.MODERATE
        assert result.overall_risk_multiplier == 1.2
        assert result.position_size_adjustment == 0.83

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
                risk_score=0.7,
                risk_multiplier=1.4,
                confidence=0.8,
            ),
        ]

        result = RiskAssessmentResult(
            signal=signal,
            overall_risk_level=RiskLevel.MODERATE,
            overall_risk_multiplier=1.2,
            position_size_adjustment=0.83,
            stop_loss_adjustment=None,
            confidence=0.8,
            assessment_timestamp=int(time.time() * 1000),
            factor_results=factor_results,
        )

        summary = result.get_risk_factors_summary()
        assert summary[RiskFactor.SIGNAL_QUALITY.value] == 0.3
        assert summary[RiskFactor.MARKET_VOLATILITY.value] == 0.7

    def test_has_high_risk_factors(self):
        """Test high risk factors detection."""
        signal = self.create_sample_signal()

        # Test with high risk factor
        high_risk_factors = [
            RiskFactorResult(
                factor=RiskFactor.MARKET_VOLATILITY,
                risk_score=0.8,  # High risk
                risk_multiplier=1.5,
                confidence=0.9,
            ),
        ]

        result = RiskAssessmentResult(
            signal=signal,
            overall_risk_level=RiskLevel.HIGH,
            overall_risk_multiplier=1.5,
            position_size_adjustment=0.67,
            stop_loss_adjustment=None,
            confidence=0.8,
            assessment_timestamp=int(time.time() * 1000),
            factor_results=high_risk_factors,
        )

        assert result.has_high_risk_factors() is True

        # Test with low risk factors
        low_risk_factors = [
            RiskFactorResult(
                factor=RiskFactor.SIGNAL_QUALITY,
                risk_score=0.2,  # Low risk
                risk_multiplier=1.1,
                confidence=0.9,
            ),
        ]

        result.factor_results = low_risk_factors
        assert result.has_high_risk_factors() is False


class TestRiskAssessor:
    """Test risk assessor functionality."""

    def create_sample_signal(self, confidence=0.8, strength=SignalStrength.STRONG):
        """Create sample trading signal."""
        return TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=strength,
            price=45000.0,
            timestamp=int(time.time() * 1000),
            strategy_name="test_strategy",
            confidence=confidence,
            reasoning="Test signal",
        )

    def create_sample_market_data(self, volatility=0.03):
        """Create sample market data."""
        return MarketData(
            symbol="BTCUSDT",
            timestamp=int(time.time() * 1000),
            price=45000.0,
            volume=1000.0,
            source="test",
            data_type="ticker",
            metadata={"volatility": volatility},
        )

    def test_risk_assessor_initialization(self):
        """Test risk assessor initialization."""
        config = RiskAssessmentConfig()
        event_hub = EventHub()

        assessor = RiskAssessor(config, event_hub)

        assert assessor._config == config
        assert assessor._event_hub == event_hub
        assert assessor._assessments_count == 0

    def test_invalid_initialization(self):
        """Test invalid initialization parameters."""
        with pytest.raises(InvalidRiskConfigError):
            RiskAssessor("invalid_config")

    def test_basic_risk_assessment(self):
        """Test basic risk assessment functionality."""
        config = RiskAssessmentConfig()
        assessor = RiskAssessor(config)

        signal = self.create_sample_signal()
        result = assessor.assess_risk(signal)

        assert isinstance(result, RiskAssessmentResult)
        assert result.signal == signal
        assert isinstance(result.overall_risk_level, RiskLevel)
        assert result.overall_risk_multiplier > 0
        assert result.position_size_adjustment > 0
        assert 0 <= result.confidence <= 1

    def test_signal_quality_assessment(self):
        """Test signal quality risk factor assessment."""
        config = RiskAssessmentConfig(
            enable_signal_quality_check=True,
            enable_volatility_check=False,
            enable_concentration_check=False,
            enable_time_context_check=False,
            enable_strategy_check=False,
        )
        assessor = RiskAssessor(config)

        # Test high confidence signal
        high_confidence_signal = self.create_sample_signal(
            confidence=0.9, strength=SignalStrength.VERY_STRONG
        )
        result = assessor.assess_risk(high_confidence_signal)

        # Should have low risk due to high confidence and strong signal
        assert result.overall_risk_level in [RiskLevel.VERY_LOW, RiskLevel.LOW]

        # Test low confidence signal
        low_confidence_signal = self.create_sample_signal(
            confidence=0.4, strength=SignalStrength.WEAK
        )
        result = assessor.assess_risk(low_confidence_signal)

        # Should have higher risk due to low confidence and weak signal
        assert result.overall_risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]

    def test_volatility_assessment(self):
        """Test market volatility risk factor assessment."""
        config = RiskAssessmentConfig(
            enable_signal_quality_check=False,
            enable_volatility_check=True,
            enable_concentration_check=False,
            enable_time_context_check=False,
            enable_strategy_check=False,
        )
        assessor = RiskAssessor(config)

        signal = self.create_sample_signal()

        # Test low volatility
        low_vol_market_data = self.create_sample_market_data(volatility=0.01)
        result = assessor.assess_risk(signal, low_vol_market_data)

        volatility_factor = next(
            f for f in result.factor_results
            if f.factor == RiskFactor.MARKET_VOLATILITY
        )
        assert volatility_factor.risk_score <= 0.3  # Low risk for low volatility

        # Test high volatility
        high_vol_market_data = self.create_sample_market_data(volatility=0.1)
        result = assessor.assess_risk(signal, high_vol_market_data)

        volatility_factor = next(
            f for f in result.factor_results
            if f.factor == RiskFactor.MARKET_VOLATILITY
        )
        assert volatility_factor.risk_score >= 0.7  # High risk for high volatility

    def test_position_concentration_assessment(self):
        """Test position concentration risk factor assessment."""
        config = RiskAssessmentConfig(
            enable_signal_quality_check=False,
            enable_volatility_check=False,
            enable_concentration_check=True,
            enable_time_context_check=False,
            enable_strategy_check=False,
            max_position_concentration=0.3,
        )
        assessor = RiskAssessor(config)

        signal = self.create_sample_signal()

        # Test low concentration
        low_concentration_portfolio = {
            "total_value": 100000.0,
            "positions": {"BTCUSDT": {"value": 10000.0}},  # 10% concentration
        }
        result = assessor.assess_risk(signal, portfolio_context=low_concentration_portfolio)

        concentration_factor = next(
            f for f in result.factor_results
            if f.factor == RiskFactor.POSITION_CONCENTRATION
        )
        assert concentration_factor.risk_score <= 0.5  # Low risk for low concentration

        # Test high concentration
        high_concentration_portfolio = {
            "total_value": 100000.0,
            "positions": {"BTCUSDT": {"value": 40000.0}},  # 40% concentration
        }
        result = assessor.assess_risk(signal, portfolio_context=high_concentration_portfolio)

        concentration_factor = next(
            f for f in result.factor_results
            if f.factor == RiskFactor.POSITION_CONCENTRATION
        )
        assert concentration_factor.risk_score >= 0.8  # High risk for high concentration

    def test_invalid_assessment_inputs(self):
        """Test validation of assessment inputs."""
        config = RiskAssessmentConfig()
        assessor = RiskAssessor(config)

        # Test invalid signal
        with pytest.raises(RiskCalculationError):
            assessor.assess_risk("invalid_signal")

        # Test invalid market data
        signal = self.create_sample_signal()
        with pytest.raises(RiskCalculationError):
            assessor.assess_risk(signal, market_data="invalid_market_data")

        # Test invalid portfolio context
        with pytest.raises(RiskCalculationError):
            assessor.assess_risk(signal, portfolio_context="invalid_context")

    def test_config_update(self):
        """Test configuration update functionality."""
        initial_config = RiskAssessmentConfig(min_confidence_threshold=0.6)
        assessor = RiskAssessor(initial_config)

        new_config = RiskAssessmentConfig(min_confidence_threshold=0.8)
        assessor.update_config(new_config)

        assert assessor._config.min_confidence_threshold == 0.8

    def test_assessment_statistics(self):
        """Test assessment statistics tracking."""
        config = RiskAssessmentConfig()
        assessor = RiskAssessor(config)

        # Initial statistics
        stats = assessor.get_assessment_statistics()
        assert stats["assessments_count"] == 0

        # Perform assessments
        signal = self.create_sample_signal()
        assessor.assess_risk(signal)
        assessor.assess_risk(signal)

        # Check updated statistics
        stats = assessor.get_assessment_statistics()
        assert stats["assessments_count"] == 2

    @patch('trading_bot.risk_management.risk_assessor.EventHub')
    def test_event_publishing(self, mock_event_hub_class):
        """Test event publishing functionality."""
        mock_event_hub = Mock()
        mock_event_hub_class.return_value = mock_event_hub

        config = RiskAssessmentConfig()
        assessor = RiskAssessor(config, mock_event_hub)

        signal = self.create_sample_signal()
        result = assessor.assess_risk(signal)

        # Verify event was published
        assert mock_event_hub.publish.called


class TestRiskAssessorFactory:
    """Test risk assessor factory function."""

    def test_create_risk_assessor_default(self):
        """Test creating risk assessor with default parameters."""
        assessor = create_risk_assessor()

        assert isinstance(assessor, RiskAssessor)
        assert assessor._config.min_confidence_threshold == 0.6
        assert assessor._config.volatility_threshold_high == 0.05
        assert assessor._config.max_position_concentration == 0.3

    def test_create_risk_assessor_custom(self):
        """Test creating risk assessor with custom parameters."""
        event_hub = EventHub()

        assessor = create_risk_assessor(
            min_confidence_threshold=0.7,
            volatility_threshold_high=0.04,
            max_position_concentration=0.25,
            event_hub=event_hub,
            weak_signal_penalty=0.25,
        )

        assert assessor._config.min_confidence_threshold == 0.7
        assert assessor._config.volatility_threshold_high == 0.04
        assert assessor._config.max_position_concentration == 0.25
        assert assessor._config.weak_signal_penalty == 0.25
        assert assessor._event_hub == event_hub


class TestRiskLevelEnum:
    """Test risk level enumeration."""

    def test_risk_level_multipliers(self):
        """Test risk level multiplier properties."""
        assert RiskLevel.VERY_LOW.multiplier == 0.5
        assert RiskLevel.LOW.multiplier == 0.75
        assert RiskLevel.MODERATE.multiplier == 1.0
        assert RiskLevel.HIGH.multiplier == 1.25
        assert RiskLevel.VERY_HIGH.multiplier == 1.5


def run_tests():
    """Run all tests."""
    print("Running Risk Assessment System Tests...")

    # Test configuration
    print("Testing RiskAssessmentConfig...")
    config_tests = TestRiskAssessmentConfig()
    config_tests.test_valid_config_creation()
    config_tests.test_invalid_confidence_threshold()
    config_tests.test_invalid_risk_multipliers()
    print("✅ RiskAssessmentConfig tests passed")

    # Test factor results
    print("Testing RiskFactorResult...")
    factor_tests = TestRiskFactorResult()
    factor_tests.test_valid_factor_result()
    factor_tests.test_invalid_risk_score()
    factor_tests.test_invalid_risk_multiplier()
    print("✅ RiskFactorResult tests passed")

    # Test assessment results
    print("Testing RiskAssessmentResult...")
    result_tests = TestRiskAssessmentResult()
    result_tests.test_valid_assessment_result()
    result_tests.test_get_risk_factors_summary()
    result_tests.test_has_high_risk_factors()
    print("✅ RiskAssessmentResult tests passed")

    # Test risk assessor
    print("Testing RiskAssessor...")
    assessor_tests = TestRiskAssessor()
    assessor_tests.test_risk_assessor_initialization()
    assessor_tests.test_basic_risk_assessment()
    assessor_tests.test_signal_quality_assessment()
    assessor_tests.test_volatility_assessment()
    assessor_tests.test_position_concentration_assessment()
    assessor_tests.test_config_update()
    assessor_tests.test_assessment_statistics()
    print("✅ RiskAssessor tests passed")

    # Test factory
    print("Testing factory function...")
    factory_tests = TestRiskAssessorFactory()
    factory_tests.test_create_risk_assessor_default()
    factory_tests.test_create_risk_assessor_custom()
    print("✅ Factory tests passed")

    # Test enum
    print("Testing RiskLevel enum...")
    enum_tests = TestRiskLevelEnum()
    enum_tests.test_risk_level_multipliers()
    print("✅ RiskLevel tests passed")

    print("\n🎉 All tests passed successfully!")


if __name__ == "__main__":
    try:
        run_tests()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)