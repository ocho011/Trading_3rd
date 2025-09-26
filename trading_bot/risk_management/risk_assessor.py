"""
Risk assessment system for trading signals and position management.

This module provides comprehensive risk assessment capabilities that analyze trading
signals and market conditions to determine appropriate risk adjustments and position
sizing parameters. Follows SOLID principles and integrates with the event-driven
architecture.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.core.logger import get_module_logger
from trading_bot.market_data.data_processor import MarketData
from trading_bot.strategies.base_strategy import SignalStrength, TradingSignal


class RiskAssessmentError(Exception):
    """Base exception for risk assessment errors."""


class InvalidRiskConfigError(RiskAssessmentError):
    """Exception raised for invalid risk assessment configuration."""


class RiskCalculationError(RiskAssessmentError):
    """Exception raised for risk calculation errors."""


class RiskFactor(Enum):
    """Risk factor types for assessment evaluation."""

    SIGNAL_QUALITY = "signal_quality"
    MARKET_VOLATILITY = "market_volatility"
    POSITION_CONCENTRATION = "position_concentration"
    TIME_CONTEXT = "time_context"
    STRATEGY_TRACK_RECORD = "strategy_track_record"
    MARKET_CORRELATION = "market_correlation"


class RiskLevel(Enum):
    """Risk level classifications for assessment results."""

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

    @property
    def multiplier(self) -> float:
        """Get risk multiplier for this level."""
        multipliers = {
            "very_low": 0.5,
            "low": 0.75,
            "moderate": 1.0,
            "high": 1.25,
            "very_high": 1.5,
        }
        return multipliers[self.value]


@dataclass
class RiskAssessmentConfig:
    """Configuration for risk assessment calculations.

    This dataclass defines the parameters and thresholds used for
    risk assessment across different risk factors.
    """

    # Signal quality thresholds
    min_confidence_threshold: float = 0.6
    high_confidence_threshold: float = 0.8
    weak_signal_penalty: float = 0.2
    strong_signal_bonus: float = 0.1

    # Market volatility parameters
    volatility_threshold_low: float = 0.02
    volatility_threshold_high: float = 0.05
    volatility_risk_multiplier: float = 1.5

    # Position concentration limits
    max_position_concentration: float = 0.3
    concentration_penalty_factor: float = 0.15

    # Time-based risk factors
    market_hours_risk_reduction: float = 0.1
    weekend_risk_increase: float = 0.2
    news_event_risk_increase: float = 0.25

    # Strategy performance parameters
    min_strategy_trades: int = 10
    good_win_rate_threshold: float = 0.6
    poor_win_rate_threshold: float = 0.4

    # Risk adjustment limits
    min_risk_multiplier: float = 0.3
    max_risk_multiplier: float = 2.0
    default_risk_multiplier: float = 1.0

    # Enable/disable specific risk factors
    enable_signal_quality_check: bool = True
    enable_volatility_check: bool = True
    enable_concentration_check: bool = True
    enable_time_context_check: bool = True
    enable_strategy_check: bool = True

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not 0.0 <= self.min_confidence_threshold <= 1.0:
            raise InvalidRiskConfigError("Min confidence threshold must be 0.0-1.0")
        if not 0.0 <= self.high_confidence_threshold <= 1.0:
            raise InvalidRiskConfigError("High confidence threshold must be 0.0-1.0")
        if self.min_confidence_threshold > self.high_confidence_threshold:
            raise InvalidRiskConfigError(
                "Min confidence cannot exceed high confidence threshold"
            )
        if self.min_risk_multiplier <= 0:
            raise InvalidRiskConfigError("Min risk multiplier must be positive")
        if self.max_risk_multiplier <= self.min_risk_multiplier:
            raise InvalidRiskConfigError(
                "Max risk multiplier must exceed min risk multiplier"
            )


@dataclass
class RiskFactorResult:
    """Result of individual risk factor assessment."""

    factor: RiskFactor
    risk_score: float
    risk_multiplier: float
    confidence: float
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result after initialization."""
        if not 0.0 <= self.risk_score <= 1.0:
            raise RiskCalculationError("Risk score must be between 0.0 and 1.0")
        if self.risk_multiplier <= 0:
            raise RiskCalculationError("Risk multiplier must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise RiskCalculationError("Confidence must be between 0.0 and 1.0")


@dataclass
class RiskAssessmentResult:
    """Comprehensive risk assessment result with recommendations.

    Contains the overall risk assessment and specific recommendations
    for position sizing and risk management.
    """

    signal: TradingSignal
    overall_risk_level: RiskLevel
    overall_risk_multiplier: float
    position_size_adjustment: float
    stop_loss_adjustment: Optional[float]
    confidence: float
    assessment_timestamp: int
    factor_results: List[RiskFactorResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result after initialization."""
        if self.overall_risk_multiplier <= 0:
            raise RiskCalculationError("Overall risk multiplier must be positive")
        if self.position_size_adjustment <= 0:
            raise RiskCalculationError("Position size adjustment must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise RiskCalculationError("Confidence must be between 0.0 and 1.0")

    def get_risk_factors_summary(self) -> Dict[str, float]:
        """Get summary of risk factor scores.

        Returns:
            Dictionary mapping factor names to risk scores
        """
        return {
            result.factor.value: result.risk_score for result in self.factor_results
        }

    def has_high_risk_factors(self) -> bool:
        """Check if any risk factors indicate high risk.

        Returns:
            True if any factor has risk score > 0.7
        """
        return any(result.risk_score > 0.7 for result in self.factor_results)


class IRiskAssessor(ABC):
    """Abstract interface for risk assessment implementations."""

    @abstractmethod
    def assess_risk(
        self,
        signal: TradingSignal,
        market_data: Optional[MarketData] = None,
        portfolio_context: Optional[Dict[str, Any]] = None,
    ) -> RiskAssessmentResult:
        """Assess risk for a trading signal.

        Args:
            signal: Trading signal to assess
            market_data: Optional current market data
            portfolio_context: Optional portfolio information

        Returns:
            RiskAssessmentResult: Comprehensive risk assessment

        Raises:
            RiskCalculationError: If risk assessment fails
        """

    @abstractmethod
    def update_config(self, config: RiskAssessmentConfig) -> None:
        """Update risk assessment configuration.

        Args:
            config: New risk assessment configuration

        Raises:
            InvalidRiskConfigError: If configuration is invalid
        """


class RiskAssessor(IRiskAssessor):
    """Comprehensive risk assessment implementation.

    This class provides multi-factor risk assessment for trading signals,
    analyzing signal quality, market conditions, and portfolio context to
    determine appropriate risk adjustments.

    Attributes:
        _config: Risk assessment configuration
        _event_hub: Event hub for publishing risk events
        _logger: Logger instance for risk assessment logging
        _assessments_count: Counter for performed assessments
    """

    def __init__(
        self,
        config: RiskAssessmentConfig,
        event_hub: Optional[EventHub] = None,
    ) -> None:
        """Initialize risk assessor with configuration and dependencies.

        Args:
            config: Risk assessment configuration
            event_hub: Optional event hub for event publishing

        Raises:
            InvalidRiskConfigError: If configuration is invalid
        """
        if not isinstance(config, RiskAssessmentConfig):
            raise InvalidRiskConfigError("Config must be RiskAssessmentConfig instance")

        self._config = config
        self._event_hub = event_hub
        self._logger = get_module_logger("risk_assessor")
        self._assessments_count = 0

        # Strategy performance tracking
        self._strategy_performance: Dict[str, Dict[str, Any]] = {}

        self._logger.info("Initialized risk assessor with comprehensive evaluation")

    def assess_risk(
        self,
        signal: TradingSignal,
        market_data: Optional[MarketData] = None,
        portfolio_context: Optional[Dict[str, Any]] = None,
    ) -> RiskAssessmentResult:
        """Assess risk for a trading signal with multi-factor analysis.

        Args:
            signal: Trading signal to assess
            market_data: Optional current market data
            portfolio_context: Optional portfolio information

        Returns:
            RiskAssessmentResult: Comprehensive risk assessment

        Raises:
            RiskCalculationError: If risk assessment fails
        """
        try:
            self._validate_assessment_inputs(signal, market_data, portfolio_context)

            # Assess individual risk factors
            factor_results = self._assess_all_risk_factors(
                signal, market_data, portfolio_context
            )

            # Calculate overall risk
            overall_result = self._calculate_overall_risk(signal, factor_results)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                overall_result, factor_results
            )

            # Create comprehensive result
            result = RiskAssessmentResult(
                signal=signal,
                overall_risk_level=overall_result["risk_level"],
                overall_risk_multiplier=overall_result["risk_multiplier"],
                position_size_adjustment=overall_result["position_adjustment"],
                stop_loss_adjustment=overall_result.get("stop_loss_adjustment"),
                confidence=overall_result["confidence"],
                assessment_timestamp=int(time.time() * 1000),
                factor_results=factor_results,
                warnings=overall_result.get("warnings", []),
                recommendations=recommendations,
                metadata=overall_result.get("metadata", {}),
            )

            # Publish event if event hub available
            if self._event_hub:
                self._publish_risk_assessment_event(result)

            # Update statistics
            self._assessments_count += 1
            self._update_strategy_performance(signal, result)

            self._logger.info(
                f"Risk assessment completed: {signal.symbol} "
                f"risk_level={result.overall_risk_level.value} "
                f"multiplier={result.overall_risk_multiplier:.3f}"
            )

            return result

        except Exception as e:
            self._logger.error(f"Error assessing risk: {e}")
            raise RiskCalculationError(f"Risk assessment failed: {e}")

    def update_config(self, config: RiskAssessmentConfig) -> None:
        """Update risk assessment configuration.

        Args:
            config: New risk assessment configuration

        Raises:
            InvalidRiskConfigError: If configuration is invalid
        """
        if not isinstance(config, RiskAssessmentConfig):
            raise InvalidRiskConfigError("Config must be RiskAssessmentConfig instance")

        self._config = config
        self._logger.info("Updated risk assessment configuration")

    def get_assessment_statistics(self) -> Dict[str, Any]:
        """Get risk assessment statistics.

        Returns:
            Dictionary containing assessment statistics
        """
        return {
            "assessments_count": self._assessments_count,
            "strategy_performance": self._strategy_performance,
            "config_summary": {
                "min_confidence_threshold": self._config.min_confidence_threshold,
                "volatility_threshold_high": self._config.volatility_threshold_high,
                "max_position_concentration": self._config.max_position_concentration,
            },
        }

    def _validate_assessment_inputs(
        self,
        signal: TradingSignal,
        market_data: Optional[MarketData],
        portfolio_context: Optional[Dict[str, Any]],
    ) -> None:
        """Validate inputs for risk assessment.

        Args:
            signal: Trading signal to validate
            market_data: Market data to validate
            portfolio_context: Portfolio context to validate

        Raises:
            RiskCalculationError: If inputs are invalid
        """
        if not isinstance(signal, TradingSignal):
            raise RiskCalculationError("Signal must be TradingSignal instance")

        if market_data and not isinstance(market_data, MarketData):
            raise RiskCalculationError("Market data must be MarketData instance")

        if portfolio_context and not isinstance(portfolio_context, dict):
            raise RiskCalculationError("Portfolio context must be dictionary")

    def _assess_all_risk_factors(
        self,
        signal: TradingSignal,
        market_data: Optional[MarketData],
        portfolio_context: Optional[Dict[str, Any]],
    ) -> List[RiskFactorResult]:
        """Assess all enabled risk factors.

        Args:
            signal: Trading signal to assess
            market_data: Optional market data
            portfolio_context: Optional portfolio context

        Returns:
            List of risk factor assessment results
        """
        factor_results = []

        if self._config.enable_signal_quality_check:
            factor_results.append(self._assess_signal_quality(signal))

        if self._config.enable_volatility_check and market_data:
            factor_results.append(self._assess_market_volatility(signal, market_data))

        if self._config.enable_concentration_check and portfolio_context:
            factor_results.append(
                self._assess_position_concentration(signal, portfolio_context)
            )

        if self._config.enable_time_context_check:
            factor_results.append(self._assess_time_context(signal))

        if self._config.enable_strategy_check:
            factor_results.append(self._assess_strategy_track_record(signal))

        return factor_results

    def _assess_signal_quality(self, signal: TradingSignal) -> RiskFactorResult:
        """Assess signal quality risk factor.

        Args:
            signal: Trading signal to assess

        Returns:
            Risk factor result for signal quality
        """
        warnings = []
        metadata = {}

        # Base risk score from confidence
        confidence_risk = 1.0 - signal.confidence

        # Adjust for signal strength
        strength_adjustment = 0.0
        if signal.strength == SignalStrength.VERY_STRONG:
            strength_adjustment = -self._config.strong_signal_bonus
            metadata["strength_bonus"] = self._config.strong_signal_bonus
        elif signal.strength == SignalStrength.WEAK:
            strength_adjustment = self._config.weak_signal_penalty
            warnings.append("Weak signal strength detected")
            metadata["strength_penalty"] = self._config.weak_signal_penalty

        # Check confidence thresholds
        if signal.confidence < self._config.min_confidence_threshold:
            warnings.append(
                f"Signal confidence {signal.confidence:.3f} below minimum threshold"
            )
        elif signal.confidence >= self._config.high_confidence_threshold:
            metadata["high_confidence"] = True

        # Calculate final risk score
        risk_score = max(0.0, min(1.0, confidence_risk + strength_adjustment))

        # Risk multiplier (higher risk = higher multiplier)
        risk_multiplier = 1.0 + (risk_score * 0.5)

        return RiskFactorResult(
            factor=RiskFactor.SIGNAL_QUALITY,
            risk_score=risk_score,
            risk_multiplier=risk_multiplier,
            confidence=0.9,
            warnings=warnings,
            metadata=metadata,
        )

    def _assess_market_volatility(
        self, signal: TradingSignal, market_data: MarketData
    ) -> RiskFactorResult:
        """Assess market volatility risk factor.

        Args:
            signal: Trading signal to assess
            market_data: Current market data

        Returns:
            Risk factor result for market volatility
        """
        warnings = []
        metadata = {}

        # Get volatility from market data metadata
        volatility = market_data.metadata.get("volatility", 0.02)
        metadata["detected_volatility"] = volatility

        # Calculate risk score based on volatility thresholds
        if volatility <= self._config.volatility_threshold_low:
            risk_score = 0.2  # Low volatility = low risk
            metadata["volatility_level"] = "low"
        elif volatility <= self._config.volatility_threshold_high:
            risk_score = 0.5  # Moderate volatility = moderate risk
            metadata["volatility_level"] = "moderate"
        else:
            risk_score = 0.8  # High volatility = high risk
            warnings.append(f"High volatility detected: {volatility:.4f}")
            metadata["volatility_level"] = "high"

        # Risk multiplier increases with volatility
        risk_multiplier = 1.0 + (volatility * self._config.volatility_risk_multiplier)

        return RiskFactorResult(
            factor=RiskFactor.MARKET_VOLATILITY,
            risk_score=risk_score,
            risk_multiplier=risk_multiplier,
            confidence=0.8,
            warnings=warnings,
            metadata=metadata,
        )

    def _assess_position_concentration(
        self, signal: TradingSignal, portfolio_context: Dict[str, Any]
    ) -> RiskFactorResult:
        """Assess position concentration risk factor.

        Args:
            signal: Trading signal to assess
            portfolio_context: Portfolio information

        Returns:
            Risk factor result for position concentration
        """
        warnings = []
        metadata = {}

        # Get current position concentration for symbol
        current_positions = portfolio_context.get("positions", {})
        total_portfolio_value = portfolio_context.get("total_value", 1.0)

        symbol_exposure = current_positions.get(signal.symbol, {})
        current_value = symbol_exposure.get("value", 0.0)
        concentration = (
            current_value / total_portfolio_value if total_portfolio_value > 0 else 0.0
        )

        metadata["current_concentration"] = concentration
        metadata["max_allowed"] = self._config.max_position_concentration

        # Calculate risk score based on concentration
        if concentration > self._config.max_position_concentration:
            risk_score = min(
                1.0, concentration / self._config.max_position_concentration
            )
            warnings.append(
                f"High position concentration: {concentration:.2%} for {signal.symbol}"
            )
        else:
            risk_score = concentration / self._config.max_position_concentration

        # Risk multiplier increases with concentration
        concentration_multiplier = 1.0 + (
            concentration * self._config.concentration_penalty_factor
        )

        return RiskFactorResult(
            factor=RiskFactor.POSITION_CONCENTRATION,
            risk_score=risk_score,
            risk_multiplier=concentration_multiplier,
            confidence=0.85,
            warnings=warnings,
            metadata=metadata,
        )

    def _assess_time_context(self, signal: TradingSignal) -> RiskFactorResult:
        """Assess time context risk factor.

        Args:
            signal: Trading signal to assess

        Returns:
            Risk factor result for time context
        """
        warnings = []
        metadata = {}

        current_time = time.time()
        signal_time = (
            signal.timestamp / 1000 if signal.timestamp > 1e10 else signal.timestamp
        )

        # Check if signal is stale
        signal_age_minutes = (current_time - signal_time) / 60
        metadata["signal_age_minutes"] = signal_age_minutes

        # Base risk score from signal age
        if signal_age_minutes > 60:  # 1 hour old
            risk_score = 0.7
            warnings.append(f"Stale signal: {signal_age_minutes:.1f} minutes old")
        elif signal_age_minutes > 30:  # 30 minutes old
            risk_score = 0.4
        else:
            risk_score = 0.2

        # TODO: Add market hours check, weekend check, news events
        # This would require additional market calendar and news data integration

        risk_multiplier = 1.0 + (risk_score * 0.3)

        return RiskFactorResult(
            factor=RiskFactor.TIME_CONTEXT,
            risk_score=risk_score,
            risk_multiplier=risk_multiplier,
            confidence=0.7,
            warnings=warnings,
            metadata=metadata,
        )

    def _assess_strategy_track_record(self, signal: TradingSignal) -> RiskFactorResult:
        """Assess strategy track record risk factor.

        Args:
            signal: Trading signal to assess

        Returns:
            Risk factor result for strategy track record
        """
        warnings = []
        metadata = {}

        strategy_name = signal.strategy_name
        performance = self._strategy_performance.get(strategy_name, {})

        total_trades = performance.get("total_trades", 0)
        win_rate = performance.get("win_rate", 0.5)  # Default neutral

        metadata["total_trades"] = total_trades
        metadata["win_rate"] = win_rate

        # Calculate risk score based on track record
        if total_trades < self._config.min_strategy_trades:
            risk_score = 0.5  # Neutral risk for new strategies
            warnings.append(f"Limited track record: {total_trades} trades")
        elif win_rate >= self._config.good_win_rate_threshold:
            risk_score = 0.2  # Low risk for good performers
            metadata["performance_tier"] = "good"
        elif win_rate <= self._config.poor_win_rate_threshold:
            risk_score = 0.8  # High risk for poor performers
            warnings.append(f"Poor win rate: {win_rate:.2%}")
            metadata["performance_tier"] = "poor"
        else:
            risk_score = 0.5  # Moderate risk for average performers
            metadata["performance_tier"] = "average"

        # Risk multiplier based on performance
        performance_multiplier = 2.0 - (
            win_rate * 1.5
        )  # Better performance = lower multiplier
        risk_multiplier = max(0.5, min(2.0, performance_multiplier))

        return RiskFactorResult(
            factor=RiskFactor.STRATEGY_TRACK_RECORD,
            risk_score=risk_score,
            risk_multiplier=risk_multiplier,
            confidence=0.6 if total_trades < self._config.min_strategy_trades else 0.9,
            warnings=warnings,
            metadata=metadata,
        )

    def _calculate_overall_risk(
        self, signal: TradingSignal, factor_results: List[RiskFactorResult]
    ) -> Dict[str, Any]:
        """Calculate overall risk from individual factor assessments.

        Args:
            signal: Trading signal being assessed
            factor_results: Individual risk factor results

        Returns:
            Dictionary with overall risk calculation results
        """
        if not factor_results:
            # Fallback to default values
            return {
                "risk_level": RiskLevel.MODERATE,
                "risk_multiplier": self._config.default_risk_multiplier,
                "position_adjustment": 1.0,
                "confidence": 0.5,
                "warnings": ["No risk factors assessed"],
                "metadata": {"fallback_used": True},
            }

        # Calculate weighted average risk score
        total_weight = sum(result.confidence for result in factor_results)
        weighted_risk_score = (
            sum(result.risk_score * result.confidence for result in factor_results)
            / total_weight
        )

        # Calculate combined risk multiplier
        combined_multiplier = 1.0
        for result in factor_results:
            # Weight by confidence and cap impact
            weight = result.confidence * 0.2  # Max 20% impact per factor
            factor_impact = (result.risk_multiplier - 1.0) * weight
            combined_multiplier += factor_impact

        # Apply limits
        combined_multiplier = max(
            self._config.min_risk_multiplier,
            min(self._config.max_risk_multiplier, combined_multiplier),
        )

        # Determine risk level
        if weighted_risk_score <= 0.2:
            risk_level = RiskLevel.VERY_LOW
        elif weighted_risk_score <= 0.4:
            risk_level = RiskLevel.LOW
        elif weighted_risk_score <= 0.6:
            risk_level = RiskLevel.MODERATE
        elif weighted_risk_score <= 0.8:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.VERY_HIGH

        # Position size adjustment (inverse of risk)
        position_adjustment = 1.0 / combined_multiplier

        # Overall confidence
        overall_confidence = sum(result.confidence for result in factor_results) / len(
            factor_results
        )

        # Collect warnings
        warnings = []
        for result in factor_results:
            warnings.extend(result.warnings)

        return {
            "risk_level": risk_level,
            "risk_multiplier": combined_multiplier,
            "position_adjustment": position_adjustment,
            "confidence": overall_confidence,
            "warnings": warnings,
            "metadata": {
                "weighted_risk_score": weighted_risk_score,
                "factors_assessed": len(factor_results),
            },
        }

    def _generate_recommendations(
        self,
        overall_result: Dict[str, Any],
        factor_results: List[RiskFactorResult],
    ) -> List[str]:
        """Generate actionable recommendations based on risk assessment.

        Args:
            overall_result: Overall risk calculation results
            factor_results: Individual factor results

        Returns:
            List of recommendation strings
        """
        recommendations = []

        risk_level = overall_result["risk_level"]
        risk_multiplier = overall_result["risk_multiplier"]

        # General recommendations based on risk level
        if risk_level == RiskLevel.VERY_HIGH:
            recommendations.append("Consider avoiding this trade due to very high risk")
            recommendations.append("If proceeding, use minimal position size")
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("Use reduced position size due to elevated risk")
            recommendations.append("Consider tighter stop-loss levels")
        elif risk_level == RiskLevel.VERY_LOW:
            recommendations.append(
                "Consider increasing position size for this low-risk opportunity"
            )

        # Specific recommendations based on individual factors
        for result in factor_results:
            if result.risk_score > 0.7:
                if result.factor == RiskFactor.SIGNAL_QUALITY:
                    recommendations.append("Wait for higher confidence signal")
                elif result.factor == RiskFactor.MARKET_VOLATILITY:
                    recommendations.append(
                        "Use wider stop-losses due to high volatility"
                    )
                elif result.factor == RiskFactor.POSITION_CONCENTRATION:
                    recommendations.append(
                        "Reduce position size to manage concentration risk"
                    )

        # Risk multiplier specific recommendations
        if risk_multiplier > 1.5:
            recommendations.append(
                f"Apply {risk_multiplier:.2f}x risk multiplier to position sizing"
            )

        return recommendations

    def _publish_risk_assessment_event(self, result: RiskAssessmentResult) -> None:
        """Publish risk assessment event to event hub.

        Args:
            result: Risk assessment result to publish
        """
        try:
            event_data = {
                "result": result,
                "symbol": result.signal.symbol,
                "risk_level": result.overall_risk_level.value,
                "risk_multiplier": result.overall_risk_multiplier,
                "timestamp": result.assessment_timestamp,
            }

            self._event_hub.publish(EventType.POSITION_SIZE_WARNING, event_data)

            # Publish high risk warning if needed
            if result.overall_risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
                self._event_hub.publish(EventType.RISK_LIMIT_EXCEEDED, event_data)

        except Exception as e:
            self._logger.error(f"Error publishing risk assessment event: {e}")

    def _update_strategy_performance(
        self, signal: TradingSignal, result: RiskAssessmentResult
    ) -> None:
        """Update strategy performance tracking.

        Args:
            signal: Trading signal
            result: Risk assessment result
        """
        strategy_name = signal.strategy_name
        if strategy_name not in self._strategy_performance:
            self._strategy_performance[strategy_name] = {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.5,
                "last_updated": int(time.time()),
            }

        # This is a simplified update - in a real system, you'd update
        # based on actual trade outcomes
        perf = self._strategy_performance[strategy_name]
        perf["total_trades"] += 1
        perf["last_updated"] = int(time.time())


def create_risk_assessor(
    min_confidence_threshold: float = 0.6,
    volatility_threshold_high: float = 0.05,
    max_position_concentration: float = 0.3,
    event_hub: Optional[EventHub] = None,
    **kwargs: Any,
) -> RiskAssessor:
    """Factory function to create RiskAssessor instance.

    Args:
        min_confidence_threshold: Minimum signal confidence threshold
        volatility_threshold_high: High volatility threshold
        max_position_concentration: Maximum position concentration allowed
        event_hub: Optional event hub for event publishing
        **kwargs: Additional configuration parameters

    Returns:
        RiskAssessor: Configured risk assessor instance

    Raises:
        InvalidRiskConfigError: If configuration is invalid
    """
    config = RiskAssessmentConfig(
        min_confidence_threshold=min_confidence_threshold,
        high_confidence_threshold=kwargs.get("high_confidence_threshold", 0.8),
        weak_signal_penalty=kwargs.get("weak_signal_penalty", 0.2),
        strong_signal_bonus=kwargs.get("strong_signal_bonus", 0.1),
        volatility_threshold_low=kwargs.get("volatility_threshold_low", 0.02),
        volatility_threshold_high=volatility_threshold_high,
        volatility_risk_multiplier=kwargs.get("volatility_risk_multiplier", 1.5),
        max_position_concentration=max_position_concentration,
        concentration_penalty_factor=kwargs.get("concentration_penalty_factor", 0.15),
        market_hours_risk_reduction=kwargs.get("market_hours_risk_reduction", 0.1),
        weekend_risk_increase=kwargs.get("weekend_risk_increase", 0.2),
        news_event_risk_increase=kwargs.get("news_event_risk_increase", 0.25),
        min_strategy_trades=kwargs.get("min_strategy_trades", 10),
        good_win_rate_threshold=kwargs.get("good_win_rate_threshold", 0.6),
        poor_win_rate_threshold=kwargs.get("poor_win_rate_threshold", 0.4),
        min_risk_multiplier=kwargs.get("min_risk_multiplier", 0.3),
        max_risk_multiplier=kwargs.get("max_risk_multiplier", 2.0),
        default_risk_multiplier=kwargs.get("default_risk_multiplier", 1.0),
        enable_signal_quality_check=kwargs.get("enable_signal_quality_check", True),
        enable_volatility_check=kwargs.get("enable_volatility_check", True),
        enable_concentration_check=kwargs.get("enable_concentration_check", True),
        enable_time_context_check=kwargs.get("enable_time_context_check", True),
        enable_strategy_check=kwargs.get("enable_strategy_check", True),
        metadata=kwargs.get("metadata", {}),
    )

    return RiskAssessor(config, event_hub)
