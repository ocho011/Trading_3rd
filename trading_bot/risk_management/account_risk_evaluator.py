"""
Account balance-based risk assessment system for trading bot portfolio management.

This module implements a comprehensive account balance and portfolio context-aware
risk evaluation system that analyzes account state, position concentration, margin
requirements, and overall portfolio health to prevent over-leveraging and
excessive risk exposure.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.core.logger import get_module_logger
from trading_bot.strategies.base_strategy import TradingSignal


class AccountRiskError(Exception):
    """Base exception for account risk evaluation errors."""


class InsufficientMarginError(AccountRiskError):
    """Exception raised when insufficient margin for position."""


class ExcessiveRiskError(AccountRiskError):
    """Exception raised when risk limits are exceeded."""


class InvalidAccountStateError(AccountRiskError):
    """Exception raised for invalid account state data."""


class RiskProfile(Enum):
    """Risk profile classifications for account evaluation."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

    @property
    def max_portfolio_risk(self) -> float:
        """Maximum portfolio risk percentage for this profile."""
        risk_limits = {
            "conservative": 0.02,  # 2% portfolio risk
            "moderate": 0.05,  # 5% portfolio risk
            "aggressive": 0.10,  # 10% portfolio risk
        }
        return risk_limits[self.value]

    @property
    def max_position_size(self) -> float:
        """Maximum position size percentage for this profile."""
        position_limits = {
            "conservative": 0.10,  # 10% max position
            "moderate": 0.20,  # 20% max position
            "aggressive": 0.30,  # 30% max position
        }
        return position_limits[self.value]

    @property
    def max_leverage(self) -> float:
        """Maximum leverage multiplier for this profile."""
        leverage_limits = {
            "conservative": 2.0,  # 2x leverage
            "moderate": 3.0,  # 3x leverage
            "aggressive": 5.0,  # 5x leverage
        }
        return leverage_limits[self.value]


class AccountRiskLevel(Enum):
    """Account risk level classifications."""

    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"

    @property
    def threshold(self) -> float:
        """Risk threshold for this level."""
        thresholds = {
            "safe": 0.20,  # 0-20% of limit
            "caution": 0.50,  # 20-50% of limit
            "warning": 0.75,  # 50-75% of limit
            "danger": 0.90,  # 75-90% of limit
            "critical": 1.0,  # 90%+ of limit
        }
        return thresholds[self.value]


@dataclass
class PositionInfo:
    """Information about an individual position in the portfolio.

    Contains all relevant position data for risk assessment including
    market value, unrealized P&L, and risk metrics.
    """

    symbol: str
    quantity: Decimal
    average_price: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    position_type: str  # "long" or "short"
    entry_timestamp: int
    margin_used: Decimal = Decimal("0")
    stop_loss_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None
    correlation_group: Optional[str] = None
    volatility: float = 0.02
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate position information after initialization."""
        if self.quantity <= 0:
            raise InvalidAccountStateError("Position quantity must be positive")
        if self.average_price <= 0:
            raise InvalidAccountStateError("Average price must be positive")
        if self.current_price <= 0:
            raise InvalidAccountStateError("Current price must be positive")
        if self.position_type not in ["long", "short"]:
            raise InvalidAccountStateError("Position type must be 'long' or 'short'")

    @property
    def position_size_percentage(self) -> float:
        """Calculate position size as percentage of portfolio."""
        total_value = self.metadata.get("portfolio_total_value", Decimal("0"))
        if total_value <= 0:
            return 0.0
        return float(abs(self.market_value) / total_value)

    @property
    def risk_amount(self) -> Decimal:
        """Calculate the amount at risk for this position."""
        if self.stop_loss_price and self.position_type == "long":
            return (
                max(Decimal("0"), self.average_price - self.stop_loss_price)
                * self.quantity
            )
        if self.stop_loss_price and self.position_type == "short":
            return (
                max(Decimal("0"), self.stop_loss_price - self.average_price)
                * self.quantity
            )
        else:
            # Default to 2% risk if no stop loss set
            return abs(self.market_value) * Decimal("0.02")


@dataclass
class AccountState:
    """Current account state information for risk assessment.

    Contains comprehensive account information including balance,
    positions, margin usage, and performance metrics.
    """

    account_id: str
    total_equity: Decimal
    available_cash: Decimal
    used_margin: Decimal
    available_margin: Decimal
    total_portfolio_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl_today: Decimal
    positions: Dict[str, PositionInfo]
    open_orders_value: Decimal = Decimal("0")
    maintenance_margin_req: Decimal = Decimal("0")
    buying_power: Decimal = Decimal("0")
    max_drawdown_today: Decimal = Decimal("0")
    max_drawdown_period: Decimal = Decimal("0")
    account_currency: str = "USD"
    last_updated: int = field(default_factory=lambda: int(time.time() * 1000))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate account state after initialization."""
        if self.total_equity <= 0:
            raise InvalidAccountStateError("Total equity must be positive")
        if self.available_cash < 0:
            raise InvalidAccountStateError("Available cash cannot be negative")
        if self.used_margin < 0:
            raise InvalidAccountStateError("Used margin cannot be negative")

        # Calculate buying power if not provided
        if self.buying_power == 0:
            self.buying_power = self.available_cash + self.available_margin

        # Update position metadata with portfolio total
        for position in self.positions.values():
            position.metadata["portfolio_total_value"] = self.total_portfolio_value

    @property
    def leverage_ratio(self) -> float:
        """Calculate current leverage ratio."""
        if self.total_equity <= 0:
            return 0.0
        total_exposure = sum(abs(pos.market_value) for pos in self.positions.values())
        return float(total_exposure / self.total_equity)

    @property
    def margin_utilization(self) -> float:
        """Calculate margin utilization percentage."""
        total_margin = self.used_margin + self.available_margin
        if total_margin <= 0:
            return 0.0
        return float(self.used_margin / total_margin)

    @property
    def portfolio_concentration(self) -> Dict[str, float]:
        """Calculate position concentration by symbol."""
        if self.total_portfolio_value <= 0:
            return {}
        return {
            symbol: float(abs(pos.market_value) / self.total_portfolio_value)
            for symbol, pos in self.positions.items()
        }

    @property
    def correlation_exposure(self) -> Dict[str, float]:
        """Calculate exposure by correlation group."""
        groups: Dict[str, Decimal] = {}
        for position in self.positions.values():
            group = position.correlation_group or "uncorrelated"
            groups[group] = groups.get(group, Decimal("0")) + abs(position.market_value)

        if self.total_portfolio_value <= 0:
            return {}
        return {
            group: float(value / self.total_portfolio_value)
            for group, value in groups.items()
        }


@dataclass
class AccountRiskConfig:
    """Configuration parameters for account risk evaluation.

    Defines all thresholds, limits, and parameters used in
    account-level risk assessment calculations.
    """

    # Portfolio risk limits
    max_portfolio_risk_pct: float = 0.05  # 5% max portfolio risk
    max_position_concentration: float = 0.20  # 20% max per position
    max_correlation_exposure: float = 0.40  # 40% max per correlation group
    max_leverage_ratio: float = 3.0  # 3x max leverage
    max_margin_utilization: float = 0.80  # 80% max margin usage

    # Drawdown protection
    max_daily_drawdown: float = 0.03  # 3% max daily drawdown
    max_period_drawdown: float = 0.15  # 15% max period drawdown
    stop_trading_drawdown: float = 0.20  # 20% emergency stop

    # Margin requirements
    maintenance_margin_buffer: float = 0.20  # 20% buffer above maintenance
    min_available_cash_pct: float = 0.10  # 10% min cash reserve
    margin_call_threshold: float = 0.30  # 30% remaining available margin

    # Position sizing limits
    min_position_value: Decimal = Decimal("100")  # $100 minimum position
    max_position_value: Decimal = Decimal("1000000")  # $1M max position
    max_positions_count: int = 50  # Max 50 open positions

    # Risk calculation parameters
    default_correlation: float = 0.30  # Default correlation assumption
    volatility_scaling_factor: float = 2.0  # Volatility risk scaling
    concentration_penalty_factor: float = 1.5  # Concentration penalty

    # Time-based limits
    max_position_age_days: int = 30  # Max position age for warnings
    stale_data_threshold_minutes: int = 15  # Stale data warning

    # Emergency controls
    enable_emergency_stops: bool = True
    enable_margin_monitoring: bool = True
    enable_concentration_limits: bool = True
    enable_correlation_analysis: bool = True
    enable_drawdown_protection: bool = True

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 < self.max_portfolio_risk_pct <= 1.0:
            raise InvalidAccountStateError("Portfolio risk must be 0-100%")
        if not 0.0 < self.max_position_concentration <= 1.0:
            raise InvalidAccountStateError("Position concentration must be 0-100%")
        if self.max_leverage_ratio <= 1.0:
            raise InvalidAccountStateError("Max leverage must be > 1.0")
        if not 0.0 < self.max_margin_utilization <= 1.0:
            raise InvalidAccountStateError("Margin utilization must be 0-100%")


@dataclass
class AccountRiskResult:
    """Result of account risk evaluation with detailed analysis.

    Contains comprehensive risk assessment results including
    individual risk factors, recommendations, and limits.
    """

    account_id: str
    risk_level: AccountRiskLevel
    overall_risk_score: float
    can_add_position: bool
    max_new_position_value: Decimal
    max_new_position_quantity: Optional[Decimal]
    margin_requirement: Decimal
    available_buying_power: Decimal
    risk_factors: Dict[str, float]
    concentration_analysis: Dict[str, float]
    correlation_analysis: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    emergency_actions: List[str] = field(default_factory=list)
    assessment_timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate risk assessment result."""
        if not 0.0 <= self.overall_risk_score <= 1.0:
            raise InvalidAccountStateError("Risk score must be 0.0-1.0")
        if self.max_new_position_value < 0:
            raise InvalidAccountStateError("Max position value cannot be negative")

    def has_critical_risks(self) -> bool:
        """Check if any risk factors are at critical levels."""
        return any(score >= 0.90 for score in self.risk_factors.values())

    def requires_immediate_action(self) -> bool:
        """Check if immediate action is required."""
        return (
            self.risk_level == AccountRiskLevel.CRITICAL
            or len(self.emergency_actions) > 0
            or self.has_critical_risks()
        )

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get summarized risk information."""
        return {
            "risk_level": self.risk_level.value,
            "overall_score": self.overall_risk_score,
            "can_trade": self.can_add_position,
            "max_position_usd": float(self.max_new_position_value),
            "warnings_count": len(self.warnings),
            "critical_risks": self.has_critical_risks(),
            "immediate_action_required": self.requires_immediate_action(),
        }


class IAccountRiskEvaluator(ABC):
    """Abstract interface for account risk evaluation implementations."""

    @abstractmethod
    def evaluate_new_position(
        self,
        signal: TradingSignal,
        account_state: AccountState,
        proposed_quantity: Optional[Decimal] = None,
    ) -> AccountRiskResult:
        """Evaluate risk for adding a new position.

        Args:
            signal: Trading signal for the new position
            account_state: Current account state
            proposed_quantity: Optional proposed position quantity

        Returns:
            AccountRiskResult: Comprehensive risk evaluation

        Raises:
            AccountRiskError: If evaluation fails
        """
        pass

    @abstractmethod
    def get_max_position_size(
        self,
        symbol: str,
        price: Decimal,
        account_state: AccountState,
    ) -> Tuple[Decimal, Decimal]:
        """Calculate maximum allowable position size.

        Args:
            symbol: Trading symbol
            price: Current price
            account_state: Current account state

        Returns:
            Tuple[Decimal, Decimal]: (max_quantity, max_value)

        Raises:
            AccountRiskError: If calculation fails
        """
        pass

    @abstractmethod
    def check_portfolio_health(
        self,
        account_state: AccountState,
    ) -> AccountRiskResult:
        """Perform comprehensive portfolio health assessment.

        Args:
            account_state: Current account state

        Returns:
            AccountRiskResult: Portfolio health evaluation

        Raises:
            AccountRiskError: If health check fails
        """
        pass

    @abstractmethod
    def validate_margin_requirements(
        self,
        signal: TradingSignal,
        quantity: Decimal,
        account_state: AccountState,
    ) -> bool:
        """Validate if sufficient margin exists for position.

        Args:
            signal: Trading signal
            quantity: Proposed position quantity
            account_state: Current account state

        Returns:
            bool: True if sufficient margin available

        Raises:
            InsufficientMarginError: If insufficient margin
        """
        pass

    @abstractmethod
    def update_config(self, config: AccountRiskConfig) -> None:
        """Update risk evaluation configuration.

        Args:
            config: New risk evaluation configuration

        Raises:
            InvalidAccountStateError: If configuration is invalid
        """
        pass


class AccountRiskEvaluator(IAccountRiskEvaluator):
    """Comprehensive account balance-based risk evaluation implementation.

    This class provides multi-dimensional risk assessment for account
    balance, portfolio concentration, margin requirements, and overall
    portfolio health to prevent over-leveraging and excessive risk exposure.

    Attributes:
        _config: Account risk evaluation configuration
        _event_hub: Event hub for publishing risk events
        _logger: Logger instance for risk evaluation logging
        _evaluations_count: Counter for performed evaluations
    """

    def __init__(
        self,
        config: AccountRiskConfig,
        event_hub: Optional[EventHub] = None,
    ) -> None:
        """Initialize account risk evaluator with configuration.

        Args:
            config: Account risk evaluation configuration
            event_hub: Optional event hub for event publishing

        Raises:
            InvalidAccountStateError: If configuration is invalid
        """
        if not isinstance(config, AccountRiskConfig):
            raise InvalidAccountStateError("Config must be AccountRiskConfig instance")

        self._config = config
        self._event_hub = event_hub
        self._logger = get_module_logger("account_risk_evaluator")
        self._evaluations_count = 0

        # Performance tracking
        self._risk_history: List[Dict[str, Any]] = []

        self._logger.info("Initialized account risk evaluator")

    def evaluate_new_position(
        self,
        signal: TradingSignal,
        account_state: AccountState,
        proposed_quantity: Optional[Decimal] = None,
    ) -> AccountRiskResult:
        """Evaluate risk for adding a new position.

        Args:
            signal: Trading signal for the new position
            account_state: Current account state
            proposed_quantity: Optional proposed position quantity

        Returns:
            AccountRiskResult: Comprehensive risk evaluation

        Raises:
            AccountRiskError: If evaluation fails
        """
        try:
            self._validate_inputs(signal, account_state)

            # Calculate proposed position size if not provided
            if proposed_quantity is None:
                max_qty, _ = self.get_max_position_size(
                    signal.symbol, Decimal(str(signal.price)), account_state
                )
                proposed_quantity = max_qty

            # Perform comprehensive risk assessment
            risk_factors = self._assess_all_risk_factors(
                signal, account_state, proposed_quantity
            )

            # Calculate margin requirements
            margin_req = self._calculate_margin_requirement(
                signal, proposed_quantity, account_state
            )

            # Determine if position can be added
            can_add, max_value, max_qty = self._evaluate_position_feasibility(
                signal, account_state, proposed_quantity, risk_factors
            )

            # Generate warnings and recommendations
            warnings, recommendations, emergency_actions = self._generate_guidance(
                risk_factors, account_state, can_add
            )

            # Calculate overall risk level
            risk_level = self._determine_risk_level(risk_factors)
            overall_score = self._calculate_overall_risk_score(risk_factors)

            # Create result
            result = AccountRiskResult(
                account_id=account_state.account_id,
                risk_level=risk_level,
                overall_risk_score=overall_score,
                can_add_position=can_add,
                max_new_position_value=max_value,
                max_new_position_quantity=max_qty,
                margin_requirement=margin_req,
                available_buying_power=account_state.buying_power,
                risk_factors=risk_factors,
                concentration_analysis=account_state.portfolio_concentration,
                correlation_analysis=account_state.correlation_exposure,
                warnings=warnings,
                recommendations=recommendations,
                emergency_actions=emergency_actions,
                metadata={
                    "signal_symbol": signal.symbol,
                    "signal_confidence": signal.confidence,
                    "proposed_quantity": float(proposed_quantity),
                    "leverage_ratio": account_state.leverage_ratio,
                    "margin_utilization": account_state.margin_utilization,
                },
            )

            # Publish events if needed
            if self._event_hub:
                self._publish_risk_events(result)

            # Update statistics
            self._evaluations_count += 1
            self._update_risk_history(result)

            self._logger.info(
                f"Position risk evaluation completed: {signal.symbol} "
                f"risk_level={risk_level.value} can_add={can_add}"
            )

            return result

        except Exception as e:
            self._logger.error("Error evaluating position risk: %s", e)
            raise AccountRiskError(f"Position risk evaluation failed: {e}") from e

    def get_max_position_size(
        self,
        symbol: str,
        price: Decimal,
        account_state: AccountState,
    ) -> Tuple[Decimal, Decimal]:
        """Calculate maximum allowable position size.

        Args:
            symbol: Trading symbol
            price: Current price
            account_state: Current account state

        Returns:
            Tuple[Decimal, Decimal]: (max_quantity, max_value)

        Raises:
            AccountRiskError: If calculation fails
        """
        try:
            # Check basic constraints
            if price <= 0:
                raise AccountRiskError("Price must be positive")

            # Calculate max value based on concentration limits
            max_concentration_value = account_state.total_portfolio_value * Decimal(
                str(self._config.max_position_concentration)
            )

            # Calculate max value based on risk limits
            max_risk_value = (
                account_state.total_equity
                * Decimal(str(self._config.max_portfolio_risk_pct))
                / Decimal("0.02")  # Assume 2% risk per position
            )

            # Calculate max value based on buying power
            max_buying_power_value = account_state.buying_power * Decimal("0.9")

            # Take the minimum of all constraints
            max_value = min(
                max_concentration_value,
                max_risk_value,
                max_buying_power_value,
                self._config.max_position_value,
            )

            # Ensure minimum position value
            max_value = max(max_value, self._config.min_position_value)

            # Calculate quantity
            max_quantity = max_value / price

            self._logger.debug(
                f"Max position size for {symbol}: "
                f"qty={max_quantity} value=${max_value}"
            )

            return max_quantity, max_value

        except Exception as e:
            self._logger.error("Error calculating max position size: %s", e)
            raise AccountRiskError(f"Max position size calculation failed: {e}") from e

    def check_portfolio_health(
        self,
        account_state: AccountState,
    ) -> AccountRiskResult:
        """Perform comprehensive portfolio health assessment.

        Args:
            account_state: Current account state

        Returns:
            AccountRiskResult: Portfolio health evaluation

        Raises:
            AccountRiskError: If health check fails
        """
        try:
            # Assess portfolio-level risk factors
            risk_factors = {
                "leverage_risk": self._assess_leverage_risk(account_state),
                "concentration_risk": self._assess_concentration_risk(account_state),
                "correlation_risk": self._assess_correlation_risk(account_state),
                "margin_risk": self._assess_margin_risk(account_state),
                "drawdown_risk": self._assess_drawdown_risk(account_state),
                "liquidity_risk": self._assess_liquidity_risk(account_state),
            }

            # Generate warnings and recommendations
            warnings, recommendations, emergency_actions = self._generate_guidance(
                risk_factors, account_state, True
            )

            # Calculate overall metrics
            risk_level = self._determine_risk_level(risk_factors)
            overall_score = self._calculate_overall_risk_score(risk_factors)

            # Create health assessment result
            result = AccountRiskResult(
                account_id=account_state.account_id,
                risk_level=risk_level,
                overall_risk_score=overall_score,
                can_add_position=risk_level
                not in [AccountRiskLevel.DANGER, AccountRiskLevel.CRITICAL],
                max_new_position_value=Decimal("0"),
                max_new_position_quantity=None,
                margin_requirement=Decimal("0"),
                available_buying_power=account_state.buying_power,
                risk_factors=risk_factors,
                concentration_analysis=account_state.portfolio_concentration,
                correlation_analysis=account_state.correlation_exposure,
                warnings=warnings,
                recommendations=recommendations,
                emergency_actions=emergency_actions,
                metadata={
                    "health_check": True,
                    "positions_count": len(account_state.positions),
                    "total_equity": float(account_state.total_equity),
                    "unrealized_pnl": float(account_state.unrealized_pnl),
                },
            )

            self._logger.info(
                f"Portfolio health check completed: "
                f"risk_level={risk_level.value} score={overall_score:.3f}"
            )

            return result

        except Exception as e:
            self._logger.error("Error checking portfolio health: %s", e)
            raise AccountRiskError(f"Portfolio health check failed: {e}") from e

    def validate_margin_requirements(
        self,
        signal: TradingSignal,
        quantity: Decimal,
        account_state: AccountState,
    ) -> bool:
        """Validate if sufficient margin exists for position.

        Args:
            signal: Trading signal
            quantity: Proposed position quantity
            account_state: Current account state

        Returns:
            bool: True if sufficient margin available

        Raises:
            InsufficientMarginError: If insufficient margin
        """
        try:
            margin_req = self._calculate_margin_requirement(
                signal, quantity, account_state
            )

            # Check available margin with buffer
            margin_buffer = margin_req * Decimal(
                str(self._config.maintenance_margin_buffer)
            )
            total_margin_needed = margin_req + margin_buffer

            has_sufficient_margin = (
                account_state.available_margin >= total_margin_needed
            )

            if not has_sufficient_margin:
                raise InsufficientMarginError(
                    f"Insufficient margin: need ${total_margin_needed}, "
                    f"available ${account_state.available_margin}"
                )

            self._logger.debug(
                f"Margin validation passed: need=${total_margin_needed} "
                f"available=${account_state.available_margin}"
            )

            return True

        except InsufficientMarginError:
            raise
        except Exception as e:
            self._logger.error("Error validating margin requirements: %s", e)
            raise AccountRiskError(f"Margin validation failed: {e}") from e

    def update_config(self, config: AccountRiskConfig) -> None:
        """Update risk evaluation configuration.

        Args:
            config: New risk evaluation configuration

        Raises:
            InvalidAccountStateError: If configuration is invalid
        """
        if not isinstance(config, AccountRiskConfig):
            raise InvalidAccountStateError("Config must be AccountRiskConfig instance")

        self._config = config
        self._logger.info("Updated account risk evaluation configuration")

    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get risk evaluation statistics.

        Returns:
            Dictionary containing evaluation statistics
        """
        return {
            "evaluations_count": self._evaluations_count,
            "recent_risk_levels": [
                entry["risk_level"] for entry in self._risk_history[-10:]
            ],
            "config_summary": {
                "max_portfolio_risk": self._config.max_portfolio_risk_pct,
                "max_position_concentration": self._config.max_position_concentration,
                "max_leverage": self._config.max_leverage_ratio,
            },
        }

    def _validate_inputs(
        self,
        signal: TradingSignal,
        account_state: AccountState,
    ) -> None:
        """Validate inputs for risk evaluation."""
        if not isinstance(signal, TradingSignal):
            raise AccountRiskError("Signal must be TradingSignal instance")
        if not isinstance(account_state, AccountState):
            raise AccountRiskError("Account state must be AccountState instance")

        # Check for stale data
        current_time = int(time.time() * 1000)
        data_age_minutes = (current_time - account_state.last_updated) / (1000 * 60)
        if data_age_minutes > self._config.stale_data_threshold_minutes:
            raise AccountRiskError(
                f"Account data is stale: {data_age_minutes:.1f} minutes old"
            )

    def _assess_all_risk_factors(
        self,
        signal: TradingSignal,
        account_state: AccountState,
        proposed_quantity: Decimal,
    ) -> Dict[str, float]:
        """Assess all account-level risk factors."""
        return {
            "leverage_risk": self._assess_leverage_risk(account_state),
            "concentration_risk": self._assess_new_position_concentration_risk(
                signal, proposed_quantity, account_state
            ),
            "correlation_risk": self._assess_correlation_risk(account_state),
            "margin_risk": self._assess_margin_risk(account_state),
            "drawdown_risk": self._assess_drawdown_risk(account_state),
            "liquidity_risk": self._assess_liquidity_risk(account_state),
            "position_size_risk": self._assess_position_size_risk(
                signal, proposed_quantity, account_state
            ),
        }

    def _assess_leverage_risk(self, account_state: AccountState) -> float:
        """Assess leverage risk factor."""
        current_leverage = account_state.leverage_ratio
        max_leverage = self._config.max_leverage_ratio

        if current_leverage <= 1.0:
            return 0.1  # Very low risk for no leverage
        elif current_leverage <= max_leverage * 0.5:
            return 0.3  # Low risk
        elif current_leverage <= max_leverage * 0.75:
            return 0.6  # Moderate risk
        elif current_leverage <= max_leverage:
            return 0.8  # High risk
        else:
            return 1.0  # Critical risk - exceeds max leverage

    def _assess_concentration_risk(self, account_state: AccountState) -> float:
        """Assess portfolio concentration risk."""
        concentrations = account_state.portfolio_concentration.values()
        if not concentrations:
            return 0.0

        max_concentration = max(concentrations)
        threshold = self._config.max_position_concentration

        if max_concentration <= threshold * 0.5:
            return 0.2  # Low concentration
        elif max_concentration <= threshold * 0.75:
            return 0.5  # Moderate concentration
        elif max_concentration <= threshold:
            return 0.7  # High concentration
        else:
            return 1.0  # Critical concentration

    def _assess_new_position_concentration_risk(
        self,
        signal: TradingSignal,
        proposed_quantity: Decimal,
        account_state: AccountState,
    ) -> float:
        """Assess concentration risk with new position added."""
        # Calculate position value
        position_value = proposed_quantity * Decimal(str(signal.price))

        # Get current position value for this symbol
        current_position = account_state.positions.get(signal.symbol)
        current_value = (
            current_position.market_value if current_position else Decimal("0")
        )

        # Calculate new total position value
        new_position_value = current_value + position_value
        new_concentration = float(
            new_position_value / account_state.total_portfolio_value
        )

        threshold = self._config.max_position_concentration

        if new_concentration <= threshold * 0.5:
            return 0.2
        elif new_concentration <= threshold * 0.75:
            return 0.5
        elif new_concentration <= threshold:
            return 0.8
        else:
            return 1.0

    def _assess_correlation_risk(self, account_state: AccountState) -> float:
        """Assess correlation risk factor."""
        correlations = account_state.correlation_exposure.values()
        if not correlations:
            return 0.0

        max_correlation_exposure = max(correlations)
        threshold = self._config.max_correlation_exposure

        if max_correlation_exposure <= threshold * 0.5:
            return 0.2
        elif max_correlation_exposure <= threshold * 0.75:
            return 0.5
        elif max_correlation_exposure <= threshold:
            return 0.8
        else:
            return 1.0

    def _assess_margin_risk(self, account_state: AccountState) -> float:
        """Assess margin utilization risk."""
        utilization = account_state.margin_utilization
        threshold = self._config.max_margin_utilization

        if utilization <= threshold * 0.4:
            return 0.1  # Very low margin usage
        elif utilization <= threshold * 0.6:
            return 0.3  # Low margin usage
        elif utilization <= threshold * 0.8:
            return 0.6  # Moderate margin usage
        elif utilization <= threshold:
            return 0.8  # High margin usage
        else:
            return 1.0  # Critical margin usage

    def _assess_drawdown_risk(self, account_state: AccountState) -> float:
        """Assess drawdown risk factor."""
        daily_dd = float(
            abs(account_state.max_drawdown_today / account_state.total_equity)
        )
        period_dd = float(
            abs(account_state.max_drawdown_period / account_state.total_equity)
        )

        daily_threshold = self._config.max_daily_drawdown
        period_threshold = self._config.max_period_drawdown

        # Take the worse of daily or period drawdown
        daily_risk = min(1.0, daily_dd / daily_threshold)
        period_risk = min(1.0, period_dd / period_threshold)

        return max(daily_risk, period_risk)

    def _assess_liquidity_risk(self, account_state: AccountState) -> float:
        """Assess liquidity risk factor."""
        cash_ratio = float(account_state.available_cash / account_state.total_equity)
        min_cash_threshold = self._config.min_available_cash_pct

        if cash_ratio >= min_cash_threshold * 2:
            return 0.1  # Excellent liquidity
        elif cash_ratio >= min_cash_threshold * 1.5:
            return 0.3  # Good liquidity
        elif cash_ratio >= min_cash_threshold:
            return 0.6  # Adequate liquidity
        elif cash_ratio >= min_cash_threshold * 0.5:
            return 0.8  # Poor liquidity
        else:
            return 1.0  # Critical liquidity

    def _assess_position_size_risk(
        self,
        signal: TradingSignal,
        proposed_quantity: Decimal,
        account_state: AccountState,
    ) -> float:
        """Assess risk of proposed position size."""
        position_value = proposed_quantity * Decimal(str(signal.price))

        if position_value < self._config.min_position_value:
            return 0.8  # High risk for very small positions (transaction costs)
        elif position_value > self._config.max_position_value:
            return 1.0  # Critical risk for oversized positions
        else:
            # Scale risk based on position size relative to equity
            size_ratio = float(position_value / account_state.total_equity)
            return min(1.0, size_ratio * 10)  # 10% of equity = 100% risk

    def _calculate_margin_requirement(
        self,
        signal: TradingSignal,
        quantity: Decimal,
        account_state: AccountState,
    ) -> Decimal:
        """Calculate margin requirement for position."""
        _ = account_state  # Used for future instrument-specific margin rates
        position_value = quantity * Decimal(str(signal.price))

        # Basic margin calculation (could be enhanced with instrument-specific rates)
        # For stocks: typically 50% margin requirement
        # For futures/forex: varies by instrument
        base_margin_rate = Decimal("0.50")  # 50% for stocks

        # Add volatility adjustment
        volatility_multiplier = Decimal(
            str(1.0 + (signal.metadata.get("volatility", 0.02) * 2))
        )

        margin_requirement = position_value * base_margin_rate * volatility_multiplier

        return margin_requirement

    def _evaluate_position_feasibility(
        self,
        signal: TradingSignal,
        account_state: AccountState,
        proposed_quantity: Decimal,
        risk_factors: Dict[str, float],
    ) -> Tuple[bool, Decimal, Optional[Decimal]]:
        """Evaluate if position can be added and calculate max allowed size."""
        # Check if any risk factors are critical
        critical_risks = [
            factor for factor, score in risk_factors.items() if score >= 0.90
        ]
        if critical_risks:
            return False, Decimal("0"), None

        # Check position count limit
        if len(account_state.positions) >= self._config.max_positions_count:
            return False, Decimal("0"), None

        # Calculate maximum allowed position
        max_quantity, max_value = self.get_max_position_size(
            signal.symbol, Decimal(str(signal.price)), account_state
        )

        # Check if proposed quantity exceeds maximum
        can_add = proposed_quantity <= max_quantity

        return can_add, max_value, max_quantity

    def _generate_guidance(
        self,
        risk_factors: Dict[str, float],
        account_state: AccountState,
        can_add_position: bool,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate warnings, recommendations, and emergency actions."""
        warnings = []
        recommendations = []
        emergency_actions = []

        # Check individual risk factors
        for factor, score in risk_factors.items():
            if score >= 0.90:
                warnings.append(f"Critical {factor.replace('_', ' ')}: {score:.1%}")
                if factor == "leverage_risk":
                    emergency_actions.append("Reduce leverage immediately")
                elif factor == "margin_risk":
                    emergency_actions.append("Close positions to free margin")
                elif factor == "drawdown_risk":
                    emergency_actions.append("Stop trading due to excessive drawdown")

            elif score >= 0.70:
                warnings.append(f"High {factor.replace('_', ' ')}: {score:.1%}")
                if factor == "concentration_risk":
                    recommendations.append("Diversify positions across more symbols")
                elif factor == "correlation_risk":
                    recommendations.append("Reduce exposure to correlated assets")

        # Position-specific recommendations
        if not can_add_position:
            recommendations.append("Cannot add position due to risk limits")
            recommendations.append("Consider reducing existing positions first")

        # Cash management recommendations
        cash_ratio = float(account_state.available_cash / account_state.total_equity)
        if cash_ratio < self._config.min_available_cash_pct:
            recommendations.append("Maintain higher cash reserves")

        # Emergency stop conditions
        if self._config.enable_emergency_stops:
            drawdown_pct = float(
                abs(account_state.max_drawdown_today) / account_state.total_equity
            )
            if drawdown_pct >= self._config.stop_trading_drawdown:
                emergency_actions.append("EMERGENCY: Stop all trading immediately")

        return warnings, recommendations, emergency_actions

    def _determine_risk_level(self, risk_factors: Dict[str, float]) -> AccountRiskLevel:
        """Determine overall risk level from individual factors."""
        max_risk = max(risk_factors.values()) if risk_factors else 0.0
        avg_risk = (
            sum(risk_factors.values()) / len(risk_factors) if risk_factors else 0.0
        )

        # Use both max and average for determination
        combined_risk = (max_risk * 0.6) + (avg_risk * 0.4)

        if combined_risk <= 0.20:
            return AccountRiskLevel.SAFE
        elif combined_risk <= 0.40:
            return AccountRiskLevel.CAUTION
        elif combined_risk <= 0.65:
            return AccountRiskLevel.WARNING
        elif combined_risk <= 0.85:
            return AccountRiskLevel.DANGER
        else:
            return AccountRiskLevel.CRITICAL

    def _calculate_overall_risk_score(self, risk_factors: Dict[str, float]) -> float:
        """Calculate weighted overall risk score."""
        if not risk_factors:
            return 0.5

        # Weight different risk factors
        weights = {
            "leverage_risk": 0.20,
            "concentration_risk": 0.15,
            "correlation_risk": 0.10,
            "margin_risk": 0.20,
            "drawdown_risk": 0.25,
            "liquidity_risk": 0.10,
            "position_size_risk": 0.00,  # Already captured in other factors
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for factor, score in risk_factors.items():
            weight = weights.get(factor, 0.10)  # Default weight
            weighted_sum += score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _publish_risk_events(self, result: AccountRiskResult) -> None:
        """Publish risk assessment events to event hub."""
        if not self._event_hub:
            return

        try:
            # Publish general risk assessment event
            event_data = {
                "account_id": result.account_id,
                "risk_level": result.risk_level.value,
                "risk_score": result.overall_risk_score,
                "can_add_position": result.can_add_position,
                "timestamp": result.assessment_timestamp,
            }

            self._event_hub.publish(EventType.POSITION_SIZE_WARNING, event_data)

            # Publish specific warnings
            if result.risk_level in [
                AccountRiskLevel.DANGER,
                AccountRiskLevel.CRITICAL,
            ]:
                self._event_hub.publish(EventType.RISK_LIMIT_EXCEEDED, event_data)

            # Publish emergency actions if needed
            if result.emergency_actions:
                emergency_data = {
                    **event_data,
                    "emergency_actions": result.emergency_actions,
                }
                self._event_hub.publish(EventType.ERROR_OCCURRED, emergency_data)

        except Exception as e:
            self._logger.error("Error publishing risk events: %s", e)

    def _update_risk_history(self, result: AccountRiskResult) -> None:
        """Update risk assessment history for tracking."""
        history_entry = {
            "timestamp": result.assessment_timestamp,
            "risk_level": result.risk_level.value,
            "risk_score": result.overall_risk_score,
            "can_add_position": result.can_add_position,
            "warnings_count": len(result.warnings),
        }

        self._risk_history.append(history_entry)

        # Keep only last 100 entries
        if len(self._risk_history) > 100:
            self._risk_history = self._risk_history[-100:]


def create_account_risk_evaluator(
    risk_profile: RiskProfile = RiskProfile.MODERATE,
    max_leverage: Optional[float] = None,
    max_position_concentration: Optional[float] = None,
    enable_emergency_stops: bool = True,
    event_hub: Optional[EventHub] = None,
    **kwargs: Any,
) -> AccountRiskEvaluator:
    """Factory function to create AccountRiskEvaluator instance.

    Args:
        risk_profile: Risk profile for default settings
        max_leverage: Override maximum leverage ratio
        max_position_concentration: Override maximum position concentration
        enable_emergency_stops: Enable emergency stop mechanisms
        event_hub: Optional event hub for event publishing
        **kwargs: Additional configuration parameters

    Returns:
        AccountRiskEvaluator: Configured account risk evaluator instance

    Raises:
        InvalidAccountStateError: If configuration is invalid
    """
    # Use risk profile defaults
    max_portfolio_risk = risk_profile.max_portfolio_risk
    max_position_size = max_position_concentration or risk_profile.max_position_size
    max_leverage_ratio = max_leverage or risk_profile.max_leverage

    config = AccountRiskConfig(
        max_portfolio_risk_pct=kwargs.get("max_portfolio_risk_pct", max_portfolio_risk),
        max_position_concentration=max_position_size,
        max_correlation_exposure=kwargs.get("max_correlation_exposure", 0.40),
        max_leverage_ratio=max_leverage_ratio,
        max_margin_utilization=kwargs.get("max_margin_utilization", 0.80),
        max_daily_drawdown=kwargs.get("max_daily_drawdown", 0.03),
        max_period_drawdown=kwargs.get("max_period_drawdown", 0.15),
        stop_trading_drawdown=kwargs.get("stop_trading_drawdown", 0.20),
        maintenance_margin_buffer=kwargs.get("maintenance_margin_buffer", 0.20),
        min_available_cash_pct=kwargs.get("min_available_cash_pct", 0.10),
        margin_call_threshold=kwargs.get("margin_call_threshold", 0.30),
        min_position_value=Decimal(str(kwargs.get("min_position_value", 100))),
        max_position_value=Decimal(str(kwargs.get("max_position_value", 1000000))),
        max_positions_count=kwargs.get("max_positions_count", 50),
        default_correlation=kwargs.get("default_correlation", 0.30),
        volatility_scaling_factor=kwargs.get("volatility_scaling_factor", 2.0),
        concentration_penalty_factor=kwargs.get("concentration_penalty_factor", 1.5),
        max_position_age_days=kwargs.get("max_position_age_days", 30),
        stale_data_threshold_minutes=kwargs.get("stale_data_threshold_minutes", 15),
        enable_emergency_stops=enable_emergency_stops,
        enable_margin_monitoring=kwargs.get("enable_margin_monitoring", True),
        enable_concentration_limits=kwargs.get("enable_concentration_limits", True),
        enable_correlation_analysis=kwargs.get("enable_correlation_analysis", True),
        enable_drawdown_protection=kwargs.get("enable_drawdown_protection", True),
        metadata=kwargs.get("metadata", {}),
    )

    return AccountRiskEvaluator(config, event_hub)
