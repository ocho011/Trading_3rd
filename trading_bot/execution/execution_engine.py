"""
Execution Engine for order execution and management in the trading bot system.

This module provides the ExecutionEngine class that handles order execution
by processing ORDER_REQUEST_GENERATED events from the EventHub, validating
order requests, and preparing them for execution via the BinanceClient.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.core.logger import get_module_logger
from trading_bot.market_data.binance_client import (
    BinanceError,
    BinanceOrderError,
    BinanceRateLimitError,
    IExchangeClient,
)
from trading_bot.risk_management.risk_manager import OrderRequest, OrderType
from trading_bot.strategies.base_strategy import SignalType


class ExecutionEngineError(Exception):
    """Base exception for execution engine errors."""


class OrderValidationError(ExecutionEngineError):
    """Exception raised for order validation errors."""


class ExecutionProcessingError(ExecutionEngineError):
    """Exception raised for execution processing errors."""


class NetworkError(ExecutionEngineError):
    """Exception raised for network-related errors."""


class InsufficientBalanceError(ExecutionEngineError):
    """Exception raised for insufficient balance errors."""


class MarketDataError(ExecutionEngineError):
    """Exception raised for market data related errors."""


class OrderTimeoutError(ExecutionEngineError):
    """Exception raised when order execution times out."""


class CircuitBreakerError(ExecutionEngineError):
    """Exception raised when circuit breaker is open."""


class ExecutionEngineConfigError(ExecutionEngineError):
    """Exception raised for configuration errors."""


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification and handling."""

    VALIDATION = "validation"
    NETWORK = "network"
    EXCHANGE = "exchange"
    RATE_LIMIT = "rate_limit"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    MARKET_DATA = "market_data"
    TIMEOUT = "timeout"
    SYSTEM = "system"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorDetails:
    """Comprehensive error information for tracking and analysis."""

    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    timestamp: int
    retry_count: int = 0
    order_id: Optional[str] = None
    symbol: Optional[str] = None
    error_type: str = ""
    stack_trace: Optional[str] = None
    recovery_suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error details to dictionary for logging/events."""
        return {
            "error_id": self.error_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "retry_count": self.retry_count,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "error_type": self.error_type,
            "stack_trace": self.stack_trace,
            "recovery_suggestion": self.recovery_suggestion,
            "metadata": self.metadata,
        }


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for preventing cascading failures."""

    is_open: bool = False
    failure_count: int = 0
    last_failure_time: Optional[int] = None
    total_requests: int = 0
    success_count: int = 0
    failure_threshold: int = 5
    timeout_seconds: int = 60
    half_open_max_calls: int = 3

    def should_allow_request(self) -> bool:
        """Check if circuit breaker should allow request."""
        current_time = int(time.time())

        if not self.is_open:
            return True

        # Check if timeout period has passed (move to half-open)
        if (
            self.last_failure_time
            and current_time - self.last_failure_time >= self.timeout_seconds
        ):
            return True

        return False

    def record_success(self) -> None:
        """Record successful request."""
        self.success_count += 1
        self.total_requests += 1
        self.failure_count = 0
        self.is_open = False

    def record_failure(self) -> None:
        """Record failed request and potentially open circuit."""
        self.failure_count += 1
        self.total_requests += 1
        self.last_failure_time = int(time.time())

        if self.failure_count >= self.failure_threshold:
            self.is_open = True

    def get_statistics(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "is_open": self.is_open,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "success_rate": (
                self.success_count / self.total_requests
                if self.total_requests > 0
                else 0.0
            ),
            "last_failure_time": self.last_failure_time,
        }


class OrderStatus(Enum):
    """Order status enumeration for tracking order lifecycle."""

    PENDING_VALIDATION = "pending_validation"
    VALIDATED = "validated"
    PENDING_EXECUTION = "pending_execution"
    EXECUTING = "executing"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class FillDetail:
    """Individual fill information from order execution."""

    price: float
    quantity: Decimal
    commission: float
    commission_asset: str
    trade_id: Optional[str] = None
    timestamp: Optional[int] = None


@dataclass
class ExecutionResult:
    """Result of order execution attempt.

    Contains comprehensive execution details, status, fill information,
    commission data, and execution quality metrics.
    """

    order_request: OrderRequest
    execution_status: OrderStatus
    execution_timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    # Core execution details
    exchange_order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    filled_quantity: Decimal = Decimal("0")
    filled_price: Optional[float] = None
    average_fill_price: Optional[float] = None

    # Enhanced fill information
    number_of_fills: int = 0
    first_fill_time: Optional[int] = None
    last_fill_time: Optional[int] = None
    individual_fills: List[FillDetail] = field(default_factory=list)

    # Commission and fee details
    commission_amount: float = 0.0
    commission_asset: str = ""
    total_commission_usd: float = 0.0
    fee_breakdown: Dict[str, float] = field(default_factory=dict)

    # Market data at execution
    market_price_at_execution: Optional[float] = None
    price_impact_percentage: Optional[float] = None
    slippage_percentage: Optional[float] = None

    # Transaction details
    transaction_time: Optional[int] = None
    order_list_id: Optional[str] = None

    # Execution quality metrics
    execution_latency_ms: Optional[float] = None
    execution_efficiency_score: Optional[float] = None
    time_to_fill_ms: Optional[float] = None

    # Status tracking
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    retry_count: int = 0
    validation_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate execution result after initialization."""
        if self.filled_quantity < 0:
            raise ExecutionEngineError("Filled quantity cannot be negative")
        if self.retry_count < 0:
            raise ExecutionEngineError("Retry count cannot be negative")
        if self.commission_amount < 0:
            raise ExecutionEngineError("Commission amount cannot be negative")
        if self.total_commission_usd < 0:
            raise ExecutionEngineError("Total commission USD cannot be negative")

    def is_successful(self) -> bool:
        """Check if execution was successful."""
        return self.execution_status == OrderStatus.EXECUTED

    def is_failed(self) -> bool:
        """Check if execution failed."""
        return self.execution_status in [
            OrderStatus.FAILED,
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
        ]

    def is_partial_fill(self) -> bool:
        """Check if order is partially filled."""
        return (
            self.filled_quantity > 0
            and self.filled_quantity < self.order_request.quantity
        )

    def is_complete_fill(self) -> bool:
        """Check if order is completely filled."""
        return self.filled_quantity >= self.order_request.quantity

    def is_over_fill(self) -> bool:
        """Check if order is over-filled (rare but possible)."""
        return self.filled_quantity > self.order_request.quantity

    def get_fill_percentage(self) -> float:
        """Calculate fill percentage of the order."""
        if self.order_request.quantity <= 0:
            return 0.0
        return float(self.filled_quantity / self.order_request.quantity) * 100.0

    def get_unfilled_quantity(self) -> Decimal:
        """Get remaining unfilled quantity."""
        return max(Decimal("0"), self.order_request.quantity - self.filled_quantity)

    def get_total_trade_value(self) -> float:
        """Calculate total trade value excluding fees."""
        if not self.average_fill_price:
            return 0.0
        return float(self.filled_quantity) * self.average_fill_price

    def get_net_trade_value(self) -> float:
        """Calculate net trade value including fees."""
        return self.get_total_trade_value() - self.total_commission_usd

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of execution result."""
        return {
            "symbol": self.order_request.symbol,
            "status": self.execution_status.value,
            "exchange_order_id": self.exchange_order_id,
            "client_order_id": self.client_order_id,
            "requested_quantity": float(self.order_request.quantity),
            "filled_quantity": float(self.filled_quantity),
            "unfilled_quantity": float(self.get_unfilled_quantity()),
            "fill_percentage": self.get_fill_percentage(),
            "is_partial_fill": self.is_partial_fill(),
            "is_complete_fill": self.is_complete_fill(),
            "average_fill_price": self.average_fill_price,
            "number_of_fills": self.number_of_fills,
            "commission_amount": self.commission_amount,
            "commission_asset": self.commission_asset,
            "total_commission_usd": self.total_commission_usd,
            "slippage_percentage": self.slippage_percentage,
            "price_impact_percentage": self.price_impact_percentage,
            "execution_latency_ms": self.execution_latency_ms,
            "time_to_fill_ms": self.time_to_fill_ms,
            "execution_efficiency_score": self.execution_efficiency_score,
            "total_trade_value": self.get_total_trade_value(),
            "net_trade_value": self.get_net_trade_value(),
            "retry_count": self.retry_count,
            "has_errors": bool(self.error_message or self.validation_errors),
            "warnings_count": len(self.warnings),
        }


@dataclass
class ExecutionEngineConfig:
    """Configuration for execution engine operations."""

    # Validation settings
    enable_pre_execution_validation: bool = True
    enable_order_amount_validation: bool = True
    enable_price_validation: bool = True
    max_price_deviation_percentage: float = 5.0

    # Execution settings
    enable_order_execution: bool = True
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    execution_timeout_seconds: float = 30.0

    # Risk limits
    max_order_value_usd: float = 10000.0
    min_order_value_usd: float = 10.0
    max_daily_order_count: int = 100

    # Monitoring settings
    enable_execution_monitoring: bool = True
    log_all_order_requests: bool = True
    log_execution_details: bool = True

    # Order status monitoring settings
    enable_background_monitoring: bool = True
    monitoring_interval_seconds: float = 5.0
    max_monitoring_duration_seconds: float = 3600.0  # 1 hour timeout

    # Error handling and resilience settings
    enable_circuit_breaker: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60
    enable_advanced_error_tracking: bool = True
    max_error_history_size: int = 1000

    # Retry strategy settings
    enable_exponential_backoff: bool = True
    max_retry_delay_seconds: float = 30.0
    backoff_multiplier: float = 2.0
    jitter_enabled: bool = True

    # Rate limiting settings
    max_requests_per_minute: int = 1200  # Binance spot trading limit
    enable_rate_limit_protection: bool = True

    # Error notification settings
    enable_error_notifications: bool = True
    critical_error_notification_threshold: int = 3
    error_notification_cooldown_seconds: int = 300  # 5 minutes

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.max_retry_attempts < 0:
            raise ExecutionEngineError("Max retry attempts cannot be negative")
        if self.retry_delay_seconds < 0:
            raise ExecutionEngineError("Retry delay cannot be negative")
        if self.execution_timeout_seconds <= 0:
            raise ExecutionEngineError("Execution timeout must be positive")
        if self.max_order_value_usd <= 0:
            raise ExecutionEngineError("Max order value must be positive")
        if self.min_order_value_usd <= 0:
            raise ExecutionEngineError("Min order value must be positive")
        if self.max_order_value_usd <= self.min_order_value_usd:
            raise ExecutionEngineError("Max order value must exceed min order value")
        if self.monitoring_interval_seconds <= 0:
            raise ExecutionEngineError("Monitoring interval must be positive")
        if self.max_monitoring_duration_seconds <= 0:
            raise ExecutionEngineError("Max monitoring duration must be positive")


class IExecutionEngine(ABC):
    """Abstract interface for execution engine implementations."""

    @abstractmethod
    async def process_order_request(
        self, order_request: OrderRequest
    ) -> ExecutionResult:
        """Process order request and execute if valid.

        Args:
            order_request: Order request to process and execute

        Returns:
            ExecutionResult: Result of execution attempt

        Raises:
            ExecutionEngineError: If processing fails
        """

    @abstractmethod
    def update_config(self, config: ExecutionEngineConfig) -> None:
        """Update execution engine configuration.

        Args:
            config: New execution engine configuration

        Raises:
            ExecutionEngineError: If configuration is invalid
        """

    @abstractmethod
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution engine statistics.

        Returns:
            Dictionary containing execution statistics
        """


class ExecutionEngine(IExecutionEngine):
    """Order execution engine for processing and executing trading orders.

    This class handles the complete order execution workflow from receiving
    ORDER_REQUEST_GENERATED events to validating and executing orders.
    It follows SOLID principles and provides comprehensive error handling.

    Attributes:
        _config: Execution engine configuration
        _event_hub: Event hub for subscribing to events
        _logger: Logger instance for execution engine logging
        _processed_orders_count: Counter for processed orders
        _successful_executions_count: Counter for successful executions
        _failed_executions_count: Counter for failed executions
        _daily_order_count: Counter for daily orders
        _execution_history: List of recent execution results
    """

    def __init__(
        self,
        config: ExecutionEngineConfig,
        event_hub: EventHub,
        binance_client: IExchangeClient,
    ) -> None:
        """Initialize execution engine with configuration and dependencies.

        Args:
            config: Execution engine configuration
            event_hub: Event hub for subscribing to events
            binance_client: Binance client for order execution

        Raises:
            ExecutionEngineError: If initialization fails
        """
        if not isinstance(config, ExecutionEngineConfig):
            raise ExecutionEngineError("Config must be ExecutionEngineConfig instance")

        self._config = config
        self._event_hub = event_hub
        self._binance_client = binance_client
        self._logger = get_module_logger("execution_engine")

        # Statistics tracking
        self._processed_orders_count = 0
        self._successful_executions_count = 0
        self._failed_executions_count = 0
        self._daily_order_count = 0
        self._execution_history: List[ExecutionResult] = []

        # Order tracking infrastructure
        self._active_orders: Dict[str, Dict[str, Any]] = {}
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_running = False

        # Error handling and resilience infrastructure
        self._error_history: List[ErrorDetails] = []
        self._circuit_breaker = CircuitBreakerState(
            failure_threshold=config.circuit_breaker_failure_threshold,
            timeout_seconds=config.circuit_breaker_timeout_seconds,
        )
        self._last_error_notification_time = 0
        self._consecutive_critical_errors = 0
        self._error_counter = 0

        # Rate limiting infrastructure
        self._request_timestamps: List[int] = []
        self._rate_limit_lock = asyncio.Lock()

        # Subscribe to order request events
        self._event_hub.subscribe(
            EventType.ORDER_REQUEST_GENERATED, self._handle_order_request_event
        )

        self._logger.info(
            "Initialized execution engine with Binance client integration and "
            f"error handling (circuit_breaker={config.enable_circuit_breaker}, "
            f"error_tracking={config.enable_advanced_error_tracking})"
        )

    async def process_order_request(
        self, order_request: OrderRequest
    ) -> ExecutionResult:
        """Process order request through complete execution workflow.

        Args:
            order_request: Order request to process and execute

        Returns:
            ExecutionResult: Result of execution attempt

        Raises:
            ExecutionEngineError: If processing fails
        """
        try:
            self._processed_orders_count += 1
            start_time = time.time()

            self._logger.info(
                f"Processing order request: {order_request.symbol} "
                f"{order_request.order_type.value} qty={order_request.quantity} "
                f"price={order_request.price}"
            )

            # Step 1: Pre-execution validation
            validation_result = await self._validate_order_request(order_request)
            if not validation_result["valid"]:
                execution_result = ExecutionResult(
                    order_request=order_request,
                    execution_status=OrderStatus.REJECTED,
                    error_message=validation_result["reason"],
                    validation_errors=validation_result.get("errors", []),
                )
                self._failed_executions_count += 1
                self._logger.warning(
                    f"Order validation failed: {validation_result['reason']}"
                )
                await self._publish_execution_result(execution_result)
                return execution_result

            # Step 2: Check daily limits
            if not await self._check_daily_limits(order_request):
                execution_result = ExecutionResult(
                    order_request=order_request,
                    execution_status=OrderStatus.REJECTED,
                    error_message="Daily order limit exceeded",
                )
                self._failed_executions_count += 1
                self._logger.warning("Order rejected: daily limit exceeded")
                await self._publish_execution_result(execution_result)
                return execution_result

            # Step 3: Prepare for execution (placeholder for BinanceClient integration)
            execution_result = await self._execute_order_request(order_request)

            # Step 4: Update statistics and publish result
            processing_time = time.time() - start_time
            execution_result.execution_latency_ms = processing_time * 1000

            if execution_result.is_successful():
                self._successful_executions_count += 1
                self._daily_order_count += 1
                self._logger.info(
                    f"Order executed successfully: {order_request.symbol} "
                    f"id={execution_result.exchange_order_id} "
                    f"filled={execution_result.filled_quantity} "
                    f"latency={execution_result.execution_latency_ms:.2f}ms"
                )
            else:
                self._failed_executions_count += 1
                self._logger.error(
                    f"Order execution failed: {order_request.symbol} "
                    f"status={execution_result.execution_status.value} "
                    f"error={execution_result.error_message}"
                )

            # Store execution history (keep last 100 results)
            self._execution_history.append(execution_result)
            if len(self._execution_history) > 100:
                self._execution_history.pop(0)

            await self._publish_execution_result(execution_result)
            return execution_result

        except Exception as e:
            self._failed_executions_count += 1
            self._logger.error(f"Error processing order request: {e}")

            error_result = ExecutionResult(
                order_request=order_request,
                execution_status=OrderStatus.FAILED,
                error_message=str(e),
            )
            await self._publish_execution_result(error_result)
            raise ExecutionEngineError(f"Order processing failed: {e}")

    def update_config(self, config: ExecutionEngineConfig) -> None:
        """Update execution engine configuration.

        Args:
            config: New execution engine configuration

        Raises:
            ExecutionEngineError: If configuration is invalid
        """
        if not isinstance(config, ExecutionEngineConfig):
            raise ExecutionEngineError("Config must be ExecutionEngineConfig instance")

        self._config = config
        self._logger.info("Updated execution engine configuration")

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive execution engine statistics.

        Including error handling metrics.

        Returns:
            Dictionary containing execution statistics
        """
        success_rate = 0.0
        if self._processed_orders_count > 0:
            success_rate = (
                self._successful_executions_count / self._processed_orders_count
            )

        return {
            "processed_orders": self._processed_orders_count,
            "successful_executions": self._successful_executions_count,
            "failed_executions": self._failed_executions_count,
            "daily_order_count": self._daily_order_count,
            "success_rate": success_rate,
            "recent_executions": len(self._execution_history),
            "monitoring_status": self.get_monitoring_status(),
            "error_statistics": self.get_error_statistics(),
            "config_summary": {
                "execution_enabled": self._config.enable_order_execution,
                "max_retry_attempts": self._config.max_retry_attempts,
                "max_order_value": self._config.max_order_value_usd,
                "daily_order_limit": self._config.max_daily_order_count,
                "monitoring_enabled": self._config.enable_background_monitoring,
                "monitoring_interval": self._config.monitoring_interval_seconds,
                # Enhanced error handling configuration
                "error_handling_enabled": {
                    "circuit_breaker": self._config.enable_circuit_breaker,
                    "advanced_error_tracking": (
                        self._config.enable_advanced_error_tracking
                    ),
                    "error_notifications": self._config.enable_error_notifications,
                    "exponential_backoff": self._config.enable_exponential_backoff,
                    "rate_limit_protection": self._config.enable_rate_limit_protection,
                },
                "retry_configuration": {
                    "max_retry_attempts": self._config.max_retry_attempts,
                    "base_retry_delay": self._config.retry_delay_seconds,
                    "max_retry_delay": self._config.max_retry_delay_seconds,
                    "backoff_multiplier": self._config.backoff_multiplier,
                    "jitter_enabled": self._config.jitter_enabled,
                },
                "circuit_breaker_config": {
                    "failure_threshold": self._config.circuit_breaker_failure_threshold,
                    "timeout_seconds": self._config.circuit_breaker_timeout_seconds,
                },
                "notification_config": {
                    "critical_error_threshold": (
                        self._config.critical_error_notification_threshold
                    ),
                    "cooldown_seconds": (
                        self._config.error_notification_cooldown_seconds
                    ),
                },
            },
        }

    async def _handle_order_request_event(self, event_data: Dict[str, Any]) -> None:
        """Handle ORDER_REQUEST_GENERATED event.

        Args:
            event_data: Event data containing order request
        """
        try:
            order_request = event_data.get("order_request")

            if not order_request:
                self._logger.warning("Received order request event without order data")
                return

            if not isinstance(order_request, OrderRequest):
                self._logger.warning(
                    f"Invalid order request type: {type(order_request)}"
                )
                return

            # Process order request asynchronously
            execution_result = await self.process_order_request(order_request)

            if execution_result.is_successful():
                self._logger.debug(
                    f"Successfully processed order request event: "
                    f"{order_request.symbol}"
                )
            else:
                self._logger.debug(
                    f"Order request processing failed: {order_request.symbol} "
                    f"status={execution_result.execution_status.value}"
                )

        except Exception as e:
            self._logger.error(f"Error handling order request event: {e}")

    async def _validate_order_request(
        self, order_request: OrderRequest
    ) -> Dict[str, Any]:
        """Validate order request before execution.

        Args:
            order_request: Order request to validate

        Returns:
            Dictionary with validation result
        """
        if not self._config.enable_pre_execution_validation:
            return {"valid": True, "reason": "Validation disabled"}

        errors = []

        # Check basic order data
        if order_request.quantity <= 0:
            errors.append("Order quantity must be positive")

        if order_request.price <= 0:
            errors.append("Order price must be positive")

        if not order_request.symbol:
            errors.append("Order symbol cannot be empty")

        # Check order value limits
        if self._config.enable_order_amount_validation:
            order_value = float(order_request.quantity) * order_request.price

            if order_value > self._config.max_order_value_usd:
                errors.append(
                    f"Order value ${order_value:.2f} exceeds maximum "
                    f"${self._config.max_order_value_usd:.2f}"
                )

            if order_value < self._config.min_order_value_usd:
                errors.append(
                    f"Order value ${order_value:.2f} below minimum "
                    f"${self._config.min_order_value_usd:.2f}"
                )

        # Check price validation (placeholder for market price comparison)
        if self._config.enable_price_validation:
            # TODO: Implement market price validation when market data is available
            pass

        # Check order type support
        supported_order_types = [OrderType.MARKET, OrderType.LIMIT]
        if order_request.order_type not in supported_order_types:
            errors.append(f"Unsupported order type: {order_request.order_type.value}")

        if errors:
            return {
                "valid": False,
                "reason": f"Validation failed: {'; '.join(errors)}",
                "errors": errors,
            }

        return {"valid": True, "reason": "Validation passed"}

    async def _check_daily_limits(self, order_request: OrderRequest) -> bool:
        """Check if order exceeds daily limits.

        Args:
            order_request: Order request to check

        Returns:
            True if within limits, False otherwise
        """
        if self._daily_order_count >= self._config.max_daily_order_count:
            self._logger.warning(
                f"Daily order limit reached: {self._daily_order_count}/"
                f"{self._config.max_daily_order_count}"
            )
            return False

        return True

    async def _execute_order_request(
        self, order_request: OrderRequest
    ) -> ExecutionResult:
        """Execute order request via BinanceClient with comprehensive error handling.

        Args:
            order_request: Order request to execute

        Returns:
            ExecutionResult: Result of execution attempt
        """
        if not self._config.enable_order_execution:
            return ExecutionResult(
                order_request=order_request,
                execution_status=OrderStatus.REJECTED,
                error_message="Order execution disabled",
            )

        # Check Binance client connection
        if not self._binance_client.is_connected():
            return ExecutionResult(
                order_request=order_request,
                execution_status=OrderStatus.FAILED,
                error_message="Binance client not connected",
            )

        # Convert order data to Binance format
        binance_params = self._convert_to_binance_format(order_request)

        # Check circuit breaker before attempting execution
        try:
            await self._check_circuit_breaker({"symbol": order_request.symbol})
            await self._check_rate_limits()
        except (CircuitBreakerError, NetworkError) as e:
            context = {
                "symbol": order_request.symbol,
                "order_type": order_request.order_type.value,
            }
            await self._handle_error(e, context)
            return self._create_error_result(order_request, str(e), 0)

        # Execute order with enhanced retry logic and error handling
        for attempt in range(self._config.max_retry_attempts):
            try:
                self._logger.info(
                    f"Executing order (attempt {attempt + 1}): {order_request.symbol} "
                    f"{order_request.order_type.value} qty={order_request.quantity} "
                    f"price={order_request.price}"
                )

                # Execute order based on type
                order_response = await self._execute_binance_order(
                    order_request.order_type, binance_params
                )

                # Record successful execution for circuit breaker
                if self._config.enable_circuit_breaker:
                    self._circuit_breaker.record_success()

                # Process successful response
                execution_result = self._process_order_response(
                    order_request, order_response, attempt
                )

                # Start monitoring for orders that need it (not immediately filled)
                if (
                    execution_result.exchange_order_id
                    and execution_result.execution_status in [OrderStatus.EXECUTING]
                    and self._config.enable_background_monitoring
                ):
                    await self._add_order_to_monitoring(execution_result)

                self._logger.info(
                    f"Order executed successfully: {order_request.symbol} "
                    f"id={execution_result.exchange_order_id} "
                    f"filled={execution_result.filled_quantity}"
                )

                return execution_result

            except BinanceRateLimitError as e:
                context = {
                    "symbol": order_request.symbol,
                    "attempt": attempt + 1,
                    "retry_count": attempt,
                    "order_type": order_request.order_type.value,
                }
                error_details = await self._handle_error(e, context)
                error_details.retry_count = attempt

                self._logger.warning(f"Rate limit hit on attempt {attempt + 1}: {e}")
                if attempt < self._config.max_retry_attempts - 1:
                    await self._handle_retry_delay(attempt, "rate_limit")
                    continue
                return self._create_error_result(
                    order_request,
                    f"Rate limit exceeded after {attempt + 1} attempts: {e}",
                    attempt,
                )

            except BinanceOrderError as e:
                context = {
                    "symbol": order_request.symbol,
                    "attempt": attempt + 1,
                    "retry_count": attempt,
                    "order_type": order_request.order_type.value,
                    "binance_error_type": "order_error",
                }
                error_details = await self._handle_error(e, context)
                error_details.retry_count = attempt

                self._logger.error(f"Order error on attempt {attempt + 1}: {e}")
                # Don't retry order errors (they're usually permanent)
                return self._create_error_result(
                    order_request, f"Order execution failed: {e}", attempt
                )

            except BinanceError as e:
                context = {
                    "symbol": order_request.symbol,
                    "attempt": attempt + 1,
                    "retry_count": attempt,
                    "order_type": order_request.order_type.value,
                    "binance_error_type": "api_error",
                }
                error_details = await self._handle_error(e, context)
                error_details.retry_count = attempt

                self._logger.error(f"Binance error on attempt {attempt + 1}: {e}")
                if attempt < self._config.max_retry_attempts - 1:
                    await self._handle_retry_delay(attempt, "api_error")
                    continue
                return self._create_error_result(
                    order_request,
                    f"Binance API error after {attempt + 1} attempts: {e}",
                    attempt,
                )

            except Exception as e:
                context = {
                    "symbol": order_request.symbol,
                    "attempt": attempt + 1,
                    "retry_count": attempt,
                    "order_type": order_request.order_type.value,
                    "error_type": "unexpected",
                }
                error_details = await self._handle_error(e, context)
                error_details.retry_count = attempt

                self._logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < self._config.max_retry_attempts - 1:
                    await self._handle_retry_delay(attempt, "general_error")
                    continue
                return self._create_error_result(
                    order_request,
                    f"Unexpected error after {attempt + 1} attempts: {e}",
                    attempt,
                )

        # This should never be reached due to the retry logic above
        return self._create_error_result(
            order_request,
            "All retry attempts exhausted",
            self._config.max_retry_attempts - 1,
        )

    def _convert_to_binance_format(self, order_request: OrderRequest) -> Dict[str, Any]:
        """Convert OrderRequest to Binance API format.

        Args:
            order_request: Order request to convert

        Returns:
            Dictionary with Binance-formatted parameters
        """
        # Convert SignalType to Binance side
        side = self._convert_signal_to_side(order_request.signal.signal_type)

        binance_params = {
            "symbol": order_request.symbol,
            "side": side,
            "quantity": str(order_request.quantity),
        }

        # Add price for limit orders
        if order_request.order_type == OrderType.LIMIT:
            binance_params["price"] = str(order_request.price)

        return binance_params

    def _convert_signal_to_side(self, signal_type: SignalType) -> str:
        """Convert trading signal type to Binance order side.

        Args:
            signal_type: Trading signal type

        Returns:
            Binance order side ('BUY' or 'SELL')

        Raises:
            ExecutionEngineError: If signal type is unsupported
        """
        if signal_type == SignalType.BUY:
            return "BUY"
        elif signal_type == SignalType.SELL:
            return "SELL"
        else:
            raise ExecutionEngineError(f"Unsupported signal type: {signal_type}")

    async def _execute_binance_order(
        self, order_type: OrderType, binance_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute order via Binance client based on order type.

        Args:
            order_type: Type of order to execute
            binance_params: Binance-formatted order parameters

        Returns:
            Binance order response

        Raises:
            ExecutionEngineError: If order type is unsupported
            BinanceError: If order execution fails
        """
        if order_type == OrderType.MARKET:
            return self._binance_client.place_market_order(
                symbol=binance_params["symbol"],
                side=binance_params["side"],
                quantity=binance_params["quantity"],
            )
        elif order_type == OrderType.LIMIT:
            return self._binance_client.place_limit_order(
                symbol=binance_params["symbol"],
                side=binance_params["side"],
                quantity=binance_params["quantity"],
                price=binance_params["price"],
            )
        else:
            raise ExecutionEngineError(f"Unsupported order type: {order_type}")

    def _process_order_response(
        self,
        order_request: OrderRequest,
        order_response: Dict[str, Any],
        retry_count: int,
    ) -> ExecutionResult:
        """Process Binance order response into comprehensive ExecutionResult.

        Args:
            order_request: Original order request
            order_response: Binance order response
            retry_count: Number of retry attempts made

        Returns:
            ExecutionResult with detailed order execution information
        """
        # Extract basic order details from Binance response
        exchange_order_id = order_response.get(
            "orderId", str(order_response.get("id", ""))
        )
        client_order_id = order_response.get("clientOrderId")
        transaction_time = order_response.get("transactTime")
        order_list_id = order_response.get("orderListId")

        # Parse filled quantity
        filled_quantity = Decimal(str(order_response.get("executedQty", "0")))

        # Process individual fills with detailed analysis
        fills = order_response.get("fills", [])
        fill_details = self._process_individual_fills(fills)

        # Calculate comprehensive fill statistics
        fill_statistics = self._calculate_fill_statistics(
            fills, filled_quantity, order_request
        )

        # Calculate execution quality metrics
        quality_metrics = self._calculate_execution_quality_metrics(
            order_request, fill_statistics, transaction_time
        )

        # Determine order status based on Binance response
        binance_status = order_response.get("status", "").upper()
        execution_status = self._determine_execution_status(binance_status)

        # Create comprehensive execution result
        execution_result = ExecutionResult(
            order_request=order_request,
            execution_status=execution_status,
            exchange_order_id=str(exchange_order_id),
            client_order_id=client_order_id,
            filled_quantity=filled_quantity,
            filled_price=fill_statistics["average_fill_price"],
            average_fill_price=fill_statistics["average_fill_price"],
            number_of_fills=len(fills),
            first_fill_time=fill_statistics["first_fill_time"],
            last_fill_time=fill_statistics["last_fill_time"],
            individual_fills=fill_details,
            commission_amount=fill_statistics["total_commission"],
            commission_asset=fill_statistics["primary_commission_asset"],
            total_commission_usd=fill_statistics["total_commission_usd"],
            fee_breakdown=fill_statistics["fee_breakdown"],
            market_price_at_execution=quality_metrics["market_price_at_execution"],
            price_impact_percentage=quality_metrics["price_impact_percentage"],
            slippage_percentage=quality_metrics["slippage_percentage"],
            transaction_time=transaction_time,
            order_list_id=order_list_id,
            execution_efficiency_score=quality_metrics["efficiency_score"],
            time_to_fill_ms=quality_metrics["time_to_fill_ms"],
            retry_count=retry_count,
            metadata={
                "binance_status": binance_status,
                "binance_response": order_response,
                "execution_method": "binance_api",
                "fill_analysis": fill_statistics,
                "quality_metrics": quality_metrics,
            },
        )

        # Add comprehensive warnings and analysis
        self._add_execution_warnings(execution_result, order_request, binance_status)

        return execution_result

    def _process_individual_fills(
        self, fills: List[Dict[str, Any]]
    ) -> List[FillDetail]:
        """Process individual fills from Binance response into FillDetail objects.

        Args:
            fills: List of fill data from Binance API response

        Returns:
            List of FillDetail objects with processed fill information
        """
        fill_details = []

        for fill in fills:
            try:
                fill_detail = FillDetail(
                    price=float(fill.get("price", "0")),
                    quantity=Decimal(str(fill.get("qty", "0"))),
                    commission=float(fill.get("commission", "0")),
                    commission_asset=fill.get("commissionAsset", ""),
                    trade_id=fill.get("tradeId"),
                    timestamp=int(time.time() * 1000),  # Current time as fallback
                )
                fill_details.append(fill_detail)

            except (ValueError, TypeError) as e:
                self._logger.warning(f"Error processing fill detail: {e}")
                continue

        return fill_details

    def _calculate_fill_statistics(
        self,
        fills: List[Dict[str, Any]],
        filled_quantity: Decimal,
        order_request: OrderRequest,
    ) -> Dict[str, Any]:
        """Calculate comprehensive fill statistics from individual fills.

        Args:
            fills: List of fill data from Binance API
            filled_quantity: Total filled quantity
            order_request: Original order request

        Returns:
            Dictionary containing detailed fill statistics
        """
        if not fills:
            return {
                "average_fill_price": order_request.price,
                "total_commission": 0.0,
                "primary_commission_asset": "",
                "total_commission_usd": 0.0,
                "fee_breakdown": {},
                "first_fill_time": None,
                "last_fill_time": None,
                "weighted_average_price": order_request.price,
                "price_variance": 0.0,
            }

        # Calculate volume-weighted average price
        total_qty = Decimal("0")
        total_value = Decimal("0")
        total_commission = 0.0
        fee_breakdown = {}
        fill_prices = []

        for fill in fills:
            try:
                fill_qty = Decimal(str(fill.get("qty", "0")))
                fill_price = Decimal(str(fill.get("price", "0")))
                commission = float(fill.get("commission", "0"))
                commission_asset = fill.get("commissionAsset", "")

                # Accumulate totals
                total_qty += fill_qty
                total_value += fill_qty * fill_price
                total_commission += commission

                # Track fees by asset
                if commission_asset:
                    fee_breakdown[commission_asset] = (
                        fee_breakdown.get(commission_asset, 0.0) + commission
                    )

                # Track price and time data
                fill_prices.append(float(fill_price))
                # Note: Binance doesn\'t provide individual fill timestamps

                # in order response
                # We'll estimate based on transaction time

            except (ValueError, TypeError) as e:
                self._logger.warning(f"Error processing fill for statistics: {e}")
                continue

        # Calculate average price
        average_fill_price = (
            float(total_value / total_qty) if total_qty > 0 else order_request.price
        )

        # Calculate price variance for execution quality
        price_variance = 0.0
        if len(fill_prices) > 1:
            avg_price = sum(fill_prices) / len(fill_prices)
            price_variance = sum((p - avg_price) ** 2 for p in fill_prices) / len(
                fill_prices
            )

        # Determine primary commission asset (most common)
        primary_commission_asset = ""
        if fee_breakdown:
            primary_commission_asset = max(fee_breakdown, key=fee_breakdown.get)

        # Estimate USD commission value (simplified - would need price
        # conversion in real system)
        total_commission_usd = self._estimate_commission_usd(
            fee_breakdown, order_request.symbol
        )

        return {
            "average_fill_price": average_fill_price,
            "total_commission": total_commission,
            "primary_commission_asset": primary_commission_asset,
            "total_commission_usd": total_commission_usd,
            "fee_breakdown": fee_breakdown,
            "first_fill_time": None,  # Would be available in real-time fills
            "last_fill_time": None,  # Would be available in real-time fills
            "weighted_average_price": average_fill_price,
            "price_variance": price_variance,
            "fill_count": len(fills),
        }

    def _calculate_execution_quality_metrics(
        self,
        order_request: OrderRequest,
        fill_statistics: Dict[str, Any],
        transaction_time: Optional[int],
    ) -> Dict[str, Any]:
        """Calculate execution quality metrics including slippage and efficiency.

        Args:
            order_request: Original order request
            fill_statistics: Calculated fill statistics
            transaction_time: Transaction timestamp from exchange

        Returns:
            Dictionary containing execution quality metrics
        """
        average_fill_price = fill_statistics["average_fill_price"]

        # Calculate slippage based on order type
        slippage_percentage = None
        if order_request.order_type == OrderType.MARKET:
            # For market orders, slippage is difference from expected market price
            # Note: In real system, would use market price at order time
            slippage_percentage = self._calculate_market_slippage(
                order_request.price,
                average_fill_price,
                order_request.signal.signal_type,
            )
        elif order_request.order_type == OrderType.LIMIT:
            # For limit orders, slippage is improvement vs limit price
            slippage_percentage = self._calculate_limit_slippage(
                order_request.price,
                average_fill_price,
                order_request.signal.signal_type,
            )

        # Calculate price impact (difference from quoted price to execution price)
        price_impact_percentage = self._calculate_price_impact(
            order_request.price, average_fill_price
        )

        # Calculate execution efficiency score
        efficiency_score = self._calculate_efficiency_score(
            fill_statistics, order_request, slippage_percentage
        )

        # Calculate time to fill
        time_to_fill_ms = None
        if transaction_time and hasattr(order_request, "creation_timestamp"):
            time_to_fill_ms = transaction_time - order_request.creation_timestamp

        return {
            "market_price_at_execution": average_fill_price,  # Simplified
            "price_impact_percentage": price_impact_percentage,
            "slippage_percentage": slippage_percentage,
            "efficiency_score": efficiency_score,
            "time_to_fill_ms": time_to_fill_ms,
            "execution_latency_estimate": time_to_fill_ms,
        }

    def _determine_execution_status(self, binance_status: str) -> OrderStatus:
        """Determine internal execution status from Binance status.

        Args:
            binance_status: Status string from Binance API

        Returns:
            Corresponding OrderStatus enum value
        """
        status_mapping = {
            "FILLED": OrderStatus.EXECUTED,
            "PARTIALLY_FILLED": OrderStatus.EXECUTED,
            "NEW": OrderStatus.EXECUTING,
            "PENDING": OrderStatus.EXECUTING,
            "CANCELED": OrderStatus.CANCELLED,
            "CANCELLED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.REJECTED,
        }

        return status_mapping.get(binance_status, OrderStatus.EXECUTING)

    def _add_execution_warnings(
        self,
        execution_result: ExecutionResult,
        order_request: OrderRequest,
        binance_status: str,
    ) -> None:
        """Add comprehensive warnings and analysis to execution result.

        Args:
            execution_result: Execution result to add warnings to
            order_request: Original order request
            binance_status: Binance order status
        """
        # Partial fill warnings
        if execution_result.is_partial_fill():
            fill_percentage = execution_result.get_fill_percentage()
            execution_result.warnings.append(
                f"Partial fill: {fill_percentage:.1f}% executed "
                f"({execution_result.filled_quantity}/{order_request.quantity})"
            )

        # Over-fill warnings (rare but possible)
        if execution_result.is_over_fill():
            over_fill_qty = execution_result.filled_quantity - order_request.quantity
            execution_result.warnings.append(
                f"Over-fill detected: {over_fill_qty} excess quantity filled"
            )

        # High slippage warnings
        if (
            execution_result.slippage_percentage
            and abs(execution_result.slippage_percentage) > 0.5
        ):
            execution_result.warnings.append(
                f"High slippage: {execution_result.slippage_percentage:.2f}%"
            )

        # Multiple fills efficiency warning
        if execution_result.number_of_fills > 5:
            execution_result.warnings.append(
                f"Order executed in {execution_result.number_of_fills} separate fills"
            )

        # High commission warning
        if execution_result.total_commission_usd > 0:
            commission_rate = (
                execution_result.total_commission_usd
                / execution_result.get_total_trade_value()
                * 100
            )
            if commission_rate > 0.1:  # More than 0.1% commission
                execution_result.warnings.append(
                    f"High commission rate: {commission_rate:.3f}%"
                )

    def _estimate_commission_usd(
        self, fee_breakdown: Dict[str, float], symbol: str
    ) -> float:
        """Estimate total commission value in USD.

        Args:
            fee_breakdown: Dictionary of commission amounts by asset
            symbol: Trading symbol for context

        Returns:
            Estimated commission value in USD
        """
        # Simplified estimation - in real system would use current market prices
        total_usd = 0.0

        for asset, amount in fee_breakdown.items():
            if asset in ["USDT", "USDC", "BUSD"]:
                # Stablecoins assumed to be 1:1 USD
                total_usd += amount
            elif asset == "BNB":
                # Simplified BNB price estimation
                total_usd += amount * 300  # Approximate BNB price
            else:
                # For other assets, make conservative estimate
                total_usd += amount * 0.1  # Very conservative estimate

        return total_usd

    def _calculate_market_slippage(
        self, expected_price: float, actual_price: float, signal_type: SignalType
    ) -> float:
        """Calculate slippage for market orders.

        Args:
            expected_price: Expected market price when order was placed
            actual_price: Actual average fill price
            signal_type: BUY or SELL signal

        Returns:
            Slippage percentage (negative means worse execution)
        """
        if expected_price <= 0:
            return 0.0

        price_diff = actual_price - expected_price

        # For buy orders, paying more is negative slippage
        # For sell orders, receiving less is negative slippage
        if signal_type == SignalType.BUY:
            slippage = -(price_diff / expected_price) * 100
        else:  # SELL
            slippage = (price_diff / expected_price) * 100

        return slippage

    def _calculate_limit_slippage(
        self, limit_price: float, actual_price: float, signal_type: SignalType
    ) -> float:
        """Calculate slippage for limit orders.

        Args:
            limit_price: Limit price specified in order
            actual_price: Actual average fill price
            signal_type: BUY or SELL signal

        Returns:
            Slippage percentage (positive means price improvement)
        """
        if limit_price <= 0:
            return 0.0

        price_diff = actual_price - limit_price

        # For buy orders, paying less is positive slippage (price improvement)
        # For sell orders, receiving more is positive slippage
        if signal_type == SignalType.BUY:
            slippage = -(price_diff / limit_price) * 100
        else:  # SELL
            slippage = (price_diff / limit_price) * 100

        return slippage

    def _calculate_price_impact(
        self, reference_price: float, execution_price: float
    ) -> float:
        """Calculate price impact as percentage difference.

        Args:
            reference_price: Reference price for comparison
            execution_price: Actual execution price

        Returns:
            Price impact percentage
        """
        if reference_price <= 0:
            return 0.0

        return ((execution_price - reference_price) / reference_price) * 100

    def _calculate_efficiency_score(
        self,
        fill_statistics: Dict[str, Any],
        order_request: OrderRequest,
        slippage_percentage: Optional[float],
    ) -> float:
        """Calculate execution efficiency score (0-100).

        Args:
            fill_statistics: Fill statistics dictionary
            order_request: Original order request
            slippage_percentage: Calculated slippage

        Returns:
            Efficiency score between 0 and 100
        """
        score = 100.0

        # Penalize for slippage (if available)
        if slippage_percentage is not None:
            score -= min(abs(slippage_percentage) * 10, 30)  # Max 30 point penalty

        # Penalize for multiple fills (fragmentation)
        fill_count = fill_statistics.get("fill_count", 1)
        if fill_count > 1:
            score -= min((fill_count - 1) * 5, 20)  # Max 20 point penalty

        # Penalize for price variance (inconsistent execution)
        price_variance = fill_statistics.get("price_variance", 0.0)
        if price_variance > 0:
            score -= min(price_variance * 1000, 15)  # Max 15 point penalty

        return max(score, 0.0)

    def _create_error_result(
        self, order_request: OrderRequest, error_message: str, retry_count: int
    ) -> ExecutionResult:
        """Create ExecutionResult for failed orders.

        Args:
            order_request: Original order request
            error_message: Error description
            retry_count: Number of retry attempts made

        Returns:
            ExecutionResult with error details
        """
        return ExecutionResult(
            order_request=order_request,
            execution_status=OrderStatus.FAILED,
            error_message=error_message,
            retry_count=retry_count,
            metadata={
                "execution_method": "binance_api",
                "error_type": "execution_failure",
            },
        )

    async def _handle_retry_delay(self, attempt: int, error_type: str) -> None:
        """Handle retry delay with enhanced exponential backoff and jitter.

        Args:
            attempt: Current attempt number (0-based)
            error_type: Type of error that triggered retry
        """
        if self._config.enable_exponential_backoff:
            # Enhanced exponential backoff with configurable multiplier
            delay = self._config.retry_delay_seconds * (
                self._config.backoff_multiplier**attempt
            )
        else:
            # Linear backoff
            delay = self._config.retry_delay_seconds * (attempt + 1)

        # Cap maximum delay
        delay = min(delay, self._config.max_retry_delay_seconds)

        # Add jitter to prevent thundering herd
        if self._config.jitter_enabled:
            import random

            jitter_factor = random.uniform(0.5, 1.5)
            delay *= jitter_factor

        backoff_type = (
            "exponential" if self._config.enable_exponential_backoff else "linear"
        )
        self._logger.info(
            f"Retrying in {delay:.2f}s due to {error_type} "
            f"(attempt {attempt + 1}/{self._config.max_retry_attempts}, "
            f"backoff={backoff_type})"
        )

        await asyncio.sleep(delay)

    async def _publish_execution_result(
        self, execution_result: ExecutionResult
    ) -> None:
        """Publish execution result event.

        Args:
            execution_result: Execution result to publish
        """
        try:
            event_data = {
                "execution_result": execution_result,
                "symbol": execution_result.order_request.symbol,
                "status": execution_result.execution_status.value,
                "exchange_order_id": execution_result.exchange_order_id,
                "filled_quantity": float(execution_result.filled_quantity),
                "execution_summary": execution_result.get_execution_summary(),
                "timestamp": execution_result.execution_timestamp,
            }

            # Determine appropriate event type based on execution status
            if execution_result.is_successful():
                event_type = EventType.ORDER_FILLED
            elif execution_result.execution_status == OrderStatus.REJECTED:
                event_type = EventType.ORDER_REJECTED
            elif execution_result.execution_status == OrderStatus.CANCELLED:
                event_type = EventType.ORDER_CANCELLED
            else:
                event_type = EventType.ORDER_REJECTED  # Default for failed states

            self._event_hub.publish(event_type, event_data)

            self._logger.debug(
                f"Published execution result event: "
                f"{execution_result.order_request.symbol} "
                f"status={execution_result.execution_status.value}"
            )

        except Exception as e:
            self._logger.error(f"Error publishing execution result event: {e}")

    async def _add_order_to_monitoring(self, execution_result: ExecutionResult) -> None:
        """Add order to active monitoring system.

        Args:
            execution_result: Execution result containing order details
        """
        if not execution_result.exchange_order_id:
            self._logger.warning("Cannot monitor order without exchange_order_id")
            return

        order_metadata = {
            "symbol": execution_result.order_request.symbol,
            "order_id": execution_result.exchange_order_id,
            "last_status": execution_result.execution_status.value,
            "start_time": int(time.time()),
            "execution_result": execution_result,
            "status_changes": [],
            "last_checked": 0,
        }

        self._active_orders[execution_result.exchange_order_id] = order_metadata

        self._logger.info(
            f"Added order {execution_result.exchange_order_id} to monitoring: "
            f"{execution_result.order_request.symbol}"
        )

        # Start monitoring task if not already running
        await self._start_order_monitoring()

    async def _start_order_monitoring(self) -> None:
        """Start background monitoring task if not already running."""
        if self._monitoring_running:
            return

        if self._monitoring_task and not self._monitoring_task.done():
            return

        self._monitoring_running = True
        self._monitoring_task = asyncio.create_task(self._monitor_orders_loop())

        self._logger.info("Started order monitoring background task")

    async def _monitor_orders_loop(self) -> None:
        """
        Enhanced continuous loop for monitoring active orders with error resilience.
        """
        consecutive_errors = 0
        max_consecutive_errors = 5
        error_backoff_delay = 5.0

        try:
            while self._monitoring_running and self._active_orders:
                try:
                    await self._check_active_orders()
                    consecutive_errors = 0  # Reset on successful iteration
                    await asyncio.sleep(self._config.monitoring_interval_seconds)

                except Exception as e:
                    consecutive_errors += 1

                    context = {
                        "operation": "monitoring_loop",
                        "consecutive_errors": consecutive_errors,
                        "active_orders_count": len(self._active_orders),
                    }

                    # Handle error with comprehensive error handling
                    await self._handle_error(e, context)

                    # Implement backoff for consecutive errors
                    if consecutive_errors >= max_consecutive_errors:
                        self._logger.critical(
                            f"Too many consecutive monitoring errors "
                            f"({consecutive_errors}), stopping monitoring loop"
                        )
                        break

                    # Use exponential backoff for error recovery
                    delay = min(
                        error_backoff_delay * (2 ** (consecutive_errors - 1)),
                        60.0
                    )
                    self._logger.warning(
                        f"Monitoring error #{consecutive_errors}, "
                        f"backing off for {delay}s"
                    )
                    await asyncio.sleep(delay)

        except asyncio.CancelledError:
            self._logger.info("Order monitoring task cancelled")
        except Exception as e:
            context = {
                "operation": "monitoring_loop_fatal",
                "consecutive_errors": consecutive_errors,
            }
            await self._handle_error(e, context)
        finally:
            self._monitoring_running = False
            self._logger.info("Order monitoring task finished")

            # Publish monitoring stopped event if there were errors
            if consecutive_errors > 0:
                try:
                    event_data = {
                        "type": "monitoring_stopped",
                        "reason": (
                            "consecutive_errors"
                            if consecutive_errors >= max_consecutive_errors
                            else "normal"
                        ),
                        "consecutive_errors": consecutive_errors,
                        "active_orders_count": len(self._active_orders),
                        "timestamp": int(time.time()),
                    }
                    self._event_hub.publish(EventType.ERROR_OCCURRED, event_data)
                except Exception as publish_error:
                    self._logger.error(
                        f"Failed to publish monitoring stopped event: {publish_error}"
                    )

    async def _check_active_orders(self) -> None:
        """Check status of all active orders and process updates."""
        if not self._active_orders:
            return

        current_time = int(time.time())
        orders_to_remove = []

        for order_id, order_metadata in list(self._active_orders.items()):
            try:
                # Check if order has timed out
                monitoring_duration = current_time - order_metadata["start_time"]
                if monitoring_duration > self._config.max_monitoring_duration_seconds:
                    self._logger.warning(
                        f"Order {order_id} monitoring timeout after "
                        f"{monitoring_duration}s, removing from monitoring"
                    )
                    orders_to_remove.append(order_id)
                    continue

                # Query order status from exchange
                symbol = order_metadata["symbol"]
                order_status_response = await self._get_order_status_safe(
                    symbol, order_id
                )

                if not order_status_response:
                    continue

                # Process status update
                updated = await self._process_order_status_update(
                    order_id, order_metadata, order_status_response
                )

                # Check if order reached terminal state
                if updated and self._is_terminal_status(
                    order_status_response.get("status", "")
                ):
                    orders_to_remove.append(order_id)

            except Exception as e:
                self._logger.error(f"Error checking order {order_id}: {e}")

        # Remove completed or timed-out orders
        for order_id in orders_to_remove:
            self._active_orders.pop(order_id, None)
            self._logger.debug(f"Removed order {order_id} from monitoring")

        # Stop monitoring if no active orders remain
        if not self._active_orders and self._monitoring_running:
            self._monitoring_running = False
            self._logger.info("No active orders remaining, stopping monitoring")

    async def _get_order_status_safe(
        self, symbol: str, order_id: str
    ) -> Optional[Dict[str, Any]]:
        """Safely query order status with comprehensive error handling and resilience.

        Args:
            symbol: Trading pair symbol
            order_id: Exchange order ID

        Returns:
            Order status response or None if failed
        """
        context = {
            "symbol": symbol,
            "order_id": order_id,
            "operation": "order_status_query",
        }

        try:
            # Check circuit breaker for monitoring operations
            if self._config.enable_circuit_breaker:
                if not self._circuit_breaker.should_allow_request():
                    self._logger.warning(
                        f"Circuit breaker open, skipping status check for {order_id}"
                    )
                    return None

            # Check rate limits before making request
            try:
                await self._check_rate_limits()
            except NetworkError as e:
                self._logger.warning(
                    f"Rate limit protection triggered for {order_id}: {e}"
                )
                return None

            if not self._binance_client.is_connected():
                error = NetworkError("Binance client not connected")
                await self._handle_error(error, context)
                return None

            order_status = self._binance_client.get_order_status(symbol, order_id)

            # Record successful request for circuit breaker
            if self._config.enable_circuit_breaker:
                self._circuit_breaker.record_success()

            return order_status

        except BinanceRateLimitError:
            # Don't treat rate limits as failures for monitoring
            self._logger.warning(f"Rate limit hit while checking order {order_id}")
            # Add small delay to prevent rapid retries
            await asyncio.sleep(1.0)
            return None

        except BinanceError as e:
            context["binance_error_type"] = "monitoring_error"
            await self._handle_error(e, context)
            return None

        except Exception as e:
            context["error_type"] = "unexpected_monitoring_error"
            await self._handle_error(e, context)
            return None

    async def _process_order_status_update(
        self,
        order_id: str,
        order_metadata: Dict[str, Any],
        status_response: Dict[str, Any],
    ) -> bool:
        """Process order status update and publish events if status changed.

        Args:
            order_id: Exchange order ID
            order_metadata: Stored order metadata
            status_response: Current status response from exchange

        Returns:
            True if status was updated, False otherwise
        """
        current_status = status_response.get("status", "").upper()
        last_status = order_metadata.get("last_status", "").upper()

        # Update last checked time
        order_metadata["last_checked"] = int(time.time())

        # Check if status changed
        if current_status == last_status:
            return False

        # Log status change
        self._logger.info(
            f"Order {order_id} status changed: {last_status} -> {current_status}"
        )

        # Record status change
        status_change = {
            "timestamp": int(time.time()),
            "from_status": last_status,
            "to_status": current_status,
            "filled_qty": status_response.get("executedQty", "0"),
        }
        order_metadata["status_changes"].append(status_change)
        order_metadata["last_status"] = current_status

        # Update execution result with latest information
        execution_result = order_metadata["execution_result"]
        self._update_execution_result_from_status(execution_result, status_response)

        # Publish appropriate events based on new status
        await self._publish_status_change_event(
            execution_result, current_status, status_response
        )

        return True

    def _update_execution_result_from_status(
        self, execution_result: ExecutionResult, status_response: Dict[str, Any]
    ) -> None:
        """Update execution result with comprehensive latest status information.

        Args:
            execution_result: Execution result to update
            status_response: Latest status response from exchange
        """
        # Update filled quantity
        executed_qty = status_response.get("executedQty", "0")
        previous_qty = execution_result.filled_quantity
        execution_result.filled_quantity = Decimal(str(executed_qty))

        # Update average fill price if available
        if "avgPrice" in status_response and status_response["avgPrice"]:
            avg_price = float(status_response["avgPrice"])
            if avg_price > 0:
                execution_result.average_fill_price = avg_price
                execution_result.filled_price = avg_price

        # Update execution status using centralized method
        binance_status = status_response.get("status", "").upper()
        execution_result.execution_status = self._determine_execution_status(
            binance_status
        )

        # Update commission information if available
        cumulative_quote_qty = status_response.get("cummulativeQuoteQty")
        if cumulative_quote_qty:
            # Estimate commission based on trade value (typical rate is 0.1%)
            trade_value = float(cumulative_quote_qty)
            estimated_commission = trade_value * 0.001  # 0.1% commission estimate
            execution_result.total_commission_usd = max(
                execution_result.total_commission_usd, estimated_commission
            )

        # Update transaction time if available
        if "updateTime" in status_response:
            execution_result.transaction_time = status_response["updateTime"]

        # Calculate time to fill if this is a new fill
        if (
            execution_result.filled_quantity > previous_qty
            and execution_result.time_to_fill_ms is None
        ):
            current_time = int(time.time() * 1000)
            if hasattr(execution_result.order_request, "creation_timestamp"):
                execution_result.time_to_fill_ms = (
                    current_time - execution_result.order_request.creation_timestamp
                )

        # Update fill tracking
        if execution_result.filled_quantity > previous_qty:
            # New fill detected
            execution_result.last_fill_time = int(time.time() * 1000)
            if execution_result.first_fill_time is None:
                execution_result.first_fill_time = execution_result.last_fill_time

        # Add warnings for status changes
        if (
            execution_result.is_partial_fill()
            and execution_result.filled_quantity != previous_qty
        ):
            fill_percentage = execution_result.get_fill_percentage()
            new_warning = f"Status update: {fill_percentage:.1f}% filled"
            if new_warning not in execution_result.warnings:
                execution_result.warnings.append(new_warning)

    async def _publish_status_change_event(
        self,
        execution_result: ExecutionResult,
        current_status: str,
        status_response: Dict[str, Any],
    ) -> None:
        """Publish comprehensive event based on order status change with
        detailed fill data.

        Args:
            execution_result: Updated execution result with comprehensive data
            current_status: Current order status
            status_response: Full status response from exchange
        """
        try:
            # Create comprehensive event data with all execution details
            event_data = {
                "execution_result": execution_result,
                "symbol": execution_result.order_request.symbol,
                "status": execution_result.execution_status.value,
                "exchange_order_id": execution_result.exchange_order_id,
                "client_order_id": execution_result.client_order_id,
                "filled_quantity": float(execution_result.filled_quantity),
                "unfilled_quantity": float(execution_result.get_unfilled_quantity()),
                "average_fill_price": execution_result.average_fill_price,
                "fill_percentage": execution_result.get_fill_percentage(),
                "execution_summary": execution_result.get_execution_summary(),
                "timestamp": int(time.time()),
                # Enhanced fill information
                "fill_details": {
                    "number_of_fills": execution_result.number_of_fills,
                    "individual_fills": [
                        {
                            "price": fill.price,
                            "quantity": float(fill.quantity),
                            "commission": fill.commission,
                            "commission_asset": fill.commission_asset,
                            "trade_id": fill.trade_id,
                            "timestamp": fill.timestamp,
                        }
                        for fill in execution_result.individual_fills
                    ],
                    "first_fill_time": execution_result.first_fill_time,
                    "last_fill_time": execution_result.last_fill_time,
                    "is_partial_fill": execution_result.is_partial_fill(),
                    "is_complete_fill": execution_result.is_complete_fill(),
                    "is_over_fill": execution_result.is_over_fill(),
                },
                # Commission and fee information
                "commission_data": {
                    "commission_amount": execution_result.commission_amount,
                    "commission_asset": execution_result.commission_asset,
                    "total_commission_usd": execution_result.total_commission_usd,
                    "fee_breakdown": execution_result.fee_breakdown,
                },
                # Execution quality metrics
                "execution_quality": {
                    "slippage_percentage": execution_result.slippage_percentage,
                    "price_impact_percentage": execution_result.price_impact_percentage,
                    "execution_efficiency_score": (
                        execution_result.execution_efficiency_score
                    ),
                    "execution_latency_ms": execution_result.execution_latency_ms,
                    "time_to_fill_ms": execution_result.time_to_fill_ms,
                    "market_price_at_execution": (
                        execution_result.market_price_at_execution
                    ),
                },
                # Trade value information
                "trade_value": {
                    "total_trade_value": execution_result.get_total_trade_value(),
                    "net_trade_value": execution_result.get_net_trade_value(),
                    "commission_rate_percentage": (
                        (
                            execution_result.total_commission_usd
                            / execution_result.get_total_trade_value()
                            * 100
                        )
                        if execution_result.get_total_trade_value() > 0
                        else 0.0
                    ),
                },
                # Status change information
                "status_change": {
                    "new_status": current_status,
                    "binance_response": status_response,
                    "warnings": execution_result.warnings,
                    "has_warnings": len(execution_result.warnings) > 0,
                },
                # Transaction details
                "transaction_details": {
                    "transaction_time": execution_result.transaction_time,
                    "order_list_id": execution_result.order_list_id,
                    "execution_timestamp": execution_result.execution_timestamp,
                },
            }

            # Determine event type and log appropriate message
            if current_status == "FILLED":
                event_type = EventType.ORDER_FILLED
                self._logger.info(
                    f"Order {execution_result.exchange_order_id} fully filled: "
                    f"{execution_result.order_request.symbol} "
                    f"qty={execution_result.filled_quantity} "
                    f"avg_price={execution_result.average_fill_price} "
                    f"fills={execution_result.number_of_fills} "
                    f"commission_usd=${execution_result.total_commission_usd:.4f} "
                    f"efficiency={execution_result.execution_efficiency_score:.1f}"
                )
            elif current_status == "PARTIALLY_FILLED":
                event_type = EventType.ORDER_FILLED
                fill_pct = execution_result.get_fill_percentage()
                self._logger.info(
                    f"Order {execution_result.exchange_order_id} partially filled: "
                    f"{execution_result.order_request.symbol} "
                    f"{fill_pct:.1f}% "
                    f"({execution_result.filled_quantity}/"
                    f"{execution_result.order_request.quantity}) "
                    f"avg_price={execution_result.average_fill_price} "
                    f"fills={execution_result.number_of_fills}"
                )
            elif current_status in ["CANCELED", "CANCELLED"]:
                event_type = EventType.ORDER_CANCELLED
                self._logger.info(
                    f"Order {execution_result.exchange_order_id} cancelled: "
                    f"{execution_result.order_request.symbol} "
                    f"filled={execution_result.filled_quantity}/"
                    f"{execution_result.order_request.quantity}"
                )
            elif current_status in ["REJECTED", "EXPIRED"]:
                event_type = EventType.ORDER_REJECTED
                self._logger.info(
                    f"Order {execution_result.exchange_order_id} rejected: "
                    f"{execution_result.order_request.symbol} "
                    f"reason={current_status}"
                )
            else:
                # For other status changes, use ORDER_FILLED event
                event_type = EventType.ORDER_FILLED
                self._logger.debug(
                    f"Order {execution_result.exchange_order_id} status update: "
                    f"{execution_result.order_request.symbol} {current_status}"
                )

            # Publish comprehensive event
            self._event_hub.publish(event_type, event_data)

            # Log any warnings
            if execution_result.warnings:
                self._logger.warning(
                    f"Order {execution_result.exchange_order_id} warnings: "
                    f"{'; '.join(execution_result.warnings)}"
                )

        except Exception as e:
            self._logger.error(f"Error publishing status change event: {e}")

    def _is_terminal_status(self, status: str) -> bool:
        """Check if order status is terminal (monitoring should stop).

        Args:
            status: Order status from exchange

        Returns:
            True if status is terminal, False otherwise
        """
        terminal_statuses = {"FILLED", "CANCELED", "CANCELLED", "REJECTED", "EXPIRED"}
        return status.upper() in terminal_statuses

    def get_active_orders_count(self) -> int:
        """Get count of currently monitored orders.

        Returns:
            Number of orders being monitored
        """
        return len(self._active_orders)

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get detailed monitoring system status.

        Returns:
            Dictionary with monitoring system information
        """
        return {
            "monitoring_enabled": self._config.enable_background_monitoring,
            "monitoring_running": self._monitoring_running,
            "active_orders_count": len(self._active_orders),
            "monitoring_interval_seconds": self._config.monitoring_interval_seconds,
            "max_monitoring_duration_seconds": (
                self._config.max_monitoring_duration_seconds
            ),
            "active_orders": [
                {
                    "order_id": order_id,
                    "symbol": metadata["symbol"],
                    "last_status": metadata["last_status"],
                    "monitoring_duration": int(time.time()) - metadata["start_time"],
                    "status_changes_count": len(metadata["status_changes"]),
                }
                for order_id, metadata in self._active_orders.items()
            ],
        }

    async def stop_monitoring(self) -> None:
        """Stop order monitoring and cleanup resources."""
        self._monitoring_running = False

        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        self._active_orders.clear()
        self._logger.info("Order monitoring stopped and resources cleaned up")

    # Enhanced Error Handling and Resilience Methods

    def _classify_error(
        self, error: Exception, context: Dict[str, Any] = None
    ) -> ErrorDetails:
        """Classify error and create detailed error information.

        Args:
            error: Exception that occurred
            context: Additional context about the error

        Returns:
            ErrorDetails object with comprehensive error information
        """
        import traceback
        import uuid

        # Generate unique error ID
        error_id = str(uuid.uuid4())[:8]
        timestamp = int(time.time())

        # Initialize context if not provided
        if context is None:
            context = {}

        # Classify error by type
        category = ErrorCategory.UNKNOWN
        severity = ErrorSeverity.MEDIUM
        recovery_suggestion = None

        if isinstance(error, BinanceRateLimitError):
            category = ErrorCategory.RATE_LIMIT
            severity = ErrorSeverity.MEDIUM
            recovery_suggestion = (
                "Wait for rate limit reset and retry with exponential backoff"
            )
        elif isinstance(error, BinanceOrderError):
            category = ErrorCategory.EXCHANGE
            severity = ErrorSeverity.HIGH
            recovery_suggestion = "Check order parameters and account balance"
        elif isinstance(error, BinanceError):
            category = ErrorCategory.EXCHANGE
            severity = ErrorSeverity.HIGH
            recovery_suggestion = "Check exchange connectivity and API status"
        elif isinstance(error, OrderValidationError):
            category = ErrorCategory.VALIDATION
            severity = ErrorSeverity.MEDIUM
            recovery_suggestion = "Review order parameters and validation rules"
        elif isinstance(error, InsufficientBalanceError):
            category = ErrorCategory.INSUFFICIENT_BALANCE
            severity = ErrorSeverity.HIGH
            recovery_suggestion = "Check account balance and reduce order size"
        elif isinstance(error, NetworkError):
            category = ErrorCategory.NETWORK
            severity = ErrorSeverity.MEDIUM
            recovery_suggestion = "Check network connectivity and retry"
        elif isinstance(error, OrderTimeoutError):
            category = ErrorCategory.TIMEOUT
            severity = ErrorSeverity.HIGH
            recovery_suggestion = (
                "Check order status manually and adjust timeout settings"
            )
        elif isinstance(error, CircuitBreakerError):
            category = ErrorCategory.SYSTEM
            severity = ErrorSeverity.CRITICAL
            recovery_suggestion = (
                "Wait for circuit breaker to reset and investigate underlying issues"
            )
        elif isinstance(error, ExecutionEngineConfigError):
            category = ErrorCategory.CONFIGURATION
            severity = ErrorSeverity.CRITICAL
            recovery_suggestion = "Review and correct configuration settings"

        # Determine if error indicates critical system failure
        critical_error_types = [
            CircuitBreakerError,
            ExecutionEngineConfigError,
        ]
        if any(isinstance(error, t) for t in critical_error_types):
            severity = ErrorSeverity.CRITICAL

        error_details = ErrorDetails(
            error_id=error_id,
            category=category,
            severity=severity,
            message=str(error),
            timestamp=timestamp,
            order_id=context.get("order_id"),
            symbol=context.get("symbol"),
            error_type=type(error).__name__,
            stack_trace=traceback.format_exc(),
            recovery_suggestion=recovery_suggestion,
            metadata=context,
        )

        return error_details

    async def _handle_error(
        self, error: Exception, context: Dict[str, Any] = None
    ) -> ErrorDetails:
        """Comprehensively handle error with classification, logging, and notification.

        Args:
            error: Exception that occurred
            context: Additional context about the error

        Returns:
            ErrorDetails object with error information
        """
        # Classify error
        error_details = self._classify_error(error, context)

        # Update error tracking if enabled
        if self._config.enable_advanced_error_tracking:
            self._add_to_error_history(error_details)

        # Update circuit breaker
        if self._config.enable_circuit_breaker:
            self._circuit_breaker.record_failure()

        # Log error with appropriate level
        self._log_error(error_details)

        # Send error notification if needed
        if self._config.enable_error_notifications:
            await self._send_error_notification(error_details)

        # Publish error event
        await self._publish_error_event(error_details)

        return error_details

    def _add_to_error_history(self, error_details: ErrorDetails) -> None:
        """Add error to history and maintain size limit.

        Args:
            error_details: Error details to add to history
        """
        self._error_history.append(error_details)

        # Maintain history size limit
        if len(self._error_history) > self._config.max_error_history_size:
            self._error_history.pop(0)

        # Track consecutive critical errors
        if error_details.severity == ErrorSeverity.CRITICAL:
            self._consecutive_critical_errors += 1
        else:
            self._consecutive_critical_errors = 0

    def _log_error(self, error_details: ErrorDetails) -> None:
        """Log error with appropriate level and details.

        Args:
            error_details: Error details to log
        """
        base_message = (
            f"[{error_details.error_id}] {error_details.category.value.upper()}: "
            f"{error_details.message}"
        )

        context_info = []
        if error_details.order_id:
            context_info.append(f"order_id={error_details.order_id}")
        if error_details.symbol:
            context_info.append(f"symbol={error_details.symbol}")
        if error_details.retry_count > 0:
            context_info.append(f"retry_count={error_details.retry_count}")

        if context_info:
            base_message += f" ({', '.join(context_info)})"

        if error_details.recovery_suggestion:
            base_message += f" | Suggestion: {error_details.recovery_suggestion}"

        # Log at appropriate level based on severity
        if error_details.severity == ErrorSeverity.CRITICAL:
            self._logger.critical(base_message)
            if error_details.stack_trace:
                self._logger.critical(f"Stack trace: {error_details.stack_trace}")
        elif error_details.severity == ErrorSeverity.HIGH:
            self._logger.error(base_message)
        elif error_details.severity == ErrorSeverity.MEDIUM:
            self._logger.warning(base_message)
        else:
            self._logger.info(base_message)

    async def _send_error_notification(self, error_details: ErrorDetails) -> None:
        """Send error notification if conditions are met.

        Args:
            error_details: Error details for notification
        """
        current_time = int(time.time())

        # Check if we should send notification based on severity and cooldown
        should_notify = False

        if error_details.severity == ErrorSeverity.CRITICAL:
            should_notify = True
        elif (
            self._consecutive_critical_errors
            >= self._config.critical_error_notification_threshold
        ):
            should_notify = True
        elif (
            current_time - self._last_error_notification_time
            >= self._config.error_notification_cooldown_seconds
        ):
            should_notify = True

        if should_notify:
            self._last_error_notification_time = current_time

            # Create notification event
            notification_data = {
                "type": "error_notification",
                "error_details": error_details.to_dict(),
                "consecutive_critical_errors": self._consecutive_critical_errors,
                "circuit_breaker_status": self._circuit_breaker.get_statistics(),
                "execution_statistics": self.get_execution_statistics(),
                "timestamp": current_time,
            }

            # Publish notification event
            self._event_hub.publish(EventType.ERROR_OCCURRED, notification_data)

            self._logger.warning(
                f"Error notification sent for {error_details.error_id} "
                f"(severity={error_details.severity.value})"
            )

    async def _publish_error_event(self, error_details: ErrorDetails) -> None:
        """Publish error event to event hub.

        Args:
            error_details: Error details to publish
        """
        try:
            event_data = {
                "error_details": error_details.to_dict(),
                "circuit_breaker_status": self._circuit_breaker.get_statistics(),
                "error_statistics": self.get_error_statistics(),
                "timestamp": error_details.timestamp,
            }

            self._event_hub.publish(EventType.ERROR_OCCURRED, event_data)

        except Exception as e:
            self._logger.error(f"Failed to publish error event: {e}")

    async def _check_circuit_breaker(self, context: Dict[str, Any] = None) -> None:
        """Check circuit breaker state and raise error if open.

        Args:
            context: Context for circuit breaker check

        Raises:
            CircuitBreakerError: If circuit breaker is open
        """
        if not self._config.enable_circuit_breaker:
            return

        if not self._circuit_breaker.should_allow_request():
            error_msg = (
                f"Circuit breaker is OPEN "
                f"(failures={self._circuit_breaker.failure_count}, "
                f"threshold={self._circuit_breaker.failure_threshold})"
            )
            raise CircuitBreakerError(error_msg)

    async def _check_rate_limits(self) -> None:
        """Check and enforce rate limits to prevent API abuse.

        Raises:
            NetworkError: If rate limit would be exceeded
        """
        if not self._config.enable_rate_limit_protection:
            return

        async with self._rate_limit_lock:
            current_time = int(time.time())

            # Remove timestamps older than 1 minute
            cutoff_time = current_time - 60
            self._request_timestamps = [
                ts for ts in self._request_timestamps if ts > cutoff_time
            ]

            # Check if we're at the limit
            if len(self._request_timestamps) >= self._config.max_requests_per_minute:
                raise NetworkError(
                    f"Rate limit exceeded: {len(self._request_timestamps)}/"
                    f"{self._config.max_requests_per_minute} requests per minute"
                )

            # Add current request timestamp
            self._request_timestamps.append(current_time)

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics.

        Returns:
            Dictionary containing error statistics
        """
        if not self._error_history:
            return {
                "total_errors": 0,
                "error_categories": {},
                "error_severities": {},
                "recent_errors": [],
                "circuit_breaker": self._circuit_breaker.get_statistics(),
                "consecutive_critical_errors": self._consecutive_critical_errors,
            }

        # Count errors by category and severity
        category_counts = {}
        severity_counts = {}

        for error in self._error_history:
            category_counts[error.category.value] = (
                category_counts.get(error.category.value, 0) + 1
            )
            severity_counts[error.severity.value] = (
                severity_counts.get(error.severity.value, 0) + 1
            )

        # Get recent errors (last 10)
        recent_errors = [
            {
                "error_id": error.error_id,
                "category": error.category.value,
                "severity": error.severity.value,
                "message": error.message,
                "timestamp": error.timestamp,
                "order_id": error.order_id,
                "symbol": error.symbol,
            }
            for error in self._error_history[-10:]
        ]

        return {
            "total_errors": len(self._error_history),
            "error_categories": category_counts,
            "error_severities": severity_counts,
            "recent_errors": recent_errors,
            "circuit_breaker": self._circuit_breaker.get_statistics(),
            "consecutive_critical_errors": self._consecutive_critical_errors,
            "rate_limit_status": {
                "requests_in_last_minute": len(self._request_timestamps),
                "max_requests_per_minute": self._config.max_requests_per_minute,
                "rate_limit_protection_enabled": (
                    self._config.enable_rate_limit_protection
                ),
            },
        }

    async def recover_from_errors(self) -> Dict[str, Any]:
        """Attempt to recover from error states and reset resilience mechanisms.

        Returns:
            Dictionary with recovery results
        """
        recovery_results = {
            "circuit_breaker_reset": False,
            "error_history_cleared": False,
            "consecutive_errors_reset": False,
            "rate_limits_cleared": False,
            "monitoring_restarted": False,
        }

        try:
            # Reset circuit breaker if conditions are met
            if self._circuit_breaker.is_open:
                current_time = int(time.time())
                if (
                    self._circuit_breaker.last_failure_time
                    and current_time - self._circuit_breaker.last_failure_time
                    >= self._circuit_breaker.timeout_seconds
                ):
                    self._circuit_breaker.is_open = False
                    self._circuit_breaker.failure_count = 0
                    recovery_results["circuit_breaker_reset"] = True
                    self._logger.info("Circuit breaker reset successfully")

            # Clear error history if requested
            if len(self._error_history) > 0:
                self._error_history.clear()
                recovery_results["error_history_cleared"] = True
                self._logger.info("Error history cleared")

            # Reset consecutive critical errors
            if self._consecutive_critical_errors > 0:
                self._consecutive_critical_errors = 0
                recovery_results["consecutive_errors_reset"] = True
                self._logger.info("Consecutive critical error count reset")

            # Clear rate limit timestamps
            if self._request_timestamps:
                self._request_timestamps.clear()
                recovery_results["rate_limits_cleared"] = True
                self._logger.info("Rate limit timestamps cleared")

            # Restart monitoring if needed
            if not self._monitoring_running and self._active_orders:
                await self._start_order_monitoring()
                recovery_results["monitoring_restarted"] = True
                self._logger.info("Order monitoring restarted")

            self._logger.info("Error recovery completed successfully")
            return recovery_results

        except Exception as e:
            self._logger.error(f"Error recovery failed: {e}")
            recovery_results["recovery_error"] = str(e)
            return recovery_results


def create_execution_engine(
    event_hub: EventHub,
    binance_client: IExchangeClient,
    enable_execution: bool = True,
    max_order_value: float = 10000.0,
    max_retry_attempts: int = 3,
    **kwargs: Any,
) -> ExecutionEngine:
    """Factory function to create ExecutionEngine with default configuration.

    Args:
        event_hub: Event hub for event communication
        binance_client: Binance client for order execution
        enable_execution: Whether to enable actual order execution
        max_order_value: Maximum order value in USD
        max_retry_attempts: Maximum retry attempts for failed orders
        **kwargs: Additional configuration parameters

    Returns:
        ExecutionEngine: Configured execution engine instance

    Raises:
        ExecutionEngineError: If creation fails
    """
    try:
        # Create execution engine configuration
        execution_config = ExecutionEngineConfig(
            enable_order_execution=enable_execution,
            max_order_value_usd=max_order_value,
            max_retry_attempts=max_retry_attempts,
            enable_pre_execution_validation=kwargs.get("enable_validation", True),
            execution_timeout_seconds=kwargs.get("timeout_seconds", 30.0),
            max_daily_order_count=kwargs.get("max_daily_orders", 100),
            log_execution_details=kwargs.get("log_details", True),
            enable_background_monitoring=kwargs.get("enable_monitoring", True),
            monitoring_interval_seconds=kwargs.get("monitoring_interval", 5.0),
            max_monitoring_duration_seconds=kwargs.get(
                "max_monitoring_duration", 3600.0
            ),
        )

        # Create execution engine
        execution_engine = ExecutionEngine(
            config=execution_config,
            event_hub=event_hub,
            binance_client=binance_client,
        )

        return execution_engine

    except Exception as e:
        raise ExecutionEngineError(f"Failed to create execution engine: {e}")
