"""
Circuit breaker pattern implementation for Discord webhook reliability.

Provides circuit breaker functionality to prevent cascading failures when Discord
webhooks become unreliable or unavailable. Follows SOLID principles with
configurable failure thresholds and recovery mechanisms.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, requests are blocked
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """
    Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        success_threshold: Number of successes to close circuit from half-open
        timeout: Time in seconds before attempting half-open
        expected_exception: Exception type that counts as failure
        monitor_window: Window in seconds for failure rate calculation
        minimum_requests: Minimum requests in window before applying failure rate
        failure_rate_threshold: Failure rate percentage (0-100) to open circuit
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0
    expected_exception: tuple = (Exception,)
    monitor_window: float = 300.0  # 5 minutes
    minimum_requests: int = 10
    failure_rate_threshold: float = 50.0  # 50%

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if self.success_threshold < 1:
            raise ValueError("success_threshold must be at least 1")
        if self.timeout < 0:
            raise ValueError("timeout must be non-negative")
        if self.monitor_window <= 0:
            raise ValueError("monitor_window must be positive")
        if not 0 <= self.failure_rate_threshold <= 100:
            raise ValueError("failure_rate_threshold must be between 0 and 100")


@dataclass
class CircuitMetrics:
    """
    Metrics for circuit breaker monitoring.

    Attributes:
        total_requests: Total requests processed
        successful_requests: Successful requests
        failed_requests: Failed requests
        state_changes: Number of state changes
        last_failure_time: Timestamp of last failure
        last_success_time: Timestamp of last success
        current_state: Current circuit state
        time_in_open: Total time spent in open state
        consecutive_failures: Current consecutive failures
        consecutive_successes: Current consecutive successes
    """

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    current_state: CircuitState = CircuitState.CLOSED
    time_in_open: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""

    def __init__(self, message: str, state: CircuitState) -> None:
        """
        Initialize circuit breaker error.

        Args:
            message: Error message
            state: Current circuit state
        """
        super().__init__(message)
        self.state = state


class ICircuitBreaker(ABC):
    """Interface for circuit breaker implementations."""

    @abstractmethod
    async def call_async(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """
        Execute async function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Any: Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """

    @abstractmethod
    def call_sync(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Any: Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """

    @abstractmethod
    def get_state(self) -> CircuitState:
        """Get current circuit state."""

    @abstractmethod
    def get_metrics(self) -> CircuitMetrics:
        """Get circuit breaker metrics."""

    @abstractmethod
    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""


class CircuitBreaker(ICircuitBreaker):
    """
    Thread-safe circuit breaker implementation.

    Provides circuit breaker pattern with configurable failure thresholds,
    timeout periods, and recovery mechanisms. Supports both time-based and
    count-based failure detection.
    """

    def __init__(self, config: CircuitBreakerConfig) -> None:
        """
        Initialize circuit breaker with configuration.

        Args:
            config: Circuit breaker configuration
        """
        self._config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._next_attempt_time: Optional[float] = None
        self._metrics = CircuitMetrics()
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)

        # For failure rate monitoring
        self._recent_requests: list[tuple[float, bool]] = []  # (timestamp, success)

    async def call_async(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """
        Execute async function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Any: Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        with self._lock:
            self._update_state()
            current_state = self._state

            if current_state == CircuitState.OPEN:
                self._metrics.total_requests += 1
                raise CircuitBreakerError(
                    "Circuit breaker is OPEN - requests are blocked", CircuitState.OPEN
                )

        # Execute the function
        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result

        except Exception as e:
            if isinstance(e, self._config.expected_exception):
                self._record_failure()
            raise

    def call_sync(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Any: Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        with self._lock:
            self._update_state()
            current_state = self._state

            if current_state == CircuitState.OPEN:
                self._metrics.total_requests += 1
                raise CircuitBreakerError(
                    "Circuit breaker is OPEN - requests are blocked", CircuitState.OPEN
                )

        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result

        except Exception as e:
            if isinstance(e, self._config.expected_exception):
                self._record_failure()
            raise

    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._update_state()
            return self._state

    def get_metrics(self) -> CircuitMetrics:
        """Get circuit breaker metrics."""
        with self._lock:
            self._update_state()
            metrics = CircuitMetrics(
                total_requests=self._metrics.total_requests,
                successful_requests=self._metrics.successful_requests,
                failed_requests=self._metrics.failed_requests,
                state_changes=self._metrics.state_changes,
                last_failure_time=self._metrics.last_failure_time,
                last_success_time=self._metrics.last_success_time,
                current_state=self._state,
                time_in_open=self._metrics.time_in_open,
                consecutive_failures=self._failure_count,
                consecutive_successes=self._success_count,
            )
            return metrics

    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        with self._lock:
            self._logger.info("Circuit breaker manually reset to CLOSED state")
            self._transition_to_closed()

    def _update_state(self) -> None:
        """Update circuit state based on current conditions."""
        current_time = time.time()

        if self._state == CircuitState.OPEN:
            if self._next_attempt_time and current_time >= self._next_attempt_time:
                self._transition_to_half_open()
            else:
                # Update time in open state
                if self._last_failure_time:
                    self._metrics.time_in_open = current_time - self._last_failure_time

        # Clean up old requests for failure rate calculation
        cutoff_time = current_time - self._config.monitor_window
        self._recent_requests = [
            (timestamp, success)
            for timestamp, success in self._recent_requests
            if timestamp >= cutoff_time
        ]

    def _record_success(self) -> None:
        """Record a successful operation."""
        current_time = time.time()

        with self._lock:
            self._metrics.total_requests += 1
            self._metrics.successful_requests += 1
            self._metrics.last_success_time = current_time

            # Reset failure count on success
            self._failure_count = 0
            self._success_count += 1

            # Record for failure rate calculation
            self._recent_requests.append((current_time, True))

            # Handle state transitions
            if self._state == CircuitState.HALF_OPEN:
                if self._success_count >= self._config.success_threshold:
                    self._transition_to_closed()
            elif self._state == CircuitState.OPEN:
                # This shouldn't happen, but handle gracefully
                self._logger.warning("Success recorded while circuit is OPEN")

            self._logger.debug(
                f"Circuit breaker recorded success (state: {self._state.value})"
            )

    def _record_failure(self) -> None:
        """Record a failed operation."""
        current_time = time.time()

        with self._lock:
            self._metrics.total_requests += 1
            self._metrics.failed_requests += 1
            self._metrics.last_failure_time = current_time
            self._last_failure_time = current_time

            # Reset success count on failure
            self._success_count = 0
            self._failure_count += 1

            # Record for failure rate calculation
            self._recent_requests.append((current_time, False))

            # Check if we should open the circuit
            should_open = False

            # Check consecutive failures threshold
            if self._failure_count >= self._config.failure_threshold:
                should_open = True
                self._logger.warning(
                    f"Circuit breaker failure threshold reached: "
                    f"{self._failure_count} failures"
                )

            # Check failure rate threshold
            if len(self._recent_requests) >= self._config.minimum_requests:
                failed_requests = sum(
                    1 for _, success in self._recent_requests if not success
                )
                failure_rate = (failed_requests / len(self._recent_requests)) * 100

                if failure_rate >= self._config.failure_rate_threshold:
                    should_open = True
                    self._logger.warning(
                        f"Circuit breaker failure rate threshold reached: "
                        f"{failure_rate:.1f}%"
                    )

            # Open circuit if threshold exceeded
            if should_open and self._state != CircuitState.OPEN:
                self._transition_to_open()

            self._logger.debug(
                f"Circuit breaker recorded failure (state: {self._state.value})"
            )

    def _transition_to_closed(self) -> None:
        """Transition circuit to CLOSED state."""
        if self._state != CircuitState.CLOSED:
            self._logger.info("Circuit breaker transitioning to CLOSED state")
            self._state = CircuitState.CLOSED
            self._metrics.state_changes += 1
            self._metrics.current_state = CircuitState.CLOSED

        self._failure_count = 0
        self._success_count = 0
        self._next_attempt_time = None

    def _transition_to_open(self) -> None:
        """Transition circuit to OPEN state."""
        self._logger.warning("Circuit breaker transitioning to OPEN state")
        self._state = CircuitState.OPEN
        self._metrics.state_changes += 1
        self._metrics.current_state = CircuitState.OPEN

        current_time = time.time()
        self._next_attempt_time = current_time + self._config.timeout

        self._logger.info(
            f"Circuit breaker will attempt recovery at {self._next_attempt_time} "
            f"(in {self._config.timeout}s)"
        )

    def _transition_to_half_open(self) -> None:
        """Transition circuit to HALF_OPEN state."""
        self._logger.info("Circuit breaker transitioning to HALF_OPEN state")
        self._state = CircuitState.HALF_OPEN
        self._metrics.state_changes += 1
        self._metrics.current_state = CircuitState.HALF_OPEN

        self._success_count = 0
        self._failure_count = 0


class DiscordCircuitBreaker(CircuitBreaker):
    """
    Specialized circuit breaker for Discord webhook operations.

    Pre-configured with Discord-specific failure patterns and recovery settings.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        timeout: float = 120.0,
        success_threshold: int = 2,
    ) -> None:
        """
        Initialize Discord circuit breaker.

        Args:
            failure_threshold: Consecutive failures before opening
            timeout: Seconds to wait before trying half-open
            success_threshold: Successes needed to close from half-open
        """
        from trading_bot.notification.discord_notifier import DiscordNotificationError

        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
            expected_exception=(
                DiscordNotificationError,
                ConnectionError,
                TimeoutError,
            ),
            monitor_window=300.0,  # 5 minutes
            minimum_requests=5,
            failure_rate_threshold=60.0,  # 60% failure rate
        )
        super().__init__(config)


def create_circuit_breaker(
    name: str = "default", failure_threshold: int = 5, timeout: float = 60.0
) -> ICircuitBreaker:
    """
    Factory function to create circuit breaker.

    Args:
        name: Circuit breaker name for logging
        failure_threshold: Failures before opening circuit
        timeout: Timeout before attempting recovery

    Returns:
        ICircuitBreaker: Configured circuit breaker
    """
    if name.lower() == "discord":
        return DiscordCircuitBreaker(
            failure_threshold=failure_threshold, timeout=timeout
        )
    else:
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold, timeout=timeout
        )
        return CircuitBreaker(config)
