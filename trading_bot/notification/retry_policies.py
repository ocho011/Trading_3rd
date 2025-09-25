"""
Advanced retry policy implementations for Discord webhook transmission.

Provides configurable retry strategies including exponential backoff with jitter,
linear backoff, and fixed intervals. Follows SOLID principles with dependency injection
for flexible retry behavior configuration.
"""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Union


class BackoffType(Enum):
    """Enumeration of supported backoff types."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"


@dataclass(frozen=True)
class RetryConfig:
    """
    Configuration for retry policies.

    Attributes:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay in seconds for backoff calculations
        max_delay: Maximum delay in seconds between attempts
        backoff_type: Type of backoff strategy to use
        jitter_enabled: Whether to add random jitter to delays
        jitter_max: Maximum jitter percentage (0.0 to 1.0)
        timeout_multiplier: Multiplier for timeout on each retry
        max_timeout: Maximum timeout value in seconds
        retry_exceptions: Tuple of exception types to retry on
        stop_exceptions: Tuple of exception types to never retry
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_type: BackoffType = BackoffType.EXPONENTIAL
    jitter_enabled: bool = True
    jitter_max: float = 0.1
    timeout_multiplier: float = 1.5
    max_timeout: float = 30.0
    retry_exceptions: tuple = (Exception,)
    stop_exceptions: tuple = (KeyboardInterrupt, SystemExit)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay < 0:
            raise ValueError("base_delay must be non-negative")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if not 0.0 <= self.jitter_max <= 1.0:
            raise ValueError("jitter_max must be between 0.0 and 1.0")
        if self.timeout_multiplier < 1.0:
            raise ValueError("timeout_multiplier must be >= 1.0")
        if self.max_timeout < 0:
            raise ValueError("max_timeout must be non-negative")


@dataclass(frozen=True)
class RetryAttempt:
    """
    Information about a retry attempt.

    Attributes:
        attempt_number: Current attempt number (1-based)
        total_attempts: Total number of attempts allowed
        delay_before: Delay before this attempt in seconds
        elapsed_time: Total elapsed time since first attempt
        last_exception: Exception from previous attempt (None for first attempt)
    """

    attempt_number: int
    total_attempts: int
    delay_before: float
    elapsed_time: float
    last_exception: Optional[Exception] = None


class IRetryPolicy(ABC):
    """Interface for retry policy implementations."""

    @abstractmethod
    def calculate_delay(
        self, attempt: int, last_exception: Optional[Exception] = None
    ) -> float:
        """
        Calculate delay for a given retry attempt.

        Args:
            attempt: Attempt number (1-based)
            last_exception: Exception from previous attempt

        Returns:
            float: Delay in seconds before next attempt
        """
        pass

    @abstractmethod
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """
        Determine if retry should be attempted.

        Args:
            attempt: Current attempt number (1-based)
            exception: Exception that occurred

        Returns:
            bool: True if retry should be attempted
        """
        pass

    @abstractmethod
    def calculate_timeout(self, attempt: int) -> float:
        """
        Calculate timeout for a given attempt.

        Args:
            attempt: Attempt number (1-based)

        Returns:
            float: Timeout in seconds for this attempt
        """
        pass


class RetryPolicy(IRetryPolicy):
    """
    Configurable retry policy implementation with multiple backoff strategies.

    Supports exponential, linear, and fixed backoff with optional jitter.
    Provides comprehensive retry logic with configurable exception handling.
    """

    def __init__(self, config: RetryConfig) -> None:
        """
        Initialize retry policy with configuration.

        Args:
            config: Retry policy configuration
        """
        self._config = config
        self._logger = logging.getLogger(__name__)

    def calculate_delay(
        self, attempt: int, last_exception: Optional[Exception] = None
    ) -> float:
        """
        Calculate delay for retry attempt using configured backoff strategy.

        Args:
            attempt: Attempt number (1-based)
            last_exception: Exception from previous attempt

        Returns:
            float: Delay in seconds before next attempt
        """
        if self._config.backoff_type == BackoffType.EXPONENTIAL:
            delay = self._config.base_delay * (2 ** (attempt - 1))
        elif self._config.backoff_type == BackoffType.LINEAR:
            delay = self._config.base_delay * attempt
        else:  # FIXED
            delay = self._config.base_delay

        # Apply maximum delay constraint
        delay = min(delay, self._config.max_delay)

        # Add jitter if enabled
        if self._config.jitter_enabled and delay > 0:
            jitter_amount = delay * self._config.jitter_max
            jitter = random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.0, delay + jitter)

        return delay

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """
        Determine if retry should be attempted based on attempt count and exception type.

        Args:
            attempt: Current attempt number (1-based)
            exception: Exception that occurred

        Returns:
            bool: True if retry should be attempted
        """
        # Check if we've exceeded maximum attempts
        if attempt >= self._config.max_attempts:
            return False

        # Check for stop exceptions (never retry these)
        if isinstance(exception, self._config.stop_exceptions):
            return False

        # Check for retry exceptions (only retry these)
        return isinstance(exception, self._config.retry_exceptions)

    def calculate_timeout(self, attempt: int) -> float:
        """
        Calculate timeout for attempt using configured multiplier.

        Args:
            attempt: Attempt number (1-based)

        Returns:
            float: Timeout in seconds for this attempt
        """
        # Start with base timeout (assume base_delay as base timeout if not specified)
        base_timeout = self._config.base_delay * 10  # 10x base delay as base timeout
        timeout = base_timeout * (self._config.timeout_multiplier ** (attempt - 1))
        return min(timeout, self._config.max_timeout)


class ExponentialBackoffPolicy(RetryPolicy):
    """
    Specialized exponential backoff retry policy.

    Provides pre-configured exponential backoff with sensible defaults
    for Discord webhook retries.
    """

    def __init__(
        self,
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        jitter_enabled: bool = True,
    ) -> None:
        """
        Initialize exponential backoff policy.

        Args:
            max_attempts: Maximum retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            jitter_enabled: Whether to add random jitter
        """
        config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            backoff_type=BackoffType.EXPONENTIAL,
            jitter_enabled=jitter_enabled,
            jitter_max=0.1,  # 10% jitter
        )
        super().__init__(config)


class LinearBackoffPolicy(RetryPolicy):
    """
    Specialized linear backoff retry policy.

    Provides pre-configured linear backoff with sensible defaults
    for Discord webhook retries.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 2.0,
        max_delay: float = 20.0,
        jitter_enabled: bool = True,
    ) -> None:
        """
        Initialize linear backoff policy.

        Args:
            max_attempts: Maximum retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            jitter_enabled: Whether to add random jitter
        """
        config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            backoff_type=BackoffType.LINEAR,
            jitter_enabled=jitter_enabled,
            jitter_max=0.15,  # 15% jitter
        )
        super().__init__(config)


class FixedDelayPolicy(RetryPolicy):
    """
    Fixed delay retry policy.

    Provides fixed delay between retry attempts with optional jitter.
    """

    def __init__(
        self, max_attempts: int = 3, delay: float = 5.0, jitter_enabled: bool = False
    ) -> None:
        """
        Initialize fixed delay policy.

        Args:
            max_attempts: Maximum retry attempts
            delay: Fixed delay in seconds between attempts
            jitter_enabled: Whether to add random jitter
        """
        config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=delay,
            max_delay=delay,
            backoff_type=BackoffType.FIXED,
            jitter_enabled=jitter_enabled,
            jitter_max=0.05,  # 5% jitter if enabled
        )
        super().__init__(config)


class RetryExecutor:
    """
    Executes functions with retry logic using configurable policies.

    Provides both synchronous and asynchronous retry execution with
    comprehensive logging and monitoring capabilities.
    """

    def __init__(self, policy: IRetryPolicy) -> None:
        """
        Initialize retry executor with policy.

        Args:
            policy: Retry policy to use for execution
        """
        self._policy = policy
        self._logger = logging.getLogger(__name__)

    async def execute_async(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """
        Execute async function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Any: Function result

        Raises:
            Exception: Last exception if all retries fail
        """
        start_time = time.time()
        last_exception = None

        for attempt in range(1, self._policy._config.max_attempts + 1):
            try:
                elapsed = time.time() - start_time

                if attempt > 1:
                    delay = self._policy.calculate_delay(attempt - 1, last_exception)
                    if delay > 0:
                        self._logger.debug(
                            f"Retrying in {delay:.2f}s (attempt {attempt}/{self._policy._config.max_attempts})"
                        )
                        await asyncio.sleep(delay)

                # Calculate timeout for this attempt
                timeout = self._policy.calculate_timeout(attempt)

                # Execute function with timeout
                try:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs), timeout=timeout
                    )

                    if attempt > 1:
                        self._logger.info(f"Function succeeded on attempt {attempt}")

                    return result

                except asyncio.TimeoutError as e:
                    raise TimeoutError(f"Function timeout after {timeout}s") from e

            except Exception as e:
                last_exception = e
                elapsed = time.time() - start_time

                self._logger.warning(
                    f"Attempt {attempt} failed after {elapsed:.2f}s: {type(e).__name__}: {e}"
                )

                # Check if we should retry
                if not self._policy.should_retry(attempt, e):
                    self._logger.error(
                        f"Not retrying after attempt {attempt}: {type(e).__name__}"
                    )
                    raise e

                # If this was the last attempt, raise the exception
                if attempt >= self._policy._config.max_attempts:
                    self._logger.error(
                        f"All {attempt} attempts failed. Last error: {type(e).__name__}: {e}"
                    )
                    raise e

        # This should never be reached, but included for completeness
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Retry executor completed without success or exception")

    def execute_sync(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute function with retry logic synchronously.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Any: Function result

        Raises:
            Exception: Last exception if all retries fail
        """
        start_time = time.time()
        last_exception = None

        for attempt in range(1, self._policy._config.max_attempts + 1):
            try:
                elapsed = time.time() - start_time

                if attempt > 1:
                    delay = self._policy.calculate_delay(attempt - 1, last_exception)
                    if delay > 0:
                        self._logger.debug(
                            f"Retrying in {delay:.2f}s (attempt {attempt}/{self._policy._config.max_attempts})"
                        )
                        time.sleep(delay)

                # Execute function
                result = func(*args, **kwargs)

                if attempt > 1:
                    self._logger.info(f"Function succeeded on attempt {attempt}")

                return result

            except Exception as e:
                last_exception = e
                elapsed = time.time() - start_time

                self._logger.warning(
                    f"Attempt {attempt} failed after {elapsed:.2f}s: {type(e).__name__}: {e}"
                )

                # Check if we should retry
                if not self._policy.should_retry(attempt, e):
                    self._logger.error(
                        f"Not retrying after attempt {attempt}: {type(e).__name__}"
                    )
                    raise e

                # If this was the last attempt, raise the exception
                if attempt >= self._policy._config.max_attempts:
                    self._logger.error(
                        f"All {attempt} attempts failed. Last error: {type(e).__name__}: {e}"
                    )
                    raise e

        # This should never be reached, but included for completeness
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Retry executor completed without success or exception")


def create_discord_retry_policy(
    max_attempts: int = 5, strategy: str = "exponential"
) -> IRetryPolicy:
    """
    Factory function to create retry policy for Discord webhooks.

    Args:
        max_attempts: Maximum retry attempts
        strategy: Retry strategy ('exponential', 'linear', 'fixed')

    Returns:
        IRetryPolicy: Configured retry policy

    Raises:
        ValueError: If strategy is not supported
    """
    if strategy.lower() == "exponential":
        return ExponentialBackoffPolicy(max_attempts=max_attempts)
    elif strategy.lower() == "linear":
        return LinearBackoffPolicy(max_attempts=max_attempts)
    elif strategy.lower() == "fixed":
        return FixedDelayPolicy(max_attempts=max_attempts)
    else:
        raise ValueError(f"Unsupported retry strategy: {strategy}")
