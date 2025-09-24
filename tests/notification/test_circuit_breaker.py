"""
Unit tests for circuit breaker implementation.

Tests circuit breaker states, failure detection, recovery mechanisms,
and thread safety for Discord webhook circuit breaker.
"""

import asyncio
import pytest
import threading
import time
from unittest.mock import Mock

from trading_bot.notification.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    DiscordCircuitBreaker,
    create_circuit_breaker
)


class TestCircuitBreakerConfig:
    """Test cases for CircuitBreakerConfig validation."""

    def test_valid_config(self):
        """Test valid configuration creation."""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout=60.0
        )
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout == 60.0

    def test_invalid_failure_threshold(self):
        """Test invalid failure_threshold validation."""
        with pytest.raises(ValueError, match="failure_threshold must be at least 1"):
            CircuitBreakerConfig(failure_threshold=0)

    def test_invalid_success_threshold(self):
        """Test invalid success_threshold validation."""
        with pytest.raises(ValueError, match="success_threshold must be at least 1"):
            CircuitBreakerConfig(success_threshold=0)

    def test_invalid_timeout(self):
        """Test invalid timeout validation."""
        with pytest.raises(ValueError, match="timeout must be non-negative"):
            CircuitBreakerConfig(timeout=-1.0)

    def test_invalid_failure_rate_threshold(self):
        """Test invalid failure_rate_threshold validation."""
        with pytest.raises(ValueError, match="failure_rate_threshold must be between 0 and 100"):
            CircuitBreakerConfig(failure_rate_threshold=150.0)


class TestCircuitBreakerStates:
    """Test cases for circuit breaker state transitions."""

    def test_initial_state_is_closed(self):
        """Test circuit breaker starts in closed state."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(config)

        assert breaker.get_state() == CircuitState.CLOSED

    def test_transition_to_open_on_failures(self):
        """Test circuit opens after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=1.0)
        breaker = CircuitBreaker(config)

        # First failure - should stay closed
        breaker._record_failure()
        assert breaker.get_state() == CircuitState.CLOSED

        # Second failure - should open
        breaker._record_failure()
        assert breaker.get_state() == CircuitState.OPEN

    def test_transition_to_half_open_after_timeout(self):
        """Test circuit transitions to half-open after timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=0.1)
        breaker = CircuitBreaker(config)

        # Trigger failure to open circuit
        breaker._record_failure()
        assert breaker.get_state() == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)
        assert breaker.get_state() == CircuitState.HALF_OPEN

    def test_transition_to_closed_from_half_open(self):
        """Test circuit closes from half-open after successes."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            timeout=0.1
        )
        breaker = CircuitBreaker(config)

        # Open circuit
        breaker._record_failure()
        time.sleep(0.15)
        assert breaker.get_state() == CircuitState.HALF_OPEN

        # First success - should stay half-open
        breaker._record_success()
        assert breaker.get_state() == CircuitState.HALF_OPEN

        # Second success - should close
        breaker._record_success()
        assert breaker.get_state() == CircuitState.CLOSED

    def test_manual_reset(self):
        """Test manual circuit breaker reset."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(config)

        # Open circuit
        breaker._record_failure()
        assert breaker.get_state() == CircuitState.OPEN

        # Manual reset
        breaker.reset()
        assert breaker.get_state() == CircuitState.CLOSED


class TestCircuitBreakerExecution:
    """Test cases for circuit breaker execution protection."""

    @pytest.mark.asyncio
    async def test_async_call_success_closed_state(self):
        """Test async call succeeds when circuit is closed."""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker(config)

        mock_func = Mock(return_value="success")

        result = await breaker.call_async(mock_func, "arg1", key="value")

        assert result == "success"
        mock_func.assert_called_once_with("arg1", key="value")
        assert breaker.get_state() == CircuitState.CLOSED

    def test_sync_call_success_closed_state(self):
        """Test sync call succeeds when circuit is closed."""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker(config)

        mock_func = Mock(return_value="success")

        result = breaker.call_sync(mock_func, "arg1", key="value")

        assert result == "success"
        mock_func.assert_called_once_with("arg1", key="value")
        assert breaker.get_state() == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_async_call_blocked_open_state(self):
        """Test async call is blocked when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(config)

        # Open the circuit
        breaker._record_failure()
        assert breaker.get_state() == CircuitState.OPEN

        mock_func = Mock()

        with pytest.raises(CircuitBreakerError, match="Circuit breaker is OPEN"):
            await breaker.call_async(mock_func)

        # Function should not be called
        mock_func.assert_not_called()

    def test_sync_call_blocked_open_state(self):
        """Test sync call is blocked when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(config)

        # Open the circuit
        breaker._record_failure()
        assert breaker.get_state() == CircuitState.OPEN

        mock_func = Mock()

        with pytest.raises(CircuitBreakerError, match="Circuit breaker is OPEN"):
            breaker.call_sync(mock_func)

        # Function should not be called
        mock_func.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_call_failure_tracking(self):
        """Test async call failure tracking."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            expected_exception=(ValueError,)
        )
        breaker = CircuitBreaker(config)

        mock_func = Mock(side_effect=ValueError("test error"))

        # First failure
        with pytest.raises(ValueError):
            await breaker.call_async(mock_func)
        assert breaker.get_state() == CircuitState.CLOSED

        # Second failure should open circuit
        with pytest.raises(ValueError):
            await breaker.call_async(mock_func)
        assert breaker.get_state() == CircuitState.OPEN

    def test_sync_call_failure_tracking(self):
        """Test sync call failure tracking."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            expected_exception=(ValueError,)
        )
        breaker = CircuitBreaker(config)

        mock_func = Mock(side_effect=ValueError("test error"))

        # First failure
        with pytest.raises(ValueError):
            breaker.call_sync(mock_func)
        assert breaker.get_state() == CircuitState.CLOSED

        # Second failure should open circuit
        with pytest.raises(ValueError):
            breaker.call_sync(mock_func)
        assert breaker.get_state() == CircuitState.OPEN

    def test_non_expected_exception_not_tracked(self):
        """Test non-expected exceptions don't trigger circuit breaker."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            expected_exception=(ValueError,)
        )
        breaker = CircuitBreaker(config)

        mock_func = Mock(side_effect=RuntimeError("not tracked"))

        # Should raise exception but not open circuit
        with pytest.raises(RuntimeError):
            breaker.call_sync(mock_func)

        assert breaker.get_state() == CircuitState.CLOSED


class TestCircuitBreakerMetrics:
    """Test cases for circuit breaker metrics tracking."""

    def test_metrics_initialization(self):
        """Test initial metrics state."""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker(config)

        metrics = breaker.get_metrics()

        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.consecutive_failures == 0
        assert metrics.consecutive_successes == 0
        assert metrics.current_state == CircuitState.CLOSED

    def test_success_metrics_tracking(self):
        """Test success metrics tracking."""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker(config)

        # Record some successes
        breaker._record_success()
        breaker._record_success()

        metrics = breaker.get_metrics()

        assert metrics.total_requests == 2
        assert metrics.successful_requests == 2
        assert metrics.failed_requests == 0
        assert metrics.consecutive_successes == 2
        assert metrics.consecutive_failures == 0

    def test_failure_metrics_tracking(self):
        """Test failure metrics tracking."""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker(config)

        # Record some failures
        breaker._record_failure()
        breaker._record_failure()

        metrics = breaker.get_metrics()

        assert metrics.total_requests == 2
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 2
        assert metrics.consecutive_failures == 2
        assert metrics.consecutive_successes == 0

    def test_mixed_metrics_tracking(self):
        """Test mixed success/failure metrics tracking."""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker(config)

        # Record mixed results
        breaker._record_success()
        breaker._record_failure()
        breaker._record_failure()
        breaker._record_success()

        metrics = breaker.get_metrics()

        assert metrics.total_requests == 4
        assert metrics.successful_requests == 2
        assert metrics.failed_requests == 2
        # Consecutive counts are reset by alternating results
        assert metrics.consecutive_successes == 1
        assert metrics.consecutive_failures == 0


class TestFailureRateThreshold:
    """Test cases for failure rate based circuit opening."""

    def test_failure_rate_triggers_opening(self):
        """Test circuit opens based on failure rate."""
        config = CircuitBreakerConfig(
            failure_threshold=10,  # High threshold for count-based
            failure_rate_threshold=60.0,  # 60% failure rate
            minimum_requests=5
        )
        breaker = CircuitBreaker(config)

        # Record requests: 3 failures out of 5 requests = 60%
        breaker._record_success()
        breaker._record_failure()
        breaker._record_failure()
        breaker._record_failure()
        breaker._record_success()

        # Should open due to failure rate
        assert breaker.get_state() == CircuitState.OPEN

    def test_failure_rate_below_minimum_requests(self):
        """Test failure rate not applied below minimum requests."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            failure_rate_threshold=50.0,
            minimum_requests=5
        )
        breaker = CircuitBreaker(config)

        # Record only 3 requests (below minimum)
        breaker._record_failure()
        breaker._record_failure()
        breaker._record_success()  # 66% failure rate

        # Should not open due to insufficient requests
        assert breaker.get_state() == CircuitState.CLOSED


class TestThreadSafety:
    """Test cases for thread safety."""

    def test_concurrent_access(self):
        """Test circuit breaker is thread-safe."""
        config = CircuitBreakerConfig(failure_threshold=10)
        breaker = CircuitBreaker(config)

        results = []
        exceptions = []

        def worker():
            try:
                for _ in range(100):
                    breaker._record_success()
                    breaker._record_failure()
                    state = breaker.get_state()
                    metrics = breaker.get_metrics()
                    results.append((state, metrics.total_requests))
            except Exception as e:
                exceptions.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=worker) for _ in range(5)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should not have any exceptions
        assert len(exceptions) == 0

        # Should have recorded all operations
        final_metrics = breaker.get_metrics()
        assert final_metrics.total_requests == 1000  # 5 threads * 100 * 2 operations


class TestDiscordCircuitBreaker:
    """Test cases for Discord-specific circuit breaker."""

    def test_discord_specific_configuration(self):
        """Test Discord circuit breaker has appropriate defaults."""
        breaker = DiscordCircuitBreaker()

        config = breaker._config
        assert config.failure_threshold <= 5  # Should be conservative
        assert config.timeout >= 60.0  # Should allow time for recovery

        # Should handle Discord-specific exceptions
        from trading_bot.notification.discord_notifier import DiscordNotificationError
        expected_exceptions = config.expected_exception
        assert DiscordNotificationError in expected_exceptions
        assert ConnectionError in expected_exceptions
        assert TimeoutError in expected_exceptions


class TestFactoryFunction:
    """Test cases for create_circuit_breaker factory."""

    def test_discord_circuit_breaker(self):
        """Test Discord circuit breaker creation."""
        breaker = create_circuit_breaker("discord")
        assert isinstance(breaker, DiscordCircuitBreaker)

    def test_default_circuit_breaker(self):
        """Test default circuit breaker creation."""
        breaker = create_circuit_breaker("default")
        assert isinstance(breaker, CircuitBreaker)
        assert not isinstance(breaker, DiscordCircuitBreaker)

    def test_custom_parameters(self):
        """Test circuit breaker with custom parameters."""
        breaker = create_circuit_breaker(
            "default",
            failure_threshold=10,
            timeout=120.0
        )

        config = breaker._config
        assert config.failure_threshold == 10
        assert config.timeout == 120.0