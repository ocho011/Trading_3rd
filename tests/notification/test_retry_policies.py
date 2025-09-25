"""
Unit tests for retry policy implementations.

Tests retry strategies, backoff calculations, timeout handling,
and execution logic for Discord webhook retry policies.
"""

import asyncio
from unittest.mock import Mock, patch

import pytest

from trading_bot.notification.retry_policies import (
    BackoffType, ExponentialBackoffPolicy, FixedDelayPolicy,
    LinearBackoffPolicy, RetryConfig, RetryExecutor, RetryPolicy,
    create_discord_retry_policy)


class TestRetryConfig:
    """Test cases for RetryConfig validation."""

    def test_valid_config(self):
        """Test valid configuration creation."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=1.0,
            max_delay=60.0,
            backoff_type=BackoffType.EXPONENTIAL,
        )
        assert config.max_attempts == 5
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_type == BackoffType.EXPONENTIAL

    def test_invalid_max_attempts(self):
        """Test invalid max_attempts validation."""
        with pytest.raises(ValueError, match="max_attempts must be at least 1"):
            RetryConfig(max_attempts=0)

    def test_invalid_base_delay(self):
        """Test invalid base_delay validation."""
        with pytest.raises(ValueError, match="base_delay must be non-negative"):
            RetryConfig(base_delay=-1.0)

    def test_invalid_max_delay(self):
        """Test invalid max_delay validation."""
        with pytest.raises(ValueError, match="max_delay must be >= base_delay"):
            RetryConfig(base_delay=10.0, max_delay=5.0)

    def test_invalid_jitter_max(self):
        """Test invalid jitter_max validation."""
        with pytest.raises(ValueError, match="jitter_max must be between 0.0 and 1.0"):
            RetryConfig(jitter_max=1.5)


class TestRetryPolicy:
    """Test cases for RetryPolicy implementation."""

    def test_exponential_backoff_calculation(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            base_delay=1.0, backoff_type=BackoffType.EXPONENTIAL, jitter_enabled=False
        )
        policy = RetryPolicy(config)

        # Test exponential progression: 1, 2, 4, 8, 16...
        assert policy.calculate_delay(1) == 1.0
        assert policy.calculate_delay(2) == 2.0
        assert policy.calculate_delay(3) == 4.0
        assert policy.calculate_delay(4) == 8.0

    def test_linear_backoff_calculation(self):
        """Test linear backoff delay calculation."""
        config = RetryConfig(
            base_delay=2.0, backoff_type=BackoffType.LINEAR, jitter_enabled=False
        )
        policy = RetryPolicy(config)

        # Test linear progression: 2, 4, 6, 8...
        assert policy.calculate_delay(1) == 2.0
        assert policy.calculate_delay(2) == 4.0
        assert policy.calculate_delay(3) == 6.0
        assert policy.calculate_delay(4) == 8.0

    def test_fixed_backoff_calculation(self):
        """Test fixed backoff delay calculation."""
        config = RetryConfig(
            base_delay=5.0, backoff_type=BackoffType.FIXED, jitter_enabled=False
        )
        policy = RetryPolicy(config)

        # Test fixed delay
        assert policy.calculate_delay(1) == 5.0
        assert policy.calculate_delay(2) == 5.0
        assert policy.calculate_delay(3) == 5.0

    def test_max_delay_constraint(self):
        """Test max_delay constraint is applied."""
        config = RetryConfig(
            base_delay=10.0,
            max_delay=15.0,
            backoff_type=BackoffType.EXPONENTIAL,
            jitter_enabled=False,
        )
        policy = RetryPolicy(config)

        # Should be capped at max_delay
        assert policy.calculate_delay(1) == 10.0  # 10
        assert policy.calculate_delay(2) == 15.0  # min(20, 15)
        assert policy.calculate_delay(3) == 15.0  # min(40, 15)

    def test_jitter_application(self):
        """Test jitter is applied when enabled."""
        config = RetryConfig(base_delay=10.0, jitter_enabled=True, jitter_max=0.1)
        policy = RetryPolicy(config)

        # Calculate multiple delays to test jitter variance
        delays = [policy.calculate_delay(1) for _ in range(10)]

        # Should have some variance due to jitter
        assert len(set(delays)) > 1  # Not all delays should be identical

        # All delays should be within jitter range
        for delay in delays:
            assert 9.0 <= delay <= 11.0  # 10 ± 10%

    def test_should_retry_logic(self):
        """Test retry decision logic."""
        config = RetryConfig(
            max_attempts=3,
            retry_exceptions=(ValueError, ConnectionError),
            stop_exceptions=(KeyboardInterrupt,),
        )
        policy = RetryPolicy(config)

        # Should retry on retryable exceptions within attempt limit
        assert policy.should_retry(1, ValueError("test"))
        assert policy.should_retry(2, ConnectionError("test"))

        # Should not retry after max attempts
        assert not policy.should_retry(3, ValueError("test"))

        # Should not retry on stop exceptions
        assert not policy.should_retry(1, KeyboardInterrupt())

        # Should not retry on non-retryable exceptions
        assert not policy.should_retry(1, RuntimeError("test"))

    def test_timeout_calculation(self):
        """Test timeout calculation with multiplier."""
        config = RetryConfig(base_delay=1.0, timeout_multiplier=2.0, max_timeout=20.0)
        policy = RetryPolicy(config)

        # Base timeout = base_delay * 10 = 10s
        # Multiplied by 2^(attempt-1)
        assert policy.calculate_timeout(1) == 10.0  # 10 * 2^0
        assert policy.calculate_timeout(2) == 20.0  # min(10 * 2^1, 20)
        assert policy.calculate_timeout(3) == 20.0  # min(10 * 2^2, 20) - capped


class TestSpecializedPolicies:
    """Test cases for specialized retry policy classes."""

    def test_exponential_backoff_policy(self):
        """Test ExponentialBackoffPolicy defaults."""
        policy = ExponentialBackoffPolicy(max_attempts=3)

        # Should use exponential backoff
        delay1 = policy.calculate_delay(1)
        delay2 = policy.calculate_delay(2)
        assert delay2 > delay1  # Exponentially increasing

    def test_linear_backoff_policy(self):
        """Test LinearBackoffPolicy defaults."""
        policy = LinearBackoffPolicy(max_attempts=3)

        # Should use linear backoff
        with patch("random.uniform", return_value=0):  # Disable jitter for test
            delay1 = policy.calculate_delay(1)
            delay2 = policy.calculate_delay(2)
            delay3 = policy.calculate_delay(3)

            # Linear progression
            assert delay2 == delay1 * 2
            assert delay3 == delay1 * 3

    def test_fixed_delay_policy(self):
        """Test FixedDelayPolicy behavior."""
        policy = FixedDelayPolicy(max_attempts=3, delay=5.0)

        # All delays should be the same
        assert policy.calculate_delay(1) == 5.0
        assert policy.calculate_delay(2) == 5.0
        assert policy.calculate_delay(3) == 5.0


class TestRetryExecutor:
    """Test cases for RetryExecutor implementation."""

    @pytest.mark.asyncio
    async def test_async_success_no_retry(self):
        """Test async execution succeeds on first attempt."""
        policy = RetryPolicy(RetryConfig(max_attempts=3))
        executor = RetryExecutor(policy)

        mock_func = Mock(return_value="success")

        result = await executor.execute_async(mock_func, "arg1", key="value")

        assert result == "success"
        mock_func.assert_called_once_with("arg1", key="value")

    @pytest.mark.asyncio
    async def test_async_success_after_retries(self):
        """Test async execution succeeds after retries."""
        policy = RetryPolicy(
            RetryConfig(
                max_attempts=3,
                base_delay=0.01,  # Short delay for testing
                jitter_enabled=False,
            )
        )
        executor = RetryExecutor(policy)

        # Mock function that fails twice then succeeds
        mock_func = Mock(
            side_effect=[ValueError("fail"), ValueError("fail"), "success"]
        )

        result = await executor.execute_async(mock_func)

        assert result == "success"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_async_all_attempts_fail(self):
        """Test async execution fails after all attempts."""
        policy = RetryPolicy(
            RetryConfig(max_attempts=2, base_delay=0.01, jitter_enabled=False)
        )
        executor = RetryExecutor(policy)

        mock_func = Mock(side_effect=ValueError("always fails"))

        with pytest.raises(ValueError, match="always fails"):
            await executor.execute_async(mock_func)

        assert mock_func.call_count == 2

    def test_sync_success_no_retry(self):
        """Test sync execution succeeds on first attempt."""
        policy = RetryPolicy(RetryConfig(max_attempts=3))
        executor = RetryExecutor(policy)

        mock_func = Mock(return_value="success")

        result = executor.execute_sync(mock_func, "arg1", key="value")

        assert result == "success"
        mock_func.assert_called_once_with("arg1", key="value")

    def test_sync_success_after_retries(self):
        """Test sync execution succeeds after retries."""
        policy = RetryPolicy(
            RetryConfig(max_attempts=3, base_delay=0.01, jitter_enabled=False)
        )
        executor = RetryExecutor(policy)

        # Mock function that fails twice then succeeds
        mock_func = Mock(
            side_effect=[ValueError("fail"), ValueError("fail"), "success"]
        )

        result = executor.execute_sync(mock_func)

        assert result == "success"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_async_timeout_handling(self):
        """Test async timeout handling."""
        policy = RetryPolicy(RetryConfig(max_attempts=2))
        executor = RetryExecutor(policy)

        # Mock function that takes too long
        async def slow_func():
            await asyncio.sleep(10)  # Longer than timeout
            return "success"

        with patch.object(policy, "calculate_timeout", return_value=0.1):
            with pytest.raises(TimeoutError):
                await executor.execute_async(slow_func)

    def test_non_retryable_exception(self):
        """Test non-retryable exception is not retried."""
        policy = RetryPolicy(
            RetryConfig(
                max_attempts=3,
                retry_exceptions=(ValueError,),
                stop_exceptions=(RuntimeError,),
            )
        )
        executor = RetryExecutor(policy)

        mock_func = Mock(side_effect=RuntimeError("don't retry"))

        with pytest.raises(RuntimeError, match="don't retry"):
            executor.execute_sync(mock_func)

        # Should only be called once (no retries)
        assert mock_func.call_count == 1


class TestFactoryFunction:
    """Test cases for create_discord_retry_policy factory."""

    def test_exponential_strategy(self):
        """Test exponential strategy creation."""
        policy = create_discord_retry_policy(strategy="exponential")
        assert isinstance(policy, ExponentialBackoffPolicy)

    def test_linear_strategy(self):
        """Test linear strategy creation."""
        policy = create_discord_retry_policy(strategy="linear")
        assert isinstance(policy, LinearBackoffPolicy)

    def test_fixed_strategy(self):
        """Test fixed strategy creation."""
        policy = create_discord_retry_policy(strategy="fixed")
        assert isinstance(policy, FixedDelayPolicy)

    def test_invalid_strategy(self):
        """Test invalid strategy raises error."""
        with pytest.raises(ValueError, match="Unsupported retry strategy"):
            create_discord_retry_policy(strategy="invalid")
