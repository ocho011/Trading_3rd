"""
Unit tests for enhanced Discord HTTP client.

Tests intelligent retry mechanisms, rate limiting, circuit breaker integration,
health monitoring, and fallback strategies for Discord webhook client.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from trading_bot.notification.enhanced_discord_client import (
    EnhancedDiscordHttpClient,
    DiscordRateLimitInfo,
    create_enhanced_discord_client
)
from trading_bot.notification.retry_policies import RetryConfig, RetryPolicy
from trading_bot.notification.circuit_breaker import CircuitBreakerConfig, CircuitBreaker
from trading_bot.notification.webhook_health import HealthThresholds, WebhookHealthMonitor
from trading_bot.notification.message_queue import QueueConfig, MessageQueue
from trading_bot.notification.webhook_config import WebhookReliabilityConfig
from trading_bot.notification.discord_notifier import DiscordNotificationError


class TestDiscordRateLimitInfo:
    """Test cases for Discord rate limit information parsing."""

    def test_rate_limit_info_creation(self):
        """Test creating rate limit info from headers."""
        headers = {
            'x-ratelimit-limit': '5',
            'x-ratelimit-remaining': '3',
            'x-ratelimit-reset': '1640995200',  # Unix timestamp
            'x-ratelimit-reset-after': '10.5',
            'retry-after': '15',
            'x-ratelimit-scope': 'webhook'
        }

        rate_limit = DiscordRateLimitInfo.from_headers(headers)

        assert rate_limit.limit == 5
        assert rate_limit.remaining == 3
        assert rate_limit.reset_timestamp == 1640995200
        assert rate_limit.reset_after == 10.5
        assert rate_limit.retry_after == 15
        assert rate_limit.scope == 'webhook'

    def test_rate_limit_info_with_missing_headers(self):
        """Test creating rate limit info with missing headers."""
        headers = {
            'x-ratelimit-remaining': '2'
        }

        rate_limit = DiscordRateLimitInfo.from_headers(headers)

        assert rate_limit.limit is None
        assert rate_limit.remaining == 2
        assert rate_limit.reset_timestamp is None
        assert rate_limit.retry_after is None

    def test_is_rate_limited(self):
        """Test rate limit detection logic."""
        # Not rate limited
        rate_limit = DiscordRateLimitInfo(remaining=3)
        assert not rate_limit.is_rate_limited()

        # Rate limited (remaining = 0)
        rate_limit = DiscordRateLimitInfo(remaining=0)
        assert rate_limit.is_rate_limited()

        # Rate limited (has retry_after)
        rate_limit = DiscordRateLimitInfo(remaining=5, retry_after=10)
        assert rate_limit.is_rate_limited()

    def test_wait_time_calculation(self):
        """Test rate limit wait time calculation."""
        # No wait needed
        rate_limit = DiscordRateLimitInfo(remaining=3)
        assert rate_limit.get_wait_time() == 0

        # Wait based on retry_after
        rate_limit = DiscordRateLimitInfo(retry_after=15)
        assert rate_limit.get_wait_time() == 15

        # Wait based on reset_after if no retry_after
        rate_limit = DiscordRateLimitInfo(reset_after=10.5)
        assert rate_limit.get_wait_time() == 10.5

        # Default wait if no timing info
        rate_limit = DiscordRateLimitInfo(remaining=0)
        assert rate_limit.get_wait_time() == 60  # Default


class TestEnhancedDiscordHttpClient:
    """Test cases for enhanced Discord HTTP client functionality."""

    def create_test_client(self, **config_overrides):
        """Create test client with default configuration."""
        config = WebhookReliabilityConfig(
            webhook_url="https://discord.com/api/webhooks/test/token",
            timeout=10,
            **config_overrides
        )
        return EnhancedDiscordHttpClient(config)

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client initialization with components."""
        client = self.create_test_client()

        assert client._config.webhook_url == "https://discord.com/api/webhooks/test/token"
        assert client._retry_policy is not None
        assert client._circuit_breaker is not None
        assert client._health_monitor is not None
        assert client._message_queue is not None

    @pytest.mark.asyncio
    async def test_successful_webhook_request(self):
        """Test successful webhook request execution."""
        client = self.create_test_client()

        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.headers = {'x-ratelimit-remaining': '5'}
        mock_response.text = AsyncMock(return_value='{"success": true}')

        with patch('aiohttp.ClientSession.post', return_value=mock_response) as mock_post:
            result = await client.send_webhook({"content": "Test message"})

            assert result is True
            mock_post.assert_called_once()

            # Verify health monitoring was updated
            metrics = client._health_monitor.get_current_metrics()
            assert metrics.total_requests == 1
            assert metrics.successful_requests == 1

    @pytest.mark.asyncio
    async def test_webhook_request_with_rate_limit(self):
        """Test webhook request handling Discord rate limits."""
        client = self.create_test_client(max_rate_limit_wait=5.0)

        # Mock rate limited response then success
        rate_limited_response = Mock()
        rate_limited_response.status = 429
        rate_limited_response.headers = {
            'x-ratelimit-remaining': '0',
            'retry-after': '2'
        }

        success_response = Mock()
        success_response.status = 200
        success_response.headers = {'x-ratelimit-remaining': '5'}
        success_response.text = AsyncMock(return_value='{"success": true}')

        with patch('aiohttp.ClientSession.post', side_effect=[rate_limited_response, success_response]) as mock_post:
            with patch('asyncio.sleep') as mock_sleep:
                result = await client.send_webhook({"content": "Test message"})

                assert result is True
                assert mock_post.call_count == 2
                mock_sleep.assert_called_once_with(2)  # Should wait for rate limit

    @pytest.mark.asyncio
    async def test_webhook_request_rate_limit_too_long(self):
        """Test webhook request when rate limit wait exceeds maximum."""
        client = self.create_test_client(max_rate_limit_wait=5.0)

        # Mock rate limited response with long wait
        rate_limited_response = Mock()
        rate_limited_response.status = 429
        rate_limited_response.headers = {
            'x-ratelimit-remaining': '0',
            'retry-after': '10'  # Exceeds max_rate_limit_wait
        }

        with patch('aiohttp.ClientSession.post', return_value=rate_limited_response):
            with pytest.raises(DiscordNotificationError, match="Rate limit wait time"):
                await client.send_webhook({"content": "Test message"})

    @pytest.mark.asyncio
    async def test_webhook_request_with_retry(self):
        """Test webhook request with retry on failure."""
        client = self.create_test_client()

        # Mock failed response then success
        failed_response = Mock()
        failed_response.status = 500
        failed_response.text = AsyncMock(return_value='Internal Server Error')

        success_response = Mock()
        success_response.status = 200
        success_response.headers = {'x-ratelimit-remaining': '5'}
        success_response.text = AsyncMock(return_value='{"success": true}')

        with patch('aiohttp.ClientSession.post', side_effect=[failed_response, success_response]) as mock_post:
            with patch('asyncio.sleep'):  # Mock retry delay
                result = await client.send_webhook({"content": "Test message"})

                assert result is True
                assert mock_post.call_count == 2  # Initial attempt + 1 retry

    @pytest.mark.asyncio
    async def test_webhook_request_circuit_breaker_open(self):
        """Test webhook request when circuit breaker is open."""
        # Create client with low failure threshold
        config = WebhookReliabilityConfig(
            webhook_url="https://discord.com/api/webhooks/test/token"
        )
        config.circuit_breaker_config.failure_threshold = 1

        client = EnhancedDiscordHttpClient(config)

        # Trigger circuit breaker to open
        client._circuit_breaker._record_failure()

        # Should raise CircuitBreakerError without making HTTP request
        with patch('aiohttp.ClientSession.post') as mock_post:
            with pytest.raises(DiscordNotificationError, match="Circuit breaker"):
                await client.send_webhook({"content": "Test message"})

            mock_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_webhook_request_with_queuing(self):
        """Test webhook request falls back to queue on failure."""
        client = self.create_test_client()

        # Mock all retries failing
        failed_response = Mock()
        failed_response.status = 500
        failed_response.text = AsyncMock(return_value='Server Error')

        with patch('aiohttp.ClientSession.post', return_value=failed_response):
            with patch('asyncio.sleep'):  # Mock retry delays
                # Should eventually queue the message
                result = await client.send_webhook({"content": "Test message"})

                # Should return False (not successfully sent) but queued
                assert result is False

                # Message should be in queue
                assert client._message_queue.size() > 0

    @pytest.mark.asyncio
    async def test_queue_processing(self):
        """Test processing queued messages."""
        client = self.create_test_client()

        # Add message to queue
        from trading_bot.notification.message_queue import QueuedMessage, MessagePriority
        message = QueuedMessage(
            content='{"content": "Queued message"}',
            webhook_url="https://discord.com/api/webhooks/test/token",
            priority=MessagePriority.NORMAL
        )
        await client._message_queue.enqueue(message)

        # Mock successful response for queue processing
        success_response = Mock()
        success_response.status = 200
        success_response.headers = {'x-ratelimit-remaining': '5'}
        success_response.text = AsyncMock(return_value='{"success": true}')

        with patch('aiohttp.ClientSession.post', return_value=success_response):
            # Start queue processing
            await client.start_queue_processing()

            # Wait a bit for processing
            await asyncio.sleep(0.1)

            # Stop processing
            await client.stop_queue_processing()

            # Queue should be empty
            assert client._message_queue.size() == 0

    @pytest.mark.asyncio
    async def test_health_monitoring_integration(self):
        """Test health monitoring integration."""
        client = self.create_test_client()

        # Mock successful response
        success_response = Mock()
        success_response.status = 200
        success_response.headers = {'x-ratelimit-remaining': '5'}
        success_response.text = AsyncMock(return_value='{"success": true}')

        # Mock failed response
        failed_response = Mock()
        failed_response.status = 500
        failed_response.text = AsyncMock(return_value='Server Error')

        with patch('aiohttp.ClientSession.post', side_effect=[success_response, failed_response]):
            # Send successful request
            await client.send_webhook({"content": "Success"})

            # Send failed request (should queue after retries)
            with patch('asyncio.sleep'):
                await client.send_webhook({"content": "Failure"})

            # Check health metrics
            metrics = client._health_monitor.get_current_metrics()
            assert metrics.total_requests >= 1  # At least one successful
            assert metrics.successful_requests >= 1
            # Failed request attempts would be recorded during retries

    @pytest.mark.asyncio
    async def test_backup_notification(self):
        """Test backup notification when all else fails."""
        client = self.create_test_client(backup_notification_enabled=True)

        # Mock backup notifier
        backup_notifier = Mock()
        client._backup_notifier = backup_notifier

        # Mock circuit breaker as open
        client._circuit_breaker._record_failure()

        # Should trigger backup notification
        result = await client.send_webhook({"content": "Test message"})

        # Should still return False (not successfully sent via webhook)
        assert result is False

    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test handling of connection errors."""
        client = self.create_test_client()

        # Mock connection error
        with patch('aiohttp.ClientSession.post', side_effect=ConnectionError("Connection failed")):
            with patch('asyncio.sleep'):  # Mock retry delays
                result = await client.send_webhook({"content": "Test message"})

                # Should fail and queue message
                assert result is False
                assert client._message_queue.size() > 0

                # Health monitor should record failure
                metrics = client._health_monitor.get_current_metrics()
                assert metrics.failed_requests > 0

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self):
        """Test handling of timeout errors."""
        client = self.create_test_client()

        # Mock timeout error
        with patch('aiohttp.ClientSession.post', side_effect=asyncio.TimeoutError("Request timed out")):
            with patch('asyncio.sleep'):  # Mock retry delays
                result = await client.send_webhook({"content": "Test message"})

                # Should fail and queue message
                assert result is False

                # Health monitor should record failure
                metrics = client._health_monitor.get_current_metrics()
                assert metrics.failed_requests > 0

    def test_get_health_status(self):
        """Test getting client health status."""
        client = self.create_test_client()

        # Initially should be healthy
        assert client.get_health_status() == "healthy"

        # Add some successful requests
        client._health_monitor._current_metrics.record_success(response_time_ms=100)
        assert client.get_health_status() == "healthy"

        # Add failures to degrade health
        for _ in range(5):
            client._health_monitor._current_metrics.record_failure("Test failure")

        # Should now show degraded health
        health_status = client.get_health_status()
        assert health_status in ["warning", "critical"]

    def test_get_health_metrics(self):
        """Test getting detailed health metrics."""
        client = self.create_test_client()

        # Add some test data
        client._health_monitor._current_metrics.record_success(response_time_ms=150)
        client._health_monitor._current_metrics.record_failure("Rate limited")

        # Get metrics
        metrics = client.get_health_metrics()

        assert metrics.total_requests == 2
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 1
        assert metrics.success_rate == 50.0
        assert "Rate limited" in metrics.failure_reasons

    @pytest.mark.asyncio
    async def test_client_lifecycle(self):
        """Test client startup and shutdown."""
        client = self.create_test_client()

        # Start client (should start queue processing)
        await client.start()

        # Verify components are started
        assert client._queue_processing_task is not None

        # Stop client
        await client.stop()

        # Verify cleanup
        assert client._queue_processing_task is None or client._queue_processing_task.done()

    @pytest.mark.asyncio
    async def test_webhook_url_validation(self):
        """Test webhook URL validation."""
        # Invalid URL should raise error during client creation
        with pytest.raises(ValueError, match="Invalid webhook URL"):
            config = WebhookReliabilityConfig(webhook_url="not-a-valid-url")
            EnhancedDiscordHttpClient(config)

        # Valid Discord webhook URL should work
        config = WebhookReliabilityConfig(
            webhook_url="https://discord.com/api/webhooks/123/token"
        )
        client = EnhancedDiscordHttpClient(config)
        assert client._config.webhook_url.startswith("https://discord.com/api/webhooks/")

    @pytest.mark.asyncio
    async def test_payload_validation(self):
        """Test webhook payload validation."""
        client = self.create_test_client()

        # Valid payload should work
        valid_payload = {"content": "Test message"}
        # This would normally make HTTP request, we just test it doesn't raise
        with patch('aiohttp.ClientSession.post', return_value=Mock(status=200, headers={}, text=AsyncMock(return_value='{}'))):
            result = await client.send_webhook(valid_payload)
            assert result is True

        # Empty payload should raise error
        with pytest.raises(ValueError, match="Payload cannot be empty"):
            await client.send_webhook({})

        # None payload should raise error
        with pytest.raises(ValueError, match="Payload cannot be empty"):
            await client.send_webhook(None)


class TestEnhancedClientIntegration:
    """Integration tests for enhanced Discord client."""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end webhook workflow."""
        # Create client with realistic configuration
        config = WebhookReliabilityConfig(
            webhook_url="https://discord.com/api/webhooks/test/token",
            timeout=5,
            max_rate_limit_wait=10.0
        )
        config.retry_config.max_attempts = 3
        config.circuit_breaker_config.failure_threshold = 5

        client = EnhancedDiscordHttpClient(config)

        # Mock various response scenarios
        responses = [
            # First request - success
            Mock(status=200, headers={'x-ratelimit-remaining': '5'}, text=AsyncMock(return_value='{}')),
            # Second request - rate limited, then success
            Mock(status=429, headers={'retry-after': '1', 'x-ratelimit-remaining': '0'}),
            Mock(status=200, headers={'x-ratelimit-remaining': '4'}, text=AsyncMock(return_value='{}')),
            # Third request - server error, then success on retry
            Mock(status=500, text=AsyncMock(return_value='Server Error')),
            Mock(status=200, headers={'x-ratelimit-remaining': '3'}, text=AsyncMock(return_value='{}'))
        ]

        with patch('aiohttp.ClientSession.post', side_effect=responses):
            with patch('asyncio.sleep'):  # Mock delays
                # Start client
                await client.start()

                try:
                    # Send multiple webhook requests
                    result1 = await client.send_webhook({"content": "Message 1"})
                    result2 = await client.send_webhook({"content": "Message 2"})
                    result3 = await client.send_webhook({"content": "Message 3"})

                    # All should succeed eventually
                    assert result1 is True
                    assert result2 is True
                    assert result3 is True

                    # Check health metrics
                    metrics = client.get_health_metrics()
                    assert metrics.successful_requests >= 3
                    assert client.get_health_status() in ["healthy", "warning"]

                finally:
                    await client.stop()

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker opening and recovery."""
        # Configure with low failure threshold for testing
        config = WebhookReliabilityConfig(
            webhook_url="https://discord.com/api/webhooks/test/token"
        )
        config.circuit_breaker_config.failure_threshold = 2
        config.circuit_breaker_config.timeout = 0.1  # Short timeout for testing

        client = EnhancedDiscordHttpClient(config)

        # Mock failures to trigger circuit breaker
        failed_response = Mock(status=500, text=AsyncMock(return_value='Server Error'))
        success_response = Mock(status=200, headers={'x-ratelimit-remaining': '5'}, text=AsyncMock(return_value='{}'))

        with patch('aiohttp.ClientSession.post', side_effect=[failed_response, failed_response, success_response]):
            with patch('asyncio.sleep'):
                # First two requests should fail and open circuit
                result1 = await client.send_webhook({"content": "Message 1"})
                result2 = await client.send_webhook({"content": "Message 2"})

                assert result1 is False
                assert result2 is False

                # Circuit should be open now
                with pytest.raises(DiscordNotificationError, match="Circuit breaker"):
                    await client.send_webhook({"content": "Message 3"})

                # Wait for circuit breaker timeout
                await asyncio.sleep(0.2)

                # Next request should succeed and close circuit
                result4 = await client.send_webhook({"content": "Message 4"})
                assert result4 is True


class TestFactoryFunction:
    """Test cases for create_enhanced_discord_client factory."""

    def test_create_with_config_object(self):
        """Test creating client with WebhookReliabilityConfig object."""
        config = WebhookReliabilityConfig(
            webhook_url="https://discord.com/api/webhooks/test/token",
            timeout=15
        )
        client = create_enhanced_discord_client(config)

        assert isinstance(client, EnhancedDiscordHttpClient)
        assert client._config.timeout == 15

    def test_create_with_webhook_url(self):
        """Test creating client with just webhook URL."""
        webhook_url = "https://discord.com/api/webhooks/test/token"
        client = create_enhanced_discord_client(webhook_url)

        assert isinstance(client, EnhancedDiscordHttpClient)
        assert client._config.webhook_url == webhook_url

    def test_create_with_custom_components(self):
        """Test creating client with custom component configurations."""
        config = WebhookReliabilityConfig(
            webhook_url="https://discord.com/api/webhooks/test/token"
        )

        # Customize retry policy
        config.retry_config.max_attempts = 7
        config.retry_config.base_delay = 2.0

        # Customize circuit breaker
        config.circuit_breaker_config.failure_threshold = 10
        config.circuit_breaker_config.timeout = 120.0

        client = create_enhanced_discord_client(config)

        assert isinstance(client, EnhancedDiscordHttpClient)
        assert client._retry_policy._config.max_attempts == 7
        assert client._circuit_breaker._config.failure_threshold == 10

    def test_create_with_invalid_config(self):
        """Test error handling for invalid configuration."""
        # Invalid webhook URL
        with pytest.raises(ValueError):
            create_enhanced_discord_client("not-a-valid-url")

        # Empty webhook URL
        with pytest.raises(ValueError):
            config = WebhookReliabilityConfig(webhook_url="")
            create_enhanced_discord_client(config)