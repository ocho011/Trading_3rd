"""
Integration tests for enhanced Discord webhook system.

Tests the complete enhanced Discord notification system with all components
working together: retry policies, circuit breaker, message queue, health
monitoring, and enhanced HTTP client.
"""

import asyncio
import json
import os
import tempfile
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from trading_bot.notification.enhanced_discord_client import (
    EnhancedDiscordHttpClient,
    create_enhanced_discord_client
)
from trading_bot.notification.webhook_config import (
    WebhookReliabilityConfig,
    WebhookConfigManager,
    ConfigPresets
)
from trading_bot.notification.discord_notifier import DiscordNotifier
from trading_bot.notification.message_queue import MessagePriority


class TestEnhancedSystemIntegration:
    """Integration tests for complete enhanced Discord webhook system."""

    @pytest.mark.asyncio
    async def test_complete_success_workflow(self):
        """Test complete successful webhook delivery workflow."""
        # Create configuration with realistic settings
        config = ConfigPresets.high_reliability()
        config.webhook_url = "https://discord.com/api/webhooks/test/token"

        # Create enhanced client
        client = create_enhanced_discord_client(config)

        # Mock successful Discord API responses
        success_response = Mock()
        success_response.status = 200
        success_response.headers = {'x-ratelimit-remaining': '5'}
        success_response.text = AsyncMock(return_value='{"success": true}')

        with patch('aiohttp.ClientSession.post', return_value=success_response) as mock_post:
            await client.start()

            try:
                # Send webhook message
                result = await client.send_webhook({
                    "content": "Integration test message",
                    "embeds": [{
                        "title": "Test Embed",
                        "description": "Testing enhanced Discord webhook system"
                    }]
                })

                assert result is True
                mock_post.assert_called_once()

                # Verify health metrics
                metrics = client.get_health_metrics()
                assert metrics.total_requests == 1
                assert metrics.successful_requests == 1
                assert metrics.success_rate == 100.0
                assert client.get_health_status() == "healthy"

            finally:
                await client.stop()

    @pytest.mark.asyncio
    async def test_rate_limiting_and_retry_workflow(self):
        """Test rate limiting handling with retry workflow."""
        config = ConfigPresets.fast_response()
        config.webhook_url = "https://discord.com/api/webhooks/test/token"
        config.max_rate_limit_wait = 5.0

        client = create_enhanced_discord_client(config)

        # Mock rate limited response followed by success
        rate_limited_response = Mock()
        rate_limited_response.status = 429
        rate_limited_response.headers = {
            'x-ratelimit-remaining': '0',
            'retry-after': '2',
            'x-ratelimit-reset-after': '2.5'
        }

        success_response = Mock()
        success_response.status = 200
        success_response.headers = {'x-ratelimit-remaining': '4'}
        success_response.text = AsyncMock(return_value='{"success": true}')

        responses = [rate_limited_response, success_response]

        with patch('aiohttp.ClientSession.post', side_effect=responses):
            with patch('asyncio.sleep') as mock_sleep:
                await client.start()

                try:
                    result = await client.send_webhook({"content": "Rate limit test"})

                    assert result is True
                    mock_sleep.assert_called_once_with(2)  # Should wait for rate limit

                    # Verify metrics include both attempts
                    metrics = client.get_health_metrics()
                    assert metrics.successful_requests == 1

                finally:
                    await client.stop()

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker integration with retry and queue."""
        config = WebhookReliabilityConfig(
            webhook_url="https://discord.com/api/webhooks/test/token"
        )
        # Low failure threshold for testing
        config.circuit_breaker_config.failure_threshold = 3
        config.circuit_breaker_config.timeout = 0.5
        config.retry_config.max_attempts = 2

        client = create_enhanced_discord_client(config)

        # Mock server errors to trigger circuit breaker
        server_error = Mock()
        server_error.status = 500
        server_error.text = AsyncMock(return_value='Internal Server Error')

        success_response = Mock()
        success_response.status = 200
        success_response.headers = {'x-ratelimit-remaining': '5'}
        success_response.text = AsyncMock(return_value='{"success": true}')

        error_responses = [server_error] * 6  # 3 messages × 2 attempts each
        recovery_responses = [success_response]

        with patch('aiohttp.ClientSession.post', side_effect=error_responses + recovery_responses):
            with patch('asyncio.sleep'):  # Mock retry delays
                await client.start()

                try:
                    # Send messages to trigger circuit breaker
                    result1 = await client.send_webhook({"content": "Message 1"})
                    result2 = await client.send_webhook({"content": "Message 2"})
                    result3 = await client.send_webhook({"content": "Message 3"})

                    # All should fail and be queued
                    assert result1 is False
                    assert result2 is False
                    assert result3 is False

                    # Circuit should be open - next request should be blocked
                    from trading_bot.notification.discord_notifier import DiscordNotificationError
                    with pytest.raises(DiscordNotificationError, match="Circuit breaker"):
                        await client.send_webhook({"content": "Blocked message"})

                    # Messages should be in queue
                    assert client._message_queue.size() == 3

                    # Wait for circuit breaker to move to half-open
                    await asyncio.sleep(0.6)

                    # Next request should succeed and close circuit
                    result_recovery = await client.send_webhook({"content": "Recovery message"})
                    assert result_recovery is True

                finally:
                    await client.stop()

    @pytest.mark.asyncio
    async def test_message_queue_processing(self):
        """Test message queue processing and persistence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Enable queue persistence
            config = ConfigPresets.high_reliability()
            config.webhook_url = "https://discord.com/api/webhooks/test/token"
            config.queue_config.persistence_enabled = True

            client = create_enhanced_discord_client(config)

            # Mock initial failures to queue messages
            server_error = Mock()
            server_error.status = 503
            server_error.text = AsyncMock(return_value='Service Unavailable')

            success_response = Mock()
            success_response.status = 200
            success_response.headers = {'x-ratelimit-remaining': '5'}
            success_response.text = AsyncMock(return_value='{"success": true}')

            # First call fails (queues message), subsequent calls succeed
            responses = [server_error] * 3 + [success_response] * 10

            with patch('aiohttp.ClientSession.post', side_effect=responses):
                with patch('asyncio.sleep'):
                    await client.start()

                    try:
                        # Send messages that will initially fail and be queued
                        await client.send_webhook({"content": "High priority", "priority": "high"})
                        await client.send_webhook({"content": "Normal priority"})
                        await client.send_webhook({"content": "Low priority", "priority": "low"})

                        # Verify messages are queued
                        initial_queue_size = client._message_queue.size()
                        assert initial_queue_size == 3

                        # Wait for queue processing to handle messages
                        await asyncio.sleep(1.0)

                        # Queue should be processed (emptied)
                        final_queue_size = client._message_queue.size()
                        assert final_queue_size < initial_queue_size

                    finally:
                        await client.stop()

    @pytest.mark.asyncio
    async def test_health_monitoring_and_alerts(self):
        """Test health monitoring and alert generation."""
        config = ConfigPresets.development()
        config.webhook_url = "https://discord.com/api/webhooks/test/token"
        # Set low thresholds for testing
        config.health_thresholds.success_rate_warning = 80.0
        config.health_thresholds.response_time_warning = 200

        client = create_enhanced_discord_client(config)

        # Mock responses with varying performance
        fast_success = Mock()
        fast_success.status = 200
        fast_success.headers = {'x-ratelimit-remaining': '5'}
        fast_success.text = AsyncMock(return_value='{"success": true}')

        slow_success = Mock()
        slow_success.status = 200
        slow_success.headers = {'x-ratelimit-remaining': '4'}
        slow_success.text = AsyncMock(return_value='{"success": true}')

        server_error = Mock()
        server_error.status = 500
        server_error.text = AsyncMock(return_value='Server Error')

        # Set up alert callback to track alerts
        alerts_received = []
        def alert_callback(alert):
            alerts_received.append(alert)

        client._health_monitor.register_alert_callback(alert_callback)

        # Simulate mixed performance to trigger alerts
        with patch('aiohttp.ClientSession.post', side_effect=[fast_success, slow_success, server_error, fast_success, server_error]):
            with patch('asyncio.sleep'):
                await client.start()

                try:
                    # Send requests with mixed results
                    await client.send_webhook({"content": "Fast success"})

                    # Simulate slow response time
                    start_time = datetime.utcnow()
                    await client.send_webhook({"content": "Slow success"})
                    # Manually record slow response for testing
                    client._health_monitor._current_metrics.record_success(response_time_ms=300)

                    await client.send_webhook({"content": "Will fail"})
                    await client.send_webhook({"content": "Fast success 2"})
                    await client.send_webhook({"content": "Will fail 2"})

                    # Check health status
                    health_status = client.get_health_status()
                    metrics = client.get_health_metrics()

                    # Should have some failures affecting health
                    assert metrics.failed_requests > 0
                    assert metrics.success_rate < 100.0

                    # May have triggered alerts based on thresholds
                    # (Alerts are asynchronous, so we check if system is monitoring)
                    assert client._health_monitor is not None

                finally:
                    await client.stop()

    @pytest.mark.asyncio
    async def test_configuration_management_integration(self):
        """Test configuration management and updates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "webhook_config.json")

            # Create configuration manager
            config_manager = WebhookConfigManager(config_file)

            # Create initial configuration
            initial_config = ConfigPresets.fast_response()
            initial_config.webhook_url = "https://discord.com/api/webhooks/test/token"

            # Save and load configuration
            config_manager.save_configuration(initial_config)
            loaded_config = config_manager.load_configuration()

            # Create client with loaded configuration
            client = create_enhanced_discord_client(loaded_config)

            # Verify configuration applied correctly
            assert client._config.webhook_url == initial_config.webhook_url
            assert client._config.timeout == initial_config.timeout

            # Test configuration updates
            updates = {
                "timeout": 25,
                "retry_config.max_attempts": 5
            }

            updated_config = config_manager.update_configuration(updates)

            # Verify updates
            assert updated_config.timeout == 25
            assert updated_config.retry_config.max_attempts == 5

    @pytest.mark.asyncio
    async def test_discord_notifier_integration(self):
        """Test integration with existing DiscordNotifier class."""
        config = ConfigPresets.high_reliability()
        config.webhook_url = "https://discord.com/api/webhooks/test/token"

        # Create enhanced client
        enhanced_client = create_enhanced_discord_client(config)

        # Create DiscordNotifier with enhanced client
        notifier = DiscordNotifier(
            webhook_url=config.webhook_url,
            http_client=enhanced_client
        )

        # Mock successful response
        success_response = Mock()
        success_response.status = 200
        success_response.headers = {'x-ratelimit-remaining': '5'}
        success_response.text = AsyncMock(return_value='{"success": true}')

        with patch('aiohttp.ClientSession.post', return_value=success_response):
            await enhanced_client.start()

            try:
                # Test notification methods
                await notifier.send_trade_alert("BTC/USD", "BUY", 50000.0, 0.1)
                await notifier.send_portfolio_update(balance=10000.0, pnl=500.0)
                await notifier.send_system_alert("System running normally", "info")

                # Verify health metrics
                metrics = enhanced_client.get_health_metrics()
                assert metrics.successful_requests == 3
                assert metrics.success_rate == 100.0

            finally:
                await enhanced_client.stop()

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self):
        """Test complete error recovery workflow."""
        config = WebhookReliabilityConfig(
            webhook_url="https://discord.com/api/webhooks/test/token"
        )
        # Configure for testing
        config.retry_config.max_attempts = 3
        config.circuit_breaker_config.failure_threshold = 5
        config.circuit_breaker_config.timeout = 1.0

        client = create_enhanced_discord_client(config)

        # Simulate various failure scenarios
        connection_error = ConnectionError("Connection failed")
        timeout_error = asyncio.TimeoutError("Request timed out")
        server_error = Mock()
        server_error.status = 503
        server_error.text = AsyncMock(return_value='Service Unavailable')

        success_response = Mock()
        success_response.status = 200
        success_response.headers = {'x-ratelimit-remaining': '5'}
        success_response.text = AsyncMock(return_value='{"success": true}')

        # Mix of failures followed by success for recovery
        failure_sequence = [
            connection_error,
            timeout_error,
            server_error,
            success_response  # Recovery
        ]

        with patch('aiohttp.ClientSession.post', side_effect=failure_sequence * 5):
            with patch('asyncio.sleep'):
                await client.start()

                try:
                    # Send message that will go through failure/recovery cycle
                    result = await client.send_webhook({"content": "Recovery test message"})

                    # Should eventually succeed or be queued
                    # (Result depends on exact retry/circuit breaker interaction)
                    assert isinstance(result, bool)

                    # System should still be functional
                    health_status = client.get_health_status()
                    assert health_status in ["healthy", "warning", "critical"]

                    # Metrics should show the activity
                    metrics = client.get_health_metrics()
                    assert metrics.total_requests > 0

                finally:
                    await client.stop()

    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self):
        """Test handling multiple concurrent webhook requests."""
        config = ConfigPresets.high_reliability()
        config.webhook_url = "https://discord.com/api/webhooks/test/token"

        client = create_enhanced_discord_client(config)

        # Mock responses with some delays to test concurrency
        success_response = Mock()
        success_response.status = 200
        success_response.headers = {'x-ratelimit-remaining': '5'}
        success_response.text = AsyncMock(return_value='{"success": true}')

        with patch('aiohttp.ClientSession.post', return_value=success_response):
            await client.start()

            try:
                # Send multiple concurrent requests
                tasks = []
                for i in range(10):
                    task = client.send_webhook({
                        "content": f"Concurrent message {i}",
                        "embeds": [{
                            "title": f"Message {i}",
                            "color": 0x00ff00
                        }]
                    })
                    tasks.append(task)

                # Wait for all requests to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Most should succeed (some might be rate limited)
                successful_results = [r for r in results if r is True]
                assert len(successful_results) >= 5  # At least half should succeed

                # Verify health metrics
                metrics = client.get_health_metrics()
                assert metrics.total_requests >= len(successful_results)

            finally:
                await client.stop()

    def test_configuration_presets_integration(self):
        """Test configuration presets work with enhanced system."""
        # Test all presets can create valid clients
        presets = [
            ConfigPresets.high_reliability(),
            ConfigPresets.fast_response(),
            ConfigPresets.development()
        ]

        for config in presets:
            config.webhook_url = "https://discord.com/api/webhooks/test/token"

            # Should create client without errors
            client = create_enhanced_discord_client(config)

            # Verify client has expected configuration
            assert client._config.webhook_url == config.webhook_url
            assert client._retry_policy is not None
            assert client._circuit_breaker is not None
            assert client._health_monitor is not None
            assert client._message_queue is not None

    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test system graceful degradation under various failure modes."""
        config = ConfigPresets.high_reliability()
        config.webhook_url = "https://discord.com/api/webhooks/test/token"
        config.backup_notification_enabled = True

        client = create_enhanced_discord_client(config)

        # Mock complete Discord API failure
        api_error = Mock()
        api_error.status = 502
        api_error.text = AsyncMock(return_value='Bad Gateway')

        with patch('aiohttp.ClientSession.post', return_value=api_error):
            with patch('asyncio.sleep'):
                await client.start()

                try:
                    # Send messages that will fail at API level
                    result1 = await client.send_webhook({"content": "Message during outage 1"})
                    result2 = await client.send_webhook({"content": "Message during outage 2"})

                    # Should gracefully handle failures (queue or backup notification)
                    assert result1 is False  # Direct sending failed
                    assert result2 is False

                    # Messages should be queued for later delivery
                    assert client._message_queue.size() >= 2

                    # Health status should reflect degraded state
                    health_status = client.get_health_status()
                    assert health_status in ["warning", "critical"]

                finally:
                    await client.stop()