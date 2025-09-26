"""
Enhanced Discord HTTP client with robust error handling and retry mechanisms.

Integrates retry policies, circuit breaker, message queue, and health monitoring
for highly reliable Discord webhook transmission. Provides intelligent rate limit
handling and graceful degradation strategies.
"""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, Optional

import aiohttp
import requests
from requests.adapters import HTTPAdapter

from trading_bot.notification.circuit_breaker import (
    CircuitBreakerError,
    ICircuitBreaker,
    create_circuit_breaker,
)
from trading_bot.notification.discord_notifier import (
    DiscordNotificationError,
    IHttpClient,
)
from trading_bot.notification.message_queue import (
    MessagePriority,
    MessageQueue,
    create_message_queue,
)
from trading_bot.notification.retry_policies import (
    IRetryPolicy,
    RetryExecutor,
    create_discord_retry_policy,
)
from trading_bot.notification.webhook_health import (
    WebhookHealthMonitor,
    create_health_monitor,
)


class RateLimitInfo:
    """Information about Discord rate limiting."""

    def __init__(self, retry_after: float, reset_after: Optional[float] = None) -> None:
        """
        Initialize rate limit info.

        Args:
            retry_after: Seconds to wait before retrying
            reset_after: Seconds until rate limit resets
        """
        self.retry_after = retry_after
        self.reset_after = reset_after or retry_after
        self.reset_time = time.time() + self.reset_after

    def is_rate_limited(self) -> bool:
        """Check if still rate limited."""
        return time.time() < self.reset_time

    def time_until_reset(self) -> float:
        """Get seconds until rate limit resets."""
        return max(0.0, self.reset_time - time.time())


class EnhancedDiscordHttpClient(IHttpClient):
    """
    Enhanced HTTP client with comprehensive reliability features.

    Integrates retry policies, circuit breaker, message queue, health monitoring,
    and intelligent rate limit handling for maximum reliability.
    """

    def __init__(
        self,
        webhook_url: str,
        timeout: int = 10,
        retry_policy: Optional[IRetryPolicy] = None,
        circuit_breaker: Optional[ICircuitBreaker] = None,
        message_queue: Optional[MessageQueue] = None,
        health_monitor: Optional[WebhookHealthMonitor] = None,
        enable_fallback_queue: bool = True,
        max_rate_limit_wait: float = 300.0,  # 5 minutes max wait for rate limits
        backup_notification_callback: Optional[
            Callable[[str, Dict[str, Any]], None]
        ] = None,
    ) -> None:
        """
        Initialize enhanced Discord HTTP client.

        Args:
            webhook_url: Discord webhook URL
            timeout: Request timeout in seconds
            retry_policy: Optional retry policy (defaults to Discord-optimized)
            circuit_breaker: Optional circuit breaker (defaults to Discord-optimized)
            message_queue: Optional message queue for failed messages
            health_monitor: Optional health monitor
            enable_fallback_queue: Whether to use fallback queue on failures
            max_rate_limit_wait: Maximum time to wait for rate limits
            backup_notification_callback: Backup notification method when Discord fails
        """
        self._webhook_url = webhook_url
        self._timeout = timeout
        self._retry_policy = retry_policy or create_discord_retry_policy()
        self._circuit_breaker = circuit_breaker or create_circuit_breaker("discord")
        self._message_queue = message_queue if enable_fallback_queue else None
        self._health_monitor = health_monitor or create_health_monitor()
        self._max_rate_limit_wait = max_rate_limit_wait
        self._backup_callback = backup_notification_callback
        self._logger = logging.getLogger(__name__)

        # Rate limiting state
        self._rate_limit_info: Optional[RateLimitInfo] = None

        # Retry executor
        self._retry_executor = RetryExecutor(self._retry_policy)

        # Async HTTP session
        self._session: Optional[aiohttp.ClientSession] = None

        # Sync HTTP session with enhanced retry
        self._requests_session = requests.Session()
        self._setup_requests_session()

        self._logger.info("Enhanced Discord HTTP client initialized for webhook")

    def _setup_requests_session(self) -> None:
        """Setup requests session with basic retry configuration."""
        # Note: We'll handle retries with our RetryExecutor, but keep basic adapter
        adapter = HTTPAdapter(max_retries=0)  # Disable requests retry, use our own
        self._requests_session.mount("http://", adapter)
        self._requests_session.mount("https://", adapter)

    async def post_async(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send async POST request with full reliability features.

        Args:
            url: Target URL (typically webhook URL)
            data: Message data to send

        Returns:
            Dict[str, Any]: Response data

        Raises:
            DiscordNotificationError: If all retry attempts fail
        """
        # Use provided URL or default webhook URL
        target_url = url or self._webhook_url

        # Check rate limiting first
        if self._rate_limit_info and self._rate_limit_info.is_rate_limited():
            wait_time = self._rate_limit_info.time_until_reset()
            if wait_time > self._max_rate_limit_wait:
                # Rate limit wait too long, queue message instead
                await self._handle_fallback_async(
                    target_url, data, "Rate limit wait too long"
                )
                raise DiscordNotificationError(
                    f"Rate limited for {wait_time:.1f}s, message queued"
                )

            self._logger.info(f"Rate limited, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)

        # Execute request with circuit breaker and retry
        try:
            return await self._circuit_breaker.call_async(
                self._retry_executor.execute_async,
                self._send_request_async,
                target_url,
                data,
            )

        except CircuitBreakerError as e:
            # Circuit breaker is open, handle fallback
            await self._handle_fallback_async(
                target_url, data, f"Circuit breaker open: {e}"
            )
            raise DiscordNotificationError(f"Circuit breaker open, message queued: {e}")

        except Exception as e:
            # All retries failed, handle fallback
            await self._handle_fallback_async(
                target_url, data, f"All retries failed: {e}"
            )
            raise

    def post_sync(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send sync POST request with full reliability features.

        Args:
            url: Target URL (typically webhook URL)
            data: Message data to send

        Returns:
            Dict[str, Any]: Response data

        Raises:
            DiscordNotificationError: If all retry attempts fail
        """
        # Use provided URL or default webhook URL
        target_url = url or self._webhook_url

        # Check rate limiting first
        if self._rate_limit_info and self._rate_limit_info.is_rate_limited():
            wait_time = self._rate_limit_info.time_until_reset()
            if wait_time > self._max_rate_limit_wait:
                # Rate limit wait too long, queue message instead
                self._handle_fallback_sync(target_url, data, "Rate limit wait too long")
                raise DiscordNotificationError(
                    f"Rate limited for {wait_time:.1f}s, message queued"
                )

            self._logger.info(f"Rate limited, waiting {wait_time:.1f}s")
            time.sleep(wait_time)

        # Execute request with circuit breaker and retry
        try:
            return self._circuit_breaker.call_sync(
                self._retry_executor.execute_sync,
                self._send_request_sync,
                target_url,
                data,
            )

        except CircuitBreakerError as e:
            # Circuit breaker is open, handle fallback
            self._handle_fallback_sync(target_url, data, f"Circuit breaker open: {e}")
            raise DiscordNotificationError(f"Circuit breaker open, message queued: {e}")

        except Exception as e:
            # All retries failed, handle fallback
            self._handle_fallback_sync(target_url, data, f"All retries failed: {e}")
            raise

    async def _send_request_async(
        self, url: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Internal async request method with health monitoring.

        Args:
            url: Target URL
            data: Request data

        Returns:
            Dict[str, Any]: Response data

        Raises:
            DiscordNotificationError: On request failure
        """
        start_time = self._health_monitor.record_request_start()

        # Initialize session if needed
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)

        try:
            async with self._session.post(
                url, json=data, headers={"Content-Type": "application/json"}
            ) as response:
                # Handle different response scenarios
                if response.status == 204:
                    # Success
                    self._health_monitor.record_request_success(start_time)
                    return {"status": "success", "status_code": response.status}

                elif response.status == 429:
                    # Rate limited - extract rate limit info
                    retry_after = float(response.headers.get("Retry-After", "1"))
                    reset_after = float(
                        response.headers.get("X-RateLimit-Reset-After", retry_after)
                    )

                    self._rate_limit_info = RateLimitInfo(retry_after, reset_after)
                    self._health_monitor.record_request_failure(
                        start_time, "rate_limited", is_rate_limited=True
                    )

                    error_msg = f"Rate limited. Retry after {retry_after}s"
                    raise DiscordNotificationError(error_msg)

                elif response.status >= 400:
                    # HTTP error
                    error_text = await response.text()
                    self._health_monitor.record_request_failure(
                        start_time, f"http_{response.status}"
                    )

                    error_msg = f"Discord API error {response.status}: {error_text}"
                    raise DiscordNotificationError(error_msg)

                else:
                    # Unexpected success response with content
                    self._health_monitor.record_request_success(start_time)
                    try:
                        response_data = await response.json()
                        return response_data
                    except json.JSONDecodeError:
                        return {"status": "success", "status_code": response.status}

        except aiohttp.ClientError as e:
            self._health_monitor.record_request_failure(start_time, "client_error")
            raise DiscordNotificationError(f"HTTP client error: {e}")

        except asyncio.TimeoutError:
            self._health_monitor.record_request_failure(
                start_time, "timeout", is_timeout=True
            )
            raise DiscordNotificationError(f"Request timeout after {self._timeout}s")

        except Exception as e:
            self._health_monitor.record_request_failure(start_time, "unexpected")
            raise DiscordNotificationError(f"Unexpected error: {e}")

    def _send_request_sync(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal sync request method with health monitoring.

        Args:
            url: Target URL
            data: Request data

        Returns:
            Dict[str, Any]: Response data

        Raises:
            DiscordNotificationError: On request failure
        """
        start_time = self._health_monitor.record_request_start()

        try:
            response = self._requests_session.post(
                url,
                json=data,
                timeout=self._timeout,
                headers={"Content-Type": "application/json"},
            )

            # Handle different response scenarios
            if response.status_code == 204:
                # Success
                self._health_monitor.record_request_success(start_time)
                return {"status": "success", "status_code": response.status_code}

            elif response.status_code == 429:
                # Rate limited - extract rate limit info
                retry_after = float(response.headers.get("Retry-After", "1"))
                reset_after = float(
                    response.headers.get("X-RateLimit-Reset-After", retry_after)
                )

                self._rate_limit_info = RateLimitInfo(retry_after, reset_after)
                self._health_monitor.record_request_failure(
                    start_time, "rate_limited", is_rate_limited=True
                )

                error_msg = f"Rate limited. Retry after {retry_after}s"
                raise DiscordNotificationError(error_msg)

            elif response.status_code >= 400:
                # HTTP error
                self._health_monitor.record_request_failure(
                    start_time, f"http_{response.status_code}"
                )

                error_msg = f"Discord API error {response.status_code}: {response.text}"
                raise DiscordNotificationError(error_msg)

            else:
                # Unexpected success response with content
                self._health_monitor.record_request_success(start_time)
                try:
                    return response.json()
                except json.JSONDecodeError:
                    return {"status": "success", "status_code": response.status_code}

        except requests.exceptions.Timeout:
            self._health_monitor.record_request_failure(
                start_time, "timeout", is_timeout=True
            )
            raise DiscordNotificationError(f"Request timeout after {self._timeout}s")

        except requests.exceptions.ConnectionError as e:
            self._health_monitor.record_request_failure(start_time, "connection_error")
            raise DiscordNotificationError(f"Connection error: {e}")

        except requests.exceptions.RequestException as e:
            self._health_monitor.record_request_failure(start_time, "request_error")
            raise DiscordNotificationError(f"HTTP request error: {e}")

        except Exception as e:
            self._health_monitor.record_request_failure(start_time, "unexpected")
            raise DiscordNotificationError(f"Unexpected error: {e}")

    async def _handle_fallback_async(
        self, url: str, data: Dict[str, Any], reason: str
    ) -> None:
        """
        Handle fallback when primary delivery fails.

        Args:
            url: Original target URL
            data: Message data
            reason: Reason for fallback
        """
        self._logger.warning(f"Handling fallback for Discord message: {reason}")

        # Try backup notification callback first
        if self._backup_callback:
            try:
                content = data.get("content", "Discord notification failed")
                self._backup_callback(content, data)
                self._logger.info("Successfully sent backup notification")
                return
            except Exception as e:
                self._logger.error(f"Backup notification failed: {e}")

        # Queue message if queue available
        if self._message_queue:
            try:
                await self._message_queue.enqueue(
                    content=data,
                    webhook_url=url,
                    priority=MessagePriority.HIGH,  # Failed messages get high priority
                    max_attempts=5,  # More attempts for queued messages
                )
                self._logger.info("Successfully queued failed Discord message")
            except Exception as e:
                self._logger.error(f"Failed to queue message: {e}")

    def _handle_fallback_sync(
        self, url: str, data: Dict[str, Any], reason: str
    ) -> None:
        """
        Handle fallback when primary delivery fails (sync version).

        Args:
            url: Original target URL
            data: Message data
            reason: Reason for fallback
        """
        self._logger.warning(f"Handling fallback for Discord message: {reason}")

        # Try backup notification callback first
        if self._backup_callback:
            try:
                content = data.get("content", "Discord notification failed")
                self._backup_callback(content, data)
                self._logger.info("Successfully sent backup notification")
                return
            except Exception as e:
                self._logger.error(f"Backup notification failed: {e}")

        # For sync version, we can't easily queue async, so just log
        if self._message_queue:
            self._logger.warning(
                "Message queuing requires async context, message not queued"
            )

    async def test_connection_async(self) -> bool:
        """
        Test Discord webhook connection asynchronously.

        Returns:
            bool: True if connection successful
        """
        try:
            test_data = {
                "content": "🔔 Enhanced Discord client connection test",
                "username": "Trading Bot Health Check",
            }

            await self.post_async(self._webhook_url, test_data)
            self._logger.info("Enhanced Discord client connection test successful")
            return True

        except Exception as e:
            self._logger.error(f"Enhanced Discord client connection test failed: {e}")
            return False

    def test_connection_sync(self) -> bool:
        """
        Test Discord webhook connection synchronously.

        Returns:
            bool: True if connection successful
        """
        try:
            test_data = {
                "content": "🔔 Enhanced Discord client connection test",
                "username": "Trading Bot Health Check",
            }

            self.post_sync(self._webhook_url, test_data)
            self._logger.info("Enhanced Discord client connection test successful")
            return True

        except Exception as e:
            self._logger.error(f"Enhanced Discord client connection test failed: {e}")
            return False

    def get_health_metrics(self) -> Dict[str, Any]:
        """
        Get current health metrics.

        Returns:
            Dict[str, Any]: Health metrics
        """
        metrics = self._health_monitor.get_health_metrics()
        return {
            **metrics.to_dict(),
            "circuit_breaker_state": self._circuit_breaker.get_state().value,
            "rate_limited": (
                self._rate_limit_info.is_rate_limited()
                if self._rate_limit_info
                else False
            ),
            "rate_limit_reset_time": (
                self._rate_limit_info.time_until_reset()
                if self._rate_limit_info
                else 0.0
            ),
            "queue_size": (
                self._message_queue.get_queue_stats()["total_messages"]
                if self._message_queue
                else 0
            ),
        }

    def get_active_alerts(self) -> list:
        """Get active health alerts."""
        return [alert.to_dict() for alert in self._health_monitor.get_active_alerts()]

    def reset_health_metrics(self) -> None:
        """Reset all health metrics and circuit breaker."""
        self._health_monitor.reset_metrics()
        self._circuit_breaker.reset()
        self._rate_limit_info = None
        self._logger.info("Health metrics and circuit breaker reset")

    async def close_async(self) -> None:
        """Close async resources."""
        if self._session:
            await self._session.close()
            self._session = None

        if self._message_queue:
            await self._message_queue.stop_background_processing()
            self._message_queue.close()

    def close_sync(self) -> None:
        """Close sync resources."""
        self._requests_session.close()

        if self._message_queue:
            self._message_queue.close()


def create_enhanced_discord_client(
    webhook_url: str,
    enable_all_features: bool = True,
    timeout: int = 10,
    max_retries: int = 5,
    enable_queue: bool = True,
    backup_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> EnhancedDiscordHttpClient:
    """
    Factory function to create enhanced Discord HTTP client.

    Args:
        webhook_url: Discord webhook URL
        enable_all_features: Whether to enable all reliability features
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        enable_queue: Whether to enable message queue fallback
        backup_callback: Backup notification callback

    Returns:
        EnhancedDiscordHttpClient: Fully configured client
    """
    # Create components based on configuration
    retry_policy = create_discord_retry_policy(max_attempts=max_retries)
    circuit_breaker = create_circuit_breaker("discord") if enable_all_features else None
    message_queue = create_message_queue() if enable_queue else None
    health_monitor = create_health_monitor() if enable_all_features else None

    return EnhancedDiscordHttpClient(
        webhook_url=webhook_url,
        timeout=timeout,
        retry_policy=retry_policy,
        circuit_breaker=circuit_breaker,
        message_queue=message_queue,
        health_monitor=health_monitor,
        enable_fallback_queue=enable_queue,
        backup_notification_callback=backup_callback,
    )
