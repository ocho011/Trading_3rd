"""
Discord webhook notification client for trading bot.

Provides Discord webhook integration for sending trading alerts, system notifications,
and error messages following SOLID principles with dependency injection.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from trading_bot.core.config_manager import ConfigManager
from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.notification.message_formatters import MessageFormatterFactory


class DiscordNotificationError(Exception):
    """Custom exception for Discord notification related errors."""

    pass


class IHttpClient(ABC):
    """Interface for HTTP client implementations."""

    @abstractmethod
    async def post_async(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send async POST request.

        Args:
            url: Target URL for POST request
            data: JSON data to send

        Returns:
            Dict[str, Any]: Response data

        Raises:
            DiscordNotificationError: If request fails
        """
        pass

    @abstractmethod
    def post_sync(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send synchronous POST request.

        Args:
            url: Target URL for POST request
            data: JSON data to send

        Returns:
            Dict[str, Any]: Response data

        Raises:
            DiscordNotificationError: If request fails
        """
        pass


class DiscordHttpClient(IHttpClient):
    """HTTP client implementation for Discord webhook requests."""

    def __init__(self, timeout: int = 10, max_retries: int = 3) -> None:
        """
        Initialize Discord HTTP client.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self._timeout = timeout
        self._max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None
        self._logger = logging.getLogger(__name__)

        # Configure requests session with retry strategy
        self._requests_session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._requests_session.mount("http://", adapter)
        self._requests_session.mount("https://", adapter)

    async def post_async(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send async POST request to Discord webhook.

        Args:
            url: Discord webhook URL
            data: Message data to send

        Returns:
            Dict[str, Any]: Response data

        Raises:
            DiscordNotificationError: If request fails
        """
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)

        try:
            async with self._session.post(
                url, json=data, headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 204:
                    # Discord returns 204 No Content on success
                    return {"status": "success", "status_code": response.status}
                elif response.status == 429:
                    # Rate limited
                    retry_after = response.headers.get("Retry-After", "1")
                    raise DiscordNotificationError(
                        f"Rate limited. Retry after {retry_after} seconds"
                    )
                elif response.status >= 400:
                    error_text = await response.text()
                    raise DiscordNotificationError(
                        f"Discord API error {response.status}: {error_text}"
                    )

                response_data = await response.json()
                return response_data

        except aiohttp.ClientError as e:
            raise DiscordNotificationError(f"HTTP client error: {e}")
        except asyncio.TimeoutError:
            raise DiscordNotificationError(f"Request timeout after {self._timeout}s")
        except Exception as e:
            raise DiscordNotificationError(f"Unexpected error: {e}")

    def post_sync(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send synchronous POST request to Discord webhook.

        Args:
            url: Discord webhook URL
            data: Message data to send

        Returns:
            Dict[str, Any]: Response data

        Raises:
            DiscordNotificationError: If request fails
        """
        try:
            response = self._requests_session.post(
                url, json=data, timeout=self._timeout
            )

            if response.status_code == 204:
                # Discord returns 204 No Content on success
                return {"status": "success", "status_code": response.status_code}
            elif response.status_code == 429:
                # Rate limited
                retry_after = response.headers.get("Retry-After", "1")
                raise DiscordNotificationError(
                    f"Rate limited. Retry after {retry_after} seconds"
                )
            elif response.status_code >= 400:
                raise DiscordNotificationError(
                    f"Discord API error {response.status_code}: {response.text}"
                )

            # Try to parse JSON response, fallback to success status
            try:
                return response.json()
            except json.JSONDecodeError:
                return {"status": "success", "status_code": response.status_code}

        except requests.exceptions.Timeout:
            raise DiscordNotificationError(f"Request timeout after {self._timeout}s")
        except requests.exceptions.ConnectionError as e:
            raise DiscordNotificationError(f"Connection error: {e}")
        except requests.exceptions.RequestException as e:
            raise DiscordNotificationError(f"HTTP request error: {e}")
        except Exception as e:
            raise DiscordNotificationError(f"Unexpected error: {e}")

    async def close(self) -> None:
        """Close async HTTP session if open."""
        if self._session:
            await self._session.close()
            self._session = None

    def close_sync(self) -> None:
        """Close synchronous HTTP session."""
        self._requests_session.close()


class IDiscordNotifier(ABC):
    """Interface for Discord notification implementations."""

    @abstractmethod
    async def send_message_async(self, message: str, **kwargs) -> bool:
        """Send message asynchronously."""
        pass

    @abstractmethod
    def send_message_sync(self, message: str, **kwargs) -> bool:
        """Send message synchronously."""
        pass

    @abstractmethod
    async def test_connection_async(self) -> bool:
        """Test Discord webhook connection asynchronously."""
        pass

    @abstractmethod
    def test_connection_sync(self) -> bool:
        """Test Discord webhook connection synchronously."""
        pass


class DiscordNotifier(IDiscordNotifier):
    """
    Discord webhook notification client for trading bot with EventHub integration.

    Provides Discord integration for sending trading alerts, system notifications,
    and error messages with proper error handling and connection testing.
    Includes EventHub integration for automatic event-driven notifications.

    This class follows the Single Responsibility Principle by focusing solely
    on Discord webhook communication and event handling.

    Attributes:
        _config_manager: Configuration manager for webhook URL
        _http_client: HTTP client for webhook requests
        _webhook_url: Discord webhook URL
        _logger: Logger instance
        _event_hub: Optional EventHub instance for event subscriptions
        _message_formatter_factory: Factory for creating message formatters
        _subscriptions_active: Track if event subscriptions are active
        _event_handlers: Dictionary of registered event handlers
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        http_client: Optional[IHttpClient] = None,
        event_hub: Optional[EventHub] = None,
        message_formatter_factory: Optional[MessageFormatterFactory] = None
    ) -> None:
        """
        Initialize Discord notifier with config manager dependency.

        Args:
            config_manager: Configuration manager instance
            http_client: Optional HTTP client implementation
            event_hub: Optional EventHub for event subscriptions
            message_formatter_factory: Optional factory for message formatting

        Raises:
            DiscordNotificationError: If webhook URL is not configured
        """
        self._config_manager = config_manager
        self._http_client = http_client or DiscordHttpClient()
        self._event_hub = event_hub
        self._message_formatter_factory = message_formatter_factory or MessageFormatterFactory()
        self._logger = logging.getLogger(__name__)

        # Track subscription state
        self._subscriptions_active = False
        self._event_handlers: Dict[str, Callable[[Dict[str, Any]], None]] = {}

        # Get webhook URL from config
        notification_config = self._config_manager.get_notification_config()
        self._webhook_url = notification_config.get("discord_webhook_url", "")

        if not self._webhook_url:
            raise DiscordNotificationError(
                "Discord webhook URL not configured. "
                "Set DISCORD_WEBHOOK_URL environment variable."
            )

    async def send_message_async(
        self,
        message: str,
        username: Optional[str] = None,
        avatar_url: Optional[str] = None,
        embeds: Optional[list] = None,
    ) -> bool:
        """
        Send message to Discord webhook asynchronously.

        Args:
            message: Message content to send
            username: Optional custom username for webhook
            avatar_url: Optional custom avatar URL for webhook
            embeds: Optional list of embed objects

        Returns:
            bool: True if message sent successfully, False otherwise

        Raises:
            DiscordNotificationError: If message sending fails
        """
        if not message.strip():
            self._logger.warning(
                "Empty message provided, skipping Discord notification"
            )
            return False

        # Build payload
        payload = {"content": message[:2000]}  # Discord message limit

        if username:
            payload["username"] = username[:80]  # Discord username limit

        if avatar_url:
            payload["avatar_url"] = avatar_url

        if embeds:
            payload["embeds"] = embeds[:10]  # Discord embed limit

        try:
            self._logger.debug(f"Sending Discord message: {message[:100]}...")
            await self._http_client.post_async(self._webhook_url, payload)
            self._logger.info("Discord message sent successfully")
            return True

        except DiscordNotificationError as e:
            self._logger.error(f"Failed to send Discord message: {e}")
            raise
        except Exception as e:
            error_msg = f"Unexpected error sending Discord message: {e}"
            self._logger.error(error_msg)
            raise DiscordNotificationError(error_msg)

    def send_message_sync(
        self,
        message: str,
        username: Optional[str] = None,
        avatar_url: Optional[str] = None,
        embeds: Optional[list] = None,
    ) -> bool:
        """
        Send message to Discord webhook synchronously.

        Args:
            message: Message content to send
            username: Optional custom username for webhook
            avatar_url: Optional custom avatar URL for webhook
            embeds: Optional list of embed objects

        Returns:
            bool: True if message sent successfully, False otherwise

        Raises:
            DiscordNotificationError: If message sending fails
        """
        if not message.strip():
            self._logger.warning(
                "Empty message provided, skipping Discord notification"
            )
            return False

        # Build payload
        payload = {"content": message[:2000]}  # Discord message limit

        if username:
            payload["username"] = username[:80]  # Discord username limit

        if avatar_url:
            payload["avatar_url"] = avatar_url

        if embeds:
            payload["embeds"] = embeds[:10]  # Discord embed limit

        try:
            self._logger.debug(f"Sending Discord message: {message[:100]}...")
            self._http_client.post_sync(self._webhook_url, payload)
            self._logger.info("Discord message sent successfully")
            return True

        except DiscordNotificationError as e:
            self._logger.error(f"Failed to send Discord message: {e}")
            raise
        except Exception as e:
            error_msg = f"Unexpected error sending Discord message: {e}"
            self._logger.error(error_msg)
            raise DiscordNotificationError(error_msg)

    async def test_connection_async(self) -> bool:
        """
        Test Discord webhook connection asynchronously.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self._logger.info("Testing Discord webhook connection...")
            test_message = (
                "🔔 Trading Bot Connection Test - Discord webhook is working!"
            )

            await self.send_message_async(
                test_message, username="Trading Bot", avatar_url=None
            )

            self._logger.info("Discord webhook connection test successful")
            return True

        except DiscordNotificationError as e:
            self._logger.error(f"Discord webhook connection test failed: {e}")
            return False
        except Exception as e:
            self._logger.error(f"Unexpected error during connection test: {e}")
            return False

    def test_connection_sync(self) -> bool:
        """
        Test Discord webhook connection synchronously.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self._logger.info("Testing Discord webhook connection...")
            test_message = (
                "🔔 Trading Bot Connection Test - Discord webhook is working!"
            )

            self.send_message_sync(
                test_message, username="Trading Bot", avatar_url=None
            )

            self._logger.info("Discord webhook connection test successful")
            return True

        except DiscordNotificationError as e:
            self._logger.error(f"Discord webhook connection test failed: {e}")
            return False
        except Exception as e:
            self._logger.error(f"Unexpected error during connection test: {e}")
            return False

    async def send_trading_alert_async(
        self,
        symbol: str,
        action: str,
        price: float,
        quantity: float,
        reason: str = "",
    ) -> bool:
        """
        Send formatted trading alert to Discord asynchronously.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            action: Trading action (e.g., 'BUY', 'SELL')
            price: Execution price
            quantity: Trade quantity
            reason: Optional reason for the trade

        Returns:
            bool: True if alert sent successfully
        """
        embed = {
            "title": f"📊 Trading Alert: {symbol}",
            "color": 0x00FF00 if action.upper() == "BUY" else 0xFF0000,
            "fields": [
                {"name": "Action", "value": action.upper(), "inline": True},
                {"name": "Symbol", "value": symbol, "inline": True},
                {"name": "Price", "value": f"${price:.8f}", "inline": True},
                {"name": "Quantity", "value": f"{quantity:.8f}", "inline": True},
            ],
            "timestamp": f"{asyncio.get_event_loop().time()}",
        }

        if reason:
            embed["fields"].append({"name": "Reason", "value": reason, "inline": False})

        message = f"🚨 {action.upper()} signal for {symbol} at ${price:.8f}"
        return await self.send_message_async(
            message, username="Trading Bot", embeds=[embed]
        )

    def send_trading_alert_sync(
        self,
        symbol: str,
        action: str,
        price: float,
        quantity: float,
        reason: str = "",
    ) -> bool:
        """
        Send formatted trading alert to Discord synchronously.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            action: Trading action (e.g., 'BUY', 'SELL')
            price: Execution price
            quantity: Trade quantity
            reason: Optional reason for the trade

        Returns:
            bool: True if alert sent successfully
        """
        import time

        embed = {
            "title": f"📊 Trading Alert: {symbol}",
            "color": 0x00FF00 if action.upper() == "BUY" else 0xFF0000,
            "fields": [
                {"name": "Action", "value": action.upper(), "inline": True},
                {"name": "Symbol", "value": symbol, "inline": True},
                {"name": "Price", "value": f"${price:.8f}", "inline": True},
                {"name": "Quantity", "value": f"{quantity:.8f}", "inline": True},
            ],
            "timestamp": f"{time.time()}",
        }

        if reason:
            embed["fields"].append({"name": "Reason", "value": reason, "inline": False})

        message = f"🚨 {action.upper()} signal for {symbol} at ${price:.8f}"
        return self.send_message_sync(message, username="Trading Bot", embeds=[embed])

    async def send_error_alert_async(
        self, error_message: str, component: str = ""
    ) -> bool:
        """
        Send error alert to Discord asynchronously.

        Args:
            error_message: Error message to send
            component: Optional component name where error occurred

        Returns:
            bool: True if alert sent successfully
        """
        embed = {
            "title": "❌ Trading Bot Error",
            "color": 0xFF0000,
            "fields": [
                {"name": "Error", "value": error_message[:1000], "inline": False}
            ],
            "timestamp": f"{asyncio.get_event_loop().time()}",
        }

        if component:
            embed["fields"].append(
                {"name": "Component", "value": component, "inline": True}
            )

        message = f"❌ Error in {component or 'Trading Bot'}: {error_message[:100]}..."
        return await self.send_message_async(
            message, username="Trading Bot Error", embeds=[embed]
        )

    def send_error_alert_sync(self, error_message: str, component: str = "") -> bool:
        """
        Send error alert to Discord synchronously.

        Args:
            error_message: Error message to send
            component: Optional component name where error occurred

        Returns:
            bool: True if alert sent successfully
        """
        import time

        embed = {
            "title": "❌ Trading Bot Error",
            "color": 0xFF0000,
            "fields": [
                {"name": "Error", "value": error_message[:1000], "inline": False}
            ],
            "timestamp": f"{time.time()}",
        }

        if component:
            embed["fields"].append(
                {"name": "Component", "value": component, "inline": True}
            )

        message = f"❌ Error in {component or 'Trading Bot'}: {error_message[:100]}..."
        return self.send_message_sync(
            message, username="Trading Bot Error", embeds=[embed]
        )

    def subscribe_to_events(self) -> None:
        """
        Subscribe to relevant EventHub events for Discord notifications.

        This method registers the DiscordNotifier to listen for specific trading bot events
        and send formatted Discord notifications when they occur.

        Raises:
            DiscordNotificationError: If EventHub is not configured or subscription fails
        """
        if not self._event_hub:
            raise DiscordNotificationError(
                "EventHub not configured. Cannot subscribe to events."
            )

        if self._subscriptions_active:
            self._logger.warning("EventHub subscriptions already active")
            return

        try:
            # Define event handlers mapping
            event_mappings = {
                EventType.ORDER_FILLED: self._handle_order_filled_async,
                EventType.ERROR_OCCURRED: self._handle_error_occurred_async,
                EventType.CONNECTION_LOST: self._handle_connection_event_async,
                EventType.CONNECTION_RESTORED: self._handle_connection_event_async,
                EventType.TRADING_SIGNAL_GENERATED: self._handle_trading_signal_async,
                EventType.RISK_LIMIT_EXCEEDED: self._handle_risk_limit_exceeded_async,
            }

            # Subscribe to each event type
            for event_type, handler in event_mappings.items():
                self._event_hub.subscribe(event_type, handler)
                self._event_handlers[event_type] = handler
                self._logger.debug(f"Subscribed to {event_type} events")

            self._subscriptions_active = True
            self._logger.info("Successfully subscribed to EventHub events")

        except Exception as e:
            error_msg = f"Failed to subscribe to EventHub events: {e}"
            self._logger.error(error_msg)
            raise DiscordNotificationError(error_msg)

    def unsubscribe_from_events(self) -> None:
        """
        Unsubscribe from all EventHub events.

        This method removes all event subscriptions for the DiscordNotifier,
        effectively stopping Discord notifications for trading bot events.
        """
        if not self._event_hub or not self._subscriptions_active:
            self._logger.debug("No active EventHub subscriptions to remove")
            return

        try:
            # Unsubscribe from all registered events
            for event_type, handler in self._event_handlers.items():
                try:
                    self._event_hub.unsubscribe(event_type, handler)
                    self._logger.debug(f"Unsubscribed from {event_type} events")
                except KeyError:
                    # Handler was not subscribed, continue with others
                    self._logger.warning(f"Handler not found for {event_type}")
                    continue

            self._event_handlers.clear()
            self._subscriptions_active = False
            self._logger.info("Successfully unsubscribed from all EventHub events")

        except Exception as e:
            self._logger.error(f"Error unsubscribing from EventHub events: {e}")

    async def _handle_order_filled_async(self, event_data: Dict[str, Any]) -> None:
        """
        Handle ORDER_FILLED events and send Discord notifications.

        Args:
            event_data: Event payload containing ExecutionResult information
        """
        try:
            formatter = self._message_formatter_factory.get_formatter(EventType.ORDER_FILLED)
            formatted_message = formatter.format_message(event_data)

            # Send message using the formatted content
            embeds = formatted_message.get("embeds", [])
            content = formatted_message.get("content", "")

            await self._send_formatted_message_async(
                content=content,
                embeds=embeds,
                username="Trading Bot - Order Execution"
            )

            self._logger.debug("Successfully sent ORDER_FILLED notification to Discord")

        except Exception as e:
            self._logger.error(f"Failed to handle ORDER_FILLED event: {e}")

    async def _handle_error_occurred_async(self, event_data: Dict[str, Any]) -> None:
        """
        Handle ERROR_OCCURRED events and send Discord notifications.

        Args:
            event_data: Event payload containing error information
        """
        try:
            formatter = self._message_formatter_factory.get_formatter(EventType.ERROR_OCCURRED)
            formatted_message = formatter.format_message(event_data)

            # Send message using the formatted content
            embeds = formatted_message.get("embeds", [])
            content = formatted_message.get("content", "")

            await self._send_formatted_message_async(
                content=content,
                embeds=embeds,
                username="Trading Bot - Error Alert"
            )

            self._logger.debug("Successfully sent ERROR_OCCURRED notification to Discord")

        except Exception as e:
            self._logger.error(f"Failed to handle ERROR_OCCURRED event: {e}")

    async def _handle_connection_event_async(self, event_data: Dict[str, Any]) -> None:
        """
        Handle CONNECTION_LOST and CONNECTION_RESTORED events and send Discord notifications.

        Args:
            event_data: Event payload containing connection information
        """
        try:
            # Determine the event type from the data
            event_type = event_data.get("event_type")
            if not event_type:
                self._logger.warning("Connection event missing event_type field")
                return

            formatter = self._message_formatter_factory.get_formatter(event_type)
            formatted_message = formatter.format_message(event_data)

            # Send message using the formatted content
            embeds = formatted_message.get("embeds", [])
            content = formatted_message.get("content", "")

            await self._send_formatted_message_async(
                content=content,
                embeds=embeds,
                username="Trading Bot - Connection Monitor"
            )

            self._logger.debug(f"Successfully sent {event_type} notification to Discord")

        except Exception as e:
            self._logger.error(f"Failed to handle connection event: {e}")

    async def _handle_trading_signal_async(self, event_data: Dict[str, Any]) -> None:
        """
        Handle TRADING_SIGNAL_GENERATED events and send Discord notifications.

        Args:
            event_data: Event payload containing TradingSignal information
        """
        try:
            formatter = self._message_formatter_factory.get_formatter(
                EventType.TRADING_SIGNAL_GENERATED
            )
            formatted_message = formatter.format_message(event_data)

            # Send message using the formatted content
            embeds = formatted_message.get("embeds", [])
            content = formatted_message.get("content", "")

            await self._send_formatted_message_async(
                content=content,
                embeds=embeds,
                username="Trading Bot - Signal Generator"
            )

            self._logger.debug("Successfully sent TRADING_SIGNAL_GENERATED notification to Discord")

        except Exception as e:
            self._logger.error(f"Failed to handle TRADING_SIGNAL_GENERATED event: {e}")

    async def _handle_risk_limit_exceeded_async(self, event_data: Dict[str, Any]) -> None:
        """
        Handle RISK_LIMIT_EXCEEDED events and send Discord notifications.

        Args:
            event_data: Event payload containing risk limit information
        """
        try:
            formatter = self._message_formatter_factory.get_formatter(EventType.RISK_LIMIT_EXCEEDED)
            formatted_message = formatter.format_message(event_data)

            # Send message using the formatted content
            embeds = formatted_message.get("embeds", [])
            content = formatted_message.get("content", "")

            await self._send_formatted_message_async(
                content=content,
                embeds=embeds,
                username="Trading Bot - Risk Manager"
            )

            self._logger.debug("Successfully sent RISK_LIMIT_EXCEEDED notification to Discord")

        except Exception as e:
            self._logger.error(f"Failed to handle RISK_LIMIT_EXCEEDED event: {e}")

    async def _send_formatted_message_async(
        self,
        content: str = "",
        embeds: Optional[List[Dict[str, Any]]] = None,
        username: Optional[str] = None
    ) -> bool:
        """
        Send a formatted message to Discord.

        Args:
            content: Message content string
            embeds: List of Discord embed dictionaries
            username: Custom username for the webhook

        Returns:
            bool: True if message sent successfully, False otherwise
        """
        try:
            # Use content if provided, otherwise use a default message from embeds
            if not content and embeds and len(embeds) > 0:
                embed = embeds[0]
                content = f"📢 {embed.get('title', 'Trading Bot Notification')}"
            elif not content:
                content = "📢 Trading Bot Notification"

            return await self.send_message_async(
                message=content,
                username=username,
                embeds=embeds
            )

        except Exception as e:
            self._logger.error(f"Failed to send formatted message to Discord: {e}")
            return False

    def initialize_notifications(self) -> None:
        """
        Initialize Discord notifications by setting up EventHub subscriptions.

        This is a convenience method that sets up all required subscriptions for
        Discord notifications. It ensures the notifier is ready to receive and
        process trading bot events.

        Raises:
            DiscordNotificationError: If EventHub is not configured or initialization fails
        """
        if not self._event_hub:
            raise DiscordNotificationError(
                "EventHub not configured. Cannot initialize notifications."
            )

        try:
            self.subscribe_to_events()
            self._logger.info("Discord notifications initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize Discord notifications: {e}"
            self._logger.error(error_msg)
            raise DiscordNotificationError(error_msg)

    def shutdown_notifications(self) -> None:
        """
        Shutdown Discord notifications by removing EventHub subscriptions.

        This method performs cleanup operations to ensure proper shutdown of
        Discord notifications. It should be called before application exit.
        """
        try:
            self.unsubscribe_from_events()
            self._logger.info("Discord notifications shutdown successfully")
        except Exception as e:
            self._logger.error(f"Error during Discord notifications shutdown: {e}")

    async def close_async(self) -> None:
        """Close async resources."""
        # Clean up event subscriptions first
        self.shutdown_notifications()

        if hasattr(self._http_client, "close"):
            await self._http_client.close()

    def close_sync(self) -> None:
        """Close synchronous resources."""
        if hasattr(self._http_client, "close_sync"):
            self._http_client.close_sync()


def create_discord_notifier(
    config_manager: ConfigManager,
    event_hub: Optional[EventHub] = None,
    message_formatter_factory: Optional[MessageFormatterFactory] = None
) -> DiscordNotifier:
    """
    Factory function to create DiscordNotifier with default HTTP client.

    Args:
        config_manager: Configuration manager instance
        event_hub: Optional EventHub for event subscriptions
        message_formatter_factory: Optional factory for message formatting

    Returns:
        DiscordNotifier: Configured Discord notifier instance
    """
    http_client = DiscordHttpClient(timeout=10, max_retries=3)
    return DiscordNotifier(
        config_manager=config_manager,
        http_client=http_client,
        event_hub=event_hub,
        message_formatter_factory=message_formatter_factory
    )
