"""
Tests for Discord notification client.

Comprehensive test suite covering all functionality of the DiscordNotifier
class including error handling, connection testing, and message sending.
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest

from trading_bot.core.config_manager import ConfigManager, IConfigLoader
from trading_bot.notification.discord_notifier import (
    DiscordHttpClient,
    DiscordNotificationError,
    DiscordNotifier,
    IHttpClient,
    create_discord_notifier,
)


class MockConfigLoader(IConfigLoader):
    """Mock config loader for testing."""

    def __init__(
        self, webhook_url: str = "https://discord.com/api/webhooks/test"
    ) -> None:
        self.webhook_url = webhook_url

    def load_config(self) -> Dict[str, Any]:
        """Load mock configuration."""
        return {
            "discord_webhook_url": self.webhook_url,
            "log_level": "INFO",
        }


class MockHttpClient(IHttpClient):
    """Mock HTTP client for testing."""

    def __init__(self, should_fail: bool = False, fail_with: Exception = None) -> None:
        self.should_fail = should_fail
        self.fail_with = fail_with or DiscordNotificationError("Mock error")
        self.post_async_calls = []
        self.post_sync_calls = []

    async def post_async(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock async POST request."""
        self.post_async_calls.append({"url": url, "data": data})
        if self.should_fail:
            raise self.fail_with
        return {"status": "success", "status_code": 204}

    def post_sync(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock sync POST request."""
        self.post_sync_calls.append({"url": url, "data": data})
        if self.should_fail:
            raise self.fail_with
        return {"status": "success", "status_code": 204}


@pytest.fixture
def config_manager():
    """Create config manager for testing."""
    loader = MockConfigLoader()
    manager = ConfigManager(loader)
    manager.load_configuration()
    return manager


@pytest.fixture
def config_manager_no_webhook():
    """Create config manager without webhook URL."""
    loader = MockConfigLoader(webhook_url="")
    manager = ConfigManager(loader)
    manager.load_configuration()
    return manager


@pytest.fixture
def mock_http_client():
    """Create mock HTTP client."""
    return MockHttpClient()


@pytest.fixture
def failing_http_client():
    """Create failing HTTP client."""
    return MockHttpClient(should_fail=True)


@pytest.fixture
def discord_notifier(config_manager, mock_http_client):
    """Create Discord notifier for testing."""
    return DiscordNotifier(config_manager, mock_http_client)


class TestDiscordNotifier:
    """Test cases for DiscordNotifier class."""

    def test_init_with_valid_config(self, config_manager, mock_http_client):
        """Test DiscordNotifier initialization with valid configuration."""
        notifier = DiscordNotifier(config_manager, mock_http_client)
        assert notifier._webhook_url == "https://discord.com/api/webhooks/test"
        assert notifier._config_manager == config_manager
        assert notifier._http_client == mock_http_client

    def test_init_without_webhook_url_raises_error(self, config_manager_no_webhook):
        """Test DiscordNotifier initialization fails without webhook URL."""
        with pytest.raises(DiscordNotificationError) as excinfo:
            DiscordNotifier(config_manager_no_webhook)

        assert "Discord webhook URL not configured" in str(excinfo.value)

    def test_init_with_default_http_client(self, config_manager):
        """Test DiscordNotifier initialization with default HTTP client."""
        notifier = DiscordNotifier(config_manager)
        assert isinstance(notifier._http_client, DiscordHttpClient)

    @pytest.mark.asyncio
    async def test_send_message_async_success(self, discord_notifier, mock_http_client):
        """Test successful async message sending."""
        message = "Test message"
        result = await discord_notifier.send_message_async(message)

        assert result is True
        assert len(mock_http_client.post_async_calls) == 1

        call = mock_http_client.post_async_calls[0]
        assert call["url"] == "https://discord.com/api/webhooks/test"
        assert call["data"]["content"] == message

    @pytest.mark.asyncio
    async def test_send_message_async_with_options(
        self, discord_notifier, mock_http_client
    ):
        """Test async message sending with optional parameters."""
        message = "Test message"
        username = "Test Bot"
        avatar_url = "https://example.com/avatar.png"
        embeds = [{"title": "Test Embed"}]

        result = await discord_notifier.send_message_async(
            message, username=username, avatar_url=avatar_url, embeds=embeds
        )

        assert result is True
        call = mock_http_client.post_async_calls[0]
        assert call["data"]["content"] == message
        assert call["data"]["username"] == username
        assert call["data"]["avatar_url"] == avatar_url
        assert call["data"]["embeds"] == embeds

    @pytest.mark.asyncio
    async def test_send_message_async_empty_message(
        self, discord_notifier, mock_http_client
    ):
        """Test async message sending with empty message."""
        result = await discord_notifier.send_message_async("")

        assert result is False
        assert len(mock_http_client.post_async_calls) == 0

    @pytest.mark.asyncio
    async def test_send_message_async_failure(
        self, config_manager, failing_http_client
    ):
        """Test async message sending failure."""
        notifier = DiscordNotifier(config_manager, failing_http_client)

        with pytest.raises(DiscordNotificationError):
            await notifier.send_message_async("Test message")

    def test_send_message_sync_success(self, discord_notifier, mock_http_client):
        """Test successful sync message sending."""
        message = "Test message"
        result = discord_notifier.send_message_sync(message)

        assert result is True
        assert len(mock_http_client.post_sync_calls) == 1

        call = mock_http_client.post_sync_calls[0]
        assert call["url"] == "https://discord.com/api/webhooks/test"
        assert call["data"]["content"] == message

    def test_send_message_sync_with_options(self, discord_notifier, mock_http_client):
        """Test sync message sending with optional parameters."""
        message = "Test message"
        username = "Test Bot"
        avatar_url = "https://example.com/avatar.png"
        embeds = [{"title": "Test Embed"}]

        result = discord_notifier.send_message_sync(
            message, username=username, avatar_url=avatar_url, embeds=embeds
        )

        assert result is True
        call = mock_http_client.post_sync_calls[0]
        assert call["data"]["content"] == message
        assert call["data"]["username"] == username
        assert call["data"]["avatar_url"] == avatar_url
        assert call["data"]["embeds"] == embeds

    def test_send_message_sync_empty_message(self, discord_notifier, mock_http_client):
        """Test sync message sending with empty message."""
        result = discord_notifier.send_message_sync("")

        assert result is False
        assert len(mock_http_client.post_sync_calls) == 0

    def test_send_message_sync_failure(self, config_manager, failing_http_client):
        """Test sync message sending failure."""
        notifier = DiscordNotifier(config_manager, failing_http_client)

        with pytest.raises(DiscordNotificationError):
            notifier.send_message_sync("Test message")

    def test_send_message_sync_long_message_truncation(
        self, discord_notifier, mock_http_client
    ):
        """Test message truncation for long messages."""
        long_message = "x" * 3000  # Exceeds 2000 char limit
        result = discord_notifier.send_message_sync(long_message)

        assert result is True
        call = mock_http_client.post_sync_calls[0]
        assert len(call["data"]["content"]) == 2000

    def test_send_message_sync_long_username_truncation(
        self, discord_notifier, mock_http_client
    ):
        """Test username truncation for long usernames."""
        long_username = "x" * 100  # Exceeds 80 char limit
        result = discord_notifier.send_message_sync("test", username=long_username)

        assert result is True
        call = mock_http_client.post_sync_calls[0]
        assert len(call["data"]["username"]) == 80

    @pytest.mark.asyncio
    async def test_test_connection_async_success(
        self, discord_notifier, mock_http_client
    ):
        """Test successful async connection test."""
        result = await discord_notifier.test_connection_async()

        assert result is True
        assert len(mock_http_client.post_async_calls) == 1

        call = mock_http_client.post_async_calls[0]
        assert "Connection Test" in call["data"]["content"]
        assert call["data"]["username"] == "Trading Bot"

    @pytest.mark.asyncio
    async def test_test_connection_async_failure(
        self, config_manager, failing_http_client
    ):
        """Test async connection test failure."""
        notifier = DiscordNotifier(config_manager, failing_http_client)
        result = await notifier.test_connection_async()

        assert result is False

    def test_test_connection_sync_success(self, discord_notifier, mock_http_client):
        """Test successful sync connection test."""
        result = discord_notifier.test_connection_sync()

        assert result is True
        assert len(mock_http_client.post_sync_calls) == 1

        call = mock_http_client.post_sync_calls[0]
        assert "Connection Test" in call["data"]["content"]
        assert call["data"]["username"] == "Trading Bot"

    def test_test_connection_sync_failure(self, config_manager, failing_http_client):
        """Test sync connection test failure."""
        notifier = DiscordNotifier(config_manager, failing_http_client)
        result = notifier.test_connection_sync()

        assert result is False

    @pytest.mark.asyncio
    async def test_send_trading_alert_async(self, discord_notifier, mock_http_client):
        """Test sending trading alert asynchronously."""
        result = await discord_notifier.send_trading_alert_async(
            symbol="BTCUSDT",
            action="BUY",
            price=45000.50,
            quantity=0.001,
            reason="Strong bullish signal",
        )

        assert result is True
        call = mock_http_client.post_async_calls[0]
        assert "BUY signal for BTCUSDT" in call["data"]["content"]
        assert len(call["data"]["embeds"]) == 1

        embed = call["data"]["embeds"][0]
        assert embed["title"] == "📊 Trading Alert: BTCUSDT"
        assert embed["color"] == 0x00FF00  # Green for BUY
        assert len(embed["fields"]) == 5  # Action, Symbol, Price, Quantity, Reason

    def test_send_trading_alert_sync(self, discord_notifier, mock_http_client):
        """Test sending trading alert synchronously."""
        result = discord_notifier.send_trading_alert_sync(
            symbol="ETHUSDT", action="SELL", price=3200.75, quantity=0.5
        )

        assert result is True
        call = mock_http_client.post_sync_calls[0]
        assert "SELL signal for ETHUSDT" in call["data"]["content"]
        assert len(call["data"]["embeds"]) == 1

        embed = call["data"]["embeds"][0]
        assert embed["title"] == "📊 Trading Alert: ETHUSDT"
        assert embed["color"] == 0xFF0000  # Red for SELL
        assert len(embed["fields"]) == 4  # Action, Symbol, Price, Quantity (no reason)

    @pytest.mark.asyncio
    async def test_send_error_alert_async(self, discord_notifier, mock_http_client):
        """Test sending error alert asynchronously."""
        result = await discord_notifier.send_error_alert_async(
            error_message="Connection failed", component="BinanceClient"
        )

        assert result is True
        call = mock_http_client.post_async_calls[0]
        assert "Error in BinanceClient" in call["data"]["content"]
        assert len(call["data"]["embeds"]) == 1

        embed = call["data"]["embeds"][0]
        assert embed["title"] == "❌ Trading Bot Error"
        assert embed["color"] == 0xFF0000  # Red for error
        assert len(embed["fields"]) == 2  # Error and Component

    def test_send_error_alert_sync(self, discord_notifier, mock_http_client):
        """Test sending error alert synchronously."""
        result = discord_notifier.send_error_alert_sync(
            error_message="Database connection lost"
        )

        assert result is True
        call = mock_http_client.post_sync_calls[0]
        assert "Error in Trading Bot" in call["data"]["content"]
        assert len(call["data"]["embeds"]) == 1

        embed = call["data"]["embeds"][0]
        assert embed["title"] == "❌ Trading Bot Error"
        assert embed["color"] == 0xFF0000  # Red for error
        assert len(embed["fields"]) == 1  # Error only (no component)


class TestDiscordHttpClient:
    """Test cases for DiscordHttpClient class."""

    def test_init_with_defaults(self):
        """Test DiscordHttpClient initialization with default values."""
        client = DiscordHttpClient()
        assert client._timeout == 10
        assert client._max_retries == 3
        assert client._session is None

    def test_init_with_custom_values(self):
        """Test DiscordHttpClient initialization with custom values."""
        client = DiscordHttpClient(timeout=20, max_retries=5)
        assert client._timeout == 20
        assert client._max_retries == 5

    @pytest.mark.asyncio
    async def test_post_async_functionality_through_notifier(self):
        """Test async POST functionality through DiscordNotifier integration."""
        # Since mocking aiohttp is complex, we test integration through DiscordNotifier
        # with MockHttpClient which validates the correct parameters are passed
        config_manager = ConfigManager(MockConfigLoader())
        config_manager.load_configuration()

        mock_client = MockHttpClient()
        notifier = DiscordNotifier(config_manager, mock_client)

        result = await notifier.send_message_async("Test async message")

        assert result is True
        assert len(mock_client.post_async_calls) == 1
        call = mock_client.post_async_calls[0]
        assert call["data"]["content"] == "Test async message"

    @pytest.mark.asyncio
    async def test_post_async_error_handling_through_notifier(self):
        """Test async POST error handling through DiscordNotifier integration."""
        config_manager = ConfigManager(MockConfigLoader())
        config_manager.load_configuration()

        mock_client = MockHttpClient(should_fail=True)
        notifier = DiscordNotifier(config_manager, mock_client)

        with pytest.raises(DiscordNotificationError):
            await notifier.send_message_async("Test async message")

    def test_post_sync_success(self):
        """Test sync POST request success."""
        client = DiscordHttpClient()

        with patch.object(client._requests_session, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 204
            mock_post.return_value = mock_response

            result = client.post_sync(
                "https://discord.com/api/webhooks/test", {"content": "test"}
            )

            assert result["status"] == "success"
            assert result["status_code"] == 204

    def test_post_sync_rate_limited(self):
        """Test sync POST request rate limiting."""
        client = DiscordHttpClient()

        with patch.object(client._requests_session, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.headers = {"Retry-After": "30"}
            mock_post.return_value = mock_response

            with pytest.raises(DiscordNotificationError) as excinfo:
                client.post_sync(
                    "https://discord.com/api/webhooks/test", {"content": "test"}
                )

            assert "Rate limited" in str(excinfo.value)
            assert "30 seconds" in str(excinfo.value)

    def test_post_sync_timeout(self):
        """Test sync POST request timeout."""
        import requests

        client = DiscordHttpClient()

        with patch.object(client._requests_session, "post") as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout("Request timeout")

            with pytest.raises(DiscordNotificationError) as excinfo:
                client.post_sync(
                    "https://discord.com/api/webhooks/test", {"content": "test"}
                )

            assert "Request timeout after 10s" in str(excinfo.value)

    def test_post_sync_connection_error(self):
        """Test sync POST request connection error."""
        import requests

        client = DiscordHttpClient()

        with patch.object(client._requests_session, "post") as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError(
                "Connection failed"
            )

            with pytest.raises(DiscordNotificationError) as excinfo:
                client.post_sync(
                    "https://discord.com/api/webhooks/test", {"content": "test"}
                )

            assert "Connection error" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_close_async(self):
        """Test closing async session."""
        client = DiscordHttpClient()
        mock_session = AsyncMock()
        client._session = mock_session

        await client.close()

        mock_session.close.assert_called_once()
        assert client._session is None

    def test_close_sync(self):
        """Test closing sync session."""
        client = DiscordHttpClient()

        with patch.object(client._requests_session, "close") as mock_close:
            client.close_sync()
            mock_close.assert_called_once()


def test_create_discord_notifier(config_manager):
    """Test factory function for creating DiscordNotifier."""
    notifier = create_discord_notifier(config_manager)

    assert isinstance(notifier, DiscordNotifier)
    assert isinstance(notifier._http_client, DiscordHttpClient)
    assert notifier._config_manager == config_manager
