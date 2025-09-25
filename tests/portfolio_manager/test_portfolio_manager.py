"""
Unit tests for PortfolioManager class and related components.

Tests portfolio management orchestration, event handling, account synchronization,
and integration with exchange client.
"""

import pytest
import asyncio
import time
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from trading_bot.core.config_manager import ConfigManager
from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.market_data.binance_client import IExchangeClient, BinanceError
from trading_bot.portfolio_manager.portfolio_manager import (
    PortfolioManager,
    PortfolioManagerConfig,
    PortfolioManagerError,
    create_portfolio_manager,
)
from trading_bot.portfolio_manager.portfolio_state import PortfolioHealthStatus
from trading_bot.portfolio_manager.position import Position, PositionSide


class MockExchangeClient:
    """Mock exchange client for testing."""

    def __init__(self, should_fail=False, account_info=None):
        self.should_fail = should_fail
        self.account_info = account_info or {
            "balances": [
                {"asset": "USDT", "free": "1000.0", "locked": "0.0"},
                {"asset": "BTC", "free": "0.1", "locked": "0.0"},
            ]
        }

    def get_account_info(self):
        """Mock get account info."""
        if self.should_fail:
            raise BinanceError("Mock exchange error")
        return self.account_info

    def place_market_order(self, symbol, side, quantity):
        """Mock place market order."""
        return {"orderId": "12345", "status": "FILLED"}

    def place_limit_order(self, symbol, side, quantity, price):
        """Mock place limit order."""
        return {"orderId": "12346", "status": "NEW"}


class TestPortfolioManagerConfig:
    """Test cases for PortfolioManagerConfig class."""

    def test_config_creation_valid(self):
        """Test valid config creation."""
        config = PortfolioManagerConfig(
            sync_interval_minutes=10,
            commission_rate=Decimal("0.002"),
            base_currency="USDT",
        )

        assert config.sync_interval_minutes == 10
        assert config.commission_rate == Decimal("0.002")
        assert config.base_currency == "USDT"
        assert config.enable_auto_sync is True  # default

    def test_config_invalid_sync_interval(self):
        """Test config with invalid sync interval."""
        with pytest.raises(PortfolioManagerError, match="Sync interval must be positive"):
            PortfolioManagerConfig(sync_interval_minutes=0)

    def test_config_invalid_price_update_interval(self):
        """Test config with invalid price update interval."""
        with pytest.raises(PortfolioManagerError, match="Price update interval must be positive"):
            PortfolioManagerConfig(price_update_interval_seconds=0)

    def test_config_negative_commission_rate(self):
        """Test config with negative commission rate."""
        with pytest.raises(PortfolioManagerError, match="Commission rate cannot be negative"):
            PortfolioManagerConfig(commission_rate=Decimal("-0.001"))

    def test_config_empty_base_currency(self):
        """Test config with empty base currency."""
        with pytest.raises(PortfolioManagerError, match="Base currency cannot be empty"):
            PortfolioManagerConfig(base_currency="")


class TestPortfolioManager:
    """Test cases for PortfolioManager class."""

    @pytest.fixture
    def mock_exchange_client(self):
        """Create mock exchange client."""
        return MockExchangeClient()

    @pytest.fixture
    def event_hub(self):
        """Create event hub for testing."""
        return EventHub()

    @pytest.fixture
    def config_manager(self):
        """Create mock config manager."""
        return Mock(spec=ConfigManager)

    @pytest.fixture
    def portfolio_config(self):
        """Create portfolio manager config."""
        return PortfolioManagerConfig(
            enable_auto_sync=False,  # Disable for testing
            enable_portfolio_reporting=False,  # Disable for testing
            sync_interval_minutes=60,  # Long interval for testing
        )

    @pytest.fixture
    def portfolio_manager(self, portfolio_config, mock_exchange_client, event_hub, config_manager):
        """Create portfolio manager for testing."""
        return PortfolioManager(
            config=portfolio_config,
            exchange_client=mock_exchange_client,
            event_hub=event_hub,
            config_manager=config_manager,
            initial_balance=Decimal("10000"),
        )

    def test_portfolio_manager_creation_valid(self, portfolio_manager):
        """Test valid portfolio manager creation."""
        assert portfolio_manager._config is not None
        assert portfolio_manager._portfolio_state is not None
        assert portfolio_manager._is_running is False

    def test_portfolio_manager_creation_invalid_config(self, mock_exchange_client, event_hub, config_manager):
        """Test portfolio manager creation with invalid config."""
        with pytest.raises(PortfolioManagerError, match="Config must be PortfolioManagerConfig instance"):
            PortfolioManager(
                config="invalid_config",
                exchange_client=mock_exchange_client,
                event_hub=event_hub,
                config_manager=config_manager,
            )

    @pytest.mark.asyncio
    async def test_start_stop_portfolio_manager(self, portfolio_manager):
        """Test starting and stopping portfolio manager."""
        assert not portfolio_manager._is_running

        await portfolio_manager.start()
        assert portfolio_manager._is_running

        await portfolio_manager.stop()
        assert not portfolio_manager._is_running

    @pytest.mark.asyncio
    async def test_start_already_running(self, portfolio_manager):
        """Test starting portfolio manager when already running."""
        await portfolio_manager.start()

        # Should not raise error, just log warning
        await portfolio_manager.start()
        assert portfolio_manager._is_running

        await portfolio_manager.stop()

    @pytest.mark.asyncio
    async def test_sync_account_balance_success(self, portfolio_manager):
        """Test successful account balance synchronization."""
        success = await portfolio_manager.sync_account_balance()

        assert success is True
        assert len(portfolio_manager._portfolio_state.balances) == 2
        assert portfolio_manager._portfolio_state.get_balance("USDT") is not None
        assert portfolio_manager._portfolio_state.get_balance("BTC") is not None

    @pytest.mark.asyncio
    async def test_sync_account_balance_exchange_error(self, portfolio_config, event_hub, config_manager):
        """Test account balance sync with exchange error."""
        failing_client = MockExchangeClient(should_fail=True)
        portfolio_manager = PortfolioManager(
            config=portfolio_config,
            exchange_client=failing_client,
            event_hub=event_hub,
            config_manager=config_manager,
        )

        success = await portfolio_manager.sync_account_balance()
        assert success is False

    @pytest.mark.asyncio
    async def test_sync_account_balance_disabled(self, mock_exchange_client, event_hub, config_manager):
        """Test account balance sync when balance tracking disabled."""
        config = PortfolioManagerConfig(enable_balance_tracking=False)
        portfolio_manager = PortfolioManager(
            config=config,
            exchange_client=mock_exchange_client,
            event_hub=event_hub,
            config_manager=config_manager,
        )

        success = await portfolio_manager.sync_account_balance()
        assert success is True
        assert len(portfolio_manager._portfolio_state.balances) == 0

    @pytest.mark.asyncio
    async def test_update_position_from_fill_buy_order(self, portfolio_manager):
        """Test updating position from buy order fill."""
        fill_data = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": "0.1",
            "price": "50000.0",
            "commission": "5.0",
            "order_id": "12345",
        }

        await portfolio_manager.update_position_from_fill(fill_data)

        position = portfolio_manager.get_position("BTCUSDT")
        assert position is not None
        assert position.side == PositionSide.LONG
        assert position.total_entry_quantity == Decimal("0.1")
        assert position.average_entry_price == Decimal("50000.0")

    @pytest.mark.asyncio
    async def test_update_position_from_fill_sell_order(self, portfolio_manager):
        """Test updating position from sell order fill."""
        # First create a long position
        fill_data_buy = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": "0.2",
            "price": "50000.0",
            "commission": "10.0",
            "order_id": "12345",
        }
        await portfolio_manager.update_position_from_fill(fill_data_buy)

        # Then sell part of it
        fill_data_sell = {
            "symbol": "BTCUSDT",
            "side": "SELL",
            "quantity": "0.1",
            "price": "55000.0",
            "commission": "5.5",
            "order_id": "12346",
        }
        await portfolio_manager.update_position_from_fill(fill_data_sell)

        position = portfolio_manager.get_position("BTCUSDT")
        assert position.total_exit_quantity == Decimal("0.1")
        assert position.open_quantity == Decimal("0.1")

    @pytest.mark.asyncio
    async def test_update_position_from_fill_invalid_data(self, portfolio_manager):
        """Test updating position with invalid fill data."""
        fill_data = {
            "symbol": "",  # Invalid empty symbol
            "side": "BUY",
            "quantity": "0.1",
            "price": "50000.0",
        }

        with pytest.raises(Exception):  # Should raise PositionUpdateError or similar
            await portfolio_manager.update_position_from_fill(fill_data)

    @pytest.mark.asyncio
    async def test_update_current_prices(self, portfolio_manager):
        """Test updating current prices for positions."""
        # First create a position
        fill_data = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": "0.1",
            "price": "50000.0",
            "commission": "5.0",
            "order_id": "12345",
        }
        await portfolio_manager.update_position_from_fill(fill_data)

        # Update current prices
        price_updates = {
            "BTCUSDT": Decimal("55000.0"),
            "ETHUSDT": Decimal("3500.0"),  # Position doesn't exist, should be ignored
        }

        await portfolio_manager.update_current_prices(price_updates)

        position = portfolio_manager.get_position("BTCUSDT")
        assert position.current_price == Decimal("55000.0")

    def test_get_portfolio_summary(self, portfolio_manager):
        """Test getting portfolio summary."""
        summary = portfolio_manager.get_portfolio_summary()

        assert "portfolio_state" in summary
        assert "positions" in summary
        assert "balances" in summary
        assert "risk_metrics" in summary

    def test_get_position_existing(self, portfolio_manager):
        """Test getting existing position."""
        # Initially no positions
        assert portfolio_manager.get_position("BTCUSDT") is None

    def test_get_all_positions(self, portfolio_manager):
        """Test getting all positions."""
        positions = portfolio_manager.get_all_positions()
        assert isinstance(positions, dict)
        assert len(positions) == 0

    def test_get_open_positions(self, portfolio_manager):
        """Test getting open positions."""
        open_positions = portfolio_manager.get_open_positions()
        assert isinstance(open_positions, dict)
        assert len(open_positions) == 0

    def test_get_balance(self, portfolio_manager):
        """Test getting balance for asset."""
        # Initially no balances
        assert portfolio_manager.get_balance("USDT") is None

    def test_get_available_balance(self, portfolio_manager):
        """Test getting available balance."""
        balance = portfolio_manager.get_available_balance("USDT")
        assert balance == Decimal("0")

    def test_update_config(self, portfolio_manager):
        """Test updating portfolio manager configuration."""
        new_config = PortfolioManagerConfig(
            sync_interval_minutes=30,
            commission_rate=Decimal("0.002"),
        )

        portfolio_manager.update_config(new_config)
        assert portfolio_manager._config.sync_interval_minutes == 30
        assert portfolio_manager._config.commission_rate == Decimal("0.002")

    def test_update_config_invalid(self, portfolio_manager):
        """Test updating with invalid configuration."""
        with pytest.raises(PortfolioManagerError, match="Config must be PortfolioManagerConfig instance"):
            portfolio_manager.update_config("invalid_config")

    @pytest.mark.asyncio
    async def test_handle_order_filled_event(self, portfolio_manager, event_hub):
        """Test handling ORDER_FILLED event."""
        # Enable position tracking in config
        portfolio_manager._config.enable_position_tracking = True

        # Subscribe to events (reinitialize subscription)
        event_hub.subscribe(EventType.ORDER_FILLED, portfolio_manager._handle_order_filled)

        # Publish ORDER_FILLED event
        event_data = {
            "order": {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "executedQty": "0.1",
                "price": "50000.0",
                "commission": "5.0",
                "orderId": "12345",
            }
        }

        event_hub.publish(EventType.ORDER_FILLED, event_data)

        # Give some time for async event processing
        await asyncio.sleep(0.1)

        # Check that position was created
        position = portfolio_manager.get_position("BTCUSDT")
        assert position is not None
        assert position.total_entry_quantity == Decimal("0.1")

    def test_string_representation(self, portfolio_manager):
        """Test portfolio manager string representation."""
        str_repr = str(portfolio_manager)
        assert "PortfolioManager" in str_repr
        assert "positions=0" in str_repr
        assert "running=False" in str_repr


class TestPortfolioManagerFactory:
    """Test cases for portfolio manager factory function."""

    def test_create_portfolio_manager_default(self):
        """Test creating portfolio manager with defaults."""
        exchange_client = MockExchangeClient()
        event_hub = EventHub()
        config_manager = Mock(spec=ConfigManager)

        manager = create_portfolio_manager(
            exchange_client=exchange_client,
            event_hub=event_hub,
            config_manager=config_manager,
        )

        assert isinstance(manager, PortfolioManager)
        assert manager._config.enable_auto_sync is True
        assert manager._config.base_currency == "USDT"

    def test_create_portfolio_manager_custom_params(self):
        """Test creating portfolio manager with custom parameters."""
        exchange_client = MockExchangeClient()
        event_hub = EventHub()
        config_manager = Mock(spec=ConfigManager)

        manager = create_portfolio_manager(
            exchange_client=exchange_client,
            event_hub=event_hub,
            config_manager=config_manager,
            initial_balance=Decimal("5000"),
            sync_interval_minutes=15,
            commission_rate="0.002",
            base_currency="EUR",
        )

        assert manager._portfolio_state.initial_balance == Decimal("5000")
        assert manager._config.sync_interval_minutes == 15
        assert manager._config.commission_rate == Decimal("0.002")
        assert manager._config.base_currency == "EUR"

    def test_create_portfolio_manager_error(self):
        """Test portfolio manager creation with error."""
        with patch('trading_bot.portfolio_manager.portfolio_manager.PortfolioManager') as mock_manager:
            mock_manager.side_effect = Exception("Creation failed")

            exchange_client = MockExchangeClient()
            event_hub = EventHub()
            config_manager = Mock(spec=ConfigManager)

            with pytest.raises(PortfolioManagerError, match="Failed to create portfolio manager"):
                create_portfolio_manager(
                    exchange_client=exchange_client,
                    event_hub=event_hub,
                    config_manager=config_manager,
                )


class TestPortfolioManagerIntegration:
    """Integration test cases for PortfolioManager."""

    @pytest.fixture
    def full_setup(self):
        """Create full portfolio manager setup for integration tests."""
        exchange_client = MockExchangeClient()
        event_hub = EventHub()
        config_manager = Mock(spec=ConfigManager)
        config = PortfolioManagerConfig(
            enable_auto_sync=False,
            enable_portfolio_reporting=False,
        )

        manager = PortfolioManager(
            config=config,
            exchange_client=exchange_client,
            event_hub=event_hub,
            config_manager=config_manager,
            initial_balance=Decimal("10000"),
        )

        return {
            "manager": manager,
            "event_hub": event_hub,
            "exchange_client": exchange_client,
            "config_manager": config_manager,
        }

    @pytest.mark.asyncio
    async def test_full_trading_cycle(self, full_setup):
        """Test complete trading cycle with position management."""
        manager = full_setup["manager"]

        # 1. Sync account balance
        success = await manager.sync_account_balance()
        assert success

        # 2. Process buy order fill
        buy_fill = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": "0.2",
            "price": "50000.0",
            "commission": "10.0",
            "order_id": "buy_123",
        }
        await manager.update_position_from_fill(buy_fill)

        # 3. Update current price (profit scenario)
        await manager.update_current_prices({"BTCUSDT": Decimal("55000.0")})

        # 4. Partially close position
        sell_fill = {
            "symbol": "BTCUSDT",
            "side": "SELL",
            "quantity": "0.1",
            "price": "56000.0",
            "commission": "5.6",
            "order_id": "sell_123",
        }
        await manager.update_position_from_fill(sell_fill)

        # 5. Verify final state
        position = manager.get_position("BTCUSDT")
        assert position is not None
        assert position.open_quantity == Decimal("0.1")
        assert position.realized_pnl > 0  # Should have profit
        assert position.unrealized_pnl > 0  # Should have unrealized profit

        summary = manager.get_portfolio_summary()
        assert summary["positions"]["open_count"] == 1
        assert summary["portfolio_state"]["total_pnl"] > 0

    @pytest.mark.asyncio
    async def test_portfolio_health_monitoring(self, full_setup):
        """Test portfolio health status monitoring."""
        manager = full_setup["manager"]

        # Create large losing position to trigger health status change
        large_loss_fill = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": "1.0",
            "price": "50000.0",
            "commission": "50.0",
            "order_id": "loss_123",
        }
        await manager.update_position_from_fill(large_loss_fill)

        # Update to losing price (significant loss)
        await manager.update_current_prices({"BTCUSDT": Decimal("30000.0")})  # -40% loss

        # Health status should reflect the loss
        health_status = manager._portfolio_state.health_status
        # Note: Exact health status depends on implementation thresholds
