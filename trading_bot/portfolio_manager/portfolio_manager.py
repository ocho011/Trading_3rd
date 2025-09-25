"""
Portfolio Manager module for comprehensive portfolio tracking and management.

This module provides the main PortfolioManager class that orchestrates portfolio
state management, position tracking, and account synchronization following
SOLID principles and event-driven architecture.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

from trading_bot.core.config_manager import ConfigManager
from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.core.logger import get_module_logger
from trading_bot.market_data.binance_client import BinanceError, IExchangeClient
from trading_bot.portfolio_manager.portfolio_state import (
    BalanceInfo,
    PortfolioHealthStatus,
    PortfolioState,
)
from trading_bot.portfolio_manager.position import (
    Position,
    PositionSide,
    PositionStatus,
)


class PortfolioManagerError(Exception):
    """Base exception for portfolio manager errors."""

    pass


class AccountSyncError(PortfolioManagerError):
    """Exception raised when account synchronization fails."""

    pass


class PositionUpdateError(PortfolioManagerError):
    """Exception raised when position updates fail."""

    pass


class InsufficientDataError(PortfolioManagerError):
    """Exception raised when insufficient data for portfolio operations."""

    pass


@dataclass
class PortfolioManagerConfig:
    """Configuration for portfolio manager operations.

    Attributes:
        enable_auto_sync: Enable automatic account synchronization
        sync_interval_minutes: Account synchronization interval in minutes
        enable_position_tracking: Enable position tracking from order fills
        enable_balance_tracking: Enable balance tracking
        enable_risk_monitoring: Enable portfolio risk monitoring
        auto_update_prices: Auto-update current prices for P&L calculations
        price_update_interval_seconds: Price update interval in seconds
        max_position_age_hours: Maximum age for stale position warnings
        enable_portfolio_reporting: Enable portfolio reporting events
        reporting_interval_minutes: Portfolio reporting interval in minutes
        commission_rate: Default commission rate for position calculations
        base_currency: Base currency for portfolio valuation
        enable_health_monitoring: Enable portfolio health status monitoring
        health_check_interval_minutes: Health check interval in minutes
        metadata: Additional configuration metadata
    """

    enable_auto_sync: bool = True
    sync_interval_minutes: int = 5
    enable_position_tracking: bool = True
    enable_balance_tracking: bool = True
    enable_risk_monitoring: bool = True
    auto_update_prices: bool = True
    price_update_interval_seconds: int = 30
    max_position_age_hours: int = 24
    enable_portfolio_reporting: bool = True
    reporting_interval_minutes: int = 15
    commission_rate: Decimal = Decimal("0.001")
    base_currency: str = "USDT"
    enable_health_monitoring: bool = True
    health_check_interval_minutes: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.sync_interval_minutes <= 0:
            raise PortfolioManagerError("Sync interval must be positive")
        if self.price_update_interval_seconds <= 0:
            raise PortfolioManagerError("Price update interval must be positive")
        if self.commission_rate < 0:
            raise PortfolioManagerError("Commission rate cannot be negative")
        if not self.base_currency:
            raise PortfolioManagerError("Base currency cannot be empty")


class IPortfolioManager(ABC):
    """Abstract interface for portfolio manager implementations."""

    @abstractmethod
    async def sync_account_balance(self) -> bool:
        """Synchronize account balance with exchange.

        Returns:
            bool: True if sync successful, False otherwise
        """
        pass

    @abstractmethod
    async def update_position_from_fill(self, fill_data: Dict[str, Any]) -> None:
        """Update position from order fill event.

        Args:
            fill_data: Order fill event data

        Raises:
            PositionUpdateError: If position update fails
        """
        pass

    @abstractmethod
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary.

        Returns:
            Dict containing portfolio state and metrics
        """
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol.

        Args:
            symbol: Symbol to retrieve

        Returns:
            Position or None if not found
        """
        pass

    @abstractmethod
    async def update_current_prices(self, price_updates: Dict[str, Decimal]) -> None:
        """Update current prices for P&L calculations.

        Args:
            price_updates: Dictionary of symbol -> current price

        Raises:
            PortfolioManagerError: If price update fails
        """
        pass


class PortfolioManager(IPortfolioManager):
    """Comprehensive portfolio management system.

    This class orchestrates portfolio state management, position tracking,
    account synchronization, and risk monitoring while following SOLID
    principles and event-driven architecture.

    Attributes:
        _config: Portfolio manager configuration
        _portfolio_state: Current portfolio state
        _exchange_client: Exchange client for account operations
        _event_hub: Event hub for publishing events
        _config_manager: Configuration manager for settings
        _logger: Logger instance for portfolio manager logging
        _last_sync_timestamp: Last successful sync timestamp
        _sync_task: Background sync task
        _price_update_task: Background price update task
        _health_check_task: Background health check task
        _is_running: Manager running status
    """

    def __init__(
        self,
        config: PortfolioManagerConfig,
        exchange_client: IExchangeClient,
        event_hub: EventHub,
        config_manager: ConfigManager,
        initial_balance: Optional[Decimal] = None,
    ) -> None:
        """Initialize portfolio manager with configuration and dependencies.

        Args:
            config: Portfolio manager configuration
            exchange_client: Exchange client for account operations
            event_hub: Event hub for publishing events
            config_manager: Configuration manager for settings
            initial_balance: Optional initial balance for return calculations

        Raises:
            PortfolioManagerError: If initialization fails
        """
        if not isinstance(config, PortfolioManagerConfig):
            raise PortfolioManagerError(
                "Config must be PortfolioManagerConfig instance"
            )

        self._config = config
        self._exchange_client = exchange_client
        self._event_hub = event_hub
        self._config_manager = config_manager
        self._logger = get_module_logger("portfolio_manager")

        # Initialize portfolio state
        self._portfolio_state = PortfolioState(
            base_currency=config.base_currency,
            initial_balance=initial_balance or Decimal("10000"),
        )

        # Background tasks
        self._sync_task: Optional[asyncio.Task] = None
        self._price_update_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._reporting_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._last_sync_timestamp = 0

        # Subscribe to events
        if config.enable_position_tracking:
            self._event_hub.subscribe(EventType.ORDER_FILLED, self._handle_order_filled)

        self._logger.info("Initialized portfolio manager with comprehensive tracking")

    async def start(self) -> None:
        """Start portfolio manager and background tasks.

        Raises:
            PortfolioManagerError: If start fails
        """
        if self._is_running:
            self._logger.warning("Portfolio manager is already running")
            return

        try:
            self._is_running = True

            # Perform initial sync
            await self.sync_account_balance()

            # Start background tasks
            if self._config.enable_auto_sync:
                self._sync_task = asyncio.create_task(self._sync_loop())

            if self._config.auto_update_prices:
                self._price_update_task = asyncio.create_task(self._price_update_loop())

            if self._config.enable_health_monitoring:
                self._health_check_task = asyncio.create_task(self._health_check_loop())

            if self._config.enable_portfolio_reporting:
                self._reporting_task = asyncio.create_task(self._reporting_loop())

            # Publish startup event
            self._event_hub.publish(
                EventType.SYSTEM_STARTUP,
                {
                    "component": "portfolio_manager",
                    "timestamp": int(time.time() * 1000),
                    "config": self._config,
                },
            )

            self._logger.info("Portfolio manager started successfully")

        except Exception as e:
            self._is_running = False
            self._logger.error(f"Failed to start portfolio manager: {e}")
            raise PortfolioManagerError(f"Start failed: {e}")

    async def stop(self) -> None:
        """Stop portfolio manager and background tasks."""
        if not self._is_running:
            return

        self._is_running = False

        # Cancel background tasks
        tasks_to_cancel = [
            self._sync_task,
            self._price_update_task,
            self._health_check_task,
            self._reporting_task,
        ]

        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Publish shutdown event
        self._event_hub.publish(
            EventType.SYSTEM_SHUTDOWN,
            {
                "component": "portfolio_manager",
                "timestamp": int(time.time() * 1000),
                "final_state": self.get_portfolio_summary(),
            },
        )

        self._logger.info("Portfolio manager stopped successfully")

    async def sync_account_balance(self) -> bool:
        """Synchronize account balance with exchange.

        Returns:
            bool: True if sync successful, False otherwise
        """
        try:
            if not self._config.enable_balance_tracking:
                self._logger.debug("Balance tracking disabled, skipping sync")
                return True

            # Get account info from exchange
            account_info = self._exchange_client.get_account_info()

            if not account_info or "balances" not in account_info:
                raise AccountSyncError("Invalid account info received from exchange")

            # Update balances
            updated_balances = {}
            for balance_data in account_info["balances"]:
                asset = balance_data["asset"]
                free = Decimal(str(balance_data["free"]))
                locked = Decimal(str(balance_data["locked"]))

                # Skip zero balances to reduce noise
                if free == 0 and locked == 0:
                    continue

                balance_info = BalanceInfo(
                    asset=asset,
                    free=free,
                    locked=locked,
                    total=free + locked,
                    # Note: USD value would need price conversion
                    usd_value=(
                        free + locked
                        if asset == self._config.base_currency
                        else Decimal("0")
                    ),
                )
                updated_balances[asset] = balance_info

            # Update portfolio state
            self._portfolio_state.update_balances(updated_balances)
            self._last_sync_timestamp = int(time.time() * 1000)

            self._logger.info(f"Account balance synced: {len(updated_balances)} assets")

            # Publish balance update event
            self._event_hub.publish(
                EventType.PORTFOLIO_REBALANCE,
                {
                    "type": "balance_sync",
                    "balances": updated_balances,
                    "timestamp": self._last_sync_timestamp,
                },
            )

            return True

        except BinanceError as e:
            self._logger.error(f"Exchange error during balance sync: {e}")
            return False
        except Exception as e:
            self._logger.error(f"Unexpected error during balance sync: {e}")
            return False

    async def update_position_from_fill(self, fill_data: Dict[str, Any]) -> None:
        """Update position from order fill event.

        Args:
            fill_data: Order fill event data containing:
                - symbol: Trading pair symbol
                - side: Order side (BUY/SELL)
                - quantity: Fill quantity
                - price: Fill price
                - commission: Trading fees
                - order_id: Order ID

        Raises:
            PositionUpdateError: If position update fails
        """
        try:
            # Extract fill data
            symbol = fill_data.get("symbol")
            side = fill_data.get("side")  # BUY or SELL
            quantity = Decimal(str(fill_data.get("quantity", 0)))
            price = Decimal(str(fill_data.get("price", 0)))
            # Handle both 'commission' and 'fees' field names
            commission = Decimal(
                str(fill_data.get("commission") or fill_data.get("fees", 0))
            )
            order_id = fill_data.get("order_id")

            if not symbol or not side:
                raise PositionUpdateError("Missing required fill data: symbol or side")

            if quantity <= 0 or price <= 0:
                raise PositionUpdateError("Invalid fill quantity or price")

            # Get or create position
            position = self._portfolio_state.get_position(symbol)
            if not position:
                # Determine position side from order side
                position_side = (
                    PositionSide.LONG if side == "BUY" else PositionSide.SHORT
                )
                position = Position(
                    symbol=symbol,
                    side=position_side,
                    commission_rate=self._config.commission_rate,
                )

            # Update position based on order side
            if (side == "BUY" and position.side == PositionSide.LONG) or (
                side == "SELL" and position.side == PositionSide.SHORT
            ):
                # Adding to position (entry)
                position.add_entry(
                    price=price,
                    quantity=quantity,
                    order_id=order_id,
                    fees=commission,
                )
                self._logger.info(f"Added entry to {symbol}: {quantity} @ {price}")

            else:
                # Closing position (exit)
                position.add_exit(
                    price=price,
                    quantity=quantity,
                    order_id=order_id,
                    fees=commission,
                )
                self._logger.info(f"Added exit to {symbol}: {quantity} @ {price}")

            # Update position in portfolio state
            self._portfolio_state.add_position(position)

            # Publish position update event
            self._event_hub.publish(
                EventType.PORTFOLIO_REBALANCE,
                {
                    "type": "position_update",
                    "symbol": symbol,
                    "position": position.get_position_summary(),
                    "fill_data": fill_data,
                    "timestamp": int(time.time() * 1000),
                },
            )

            self._logger.debug(f"Position updated for {symbol}: {position}")

        except Exception as e:
            self._logger.error(f"Error updating position from fill: {e}")
            raise PositionUpdateError(f"Position update failed: {e}")

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary.

        Returns:
            Dict containing portfolio state and metrics
        """
        try:
            return self._portfolio_state.get_portfolio_summary()
        except Exception as e:
            self._logger.error(f"Error generating portfolio summary: {e}")
            return {
                "error": f"Failed to generate portfolio summary: {e}",
                "timestamp": int(time.time() * 1000),
            }

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol.

        Args:
            symbol: Symbol to retrieve

        Returns:
            Position or None if not found
        """
        return self._portfolio_state.get_position(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions.

        Returns:
            Dictionary of all positions by symbol
        """
        return self._portfolio_state.positions.copy()

    def get_open_positions(self) -> Dict[str, Position]:
        """Get all open positions.

        Returns:
            Dictionary of open positions by symbol
        """
        return self._portfolio_state.open_positions

    def get_balance(self, asset: str) -> Optional[BalanceInfo]:
        """Get balance information for an asset.

        Args:
            asset: Asset symbol

        Returns:
            BalanceInfo or None if not found
        """
        return self._portfolio_state.get_balance(asset)

    def get_available_balance(self, asset: str) -> Decimal:
        """Get available balance for an asset.

        Args:
            asset: Asset symbol

        Returns:
            Available balance or 0 if not found
        """
        return self._portfolio_state.get_available_balance(asset)

    async def update_current_prices(self, price_updates: Dict[str, Decimal]) -> None:
        """Update current prices for P&L calculations.

        Args:
            price_updates: Dictionary of symbol -> current price

        Raises:
            PortfolioManagerError: If price update fails
        """
        try:
            updated_positions = []

            for symbol, price in price_updates.items():
                position = self._portfolio_state.get_position(symbol)
                if position:
                    position.update_current_price(price)
                    updated_positions.append(symbol)

            if updated_positions:
                self._logger.debug(
                    f"Updated prices for {len(updated_positions)} positions"
                )

                # Publish price update event
                self._event_hub.publish(
                    EventType.PRICE_UPDATE,
                    {
                        "type": "portfolio_prices",
                        "updated_symbols": updated_positions,
                        "price_updates": {
                            k: float(v) for k, v in price_updates.items()
                        },
                        "timestamp": int(time.time() * 1000),
                    },
                )

        except Exception as e:
            self._logger.error(f"Error updating current prices: {e}")
            raise PortfolioManagerError(f"Price update failed: {e}")

    def update_config(self, config: PortfolioManagerConfig) -> None:
        """Update portfolio manager configuration.

        Args:
            config: New portfolio manager configuration

        Raises:
            PortfolioManagerError: If configuration is invalid
        """
        if not isinstance(config, PortfolioManagerConfig):
            raise PortfolioManagerError(
                "Config must be PortfolioManagerConfig instance"
            )

        self._config = config
        self._logger.info("Updated portfolio manager configuration")

    async def _handle_order_filled(self, event_data: Dict[str, Any]) -> None:
        """Handle ORDER_FILLED event.

        Args:
            event_data: Order filled event data
        """
        try:
            # Extract order fill information
            order_data = event_data.get("order", {})
            fill_data = {
                "symbol": order_data.get("symbol"),
                "side": order_data.get("side"),
                "quantity": order_data.get("executedQty", order_data.get("quantity")),
                "price": order_data.get("price"),
                "commission": order_data.get("commission", 0),
                "order_id": order_data.get("orderId"),
            }

            await self.update_position_from_fill(fill_data)

        except Exception as e:
            self._logger.error(f"Error handling order filled event: {e}")

    async def _sync_loop(self) -> None:
        """Background task for periodic account synchronization."""
        while self._is_running:
            try:
                await asyncio.sleep(self._config.sync_interval_minutes * 60)
                if self._is_running:
                    await self.sync_account_balance()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in sync loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

    async def _price_update_loop(self) -> None:
        """Background task for periodic price updates."""
        while self._is_running:
            try:
                await asyncio.sleep(self._config.price_update_interval_seconds)
                if self._is_running:
                    # This would typically fetch current prices from exchange
                    # For now, we just log the intent
                    self._logger.debug("Price update loop iteration")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in price update loop: {e}")
                await asyncio.sleep(30)  # Wait 30 seconds before retry

    async def _health_check_loop(self) -> None:
        """Background task for portfolio health monitoring."""
        while self._is_running:
            try:
                await asyncio.sleep(self._config.health_check_interval_minutes * 60)
                if self._is_running:
                    # Check portfolio health
                    current_health = self._portfolio_state.health_status
                    self._logger.debug(
                        f"Portfolio health status: {current_health.value}"
                    )

                    # Publish health status if critical
                    if current_health in [
                        PortfolioHealthStatus.CRITICAL,
                        PortfolioHealthStatus.EMERGENCY,
                    ]:
                        self._event_hub.publish(
                            EventType.RISK_LIMIT_EXCEEDED,
                            {
                                "type": "portfolio_health",
                                "health_status": current_health.value,
                                "portfolio_summary": self.get_portfolio_summary(),
                                "timestamp": int(time.time() * 1000),
                            },
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    async def _reporting_loop(self) -> None:
        """Background task for periodic portfolio reporting."""
        while self._is_running:
            try:
                await asyncio.sleep(self._config.reporting_interval_minutes * 60)
                if self._is_running:
                    # Generate and publish portfolio report
                    portfolio_summary = self.get_portfolio_summary()
                    self._event_hub.publish(
                        EventType.PORTFOLIO_REBALANCE,
                        {
                            "type": "portfolio_report",
                            "summary": portfolio_summary,
                            "timestamp": int(time.time() * 1000),
                        },
                    )
                    self._logger.debug("Published periodic portfolio report")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in reporting loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    def __str__(self) -> str:
        """String representation of portfolio manager."""
        return (
            f"PortfolioManager(positions={len(self._portfolio_state.positions)} "
            f"health={self._portfolio_state.health_status.value} running={self._is_running})"
        )


def create_portfolio_manager(
    exchange_client: IExchangeClient,
    event_hub: EventHub,
    config_manager: ConfigManager,
    initial_balance: Optional[Decimal] = None,
    **kwargs: Any,
) -> PortfolioManager:
    """Factory function to create PortfolioManager with default configuration.

    Args:
        exchange_client: Exchange client for account operations
        event_hub: Event hub for publishing events
        config_manager: Configuration manager for settings
        initial_balance: Optional initial balance for return calculations
        **kwargs: Additional configuration parameters

    Returns:
        PortfolioManager: Configured portfolio manager instance

    Raises:
        PortfolioManagerError: If creation fails
    """
    try:
        # Create portfolio manager configuration
        config = PortfolioManagerConfig(
            enable_auto_sync=kwargs.get("enable_auto_sync", True),
            sync_interval_minutes=kwargs.get("sync_interval_minutes", 5),
            enable_position_tracking=kwargs.get("enable_position_tracking", True),
            enable_balance_tracking=kwargs.get("enable_balance_tracking", True),
            auto_update_prices=kwargs.get("auto_update_prices", True),
            commission_rate=Decimal(str(kwargs.get("commission_rate", "0.001"))),
            base_currency=kwargs.get("base_currency", "USDT"),
            enable_health_monitoring=kwargs.get("enable_health_monitoring", True),
        )

        # Create portfolio manager
        portfolio_manager = PortfolioManager(
            config=config,
            exchange_client=exchange_client,
            event_hub=event_hub,
            config_manager=config_manager,
            initial_balance=initial_balance,
        )

        return portfolio_manager

    except Exception as e:
        raise PortfolioManagerError(f"Failed to create portfolio manager: {e}")
