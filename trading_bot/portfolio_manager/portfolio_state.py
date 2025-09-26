"""
Portfolio state management module for comprehensive portfolio tracking.

This module provides centralized portfolio state management with balance tracking,
position aggregation, risk metrics, and performance analytics following
SOLID principles.
"""

import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from trading_bot.portfolio_manager.position import (
    Position,
    PositionSide,
    PositionStatus,
)


class PortfolioError(Exception):
    """Base exception for portfolio-related errors."""


class PortfolioStateError(PortfolioError):
    """Exception raised for portfolio state management errors."""


class InsufficientBalanceError(PortfolioError):
    """Exception raised when insufficient balance for operations."""


class PortfolioHealthStatus(Enum):
    """Portfolio health status enumeration."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class BalanceInfo:
    """Represents account balance information for a specific asset.

    Attributes:
        asset: Asset symbol (e.g., 'USDT', 'BTC')
        free: Available balance for trading
        locked: Balance locked in orders
        total: Total balance (free + locked)
        usd_value: USD equivalent value
        last_updated: Last update timestamp
    """

    asset: str
    free: Decimal = Decimal("0")
    locked: Decimal = Decimal("0")
    total: Decimal = Decimal("0")
    usd_value: Decimal = Decimal("0")
    last_updated: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        """Validate balance info after initialization."""
        if not self.asset:
            raise PortfolioStateError("Asset symbol cannot be empty")
        if any(balance < 0 for balance in [self.free, self.locked, self.total]):
            raise PortfolioStateError("Balance amounts cannot be negative")

        # Auto-calculate total if not provided
        if self.total == 0 and (self.free > 0 or self.locked > 0):
            self.total = self.free + self.locked

    @property
    def utilization_percentage(self) -> Decimal:
        """Calculate balance utilization percentage."""
        if self.total == 0:
            return Decimal("0")
        return (self.locked / self.total) * Decimal("100")


@dataclass
class PortfolioMetrics:
    """Portfolio performance and risk metrics.

    Attributes:
        total_value: Total portfolio value in USD
        total_pnl: Total unrealized + realized P&L
        total_unrealized_pnl: Total unrealized P&L
        total_realized_pnl: Total realized P&L
        total_fees: Total trading fees paid
        open_positions_count: Number of open positions
        winning_positions: Number of positions with positive P&L
        losing_positions: Number of positions with negative P&L
        largest_winner: Largest winning position P&L
        largest_loser: Largest losing position P&L
        portfolio_return: Overall portfolio return percentage
        sharpe_ratio: Risk-adjusted return metric
        max_drawdown: Maximum drawdown percentage
        value_at_risk: Portfolio Value at Risk (VaR)
        exposure_by_symbol: Position exposure by symbol
        concentration_risk: Portfolio concentration risk score
    """

    total_value: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")
    total_unrealized_pnl: Decimal = Decimal("0")
    total_realized_pnl: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")
    open_positions_count: int = 0
    winning_positions: int = 0
    losing_positions: int = 0
    largest_winner: Decimal = Decimal("0")
    largest_loser: Decimal = Decimal("0")
    portfolio_return: Decimal = Decimal("0")
    sharpe_ratio: Optional[Decimal] = None
    max_drawdown: Decimal = Decimal("0")
    value_at_risk: Optional[Decimal] = None
    exposure_by_symbol: Dict[str, Decimal] = field(default_factory=dict)
    concentration_risk: Decimal = Decimal("0")


@dataclass
class PortfolioState:
    """Comprehensive portfolio state management.

    This class maintains the complete state of the trading portfolio including
    positions, balances, metrics, and risk assessments following SOLID principles.

    Attributes:
        positions: Dictionary of positions by symbol
        balances: Dictionary of balance information by asset
        base_currency: Base currency for portfolio valuation (default: USDT)
        initial_balance: Initial portfolio balance for return calculations
        total_deposits: Total deposits made to the portfolio
        total_withdrawals: Total withdrawals from the portfolio
        health_status: Current portfolio health status
        last_sync_timestamp: Last balance synchronization timestamp
        created_timestamp: Portfolio creation timestamp
        updated_timestamp: Last update timestamp
        metadata: Additional portfolio metadata
    """

    positions: Dict[str, Position] = field(default_factory=dict)
    balances: Dict[str, BalanceInfo] = field(default_factory=dict)
    base_currency: str = "USDT"
    initial_balance: Decimal = Decimal("0")
    total_deposits: Decimal = Decimal("0")
    total_withdrawals: Decimal = Decimal("0")
    health_status: PortfolioHealthStatus = PortfolioHealthStatus.HEALTHY
    last_sync_timestamp: int = 0
    created_timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize portfolio state after creation."""
        if not self.base_currency:
            raise PortfolioStateError("Base currency cannot be empty")
        self.updated_timestamp = int(time.time() * 1000)

    def add_position(self, position: Position) -> None:
        """Add or update position in portfolio.

        Args:
            position: Position to add or update

        Raises:
            PortfolioStateError: If position is invalid
        """
        if not isinstance(position, Position):
            raise PortfolioStateError("Position must be Position instance")

        validation_errors = position.validate_position()
        if validation_errors:
            raise PortfolioStateError(
                f"Position validation failed: {validation_errors}"
            )

        self.positions[position.symbol] = position
        self.updated_timestamp = int(time.time() * 1000)
        self._update_health_status()

    def remove_position(self, symbol: str) -> Optional[Position]:
        """Remove position from portfolio.

        Args:
            symbol: Symbol of position to remove

        Returns:
            Removed position or None if not found
        """
        removed_position = self.positions.pop(symbol, None)
        if removed_position:
            self.updated_timestamp = int(time.time() * 1000)
            self._update_health_status()
        return removed_position

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol.

        Args:
            symbol: Symbol to retrieve

        Returns:
            Position or None if not found
        """
        return self.positions.get(symbol)

    def update_balance(self, balance_info: BalanceInfo) -> None:
        """Update balance information for an asset.

        Args:
            balance_info: Balance information to update

        Raises:
            PortfolioStateError: If balance info is invalid
        """
        if not isinstance(balance_info, BalanceInfo):
            raise PortfolioStateError("Balance info must be BalanceInfo instance")

        self.balances[balance_info.asset] = balance_info
        self.updated_timestamp = int(time.time() * 1000)
        self._update_health_status()

    def update_balances(self, balances: Dict[str, BalanceInfo]) -> None:
        """Update multiple balance information entries.

        Args:
            balances: Dictionary of balance information by asset

        Raises:
            PortfolioStateError: If any balance info is invalid
        """
        for asset, balance_info in balances.items():
            if not isinstance(balance_info, BalanceInfo):
                raise PortfolioStateError(
                    f"Balance info for {asset} must be BalanceInfo instance"
                )

        self.balances.update(balances)
        self.last_sync_timestamp = int(time.time() * 1000)
        self.updated_timestamp = int(time.time() * 1000)
        self._update_health_status()

    def get_balance(self, asset: str) -> Optional[BalanceInfo]:
        """Get balance information for an asset.

        Args:
            asset: Asset symbol to retrieve

        Returns:
            BalanceInfo or None if not found
        """
        return self.balances.get(asset)

    def get_available_balance(self, asset: str) -> Decimal:
        """Get available (free) balance for an asset.

        Args:
            asset: Asset symbol

        Returns:
            Available balance or 0 if asset not found
        """
        balance_info = self.balances.get(asset)
        return balance_info.free if balance_info else Decimal("0")

    def get_total_balance(self, asset: str) -> Decimal:
        """Get total balance for an asset.

        Args:
            asset: Asset symbol

        Returns:
            Total balance or 0 if asset not found
        """
        balance_info = self.balances.get(asset)
        return balance_info.total if balance_info else Decimal("0")

    @property
    def open_positions(self) -> Dict[str, Position]:
        """Get all open positions."""
        return {
            symbol: pos
            for symbol, pos in self.positions.items()
            if pos.status in [PositionStatus.OPEN, PositionStatus.PARTIALLY_CLOSED]
        }

    @property
    def closed_positions(self) -> Dict[str, Position]:
        """Get all closed positions."""
        return {
            symbol: pos
            for symbol, pos in self.positions.items()
            if pos.status == PositionStatus.CLOSED
        }

    @property
    def long_positions(self) -> Dict[str, Position]:
        """Get all long positions."""
        return {
            symbol: pos
            for symbol, pos in self.positions.items()
            if pos.side == PositionSide.LONG
        }

    @property
    def short_positions(self) -> Dict[str, Position]:
        """Get all short positions."""
        return {
            symbol: pos
            for symbol, pos in self.positions.items()
            if pos.side == PositionSide.SHORT
        }

    @property
    def total_portfolio_value(self) -> Decimal:
        """Calculate total portfolio value in base currency."""
        total_value = Decimal("0")

        # Add balance values
        for balance_info in self.balances.values():
            total_value += balance_info.usd_value

        # Add position values
        for position in self.positions.values():
            total_value += position.notional_value

        return total_value

    @property
    def total_unrealized_pnl(self) -> Decimal:
        """Calculate total unrealized P&L across all positions."""
        return sum(pos.unrealized_pnl for pos in self.open_positions.values())

    @property
    def total_realized_pnl(self) -> Decimal:
        """Calculate total realized P&L across all positions."""
        return sum(pos.realized_pnl for pos in self.positions.values())

    @property
    def total_pnl(self) -> Decimal:
        """Calculate total P&L (realized + unrealized)."""
        return self.total_realized_pnl + self.total_unrealized_pnl

    @property
    def total_fees(self) -> Decimal:
        """Calculate total fees paid across all positions."""
        return sum(pos.total_fees for pos in self.positions.values())

    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics.

        Returns:
            PortfolioMetrics with current portfolio analytics
        """
        open_positions = list(self.open_positions.values())
        all_positions = list(self.positions.values())

        # Basic metrics
        total_value = self.total_portfolio_value
        total_pnl = self.total_pnl
        total_unrealized = self.total_unrealized_pnl
        total_realized = self.total_realized_pnl
        total_fees = self.total_fees

        # Position statistics
        winning_positions = len([pos for pos in all_positions if pos.total_pnl > 0])
        losing_positions = len([pos for pos in all_positions if pos.total_pnl < 0])

        # Find largest winner/loser
        largest_winner = max(
            (pos.total_pnl for pos in all_positions), default=Decimal("0")
        )
        largest_loser = min(
            (pos.total_pnl for pos in all_positions), default=Decimal("0")
        )

        # Portfolio return calculation
        portfolio_return = Decimal("0")
        if self.initial_balance > 0:
            portfolio_return = (total_pnl / self.initial_balance) * Decimal("100")

        # Exposure by symbol
        exposure_by_symbol = {}
        for position in open_positions:
            if total_value > 0:
                exposure_pct = (position.notional_value / total_value) * Decimal("100")
                exposure_by_symbol[position.symbol] = exposure_pct

        # Concentration risk (Herfindahl Index)
        concentration_risk = Decimal("0")
        if exposure_by_symbol:
            concentration_risk = sum(
                (exposure / Decimal("100")) ** 2
                for exposure in exposure_by_symbol.values()
            ) * Decimal("100")

        return PortfolioMetrics(
            total_value=total_value,
            total_pnl=total_pnl,
            total_unrealized_pnl=total_unrealized,
            total_realized_pnl=total_realized,
            total_fees=total_fees,
            open_positions_count=len(open_positions),
            winning_positions=winning_positions,
            losing_positions=losing_positions,
            largest_winner=largest_winner,
            largest_loser=largest_loser,
            portfolio_return=portfolio_return,
            exposure_by_symbol=exposure_by_symbol,
            concentration_risk=concentration_risk,
        )

    def get_symbols_with_positions(self) -> Set[str]:
        """Get set of all symbols with positions.

        Returns:
            Set of symbol strings
        """
        return set(self.positions.keys())

    def get_position_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position summary for a specific symbol.

        Args:
            symbol: Symbol to get summary for

        Returns:
            Position summary dict or None if position not found
        """
        position = self.get_position(symbol)
        return position.get_position_summary() if position else None

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary.

        Returns:
            Dictionary containing portfolio state and metrics
        """
        metrics = self.get_portfolio_metrics()
        open_positions = self.open_positions
        closed_positions = self.closed_positions

        return {
            "portfolio_state": {
                "total_value": float(metrics.total_value),
                "total_pnl": float(metrics.total_pnl),
                "unrealized_pnl": float(metrics.total_unrealized_pnl),
                "realized_pnl": float(metrics.total_realized_pnl),
                "portfolio_return": float(metrics.portfolio_return),
                "total_fees": float(metrics.total_fees),
                "health_status": self.health_status.value,
                "base_currency": self.base_currency,
                "last_sync": self.last_sync_timestamp,
            },
            "positions": {
                "total_count": len(self.positions),
                "open_count": len(open_positions),
                "closed_count": len(closed_positions),
                "winning_count": metrics.winning_positions,
                "losing_count": metrics.losing_positions,
                "open_positions": {
                    symbol: pos.get_position_summary()
                    for symbol, pos in open_positions.items()
                },
            },
            "balances": {
                asset: {
                    "free": float(balance.free),
                    "locked": float(balance.locked),
                    "total": float(balance.total),
                    "usd_value": float(balance.usd_value),
                    "utilization": float(balance.utilization_percentage),
                }
                for asset, balance in self.balances.items()
            },
            "risk_metrics": {
                "concentration_risk": float(metrics.concentration_risk),
                "exposure_by_symbol": {
                    symbol: float(exposure)
                    for symbol, exposure in metrics.exposure_by_symbol.items()
                },
                "largest_winner": float(metrics.largest_winner),
                "largest_loser": float(metrics.largest_loser),
            },
            "timestamps": {
                "created": self.created_timestamp,
                "updated": self.updated_timestamp,
                "last_sync": self.last_sync_timestamp,
            },
        }

    def _update_health_status(self) -> None:
        """Update portfolio health status based on current metrics."""
        try:
            metrics = self.get_portfolio_metrics()

            # Simple health assessment logic
            if metrics.total_value <= 0:
                self.health_status = PortfolioHealthStatus.EMERGENCY
            elif metrics.portfolio_return < -20:  # More than 20% loss
                self.health_status = PortfolioHealthStatus.CRITICAL
            elif metrics.portfolio_return < -10:  # More than 10% loss
                self.health_status = PortfolioHealthStatus.WARNING
            elif metrics.concentration_risk > 50:  # High concentration risk
                self.health_status = PortfolioHealthStatus.WARNING
            else:
                self.health_status = PortfolioHealthStatus.HEALTHY

        except Exception:
            # If we can't calculate metrics, assume warning status
            self.health_status = PortfolioHealthStatus.WARNING

    def validate_portfolio_state(self) -> List[str]:
        """Validate portfolio state integrity.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate positions
        for symbol, position in self.positions.items():
            position_errors = position.validate_position()
            if position_errors:
                errors.extend(
                    [f"Position {symbol}: {error}" for error in position_errors]
                )

        # Validate balances
        for asset, balance in self.balances.items():
            try:
                balance.__post_init__()
            except PortfolioStateError as e:
                errors.append(f"Balance {asset}: {e}")

        # Check consistency
        if self.initial_balance < 0:
            errors.append("Initial balance cannot be negative")

        if self.total_deposits < 0:
            errors.append("Total deposits cannot be negative")

        if self.total_withdrawals < 0:
            errors.append("Total withdrawals cannot be negative")

        return errors

    def reset_realized_pnl(self) -> None:
        """Reset realized P&L tracking (e.g., for new accounting period)."""
        for position in self.positions.values():
            if position.status == PositionStatus.CLOSED:
                position.exit_levels.clear()
                position.status = PositionStatus.OPEN

        self.updated_timestamp = int(time.time() * 1000)

    def __str__(self) -> str:
        """String representation of portfolio state."""
        metrics = self.get_portfolio_metrics()
        return (
            f"PortfolioState(value={metrics.total_value:.2f} "
            f"pnl={metrics.total_pnl:.2f} positions={len(self.positions)} "
            f"health={self.health_status.value})"
        )
