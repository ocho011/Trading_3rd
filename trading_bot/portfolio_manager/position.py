"""
Position tracking module for portfolio management.

This module provides comprehensive position tracking with accurate P&L calculations,
risk metrics, and validation following SOLID principles and dependency
injection patterns.
"""

import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional


class PositionError(Exception):
    """Base exception for position-related errors."""


class InvalidPositionError(PositionError):
    """Exception raised when position data is invalid."""


class PositionCalculationError(PositionError):
    """Exception raised when position calculations fail."""


class PositionSide(Enum):
    """Position side enumeration."""

    LONG = "long"
    SHORT = "short"


class PositionStatus(Enum):
    """Position status enumeration."""

    OPEN = "open"
    CLOSED = "closed"
    PARTIALLY_CLOSED = "partially_closed"


@dataclass
class PositionLevel:
    """Represents a specific price level within a position.

    Used for tracking partial entries and exits at different price levels.
    """

    price: Decimal
    quantity: Decimal
    timestamp: int
    order_id: Optional[str] = None
    fees: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        """Validate position level after initialization."""
        if self.quantity <= 0:
            raise InvalidPositionError("Position level quantity must be positive")
        if self.price <= 0:
            raise InvalidPositionError("Position level price must be positive")
        if self.fees < 0:
            raise InvalidPositionError("Position level fees cannot be negative")

    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value of this position level."""
        return self.price * self.quantity

    @property
    def net_value(self) -> Decimal:
        """Calculate net value after fees."""
        return self.notional_value - self.fees


@dataclass
class Position:
    """Comprehensive position tracking with P&L calculations and risk metrics.

    This class maintains detailed position state including entry/exit levels,
    realized/unrealized P&L, and risk metrics while adhering to SOLID principles.

    Attributes:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        side: Position side (LONG or SHORT)
        entry_levels: List of entry price levels
        exit_levels: List of exit price levels
        current_price: Current market price for P&L calculations
        commission_rate: Commission rate as decimal (e.g., 0.001 for 0.1%)
        status: Position status
        metadata: Additional position metadata
        created_timestamp: Position creation timestamp
        updated_timestamp: Last update timestamp
    """

    symbol: str
    side: PositionSide
    entry_levels: List[PositionLevel] = field(default_factory=list)
    exit_levels: List[PositionLevel] = field(default_factory=list)
    current_price: Decimal = Decimal("0")
    commission_rate: Decimal = Decimal("0.001")
    status: PositionStatus = PositionStatus.OPEN
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        """Validate position after initialization."""
        if not self.symbol:
            raise InvalidPositionError("Symbol cannot be empty")
        if not isinstance(self.side, PositionSide):
            raise InvalidPositionError("Side must be PositionSide enum")
        if self.commission_rate < 0:
            raise InvalidPositionError("Commission rate cannot be negative")

        # Update timestamp on initialization
        self.updated_timestamp = int(time.time() * 1000)

    def add_entry(
        self,
        price: Decimal,
        quantity: Decimal,
        order_id: Optional[str] = None,
        fees: Decimal = Decimal("0"),
    ) -> None:
        """Add entry level to position.

        Args:
            price: Entry price
            quantity: Entry quantity
            order_id: Optional order ID for tracking
            fees: Trading fees for this entry

        Raises:
            InvalidPositionError: If position is closed or parameters invalid
        """
        if self.status == PositionStatus.CLOSED:
            raise InvalidPositionError("Cannot add entry to closed position")

        entry_level = PositionLevel(
            price=price,
            quantity=quantity,
            timestamp=int(time.time() * 1000),
            order_id=order_id,
            fees=fees,
        )

        self.entry_levels.append(entry_level)
        self.updated_timestamp = int(time.time() * 1000)

        # Update status
        if self.status == PositionStatus.PARTIALLY_CLOSED:
            # Keep partially closed status if we still have exits
            pass
        else:
            self.status = PositionStatus.OPEN

    def add_exit(
        self,
        price: Decimal,
        quantity: Decimal,
        order_id: Optional[str] = None,
        fees: Decimal = Decimal("0"),
    ) -> None:
        """Add exit level to position.

        Args:
            price: Exit price
            quantity: Exit quantity
            order_id: Optional order ID for tracking
            fees: Trading fees for this exit

        Raises:
            InvalidPositionError: If exit quantity exceeds open quantity
        """
        if quantity > self.open_quantity:
            raise InvalidPositionError(
                f"Exit quantity {quantity} exceeds open quantity {self.open_quantity}"
            )

        exit_level = PositionLevel(
            price=price,
            quantity=quantity,
            timestamp=int(time.time() * 1000),
            order_id=order_id,
            fees=fees,
        )

        self.exit_levels.append(exit_level)
        self.updated_timestamp = int(time.time() * 1000)

        # Update status
        self._update_position_status()

    def update_current_price(self, price: Decimal) -> None:
        """Update current market price for P&L calculations.

        Args:
            price: Current market price

        Raises:
            InvalidPositionError: If price is invalid
        """
        if price <= 0:
            raise InvalidPositionError("Current price must be positive")

        self.current_price = price
        self.updated_timestamp = int(time.time() * 1000)

    @property
    def total_entry_quantity(self) -> Decimal:
        """Calculate total entry quantity."""
        return sum(level.quantity for level in self.entry_levels)

    @property
    def total_exit_quantity(self) -> Decimal:
        """Calculate total exit quantity."""
        return sum(level.quantity for level in self.exit_levels)

    @property
    def open_quantity(self) -> Decimal:
        """Calculate current open quantity."""
        return self.total_entry_quantity - self.total_exit_quantity

    @property
    def average_entry_price(self) -> Decimal:
        """Calculate quantity-weighted average entry price.

        Returns:
            Decimal: Average entry price

        Raises:
            PositionCalculationError: If no entry levels exist
        """
        if not self.entry_levels:
            raise PositionCalculationError("No entry levels to calculate average")

        total_notional = sum(level.notional_value for level in self.entry_levels)
        total_quantity = sum(level.quantity for level in self.entry_levels)

        if total_quantity == 0:
            raise PositionCalculationError("Total entry quantity is zero")

        return total_notional / total_quantity

    @property
    def average_exit_price(self) -> Optional[Decimal]:
        """Calculate quantity-weighted average exit price.

        Returns:
            Decimal: Average exit price or None if no exits
        """
        if not self.exit_levels:
            return None

        total_notional = sum(level.notional_value for level in self.exit_levels)
        total_quantity = sum(level.quantity for level in self.exit_levels)

        if total_quantity == 0:
            return None

        return total_notional / total_quantity

    @property
    def notional_value(self) -> Decimal:
        """Calculate current notional value of open position."""
        if self.current_price <= 0:
            return Decimal("0")

        return self.open_quantity * self.current_price

    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate unrealized P&L for open quantity.

        Returns:
            Decimal: Unrealized P&L including fees
        """
        if self.open_quantity <= 0 or self.current_price <= 0:
            return Decimal("0")

        try:
            avg_entry = self.average_entry_price

            if self.side == PositionSide.LONG:
                price_pnl = (self.current_price - avg_entry) * self.open_quantity
            else:  # SHORT
                price_pnl = (avg_entry - self.current_price) * self.open_quantity

            # Calculate estimated fees for closing the position
            estimated_exit_fees = (
                self.current_price * self.open_quantity * self.commission_rate
            )

            return price_pnl - estimated_exit_fees

        except PositionCalculationError:
            return Decimal("0")

    @property
    def realized_pnl(self) -> Decimal:
        """Calculate realized P&L from closed portions.

        Returns:
            Decimal: Realized P&L including all fees
        """
        if not self.exit_levels:
            return Decimal("0")

        realized_pnl = Decimal("0")

        try:
            avg_entry = self.average_entry_price

            for exit_level in self.exit_levels:
                if self.side == PositionSide.LONG:
                    price_pnl = (exit_level.price - avg_entry) * exit_level.quantity
                else:  # SHORT
                    price_pnl = (avg_entry - exit_level.price) * exit_level.quantity

                # Subtract fees
                total_fees = exit_level.fees
                # Add proportional entry fees
                entry_fees = (
                    exit_level.quantity / self.total_entry_quantity
                ) * self.total_entry_fees

                realized_pnl += price_pnl - total_fees - entry_fees

        except PositionCalculationError:
            pass

        return realized_pnl

    @property
    def total_pnl(self) -> Decimal:
        """Calculate total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def total_entry_fees(self) -> Decimal:
        """Calculate total entry fees."""
        return sum(level.fees for level in self.entry_levels)

    @property
    def total_exit_fees(self) -> Decimal:
        """Calculate total exit fees."""
        return sum(level.fees for level in self.exit_levels)

    @property
    def total_fees(self) -> Decimal:
        """Calculate total fees paid."""
        return self.total_entry_fees + self.total_exit_fees

    @property
    def return_percentage(self) -> Decimal:
        """Calculate position return as percentage.

        Returns:
            Decimal: Return percentage (e.g., 5.25 for 5.25%)
        """
        try:
            if self.total_entry_quantity == 0:
                return Decimal("0")

            total_entry_value = sum(level.notional_value for level in self.entry_levels)
            if total_entry_value == 0:
                return Decimal("0")

            return (self.total_pnl / total_entry_value) * Decimal("100")

        except (ZeroDivisionError, PositionCalculationError):
            return Decimal("0")

    def _update_position_status(self) -> None:
        """Update position status based on entry/exit levels."""
        if not self.entry_levels:
            self.status = PositionStatus.OPEN
        elif self.total_exit_quantity == 0:
            self.status = PositionStatus.OPEN
        elif self.total_exit_quantity >= self.total_entry_quantity:
            self.status = PositionStatus.CLOSED
        else:
            self.status = PositionStatus.PARTIALLY_CLOSED

    def get_position_summary(self) -> Dict[str, Any]:
        """Get comprehensive position summary for reporting.

        Returns:
            Dict containing position metrics and state
        """
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "status": self.status.value,
            "open_quantity": float(self.open_quantity),
            "total_entry_quantity": float(self.total_entry_quantity),
            "total_exit_quantity": float(self.total_exit_quantity),
            "average_entry_price": (
                float(self.average_entry_price) if self.entry_levels else 0.0
            ),
            "average_exit_price": (
                float(self.average_exit_price) if self.average_exit_price else None
            ),
            "current_price": float(self.current_price),
            "notional_value": float(self.notional_value),
            "unrealized_pnl": float(self.unrealized_pnl),
            "realized_pnl": float(self.realized_pnl),
            "total_pnl": float(self.total_pnl),
            "return_percentage": float(self.return_percentage),
            "total_fees": float(self.total_fees),
            "entry_levels_count": len(self.entry_levels),
            "exit_levels_count": len(self.exit_levels),
            "created_timestamp": self.created_timestamp,
            "updated_timestamp": self.updated_timestamp,
        }

    def validate_position(self) -> List[str]:
        """Validate position integrity and return any issues.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check basic integrity
        if not self.symbol:
            errors.append("Symbol is empty")

        if self.total_exit_quantity > self.total_entry_quantity:
            errors.append("Total exit quantity exceeds entry quantity")

        # Check level integrity
        for i, level in enumerate(self.entry_levels):
            try:
                level.__post_init__()
            except InvalidPositionError as e:
                errors.append(f"Entry level {i} invalid: {e}")

        for i, level in enumerate(self.exit_levels):
            try:
                level.__post_init__()
            except InvalidPositionError as e:
                errors.append(f"Exit level {i} invalid: {e}")

        # Check status consistency
        expected_status = self._calculate_expected_status()
        if self.status != expected_status:
            errors.append(
                f"Status mismatch: {self.status.value} vs expected "
                f"{expected_status.value}"
            )

        return errors

    def _calculate_expected_status(self) -> PositionStatus:
        """Calculate expected status based on entry/exit levels."""
        if not self.entry_levels:
            return PositionStatus.OPEN
        elif self.total_exit_quantity == 0:
            return PositionStatus.OPEN
        elif self.total_exit_quantity >= self.total_entry_quantity:
            return PositionStatus.CLOSED
        else:
            return PositionStatus.PARTIALLY_CLOSED

    def __str__(self) -> str:
        """String representation of position."""
        return (
            f"Position({self.symbol} {self.side.value} "
            f"qty={self.open_quantity} @ "
            f"{self.average_entry_price if self.entry_levels else 'N/A'} "
            f"P&L={self.total_pnl:.2f} {self.status.value})"
        )

    def __repr__(self) -> str:
        """Detailed representation of position."""
        return (
            f"Position(symbol='{self.symbol}', side={self.side}, "
            f"entry_levels={len(self.entry_levels)}, "
            f"exit_levels={len(self.exit_levels)}, "
            f"status={self.status}, open_qty={self.open_quantity})"
        )
