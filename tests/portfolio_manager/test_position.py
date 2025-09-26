"""
Unit tests for Position class and related components.

Tests position tracking, P&L calculations, validation, and state management.
"""

import time
from decimal import Decimal

import pytest

from trading_bot.portfolio_manager.position import (
    InvalidPositionError,
    Position,
    PositionCalculationError,
    PositionLevel,
    PositionSide,
    PositionStatus,
)


class TestPositionLevel:
    """Test cases for PositionLevel class."""

    def test_position_level_creation_valid(self):
        """Test valid position level creation."""
        level = PositionLevel(
            price=Decimal("100.0"),
            quantity=Decimal("1.5"),
            timestamp=int(time.time() * 1000),
            order_id="test_order_123",
            fees=Decimal("0.15"),
        )

        assert level.price == Decimal("100.0")
        assert level.quantity == Decimal("1.5")
        assert level.order_id == "test_order_123"
        assert level.fees == Decimal("0.15")
        assert level.notional_value == Decimal("150.0")
        assert level.net_value == Decimal("149.85")

    def test_position_level_invalid_quantity(self):
        """Test position level creation with invalid quantity."""
        with pytest.raises(InvalidPositionError, match="quantity must be positive"):
            PositionLevel(
                price=Decimal("100.0"),
                quantity=Decimal("0"),
                timestamp=int(time.time() * 1000),
            )

    def test_position_level_invalid_price(self):
        """Test position level creation with invalid price."""
        with pytest.raises(InvalidPositionError, match="price must be positive"):
            PositionLevel(
                price=Decimal("0"),
                quantity=Decimal("1.0"),
                timestamp=int(time.time() * 1000),
            )

    def test_position_level_negative_fees(self):
        """Test position level creation with negative fees."""
        with pytest.raises(InvalidPositionError, match="fees cannot be negative"):
            PositionLevel(
                price=Decimal("100.0"),
                quantity=Decimal("1.0"),
                timestamp=int(time.time() * 1000),
                fees=Decimal("-1.0"),
            )


class TestPosition:
    """Test cases for Position class."""

    @pytest.fixture
    def sample_position(self):
        """Create sample position for testing."""
        return Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            commission_rate=Decimal("0.001"),
        )

    def test_position_creation_valid(self, sample_position):
        """Test valid position creation."""
        assert sample_position.symbol == "BTCUSDT"
        assert sample_position.side == PositionSide.LONG
        assert sample_position.commission_rate == Decimal("0.001")
        assert sample_position.status == PositionStatus.OPEN
        assert len(sample_position.entry_levels) == 0
        assert len(sample_position.exit_levels) == 0

    def test_position_creation_invalid_symbol(self):
        """Test position creation with invalid symbol."""
        with pytest.raises(InvalidPositionError, match="Symbol cannot be empty"):
            Position(symbol="", side=PositionSide.LONG)

    def test_position_creation_invalid_commission_rate(self):
        """Test position creation with negative commission rate."""
        with pytest.raises(
            InvalidPositionError, match="Commission rate cannot be negative"
        ):
            Position(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                commission_rate=Decimal("-0.001"),
            )

    def test_add_entry_valid(self, sample_position):
        """Test adding valid entry to position."""
        sample_position.add_entry(
            price=Decimal("50000.0"),
            quantity=Decimal("0.1"),
            order_id="order_1",
            fees=Decimal("5.0"),
        )

        assert len(sample_position.entry_levels) == 1
        assert sample_position.total_entry_quantity == Decimal("0.1")
        assert sample_position.average_entry_price == Decimal("50000.0")
        assert sample_position.open_quantity == Decimal("0.1")
        assert sample_position.status == PositionStatus.OPEN

    def test_add_multiple_entries(self, sample_position):
        """Test adding multiple entries to position."""
        sample_position.add_entry(Decimal("50000.0"), Decimal("0.1"))
        sample_position.add_entry(Decimal("51000.0"), Decimal("0.1"))

        assert len(sample_position.entry_levels) == 2
        assert sample_position.total_entry_quantity == Decimal("0.2")
        assert sample_position.average_entry_price == Decimal(
            "50500.0"
        )  # (50000*0.1 + 51000*0.1) / 0.2

    def test_add_exit_valid(self, sample_position):
        """Test adding valid exit to position."""
        sample_position.add_entry(Decimal("50000.0"), Decimal("0.2"))
        sample_position.add_exit(
            price=Decimal("52000.0"),
            quantity=Decimal("0.1"),
            order_id="exit_1",
            fees=Decimal("5.2"),
        )

        assert len(sample_position.exit_levels) == 1
        assert sample_position.total_exit_quantity == Decimal("0.1")
        assert sample_position.open_quantity == Decimal("0.1")
        assert sample_position.status == PositionStatus.PARTIALLY_CLOSED

    def test_add_exit_exceeds_quantity(self, sample_position):
        """Test adding exit that exceeds open quantity."""
        sample_position.add_entry(Decimal("50000.0"), Decimal("0.1"))

        with pytest.raises(InvalidPositionError, match="exceeds open quantity"):
            sample_position.add_exit(Decimal("52000.0"), Decimal("0.2"))

    def test_fully_close_position(self, sample_position):
        """Test fully closing a position."""
        sample_position.add_entry(Decimal("50000.0"), Decimal("0.1"))
        sample_position.add_exit(Decimal("52000.0"), Decimal("0.1"))

        assert sample_position.total_exit_quantity == Decimal("0.1")
        assert sample_position.open_quantity == Decimal("0")
        assert sample_position.status == PositionStatus.CLOSED

    def test_update_current_price_valid(self, sample_position):
        """Test updating current price."""
        sample_position.update_current_price(Decimal("55000.0"))
        assert sample_position.current_price == Decimal("55000.0")

    def test_update_current_price_invalid(self, sample_position):
        """Test updating current price with invalid value."""
        with pytest.raises(
            InvalidPositionError, match="Current price must be positive"
        ):
            sample_position.update_current_price(Decimal("0"))

    def test_unrealized_pnl_long_profit(self, sample_position):
        """Test unrealized P&L calculation for long position with profit."""
        sample_position.add_entry(Decimal("50000.0"), Decimal("0.1"))
        sample_position.update_current_price(Decimal("55000.0"))

        # P&L = (55000 - 50000) * 0.1 = 500
        # Estimated exit fees = 55000 * 0.1 * 0.001 = 5.5
        # Net P&L = 500 - 5.5 = 494.5
        expected_pnl = Decimal("494.5")
        assert sample_position.unrealized_pnl == expected_pnl

    def test_unrealized_pnl_long_loss(self, sample_position):
        """Test unrealized P&L calculation for long position with loss."""
        sample_position.add_entry(Decimal("50000.0"), Decimal("0.1"))
        sample_position.update_current_price(Decimal("45000.0"))

        # P&L = (45000 - 50000) * 0.1 = -500
        # Estimated exit fees = 45000 * 0.1 * 0.001 = 4.5
        # Net P&L = -500 - 4.5 = -504.5
        expected_pnl = Decimal("-504.5")
        assert sample_position.unrealized_pnl == expected_pnl

    def test_unrealized_pnl_short_profit(self):
        """Test unrealized P&L calculation for short position with profit."""
        position = Position(symbol="BTCUSDT", side=PositionSide.SHORT)
        position.add_entry(Decimal("50000.0"), Decimal("0.1"))
        position.update_current_price(Decimal("45000.0"))

        # P&L = (50000 - 45000) * 0.1 = 500
        # Estimated exit fees = 45000 * 0.1 * 0.001 = 4.5
        # Net P&L = 500 - 4.5 = 495.5
        expected_pnl = Decimal("495.5")
        assert position.unrealized_pnl == expected_pnl

    def test_realized_pnl_long_profit(self, sample_position):
        """Test realized P&L calculation for long position with profit."""
        sample_position.add_entry(
            Decimal("50000.0"), Decimal("0.2"), fees=Decimal("10.0")
        )
        sample_position.add_exit(
            Decimal("55000.0"), Decimal("0.1"), fees=Decimal("5.5")
        )

        # Price P&L = (55000 - 50000) * 0.1 = 500
        # Exit fees = 5.5
        # Proportional entry fees = (0.1 / 0.2) * 10.0 = 5.0
        # Net P&L = 500 - 5.5 - 5.0 = 489.5
        expected_pnl = Decimal("489.5")
        assert sample_position.realized_pnl == expected_pnl

    def test_total_pnl_calculation(self, sample_position):
        """Test total P&L calculation (realized + unrealized)."""
        sample_position.add_entry(Decimal("50000.0"), Decimal("0.2"))
        sample_position.add_exit(Decimal("55000.0"), Decimal("0.1"))
        sample_position.update_current_price(Decimal("60000.0"))

        total_pnl = sample_position.total_pnl
        expected_total = sample_position.realized_pnl + sample_position.unrealized_pnl
        assert total_pnl == expected_total

    def test_return_percentage_calculation(self, sample_position):
        """Test return percentage calculation."""
        sample_position.add_entry(Decimal("50000.0"), Decimal("0.1"))  # Notional: 5000
        sample_position.update_current_price(Decimal("55000.0"))

        # Unrealized P&L should be about 494.5
        # Return % = (494.5 / 5000) * 100 = 9.89%
        return_pct = sample_position.return_percentage
        assert abs(return_pct - Decimal("9.89")) < Decimal("0.01")

    def test_notional_value_calculation(self, sample_position):
        """Test notional value calculation."""
        sample_position.add_entry(Decimal("50000.0"), Decimal("0.1"))
        sample_position.add_exit(Decimal("55000.0"), Decimal("0.05"))
        sample_position.update_current_price(Decimal("60000.0"))

        # Open quantity = 0.1 - 0.05 = 0.05
        # Notional = 0.05 * 60000 = 3000
        assert sample_position.notional_value == Decimal("3000.0")

    def test_average_exit_price_calculation(self, sample_position):
        """Test average exit price calculation."""
        sample_position.add_entry(Decimal("50000.0"), Decimal("0.2"))
        sample_position.add_exit(Decimal("55000.0"), Decimal("0.1"))
        sample_position.add_exit(Decimal("57000.0"), Decimal("0.05"))

        # Average exit = (55000*0.1 + 57000*0.05) / 0.15 = 8350 / 0.15 = 55666.67
        expected_avg = Decimal("8350.0") / Decimal("0.15")
        assert abs(sample_position.average_exit_price - expected_avg) < Decimal("0.01")

    def test_position_validation_valid(self, sample_position):
        """Test position validation for valid position."""
        sample_position.add_entry(Decimal("50000.0"), Decimal("0.1"))

        errors = sample_position.validate_position()
        assert len(errors) == 0

    def test_position_validation_invalid(self):
        """Test position validation for invalid position."""
        position = Position(symbol="BTCUSDT", side=PositionSide.LONG)
        # Manually create invalid state
        position.entry_levels.append(
            PositionLevel(Decimal("100"), Decimal("0.1"), int(time.time() * 1000))
        )
        position.exit_levels.append(
            PositionLevel(Decimal("110"), Decimal("0.2"), int(time.time() * 1000))
        )

        errors = position.validate_position()
        assert any("exceeds entry quantity" in error for error in errors)

    def test_position_summary(self, sample_position):
        """Test position summary generation."""
        sample_position.add_entry(Decimal("50000.0"), Decimal("0.1"))
        sample_position.update_current_price(Decimal("55000.0"))

        summary = sample_position.get_position_summary()

        assert summary["symbol"] == "BTCUSDT"
        assert summary["side"] == "long"
        assert summary["status"] == "open"
        assert summary["open_quantity"] == 0.1
        assert summary["average_entry_price"] == 50000.0
        assert summary["current_price"] == 55000.0
        assert summary["notional_value"] == 5500.0

    def test_position_string_representation(self, sample_position):
        """Test position string representation."""
        sample_position.add_entry(Decimal("50000.0"), Decimal("0.1"))

        str_repr = str(sample_position)
        assert "BTCUSDT" in str_repr
        assert "long" in str_repr
        assert "0.1" in str_repr
        assert "50000.0" in str_repr

    def test_position_closed_no_new_entries(self, sample_position):
        """Test that closed position cannot accept new entries."""
        sample_position.add_entry(Decimal("50000.0"), Decimal("0.1"))
        sample_position.add_exit(Decimal("55000.0"), Decimal("0.1"))

        with pytest.raises(
            InvalidPositionError, match="Cannot add entry to closed position"
        ):
            sample_position.add_entry(Decimal("60000.0"), Decimal("0.1"))

    def test_no_entry_levels_calculation_error(self, sample_position):
        """Test calculation error when no entry levels exist."""
        with pytest.raises(PositionCalculationError, match="No entry levels"):
            sample_position.average_entry_price

    def test_zero_current_price_unrealized_pnl(self, sample_position):
        """Test unrealized P&L with zero current price."""
        sample_position.add_entry(Decimal("50000.0"), Decimal("0.1"))
        # current_price defaults to 0

        assert sample_position.unrealized_pnl == Decimal("0")
        assert sample_position.notional_value == Decimal("0")

    def test_position_fees_tracking(self, sample_position):
        """Test position fees tracking."""
        sample_position.add_entry(
            Decimal("50000.0"), Decimal("0.1"), fees=Decimal("10.0")
        )
        sample_position.add_entry(
            Decimal("51000.0"), Decimal("0.1"), fees=Decimal("11.0")
        )
        sample_position.add_exit(
            Decimal("55000.0"), Decimal("0.05"), fees=Decimal("5.5")
        )

        assert sample_position.total_entry_fees == Decimal("21.0")
        assert sample_position.total_exit_fees == Decimal("5.5")
        assert sample_position.total_fees == Decimal("26.5")
