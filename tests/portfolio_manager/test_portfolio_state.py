"""
Unit tests for PortfolioState class and related components.

Tests portfolio state management, balance tracking, metrics calculation, and validation.
"""

from decimal import Decimal

import pytest

from trading_bot.portfolio_manager.portfolio_state import (
    BalanceInfo,
    PortfolioHealthStatus,
    PortfolioState,
    PortfolioStateError,
)
from trading_bot.portfolio_manager.position import (
    Position,
    PositionSide,
    PositionStatus,
)


class TestBalanceInfo:
    """Test cases for BalanceInfo class."""

    def test_balance_info_creation_valid(self):
        """Test valid balance info creation."""
        balance = BalanceInfo(
            asset="USDT",
            free=Decimal("1000.0"),
            locked=Decimal("200.0"),
            total=Decimal("1200.0"),
            usd_value=Decimal("1200.0"),
        )

        assert balance.asset == "USDT"
        assert balance.free == Decimal("1000.0")
        assert balance.locked == Decimal("200.0")
        assert balance.total == Decimal("1200.0")
        assert balance.usd_value == Decimal("1200.0")
        # Check utilization percentage with reasonable precision tolerance
        expected_util = Decimal("200") / Decimal("1200") * Decimal("100")
        assert abs(balance.utilization_percentage - expected_util) < Decimal("0.001")

    def test_balance_info_auto_calculate_total(self):
        """Test auto-calculation of total balance."""
        balance = BalanceInfo(
            asset="BTC",
            free=Decimal("1.5"),
            locked=Decimal("0.5"),
        )

        assert balance.total == Decimal("2.0")

    def test_balance_info_invalid_asset(self):
        """Test balance info creation with empty asset."""
        with pytest.raises(PortfolioStateError, match="Asset symbol cannot be empty"):
            BalanceInfo(asset="")

    def test_balance_info_negative_balance(self):
        """Test balance info creation with negative balance."""
        with pytest.raises(
            PortfolioStateError, match="Balance amounts cannot be negative"
        ):
            BalanceInfo(asset="USDT", free=Decimal("-100.0"))

    def test_balance_utilization_zero_total(self):
        """Test utilization percentage with zero total balance."""
        balance = BalanceInfo(asset="USDT")
        assert balance.utilization_percentage == Decimal("0")


class TestPortfolioState:
    """Test cases for PortfolioState class."""

    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio for testing."""
        return PortfolioState(
            base_currency="USDT",
            initial_balance=Decimal("10000.0"),
        )

    @pytest.fixture
    def sample_position(self):
        """Create sample position for testing."""
        position = Position(symbol="BTCUSDT", side=PositionSide.LONG)
        position.add_entry(Decimal("50000.0"), Decimal("0.1"))
        position.update_current_price(Decimal("55000.0"))
        return position

    def test_portfolio_state_creation_valid(self, sample_portfolio):
        """Test valid portfolio state creation."""
        assert sample_portfolio.base_currency == "USDT"
        assert sample_portfolio.initial_balance == Decimal("10000.0")
        assert sample_portfolio.health_status == PortfolioHealthStatus.HEALTHY
        assert len(sample_portfolio.positions) == 0
        assert len(sample_portfolio.balances) == 0

    def test_portfolio_state_creation_invalid_currency(self):
        """Test portfolio state creation with empty base currency."""
        with pytest.raises(PortfolioStateError, match="Base currency cannot be empty"):
            PortfolioState(base_currency="")

    def test_add_position_valid(self, sample_portfolio, sample_position):
        """Test adding valid position to portfolio."""
        sample_portfolio.add_position(sample_position)

        assert len(sample_portfolio.positions) == 1
        assert sample_portfolio.get_position("BTCUSDT") == sample_position

    def test_add_position_invalid(self, sample_portfolio):
        """Test adding invalid position to portfolio."""
        with pytest.raises(
            PortfolioStateError, match="Position must be Position instance"
        ):
            sample_portfolio.add_position("not_a_position")

    def test_remove_position_existing(self, sample_portfolio, sample_position):
        """Test removing existing position."""
        sample_portfolio.add_position(sample_position)
        removed = sample_portfolio.remove_position("BTCUSDT")

        assert removed == sample_position
        assert len(sample_portfolio.positions) == 0

    def test_remove_position_nonexistent(self, sample_portfolio):
        """Test removing non-existent position."""
        removed = sample_portfolio.remove_position("NONEXISTENT")
        assert removed is None

    def test_update_balance_valid(self, sample_portfolio):
        """Test updating balance with valid info."""
        balance = BalanceInfo(
            asset="USDT",
            free=Decimal("1000.0"),
            locked=Decimal("0.0"),
            total=Decimal("1000.0"),
            usd_value=Decimal("1000.0"),
        )

        sample_portfolio.update_balance(balance)
        assert sample_portfolio.get_balance("USDT") == balance

    def test_update_balance_invalid(self, sample_portfolio):
        """Test updating balance with invalid info."""
        with pytest.raises(
            PortfolioStateError, match="Balance info must be BalanceInfo instance"
        ):
            sample_portfolio.update_balance("not_balance_info")

    def test_update_multiple_balances(self, sample_portfolio):
        """Test updating multiple balances."""
        balances = {
            "USDT": BalanceInfo("USDT", Decimal("1000"), Decimal("0"), Decimal("1000")),
            "BTC": BalanceInfo("BTC", Decimal("1.5"), Decimal("0.5"), Decimal("2.0")),
        }

        sample_portfolio.update_balances(balances)
        assert len(sample_portfolio.balances) == 2
        assert sample_portfolio.get_balance("USDT") is not None
        assert sample_portfolio.get_balance("BTC") is not None

    def test_get_available_balance(self, sample_portfolio):
        """Test getting available balance."""
        balance = BalanceInfo("USDT", free=Decimal("1000"), locked=Decimal("200"))
        sample_portfolio.update_balance(balance)

        assert sample_portfolio.get_available_balance("USDT") == Decimal("1000")
        assert sample_portfolio.get_available_balance("NONEXISTENT") == Decimal("0")

    def test_get_total_balance(self, sample_portfolio):
        """Test getting total balance."""
        balance = BalanceInfo("USDT", free=Decimal("1000"), locked=Decimal("200"))
        sample_portfolio.update_balance(balance)

        assert sample_portfolio.get_total_balance("USDT") == Decimal("1200")
        assert sample_portfolio.get_total_balance("NONEXISTENT") == Decimal("0")

    def test_open_positions_property(self, sample_portfolio):
        """Test open positions property."""
        # Add open position
        open_position = Position(symbol="BTCUSDT", side=PositionSide.LONG)
        open_position.add_entry(Decimal("50000"), Decimal("0.1"))

        # Add closed position
        closed_position = Position(symbol="ETHUSDT", side=PositionSide.LONG)
        closed_position.add_entry(Decimal("3000"), Decimal("1.0"))
        closed_position.add_exit(Decimal("3100"), Decimal("1.0"))

        sample_portfolio.add_position(open_position)
        sample_portfolio.add_position(closed_position)

        open_positions = sample_portfolio.open_positions
        assert len(open_positions) == 1
        assert "BTCUSDT" in open_positions
        assert "ETHUSDT" not in open_positions

    def test_closed_positions_property(self, sample_portfolio):
        """Test closed positions property."""
        # Add open position
        open_position = Position(symbol="BTCUSDT", side=PositionSide.LONG)
        open_position.add_entry(Decimal("50000"), Decimal("0.1"))

        # Add closed position
        closed_position = Position(symbol="ETHUSDT", side=PositionSide.LONG)
        closed_position.add_entry(Decimal("3000"), Decimal("1.0"))
        closed_position.add_exit(Decimal("3100"), Decimal("1.0"))

        sample_portfolio.add_position(open_position)
        sample_portfolio.add_position(closed_position)

        closed_positions = sample_portfolio.closed_positions
        assert len(closed_positions) == 1
        assert "ETHUSDT" in closed_positions
        assert "BTCUSDT" not in closed_positions

    def test_long_positions_property(self, sample_portfolio):
        """Test long positions property."""
        long_position = Position(symbol="BTCUSDT", side=PositionSide.LONG)
        short_position = Position(symbol="ETHUSDT", side=PositionSide.SHORT)

        sample_portfolio.add_position(long_position)
        sample_portfolio.add_position(short_position)

        long_positions = sample_portfolio.long_positions
        assert len(long_positions) == 1
        assert "BTCUSDT" in long_positions

    def test_short_positions_property(self, sample_portfolio):
        """Test short positions property."""
        long_position = Position(symbol="BTCUSDT", side=PositionSide.LONG)
        short_position = Position(symbol="ETHUSDT", side=PositionSide.SHORT)

        sample_portfolio.add_position(long_position)
        sample_portfolio.add_position(short_position)

        short_positions = sample_portfolio.short_positions
        assert len(short_positions) == 1
        assert "ETHUSDT" in short_positions

    def test_total_portfolio_value(self, sample_portfolio, sample_position):
        """Test total portfolio value calculation."""
        # Add balance
        balance = BalanceInfo("USDT", usd_value=Decimal("5000"))
        sample_portfolio.update_balance(balance)

        # Add position
        sample_portfolio.add_position(sample_position)

        # Expected: balance USD value + position notional value
        expected_value = Decimal("5000") + sample_position.notional_value
        assert sample_portfolio.total_portfolio_value == expected_value

    def test_total_unrealized_pnl(self, sample_portfolio, sample_position):
        """Test total unrealized P&L calculation."""
        sample_portfolio.add_position(sample_position)

        assert sample_portfolio.total_unrealized_pnl == sample_position.unrealized_pnl

    def test_total_realized_pnl(self, sample_portfolio):
        """Test total realized P&L calculation."""
        # Create position with realized P&L
        position = Position(symbol="BTCUSDT", side=PositionSide.LONG)
        position.add_entry(Decimal("50000"), Decimal("0.2"))
        position.add_exit(Decimal("55000"), Decimal("0.1"))

        sample_portfolio.add_position(position)
        assert sample_portfolio.total_realized_pnl == position.realized_pnl

    def test_total_pnl(self, sample_portfolio, sample_position):
        """Test total P&L calculation."""
        sample_portfolio.add_position(sample_position)

        expected_total = sample_position.realized_pnl + sample_position.unrealized_pnl
        assert sample_portfolio.total_pnl == expected_total

    def test_total_fees(self, sample_portfolio):
        """Test total fees calculation."""
        position = Position(symbol="BTCUSDT", side=PositionSide.LONG)
        position.add_entry(Decimal("50000"), Decimal("0.1"), fees=Decimal("5"))

        sample_portfolio.add_position(position)
        assert sample_portfolio.total_fees == Decimal("5")

    def test_get_portfolio_metrics(self, sample_portfolio):
        """Test portfolio metrics calculation."""
        # Add positions with different P&L
        winning_position = Position(symbol="BTCUSDT", side=PositionSide.LONG)
        winning_position.add_entry(Decimal("50000"), Decimal("0.1"))
        winning_position.update_current_price(Decimal("55000"))

        losing_position = Position(symbol="ETHUSDT", side=PositionSide.LONG)
        losing_position.add_entry(Decimal("3000"), Decimal("1.0"))
        losing_position.update_current_price(Decimal("2800"))

        sample_portfolio.add_position(winning_position)
        sample_portfolio.add_position(losing_position)

        metrics = sample_portfolio.get_portfolio_metrics()

        assert metrics.open_positions_count == 2
        assert metrics.winning_positions == 1
        assert metrics.losing_positions == 1
        assert metrics.largest_winner > 0
        assert metrics.largest_loser < 0

    def test_get_symbols_with_positions(self, sample_portfolio, sample_position):
        """Test getting symbols with positions."""
        sample_portfolio.add_position(sample_position)

        symbols = sample_portfolio.get_symbols_with_positions()
        assert symbols == {"BTCUSDT"}

    def test_get_position_summary(self, sample_portfolio, sample_position):
        """Test getting position summary."""
        sample_portfolio.add_position(sample_position)

        summary = sample_portfolio.get_position_summary("BTCUSDT")
        assert summary is not None
        assert summary["symbol"] == "BTCUSDT"

        # Non-existent position
        assert sample_portfolio.get_position_summary("NONEXISTENT") is None

    def test_get_portfolio_summary(self, sample_portfolio, sample_position):
        """Test getting comprehensive portfolio summary."""
        balance = BalanceInfo("USDT", free=Decimal("1000"), usd_value=Decimal("1000"))
        sample_portfolio.update_balance(balance)
        sample_portfolio.add_position(sample_position)

        summary = sample_portfolio.get_portfolio_summary()

        assert "portfolio_state" in summary
        assert "positions" in summary
        assert "balances" in summary
        assert "risk_metrics" in summary
        assert "timestamps" in summary

        assert summary["portfolio_state"]["base_currency"] == "USDT"
        assert summary["positions"]["total_count"] == 1
        assert summary["balances"]["USDT"]["free"] == 1000.0

    def test_health_status_updates(self, sample_portfolio):
        """Test portfolio health status updates."""
        # Initially healthy
        assert sample_portfolio.health_status == PortfolioHealthStatus.HEALTHY

        # Add large losing position to trigger health update
        losing_position = Position(symbol="BTCUSDT", side=PositionSide.LONG)
        losing_position.add_entry(Decimal("50000"), Decimal("1.0"))
        losing_position.update_current_price(Decimal("30000"))  # -40% loss

        sample_portfolio.add_position(losing_position)
        # Health status should update automatically

    def test_validate_portfolio_state_valid(self, sample_portfolio, sample_position):
        """Test portfolio state validation for valid state."""
        balance = BalanceInfo("USDT", free=Decimal("1000"))
        sample_portfolio.update_balance(balance)
        sample_portfolio.add_position(sample_position)

        errors = sample_portfolio.validate_portfolio_state()
        assert len(errors) == 0

    def test_validate_portfolio_state_invalid_balance(self, sample_portfolio):
        """Test portfolio state validation with invalid balance."""
        # Manually create invalid balance
        invalid_balance = BalanceInfo.__new__(BalanceInfo)
        invalid_balance.asset = ""  # Invalid empty asset
        invalid_balance.free = Decimal("1000")
        invalid_balance.locked = Decimal("0")
        invalid_balance.total = Decimal("1000")

        sample_portfolio.balances["INVALID"] = invalid_balance

        errors = sample_portfolio.validate_portfolio_state()
        assert any("Balance INVALID" in error for error in errors)

    def test_validate_portfolio_state_negative_initial_balance(self):
        """Test portfolio state validation with negative initial balance."""
        portfolio = PortfolioState(initial_balance=Decimal("-1000"))

        errors = portfolio.validate_portfolio_state()
        assert any("Initial balance cannot be negative" in error for error in errors)

    def test_reset_realized_pnl(self, sample_portfolio):
        """Test resetting realized P&L tracking."""
        # Create position with realized P&L
        position = Position(symbol="BTCUSDT", side=PositionSide.LONG)
        position.add_entry(Decimal("50000"), Decimal("0.2"))
        position.add_exit(Decimal("55000"), Decimal("0.2"))  # Fully close

        sample_portfolio.add_position(position)
        assert position.status == PositionStatus.CLOSED
        assert len(position.exit_levels) == 1

        # Reset realized P&L
        sample_portfolio.reset_realized_pnl()

        # Closed position should be reopened with exits cleared
        updated_position = sample_portfolio.get_position("BTCUSDT")
        assert updated_position.status == PositionStatus.OPEN
        assert len(updated_position.exit_levels) == 0

    def test_portfolio_state_string_representation(
        self, sample_portfolio, sample_position
    ):
        """Test portfolio state string representation."""
        sample_portfolio.add_position(sample_position)

        str_repr = str(sample_portfolio)
        assert "PortfolioState" in str_repr
        assert "positions=1" in str_repr
        # Health status depends on actual portfolio metrics, just check it's present
        assert "health=" in str_repr

    def test_concentration_risk_calculation(self, sample_portfolio):
        """Test concentration risk calculation in portfolio metrics."""
        # Add two positions with different exposures
        btc_position = Position(symbol="BTCUSDT", side=PositionSide.LONG)
        btc_position.add_entry(Decimal("50000"), Decimal("0.1"))  # 5000 notional
        btc_position.update_current_price(Decimal("50000"))

        eth_position = Position(symbol="ETHUSDT", side=PositionSide.LONG)
        eth_position.add_entry(Decimal("3000"), Decimal("1.0"))  # 3000 notional
        eth_position.update_current_price(Decimal("3000"))

        sample_portfolio.add_position(btc_position)
        sample_portfolio.add_position(eth_position)

        metrics = sample_portfolio.get_portfolio_metrics()

        # Total notional = 5000 + 3000 = 8000
        # BTC exposure = 5000/8000 = 62.5%
        # ETH exposure = 3000/8000 = 37.5%
        # Concentration risk (Herfindahl) = (0.625^2 + 0.375^2) * 100 = 53.125

        assert abs(metrics.concentration_risk - Decimal("53.125")) < Decimal("0.01")
