"""
Unit tests for AccountRiskEvaluator module.

Tests comprehensive account balance-based risk assessment system
including position evaluation, portfolio health checks, and margin validation.
"""

import time
import unittest
from decimal import Decimal
from unittest.mock import Mock, patch

from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.risk_management.account_risk_evaluator import (
    AccountRiskEvaluator,
    AccountRiskConfig,
    AccountRiskResult,
    AccountState,
    PositionInfo,
    RiskProfile,
    AccountRiskLevel,
    AccountRiskError,
    InsufficientMarginError,
    ExcessiveRiskError,
    InvalidAccountStateError,
    create_account_risk_evaluator,
)
from trading_bot.strategies.base_strategy import TradingSignal, SignalType, SignalStrength


class TestPositionInfo(unittest.TestCase):
    """Test cases for PositionInfo dataclass."""

    def test_position_info_creation(self) -> None:
        """Test creating valid PositionInfo instance."""
        position = PositionInfo(
            symbol="AAPL",
            quantity=Decimal('100'),
            average_price=Decimal('150.00'),
            current_price=Decimal('155.00'),
            market_value=Decimal('15500.00'),
            unrealized_pnl=Decimal('500.00'),
            position_type="long",
            entry_timestamp=int(time.time() * 1000),
        )

        self.assertEqual(position.symbol, "AAPL")
        self.assertEqual(position.quantity, Decimal('100'))
        self.assertEqual(position.position_type, "long")

    def test_position_info_validation(self) -> None:
        """Test PositionInfo validation."""
        # Test invalid quantity
        with self.assertRaises(InvalidAccountStateError):
            PositionInfo(
                symbol="AAPL",
                quantity=Decimal('-100'),  # Invalid negative quantity
                average_price=Decimal('150.00'),
                current_price=Decimal('155.00'),
                market_value=Decimal('15500.00'),
                unrealized_pnl=Decimal('500.00'),
                position_type="long",
                entry_timestamp=int(time.time() * 1000),
            )

        # Test invalid position type
        with self.assertRaises(InvalidAccountStateError):
            PositionInfo(
                symbol="AAPL",
                quantity=Decimal('100'),
                average_price=Decimal('150.00'),
                current_price=Decimal('155.00'),
                market_value=Decimal('15500.00'),
                unrealized_pnl=Decimal('500.00'),
                position_type="invalid",  # Invalid position type
                entry_timestamp=int(time.time() * 1000),
            )

    def test_position_risk_amount(self) -> None:
        """Test position risk amount calculation."""
        # Test long position with stop loss
        position = PositionInfo(
            symbol="AAPL",
            quantity=Decimal('100'),
            average_price=Decimal('150.00'),
            current_price=Decimal('155.00'),
            market_value=Decimal('15500.00'),
            unrealized_pnl=Decimal('500.00'),
            position_type="long",
            entry_timestamp=int(time.time() * 1000),
            stop_loss_price=Decimal('145.00'),
        )

        expected_risk = (Decimal('150.00') - Decimal('145.00')) * Decimal('100')
        self.assertEqual(position.risk_amount, expected_risk)

        # Test position without stop loss (default 2% risk)
        position_no_stop = PositionInfo(
            symbol="AAPL",
            quantity=Decimal('100'),
            average_price=Decimal('150.00'),
            current_price=Decimal('155.00'),
            market_value=Decimal('15500.00'),
            unrealized_pnl=Decimal('500.00'),
            position_type="long",
            entry_timestamp=int(time.time() * 1000),
        )

        expected_default_risk = abs(Decimal('15500.00')) * Decimal('0.02')
        self.assertEqual(position_no_stop.risk_amount, expected_default_risk)


class TestAccountState(unittest.TestCase):
    """Test cases for AccountState dataclass."""

    def create_sample_account_state(self) -> AccountState:
        """Create a sample AccountState for testing."""
        positions = {
            "AAPL": PositionInfo(
                symbol="AAPL",
                quantity=Decimal('100'),
                average_price=Decimal('150.00'),
                current_price=Decimal('155.00'),
                market_value=Decimal('15500.00'),
                unrealized_pnl=Decimal('500.00'),
                position_type="long",
                entry_timestamp=int(time.time() * 1000),
                correlation_group="tech",
            ),
            "GOOGL": PositionInfo(
                symbol="GOOGL",
                quantity=Decimal('50'),
                average_price=Decimal('2500.00'),
                current_price=Decimal('2600.00'),
                market_value=Decimal('130000.00'),
                unrealized_pnl=Decimal('5000.00'),
                position_type="long",
                entry_timestamp=int(time.time() * 1000),
                correlation_group="tech",
            ),
        }

        return AccountState(
            account_id="test_account",
            total_equity=Decimal('200000.00'),
            available_cash=Decimal('50000.00'),
            used_margin=Decimal('20000.00'),
            available_margin=Decimal('30000.00'),
            total_portfolio_value=Decimal('200000.00'),
            unrealized_pnl=Decimal('5500.00'),
            realized_pnl_today=Decimal('1000.00'),
            positions=positions,
            buying_power=Decimal('80000.00'),
        )

    def test_account_state_creation(self) -> None:
        """Test creating valid AccountState instance."""
        account_state = self.create_sample_account_state()

        self.assertEqual(account_state.account_id, "test_account")
        self.assertEqual(account_state.total_equity, Decimal('200000.00'))
        self.assertEqual(len(account_state.positions), 2)

    def test_leverage_ratio_calculation(self) -> None:
        """Test leverage ratio calculation."""
        account_state = self.create_sample_account_state()

        # Total exposure = 15500 + 130000 = 145500
        # Total equity = 200000
        # Leverage = 145500 / 200000 = 0.7275
        expected_leverage = 145500.0 / 200000.0
        self.assertAlmostEqual(account_state.leverage_ratio, expected_leverage, places=4)

    def test_margin_utilization(self) -> None:
        """Test margin utilization calculation."""
        account_state = self.create_sample_account_state()

        # Used margin = 20000, Available margin = 30000
        # Total margin = 50000, Utilization = 20000/50000 = 0.4
        expected_utilization = 20000.0 / 50000.0
        self.assertAlmostEqual(account_state.margin_utilization, expected_utilization, places=4)

    def test_portfolio_concentration(self) -> None:
        """Test portfolio concentration calculation."""
        account_state = self.create_sample_account_state()
        concentrations = account_state.portfolio_concentration

        # AAPL: 15500 / 200000 = 0.0775
        # GOOGL: 130000 / 200000 = 0.65
        self.assertAlmostEqual(concentrations["AAPL"], 0.0775, places=4)
        self.assertAlmostEqual(concentrations["GOOGL"], 0.65, places=4)

    def test_correlation_exposure(self) -> None:
        """Test correlation exposure calculation."""
        account_state = self.create_sample_account_state()
        correlations = account_state.correlation_exposure

        # Both positions are in "tech" group: (15500 + 130000) / 200000 = 0.7275
        self.assertAlmostEqual(correlations["tech"], 0.7275, places=4)


class TestAccountRiskConfig(unittest.TestCase):
    """Test cases for AccountRiskConfig dataclass."""

    def test_config_creation(self) -> None:
        """Test creating valid AccountRiskConfig instance."""
        config = AccountRiskConfig(
            max_portfolio_risk_pct=0.05,
            max_position_concentration=0.20,
            max_leverage_ratio=3.0,
        )

        self.assertEqual(config.max_portfolio_risk_pct, 0.05)
        self.assertEqual(config.max_position_concentration, 0.20)
        self.assertEqual(config.max_leverage_ratio, 3.0)

    def test_config_validation(self) -> None:
        """Test AccountRiskConfig validation."""
        # Test invalid portfolio risk
        with self.assertRaises(InvalidAccountStateError):
            AccountRiskConfig(max_portfolio_risk_pct=1.5)  # > 100%

        # Test invalid leverage
        with self.assertRaises(InvalidAccountStateError):
            AccountRiskConfig(max_leverage_ratio=0.5)  # <= 1.0


class TestAccountRiskEvaluator(unittest.TestCase):
    """Test cases for AccountRiskEvaluator class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = AccountRiskConfig(
            max_portfolio_risk_pct=0.05,
            max_position_concentration=0.20,
            max_leverage_ratio=3.0,
            max_margin_utilization=0.80,
        )
        self.event_hub = Mock(spec=EventHub)
        self.evaluator = AccountRiskEvaluator(self.config, self.event_hub)

        # Sample account state
        self.account_state = self._create_sample_account_state()

        # Sample trading signal
        self.signal = TradingSignal(
            symbol="TSLA",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=0.85,
            price=200.0,
            timestamp=int(time.time() * 1000),
            strategy_name="test_strategy",
            metadata={"volatility": 0.03},
        )

    def _create_sample_account_state(self) -> AccountState:
        """Create a sample AccountState for testing."""
        positions = {
            "AAPL": PositionInfo(
                symbol="AAPL",
                quantity=Decimal('100'),
                average_price=Decimal('150.00'),
                current_price=Decimal('155.00'),
                market_value=Decimal('15500.00'),
                unrealized_pnl=Decimal('500.00'),
                position_type="long",
                entry_timestamp=int(time.time() * 1000),
            ),
        }

        return AccountState(
            account_id="test_account",
            total_equity=Decimal('100000.00'),
            available_cash=Decimal('70000.00'),
            used_margin=Decimal('10000.00'),
            available_margin=Decimal('40000.00'),
            total_portfolio_value=Decimal('100000.00'),
            unrealized_pnl=Decimal('500.00'),
            realized_pnl_today=Decimal('200.00'),
            positions=positions,
            buying_power=Decimal('110000.00'),
        )

    def test_evaluator_initialization(self) -> None:
        """Test AccountRiskEvaluator initialization."""
        self.assertIsInstance(self.evaluator, AccountRiskEvaluator)
        self.assertEqual(self.evaluator._config, self.config)
        self.assertEqual(self.evaluator._event_hub, self.event_hub)

    def test_invalid_initialization(self) -> None:
        """Test invalid AccountRiskEvaluator initialization."""
        with self.assertRaises(InvalidAccountStateError):
            AccountRiskEvaluator("invalid_config")

    def test_get_max_position_size(self) -> None:
        """Test maximum position size calculation."""
        max_qty, max_value = self.evaluator.get_max_position_size(
            "TSLA", Decimal('200.00'), self.account_state
        )

        # Max value should be constrained by concentration limit
        # 20% of 100,000 = 20,000
        expected_max_value = Decimal('20000.00')
        expected_max_qty = expected_max_value / Decimal('200.00')

        self.assertEqual(max_value, expected_max_value)
        self.assertEqual(max_qty, expected_max_qty)

    def test_validate_margin_requirements_sufficient(self) -> None:
        """Test margin validation with sufficient margin."""
        result = self.evaluator.validate_margin_requirements(
            self.signal, Decimal('50'), self.account_state
        )

        self.assertTrue(result)

    def test_validate_margin_requirements_insufficient(self) -> None:
        """Test margin validation with insufficient margin."""
        # Create account state with low available margin
        low_margin_state = AccountState(
            account_id="test_account",
            total_equity=Decimal('100000.00'),
            available_cash=Decimal('5000.00'),
            used_margin=Decimal('45000.00'),
            available_margin=Decimal('1000.00'),  # Very low available margin
            total_portfolio_value=Decimal('100000.00'),
            unrealized_pnl=Decimal('500.00'),
            realized_pnl_today=Decimal('200.00'),
            positions={},
            buying_power=Decimal('6000.00'),
        )

        with self.assertRaises(InsufficientMarginError):
            self.evaluator.validate_margin_requirements(
                self.signal, Decimal('1000'), low_margin_state  # Large position
            )

    def test_evaluate_new_position_safe(self) -> None:
        """Test evaluating a new position under safe conditions."""
        # Use a smaller position size to avoid critical position size risk
        # 20 shares * $200 = $4000, which is 4% of $100k equity (reasonable)
        result = self.evaluator.evaluate_new_position(
            self.signal, self.account_state, Decimal('20')
        )

        self.assertIsInstance(result, AccountRiskResult)
        self.assertEqual(result.account_id, "test_account")
        self.assertTrue(result.can_add_position)
        self.assertLessEqual(result.overall_risk_score, 0.6)  # Should be reasonable risk

    def test_evaluate_new_position_high_concentration(self) -> None:
        """Test evaluating a position that would cause high concentration."""
        # Try to add a very large position
        large_quantity = Decimal('2000')  # $400,000 worth at $200/share

        result = self.evaluator.evaluate_new_position(
            self.signal, self.account_state, large_quantity
        )

        # Should have concentration warnings
        self.assertIn("concentration", str(result.warnings).lower())
        self.assertGreater(result.risk_factors.get("concentration_risk", 0), 0.5)

    def test_check_portfolio_health(self) -> None:
        """Test portfolio health assessment."""
        result = self.evaluator.check_portfolio_health(self.account_state)

        self.assertIsInstance(result, AccountRiskResult)
        self.assertEqual(result.account_id, "test_account")
        self.assertIn("health_check", result.metadata)
        self.assertTrue(result.metadata["health_check"])

    def test_leverage_risk_assessment(self) -> None:
        """Test leverage risk factor assessment."""
        # Create high leverage account state
        high_leverage_positions = {
            "AAPL": PositionInfo(
                symbol="AAPL",
                quantity=Decimal('1000'),
                average_price=Decimal('150.00'),
                current_price=Decimal('155.00'),
                market_value=Decimal('155000.00'),  # Large position
                unrealized_pnl=Decimal('5000.00'),
                position_type="long",
                entry_timestamp=int(time.time() * 1000),
            ),
            "GOOGL": PositionInfo(
                symbol="GOOGL",
                quantity=Decimal('100'),
                average_price=Decimal('2500.00'),
                current_price=Decimal('2600.00'),
                market_value=Decimal('260000.00'),  # Another large position
                unrealized_pnl=Decimal('10000.00'),
                position_type="long",
                entry_timestamp=int(time.time() * 1000),
            ),
        }

        high_leverage_state = AccountState(
            account_id="test_account",
            total_equity=Decimal('100000.00'),  # Small equity base
            available_cash=Decimal('10000.00'),
            used_margin=Decimal('80000.00'),
            available_margin=Decimal('10000.00'),
            total_portfolio_value=Decimal('100000.00'),
            unrealized_pnl=Decimal('15000.00'),
            realized_pnl_today=Decimal('1000.00'),
            positions=high_leverage_positions,
            buying_power=Decimal('20000.00'),
        )

        leverage_risk = self.evaluator._assess_leverage_risk(high_leverage_state)

        # Should detect high leverage (total exposure ~415k vs 100k equity = 4.15x)
        self.assertGreater(leverage_risk, 0.5)

    def test_concentration_risk_assessment(self) -> None:
        """Test concentration risk factor assessment."""
        # Create account with one highly concentrated position
        concentrated_position = {
            "AAPL": PositionInfo(
                symbol="AAPL",
                quantity=Decimal('500'),
                average_price=Decimal('150.00'),
                current_price=Decimal('155.00'),
                market_value=Decimal('77500.00'),  # 77.5% of portfolio
                unrealized_pnl=Decimal('2500.00'),
                position_type="long",
                entry_timestamp=int(time.time() * 1000),
            ),
        }

        concentrated_state = AccountState(
            account_id="test_account",
            total_equity=Decimal('100000.00'),
            available_cash=Decimal('22500.00'),
            used_margin=Decimal('0.00'),
            available_margin=Decimal('50000.00'),
            total_portfolio_value=Decimal('100000.00'),
            unrealized_pnl=Decimal('2500.00'),
            realized_pnl_today=Decimal('0.00'),
            positions=concentrated_position,
            buying_power=Decimal('72500.00'),
        )

        concentration_risk = self.evaluator._assess_concentration_risk(concentrated_state)

        # Should detect high concentration (77.5% vs 20% limit)
        self.assertGreater(concentration_risk, 0.8)

    def test_event_publishing(self) -> None:
        """Test event publishing during risk evaluation."""
        # Evaluate a risky position to trigger events
        large_quantity = Decimal('1000')

        result = self.evaluator.evaluate_new_position(
            self.signal, self.account_state, large_quantity
        )

        # Check that events were published
        self.event_hub.publish.assert_called()

    def test_update_config(self) -> None:
        """Test updating evaluator configuration."""
        new_config = AccountRiskConfig(
            max_portfolio_risk_pct=0.10,  # Higher risk tolerance
            max_position_concentration=0.25,
            max_leverage_ratio=4.0,
        )

        self.evaluator.update_config(new_config)
        self.assertEqual(self.evaluator._config, new_config)

    def test_statistics_tracking(self) -> None:
        """Test evaluation statistics tracking."""
        # Perform a few evaluations
        self.evaluator.evaluate_new_position(self.signal, self.account_state, Decimal('50'))
        self.evaluator.evaluate_new_position(self.signal, self.account_state, Decimal('75'))

        stats = self.evaluator.get_evaluation_statistics()

        self.assertEqual(stats["evaluations_count"], 2)
        self.assertIn("config_summary", stats)
        self.assertIn("recent_risk_levels", stats)

    def test_stale_data_detection(self) -> None:
        """Test detection of stale account data."""
        # Create account state with old timestamp
        old_timestamp = int((time.time() - 30 * 60) * 1000)  # 30 minutes ago
        stale_account_state = AccountState(
            account_id="test_account",
            total_equity=Decimal('100000.00'),
            available_cash=Decimal('70000.00'),
            used_margin=Decimal('10000.00'),
            available_margin=Decimal('40000.00'),
            total_portfolio_value=Decimal('100000.00'),
            unrealized_pnl=Decimal('500.00'),
            realized_pnl_today=Decimal('200.00'),
            positions={},
            last_updated=old_timestamp,
        )

        with self.assertRaises(AccountRiskError):
            self.evaluator.evaluate_new_position(self.signal, stale_account_state)


class TestCreateAccountRiskEvaluator(unittest.TestCase):
    """Test cases for create_account_risk_evaluator factory function."""

    def test_factory_function_default(self) -> None:
        """Test factory function with default parameters."""
        evaluator = create_account_risk_evaluator()

        self.assertIsInstance(evaluator, AccountRiskEvaluator)
        self.assertEqual(evaluator._config.max_leverage_ratio, RiskProfile.MODERATE.max_leverage)

    def test_factory_function_conservative(self) -> None:
        """Test factory function with conservative risk profile."""
        evaluator = create_account_risk_evaluator(risk_profile=RiskProfile.CONSERVATIVE)

        self.assertEqual(evaluator._config.max_leverage_ratio, RiskProfile.CONSERVATIVE.max_leverage)
        self.assertEqual(
            evaluator._config.max_position_concentration,
            RiskProfile.CONSERVATIVE.max_position_size
        )

    def test_factory_function_aggressive(self) -> None:
        """Test factory function with aggressive risk profile."""
        evaluator = create_account_risk_evaluator(risk_profile=RiskProfile.AGGRESSIVE)

        self.assertEqual(evaluator._config.max_leverage_ratio, RiskProfile.AGGRESSIVE.max_leverage)
        self.assertEqual(
            evaluator._config.max_position_concentration,
            RiskProfile.AGGRESSIVE.max_position_size
        )

    def test_factory_function_custom_params(self) -> None:
        """Test factory function with custom parameters."""
        evaluator = create_account_risk_evaluator(
            max_leverage=2.5,
            max_position_concentration=0.15,
            enable_emergency_stops=False,
        )

        self.assertEqual(evaluator._config.max_leverage_ratio, 2.5)
        self.assertEqual(evaluator._config.max_position_concentration, 0.15)
        self.assertFalse(evaluator._config.enable_emergency_stops)

    def test_factory_function_with_event_hub(self) -> None:
        """Test factory function with event hub."""
        event_hub = Mock(spec=EventHub)
        evaluator = create_account_risk_evaluator(event_hub=event_hub)

        self.assertEqual(evaluator._event_hub, event_hub)


class TestRiskProfile(unittest.TestCase):
    """Test cases for RiskProfile enum."""

    def test_risk_profile_properties(self) -> None:
        """Test risk profile properties."""
        # Conservative profile
        conservative = RiskProfile.CONSERVATIVE
        self.assertEqual(conservative.max_portfolio_risk, 0.02)
        self.assertEqual(conservative.max_position_size, 0.10)
        self.assertEqual(conservative.max_leverage, 2.0)

        # Moderate profile
        moderate = RiskProfile.MODERATE
        self.assertEqual(moderate.max_portfolio_risk, 0.05)
        self.assertEqual(moderate.max_position_size, 0.20)
        self.assertEqual(moderate.max_leverage, 3.0)

        # Aggressive profile
        aggressive = RiskProfile.AGGRESSIVE
        self.assertEqual(aggressive.max_portfolio_risk, 0.10)
        self.assertEqual(aggressive.max_position_size, 0.30)
        self.assertEqual(aggressive.max_leverage, 5.0)


class TestAccountRiskLevel(unittest.TestCase):
    """Test cases for AccountRiskLevel enum."""

    def test_risk_level_thresholds(self) -> None:
        """Test risk level threshold properties."""
        self.assertEqual(AccountRiskLevel.SAFE.threshold, 0.20)
        self.assertEqual(AccountRiskLevel.CAUTION.threshold, 0.50)
        self.assertEqual(AccountRiskLevel.WARNING.threshold, 0.75)
        self.assertEqual(AccountRiskLevel.DANGER.threshold, 0.90)
        self.assertEqual(AccountRiskLevel.CRITICAL.threshold, 1.0)


class TestAccountRiskResult(unittest.TestCase):
    """Test cases for AccountRiskResult dataclass."""

    def create_sample_result(self) -> AccountRiskResult:
        """Create sample AccountRiskResult for testing."""
        return AccountRiskResult(
            account_id="test_account",
            risk_level=AccountRiskLevel.SAFE,
            overall_risk_score=0.25,
            can_add_position=True,
            max_new_position_value=Decimal('20000.00'),
            max_new_position_quantity=Decimal('100'),
            margin_requirement=Decimal('10000.00'),
            available_buying_power=Decimal('50000.00'),
            risk_factors={"leverage_risk": 0.2, "concentration_risk": 0.3},
            concentration_analysis={"AAPL": 0.15, "GOOGL": 0.20},
            correlation_analysis={"tech": 0.35},
            warnings=["Test warning"],
            recommendations=["Test recommendation"],
        )

    def test_result_creation(self) -> None:
        """Test creating valid AccountRiskResult."""
        result = self.create_sample_result()

        self.assertEqual(result.account_id, "test_account")
        self.assertEqual(result.risk_level, AccountRiskLevel.SAFE)
        self.assertTrue(result.can_add_position)

    def test_critical_risks_detection(self) -> None:
        """Test critical risks detection."""
        result = self.create_sample_result()
        result.risk_factors["leverage_risk"] = 0.95  # Critical level

        self.assertTrue(result.has_critical_risks())

    def test_immediate_action_required(self) -> None:
        """Test immediate action requirement detection."""
        result = self.create_sample_result()
        result.risk_level = AccountRiskLevel.CRITICAL
        result.emergency_actions = ["Stop trading immediately"]

        self.assertTrue(result.requires_immediate_action())

    def test_risk_summary(self) -> None:
        """Test risk summary generation."""
        result = self.create_sample_result()
        summary = result.get_risk_summary()

        self.assertEqual(summary["risk_level"], "safe")
        self.assertEqual(summary["overall_score"], 0.25)
        self.assertTrue(summary["can_trade"])
        self.assertFalse(summary["critical_risks"])


if __name__ == "__main__":
    unittest.main()