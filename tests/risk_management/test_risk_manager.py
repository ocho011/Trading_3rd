"""
Unit tests for RiskManager orchestrator.

Tests the complete workflow from trading signal processing to order request generation.
"""

import asyncio
import time
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from trading_bot.core.config_manager import ConfigManager, EnvConfigLoader
from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.market_data.data_processor import MarketData
from trading_bot.risk_management.account_risk_evaluator import AccountState, PositionInfo
from trading_bot.risk_management.risk_manager import (
    OrderRequest,
    OrderType,
    RiskManager,
    RiskManagerConfig,
    create_risk_manager,
)
from trading_bot.strategies.base_strategy import (
    SignalStrength,
    SignalType,
    TradingSignal,
)


class TestRiskManagerConfig(unittest.TestCase):
    """Test RiskManagerConfig validation and functionality."""

    def test_valid_config_creation(self):
        """Test creating valid configuration."""
        config = RiskManagerConfig(
            max_position_risk_percentage=2.0,
            min_confidence_threshold=0.7,
            require_stop_loss=True,
        )
        self.assertEqual(config.max_position_risk_percentage, 2.0)
        self.assertEqual(config.min_confidence_threshold, 0.7)
        self.assertTrue(config.require_stop_loss)

    def test_invalid_config_validation(self):
        """Test configuration validation with invalid values."""
        # Test invalid risk percentage
        with self.assertRaises(Exception):
            RiskManagerConfig(max_position_risk_percentage=0.05)  # Too low

        with self.assertRaises(Exception):
            RiskManagerConfig(max_position_risk_percentage=60.0)  # Too high

        # Test invalid confidence threshold
        with self.assertRaises(Exception):
            RiskManagerConfig(min_confidence_threshold=1.5)  # Too high

        # Test invalid risk-reward ratio
        with self.assertRaises(Exception):
            RiskManagerConfig(min_risk_reward_ratio=0.2)  # Too low


class TestOrderRequest(unittest.TestCase):
    """Test OrderRequest data structure and methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.signal = TradingSignal(
            symbol="BTCUSD",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            price=45000.0,
            timestamp=int(time.time() * 1000),
            strategy_name="test_strategy",
            confidence=0.8,
        )

    def test_valid_order_request_creation(self):
        """Test creating valid order request."""
        order_request = OrderRequest(
            signal=self.signal,
            symbol="BTCUSD",
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=45000.0,
            position_size_result=None,
            risk_assessment_result=None,
            stop_loss_result=None,
            account_risk_result=None,
            entry_price=45000.0,
            stop_loss_price=43000.0,
            take_profit_price=48000.0,
            confidence_score=0.8,
        )

        self.assertEqual(order_request.symbol, "BTCUSD")
        self.assertEqual(order_request.quantity, Decimal("0.1"))
        self.assertTrue(order_request.has_stop_loss())
        self.assertTrue(order_request.has_take_profit())

    def test_order_request_validation(self):
        """Test order request validation."""
        # Test invalid quantity
        with self.assertRaises(Exception):
            OrderRequest(
                signal=self.signal,
                symbol="BTCUSD",
                order_type=OrderType.MARKET,
                quantity=Decimal("0"),  # Invalid
                price=45000.0,
                position_size_result=None,
                risk_assessment_result=None,
                stop_loss_result=None,
                account_risk_result=None,
                entry_price=45000.0,
            )

        # Test invalid confidence score
        with self.assertRaises(Exception):
            OrderRequest(
                signal=self.signal,
                symbol="BTCUSD",
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
                price=45000.0,
                position_size_result=None,
                risk_assessment_result=None,
                stop_loss_result=None,
                account_risk_result=None,
                entry_price=45000.0,
                confidence_score=1.5,  # Invalid
            )

    def test_risk_reward_ratio_calculation(self):
        """Test risk-reward ratio calculation."""
        order_request = OrderRequest(
            signal=self.signal,
            symbol="BTCUSD",
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=45000.0,
            position_size_result=None,
            risk_assessment_result=None,
            stop_loss_result=None,
            account_risk_result=None,
            entry_price=45000.0,
            stop_loss_price=43000.0,  # Risk: 2000
            take_profit_price=49000.0,  # Reward: 4000
        )

        rr_ratio = order_request.calculate_risk_reward_ratio()
        self.assertAlmostEqual(rr_ratio, 2.0, places=1)

    def test_order_summary(self):
        """Test order summary generation."""
        order_request = OrderRequest(
            signal=self.signal,
            symbol="BTCUSD",
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=45000.0,
            position_size_result=None,
            risk_assessment_result=None,
            stop_loss_result=None,
            account_risk_result=None,
            entry_price=45000.0,
            total_risk_amount=Decimal("500"),
            risk_percentage=1.0,
        )

        summary = order_request.get_order_summary()
        self.assertEqual(summary["symbol"], "BTCUSD")
        self.assertEqual(summary["quantity"], 0.1)
        self.assertEqual(summary["risk_amount"], 500.0)
        self.assertEqual(summary["risk_percentage"], 1.0)


class TestRiskManager(unittest.TestCase):
    """Test RiskManager orchestrator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.event_hub = EventHub()
        self.config_manager = ConfigManager(EnvConfigLoader())
        self.config = RiskManagerConfig()

        # Create mock components
        self.mock_position_sizer = MagicMock()
        self.mock_risk_assessor = MagicMock()
        self.mock_stop_loss_calculator = MagicMock()
        self.mock_account_risk_evaluator = MagicMock()

        self.risk_manager = RiskManager(
            config=self.config,
            event_hub=self.event_hub,
            config_manager=self.config_manager,
            position_sizer=self.mock_position_sizer,
            risk_assessor=self.mock_risk_assessor,
            stop_loss_calculator=self.mock_stop_loss_calculator,
            account_risk_evaluator=self.mock_account_risk_evaluator,
        )

        # Create test signal
        self.signal = TradingSignal(
            symbol="BTCUSD",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            price=45000.0,
            timestamp=int(time.time() * 1000),
            strategy_name="test_strategy",
            confidence=0.8,
        )

        # Create test market data
        self.market_data = MarketData(
            symbol="BTCUSD",
            timestamp=int(time.time() * 1000),
            price=45000.0,
            volume=1000000,
            source="test",
            data_type="tick",
            metadata={"atr": 1200.0, "volatility": 0.03},
        )

        # Create test account state
        self.account_state = AccountState(
            account_id="test_account",
            total_equity=Decimal("50000"),
            available_cash=Decimal("30000"),
            used_margin=Decimal("15000"),
            available_margin=Decimal("20000"),
            total_portfolio_value=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl_today=Decimal("0"),
            positions={},
        )

    def test_risk_manager_initialization(self):
        """Test RiskManager initialization."""
        self.assertIsInstance(self.risk_manager, RiskManager)
        self.assertEqual(self.risk_manager._config, self.config)
        self.assertEqual(self.risk_manager._event_hub, self.event_hub)

    def test_get_risk_statistics(self):
        """Test risk statistics retrieval."""
        stats = self.risk_manager.get_risk_statistics()
        self.assertIn("processed_signals", stats)
        self.assertIn("generated_orders", stats)
        self.assertIn("rejected_signals", stats)
        self.assertIn("success_rate", stats)
        self.assertIn("components_status", stats)

    def test_config_update(self):
        """Test configuration update."""
        new_config = RiskManagerConfig(max_position_risk_percentage=3.0)
        self.risk_manager.update_config(new_config)
        self.assertEqual(self.risk_manager._config.max_position_risk_percentage, 3.0)

    @patch("trading_bot.risk_management.risk_manager.time")
    async def test_signal_validation_age_check(self, mock_time):
        """Test signal age validation."""
        # Set current time to make signal appear old
        mock_time.time.return_value = time.time() + 600  # 10 minutes later

        old_signal = TradingSignal(
            symbol="BTCUSD",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            price=45000.0,
            timestamp=int(time.time() * 1000),  # Old timestamp
            strategy_name="test_strategy",
            confidence=0.8,
        )

        result = await self.risk_manager._validate_signal_inputs(
            old_signal, self.market_data, self.account_state
        )
        self.assertFalse(result["valid"])
        self.assertIn("too old", result["reason"])

    async def test_signal_validation_confidence_check(self):
        """Test signal confidence validation."""
        low_confidence_signal = TradingSignal(
            symbol="BTCUSD",
            signal_type=SignalType.BUY,
            strength=SignalStrength.WEAK,
            price=45000.0,
            timestamp=int(time.time() * 1000),
            strategy_name="test_strategy",
            confidence=0.3,  # Below threshold
        )

        result = await self.risk_manager._validate_signal_inputs(
            low_confidence_signal, self.market_data, self.account_state
        )
        self.assertFalse(result["valid"])
        self.assertIn("confidence too low", result["reason"])

    async def test_signal_validation_hold_signal(self):
        """Test HOLD signal rejection."""
        hold_signal = TradingSignal(
            symbol="BTCUSD",
            signal_type=SignalType.HOLD,
            strength=SignalStrength.MODERATE,
            price=45000.0,
            timestamp=int(time.time() * 1000),
            strategy_name="test_strategy",
            confidence=0.8,
        )

        result = await self.risk_manager._validate_signal_inputs(
            hold_signal, self.market_data, self.account_state
        )
        self.assertFalse(result["valid"])
        self.assertIn("HOLD signals", result["reason"])

    async def test_process_trading_signal_success(self):
        """Test successful signal processing."""
        # Configure mock components to return successful results
        from trading_bot.risk_management.position_sizer import PositionSizingResult, PositionSizingMethod
        from trading_bot.risk_management.risk_assessor import RiskAssessmentResult, RiskLevel
        from trading_bot.risk_management.stop_loss_calculator import StopLossResult, StopLossLevel, PositionType

        # Mock position sizing result
        mock_position_result = PositionSizingResult(
            position_size=0.1,
            risk_amount=500.0,
            entry_price=45000.0,
            stop_loss_price=43000.0,
            method_used=PositionSizingMethod.FIXED_PERCENTAGE,
            account_balance=50000.0,
            risk_percentage=1.0,
            calculation_timestamp=int(time.time() * 1000),
        )
        self.mock_position_sizer.calculate_position_size.return_value = mock_position_result

        # Mock risk assessment result
        mock_risk_result = MagicMock()
        mock_risk_result.overall_risk_level = RiskLevel.MODERATE
        mock_risk_result.overall_risk_multiplier = 1.0
        mock_risk_result.position_size_adjustment = 1.0
        mock_risk_result.confidence = 0.8
        mock_risk_result.warnings = []
        mock_risk_result.recommendations = []
        self.mock_risk_assessor.assess_risk.return_value = mock_risk_result

        # Mock stop-loss result
        mock_stop_result = MagicMock()
        mock_stop_result.stop_loss_level = MagicMock()
        mock_stop_result.stop_loss_level.price = 43000.0
        mock_stop_result.take_profit_level = MagicMock()
        mock_stop_result.take_profit_level.price = 48000.0
        mock_stop_result.warnings = []
        mock_stop_result.recommendations = []
        self.mock_stop_loss_calculator.calculate_levels.return_value = mock_stop_result

        # Mock account risk result
        mock_account_result = MagicMock()
        mock_account_result.can_add_position = True
        mock_account_result.max_new_position_value = Decimal("5000")
        self.mock_account_risk_evaluator.evaluate_new_position.return_value = mock_account_result

        # Process signal
        order_request = await self.risk_manager.process_trading_signal(
            self.signal, self.market_data, self.account_state
        )

        # Verify successful processing
        self.assertIsNotNone(order_request)
        self.assertIsInstance(order_request, OrderRequest)
        self.assertEqual(order_request.symbol, "BTCUSD")
        self.assertGreater(order_request.quantity, 0)
        self.assertEqual(order_request.stop_loss_price, 43000.0)
        self.assertEqual(order_request.take_profit_price, 48000.0)

    async def test_process_trading_signal_rejection(self):
        """Test signal rejection due to low confidence."""
        low_confidence_signal = TradingSignal(
            symbol="BTCUSD",
            signal_type=SignalType.BUY,
            strength=SignalStrength.WEAK,
            price=45000.0,
            timestamp=int(time.time() * 1000),
            strategy_name="test_strategy",
            confidence=0.4,  # Below threshold
        )

        order_request = await self.risk_manager.process_trading_signal(
            low_confidence_signal, self.market_data, self.account_state
        )

        self.assertIsNone(order_request)
        self.assertEqual(self.risk_manager._rejected_signals_count, 1)

    async def test_event_publishing(self):
        """Test ORDER_REQUEST_GENERATED event publishing."""
        event_data = None

        def capture_event(data):
            nonlocal event_data
            event_data = data

        # Subscribe to events
        self.event_hub.subscribe(EventType.ORDER_REQUEST_GENERATED, capture_event)

        # Create simple order request
        order_request = OrderRequest(
            signal=self.signal,
            symbol="BTCUSD",
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=45000.0,
            position_size_result=None,
            risk_assessment_result=None,
            stop_loss_result=None,
            account_risk_result=None,
            entry_price=45000.0,
        )

        # Publish event
        await self.risk_manager._publish_order_request_event(order_request)

        # Verify event was published
        self.assertIsNotNone(event_data)
        self.assertEqual(event_data["symbol"], "BTCUSD")
        self.assertEqual(event_data["order_type"], "market")


class TestRiskManagerFactory(unittest.TestCase):
    """Test RiskManager factory function."""

    def test_create_risk_manager_success(self):
        """Test successful risk manager creation with factory."""
        event_hub = EventHub()
        config_manager = ConfigManager(EnvConfigLoader())

        risk_manager = create_risk_manager(
            event_hub=event_hub,
            config_manager=config_manager,
            account_balance=25000.0,
            max_position_risk=1.5,
            min_confidence=0.75,
        )

        self.assertIsInstance(risk_manager, RiskManager)
        self.assertEqual(risk_manager._config.max_position_risk_percentage, 1.5)
        self.assertEqual(risk_manager._config.min_confidence_threshold, 0.75)

    def test_create_risk_manager_with_options(self):
        """Test risk manager creation with additional options."""
        event_hub = EventHub()
        config_manager = ConfigManager(EnvConfigLoader())

        risk_manager = create_risk_manager(
            event_hub=event_hub,
            config_manager=config_manager,
            require_stop_loss=False,
            require_take_profit=True,
            min_risk_reward_ratio=3.0,
            max_daily_trades=25,
        )

        self.assertIsInstance(risk_manager, RiskManager)
        self.assertFalse(risk_manager._config.require_stop_loss)
        self.assertTrue(risk_manager._config.require_take_profit)
        self.assertEqual(risk_manager._config.min_risk_reward_ratio, 3.0)
        self.assertEqual(risk_manager._config.max_daily_trades, 25)


def run_async_test(coro):
    """Helper to run async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


if __name__ == "__main__":
    # Custom test runner for async tests
    import sys

    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        sys.exit(1)