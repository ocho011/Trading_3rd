"""
Unit tests for ICT Strategy implementation.

Tests the ICTStrategy class functionality including configuration validation,
signal generation, order block analysis, and performance metrics.
"""

import time

import pytest

from trading_bot.core.event_hub import EventHub
from trading_bot.market_data.data_processor import MarketData
from trading_bot.strategies.base_strategy import (
    SignalStrength,
    SignalType,
    StrategyConfiguration,
    TradingSignal,
)
from trading_bot.strategies.ict_patterns import (
    Direction,
    IctConfiguration,
    OrderBlock,
    OrderBlockType,
)
from trading_bot.strategies.ict_strategy import (
    IctConfigurationError,
    IctSignalMetrics,
    ICTStrategy,
    IctStrategyConfiguration,
    create_ict_strategy,
)


class TestIctStrategyConfiguration:
    """Test ICT strategy configuration validation."""

    def test_valid_configuration(self):
        """Test valid ICT strategy configuration creation."""
        config = IctStrategyConfiguration(
            min_order_block_confidence=0.8,
            max_order_blocks_for_signals=5,
            base_position_size=0.02,
            target_multiplier=2.5,
        )

        assert config.min_order_block_confidence == 0.8
        assert config.max_order_blocks_for_signals == 5
        assert config.base_position_size == 0.02
        assert config.target_multiplier == 2.5

    def test_invalid_confidence_range(self):
        """Test invalid confidence range validation."""
        with pytest.raises(IctConfigurationError):
            IctStrategyConfiguration(min_order_block_confidence=1.5)

        with pytest.raises(IctConfigurationError):
            IctStrategyConfiguration(min_order_block_confidence=-0.1)

    def test_invalid_max_order_blocks(self):
        """Test invalid max order blocks validation."""
        with pytest.raises(IctConfigurationError):
            IctStrategyConfiguration(max_order_blocks_for_signals=0)

        with pytest.raises(IctConfigurationError):
            IctStrategyConfiguration(max_order_blocks_for_signals=-1)

    def test_invalid_position_size(self):
        """Test invalid position size validation."""
        with pytest.raises(IctConfigurationError):
            IctStrategyConfiguration(base_position_size=0)

        with pytest.raises(IctConfigurationError):
            IctStrategyConfiguration(base_position_size=-0.01)

    def test_invalid_target_multiplier(self):
        """Test invalid target multiplier validation."""
        with pytest.raises(IctConfigurationError):
            IctStrategyConfiguration(target_multiplier=0)

        with pytest.raises(IctConfigurationError):
            IctStrategyConfiguration(target_multiplier=-1)


class TestIctSignalMetrics:
    """Test ICT signal metrics functionality."""

    def test_metrics_initialization(self):
        """Test metrics initialization with default values."""
        metrics = IctSignalMetrics()

        assert metrics.total_signals == 0
        assert metrics.signals_by_type == {}
        assert metrics.average_signal_confidence == 0.0
        assert metrics.last_signal_time == 0

    def test_update_signal_metrics(self):
        """Test signal metrics update functionality."""
        metrics = IctSignalMetrics()

        # Create test signal
        signal = TradingSignal(
            symbol="EURUSD",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            price=1.1050,
            timestamp=int(time.time() * 1000),
            strategy_name="test_strategy",
            confidence=0.85,
        )

        # Update metrics
        metrics.update_signal_metrics(signal)

        assert metrics.total_signals == 1
        assert metrics.signals_by_type["buy"] == 1
        assert metrics.average_signal_confidence == 0.85
        assert metrics.last_signal_time == signal.timestamp

        # Add another signal
        signal2 = TradingSignal(
            symbol="EURUSD",
            signal_type=SignalType.SELL,
            strength=SignalStrength.MODERATE,
            price=1.1040,
            timestamp=int(time.time() * 1000),
            strategy_name="test_strategy",
            confidence=0.75,
        )

        metrics.update_signal_metrics(signal2)

        assert metrics.total_signals == 2
        assert metrics.signals_by_type["buy"] == 1
        assert metrics.signals_by_type["sell"] == 1
        assert metrics.average_signal_confidence == 0.8  # (0.85 + 0.75) / 2


class TestICTStrategy:
    """Test ICT Strategy implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.event_hub = EventHub()
        self.base_config = StrategyConfiguration(
            name="test_ict_strategy",
            symbol="EURUSD",
            timeframe="5m",
            min_confidence=0.7,
        )
        self.ict_config = IctConfiguration()
        self.ict_strategy_config = IctStrategyConfiguration()

    def test_strategy_initialization(self):
        """Test ICT strategy initialization."""
        strategy = ICTStrategy(
            config=self.base_config,
            event_hub=self.event_hub,
            ict_config=self.ict_config,
            ict_strategy_config=self.ict_strategy_config,
        )

        assert strategy._config.name == "test_ict_strategy"
        assert strategy._config.symbol == "EURUSD"
        assert strategy._ict_config is not None
        assert strategy._ict_strategy_config is not None
        assert len(strategy._order_blocks) == 0

    def test_strategy_initialization_success(self):
        """Test successful strategy initialization."""
        strategy = ICTStrategy(
            config=self.base_config,
            event_hub=self.event_hub,
            ict_config=self.ict_config,
            ict_strategy_config=self.ict_strategy_config,
        )

        strategy.initialize()

        assert strategy._is_initialized is True
        assert len(strategy._market_data_history) == 0
        assert len(strategy._order_blocks) == 0

    def test_market_data_conversion(self):
        """Test market data to candle conversion."""
        strategy = ICTStrategy(
            config=self.base_config,
            event_hub=self.event_hub,
            ict_config=self.ict_config,
            ict_strategy_config=self.ict_strategy_config,
        )

        # Test with OHLC metadata
        market_data = MarketData(
            symbol="EURUSD",
            timestamp=int(time.time() * 1000),
            price=1.1050,
            volume=1000.0,
            source="test",
            data_type="tick",
            raw_data={},
            metadata={
                "open_price": 1.1045,
                "high_price": 1.1055,
                "low_price": 1.1040,
                "is_closed": True,
            },
        )

        candle_data = strategy._convert_to_candle_data(market_data)

        assert candle_data is not None
        assert candle_data["open"] == 1.1045
        assert candle_data["high"] == 1.1055
        assert candle_data["low"] == 1.1040
        assert candle_data["close"] == 1.1050
        assert candle_data["volume"] == 1000.0

    def test_market_data_conversion_simple(self):
        """Test market data conversion without OHLC metadata."""
        strategy = ICTStrategy(
            config=self.base_config,
            event_hub=self.event_hub,
            ict_config=self.ict_config,
            ict_strategy_config=self.ict_strategy_config,
        )

        # Test without OHLC metadata
        market_data = MarketData(
            symbol="EURUSD",
            timestamp=int(time.time() * 1000),
            price=1.1050,
            volume=1000.0,
            source="test",
            data_type="tick",
            raw_data={},
            metadata={},
        )

        candle_data = strategy._convert_to_candle_data(market_data)

        assert candle_data is not None
        assert candle_data["open"] == 1.1050
        assert candle_data["high"] == 1.1050
        assert candle_data["low"] == 1.1050
        assert candle_data["close"] == 1.1050

    def test_position_size_calculation(self):
        """Test position size calculation based on confidence."""
        strategy = ICTStrategy(
            config=self.base_config,
            event_hub=self.event_hub,
            ict_config=self.ict_config,
            ict_strategy_config=self.ict_strategy_config,
        )

        # Test with confidence-based sizing enabled
        strategy._ict_strategy_config.confidence_based_sizing = True
        strategy._ict_strategy_config.base_position_size = 0.01
        strategy._ict_strategy_config.min_confidence_multiplier = 0.5
        strategy._ict_strategy_config.max_confidence_multiplier = 2.0

        # High confidence
        size_high = strategy._calculate_position_size(0.9)
        expected_high = 0.01 * (0.5 + (2.0 - 0.5) * 0.9)
        assert abs(size_high - expected_high) < 1e-6

        # Low confidence
        size_low = strategy._calculate_position_size(0.1)
        expected_low = 0.01 * (0.5 + (2.0 - 0.5) * 0.1)
        assert abs(size_low - expected_low) < 1e-6

        # Test with confidence-based sizing disabled
        strategy._ict_strategy_config.confidence_based_sizing = False
        size_fixed = strategy._calculate_position_size(0.9)
        assert size_fixed == 0.01

    def test_signal_strength_determination(self):
        """Test signal strength determination based on confidence."""
        strategy = ICTStrategy(
            config=self.base_config,
            event_hub=self.event_hub,
            ict_config=self.ict_config,
            ict_strategy_config=self.ict_strategy_config,
        )

        assert strategy._determine_signal_strength(0.95) == SignalStrength.VERY_STRONG
        assert strategy._determine_signal_strength(0.85) == SignalStrength.STRONG
        assert strategy._determine_signal_strength(0.75) == SignalStrength.MODERATE
        assert strategy._determine_signal_strength(0.65) == SignalStrength.WEAK

    def test_risk_reward_calculation_buy(self):
        """Test risk/reward level calculation for buy signals."""
        strategy = ICTStrategy(
            config=self.base_config,
            event_hub=self.event_hub,
            ict_config=self.ict_config,
            ict_strategy_config=self.ict_strategy_config,
        )

        # Create test order block
        order_block = OrderBlock(
            start_time=int(time.time() * 1000),
            end_time=int(time.time() * 1000),
            high_price=1.1060,
            low_price=1.1040,
            order_block_type=OrderBlockType.DEMAND_ZONE,
            direction=Direction.BULLISH,
            volume=1000.0,
            formation_candle_index=10,
            structure_break_index=15,
            confidence=0.8,
        )

        entry_price = 1.1050
        stop_loss, target_price = strategy._calculate_risk_reward_levels(
            order_block, entry_price, SignalType.BUY
        )

        # Stop loss should be below order block low with buffer
        expected_stop = 1.1040 * (1 - 0.001)  # Default buffer
        assert abs(stop_loss - expected_stop) < 1e-6

        # Target should be entry + (risk * multiplier)
        risk = entry_price - stop_loss
        expected_target = entry_price + (risk * 2.0)  # Default multiplier
        assert abs(target_price - expected_target) < 1e-6

    def test_risk_reward_calculation_sell(self):
        """Test risk/reward level calculation for sell signals."""
        strategy = ICTStrategy(
            config=self.base_config,
            event_hub=self.event_hub,
            ict_config=self.ict_config,
            ict_strategy_config=self.ict_strategy_config,
        )

        # Create test order block
        order_block = OrderBlock(
            start_time=int(time.time() * 1000),
            end_time=int(time.time() * 1000),
            high_price=1.1060,
            low_price=1.1040,
            order_block_type=OrderBlockType.SUPPLY_ZONE,
            direction=Direction.BEARISH,
            volume=1000.0,
            formation_candle_index=10,
            structure_break_index=15,
            confidence=0.8,
        )

        entry_price = 1.1050
        stop_loss, target_price = strategy._calculate_risk_reward_levels(
            order_block, entry_price, SignalType.SELL
        )

        # Stop loss should be above order block high with buffer
        expected_stop = 1.1060 * (1 + 0.001)  # Default buffer
        assert abs(stop_loss - expected_stop) < 1e-6

        # Target should be entry - (risk * multiplier)
        risk = stop_loss - entry_price
        expected_target = entry_price - (risk * 2.0)  # Default multiplier
        assert abs(target_price - expected_target) < 1e-6

    def test_signal_cooldown(self):
        """Test signal generation cooldown functionality."""
        strategy = ICTStrategy(
            config=self.base_config,
            event_hub=self.event_hub,
            ict_config=self.ict_config,
            ict_strategy_config=self.ict_strategy_config,
        )

        # Set cooldown period
        strategy._ict_strategy_config.signal_cooldown_minutes = 5
        current_time = int(time.time() * 1000)

        # No previous signal - should not be in cooldown
        assert not strategy._is_signal_in_cooldown(current_time)

        # Set last signal time to 3 minutes ago
        strategy._last_signal_timestamp = current_time - (3 * 60 * 1000)
        assert strategy._is_signal_in_cooldown(current_time)

        # Set last signal time to 6 minutes ago
        strategy._last_signal_timestamp = current_time - (6 * 60 * 1000)
        assert not strategy._is_signal_in_cooldown(current_time)

    def test_order_block_invalidation(self):
        """Test order block invalidation logic."""
        strategy = ICTStrategy(
            config=self.base_config,
            event_hub=self.event_hub,
            ict_config=self.ict_config,
            ict_strategy_config=self.ict_strategy_config,
        )

        # Bullish order block
        bullish_block = OrderBlock(
            start_time=int(time.time() * 1000),
            end_time=int(time.time() * 1000),
            high_price=1.1060,
            low_price=1.1040,
            order_block_type=OrderBlockType.DEMAND_ZONE,
            direction=Direction.BULLISH,
            volume=1000.0,
            formation_candle_index=10,
            structure_break_index=15,
            confidence=0.8,
        )

        # Price above order block - should not invalidate
        assert not strategy._check_order_block_invalidation(bullish_block, 1.1055)

        # Price significantly below order block - should invalidate
        # 1.1040 * 0.998 = 1.101792, so 1.100 should clearly invalidate
        assert strategy._check_order_block_invalidation(bullish_block, 1.100)

        # Bearish order block
        bearish_block = OrderBlock(
            start_time=int(time.time() * 1000),
            end_time=int(time.time() * 1000),
            high_price=1.1060,
            low_price=1.1040,
            order_block_type=OrderBlockType.SUPPLY_ZONE,
            direction=Direction.BEARISH,
            volume=1000.0,
            formation_candle_index=10,
            structure_break_index=15,
            confidence=0.8,
        )

        # Price below order block - should not invalidate
        assert not strategy._check_order_block_invalidation(bearish_block, 1.1045)

        # Price significantly above order block - should invalidate
        # 1.1060 * 1.002 = 1.108212, so 1.110 should clearly invalidate
        assert strategy._check_order_block_invalidation(bearish_block, 1.110)

    def test_performance_metrics(self):
        """Test ICT strategy performance metrics collection."""
        strategy = ICTStrategy(
            config=self.base_config,
            event_hub=self.event_hub,
            ict_config=self.ict_config,
            ict_strategy_config=self.ict_strategy_config,
        )

        strategy.initialize()

        # Get initial metrics
        metrics = strategy.get_ict_performance_metrics()

        assert "ict_order_blocks_detected" in metrics
        assert "ict_successful_mitigations" in metrics
        assert "ict_failed_mitigations" in metrics
        assert "ict_mitigation_success_rate" in metrics
        assert "ict_average_signal_confidence" in metrics
        assert "ict_active_order_blocks" in metrics
        assert "ict_configuration" in metrics

        # Check configuration values
        config = metrics["ict_configuration"]
        assert "min_confidence" in config
        assert "max_order_blocks" in config
        assert "position_sizing" in config
        assert "target_multiplier" in config

    def test_cleanup(self):
        """Test strategy cleanup functionality."""
        strategy = ICTStrategy(
            config=self.base_config,
            event_hub=self.event_hub,
            ict_config=self.ict_config,
            ict_strategy_config=self.ict_strategy_config,
        )

        strategy.initialize()

        # Add some test data
        strategy._market_data_history.append(
            MarketData(
                symbol="EURUSD",
                timestamp=int(time.time() * 1000),
                price=1.1050,
                volume=1000.0,
                source="test",
                data_type="tick",
                raw_data={},
                metadata={},
            )
        )

        # Cleanup should not raise exceptions
        strategy.cleanup()

        assert not strategy._is_initialized


class TestICTStrategyFactory:
    """Test ICT strategy factory function."""

    def test_create_ict_strategy_minimal(self):
        """Test ICT strategy creation with minimal parameters."""
        event_hub = EventHub()

        strategy = create_ict_strategy(
            name="test_strategy",
            symbol="EURUSD",
            timeframe="5m",
            event_hub=event_hub,
        )

        assert isinstance(strategy, ICTStrategy)
        assert strategy._config.name == "test_strategy"
        assert strategy._config.symbol == "EURUSD"
        assert strategy._config.timeframe == "5m"

    def test_create_ict_strategy_full_config(self):
        """Test ICT strategy creation with full configuration."""
        event_hub = EventHub()
        ict_config = IctConfiguration(min_candles_for_structure=15)
        ict_strategy_config = IctStrategyConfiguration(min_order_block_confidence=0.8)

        strategy = create_ict_strategy(
            name="test_strategy",
            symbol="EURUSD",
            timeframe="5m",
            event_hub=event_hub,
            ict_config=ict_config,
            ict_strategy_config=ict_strategy_config,
            risk_tolerance=0.03,
        )

        assert isinstance(strategy, ICTStrategy)
        assert strategy._config.risk_tolerance == 0.03
        assert strategy._ict_config.min_candles_for_structure == 15
        assert strategy._ict_strategy_config.min_order_block_confidence == 0.8


if __name__ == "__main__":
    pytest.main([__file__])
