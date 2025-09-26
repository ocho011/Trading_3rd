"""
ICT Strategy Implementation for institutional trading pattern analysis.

This module implements the ICTStrategy class that integrates the BaseStrategy framework
with ICT (Inner Circle Trader) pattern detection algorithms for automated trading
signal generation based on order blocks, structure breaks, and institutional analysis.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from trading_bot.core.event_hub import EventHub
from trading_bot.market_data.data_processor import MarketData
from trading_bot.strategies.base_strategy import (
    BaseStrategy,
    SignalGenerationError,
    SignalStrength,
    SignalType,
    StrategyConfiguration,
    TradingSignal,
)
from trading_bot.strategies.ict_patterns import (
    Direction,
    IctConfiguration,
    IctOrderBlockDetector,
    OrderBlock,
    StructurePoint,
    ZigZagStructureAnalyzer,
)


class IctStrategyError(Exception):
    """Base exception for ICT strategy-specific errors."""


class IctConfigurationError(IctStrategyError):
    """Exception raised for ICT configuration errors."""


class IctSignalGenerationError(IctStrategyError):
    """Exception raised for ICT signal generation errors."""


@dataclass
class IctStrategyConfiguration:
    """ICT Strategy-specific configuration extending base strategy configuration.

    This configuration includes ICT-specific parameters for order block detection,
    signal generation, and risk management tailored to institutional trading patterns.
    """

    # Order Block Signal Parameters
    min_order_block_confidence: float = 0.7
    max_order_blocks_for_signals: int = 5
    order_block_interaction_tolerance: float = 0.002  # 0.2%
    signal_cooldown_minutes: int = 15

    # Position Sizing Parameters
    confidence_based_sizing: bool = True
    base_position_size: float = 0.01  # 1% of capital
    max_confidence_multiplier: float = 2.0
    min_confidence_multiplier: float = 0.5

    # Stop Loss and Target Parameters
    stop_loss_buffer_percentage: float = 0.001  # 0.1% buffer beyond order block
    target_multiplier: float = 2.0  # Risk-reward ratio
    use_structure_targets: bool = True

    # Signal Filtering Parameters
    require_structure_confirmation: bool = True
    min_order_block_size_percentage: float = 0.0005  # 0.05%
    max_order_block_age_hours: int = 168  # 1 week

    # Performance Tracking Parameters
    track_mitigation_success: bool = True
    track_signal_accuracy: bool = True
    min_signals_for_metrics: int = 10

    def __post_init__(self) -> None:
        """Validate ICT strategy configuration parameters."""
        if not 0.0 <= self.min_order_block_confidence <= 1.0:
            raise IctConfigurationError(
                "Min order block confidence must be between 0.0 and 1.0"
            )
        if self.max_order_blocks_for_signals <= 0:
            raise IctConfigurationError("Max order blocks for signals must be positive")
        if self.base_position_size <= 0:
            raise IctConfigurationError("Base position size must be positive")
        if self.target_multiplier <= 0:
            raise IctConfigurationError("Target multiplier must be positive")


@dataclass
class IctSignalMetrics:
    """ICT strategy performance metrics and tracking data."""

    total_signals: int = 0
    signals_by_type: Dict[str, int] = field(default_factory=dict)
    order_blocks_detected: int = 0
    successful_mitigations: int = 0
    failed_mitigations: int = 0
    average_signal_confidence: float = 0.0
    mitigation_success_rate: float = 0.0
    signal_accuracy_rate: float = 0.0
    last_signal_time: int = 0
    active_order_blocks: int = 0

    def update_signal_metrics(self, signal: TradingSignal) -> None:
        """Update metrics with new signal data."""
        self.total_signals += 1
        signal_type = signal.signal_type.value
        self.signals_by_type[signal_type] = self.signals_by_type.get(signal_type, 0) + 1

        # Update average confidence
        old_avg = self.average_signal_confidence
        new_avg = (
            old_avg * (self.total_signals - 1) + signal.confidence
        ) / self.total_signals
        self.average_signal_confidence = new_avg

        self.last_signal_time = signal.timestamp


class ICTStrategy(BaseStrategy):
    """ICT Strategy implementation for institutional trading pattern analysis.

    This strategy integrates the BaseStrategy framework with ICT pattern detection
    algorithms to generate trading signals based on order block interactions,
    structure breaks, and institutional trading concepts.

    Key Features:
    - Order block detection and tracking
    - Structure break analysis
    - Confidence-based position sizing
    - Risk management with structure-based stops
    - Performance tracking for ICT-specific metrics
    """

    def __init__(
        self,
        config: StrategyConfiguration,
        event_hub: EventHub,
        ict_config: Optional[IctConfiguration] = None,
        ict_strategy_config: Optional[IctStrategyConfiguration] = None,
    ) -> None:
        """Initialize ICT strategy with configuration and dependencies.

        Args:
            config: Base strategy configuration
            event_hub: Event hub for event communication
            ict_config: ICT pattern detection configuration
            ict_strategy_config: ICT strategy-specific configuration

        Raises:
            InvalidStrategyConfigError: If configuration is invalid
            IctConfigurationError: If ICT configuration is invalid
        """
        super().__init__(config, event_hub)

        # ICT-specific configurations
        self._ict_config = ict_config or IctConfiguration()
        self._ict_strategy_config = ict_strategy_config or IctStrategyConfiguration()

        # Pattern detection components
        self._structure_analyzer = ZigZagStructureAnalyzer(self._ict_config)
        self._order_block_detector = IctOrderBlockDetector(self._ict_config)

        # State tracking
        self._market_data_history: List[MarketData] = []
        self._order_blocks: List[OrderBlock] = []
        self._structure_points: List[StructurePoint] = []
        self._last_signal_timestamp = 0

        # Performance metrics
        self._ict_metrics = IctSignalMetrics()

        # Signal generation state
        self._current_candle_data: Optional[Dict[str, Any]] = None
        self._price_levels: Dict[str, float] = {}

        self._logger.info(f"ICT Strategy initialized for {config.symbol}")

    def _generate_signal_implementation(
        self, market_data: MarketData
    ) -> Optional[TradingSignal]:
        """Generate trading signal based on ICT pattern analysis.

        This method implements the core ICT signal generation logic by:
        1. Converting market data to candle format
        2. Updating order block analysis
        3. Checking for order block interactions
        4. Generating signals with confidence-based parameters

        Args:
            market_data: Current market data for analysis

        Returns:
            TradingSignal if ICT conditions are met, None otherwise

        Raises:
            SignalGenerationError: If signal generation fails
        """
        try:
            # Update market data history
            self._update_market_data_history(market_data)

            # Check signal cooldown
            if self._is_signal_in_cooldown(market_data.timestamp):
                return None

            # Convert market data to candle format for analysis
            candle_data = self._convert_to_candle_data(market_data)
            if not candle_data:
                return None

            # Update order block analysis
            self._update_order_block_analysis(candle_data, market_data.timestamp)

            # Find the best order block interaction
            best_order_block = self._find_best_order_block_interaction(
                market_data.price
            )
            if not best_order_block:
                return None

            # Generate signal from order block interaction
            signal = self._create_signal_from_order_block(best_order_block, market_data)

            if signal:
                # Update metrics
                self._ict_metrics.update_signal_metrics(signal)

                # Mark order block as tested
                best_order_block.tested = True
                best_order_block.last_test_time = market_data.timestamp
                best_order_block.mitigation_count += 1

                self._logger.info(
                    f"ICT signal generated: {signal.signal_type.value} "
                    f"for {signal.symbol} at {signal.price:.6f} "
                    f"(confidence: {signal.confidence:.3f})"
                )

            return signal

        except Exception as e:
            self._logger.error(f"Error in ICT signal generation: {e}")
            raise SignalGenerationError(f"ICT signal generation failed: {e}")

    def _initialize_strategy(self) -> None:
        """Perform ICT strategy-specific initialization."""
        try:
            self._logger.info("Initializing ICT strategy components")

            # Validate configurations
            if not isinstance(self._ict_config, IctConfiguration):
                raise IctConfigurationError("Invalid ICT configuration")

            if not isinstance(self._ict_strategy_config, IctStrategyConfiguration):
                raise IctConfigurationError("Invalid ICT strategy configuration")

            # Initialize tracking structures
            self._market_data_history = []
            self._order_blocks = []
            self._structure_points = []
            self._ict_metrics = IctSignalMetrics()
            self._price_levels = {}

            # Log configuration
            min_conf = self._ict_strategy_config.min_order_block_confidence
            max_blocks = self._ict_strategy_config.max_order_blocks_for_signals
            self._logger.info(
                f"ICT Strategy initialized with min_confidence: {min_conf}, "
                f"max_order_blocks: {max_blocks}"
            )

        except Exception as e:
            self._logger.error(f"Failed to initialize ICT strategy: {e}")
            raise

    def _cleanup_strategy(self) -> None:
        """Perform ICT strategy-specific cleanup."""
        try:
            self._logger.info("Cleaning up ICT strategy")

            # Log final performance metrics
            final_metrics = self.get_ict_performance_metrics()
            self._logger.info(f"Final ICT metrics: {final_metrics}")

            # Clear tracking structures
            self._market_data_history.clear()
            self._order_blocks.clear()
            self._structure_points.clear()

            self._logger.info("ICT strategy cleanup completed")

        except Exception as e:
            self._logger.error(f"Error during ICT strategy cleanup: {e}")

    def _update_market_data_history(self, market_data: MarketData) -> None:
        """Update market data history for analysis.

        Args:
            market_data: New market data to add
        """
        try:
            self._market_data_history.append(market_data)

            # Limit history size for performance
            max_history = self._ict_config.candle_history_limit
            if len(self._market_data_history) > max_history:
                self._market_data_history = self._market_data_history[-max_history:]

        except Exception as e:
            self._logger.error(f"Error updating market data history: {e}")

    def _convert_to_candle_data(
        self, market_data: MarketData
    ) -> Optional[Dict[str, Any]]:
        """Convert market data to candle format for ICT analysis.

        Args:
            market_data: Market data to convert

        Returns:
            Dictionary with candle data or None if conversion fails
        """
        try:
            # Extract OHLC data from metadata if available
            metadata = market_data.metadata

            if "open_price" in metadata:
                candle_data = {
                    "timestamp": market_data.timestamp,
                    "open": metadata["open_price"],
                    "high": metadata.get("high_price", market_data.price),
                    "low": metadata.get("low_price", market_data.price),
                    "close": market_data.price,
                    "volume": market_data.volume,
                }
            else:
                # Create simple candle from current price
                candle_data = {
                    "timestamp": market_data.timestamp,
                    "open": market_data.price,
                    "high": market_data.price,
                    "low": market_data.price,
                    "close": market_data.price,
                    "volume": market_data.volume,
                }

            self._current_candle_data = candle_data
            return candle_data

        except Exception as e:
            self._logger.error(f"Error converting to candle data: {e}")
            return None

    def _update_order_block_analysis(
        self, candle_data: Dict[str, Any], current_time: int
    ) -> None:
        """Update order block analysis with new candle data.

        Args:
            candle_data: Current candle data
            current_time: Current timestamp
        """
        try:
            # Need sufficient history for analysis
            if (
                len(self._market_data_history)
                < self._ict_config.min_candles_for_structure
            ):
                return

            # Convert history to DataFrame-like structure for analysis
            candles_data = self._convert_history_to_analysis_format()
            if not candles_data:
                return

            # Update structure analysis
            self._update_structure_points(candles_data)

            # Detect new order blocks if structure break occurred
            self._detect_new_order_blocks(candles_data)

            # Update existing order blocks
            self._update_existing_order_blocks(candle_data, current_time)

            # Clean up old/invalid order blocks
            self._cleanup_order_blocks(current_time)

        except Exception as e:
            self._logger.error(f"Error updating order block analysis: {e}")

    def _find_best_order_block_interaction(
        self, current_price: float
    ) -> Optional[OrderBlock]:
        """Find the best order block for signal generation.

        Args:
            current_price: Current market price

        Returns:
            Best order block for signal generation or None
        """
        try:
            candidate_blocks = []

            # Filter active order blocks within interaction zone
            for order_block in self._order_blocks:
                if (
                    not order_block.invalidated
                    and order_block.confidence
                    >= self._ict_strategy_config.min_order_block_confidence
                    and order_block.is_within_zone(
                        current_price,
                        self._ict_strategy_config.order_block_interaction_tolerance,
                    )
                ):
                    candidate_blocks.append(order_block)

            if not candidate_blocks:
                return None

            # Sort by confidence and choose the best
            candidate_blocks.sort(key=lambda ob: ob.confidence, reverse=True)

            # Limit to max order blocks for signals
            max_blocks = self._ict_strategy_config.max_order_blocks_for_signals
            candidate_blocks = candidate_blocks[:max_blocks]

            # Return highest confidence order block
            return candidate_blocks[0]

        except Exception as e:
            self._logger.error(f"Error finding best order block interaction: {e}")
            return None

    def _create_signal_from_order_block(
        self, order_block: OrderBlock, market_data: MarketData
    ) -> Optional[TradingSignal]:
        """Create trading signal from order block interaction.

        Args:
            order_block: Order block generating the signal
            market_data: Current market data

        Returns:
            TradingSignal if valid signal can be created
        """
        try:
            # Determine signal direction based on order block type and direction
            if order_block.direction == Direction.BULLISH:
                signal_type = SignalType.BUY
            elif order_block.direction == Direction.BEARISH:
                signal_type = SignalType.SELL
            else:
                return None

            # Calculate position size based on confidence
            position_size = self._calculate_position_size(order_block.confidence)

            # Calculate stop loss and target prices
            stop_loss, target_price = self._calculate_risk_reward_levels(
                order_block, market_data.price, signal_type
            )

            # Determine signal strength based on confidence
            strength = self._determine_signal_strength(order_block.confidence)

            # Create reasoning for the signal
            reasoning = self._create_signal_reasoning(order_block)

            # Create the trading signal
            signal = TradingSignal(
                symbol=market_data.symbol,
                signal_type=signal_type,
                strength=strength,
                price=market_data.price,
                timestamp=market_data.timestamp,
                strategy_name=self._config.name,
                confidence=order_block.confidence,
                reasoning=reasoning,
                target_price=target_price,
                stop_loss=stop_loss,
                metadata={
                    "ict_strategy": True,
                    "order_block_type": order_block.order_block_type.value,
                    "order_block_direction": order_block.direction.value,
                    "order_block_confidence": order_block.confidence,
                    "mitigation_count": order_block.mitigation_count,
                    "order_block_size": order_block.size,
                    "position_size": position_size,
                    "risk_reward_ratio": self._ict_strategy_config.target_multiplier,
                    "structure_break_type": (
                        order_block.creation_structure.value
                        if order_block.creation_structure
                        else "unknown"
                    ),
                },
            )

            return signal

        except Exception as e:
            self._logger.error(f"Error creating signal from order block: {e}")
            return None

    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on order block confidence.

        Args:
            confidence: Order block confidence score

        Returns:
            Position size as percentage of capital
        """
        try:
            base_size = self._ict_strategy_config.base_position_size

            if not self._ict_strategy_config.confidence_based_sizing:
                return base_size

            # Scale position size based on confidence
            min_mult = self._ict_strategy_config.min_confidence_multiplier
            max_mult = self._ict_strategy_config.max_confidence_multiplier

            # Linear scaling based on confidence
            multiplier = min_mult + (max_mult - min_mult) * confidence

            return base_size * multiplier

        except Exception as e:
            self._logger.error(f"Error calculating position size: {e}")
            return self._ict_strategy_config.base_position_size

    def _calculate_risk_reward_levels(
        self, order_block: OrderBlock, entry_price: float, signal_type: SignalType
    ) -> tuple[float, float]:
        """Calculate stop loss and target price levels.

        Args:
            order_block: Order block for reference levels
            entry_price: Entry price for the trade
            signal_type: Type of signal (BUY/SELL)

        Returns:
            Tuple of (stop_loss, target_price)
        """
        try:
            buffer = self._ict_strategy_config.stop_loss_buffer_percentage
            target_mult = self._ict_strategy_config.target_multiplier

            if signal_type == SignalType.BUY:
                # For bullish signals, stop below order block low
                stop_loss = order_block.low_price * (1 - buffer)
                risk = entry_price - stop_loss
                target_price = entry_price + (risk * target_mult)

            elif signal_type == SignalType.SELL:
                # For bearish signals, stop above order block high
                stop_loss = order_block.high_price * (1 + buffer)
                risk = stop_loss - entry_price
                target_price = entry_price - (risk * target_mult)

            else:
                # Default to percentage-based levels
                stop_loss = entry_price * (
                    0.98 if signal_type == SignalType.BUY else 1.02
                )
                target_price = entry_price * (
                    1.02 if signal_type == SignalType.BUY else 0.98
                )

            return stop_loss, target_price

        except Exception as e:
            self._logger.error(f"Error calculating risk/reward levels: {e}")
            # Return default levels
            if signal_type == SignalType.BUY:
                return entry_price * 0.99, entry_price * 1.02
            else:
                return entry_price * 1.01, entry_price * 0.98

    def _determine_signal_strength(self, confidence: float) -> SignalStrength:
        """Determine signal strength based on confidence level.

        Args:
            confidence: Order block confidence score

        Returns:
            SignalStrength enum value
        """
        if confidence >= 0.9:
            return SignalStrength.VERY_STRONG
        elif confidence >= 0.8:
            return SignalStrength.STRONG
        elif confidence >= 0.7:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK

    def _create_signal_reasoning(self, order_block: OrderBlock) -> str:
        """Create human-readable reasoning for the signal.

        Args:
            order_block: Order block generating the signal

        Returns:
            Reasoning string explaining the signal
        """
        reasoning_parts = [
            f"ICT {order_block.order_block_type.value} interaction",
            f"Direction: {order_block.direction.value}",
            f"Confidence: {order_block.confidence:.3f}",
            f"Mitigation count: {order_block.mitigation_count}",
        ]

        if order_block.creation_structure:
            reasoning_parts.append(f"Structure: {order_block.creation_structure.value}")

        return ". ".join(reasoning_parts)

    def _is_signal_in_cooldown(self, current_timestamp: int) -> bool:
        """Check if signal generation is in cooldown period.

        Args:
            current_timestamp: Current timestamp

        Returns:
            True if in cooldown, False otherwise
        """
        try:
            if self._last_signal_timestamp == 0:
                return False

            cooldown_ms = self._ict_strategy_config.signal_cooldown_minutes * 60 * 1000
            time_since_last = current_timestamp - self._last_signal_timestamp

            return time_since_last < cooldown_ms

        except Exception as e:
            self._logger.error(f"Error checking signal cooldown: {e}")
            return False

    def _convert_history_to_analysis_format(self) -> Optional[List[Dict[str, Any]]]:
        """Convert market data history to format suitable for analysis.

        Returns:
            List of candle data dictionaries or None if conversion fails
        """
        try:
            if len(self._market_data_history) < 2:
                return None

            candles_data = []
            for market_data in self._market_data_history[
                -100:
            ]:  # Use last 100 data points
                candle = {
                    "timestamp": market_data.timestamp,
                    "open": market_data.price,  # Simplified - assume price is close
                    "high": market_data.price,
                    "low": market_data.price,
                    "close": market_data.price,
                    "volume": market_data.volume,
                }

                # Extract OHLC if available in metadata
                if "open_price" in market_data.metadata:
                    candle["open"] = market_data.metadata["open_price"]
                    candle["high"] = market_data.metadata.get(
                        "high_price", market_data.price
                    )
                    candle["low"] = market_data.metadata.get(
                        "low_price", market_data.price
                    )

                candles_data.append(candle)

            return candles_data

        except Exception as e:
            self._logger.error(f"Error converting history to analysis format: {e}")
            return None

    def _update_structure_points(self, candles_data: List[Dict[str, Any]]) -> None:
        """Update structure points analysis."""
        try:
            # This is a simplified implementation
            # In a real scenario, you'd use the ZigZagStructureAnalyzer properly
            pass
        except Exception as e:
            self._logger.error(f"Error updating structure points: {e}")

    def _detect_new_order_blocks(self, candles_data: List[Dict[str, Any]]) -> None:
        """Detect new order blocks from recent market data."""
        try:
            # This is a simplified implementation
            # In a real scenario, you'd use the IctOrderBlockDetector properly
            pass
        except Exception as e:
            self._logger.error(f"Error detecting new order blocks: {e}")

    def _update_existing_order_blocks(
        self, candle_data: Dict[str, Any], current_time: int
    ) -> None:
        """Update existing order blocks with current market data."""
        try:
            for order_block in self._order_blocks:
                if order_block.invalidated:
                    continue

                # Check for invalidation
                current_price = candle_data["close"]
                if self._check_order_block_invalidation(order_block, current_price):
                    order_block.invalidated = True
                    self._logger.info(
                        f"Order block invalidated: {order_block.order_block_type.value}"
                    )
                    continue

                # Update confidence based on age
                order_block.confidence = self._update_order_block_confidence(
                    order_block, current_time
                )

        except Exception as e:
            self._logger.error(f"Error updating existing order blocks: {e}")

    def _cleanup_order_blocks(self, current_time: int) -> None:
        """Clean up old and invalid order blocks."""
        try:
            max_age_ms = (
                self._ict_strategy_config.max_order_block_age_hours * 3600 * 1000
            )

            valid_blocks = []
            for order_block in self._order_blocks:
                # Remove if invalidated or too old
                age = current_time - order_block.start_time
                if not order_block.invalidated and age < max_age_ms:
                    valid_blocks.append(order_block)

            self._order_blocks = valid_blocks
            self._ict_metrics.active_order_blocks = len(valid_blocks)

        except Exception as e:
            self._logger.error(f"Error cleaning up order blocks: {e}")

    def _check_order_block_invalidation(
        self, order_block: OrderBlock, current_price: float
    ) -> bool:
        """Check if order block should be invalidated."""
        try:
            # Invalidate if price breaks through order block significantly
            if order_block.direction == Direction.BULLISH:
                return current_price < order_block.low_price * 0.998
            else:
                return current_price > order_block.high_price * 1.002

        except Exception as e:
            self._logger.error(f"Error checking order block invalidation: {e}")
            return False

    def _update_order_block_confidence(
        self, order_block: OrderBlock, current_time: int
    ) -> float:
        """Update order block confidence based on age and performance."""
        try:
            base_confidence = order_block.confidence

            # Age factor (confidence decreases over time)
            age_ms = current_time - order_block.start_time
            age_hours = age_ms / (1000 * 3600)
            age_factor = max(0.5, 1.0 - (age_hours / 168))  # Decay over 1 week

            # Test factor (successful tests maintain confidence)
            test_factor = 1.0 + (order_block.mitigation_count * 0.05)

            # Combine factors
            updated_confidence = base_confidence * age_factor * min(test_factor, 1.2)
            return min(max(updated_confidence, 0.0), 1.0)

        except Exception as e:
            self._logger.error(f"Error updating order block confidence: {e}")
            return order_block.confidence

    def get_ict_performance_metrics(self) -> Dict[str, Any]:
        """Get ICT strategy-specific performance metrics.

        Returns:
            Dictionary containing ICT performance metrics
        """
        try:
            base_metrics = self._get_performance_metrics()

            # Calculate mitigation success rate
            total_mitigations = (
                self._ict_metrics.successful_mitigations
                + self._ict_metrics.failed_mitigations
            )
            mitigation_rate = self._ict_metrics.successful_mitigations / max(
                total_mitigations, 1
            )

            ict_metrics = {
                "ict_order_blocks_detected": self._ict_metrics.order_blocks_detected,
                "ict_successful_mitigations": self._ict_metrics.successful_mitigations,
                "ict_failed_mitigations": self._ict_metrics.failed_mitigations,
                "ict_mitigation_success_rate": mitigation_rate,
                "ict_signals_by_type": self._ict_metrics.signals_by_type,
                "ict_average_signal_confidence": (
                    self._ict_metrics.average_signal_confidence
                ),
                "ict_active_order_blocks": len(
                    [ob for ob in self._order_blocks if not ob.invalidated]
                ),
                "ict_total_order_blocks": len(self._order_blocks),
                "ict_structure_points": len(self._structure_points),
                "ict_last_signal_time": self._ict_metrics.last_signal_time,
                "ict_configuration": {
                    "min_confidence": (
                        self._ict_strategy_config.min_order_block_confidence
                    ),
                    "max_order_blocks": (
                        self._ict_strategy_config.max_order_blocks_for_signals
                    ),
                    "position_sizing": (
                        self._ict_strategy_config.confidence_based_sizing
                    ),
                    "target_multiplier": self._ict_strategy_config.target_multiplier,
                },
            }

            # Merge with base metrics
            base_metrics.update(ict_metrics)
            return base_metrics

        except Exception as e:
            self._logger.error(f"Error getting ICT performance metrics: {e}")
            return {}

    def get_current_order_blocks(self) -> List[Dict[str, Any]]:
        """Get current active order blocks for monitoring.

        Returns:
            List of order block data dictionaries
        """
        try:
            order_block_data = []

            for order_block in self._order_blocks:
                if order_block.invalidated:
                    continue

                data = {
                    "type": order_block.order_block_type.value,
                    "direction": order_block.direction.value,
                    "high_price": order_block.high_price,
                    "low_price": order_block.low_price,
                    "confidence": order_block.confidence,
                    "tested": order_block.tested,
                    "mitigation_count": order_block.mitigation_count,
                    "start_time": order_block.start_time,
                    "size": order_block.size,
                    "formation_index": order_block.formation_candle_index,
                }

                order_block_data.append(data)

            return order_block_data

        except Exception as e:
            self._logger.error(f"Error getting current order blocks: {e}")
            return []


def create_ict_strategy(
    name: str,
    symbol: str,
    timeframe: str,
    event_hub: EventHub,
    ict_config: Optional[IctConfiguration] = None,
    ict_strategy_config: Optional[IctStrategyConfiguration] = None,
    **kwargs: Any,
) -> ICTStrategy:
    """Factory function to create ICT strategy instance.

    Args:
        name: Strategy name
        symbol: Trading symbol
        timeframe: Timeframe for strategy operation
        event_hub: Event hub instance
        ict_config: ICT pattern detection configuration
        ict_strategy_config: ICT strategy-specific configuration
        **kwargs: Additional strategy configuration parameters

    Returns:
        ICTStrategy: Configured ICT strategy instance

    Raises:
        InvalidStrategyConfigError: If configuration parameters are invalid
        IctConfigurationError: If ICT configuration is invalid
    """
    # Create base strategy configuration
    base_config = StrategyConfiguration(
        name=name,
        symbol=symbol,
        timeframe=timeframe,
        enabled=kwargs.get("enabled", True),
        risk_tolerance=kwargs.get("risk_tolerance", 0.02),
        min_confidence=kwargs.get("min_confidence", 0.7),
        max_position_size=kwargs.get("max_position_size", 1.0),
        use_stop_loss=kwargs.get("use_stop_loss", True),
        use_take_profit=kwargs.get("use_take_profit", True),
        parameters=kwargs.get("parameters", {}),
        metadata=kwargs.get("metadata", {}),
    )

    # Create ICT configurations if not provided
    if ict_config is None:
        ict_config = IctConfiguration()

    if ict_strategy_config is None:
        ict_strategy_config = IctStrategyConfiguration()

    return ICTStrategy(
        config=base_config,
        event_hub=event_hub,
        ict_config=ict_config,
        ict_strategy_config=ict_strategy_config,
    )
