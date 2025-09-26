"""
ICT (Inner Circle Trader) Order Block Pattern Detection Algorithm.

This module implements institutional trading pattern detection based on ICT concepts,
focusing on order blocks, break of structure, and fair value gaps. The algorithm
identifies key institutional levels and mitigation zones for algorithmic trading.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from trading_bot.core.event_hub import EventHub
from trading_bot.core.logger import get_module_logger
from trading_bot.market_data.data_processor import CandleData, MarketData, Timeframe
from trading_bot.strategies.base_strategy import (
    BaseStrategy,
    InvalidStrategyConfigError,
    SignalGenerationError,
    SignalStrength,
    SignalType,
    StrategyConfiguration,
    StrategyError,
    TradingSignal,
)


class IctPatternError(StrategyError):
    """Base exception for ICT pattern detection errors."""


class OrderBlockError(IctPatternError):
    """Exception raised for order block detection errors."""


class StructureAnalysisError(IctPatternError):
    """Exception raised for market structure analysis errors."""


class Direction(Enum):
    """Market direction enumeration."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class OrderBlockType(Enum):
    """Order block type classification."""

    BULLISH_BREAKER = "bullish_breaker"
    BEARISH_BREAKER = "bearish_breaker"
    DEMAND_ZONE = "demand_zone"
    SUPPLY_ZONE = "supply_zone"


class StructureType(Enum):
    """Market structure break type."""

    BREAK_OF_STRUCTURE = "bos"  # BOS
    CHANGE_OF_CHARACTER = "choch"  # CHoCH
    LIQUIDITY_SWEEP = "liquidity_sweep"


@dataclass
class StructurePoint:
    """Market structure point representation."""

    timestamp: int
    price: float
    direction: Direction
    volume: float
    candle_index: int
    strength: float = 0.0
    confirmed: bool = False

    def __post_init__(self) -> None:
        """Validate structure point data."""
        if self.price <= 0:
            raise OrderBlockError("Price must be positive")
        if self.strength < 0.0 or self.strength > 1.0:
            raise OrderBlockError("Strength must be between 0.0 and 1.0")


@dataclass
class FairValueGap:
    """Fair Value Gap (imbalance) representation."""

    start_time: int
    end_time: int
    high_price: float
    low_price: float
    direction: Direction
    volume: float
    filled: bool = False
    fill_percentage: float = 0.0
    creation_candle_index: int = 0

    def __post_init__(self) -> None:
        """Validate Fair Value Gap data."""
        if self.high_price <= self.low_price:
            raise OrderBlockError("High price must be greater than low price")
        if self.fill_percentage < 0.0 or self.fill_percentage > 1.0:
            raise OrderBlockError("Fill percentage must be between 0.0 and 1.0")

    @property
    def gap_size(self) -> float:
        """Calculate gap size."""
        return self.high_price - self.low_price

    @property
    def midpoint(self) -> float:
        """Calculate gap midpoint."""
        return (self.high_price + self.low_price) / 2.0


@dataclass
class OrderBlock:
    """Order Block representation with institutional trading context."""

    start_time: int
    end_time: int
    high_price: float
    low_price: float
    order_block_type: OrderBlockType
    direction: Direction
    volume: float
    formation_candle_index: int
    structure_break_index: int
    confidence: float = 0.0
    tested: bool = False
    invalidated: bool = False
    mitigation_count: int = 0
    last_test_time: int = 0
    creation_structure: Optional[StructureType] = None

    def __post_init__(self) -> None:
        """Validate order block data."""
        if self.high_price <= self.low_price:
            raise OrderBlockError("High price must be greater than low price")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise OrderBlockError("Confidence must be between 0.0 and 1.0")
        if self.mitigation_count < 0:
            raise OrderBlockError("Mitigation count cannot be negative")

    @property
    def size(self) -> float:
        """Calculate order block size."""
        return self.high_price - self.low_price

    @property
    def midpoint(self) -> float:
        """Calculate order block midpoint."""
        return (self.high_price + self.low_price) / 2.0

    def is_within_zone(self, price: float, tolerance: float = 0.001) -> bool:
        """Check if price is within order block zone.

        Args:
            price: Price to check
            tolerance: Tolerance percentage (default 0.1%)

        Returns:
            True if price is within the order block zone
        """
        tolerance_amount = self.size * tolerance
        return (
            self.low_price - tolerance_amount
            <= price
            <= self.high_price + tolerance_amount
        )


@dataclass
class IctConfiguration:
    """ICT pattern detection configuration."""

    # Order Block Detection Parameters
    min_candles_for_structure: int = 10
    max_candles_for_structure: int = 200
    volume_confirmation_factor: float = 1.5
    structure_break_percentage: float = 0.001  # 0.1%
    order_block_min_size_percentage: float = 0.0005  # 0.05%

    # Fair Value Gap Parameters
    min_gap_size_percentage: float = 0.001  # 0.1%
    max_gap_fill_lookback: int = 50
    gap_confirmation_candles: int = 2

    # Confidence Scoring Parameters
    volume_weight: float = 0.3
    structure_strength_weight: float = 0.4
    gap_confirmation_weight: float = 0.3
    min_confidence_threshold: float = 0.6

    # Risk Management
    max_order_blocks_tracked: int = 10
    max_fair_value_gaps_tracked: int = 20
    candle_history_limit: int = 1000

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.min_candles_for_structure <= 0:
            raise InvalidStrategyConfigError(
                "Min candles for structure must be positive"
            )
        if self.max_candles_for_structure <= self.min_candles_for_structure:
            raise InvalidStrategyConfigError(
                "Max candles must be greater than min candles"
            )
        if not 0.0 <= self.min_confidence_threshold <= 1.0:
            raise InvalidStrategyConfigError(
                "Min confidence threshold must be between 0.0 and 1.0"
            )


class IMarketStructureAnalyzer(ABC):
    """Interface for market structure analysis strategies."""

    @abstractmethod
    def find_structure_points(self, candles: pd.DataFrame) -> List[StructurePoint]:
        """Find significant structure points in price data."""

    @abstractmethod
    def detect_structure_break(
        self,
        candles: pd.DataFrame,
        structure_points: List[StructurePoint],
        current_index: int,
    ) -> Optional[StructureType]:
        """Detect break of structure or change of character."""


class ZigZagStructureAnalyzer(IMarketStructureAnalyzer):
    """ZigZag-based market structure analyzer."""

    def __init__(self, config: IctConfiguration) -> None:
        """Initialize ZigZag structure analyzer.

        Args:
            config: ICT pattern detection configuration
        """
        self._config = config
        self._logger = get_module_logger("zigzag_structure_analyzer")

    def find_structure_points(self, candles: pd.DataFrame) -> List[StructurePoint]:
        """Find significant structure points using ZigZag algorithm.

        Args:
            candles: DataFrame with OHLCV data

        Returns:
            List of significant structure points
        """
        try:
            if len(candles) < self._config.min_candles_for_structure:
                return []

            structure_points = []
            min_change = self._config.structure_break_percentage

            # Find peaks and troughs using rolling windows
            highs = candles["high"].values
            lows = candles["low"].values
            volumes = candles["volume"].values
            timestamps = candles["timestamp"].values

            # Detect local maxima (peaks)
            for i in range(5, len(candles) - 5):
                is_peak = all(
                    highs[i] > highs[j] for j in range(i - 5, i + 6) if j != i
                )
                if is_peak:
                    # Check if change is significant enough
                    prev_low = min(lows[max(0, i - 20) : i])
                    if (highs[i] - prev_low) / prev_low >= min_change:
                        strength = self._calculate_structure_strength(
                            candles.iloc[i], candles, i
                        )
                        structure_points.append(
                            StructurePoint(
                                timestamp=int(timestamps[i]),
                                price=highs[i],
                                direction=Direction.BEARISH,  # Peak = reversal down
                                volume=volumes[i],
                                candle_index=i,
                                strength=strength,
                                confirmed=True,
                            )
                        )

            # Detect local minima (troughs)
            for i in range(5, len(candles) - 5):
                is_trough = all(
                    lows[i] < lows[j] for j in range(i - 5, i + 6) if j != i
                )
                if is_trough:
                    # Check if change is significant enough
                    prev_high = max(highs[max(0, i - 20) : i])
                    if (prev_high - lows[i]) / prev_high >= min_change:
                        strength = self._calculate_structure_strength(
                            candles.iloc[i], candles, i
                        )
                        structure_points.append(
                            StructurePoint(
                                timestamp=int(timestamps[i]),
                                price=lows[i],
                                direction=Direction.BULLISH,  # Trough = reversal up
                                volume=volumes[i],
                                candle_index=i,
                                strength=strength,
                                confirmed=True,
                            )
                        )

            # Sort by candle index
            structure_points.sort(key=lambda x: x.candle_index)

            # Limit results
            return structure_points[-50:]  # Keep last 50 structure points

        except Exception as e:
            self._logger.error(f"Error finding structure points: {e}")
            raise StructureAnalysisError(f"Failed to find structure points: {e}")

    def detect_structure_break(
        self,
        candles: pd.DataFrame,
        structure_points: List[StructurePoint],
        current_index: int,
    ) -> Optional[StructureType]:
        """Detect break of structure or change of character.

        Args:
            candles: DataFrame with OHLCV data
            structure_points: List of structure points
            current_index: Current candle index

        Returns:
            Structure break type if detected, None otherwise
        """
        try:
            if current_index < 10 or not structure_points:
                return None

            current_candle = candles.iloc[current_index]
            current_high = current_candle["high"]
            current_low = current_candle["low"]

            # Find recent structure points
            recent_points = [
                sp
                for sp in structure_points
                if current_index - 50 <= sp.candle_index < current_index
            ]

            if len(recent_points) < 2:
                return None

            # Check for break of structure (continuation)
            # Bullish BOS: Break above previous higher high
            # Bearish BOS: Break below previous lower low

            recent_highs = [
                sp for sp in recent_points if sp.direction == Direction.BEARISH
            ]
            recent_lows = [
                sp for sp in recent_points if sp.direction == Direction.BULLISH
            ]

            if recent_highs:
                last_high = max(recent_highs, key=lambda x: x.price)
                if current_high > last_high.price * (
                    1 + self._config.structure_break_percentage
                ):
                    return StructureType.BREAK_OF_STRUCTURE

            if recent_lows:
                last_low = min(recent_lows, key=lambda x: x.price)
                if current_low < last_low.price * (
                    1 - self._config.structure_break_percentage
                ):
                    return StructureType.BREAK_OF_STRUCTURE

            # Check for change of character (reversal)
            # This is more complex and requires analyzing the sequence of highs/lows
            if self._detect_change_of_character(candles, recent_points, current_index):
                return StructureType.CHANGE_OF_CHARACTER

            return None

        except Exception as e:
            self._logger.error(f"Error detecting structure break: {e}")
            return None

    def _calculate_structure_strength(
        self, candle: pd.Series, candles: pd.DataFrame, index: int
    ) -> float:
        """Calculate structure point strength based on volume and price action.

        Args:
            candle: Current candle
            candles: Full candle DataFrame
            index: Current candle index

        Returns:
            Strength score between 0.0 and 1.0
        """
        try:
            # Volume strength (compared to recent average)
            recent_volumes = candles["volume"].iloc[max(0, index - 20) : index].mean()
            volume_strength = min(candle["volume"] / max(recent_volumes, 1), 3.0) / 3.0

            # Price action strength (body vs wick ratio)
            body_size = abs(candle["close"] - candle["open"])
            total_size = candle["high"] - candle["low"]
            body_ratio = body_size / max(total_size, 0.0001)

            # Distance from moving average
            if len(candles) >= 20:
                ma20 = candles["close"].iloc[max(0, index - 19) : index + 1].mean()
                distance_ratio = abs(candle["close"] - ma20) / ma20
                distance_strength = min(distance_ratio / 0.05, 1.0)  # Normalize to 5%
            else:
                distance_strength = 0.5

            # Combine factors
            strength = (
                volume_strength * 0.4 + body_ratio * 0.3 + distance_strength * 0.3
            )
            return min(max(strength, 0.0), 1.0)

        except Exception as e:
            self._logger.error(f"Error calculating structure strength: {e}")
            return 0.5

    def _detect_change_of_character(
        self,
        candles: pd.DataFrame,
        structure_points: List[StructurePoint],
        current_index: int,
    ) -> bool:
        """Detect change of character pattern.

        Args:
            candles: DataFrame with OHLCV data
            structure_points: List of structure points
            current_index: Current candle index

        Returns:
            True if CHoCH pattern detected
        """
        try:
            # Simplified CHoCH detection
            # Look for failure to make new highs/lows followed by opposite break
            if len(structure_points) < 3:
                return False

            # Sort by time
            sorted_points = sorted(structure_points, key=lambda x: x.candle_index)
            last_three = sorted_points[-3:]

            # Check for failure pattern
            if len(last_three) == 3:
                p1, p2, p3 = last_three
                current_price = candles.iloc[current_index]["close"]

                # Bullish CHoCH: Lower high followed by break of structure to upside
                if (
                    p1.direction == Direction.BEARISH
                    and p2.direction == Direction.BULLISH
                    and p3.direction == Direction.BEARISH
                    and p3.price < p1.price  # Lower high
                    and current_price > p1.price
                ):  # Break above
                    return True

                # Bearish CHoCH: Higher low followed by break of structure to downside
                if (
                    p1.direction == Direction.BULLISH
                    and p2.direction == Direction.BEARISH
                    and p3.direction == Direction.BULLISH
                    and p3.price > p1.price  # Higher low
                    and current_price < p1.price
                ):  # Break below
                    return True

            return False

        except Exception as e:
            self._logger.error(f"Error detecting change of character: {e}")
            return False


class IOrderBlockDetector(ABC):
    """Interface for order block detection strategies."""

    @abstractmethod
    def detect_order_blocks(
        self,
        candles: pd.DataFrame,
        structure_points: List[StructurePoint],
        structure_break: Optional[StructureType],
        current_index: int,
    ) -> List[OrderBlock]:
        """Detect order blocks based on structure analysis."""

    @abstractmethod
    def update_order_blocks(
        self,
        order_blocks: List[OrderBlock],
        current_candle: pd.Series,
        current_time: int,
    ) -> List[OrderBlock]:
        """Update existing order blocks with current market data."""


class IctOrderBlockDetector(IOrderBlockDetector):
    """ICT-based order block detector implementation."""

    def __init__(self, config: IctConfiguration) -> None:
        """Initialize ICT order block detector.

        Args:
            config: ICT pattern detection configuration
        """
        self._config = config
        self._logger = get_module_logger("ict_order_block_detector")

    def detect_order_blocks(
        self,
        candles: pd.DataFrame,
        structure_points: List[StructurePoint],
        structure_break: Optional[StructureType],
        current_index: int,
    ) -> List[OrderBlock]:
        """Detect order blocks based on structure analysis.

        Args:
            candles: DataFrame with OHLCV data
            structure_points: List of structure points
            structure_break: Type of structure break detected
            current_index: Current candle index

        Returns:
            List of detected order blocks
        """
        try:
            order_blocks = []

            if not structure_break or not structure_points:
                return order_blocks

            # Find the last opposing candle before structure break
            break_candle = candles.iloc[current_index]
            break_direction = self._determine_break_direction(
                break_candle, structure_break
            )

            if break_direction == Direction.NEUTRAL:
                return order_blocks

            # Look for order block formation candle
            ob_candle_index = self._find_order_block_candle(
                candles, current_index, break_direction
            )

            if ob_candle_index is None:
                return order_blocks

            ob_candle = candles.iloc[ob_candle_index]

            # Create order block
            order_block = self._create_order_block(
                ob_candle,
                ob_candle_index,
                current_index,
                break_direction,
                structure_break,
                candles,
            )

            if order_block and self._validate_order_block(order_block, candles):
                order_blocks.append(order_block)

            return order_blocks

        except Exception as e:
            self._logger.error(f"Error detecting order blocks: {e}")
            raise OrderBlockError(f"Failed to detect order blocks: {e}")

    def update_order_blocks(
        self,
        order_blocks: List[OrderBlock],
        current_candle: pd.Series,
        current_time: int,
    ) -> List[OrderBlock]:
        """Update existing order blocks with current market data.

        Args:
            order_blocks: List of existing order blocks
            current_candle: Current candle data
            current_time: Current timestamp

        Returns:
            Updated list of order blocks
        """
        try:
            updated_blocks = []
            current_price = current_candle["close"]

            for order_block in order_blocks:
                if order_block.invalidated:
                    continue

                # Check for mitigation (price returning to order block)
                if order_block.is_within_zone(current_price):
                    if not order_block.tested:
                        order_block.tested = True
                        order_block.last_test_time = current_time
                        order_block.mitigation_count += 1
                        self._logger.info(
                            f"OB mitigated: {order_block.order_block_type.value}"
                            f" at {current_price}"
                        )

                # Check for invalidation
                if self._check_order_block_invalidation(order_block, current_candle):
                    order_block.invalidated = True
                    self._logger.info(
                        f"Order block invalidated: {order_block.order_block_type.value}"
                    )
                    continue

                # Update confidence based on age and tests
                order_block.confidence = self._update_order_block_confidence(
                    order_block, current_time
                )

                updated_blocks.append(order_block)

            return updated_blocks

        except Exception as e:
            self._logger.error(f"Error updating order blocks: {e}")
            return order_blocks

    def _determine_break_direction(
        self, break_candle: pd.Series, structure_break: StructureType
    ) -> Direction:
        """Determine the direction of structure break.

        Args:
            break_candle: Candle that broke structure
            structure_break: Type of structure break

        Returns:
            Direction of the break
        """
        # Simplified logic - can be enhanced
        if break_candle["close"] > break_candle["open"]:
            return Direction.BULLISH
        elif break_candle["close"] < break_candle["open"]:
            return Direction.BEARISH
        else:
            return Direction.NEUTRAL

    def _find_order_block_candle(
        self, candles: pd.DataFrame, break_index: int, break_direction: Direction
    ) -> Optional[int]:
        """Find the last opposing candle before structure break.

        Args:
            candles: DataFrame with OHLCV data
            break_index: Index of structure break candle
            break_direction: Direction of structure break

        Returns:
            Index of order block formation candle if found
        """
        try:
            # Look back for opposing candle
            lookback_limit = min(20, break_index)

            for i in range(break_index - 1, break_index - lookback_limit - 1, -1):
                if i < 0:
                    break

                candle = candles.iloc[i]
                candle_direction = (
                    Direction.BULLISH
                    if candle["close"] > candle["open"]
                    else Direction.BEARISH
                )

                # Find opposing direction candle
                if (
                    break_direction == Direction.BULLISH
                    and candle_direction == Direction.BEARISH
                ):
                    return i
                elif (
                    break_direction == Direction.BEARISH
                    and candle_direction == Direction.BULLISH
                ):
                    return i

            return None

        except Exception as e:
            self._logger.error(f"Error finding order block candle: {e}")
            return None

    def _create_order_block(
        self,
        ob_candle: pd.Series,
        ob_index: int,
        break_index: int,
        break_direction: Direction,
        structure_break: StructureType,
        candles: pd.DataFrame,
    ) -> Optional[OrderBlock]:
        """Create order block from formation candle.

        Args:
            ob_candle: Order block formation candle
            ob_index: Index of formation candle
            break_index: Index of structure break
            break_direction: Direction of structure break
            structure_break: Type of structure break
            candles: Full candle DataFrame

        Returns:
            Created OrderBlock if valid
        """
        try:
            # Determine order block type
            if break_direction == Direction.BULLISH:
                ob_type = OrderBlockType.DEMAND_ZONE
                high_price = ob_candle["high"]
                low_price = ob_candle["low"]
            else:
                ob_type = OrderBlockType.SUPPLY_ZONE
                high_price = ob_candle["high"]
                low_price = ob_candle["low"]

            # Calculate confidence
            confidence = self._calculate_order_block_confidence(
                ob_candle, candles, ob_index, break_index
            )

            # Create order block
            order_block = OrderBlock(
                start_time=int(ob_candle["timestamp"]),
                end_time=int(ob_candle["timestamp"]),  # Will be extended later
                high_price=high_price,
                low_price=low_price,
                order_block_type=ob_type,
                direction=break_direction,
                volume=ob_candle["volume"],
                formation_candle_index=ob_index,
                structure_break_index=break_index,
                confidence=confidence,
                creation_structure=structure_break,
            )

            return order_block

        except Exception as e:
            self._logger.error(f"Error creating order block: {e}")
            return None

    def _calculate_order_block_confidence(
        self,
        ob_candle: pd.Series,
        candles: pd.DataFrame,
        ob_index: int,
        break_index: int,
    ) -> float:
        """Calculate order block confidence score.

        Args:
            ob_candle: Order block formation candle
            candles: Full candle DataFrame
            ob_index: Formation candle index
            break_index: Structure break index

        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            scores = []

            # Volume factor
            recent_avg_volume = (
                candles["volume"].iloc[max(0, ob_index - 20) : ob_index].mean()
            )
            volume_ratio = ob_candle["volume"] / max(recent_avg_volume, 1)
            volume_score = min(
                volume_ratio / self._config.volume_confirmation_factor, 1.0
            )
            scores.append(volume_score * self._config.volume_weight)

            # Distance to break factor
            distance_candles = break_index - ob_index
            distance_score = max(0, 1.0 - (distance_candles / 20.0))
            scores.append(distance_score * self._config.structure_strength_weight)

            # Candle body size factor
            body_size = abs(ob_candle["close"] - ob_candle["open"])
            total_size = ob_candle["high"] - ob_candle["low"]
            body_ratio = body_size / max(total_size, 0.0001)
            scores.append(body_ratio * self._config.gap_confirmation_weight)

            # Combine scores
            total_score = sum(scores)
            return min(max(total_score, 0.0), 1.0)

        except Exception as e:
            self._logger.error(f"Error calculating order block confidence: {e}")
            return 0.5

    def _validate_order_block(
        self, order_block: OrderBlock, candles: pd.DataFrame
    ) -> bool:
        """Validate order block meets minimum criteria.

        Args:
            order_block: Order block to validate
            candles: DataFrame with OHLCV data

        Returns:
            True if order block is valid
        """
        try:
            # Check minimum size
            current_price = candles.iloc[-1]["close"]
            min_size = current_price * self._config.order_block_min_size_percentage
            if order_block.size < min_size:
                return False

            # Check confidence threshold
            if order_block.confidence < self._config.min_confidence_threshold:
                return False

            return True

        except Exception as e:
            self._logger.error(f"Error validating order block: {e}")
            return False

    def _check_order_block_invalidation(
        self, order_block: OrderBlock, current_candle: pd.Series
    ) -> bool:
        """Check if order block should be invalidated.

        Args:
            order_block: Order block to check
            current_candle: Current candle data

        Returns:
            True if order block should be invalidated
        """
        try:
            current_price = current_candle["close"]

            # Invalidate if price breaks through order block significantly
            if order_block.direction == Direction.BULLISH:
                # Bullish order block invalidated if price closes below low
                return current_price < order_block.low_price * 0.999
            else:
                # Bearish order block invalidated if price closes above high
                return current_price > order_block.high_price * 1.001

        except Exception as e:
            self._logger.error(f"Error checking order block invalidation: {e}")
            return False

    def _update_order_block_confidence(
        self, order_block: OrderBlock, current_time: int
    ) -> float:
        """Update order block confidence based on age and performance.

        Args:
            order_block: Order block to update
            current_time: Current timestamp

        Returns:
            Updated confidence score
        """
        try:
            base_confidence = order_block.confidence

            # Age factor (confidence decreases over time)
            age_ms = current_time - order_block.start_time
            age_hours = age_ms / (1000 * 3600)
            age_factor = max(0.5, 1.0 - (age_hours / 168))  # Decay over 1 week

            # Test factor (successful tests increase confidence)
            test_factor = 1.0 + (order_block.mitigation_count * 0.1)

            # Combine factors
            updated_confidence = base_confidence * age_factor * test_factor
            return min(max(updated_confidence, 0.0), 1.0)

        except Exception as e:
            self._logger.error(f"Error updating order block confidence: {e}")
            return order_block.confidence


class IctPatternStrategy(BaseStrategy):
    """ICT Order Block pattern detection trading strategy.

    This strategy implements institutional trading concepts including order blocks,
    break of structure, and fair value gaps for algorithmic trading signal generation.
    """

    def __init__(
        self,
        config: StrategyConfiguration,
        event_hub: EventHub,
        ict_config: Optional[IctConfiguration] = None,
    ) -> None:
        """Initialize ICT pattern strategy.

        Args:
            config: Base strategy configuration
            event_hub: Event hub for event communication
            ict_config: ICT-specific configuration parameters

        Raises:
            InvalidStrategyConfigError: If configuration is invalid
        """
        super().__init__(config, event_hub)

        # ICT-specific configuration
        self._ict_config = ict_config or IctConfiguration()

        # Pattern detection components
        self._structure_analyzer = ZigZagStructureAnalyzer(self._ict_config)
        self._order_block_detector = IctOrderBlockDetector(self._ict_config)

        # State tracking
        self._candle_history: List[CandleData] = []
        self._order_blocks: List[OrderBlock] = []
        self._fair_value_gaps: List[FairValueGap] = []
        self._structure_points: List[StructurePoint] = []

        # Performance tracking
        self._patterns_detected = 0
        self._signals_from_order_blocks = 0
        self._successful_mitigations = 0

        self._logger.info("ICT pattern strategy initialized")

    def _generate_signal_implementation(
        self, market_data: MarketData
    ) -> Optional[TradingSignal]:
        """Generate trading signal based on ICT pattern analysis.

        Args:
            market_data: Current market data for analysis

        Returns:
            TradingSignal if pattern conditions are met, None otherwise
        """
        try:
            # Convert market data to candle format for analysis
            current_candle = self._market_data_to_candle(market_data)
            if not current_candle:
                return None

            # Update candle history
            self._update_candle_history(current_candle)

            # Need minimum candles for analysis
            if len(self._candle_history) < self._ict_config.min_candles_for_structure:
                return None

            # Convert to DataFrame for analysis
            candles_df = self._candles_to_dataframe()

            # Analyze market structure
            self._update_structure_analysis(candles_df)

            # Detect new order blocks
            self._detect_new_order_blocks(candles_df)

            # Update existing order blocks
            self._update_existing_order_blocks(current_candle, market_data.timestamp)

            # Generate signal based on order block interactions
            signal = self._analyze_order_block_signals(market_data)

            if signal:
                self._signals_from_order_blocks += 1

            return signal

        except Exception as e:
            self._logger.error(f"Error generating ICT signal: {e}")
            raise SignalGenerationError(f"Failed to generate ICT signal: {e}")

    def _initialize_strategy(self) -> None:
        """Perform ICT strategy-specific initialization."""
        self._logger.info("Initializing ICT pattern strategy components")

        # Validate ICT configuration
        if not isinstance(self._ict_config, IctConfiguration):
            raise InvalidStrategyConfigError("Invalid ICT configuration")

        # Initialize pattern tracking lists
        self._order_blocks = []
        self._fair_value_gaps = []
        self._structure_points = []
        self._candle_history = []

        self._logger.info("ICT pattern strategy components initialized")

    def _cleanup_strategy(self) -> None:
        """Perform ICT strategy-specific cleanup."""
        self._logger.info("Cleaning up ICT pattern strategy")

        # Log final statistics
        stats = {
            "patterns_detected": self._patterns_detected,
            "signals_from_order_blocks": self._signals_from_order_blocks,
            "successful_mitigations": self._successful_mitigations,
            "active_order_blocks": len(
                [ob for ob in self._order_blocks if not ob.invalidated]
            ),
            "active_fair_value_gaps": len(
                [fvg for fvg in self._fair_value_gaps if not fvg.filled]
            ),
        }

        self._logger.info(f"ICT pattern strategy final statistics: {stats}")

    def _market_data_to_candle(self, market_data: MarketData) -> Optional[CandleData]:
        """Convert MarketData to CandleData for analysis.

        Args:
            market_data: Market data to convert

        Returns:
            CandleData if conversion successful, None otherwise
        """
        try:
            # Extract OHLC data from metadata if available
            metadata = market_data.metadata
            if "open_price" in metadata:
                return CandleData(
                    symbol=market_data.symbol,
                    timeframe=Timeframe.ONE_MINUTE,  # Default timeframe
                    open_time=market_data.timestamp,
                    close_time=market_data.timestamp,
                    open_price=metadata["open_price"],
                    high_price=metadata["high_price"],
                    low_price=metadata["low_price"],
                    close_price=market_data.price,
                    volume=market_data.volume,
                    is_closed=metadata.get("is_closed", True),
                )

            # If no OHLC data, create simple candle
            return CandleData(
                symbol=market_data.symbol,
                timeframe=Timeframe.ONE_MINUTE,
                open_time=market_data.timestamp,
                close_time=market_data.timestamp,
                open_price=market_data.price,
                high_price=market_data.price,
                low_price=market_data.price,
                close_price=market_data.price,
                volume=market_data.volume,
                is_closed=True,
            )

        except Exception as e:
            self._logger.error(f"Error converting market data to candle: {e}")
            return None

    def _update_candle_history(self, candle: CandleData) -> None:
        """Update candle history with new candle.

        Args:
            candle: New candle to add
        """
        try:
            self._candle_history.append(candle)

            # Limit history size
            if len(self._candle_history) > self._ict_config.candle_history_limit:
                self._candle_history = self._candle_history[
                    -self._ict_config.candle_history_limit :
                ]

        except Exception as e:
            self._logger.error(f"Error updating candle history: {e}")

    def _candles_to_dataframe(self) -> pd.DataFrame:
        """Convert candle history to pandas DataFrame.

        Returns:
            DataFrame with OHLCV data
        """
        try:
            data = []
            for candle in self._candle_history:
                data.append(
                    {
                        "timestamp": candle.open_time,
                        "open": candle.open_price,
                        "high": candle.high_price,
                        "low": candle.low_price,
                        "close": candle.close_price,
                        "volume": candle.volume,
                    }
                )

            return pd.DataFrame(data)

        except Exception as e:
            self._logger.error(f"Error converting candles to DataFrame: {e}")
            return pd.DataFrame()

    def _update_structure_analysis(self, candles_df: pd.DataFrame) -> None:
        """Update market structure analysis.

        Args:
            candles_df: DataFrame with OHLCV data
        """
        try:
            # Find structure points
            self._structure_points = self._structure_analyzer.find_structure_points(
                candles_df
            )

            # Detect structure breaks
            current_index = len(candles_df) - 1
            structure_break = self._structure_analyzer.detect_structure_break(
                candles_df, self._structure_points, current_index
            )

            if structure_break:
                self._logger.info(f"Structure break detected: {structure_break.value}")

        except Exception as e:
            self._logger.error(f"Error updating structure analysis: {e}")

    def _detect_new_order_blocks(self, candles_df: pd.DataFrame) -> None:
        """Detect new order blocks.

        Args:
            candles_df: DataFrame with OHLCV data
        """
        try:
            current_index = len(candles_df) - 1

            # Detect structure break
            structure_break = self._structure_analyzer.detect_structure_break(
                candles_df, self._structure_points, current_index
            )

            if structure_break:
                # Detect order blocks
                new_order_blocks = self._order_block_detector.detect_order_blocks(
                    candles_df, self._structure_points, structure_break, current_index
                )

                # Add new order blocks
                for order_block in new_order_blocks:
                    self._order_blocks.append(order_block)
                    self._patterns_detected += 1
                    self._logger.info(
                        f"New OB detected: {order_block.order_block_type.value}"
                        f" at {order_block.low_price}-{order_block.high_price}"
                    )

                # Limit order blocks tracked
                if len(self._order_blocks) > self._ict_config.max_order_blocks_tracked:
                    # Remove oldest invalidated order blocks first
                    self._order_blocks = [
                        ob for ob in self._order_blocks if not ob.invalidated
                    ][-self._ict_config.max_order_blocks_tracked :]

        except Exception as e:
            self._logger.error(f"Error detecting new order blocks: {e}")

    def _update_existing_order_blocks(
        self, current_candle: CandleData, current_time: int
    ) -> None:
        """Update existing order blocks with current market data.

        Args:
            current_candle: Current candle data
            current_time: Current timestamp
        """
        try:
            # Convert candle to series for compatibility
            candle_series = pd.Series(
                {
                    "open": current_candle.open_price,
                    "high": current_candle.high_price,
                    "low": current_candle.low_price,
                    "close": current_candle.close_price,
                    "volume": current_candle.volume,
                }
            )

            # Update order blocks
            self._order_blocks = self._order_block_detector.update_order_blocks(
                self._order_blocks, candle_series, current_time
            )

        except Exception as e:
            self._logger.error(f"Error updating existing order blocks: {e}")

    def _analyze_order_block_signals(
        self, market_data: MarketData
    ) -> Optional[TradingSignal]:
        """Analyze order blocks for trading signals.

        Args:
            market_data: Current market data

        Returns:
            TradingSignal if conditions are met, None otherwise
        """
        try:
            current_price = market_data.price
            best_signal = None
            highest_confidence = 0.0

            # Check each active order block
            for order_block in self._order_blocks:
                if order_block.invalidated:
                    continue

                # Check if price is interacting with order block
                if order_block.is_within_zone(current_price):
                    signal = self._create_order_block_signal(order_block, market_data)

                    if signal and signal.confidence > highest_confidence:
                        highest_confidence = signal.confidence
                        best_signal = signal

            return best_signal

        except Exception as e:
            self._logger.error(f"Error analyzing order block signals: {e}")
            return None

    def _create_order_block_signal(
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
            # Determine signal type based on order block direction
            if order_block.direction == Direction.BULLISH:
                signal_type = SignalType.BUY
                target_price = order_block.high_price * 1.01  # 1% above
                stop_loss = order_block.low_price * 0.999  # Below order block
            else:
                signal_type = SignalType.SELL
                target_price = order_block.low_price * 0.99  # 1% below
                stop_loss = order_block.high_price * 1.001  # Above order block

            # Determine signal strength based on order block confidence
            if order_block.confidence >= 0.8:
                strength = SignalStrength.STRONG
            elif order_block.confidence >= 0.7:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK

            # Create reasoning
            reasoning = (
                f"ICT Order Block {order_block.order_block_type.value} interaction. "
                f"Direction: {order_block.direction.value}, "
                f"Confidence: {order_block.confidence:.3f}, "
                f"Tests: {order_block.mitigation_count}"
            )

            # Create signal
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
                    "order_block_type": order_block.order_block_type.value,
                    "order_block_direction": order_block.direction.value,
                    "order_block_confidence": order_block.confidence,
                    "mitigation_count": order_block.mitigation_count,
                    "order_block_size": order_block.size,
                    "formation_index": order_block.formation_candle_index,
                },
            )

            return signal

        except Exception as e:
            self._logger.error(f"Error creating order block signal: {e}")
            return None

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get ICT pattern detection statistics.

        Returns:
            Dictionary containing pattern statistics
        """
        try:
            active_order_blocks = [
                ob for ob in self._order_blocks if not ob.invalidated
            ]
            tested_order_blocks = [ob for ob in active_order_blocks if ob.tested]

            return {
                "total_patterns_detected": self._patterns_detected,
                "signals_from_order_blocks": self._signals_from_order_blocks,
                "successful_mitigations": self._successful_mitigations,
                "active_order_blocks": len(active_order_blocks),
                "tested_order_blocks": len(tested_order_blocks),
                "structure_points": len(self._structure_points),
                "candle_history_length": len(self._candle_history),
                "average_order_block_confidence": (
                    sum(ob.confidence for ob in active_order_blocks)
                    / max(len(active_order_blocks), 1)
                ),
            }

        except Exception as e:
            self._logger.error(f"Error getting pattern statistics: {e}")
            return {}


def create_ict_pattern_strategy(
    name: str,
    symbol: str,
    timeframe: str,
    event_hub: EventHub,
    ict_config: Optional[IctConfiguration] = None,
    **kwargs: Any,
) -> IctPatternStrategy:
    """Factory function to create ICT pattern strategy.

    Args:
        name: Strategy name
        symbol: Trading symbol
        timeframe: Timeframe for strategy operation
        event_hub: Event hub instance
        ict_config: ICT-specific configuration
        **kwargs: Additional strategy configuration parameters

    Returns:
        IctPatternStrategy: Configured ICT pattern strategy instance

    Raises:
        InvalidStrategyConfigError: If configuration parameters are invalid
    """
    # Create base strategy configuration
    strategy_config = StrategyConfiguration(
        name=name,
        symbol=symbol,
        timeframe=timeframe,
        enabled=kwargs.get("enabled", True),
        risk_tolerance=kwargs.get("risk_tolerance", 0.02),
        min_confidence=kwargs.get("min_confidence", 0.6),
        max_position_size=kwargs.get("max_position_size", 1.0),
        use_stop_loss=kwargs.get("use_stop_loss", True),
        use_take_profit=kwargs.get("use_take_profit", True),
        parameters=kwargs.get("parameters", {}),
        metadata=kwargs.get("metadata", {}),
    )

    # Create ICT configuration if not provided
    if ict_config is None:
        ict_config = IctConfiguration()

    return IctPatternStrategy(
        config=strategy_config,
        event_hub=event_hub,
        ict_config=ict_config,
    )
