"""Portfolio management module for the trading bot system.

This module provides comprehensive portfolio tracking including position management,
balance synchronization, P&L calculations, and risk monitoring following SOLID
principles and event-driven architecture.
"""

from .portfolio_manager import (
    IPortfolioManager,
    PortfolioManager,
    PortfolioManagerConfig,
    PortfolioManagerError,
    create_portfolio_manager,
)
from .portfolio_state import (
    BalanceInfo,
    PortfolioHealthStatus,
    PortfolioMetrics,
    PortfolioState,
    PortfolioStateError,
)
from .position import (
    Position,
    PositionError,
    PositionLevel,
    PositionSide,
    PositionStatus,
)

__all__ = [
    # Portfolio Manager
    "IPortfolioManager",
    "PortfolioManager",
    "PortfolioManagerConfig",
    "PortfolioManagerError",
    "create_portfolio_manager",
    # Portfolio State
    "BalanceInfo",
    "PortfolioHealthStatus",
    "PortfolioMetrics",
    "PortfolioState",
    "PortfolioStateError",
    # Position
    "Position",
    "PositionError",
    "PositionLevel",
    "PositionSide",
    "PositionStatus",
]
