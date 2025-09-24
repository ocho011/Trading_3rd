"""Order execution module for the trading bot system.

This module provides order execution components that translate trading signals
into actual exchange orders, monitor order status, and handle order lifecycle events.
"""

from .execution_engine import (
    ExecutionEngine,
    ExecutionEngineConfig,
    ExecutionResult,
    OrderStatus,
    create_execution_engine,
)

__all__ = [
    "ExecutionEngine",
    "ExecutionEngineConfig",
    "ExecutionResult",
    "OrderStatus",
    "create_execution_engine",
]
