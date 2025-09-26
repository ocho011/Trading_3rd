"""
Performance Logger for Trade Execution Tracking.

This module provides the PerformanceLogger class that subscribes to ORDER_FILLED
events and logs trade execution details to CSV files for analysis.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.core.logger import get_module_logger


class PerformanceLoggerError(Exception):
    """Custom exception for performance logger related errors."""


class PerformanceLogger:
    """
    Performance logger for tracking trade execution details.

    This class subscribes to ORDER_FILLED events and logs structured
    trade data to CSV files for later analysis and performance evaluation.

    Follows SOLID principles:
    - Single Responsibility: Only handles trade logging
    - Open/Closed: Extensible for different log formats
    - Dependency Inversion: Depends on EventHub abstraction
    """

    def __init__(
        self,
        event_hub: EventHub,
        log_directory: str = "logs",
        csv_filename: str = "trades.csv",
        portfolio_csv_filename: str = "portfolio_history.csv",
    ) -> None:
        """
        Initialize Performance Logger.

        Args:
            event_hub: EventHub instance for subscribing to events
            log_directory: Directory to store log files
            csv_filename: Name of the CSV file for trade logs
            portfolio_csv_filename: Name of the CSV file for portfolio history logs

        Raises:
            PerformanceLoggerError: If initialization fails
        """
        try:
            self._event_hub = event_hub
            self._logger = get_module_logger(__name__)

            # Setup logging paths
            self._log_directory = Path(log_directory)
            self._csv_filepath = self._log_directory / csv_filename
            self._portfolio_csv_filepath = self._log_directory / portfolio_csv_filename

            # Create log directory if it doesn't exist
            self._log_directory.mkdir(parents=True, exist_ok=True)

            # CSV field names for trade data
            self._csv_fieldnames = [
                "timestamp",
                "symbol",
                "direction",
                "entry_price",
                "exit_price",
                "size",
                "pnl",
                "entry_reason",
                "exit_reason",
                "order_id",
                "execution_time",
                "fees",
                "trading_mode"
            ]

            # CSV field names for portfolio history
            self._portfolio_csv_fieldnames = [
                "timestamp",
                "total_balance",
                "unrealized_pnl",
                "open_positions_count",
                "base_currency",
                "snapshot_type"
            ]

            # Initialize CSV files with headers if they don't exist
            self._initialize_csv_file()
            self._initialize_portfolio_csv_file()

            # Track subscription status
            self._subscribed = False

            self._logger.info(
                f"PerformanceLogger initialized with trade log: {self._csv_filepath} "
                f"and portfolio log: {self._portfolio_csv_filepath}"
            )

        except Exception as e:
            error_msg = f"Failed to initialize PerformanceLogger: {e}"
            self._logger.error(error_msg) if hasattr(self, "_logger") else None
            raise PerformanceLoggerError(error_msg) from e

    def _initialize_csv_file(self) -> None:
        """
        Initialize CSV file with headers if it doesn't exist.

        Creates the CSV file with proper headers for trade logging.
        """
        try:
            if not self._csv_filepath.exists():
                with open(self._csv_filepath, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self._csv_fieldnames)
                    writer.writeheader()

                self._logger.info(f"Created new trade log file: {self._csv_filepath}")
            else:
                self._logger.info(f"Using existing trade log file: {self._csv_filepath}")

        except Exception as e:
            error_msg = f"Failed to initialize CSV file: {e}"
            self._logger.error(error_msg)
            raise PerformanceLoggerError(error_msg) from e

    def _initialize_portfolio_csv_file(self) -> None:
        """
        Initialize portfolio history CSV file with headers if it doesn't exist.

        Creates the CSV file with proper headers for portfolio snapshot logging.
        """
        try:
            if not self._portfolio_csv_filepath.exists():
                with open(
                    self._portfolio_csv_filepath, "w", newline="", encoding="utf-8"
                ) as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self._portfolio_csv_fieldnames)
                    writer.writeheader()

                self._logger.info(
                    f"Created new portfolio history file: {self._portfolio_csv_filepath}"
                )
            else:
                self._logger.info(
                    f"Using existing portfolio history file: {self._portfolio_csv_filepath}"
                )

        except Exception as e:
            error_msg = f"Failed to initialize portfolio CSV file: {e}"
            self._logger.error(error_msg)
            raise PerformanceLoggerError(error_msg) from e

    def subscribe_to_events(self) -> None:
        """
        Subscribe to ORDER_FILLED and PORTFOLIO_SNAPSHOT events from the EventHub.

        Sets up event subscription for trade execution and portfolio snapshot logging.

        Raises:
            PerformanceLoggerError: If subscription fails
        """
        try:
            if self._subscribed:
                self._logger.warning("PerformanceLogger already subscribed to events")
                return

            # Subscribe to ORDER_FILLED events
            self._event_hub.subscribe(EventType.ORDER_FILLED, self._handle_order_filled)

            # Subscribe to PORTFOLIO_SNAPSHOT events
            self._event_hub.subscribe(EventType.PORTFOLIO_SNAPSHOT, self._handle_portfolio_snapshot)

            self._subscribed = True
            self._logger.info(
                "PerformanceLogger subscribed to ORDER_FILLED and PORTFOLIO_SNAPSHOT events"
            )

        except Exception as e:
            error_msg = f"Failed to subscribe to events: {e}"
            self._logger.error(error_msg)
            raise PerformanceLoggerError(error_msg) from e

    def _handle_order_filled(self, event_data: Dict[str, Any]) -> None:
        """
        Handle ORDER_FILLED events and log trade data.

        Args:
            event_data: Event payload containing order execution details

        Note:
            This method is called by the EventHub when an ORDER_FILLED event occurs.
        """
        try:
            self._logger.debug(f"Processing ORDER_FILLED event: {event_data}")

            # Extract trade data from event payload
            trade_record = self._extract_trade_data(event_data)

            # Log trade to CSV file
            self._log_trade_to_csv(trade_record)

            self._logger.info(
                f"Logged trade: {trade_record['symbol']} {trade_record['direction']} "
                f"Size: {trade_record['size']} Price: {trade_record['exit_price']}"
            )

        except Exception as e:
            self._logger.error(f"Failed to handle ORDER_FILLED event: {e}")
            # Don't re-raise to avoid disrupting other event handlers

    def _handle_portfolio_snapshot(self, event_data: Dict[str, Any]) -> None:
        """
        Handle PORTFOLIO_SNAPSHOT events and log portfolio data.

        Args:
            event_data: Event payload containing portfolio snapshot details

        Note:
            This method is called by the EventHub when a PORTFOLIO_SNAPSHOT event occurs.
        """
        try:
            self._logger.debug(f"Processing PORTFOLIO_SNAPSHOT event: {event_data}")

            # Extract portfolio snapshot data from event payload
            portfolio_record = self._extract_portfolio_data(event_data)

            # Log portfolio snapshot to CSV file
            self._log_portfolio_to_csv(portfolio_record)

            self._logger.info(
                f"Logged portfolio snapshot: Total Balance: {portfolio_record['total_balance']} "
                f"Unrealized P&L: {portfolio_record['unrealized_pnl']} "
                f"Open Positions: {portfolio_record['open_positions_count']}"
            )

        except Exception as e:
            self._logger.error(f"Failed to handle PORTFOLIO_SNAPSHOT event: {e}")
            # Don't re-raise to avoid disrupting other event handlers

    def _extract_trade_data(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and structure trade data from ORDER_FILLED event.

        Args:
            event_data: Raw event data from ORDER_FILLED event

        Returns:
            Dict[str, Any]: Structured trade record for CSV logging

        Raises:
            PerformanceLoggerError: If required data is missing
        """
        try:
            # Current timestamp
            current_timestamp = datetime.now().isoformat()

            # Extract order details with defaults
            order_id = event_data.get("order_id", "unknown")
            symbol = event_data.get("symbol", "UNKNOWN")
            side = event_data.get("side", "UNKNOWN")

            # Convert side to direction
            direction = "BUY" if side.upper() in ["BUY", "LONG"] else "SELL"

            # Extract prices and quantities
            executed_qty = float(event_data.get("executed_qty", 0))
            price = float(event_data.get("price", 0))

            # For filled orders, entry and exit price are the same
            # In a real system, you'd track position lifecycle
            entry_price = price
            exit_price = price

            # Calculate basic metrics
            fees = float(event_data.get("commission", 0))

            # PnL calculation (simplified for individual orders)
            # In reality, this would be calculated at position close
            pnl = 0.0  # Will be calculated when position closes

            # Extract additional details
            entry_reason = event_data.get("entry_reason", "signal_generated")
            exit_reason = event_data.get("exit_reason", "order_filled")
            execution_time = event_data.get("transact_time", current_timestamp)
            trading_mode = event_data.get("trading_mode", "paper")

            trade_record = {
                "timestamp": current_timestamp,
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "size": executed_qty,
                "pnl": pnl,
                "entry_reason": entry_reason,
                "exit_reason": exit_reason,
                "order_id": order_id,
                "execution_time": execution_time,
                "fees": fees,
                "trading_mode": trading_mode
            }

            return trade_record

        except (KeyError, ValueError, TypeError) as e:
            error_msg = f"Failed to extract trade data from event: {e}"
            self._logger.error(error_msg)
            raise PerformanceLoggerError(error_msg) from e

    def _log_trade_to_csv(self, trade_record: Dict[str, Any]) -> None:
        """
        Log trade record to CSV file.

        Args:
            trade_record: Structured trade data to log

        Raises:
            PerformanceLoggerError: If logging fails
        """
        try:
            # Append trade record to CSV file
            with open(self._csv_filepath, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self._csv_fieldnames)
                writer.writerow(trade_record)

            self._logger.debug(f"Trade record written to {self._csv_filepath}")

        except Exception as e:
            error_msg = f"Failed to write trade record to CSV: {e}"
            self._logger.error(error_msg)
            raise PerformanceLoggerError(error_msg) from e

    def _extract_portfolio_data(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and structure portfolio data from PORTFOLIO_SNAPSHOT event.

        Args:
            event_data: Raw event data from PORTFOLIO_SNAPSHOT event

        Returns:
            Dict[str, Any]: Structured portfolio record for CSV logging

        Raises:
            PerformanceLoggerError: If required data is missing
        """
        try:
            # Convert timestamp to human-readable format
            timestamp_ms = event_data.get("timestamp", int(datetime.now().timestamp() * 1000))
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000).isoformat()

            # Extract portfolio metrics with defaults
            total_balance = float(event_data.get("total_balance", 0))
            unrealized_pnl = float(event_data.get("unrealized_pnl", 0))
            open_positions_count = int(event_data.get("open_positions_count", 0))
            base_currency = event_data.get("base_currency", "USDT")
            snapshot_type = event_data.get("snapshot_type", "hourly")

            portfolio_record = {
                "timestamp": timestamp,
                "total_balance": total_balance,
                "unrealized_pnl": unrealized_pnl,
                "open_positions_count": open_positions_count,
                "base_currency": base_currency,
                "snapshot_type": snapshot_type
            }

            return portfolio_record

        except (KeyError, ValueError, TypeError) as e:
            error_msg = f"Failed to extract portfolio data from event: {e}"
            self._logger.error(error_msg)
            raise PerformanceLoggerError(error_msg) from e

    def _log_portfolio_to_csv(self, portfolio_record: Dict[str, Any]) -> None:
        """
        Log portfolio record to CSV file.

        Args:
            portfolio_record: Structured portfolio data to log

        Raises:
            PerformanceLoggerError: If logging fails
        """
        try:
            # Append portfolio record to CSV file
            with open(self._portfolio_csv_filepath, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self._portfolio_csv_fieldnames)
                writer.writerow(portfolio_record)

            self._logger.debug(f"Portfolio record written to {self._portfolio_csv_filepath}")

        except Exception as e:
            error_msg = f"Failed to write portfolio record to CSV: {e}"
            self._logger.error(error_msg)
            raise PerformanceLoggerError(error_msg) from e

    def get_csv_filepath(self) -> Path:
        """
        Get the path to the CSV log file.

        Returns:
            Path: Path to the trades CSV file
        """
        return self._csv_filepath

    def get_portfolio_csv_filepath(self) -> Path:
        """
        Get the path to the portfolio history CSV file.

        Returns:
            Path: Path to the portfolio history CSV file
        """
        return self._portfolio_csv_filepath

    def is_subscribed(self) -> bool:
        """
        Check if logger is subscribed to events.

        Returns:
            bool: True if subscribed to ORDER_FILLED events
        """
        return self._subscribed

    def get_trade_count(self) -> int:
        """
        Get the number of trades logged.

        Returns:
            int: Number of trade records in CSV file (excluding header)

        Raises:
            PerformanceLoggerError: If file cannot be read
        """
        try:
            if not self._csv_filepath.exists():
                return 0

            with open(self._csv_filepath, "r", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                # Count rows minus header
                row_count = sum(1 for _ in reader) - 1
                return max(0, row_count)  # Ensure non-negative

        except Exception as e:
            error_msg = f"Failed to count trades in CSV file: {e}"
            self._logger.error(error_msg)
            raise PerformanceLoggerError(error_msg) from e

    def shutdown(self) -> None:
        """
        Gracefully shutdown the performance logger.

        Cleans up resources and logs final statistics.
        """
        try:
            if self._subscribed:
                # Note: EventHub doesn't provide unsubscribe method
                # This would ideally unsubscribe from events
                self._subscribed = False

            # Log final statistics
            trade_count = self.get_trade_count()
            self._logger.info(
                f"PerformanceLogger shutdown complete. "
                f"Total trades logged: {trade_count}"
            )

        except Exception as e:
            self._logger.error(f"Error during PerformanceLogger shutdown: {e}")


def create_performance_logger(
    event_hub: EventHub,
    log_directory: str = "logs",
    csv_filename: str = "trades.csv",
    portfolio_csv_filename: str = "portfolio_history.csv"
) -> PerformanceLogger:
    """
    Factory function to create and configure a PerformanceLogger.

    Args:
        event_hub: EventHub instance for event subscription
        log_directory: Directory for log files
        csv_filename: Name of the CSV trade log file
        portfolio_csv_filename: Name of the CSV portfolio history file

    Returns:
        PerformanceLogger: Configured performance logger instance

    Raises:
        PerformanceLoggerError: If creation fails
    """
    logger = PerformanceLogger(
        event_hub=event_hub,
        log_directory=log_directory,
        csv_filename=csv_filename,
        portfolio_csv_filename=portfolio_csv_filename
    )

    # Automatically subscribe to events
    logger.subscribe_to_events()

    return logger
