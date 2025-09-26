"""
Logging system for trading bot application.

Provides centralized logging functionality with configurable outputs,
log levels, and formatting following SOLID principles.
"""

import logging
import logging.handlers
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class LogFormatterError(Exception):
    """Custom exception for log formatting errors."""


class ILogFormatter(ABC):
    """Interface for log formatting strategies."""

    @abstractmethod
    def get_formatter(self) -> logging.Formatter:
        """
        Get configured log formatter.

        Returns:
            logging.Formatter: Configured formatter instance
        """


class StandardLogFormatter(ILogFormatter):
    """Standard log formatter with timestamp, level, and message."""

    def __init__(self, include_module: bool = True) -> None:
        """
        Initialize standard log formatter.

        Args:
            include_module: Whether to include module name in format
        """
        self._include_module = include_module

    def get_formatter(self) -> logging.Formatter:
        """
        Get standard log formatter.

        Returns:
            logging.Formatter: Standard formatter instance
        """
        if self._include_module:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            format_string = "%(asctime)s - %(levelname)s - %(message)s"

        return logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")


class TradingLogFormatter(ILogFormatter):
    """Trading-specific log formatter with additional context."""

    def get_formatter(self) -> logging.Formatter:
        """
        Get trading-specific log formatter.

        Returns:
            logging.Formatter: Trading formatter instance
        """
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)-20s | "
            "%(funcName)-15s | %(message)s"
        )
        return logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")


class ILogHandler(ABC):
    """Interface for log handler creation strategies."""

    @abstractmethod
    def create_handler(self, formatter: logging.Formatter) -> logging.Handler:
        """
        Create configured log handler.

        Args:
            formatter: Log formatter to use

        Returns:
            logging.Handler: Configured handler instance
        """


class ConsoleLogHandler(ILogHandler):
    """Creates console log handler for stdout output."""

    def __init__(self, level: int = logging.INFO) -> None:
        """
        Initialize console log handler.

        Args:
            level: Logging level for console output
        """
        self._level = level

    def create_handler(self, formatter: logging.Formatter) -> logging.Handler:
        """
        Create console log handler.

        Args:
            formatter: Log formatter to use

        Returns:
            logging.Handler: Console handler instance
        """
        handler = logging.StreamHandler()
        handler.setLevel(self._level)
        handler.setFormatter(formatter)
        return handler


class FileLogHandler(ILogHandler):
    """Creates file log handler with rotation support."""

    def __init__(
        self,
        log_file_path: str,
        level: int = logging.DEBUG,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ) -> None:
        """
        Initialize file log handler.

        Args:
            log_file_path: Path to log file
            level: Logging level for file output
            max_bytes: Maximum file size before rotation
            backup_count: Number of backup files to keep
        """
        self._log_file_path = Path(log_file_path)
        self._level = level
        self._max_bytes = max_bytes
        self._backup_count = backup_count

    def create_handler(self, formatter: logging.Formatter) -> logging.Handler:
        """
        Create rotating file log handler.

        Args:
            formatter: Log formatter to use

        Returns:
            logging.Handler: File handler instance
        """
        # Ensure log directory exists
        self._log_file_path.parent.mkdir(parents=True, exist_ok=True)

        handler = logging.handlers.RotatingFileHandler(
            filename=str(self._log_file_path),
            maxBytes=self._max_bytes,
            backupCount=self._backup_count,
        )
        handler.setLevel(self._level)
        handler.setFormatter(formatter)
        return handler


class LoggerManager:
    """
    Central logger manager for trading bot application.

    Manages logger configuration and provides centralized logging
    functionality following the Single Responsibility Principle.
    """

    def __init__(self, name: str = "trading_bot") -> None:
        """
        Initialize logger manager.

        Args:
            name: Logger name (typically application name)
        """
        self._logger_name = name
        self._logger: Optional[logging.Logger] = None
        self._handlers: Dict[str, logging.Handler] = {}
        self._is_configured = False

    def configure_logger(
        self,
        level: int = logging.INFO,
        formatter: Optional[ILogFormatter] = None,
        handlers: Optional[Dict[str, ILogHandler]] = None,
    ) -> None:
        """
        Configure logger with specified settings.

        Args:
            level: Base logging level
            formatter: Log formatter strategy
            handlers: Dictionary of handler name to handler strategy
        """
        self._logger = logging.getLogger(self._logger_name)
        self._logger.setLevel(level)

        # Clear existing handlers
        self._logger.handlers.clear()
        self._handlers.clear()

        # Use default formatter if none provided
        if formatter is None:
            formatter = StandardLogFormatter()

        log_formatter = formatter.get_formatter()

        # Use default handlers if none provided
        if handlers is None:
            handlers = self._get_default_handlers()

        # Create and add handlers
        for handler_name, handler_strategy in handlers.items():
            handler = handler_strategy.create_handler(log_formatter)
            self._logger.addHandler(handler)
            self._handlers[handler_name] = handler

        self._is_configured = True

    def get_logger(self) -> logging.Logger:
        """
        Get configured logger instance.

        Returns:
            logging.Logger: Configured logger

        Raises:
            RuntimeError: If logger not configured
        """
        if not self._is_configured or self._logger is None:
            raise RuntimeError("Logger not configured. Call configure_logger() first.")

        return self._logger

    def update_log_level(self, level: int) -> None:
        """
        Update logging level for all handlers.

        Args:
            level: New logging level
        """
        if self._logger:
            self._logger.setLevel(level)
            for handler in self._handlers.values():
                handler.setLevel(level)

    def get_handler(self, handler_name: str) -> Optional[logging.Handler]:
        """
        Get specific handler by name.

        Args:
            handler_name: Name of handler to retrieve

        Returns:
            Optional[logging.Handler]: Handler instance if found
        """
        return self._handlers.get(handler_name)

    def _get_default_handlers(self) -> Dict[str, ILogHandler]:
        """
        Get default handler configuration.

        Returns:
            Dict[str, ILogHandler]: Default handlers
        """
        log_dir = Path("logs")
        log_file = (
            log_dir / f"{self._logger_name}_{datetime.now().strftime('%Y%m%d')}.log"
        )

        return {
            "console": ConsoleLogHandler(level=logging.INFO),
            "file": FileLogHandler(str(log_file), level=logging.DEBUG),
        }


def create_trading_logger(
    name: str = "trading_bot", log_level: str = "INFO", log_dir: str = "logs"
) -> logging.Logger:
    """
    Factory function to create pre-configured trading bot logger.

    Args:
        name: Logger name
        log_level: Logging level as string
        log_dir: Directory for log files

    Returns:
        logging.Logger: Configured logger instance
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    level = level_map.get(log_level.upper(), logging.INFO)
    log_path = Path(log_dir)
    log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"

    # Create logger manager
    manager = LoggerManager(name)

    # Configure with trading-specific formatter and handlers
    formatter = TradingLogFormatter()
    handlers = {
        "console": ConsoleLogHandler(level=logging.INFO),
        "file": FileLogHandler(str(log_file), level=level),
    }

    manager.configure_logger(level=level, formatter=formatter, handlers=handlers)
    return manager.get_logger()


def get_module_logger(module_name: str) -> logging.Logger:
    """
    Get logger for specific module.

    Args:
        module_name: Name of the module

    Returns:
        logging.Logger: Module-specific logger
    """
    return logging.getLogger(f"trading_bot.{module_name}")
