"""
Unit tests for Logger functionality.

Tests logger configuration, handler creation, and logging output
to both console and file destinations.
"""

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from trading_bot.core.logger import (ConsoleLogHandler, FileLogHandler,
                                     LoggerManager, StandardLogFormatter,
                                     TradingLogFormatter,
                                     create_trading_logger, get_module_logger)


class TestStandardLogFormatter(unittest.TestCase):
    """Test cases for StandardLogFormatter."""

    def test_formatter_with_module(self):
        """Test formatter with module name included."""
        formatter_strategy = StandardLogFormatter(include_module=True)
        formatter = formatter_strategy.get_formatter()

        self.assertIsInstance(formatter, logging.Formatter)
        self.assertIn("%(name)s", formatter._fmt)

    def test_formatter_without_module(self):
        """Test formatter without module name."""
        formatter_strategy = StandardLogFormatter(include_module=False)
        formatter = formatter_strategy.get_formatter()

        self.assertIsInstance(formatter, logging.Formatter)
        self.assertNotIn("%(name)s", formatter._fmt)


class TestTradingLogFormatter(unittest.TestCase):
    """Test cases for TradingLogFormatter."""

    def test_trading_formatter(self):
        """Test trading-specific formatter."""
        formatter_strategy = TradingLogFormatter()
        formatter = formatter_strategy.get_formatter()

        self.assertIsInstance(formatter, logging.Formatter)
        self.assertIn("%(funcName)", formatter._fmt)
        self.assertIn("%(levelname)", formatter._fmt)


class TestConsoleLogHandler(unittest.TestCase):
    """Test cases for ConsoleLogHandler."""

    def test_create_console_handler(self):
        """Test console handler creation."""
        handler_strategy = ConsoleLogHandler(level=logging.DEBUG)
        formatter = StandardLogFormatter().get_formatter()

        handler = handler_strategy.create_handler(formatter)

        self.assertIsInstance(handler, logging.StreamHandler)
        self.assertEqual(handler.level, logging.DEBUG)


class TestFileLogHandler(unittest.TestCase):
    """Test cases for FileLogHandler."""

    def test_create_file_handler(self):
        """Test file handler creation."""
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            try:
                handler_strategy = FileLogHandler(
                    f.name, level=logging.INFO, max_bytes=1024, backup_count=3
                )
                formatter = StandardLogFormatter().get_formatter()

                handler = handler_strategy.create_handler(formatter)

                self.assertIsInstance(
                    handler, logging.handlers.RotatingFileHandler
                )
                self.assertEqual(handler.level, logging.INFO)
                self.assertEqual(handler.maxBytes, 1024)
                self.assertEqual(handler.backupCount, 3)
            finally:
                Path(f.name).unlink(missing_ok=True)


class TestLoggerManager(unittest.TestCase):
    """Test cases for LoggerManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger_manager = LoggerManager("test_logger")

    def test_configure_logger(self):
        """Test logger configuration."""
        formatter = StandardLogFormatter()
        handlers = {"console": ConsoleLogHandler(level=logging.INFO)}

        self.logger_manager.configure_logger(
            level=logging.DEBUG, formatter=formatter, handlers=handlers
        )

        logger = self.logger_manager.get_logger()
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.level, logging.DEBUG)
        self.assertEqual(len(logger.handlers), 1)

    def test_get_logger_before_configuration(self):
        """Test error when getting logger before configuration."""
        with self.assertRaises(RuntimeError):
            self.logger_manager.get_logger()

    def test_update_log_level(self):
        """Test updating log level."""
        self.logger_manager.configure_logger(level=logging.INFO)

        self.logger_manager.update_log_level(logging.ERROR)

        logger = self.logger_manager.get_logger()
        self.assertEqual(logger.level, logging.ERROR)

    def test_get_handler(self):
        """Test getting specific handler."""
        handlers = {
            "console": ConsoleLogHandler(level=logging.INFO),
            "file": FileLogHandler("test.log", level=logging.DEBUG),
        }

        self.logger_manager.configure_logger(handlers=handlers)

        console_handler = self.logger_manager.get_handler("console")
        self.assertIsNotNone(console_handler)

        nonexistent_handler = self.logger_manager.get_handler("nonexistent")
        self.assertIsNone(nonexistent_handler)


class TestCreateTradingLogger(unittest.TestCase):
    """Test cases for create_trading_logger factory function."""

    @patch("trading_bot.core.logger.LoggerManager")
    def test_create_trading_logger(self, mock_manager_class):
        """Test trading logger creation."""
        mock_manager = MagicMock()
        mock_logger = MagicMock()
        mock_manager.get_logger.return_value = mock_logger
        mock_manager_class.return_value = mock_manager

        logger = create_trading_logger(
            name="test_bot", log_level="DEBUG", log_dir="test_logs"
        )

        mock_manager_class.assert_called_once_with("test_bot")
        mock_manager.configure_logger.assert_called_once()
        mock_manager.get_logger.assert_called_once()
        self.assertEqual(logger, mock_logger)

    def test_create_trading_logger_invalid_level(self):
        """Test trading logger creation with invalid log level."""
        # Should default to INFO for invalid levels
        logger = create_trading_logger(log_level="INVALID")
        self.assertIsInstance(logger, logging.Logger)


class TestGetModuleLogger(unittest.TestCase):
    """Test cases for get_module_logger function."""

    def test_get_module_logger(self):
        """Test getting module-specific logger."""
        logger = get_module_logger("market_data")

        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "trading_bot.market_data")


if __name__ == "__main__":
    unittest.main()
