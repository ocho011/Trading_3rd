"""
Integration tests for core infrastructure components.

Tests the interaction between ConfigManager and Logger to ensure
they work together correctly in the trading bot application.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trading_bot.core.config_manager import create_config_manager
from trading_bot.core.logger import create_trading_logger


class TestCoreInfrastructureIntegration(unittest.TestCase):
    """Integration tests for core infrastructure components."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_log_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up log files
        log_dir = Path(self.test_log_dir)
        if log_dir.exists():
            for log_file in log_dir.glob("*.log"):
                log_file.unlink()
            log_dir.rmdir()

    @patch.dict(
        os.environ,
        {
            "BINANCE_API_KEY": "test_api_key",
            "BINANCE_SECRET_KEY": "test_secret_key",
            "DISCORD_WEBHOOK_URL": "https://discord.com/webhook/test",
            "LOG_LEVEL": "DEBUG",
            "TRADING_MODE": "paper",
            "MAX_POSITION_SIZE": "0.1",
            "RISK_PERCENTAGE": "2.0",
        },
    )
    def test_config_and_logger_integration(self):
        """Test ConfigManager and Logger working together."""
        # Create and configure ConfigManager
        config_manager = create_config_manager("env")
        config_manager.load_configuration()

        # Get log level from config
        log_level = config_manager.get_config_value("log_level", "INFO")

        # Create logger with config-driven settings
        logger = create_trading_logger(
            name="trading_bot_test",
            log_level=log_level,
            log_dir=self.test_log_dir,
        )

        # Test that logger is properly configured
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "trading_bot_test")

        # Test logging functionality
        logger.info("Test info message")
        logger.debug("Test debug message")
        logger.warning("Test warning message")

        # Verify log file was created
        log_files = list(Path(self.test_log_dir).glob("*.log"))
        self.assertTrue(len(log_files) > 0)

        # Test config values can be retrieved
        api_credentials = config_manager.get_api_credentials()
        self.assertEqual(api_credentials["api_key"], "test_api_key")
        self.assertEqual(api_credentials["secret_key"], "test_secret_key")

        notification_config = config_manager.get_notification_config()
        self.assertEqual(
            notification_config["discord_webhook_url"],
            "https://discord.com/webhook/test",
        )

        trading_config = config_manager.get_trading_config()
        self.assertEqual(trading_config["trading_mode"], "paper")
        self.assertEqual(trading_config["max_position_size"], 0.1)
        self.assertEqual(trading_config["risk_percentage"], 2.0)

        # Log configuration values to demonstrate integration
        logger.info(f"Trading mode: {trading_config['trading_mode']}")
        logger.info(f"Max position size: {trading_config['max_position_size']}")
        logger.info(f"Risk percentage: {trading_config['risk_percentage']}")

    def test_directory_structure_exists(self):
        """Test that the required directory structure exists."""
        base_path = Path(__file__).parent.parent / "trading_bot"

        # Check main directories exist
        self.assertTrue((base_path / "core").exists())
        self.assertTrue((base_path / "market_data").exists())
        self.assertTrue((base_path / "strategies").exists())
        self.assertTrue((base_path / "risk_management").exists())
        self.assertTrue((base_path / "notification").exists())

        # Check __init__.py files exist
        self.assertTrue((base_path / "__init__.py").exists())
        self.assertTrue((base_path / "core" / "__init__.py").exists())
        self.assertTrue((base_path / "market_data" / "__init__.py").exists())
        self.assertTrue((base_path / "strategies" / "__init__.py").exists())
        self.assertTrue((base_path / "risk_management" / "__init__.py").exists())
        self.assertTrue((base_path / "notification" / "__init__.py").exists())

        # Check core modules exist
        self.assertTrue((base_path / "core" / "config_manager.py").exists())
        self.assertTrue((base_path / "core" / "logger.py").exists())


if __name__ == "__main__":
    unittest.main()
