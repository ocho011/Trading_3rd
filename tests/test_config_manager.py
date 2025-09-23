"""
Unit tests for ConfigManager functionality.

Tests configuration loading from environment variables and INI files,
validation, and error handling.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from trading_bot.core.config_manager import (
    ConfigManager,
    ConfigurationError,
    EnvConfigLoader,
    IniConfigLoader,
    create_config_manager,
)


class TestEnvConfigLoader(unittest.TestCase):
    """Test cases for EnvConfigLoader."""

    @patch.dict(
        os.environ,
        {
            "BINANCE_API_KEY": "test_api_key",
            "BINANCE_SECRET_KEY": "test_secret_key",
            "DISCORD_WEBHOOK_URL": "test_webhook_url",
            "LOG_LEVEL": "DEBUG",
            "TRADING_MODE": "live",
            "MAX_POSITION_SIZE": "0.2",
            "RISK_PERCENTAGE": "3.0",
        },
    )
    def test_load_config_from_env(self):
        """Test loading configuration from environment variables."""
        loader = EnvConfigLoader()
        config = loader.load_config()

        self.assertEqual(config["binance_api_key"], "test_api_key")
        self.assertEqual(config["binance_secret_key"], "test_secret_key")
        self.assertEqual(config["discord_webhook_url"], "test_webhook_url")
        self.assertEqual(config["log_level"], "DEBUG")
        self.assertEqual(config["trading_mode"], "live")
        self.assertEqual(config["max_position_size"], 0.2)
        self.assertEqual(config["risk_percentage"], 3.0)


class TestIniConfigLoader(unittest.TestCase):
    """Test cases for IniConfigLoader."""

    def test_load_config_from_ini_file(self):
        """Test loading configuration from INI file."""
        # Create temporary INI file
        ini_content = """
[api]
binance_api_key = test_api_key
binance_secret_key = test_secret_key

[notification]
discord_webhook_url = test_webhook_url

[logging]
log_level = DEBUG

[trading]
trading_mode = live
max_position_size = 0.2
risk_percentage = 3.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            f.write(ini_content)
            f.flush()

            try:
                loader = IniConfigLoader(f.name)
                config = loader.load_config()

                self.assertEqual(config["binance_api_key"], "test_api_key")
                self.assertEqual(config["binance_secret_key"], "test_secret_key")
                self.assertEqual(config["discord_webhook_url"], "test_webhook_url")
                self.assertEqual(config["log_level"], "DEBUG")
                self.assertEqual(config["trading_mode"], "live")
                self.assertEqual(config["max_position_size"], 0.2)
                self.assertEqual(config["risk_percentage"], 3.0)
            finally:
                os.unlink(f.name)

    def test_ini_file_not_found(self):
        """Test error handling when INI file is not found."""
        loader = IniConfigLoader("non_existent_file.ini")

        with self.assertRaises(ConfigurationError):
            loader.load_config()


class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_loader = MagicMock()
        self.config_manager = ConfigManager(self.mock_loader)

    def test_load_configuration_success(self):
        """Test successful configuration loading."""
        mock_config = {
            "binance_api_key": "test_key",
            "binance_secret_key": "test_secret",
            "discord_webhook_url": "test_url",
        }
        self.mock_loader.load_config.return_value = mock_config

        self.config_manager.load_configuration()

        self.mock_loader.load_config.assert_called_once()
        self.assertTrue(self.config_manager._is_loaded)

    def test_get_config_value_before_loading(self):
        """Test error when getting config value before loading."""
        with self.assertRaises(ConfigurationError):
            self.config_manager.get_config_value("test_key")

    def test_get_api_credentials(self):
        """Test getting API credentials."""
        mock_config = {
            "binance_api_key": "test_key",
            "binance_secret_key": "test_secret",
        }
        self.mock_loader.load_config.return_value = mock_config
        self.config_manager.load_configuration()

        credentials = self.config_manager.get_api_credentials()

        self.assertEqual(credentials["api_key"], "test_key")
        self.assertEqual(credentials["secret_key"], "test_secret")

    def test_missing_api_credentials(self):
        """Test error when API credentials are missing."""
        mock_config = {"other_key": "value"}
        self.mock_loader.load_config.return_value = mock_config
        self.config_manager.load_configuration()

        with self.assertRaises(ConfigurationError):
            self.config_manager.get_api_credentials()


class TestCreateConfigManager(unittest.TestCase):
    """Test cases for create_config_manager factory function."""

    def test_create_env_config_manager(self):
        """Test creating config manager with env loader."""
        manager = create_config_manager("env")
        self.assertIsInstance(manager, ConfigManager)

    def test_create_ini_config_manager(self):
        """Test creating config manager with ini loader."""
        manager = create_config_manager("ini")
        self.assertIsInstance(manager, ConfigManager)

    def test_invalid_config_source(self):
        """Test error with invalid config source."""
        with self.assertRaises(ValueError):
            create_config_manager("invalid")


if __name__ == "__main__":
    unittest.main()
