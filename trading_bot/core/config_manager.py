"""
Configuration manager for trading bot application.

Handles loading and managing configuration settings from environment variables
and configuration files following SOLID principles and dependency injection.
"""

import configparser
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""


class IConfigLoader:
    """Interface for configuration loading strategies."""

    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from source.

        Returns:
            Dict[str, Any]: Configuration dictionary

        Raises:
            ConfigurationError: If configuration loading fails
        """
        raise NotImplementedError


class EnvConfigLoader(IConfigLoader):
    """Loads configuration from environment variables and .env files."""

    def __init__(self, env_file_path: Optional[str] = None) -> None:
        """
        Initialize environment configuration loader.

        Args:
            env_file_path: Optional path to .env file
        """
        self._env_file_path = env_file_path

    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.

        Returns:
            Dict[str, Any]: Configuration from environment

        Raises:
            ConfigurationError: If required environment variables are missing
        """
        if self._env_file_path:
            load_dotenv(self._env_file_path)
        else:
            load_dotenv()

        return {
            # Legacy keys for backward compatibility
            "binance_api_key": os.getenv("BINANCE_API_KEY"),
            "binance_secret_key": os.getenv("BINANCE_SECRET_KEY"),

            # Mainnet API credentials
            "binance_mainnet_api_key": os.getenv("BINANCE_MAINNET_API_KEY"),
            "binance_mainnet_secret_key": os.getenv("BINANCE_MAINNET_SECRET_KEY"),

            # Testnet API credentials
            "binance_testnet_api_key": os.getenv("BINANCE_TESTNET_API_KEY"),
            "binance_testnet_secret_key": os.getenv("BINANCE_TESTNET_SECRET_KEY"),

            # Network selection
            "binance_testnet": os.getenv("BINANCE_TESTNET", "true"),
            "binance_mainnet_base_url": os.getenv("BINANCE_MAINNET_BASE_URL", "https://api.binance.com/api"),
            "binance_testnet_base_url": os.getenv("BINANCE_TESTNET_BASE_URL", "https://testnet.binance.vision/api"),

            # Other configuration
            "discord_webhook_url": os.getenv("DISCORD_WEBHOOK_URL"),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "trading_mode": os.getenv("TRADING_MODE", "paper"),
            "max_position_size": float(os.getenv("MAX_POSITION_SIZE", "1000.0")),
            "risk_percentage": float(os.getenv("RISK_PERCENTAGE", "2.0")),
        }


class IniConfigLoader(IConfigLoader):
    """Loads configuration from INI configuration files."""

    def __init__(self, config_file_path: str) -> None:
        """
        Initialize INI configuration loader.

        Args:
            config_file_path: Path to configuration INI file
        """
        self._config_file_path = Path(config_file_path)

    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from INI file.

        Returns:
            Dict[str, Any]: Configuration from INI file

        Raises:
            ConfigurationError: If config file is missing or invalid
        """
        if not self._config_file_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {self._config_file_path}"
            )

        config = configparser.ConfigParser()
        config.read(self._config_file_path)

        return {
            "binance_api_key": config.get("api", "binance_api_key", fallback=None),
            "binance_secret_key": config.get(
                "api", "binance_secret_key", fallback=None
            ),
            "discord_webhook_url": config.get(
                "notification", "discord_webhook_url", fallback=None
            ),
            "log_level": config.get("logging", "log_level", fallback="INFO"),
            "trading_mode": config.get("trading", "trading_mode", fallback="paper"),
            "max_position_size": config.getfloat(
                "trading", "max_position_size", fallback=0.1
            ),
            "risk_percentage": config.getfloat(
                "trading", "risk_percentage", fallback=2.0
            ),
        }


class ConfigManager:
    """
    Central configuration manager for trading bot application.

    Manages application settings loaded from various sources following
    the Dependency Inversion Principle.
    """

    def __init__(self, config_loader: IConfigLoader) -> None:
        """
        Initialize configuration manager with a config loader.

        Args:
            config_loader: Implementation of IConfigLoader interface
        """
        self._config_loader = config_loader
        self._config: Dict[str, Any] = {}
        self._is_loaded = False

    def load_configuration(self) -> None:
        """
        Load configuration using the injected config loader.

        Raises:
            ConfigurationError: If configuration loading fails
        """
        try:
            self._config = self._config_loader.load_config()
            self._is_loaded = True
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Any: Configuration value

        Raises:
            ConfigurationError: If configuration not loaded
        """
        if not self._is_loaded:
            raise ConfigurationError(
                "Configuration not loaded. Call load_configuration() first."
            )

        return self._config.get(key, default)

    def get_api_credentials(self) -> Dict[str, str]:
        """
        Get API credentials for trading exchanges.

        Dynamically selects between mainnet and testnet credentials
        based on BINANCE_TESTNET configuration.

        Returns:
            Dict[str, str]: API credentials with network info
                - api_key: The API key
                - secret_key: The secret key
                - network: 'testnet' or 'mainnet'
                - base_url: The appropriate base URL

        Raises:
            ConfigurationError: If credentials are missing
        """
        # Check if we should use testnet or mainnet
        is_testnet = self.get_config_value("binance_testnet", "true").lower() == "true"

        if is_testnet:
            # Use testnet credentials
            api_key = self.get_config_value("binance_testnet_api_key")
            secret_key = self.get_config_value("binance_testnet_secret_key")
            base_url = self.get_config_value("binance_testnet_base_url", "https://testnet.binance.vision/api")
            network = "testnet"
        else:
            # Use mainnet credentials
            api_key = self.get_config_value("binance_mainnet_api_key")
            secret_key = self.get_config_value("binance_mainnet_secret_key")
            base_url = self.get_config_value("binance_mainnet_base_url", "https://api.binance.com/api")
            network = "mainnet"

        # Fallback to legacy environment variables if new ones are not set
        if not api_key or not secret_key:
            api_key = self.get_config_value("binance_api_key")
            secret_key = self.get_config_value("binance_secret_key")
            network = "legacy"
            base_url = "https://testnet.binance.vision/api" if is_testnet else "https://api.binance.com/api"

        if not api_key or not secret_key:
            raise ConfigurationError(
                f"Binance {network} API credentials are missing. "
                f"Please set BINANCE_{network.upper()}_API_KEY and BINANCE_{network.upper()}_SECRET_KEY"
            )

        return {
            "api_key": api_key,
            "secret_key": secret_key,
            "network": network,
            "base_url": base_url
        }

    def get_notification_config(self) -> Dict[str, str]:
        """
        Get notification configuration.

        Returns:
            Dict[str, str]: Notification settings
        """
        return {
            "discord_webhook_url": self.get_config_value("discord_webhook_url", ""),
        }

    def get_trading_config(self) -> Dict[str, Any]:
        """
        Get trading configuration parameters.

        Returns:
            Dict[str, Any]: Trading configuration
        """
        return {
            "trading_mode": self.get_config_value("trading_mode", "paper"),
            "max_position_size": self.get_config_value("max_position_size", 0.1),
            "risk_percentage": self.get_config_value("risk_percentage", 2.0),
        }


def create_config_manager(config_source: str = "env") -> ConfigManager:
    """
    Factory function to create ConfigManager with appropriate loader.

    Args:
        config_source: Configuration source type ('env' or 'ini')

    Returns:
        ConfigManager: Configured instance

    Raises:
        ValueError: If config_source is invalid
    """
    if config_source == "env":
        loader = EnvConfigLoader()
    elif config_source == "ini":
        loader = IniConfigLoader("config.ini")
    else:
        raise ValueError(f"Unsupported config source: {config_source}")

    return ConfigManager(loader)
