"""
Comprehensive configuration support for Discord webhook reliability features.

Provides centralized configuration management for retry policies, circuit breaker,
message queue, and health monitoring. Supports environment variables, configuration
files, and runtime configuration updates.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from trading_bot.notification.circuit_breaker import CircuitBreakerConfig
from trading_bot.notification.message_queue import QueueConfig
from trading_bot.notification.retry_policies import BackoffType, RetryConfig
from trading_bot.notification.webhook_health import HealthThresholds


class ConfigurationSource(Enum):
    """Configuration source types."""

    ENVIRONMENT = "environment"
    FILE = "file"
    RUNTIME = "runtime"
    DEFAULT = "default"


@dataclass
class WebhookReliabilityConfig:
    """
    Complete configuration for Discord webhook reliability features.

    Attributes:
        enabled: Whether enhanced reliability features are enabled
        webhook_url: Discord webhook URL
        timeout: Request timeout in seconds
        max_rate_limit_wait: Maximum wait time for rate limits
        backup_notification_enabled: Whether backup notifications are enabled
        retry_config: Retry policy configuration
        circuit_breaker_config: Circuit breaker configuration
        queue_config: Message queue configuration
        health_thresholds: Health monitoring thresholds
        logging_config: Logging configuration for webhook components
    """

    # Basic settings
    enabled: bool = True
    webhook_url: str = ""
    timeout: int = 10
    max_rate_limit_wait: float = 300.0
    backup_notification_enabled: bool = True

    # Component configurations
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    circuit_breaker_config: CircuitBreakerConfig = field(
        default_factory=CircuitBreakerConfig
    )
    queue_config: QueueConfig = field(default_factory=QueueConfig)
    health_thresholds: HealthThresholds = field(default_factory=HealthThresholds)

    # Logging configuration
    logging_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "webhook_logger": "trading_bot.notification",
            "enable_file_logging": False,
            "log_file_path": "logs/webhook_reliability.log",
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = asdict(self)

        # Handle enum serialization
        if (
            "retry_config" in config_dict
            and "backoff_type" in config_dict["retry_config"]
        ):
            config_dict["retry_config"][
                "backoff_type"
            ] = self.retry_config.backoff_type.value

        return config_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebhookReliabilityConfig":
        """Create configuration from dictionary."""
        # Handle nested configurations
        config_data = data.copy()

        # Handle retry config
        if "retry_config" in config_data:
            retry_data = config_data["retry_config"]
            if "backoff_type" in retry_data and isinstance(
                retry_data["backoff_type"], str
            ):
                retry_data["backoff_type"] = BackoffType(retry_data["backoff_type"])
            config_data["retry_config"] = RetryConfig(**retry_data)

        # Handle circuit breaker config
        if "circuit_breaker_config" in config_data:
            cb_data = config_data["circuit_breaker_config"]
            config_data["circuit_breaker_config"] = CircuitBreakerConfig(**cb_data)

        # Handle queue config
        if "queue_config" in config_data:
            queue_data = config_data["queue_config"]
            config_data["queue_config"] = QueueConfig(**queue_data)

        # Handle health thresholds
        if "health_thresholds" in config_data:
            health_data = config_data["health_thresholds"]
            config_data["health_thresholds"] = HealthThresholds(**health_data)

        return cls(**config_data)

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.

        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []

        # Validate basic settings
        if not self.webhook_url:
            errors.append("webhook_url is required")

        if self.timeout <= 0:
            errors.append("timeout must be positive")

        if self.max_rate_limit_wait < 0:
            errors.append("max_rate_limit_wait must be non-negative")

        # Validate retry config
        try:
            # This will raise ValueError if invalid
            RetryConfig(**asdict(self.retry_config))
        except ValueError as e:
            errors.append(f"Invalid retry_config: {e}")

        # Validate circuit breaker config
        try:
            CircuitBreakerConfig(**asdict(self.circuit_breaker_config))
        except ValueError as e:
            errors.append(f"Invalid circuit_breaker_config: {e}")

        # Validate queue config
        if self.queue_config.max_size < 0:
            errors.append("queue_config.max_size must be non-negative")

        if self.queue_config.retention_hours < 0:
            errors.append("queue_config.retention_hours must be non-negative")

        # Validate health thresholds
        if not 0 <= self.health_thresholds.success_rate_warning <= 100:
            errors.append("health_thresholds.success_rate_warning must be 0-100")

        if not 0 <= self.health_thresholds.success_rate_critical <= 100:
            errors.append("health_thresholds.success_rate_critical must be 0-100")

        if (
            self.health_thresholds.success_rate_critical
            > self.health_thresholds.success_rate_warning
        ):
            errors.append(
                "health_thresholds.success_rate_critical must be "
                "<= success_rate_warning"
            )

        return errors


class WebhookConfigManager:
    """
    Configuration manager for Discord webhook reliability features.

    Provides centralized configuration loading from multiple sources,
    validation, and runtime updates with change notifications.
    """

    def __init__(self, config_file_path: Optional[str] = None) -> None:
        """
        Initialize configuration manager.

        Args:
            config_file_path: Optional path to configuration file
        """
        self._config_file_path = config_file_path or "webhook_reliability_config.json"
        self._config = WebhookReliabilityConfig()
        self._logger = logging.getLogger(__name__)

        # Configuration change callbacks
        self._change_callbacks: List[callable] = []

    def load_configuration(self) -> WebhookReliabilityConfig:
        """
        Load configuration from all available sources.

        Priority order:
        1. Environment variables (highest)
        2. Configuration file
        3. Default values (lowest)

        Returns:
            WebhookReliabilityConfig: Loaded configuration
        """
        # Start with default configuration
        config = WebhookReliabilityConfig()

        # Load from file if exists
        if Path(self._config_file_path).exists():
            try:
                config = self._load_from_file(self._config_file_path)
                self._logger.info(
                    f"Loaded configuration from file: {self._config_file_path}"
                )
            except Exception as e:
                self._logger.error(f"Failed to load config file: {e}")

        # Override with environment variables
        config = self._load_from_environment(config)

        # Validate configuration
        errors = config.validate()
        if errors:
            error_msg = f"Configuration validation failed: {', '.join(errors)}"
            self._logger.error(error_msg)
            raise ValueError(error_msg)

        self._config = config
        self._logger.info("Configuration loaded successfully")
        return config

    def _load_from_file(self, file_path: str) -> WebhookReliabilityConfig:
        """Load configuration from JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return WebhookReliabilityConfig.from_dict(data)

    def _load_from_environment(
        self, base_config: WebhookReliabilityConfig
    ) -> WebhookReliabilityConfig:
        """Load configuration from environment variables."""
        config_dict = base_config.to_dict()

        # Map of environment variables to config paths
        env_mappings = {
            # Basic settings
            "DISCORD_WEBHOOK_RELIABILITY_ENABLED": ("enabled", bool),
            "DISCORD_WEBHOOK_URL": ("webhook_url", str),
            "DISCORD_WEBHOOK_TIMEOUT": ("timeout", int),
            "DISCORD_WEBHOOK_MAX_RATE_LIMIT_WAIT": ("max_rate_limit_wait", float),
            "DISCORD_BACKUP_NOTIFICATION_ENABLED": (
                "backup_notification_enabled",
                bool,
            ),
            # Retry configuration
            "DISCORD_RETRY_MAX_ATTEMPTS": ("retry_config.max_attempts", int),
            "DISCORD_RETRY_BASE_DELAY": ("retry_config.base_delay", float),
            "DISCORD_RETRY_MAX_DELAY": ("retry_config.max_delay", float),
            "DISCORD_RETRY_BACKOFF_TYPE": ("retry_config.backoff_type", str),
            "DISCORD_RETRY_JITTER_ENABLED": ("retry_config.jitter_enabled", bool),
            # Circuit breaker configuration
            "DISCORD_CB_FAILURE_THRESHOLD": (
                "circuit_breaker_config.failure_threshold",
                int,
            ),
            "DISCORD_CB_SUCCESS_THRESHOLD": (
                "circuit_breaker_config.success_threshold",
                int,
            ),
            "DISCORD_CB_TIMEOUT": ("circuit_breaker_config.timeout", float),
            # Queue configuration
            "DISCORD_QUEUE_MAX_SIZE": ("queue_config.max_size", int),
            "DISCORD_QUEUE_RETENTION_HOURS": ("queue_config.retention_hours", int),
            "DISCORD_QUEUE_PERSISTENCE_ENABLED": (
                "queue_config.persistence_enabled",
                bool,
            ),
            "DISCORD_QUEUE_STORAGE_PATH": ("queue_config.storage_path", str),
            # Health monitoring
            "DISCORD_HEALTH_SUCCESS_RATE_WARNING": (
                "health_thresholds.success_rate_warning",
                float,
            ),
            "DISCORD_HEALTH_SUCCESS_RATE_CRITICAL": (
                "health_thresholds.success_rate_critical",
                float,
            ),
            "DISCORD_HEALTH_RESPONSE_TIME_WARNING": (
                "health_thresholds.response_time_warning",
                float,
            ),
            "DISCORD_HEALTH_CONSECUTIVE_FAILURES_WARNING": (
                "health_thresholds.consecutive_failures_warning",
                int,
            ),
        }

        # Apply environment variable overrides
        for env_var, (config_path, value_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    # Convert value to appropriate type
                    if value_type == bool:
                        converted_value = env_value.lower() in (
                            "true",
                            "1",
                            "yes",
                            "on",
                        )
                    elif (
                        value_type == str and config_path == "retry_config.backoff_type"
                    ):
                        converted_value = BackoffType(env_value)
                    else:
                        converted_value = value_type(env_value)

                    # Set nested configuration value
                    self._set_nested_value(config_dict, config_path, converted_value)
                    self._logger.debug(
                        f"Applied environment override: {env_var}={converted_value}"
                    )

                except (ValueError, TypeError) as e:
                    self._logger.warning(
                        f"Invalid environment variable {env_var}={env_value}: {e}"
                    )

        return WebhookReliabilityConfig.from_dict(config_dict)

    def _set_nested_value(
        self, config_dict: Dict[str, Any], path: str, value: Any
    ) -> None:
        """Set nested dictionary value using dot notation."""
        keys = path.split(".")
        current = config_dict

        # Navigate to the nested dictionary
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the value
        current[keys[-1]] = value

    def save_configuration(
        self, config: Optional[WebhookReliabilityConfig] = None
    ) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration to save (uses current if None)
        """
        config_to_save = config or self._config

        # Validate before saving
        errors = config_to_save.validate()
        if errors:
            raise ValueError(f"Cannot save invalid configuration: {', '.join(errors)}")

        # Ensure directory exists
        Path(self._config_file_path).parent.mkdir(parents=True, exist_ok=True)

        # Save to file
        with open(self._config_file_path, "w") as f:
            json.dump(config_to_save.to_dict(), f, indent=2)

        self._logger.info(f"Configuration saved to: {self._config_file_path}")

    def update_configuration(
        self,
        updates: Dict[str, Any],
        source: ConfigurationSource = ConfigurationSource.RUNTIME,
    ) -> WebhookReliabilityConfig:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates
            source: Source of the configuration update

        Returns:
            WebhookReliabilityConfig: Updated configuration
        """
        # Apply updates to current configuration
        config_dict = self._config.to_dict()

        for key, value in updates.items():
            self._set_nested_value(config_dict, key, value)

        # Create new configuration
        new_config = WebhookReliabilityConfig.from_dict(config_dict)

        # Validate new configuration
        errors = new_config.validate()
        if errors:
            raise ValueError(
                f"Configuration update failed validation: {', '.join(errors)}"
            )

        # Update current configuration
        old_config = self._config
        self._config = new_config

        # Notify change callbacks
        self._notify_config_change(old_config, new_config, source)

        self._logger.info(f"Configuration updated from {source.value}")
        return new_config

    def get_configuration(self) -> WebhookReliabilityConfig:
        """Get current configuration."""
        return self._config

    def register_change_callback(self, callback: callable) -> None:
        """
        Register callback for configuration changes.

        Args:
            callback: Function to call on configuration changes
                     Signature: callback(old_config, new_config, source)
        """
        self._change_callbacks.append(callback)

    def _notify_config_change(
        self,
        old_config: WebhookReliabilityConfig,
        new_config: WebhookReliabilityConfig,
        source: ConfigurationSource,
    ) -> None:
        """Notify registered callbacks of configuration change."""
        for callback in self._change_callbacks:
            try:
                callback(old_config, new_config, source)
            except Exception as e:
                self._logger.error(f"Error in configuration change callback: {e}")

    def create_default_config_file(self, overwrite: bool = False) -> None:
        """
        Create default configuration file.

        Args:
            overwrite: Whether to overwrite existing file
        """
        if Path(self._config_file_path).exists() and not overwrite:
            self._logger.info(
                f"Configuration file already exists: {self._config_file_path}"
            )
            return

        default_config = WebhookReliabilityConfig()
        self.save_configuration(default_config)
        self._logger.info(
            f"Created default configuration file: {self._config_file_path}"
        )

    def get_environment_template(self) -> str:
        """
        Get environment variable template for configuration.

        Returns:
            str: Template with all available environment variables
        """
        template = """# Discord Webhook Reliability Configuration
# Copy to .env file and customize as needed

# Basic Settings
DISCORD_WEBHOOK_RELIABILITY_ENABLED=true
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_URL
DISCORD_WEBHOOK_TIMEOUT=10
DISCORD_WEBHOOK_MAX_RATE_LIMIT_WAIT=300.0
DISCORD_BACKUP_NOTIFICATION_ENABLED=true

# Retry Policy
DISCORD_RETRY_MAX_ATTEMPTS=5
DISCORD_RETRY_BASE_DELAY=1.0
DISCORD_RETRY_MAX_DELAY=30.0
DISCORD_RETRY_BACKOFF_TYPE=exponential
DISCORD_RETRY_JITTER_ENABLED=true

# Circuit Breaker
DISCORD_CB_FAILURE_THRESHOLD=3
DISCORD_CB_SUCCESS_THRESHOLD=2
DISCORD_CB_TIMEOUT=120.0

# Message Queue
DISCORD_QUEUE_MAX_SIZE=1000
DISCORD_QUEUE_RETENTION_HOURS=24
DISCORD_QUEUE_PERSISTENCE_ENABLED=true
DISCORD_QUEUE_STORAGE_PATH=discord_message_queue.db

# Health Monitoring
DISCORD_HEALTH_SUCCESS_RATE_WARNING=85.0
DISCORD_HEALTH_SUCCESS_RATE_CRITICAL=70.0
DISCORD_HEALTH_RESPONSE_TIME_WARNING=5.0
DISCORD_HEALTH_CONSECUTIVE_FAILURES_WARNING=3
"""
        return template


def create_webhook_config_manager(
    config_file_path: Optional[str] = None,
) -> WebhookConfigManager:
    """
    Factory function to create configuration manager.

    Args:
        config_file_path: Optional path to configuration file

    Returns:
        WebhookConfigManager: Configured manager instance
    """
    return WebhookConfigManager(config_file_path)


def load_webhook_config(
    config_file_path: Optional[str] = None,
) -> WebhookReliabilityConfig:
    """
    Convenience function to load webhook configuration.

    Args:
        config_file_path: Optional path to configuration file

    Returns:
        WebhookReliabilityConfig: Loaded configuration
    """
    manager = create_webhook_config_manager(config_file_path)
    return manager.load_configuration()


# Configuration presets for common scenarios
class ConfigPresets:
    """Predefined configuration presets for common use cases."""

    @staticmethod
    def high_reliability() -> WebhookReliabilityConfig:
        """Configuration preset for high reliability requirements."""
        return WebhookReliabilityConfig(
            retry_config=RetryConfig(
                max_attempts=7,
                base_delay=1.0,
                max_delay=60.0,
                backoff_type=BackoffType.EXPONENTIAL,
                jitter_enabled=True,
            ),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=5, success_threshold=3, timeout=180.0
            ),
            queue_config=QueueConfig(
                max_size=2000, retention_hours=48, persistence_enabled=True
            ),
            health_thresholds=HealthThresholds(
                success_rate_warning=90.0,
                success_rate_critical=80.0,
                response_time_warning=3.0,
                consecutive_failures_warning=2,
            ),
        )

    @staticmethod
    def fast_response() -> WebhookReliabilityConfig:
        """Configuration preset for fast response requirements."""
        return WebhookReliabilityConfig(
            timeout=5,
            max_rate_limit_wait=30.0,
            retry_config=RetryConfig(
                max_attempts=3,
                base_delay=0.5,
                max_delay=5.0,
                backoff_type=BackoffType.LINEAR,
            ),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=3, timeout=60.0
            ),
            queue_config=QueueConfig(max_size=500, retention_hours=12),
        )

    @staticmethod
    def development() -> WebhookReliabilityConfig:
        """Configuration preset for development environment."""
        return WebhookReliabilityConfig(
            retry_config=RetryConfig(max_attempts=2, base_delay=1.0, max_delay=5.0),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=5, timeout=30.0
            ),
            queue_config=QueueConfig(
                max_size=100,
                retention_hours=2,
                persistence_enabled=False,  # Use in-memory for dev
            ),
            logging_config={
                "level": "DEBUG",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "webhook_logger": "trading_bot.notification",
                "enable_file_logging": True,
                "log_file_path": "logs/webhook_dev.log",
            },
        )
