"""
Unit tests for webhook configuration management.

Tests configuration loading, validation, environment variable handling,
and configuration updates for Discord webhook reliability features.
"""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from trading_bot.notification.circuit_breaker import CircuitBreakerConfig
from trading_bot.notification.message_queue import QueueConfig
from trading_bot.notification.retry_policies import BackoffType, RetryConfig
from trading_bot.notification.webhook_config import (
    ConfigPresets,
    ConfigurationSource,
    WebhookConfigManager,
    WebhookReliabilityConfig,
    create_webhook_config_manager,
    load_webhook_config,
)
from trading_bot.notification.webhook_health import HealthThresholds


class TestWebhookReliabilityConfig:
    """Test cases for WebhookReliabilityConfig."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = WebhookReliabilityConfig()

        assert config.enabled is True
        assert config.webhook_url == ""
        assert config.timeout == 10
        assert config.max_rate_limit_wait == 300.0
        assert config.backup_notification_enabled is True

        # Check nested configurations exist
        assert isinstance(config.retry_config, RetryConfig)
        assert isinstance(config.circuit_breaker_config, CircuitBreakerConfig)
        assert isinstance(config.queue_config, QueueConfig)
        assert isinstance(config.health_thresholds, HealthThresholds)

    def test_to_dict_conversion(self):
        """Test configuration to dictionary conversion."""
        config = WebhookReliabilityConfig(
            webhook_url="https://discord.com/api/webhooks/test", timeout=15
        )

        config_dict = config.to_dict()

        assert config_dict["webhook_url"] == "https://discord.com/api/webhooks/test"
        assert config_dict["timeout"] == 15
        assert "retry_config" in config_dict
        assert "circuit_breaker_config" in config_dict

        # Enum should be serialized as value
        assert config_dict["retry_config"]["backoff_type"] == "exponential"

    def test_from_dict_conversion(self):
        """Test configuration from dictionary conversion."""
        config_dict = {
            "webhook_url": "https://discord.com/api/webhooks/test",
            "timeout": 15,
            "retry_config": {"max_attempts": 7, "backoff_type": "linear"},
            "circuit_breaker_config": {"failure_threshold": 5},
            "queue_config": {"max_size": 2000},
            "health_thresholds": {"success_rate_warning": 90.0},
        }

        config = WebhookReliabilityConfig.from_dict(config_dict)

        assert config.webhook_url == "https://discord.com/api/webhooks/test"
        assert config.timeout == 15
        assert config.retry_config.max_attempts == 7
        assert config.retry_config.backoff_type == BackoffType.LINEAR
        assert config.circuit_breaker_config.failure_threshold == 5
        assert config.queue_config.max_size == 2000
        assert config.health_thresholds.success_rate_warning == 90.0

    def test_validation_success(self):
        """Test successful configuration validation."""
        config = WebhookReliabilityConfig(
            webhook_url="https://discord.com/api/webhooks/test"
        )

        errors = config.validate()
        assert len(errors) == 0

    def test_validation_errors(self):
        """Test configuration validation errors."""
        config = WebhookReliabilityConfig(
            webhook_url="",  # Missing webhook URL
            timeout=-5,  # Invalid timeout
            max_rate_limit_wait=-1.0,  # Invalid wait time
        )

        errors = config.validate()

        assert len(errors) >= 3
        assert any("webhook_url is required" in error for error in errors)
        assert any("timeout must be positive" in error for error in errors)
        assert any(
            "max_rate_limit_wait must be non-negative" in error for error in errors
        )

    def test_nested_validation_errors(self):
        """Test nested configuration validation errors."""
        # Create config with invalid nested values
        config = WebhookReliabilityConfig(
            webhook_url="https://discord.com/api/webhooks/test"
        )

        # Manually set invalid nested values (bypassing dataclass validation)
        config.health_thresholds.success_rate_warning = 150.0  # Invalid percentage
        config.health_thresholds.success_rate_critical = 200.0  # Invalid percentage

        errors = config.validate()

        assert len(errors) >= 2
        assert any("success_rate_warning must be 0-100" in error for error in errors)
        assert any("success_rate_critical must be 0-100" in error for error in errors)


class TestWebhookConfigManager:
    """Test cases for WebhookConfigManager."""

    def test_initialization(self):
        """Test config manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "test_config.json")
            manager = WebhookConfigManager(config_file)

            assert manager._config_file_path == config_file

    def test_load_default_configuration(self):
        """Test loading default configuration when no file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "nonexistent.json")
            manager = WebhookConfigManager(config_file)

            # Mock environment to provide webhook URL
            with patch.dict(
                os.environ, {"DISCORD_WEBHOOK_URL": "https://discord.com/test"}
            ):
                config = manager.load_configuration()

            assert isinstance(config, WebhookReliabilityConfig)
            assert config.webhook_url == "https://discord.com/test"

    def test_load_from_file(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "webhook_url": "https://discord.com/api/webhooks/test",
            "timeout": 20,
            "retry_config": {"max_attempts": 5, "backoff_type": "exponential"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            manager = WebhookConfigManager(config_file)
            config = manager.load_configuration()

            assert config.webhook_url == "https://discord.com/api/webhooks/test"
            assert config.timeout == 20
            assert config.retry_config.max_attempts == 5
        finally:
            os.unlink(config_file)

    def test_environment_variable_overrides(self):
        """Test environment variable overrides."""
        config_data = {
            "webhook_url": "https://discord.com/api/webhooks/file",
            "timeout": 10,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            manager = WebhookConfigManager(config_file)

            # Override with environment variables
            env_overrides = {
                "DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/env",
                "DISCORD_WEBHOOK_TIMEOUT": "25",
                "DISCORD_RETRY_MAX_ATTEMPTS": "7",
                "DISCORD_CB_FAILURE_THRESHOLD": "10",
            }

            with patch.dict(os.environ, env_overrides):
                config = manager.load_configuration()

            # Environment should override file values
            assert config.webhook_url == "https://discord.com/api/webhooks/env"
            assert config.timeout == 25
            assert config.retry_config.max_attempts == 7
            assert config.circuit_breaker_config.failure_threshold == 10

        finally:
            os.unlink(config_file)

    def test_boolean_environment_variables(self):
        """Test boolean environment variable parsing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "test.json")
            manager = WebhookConfigManager(config_file)

            env_vars = {
                "DISCORD_WEBHOOK_URL": "https://discord.com/test",
                "DISCORD_WEBHOOK_RELIABILITY_ENABLED": "true",
                "DISCORD_RETRY_JITTER_ENABLED": "false",
                "DISCORD_QUEUE_PERSISTENCE_ENABLED": "1",
                "DISCORD_BACKUP_NOTIFICATION_ENABLED": "0",
            }

            with patch.dict(os.environ, env_vars):
                config = manager.load_configuration()

            assert config.enabled is True
            assert config.retry_config.jitter_enabled is False
            assert config.queue_config.persistence_enabled is True
            assert config.backup_notification_enabled is False

    def test_invalid_environment_variables(self):
        """Test handling of invalid environment variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "test.json")
            manager = WebhookConfigManager(config_file)

            env_vars = {
                "DISCORD_WEBHOOK_URL": "https://discord.com/test",
                "DISCORD_WEBHOOK_TIMEOUT": "invalid_number",
                "DISCORD_RETRY_MAX_ATTEMPTS": "not_a_number",
            }

            with patch.dict(os.environ, env_vars):
                # Should not raise exception, but log warnings
                with patch.object(manager._logger, "warning") as mock_warning:
                    config = manager.load_configuration()

                # Should have logged warnings for invalid values
                assert mock_warning.call_count >= 2

            # Should use default values for invalid env vars
            assert config.timeout == 10  # Default value
            assert config.retry_config.max_attempts == 3  # Default value

    def test_save_configuration(self):
        """Test saving configuration to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "save_test.json")
            manager = WebhookConfigManager(config_file)

            config = WebhookReliabilityConfig(
                webhook_url="https://discord.com/api/webhooks/save_test", timeout=30
            )

            manager.save_configuration(config)

            # Verify file was created and contains expected data
            assert os.path.exists(config_file)

            with open(config_file, "r") as f:
                saved_data = json.load(f)

            assert (
                saved_data["webhook_url"]
                == "https://discord.com/api/webhooks/save_test"
            )
            assert saved_data["timeout"] == 30

    def test_save_invalid_configuration(self):
        """Test saving invalid configuration raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "invalid_test.json")
            manager = WebhookConfigManager(config_file)

            # Create invalid config
            config = WebhookReliabilityConfig(
                webhook_url="", timeout=-5  # Missing required field  # Invalid value
            )

            with pytest.raises(ValueError, match="Cannot save invalid configuration"):
                manager.save_configuration(config)

    def test_update_configuration(self):
        """Test updating configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "update_test.json")
            manager = WebhookConfigManager(config_file)

            # Load initial config
            with patch.dict(
                os.environ, {"DISCORD_WEBHOOK_URL": "https://discord.com/test"}
            ):
                initial_config = manager.load_configuration()

            assert initial_config.timeout == 10

            # Update configuration
            updates = {"timeout": 25, "retry_config.max_attempts": 7}

            updated_config = manager.update_configuration(updates)

            assert updated_config.timeout == 25
            assert updated_config.retry_config.max_attempts == 7
            # Other values should remain unchanged
            assert updated_config.webhook_url == "https://discord.com/test"

    def test_update_invalid_configuration(self):
        """Test updating with invalid values raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "update_test.json")
            manager = WebhookConfigManager(config_file)

            with patch.dict(
                os.environ, {"DISCORD_WEBHOOK_URL": "https://discord.com/test"}
            ):
                manager.load_configuration()

            # Try to update with invalid value
            updates = {"timeout": -10}

            with pytest.raises(
                ValueError, match="Configuration update failed validation"
            ):
                manager.update_configuration(updates)

    def test_configuration_change_callbacks(self):
        """Test configuration change callbacks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "callback_test.json")
            manager = WebhookConfigManager(config_file)

            callback_mock = Mock()
            manager.register_change_callback(callback_mock)

            # Load initial config
            with patch.dict(
                os.environ, {"DISCORD_WEBHOOK_URL": "https://discord.com/test"}
            ):
                manager.load_configuration()

            # Update configuration
            updates = {"timeout": 25}
            manager.update_configuration(updates)

            # Callback should have been called
            callback_mock.assert_called_once()
            args = callback_mock.call_args[0]
            old_config, new_config, source = args

            assert old_config.timeout == 10
            assert new_config.timeout == 25
            assert source == ConfigurationSource.RUNTIME

    def test_create_default_config_file(self):
        """Test creating default configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "default_test.json")
            manager = WebhookConfigManager(config_file)

            assert not os.path.exists(config_file)

            manager.create_default_config_file()

            assert os.path.exists(config_file)

            # Verify file contains default configuration
            with open(config_file, "r") as f:
                data = json.load(f)

            default_config = WebhookReliabilityConfig()
            expected_data = default_config.to_dict()

            assert data["enabled"] == expected_data["enabled"]
            assert data["timeout"] == expected_data["timeout"]

    def test_environment_template_generation(self):
        """Test environment template generation."""
        manager = WebhookConfigManager()
        template = manager.get_environment_template()

        # Should contain key environment variables
        assert "DISCORD_WEBHOOK_URL" in template
        assert "DISCORD_WEBHOOK_TIMEOUT" in template
        assert "DISCORD_RETRY_MAX_ATTEMPTS" in template
        assert "DISCORD_CB_FAILURE_THRESHOLD" in template

        # Should have comments and structure
        assert template.startswith("# Discord Webhook Reliability Configuration")
        assert "# Basic Settings" in template
        assert "# Retry Policy" in template


class TestConfigPresets:
    """Test cases for configuration presets."""

    def test_high_reliability_preset(self):
        """Test high reliability configuration preset."""
        config = ConfigPresets.high_reliability()

        assert config.retry_config.max_attempts >= 5
        assert config.circuit_breaker_config.failure_threshold >= 3
        assert config.queue_config.max_size >= 1000
        assert config.health_thresholds.success_rate_warning >= 90.0

        # Should pass validation
        errors = config.validate()
        assert len(errors) == 0

    def test_fast_response_preset(self):
        """Test fast response configuration preset."""
        config = ConfigPresets.fast_response()

        assert config.timeout <= 10
        assert config.max_rate_limit_wait <= 60.0
        assert config.retry_config.max_attempts <= 5
        assert config.retry_config.max_delay <= 10.0

        # Should pass validation
        errors = config.validate()
        # Note: might have webhook_url error, which is expected for preset
        assert all("webhook_url" in error for error in errors) or len(errors) == 0

    def test_development_preset(self):
        """Test development configuration preset."""
        config = ConfigPresets.development()

        assert config.retry_config.max_attempts <= 3
        assert config.queue_config.persistence_enabled is False
        assert config.logging_config["level"] == "DEBUG"

        # Should pass validation (except webhook_url)
        errors = config.validate()
        assert all("webhook_url" in error for error in errors) or len(errors) == 0


class TestFactoryFunctions:
    """Test cases for factory functions."""

    def test_create_webhook_config_manager(self):
        """Test webhook config manager factory."""
        manager = create_webhook_config_manager()
        assert isinstance(manager, WebhookConfigManager)

        # Test with custom path
        custom_path = "custom_config.json"
        manager = create_webhook_config_manager(custom_path)
        assert manager._config_file_path == custom_path

    def test_load_webhook_config(self):
        """Test webhook config loading function."""
        with patch.dict(
            os.environ, {"DISCORD_WEBHOOK_URL": "https://discord.com/test"}
        ):
            config = load_webhook_config()

        assert isinstance(config, WebhookReliabilityConfig)
        assert config.webhook_url == "https://discord.com/test"
