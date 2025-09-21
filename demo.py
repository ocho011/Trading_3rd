"""
Demo script for trading bot core infrastructure.

Demonstrates how to use ConfigManager and Logger components
in a real trading bot application scenario.
"""

import os
from pathlib import Path

from trading_bot.core.config_manager import create_config_manager, ConfigurationError
from trading_bot.core.logger import create_trading_logger, get_module_logger


def main():
    """
    Demonstrate core infrastructure components.

    Shows how to initialize and use ConfigManager and Logger
    in a typical trading bot application setup.
    """
    print("Trading Bot Core Infrastructure Demo")
    print("=" * 40)

    # Initialize logger
    logger = create_trading_logger(
        name="trading_bot_demo",
        log_level="INFO",
        log_dir="logs"
    )

    logger.info("Starting trading bot core infrastructure demo")

    try:
        # Initialize configuration manager
        logger.info("Loading configuration...")

        # Try environment variables first
        config_manager = create_config_manager("env")
        config_manager.load_configuration()

        logger.info("Configuration loaded successfully from environment variables")

        # Demonstrate getting configuration values
        log_level = config_manager.get_config_value('log_level', 'INFO')
        trading_mode = config_manager.get_config_value('trading_mode', 'paper')

        logger.info(f"Log level: {log_level}")
        logger.info(f"Trading mode: {trading_mode}")

        # Try to get API credentials (will fail if not set)
        try:
            api_credentials = config_manager.get_api_credentials()
            logger.info("API credentials loaded successfully")
            logger.info(f"API key prefix: {api_credentials['api_key'][:8]}...")
        except ConfigurationError as e:
            logger.warning(f"API credentials not available: {e}")

        # Get notification configuration
        notification_config = config_manager.get_notification_config()
        if notification_config['discord_webhook_url']:
            logger.info("Discord webhook URL configured")
        else:
            logger.warning("Discord webhook URL not configured")

        # Get trading configuration
        trading_config = config_manager.get_trading_config()
        logger.info(f"Max position size: {trading_config['max_position_size']}")
        logger.info(f"Risk percentage: {trading_config['risk_percentage']}")

        # Demonstrate module-specific loggers
        market_data_logger = get_module_logger("market_data")
        strategy_logger = get_module_logger("strategies")

        market_data_logger.info("Market data module logger test")
        strategy_logger.info("Strategy module logger test")

        # Show directory structure
        print("\nProject Directory Structure:")
        print("-" * 30)
        base_path = Path("trading_bot")
        if base_path.exists():
            for item in sorted(base_path.rglob("*")):
                if item.is_file() and item.name.endswith(('.py', '.txt')):
                    print(f"  {item}")

        logger.info("Demo completed successfully")

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        print(f"\nError: {e}")
        print("\nTo run this demo successfully, set environment variables:")
        print("  export BINANCE_API_KEY='your_api_key'")
        print("  export BINANCE_SECRET_KEY='your_secret_key'")
        print("  export DISCORD_WEBHOOK_URL='your_webhook_url'")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()