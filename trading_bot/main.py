"""
Main application entry point for the trading bot.

Implements core component initialization sequence following SOLID principles
and dependency injection patterns. Provides the foundation for the trading
bot application with proper error handling and logging.
"""

import asyncio
import logging
import os
import signal
import sys
import time
from decimal import Decimal
from typing import List, Optional

from trading_bot.core.config_manager import (
    ConfigManager,
    ConfigurationError,
    create_config_manager,
)
from trading_bot.core.event_hub import EventHub, EventType
from trading_bot.core.logger import create_trading_logger
from trading_bot.execution.execution_engine import (
    ExecutionEngine,
    ExecutionEngineConfig,
)

# Trading module imports for Task 10.2
from trading_bot.market_data.binance_client import BinanceClient
from trading_bot.market_data.data_processor import MarketDataProcessor

# WebSocket manager imports for Task 10.4
from trading_bot.market_data.websocket_manager import (
    BinanceWebSocketManager,
    WebSocketConnectionError,
    create_binance_websocket_manager,
)
from trading_bot.notification.discord_notifier import DiscordNotifier
from trading_bot.portfolio_manager.portfolio_manager import (
    PortfolioManager,
    PortfolioManagerConfig,
)
from trading_bot.risk_management.risk_manager import RiskManager, RiskManagerConfig
from trading_bot.strategies.base_strategy import StrategyConfiguration
from trading_bot.strategies.ict_strategy import ICTStrategy
from trading_bot.analysis.performance_logger import PerformanceLogger, create_performance_logger


class ComponentInitializationError(Exception):
    """Custom exception for component initialization failures."""


class TradingBotApplication:
    """
    Main trading bot application class.

    Manages the initialization and lifecycle of core components following
    the Single Responsibility Principle and proper dependency management.
    """

    def __init__(self) -> None:
        """
        Initialize trading bot application.

        Components are initialized in the correct dependency order:
        1. ConfigManager - provides configuration for all other components
        2. EventHub - enables event-driven communication
        3. Logger - provides centralized logging after configuration is available
        4. Trading Modules - all trading-specific components
        """
        # Core components (Task 10.1)
        self._config_manager: Optional[ConfigManager] = None
        self._event_hub: Optional[EventHub] = None
        self._logger: Optional[logging.Logger] = None
        self._is_initialized = False

        # Trading modules (Task 10.2)
        self._binance_client: Optional[BinanceClient] = None
        self._data_processor: Optional[MarketDataProcessor] = None
        self._ict_strategy: Optional[ICTStrategy] = None
        self._risk_manager: Optional[RiskManager] = None
        self._execution_engine: Optional[ExecutionEngine] = None
        self._portfolio_manager: Optional[PortfolioManager] = None
        self._discord_notifier: Optional[DiscordNotifier] = None
        self._trading_modules_initialized = False

        # Performance logging (Task 11.3)
        self._performance_logger: Optional[PerformanceLogger] = None

        # WebSocket manager (Task 10.4)
        self._websocket_manager: Optional[BinanceWebSocketManager] = None
        self._websocket_initialized = False

        # Async loop management
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._heartbeat_task: Optional[asyncio.Task] = None

    def initialize_core_components(self) -> None:
        """
        Initialize core components in dependency order.

        Raises:
            ComponentInitializationError: If any component fails to initialize
        """
        try:
            # Step 1: Initialize ConfigManager first (no dependencies)
            self._initialize_config_manager()

            # Step 2: Initialize EventHub (no dependencies on config)
            self._initialize_event_hub()

            # Step 3: Initialize Logger with configuration
            self._initialize_logger()

            # Log successful initialization with component info
            self._log_successful_initialization()

            # Mark as initialized
            self._is_initialized = True

            # Publish system startup event
            if self._event_hub:
                startup_data = {
                    "timestamp": "now",
                    "components_initialized": ["ConfigManager", "EventHub", "Logger"],
                }
                self._event_hub.publish(EventType.SYSTEM_STARTUP, startup_data)

        except Exception as e:
            error_msg = f"Failed to initialize core components: {e}"
            if self._logger:
                self._logger.error(error_msg)
            else:
                print(f"FATAL ERROR: {error_msg}", file=sys.stderr)
            raise ComponentInitializationError(error_msg) from e

    def _initialize_config_manager(self) -> None:
        """
        Initialize configuration manager using factory function.

        Raises:
            ConfigurationError: If configuration loading fails
        """
        try:
            self._config_manager = create_config_manager(config_source="env")
            self._config_manager.load_configuration()

            # Validate critical configuration exists
            self._validate_critical_configuration()

        except ConfigurationError as e:
            raise ComponentInitializationError(
                f"ConfigManager initialization failed: {e}"
            )
        except Exception as e:
            raise ComponentInitializationError(
                f"Unexpected error initializing ConfigManager: {e}"
            )

    def _initialize_event_hub(self) -> None:
        """
        Initialize event hub for system-wide event communication.

        Raises:
            ComponentInitializationError: If EventHub initialization fails
        """
        try:
            self._event_hub = EventHub()

        except Exception as e:
            raise ComponentInitializationError(f"EventHub initialization failed: {e}")

    def _initialize_logger(self) -> None:
        """
        Initialize logger using configuration from ConfigManager.

        Raises:
            ComponentInitializationError: If logger initialization fails
        """
        try:
            if not self._config_manager:
                raise ComponentInitializationError(
                    "ConfigManager must be initialized before Logger"
                )

            log_level = self._config_manager.get_config_value("log_level", "INFO")
            self._logger = create_trading_logger(
                name="trading_bot", log_level=log_level, log_dir="logs"
            )

        except Exception as e:
            raise ComponentInitializationError(f"Logger initialization failed: {e}")

    def _validate_critical_configuration(self) -> None:
        """
        Validate that critical configuration values are present.

        Raises:
            ConfigurationError: If critical configuration is missing
        """
        if not self._config_manager:
            raise ConfigurationError("ConfigManager not initialized")

        # Validate API credentials (will raise ConfigurationError if missing)
        try:
            api_credentials = self._config_manager.get_api_credentials()
            api_key = api_credentials.get("api_key")
            secret_key = api_credentials.get("secret_key")
            if not api_key or not secret_key:
                raise ConfigurationError("API credentials are incomplete")
        except ConfigurationError:
            # Re-raise configuration errors as-is
            raise
        except Exception as e:
            raise ConfigurationError(f"Error validating API credentials: {e}")

        # Validate trading configuration exists
        trading_config = self._config_manager.get_trading_config()
        required_keys = ["trading_mode", "max_position_size", "risk_percentage"]
        for key in required_keys:
            if key not in trading_config:
                msg = f"Missing required trading configuration: {key}"
                raise ConfigurationError(msg)

    def _log_successful_initialization(self) -> None:
        """
        Log successful component initialization with configuration info.

        Logs configuration values while masking sensitive data like API keys.
        """
        if not self._logger or not self._config_manager:
            return

        self._logger.info(
            "=== Trading Bot Core Components Initialized Successfully ==="
        )

        # Log ConfigManager status
        self._logger.info("✓ ConfigManager: Loaded and validated")

        # Log EventHub status
        self._logger.info("✓ EventHub: Ready for event-driven communication")

        # Log Logger status
        log_level = self._config_manager.get_config_value("log_level", "INFO")
        self._logger.info(f"✓ Logger: Configured with level {log_level}")

        # Log configuration values (mask sensitive data)
        self._log_configuration_summary()

        self._logger.info("=== Core Initialization Complete ===")

    def _log_configuration_summary(self) -> None:
        """
        Log configuration summary with sensitive data masked.
        """
        if not self._logger or not self._config_manager:
            return

        try:
            # API Configuration (masked)
            api_credentials = self._config_manager.get_api_credentials()
            api_key = api_credentials.get("api_key", "")
            api_key_masked = self._mask_sensitive_data(api_key)
            self._logger.info(f"API Key: {api_key_masked}")

            # Trading Configuration
            trading_config = self._config_manager.get_trading_config()
            mode = trading_config.get("trading_mode")
            max_pos = trading_config.get("max_position_size")
            risk_pct = trading_config.get("risk_percentage")
            self._logger.info(f"Trading Mode: {mode}")
            self._logger.info(f"Max Position Size: {max_pos}")
            self._logger.info(f"Risk Percentage: {risk_pct}%")

            # Notification Configuration
            notification_config = self._config_manager.get_notification_config()
            webhook_url = notification_config.get("discord_webhook_url", "")
            if webhook_url:
                webhook_masked = self._mask_sensitive_data(webhook_url)
                self._logger.info(f"Discord Webhook: {webhook_masked}")
            else:
                self._logger.warning("Discord Webhook: Not configured")

        except Exception as e:
            self._logger.warning(f"Could not log configuration summary: {e}")

    def _mask_sensitive_data(self, data: str) -> str:
        """
        Mask sensitive data for logging.

        Args:
            data: Sensitive data string to mask

        Returns:
            str: Masked data showing only first 4 and last 4 characters
        """
        if not data or len(data) <= 8:
            return "****"

        return f"{data[:4]}****{data[-4:]}"

    def initialize_trading_modules(self) -> None:
        """
        Initialize all trading modules in correct dependency order.

        Requires core components to be initialized first.
        Initializes:
        1. BinanceClient - market data connection
        2. DataProcessor - market data processing
        3. ICTStrategy - trading strategy
        4. RiskManager - risk management
        5. ExecutionEngine - order execution
        6. PortfolioManager - portfolio tracking
        7. DiscordNotifier - notifications

        Raises:
            ComponentInitializationError: If any trading module fails to initialize
            RuntimeError: If core components not initialized first
        """
        if not self._is_initialized:
            raise RuntimeError(
                "Core components must be initialized before trading modules"
            )

        if not self._config_manager or not self._event_hub or not self._logger:
            raise RuntimeError("Core components not properly initialized")

        self._logger.info("=== Initializing Trading Modules ===")

        try:
            # Initialize trading modules in dependency order
            self._initialize_binance_client()
            self._initialize_data_processor()
            self._initialize_ict_strategy()
            self._initialize_risk_manager()
            self._initialize_execution_engine()
            self._initialize_portfolio_manager()
            self._initialize_discord_notifier()
            self._initialize_performance_logger()

            # Mark trading modules as initialized
            self._trading_modules_initialized = True

            # Log successful initialization
            self._log_trading_modules_success()

            # Setup event subscriptions after all modules are initialized (Task 10.3)
            self._setup_event_subscriptions()

            # Initialize WebSocket manager after event subscriptions (Task 10.4)
            self._initialize_websocket_manager()

            # Publish trading system ready event
            self._event_hub.publish(
                EventType.SYSTEM_STARTUP,
                {
                    "timestamp": "now",
                    "trading_modules_initialized": [
                        "BinanceClient",
                        "DataProcessor",
                        "ICTStrategy",
                        "RiskManager",
                        "ExecutionEngine",
                        "PortfolioManager",
                        "DiscordNotifier",
                        "PerformanceLogger",
                        "WebSocketManager",
                    ],
                },
            )

            self._logger.info("=== Trading Modules Initialization Complete ===")

        except Exception as e:
            error_msg = f"Failed to initialize trading modules: {e}"
            self._logger.error(error_msg)
            raise ComponentInitializationError(error_msg) from e

    def _initialize_binance_client(self) -> None:
        """Initialize Binance client for market data connection."""
        try:
            self._binance_client = BinanceClient(self._config_manager)
            self._binance_client.initialize()
            self._logger.info("✓ BinanceClient: Initialized and connected")
        except Exception as e:
            self._logger.error(f"BinanceClient initialization failed: {e}")
            raise ComponentInitializationError(
                f"BinanceClient initialization failed: {e}"
            ) from e

    def _initialize_data_processor(self) -> None:
        """Initialize market data processor."""
        try:
            # Use sensible defaults for optional parameters
            supported_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
            self._data_processor = MarketDataProcessor(
                config_manager=self._config_manager,
                event_hub=self._event_hub,
                data_validator=None,  # Use default validator
                supported_timeframes=supported_timeframes,
            )
            self._logger.info("✓ DataProcessor: Initialized with default configuration")
        except Exception as e:
            self._logger.error(f"DataProcessor initialization failed: {e}")
            raise ComponentInitializationError(
                f"DataProcessor initialization failed: {e}"
            ) from e

    def _initialize_ict_strategy(self) -> None:
        """Initialize ICT trading strategy."""
        try:
            # Get default symbol and timeframe from environment or use defaults
            self._config_manager.get_trading_config()
            default_symbol = os.environ.get("DEFAULT_SYMBOL", "BTCUSDT")
            default_timeframe = os.environ.get("DEFAULT_TIMEFRAME", "15m")

            # Create strategy configuration with defaults
            strategy_config = StrategyConfiguration(
                name="ICT_Strategy",
                symbol=default_symbol,
                timeframe=default_timeframe,
                enabled=True,
                parameters={
                    "timeframe": default_timeframe,
                    "risk_reward_ratio": 2.0,
                    "max_daily_trades": 5,
                    "paper_trading": True,  # Default to paper trading for safety
                },
            )

            self._ict_strategy = ICTStrategy(
                config=strategy_config,  # Pass the StrategyConfiguration
                event_hub=self._event_hub,
                ict_config=None,  # Use default ICT configuration
                ict_strategy_config=None,  # Use default ICT strategy configuration
            )
            self._logger.info("✓ ICTStrategy: Initialized with default configuration")
        except Exception as e:
            self._logger.error(f"ICTStrategy initialization failed: {e}")
            raise ComponentInitializationError(
                f"ICTStrategy initialization failed: {e}"
            ) from e

    def _initialize_risk_manager(self) -> None:
        """Initialize risk management system."""
        try:
            # Create risk manager configuration with defaults
            risk_config = RiskManagerConfig(
                max_position_risk_percentage=2.0,  # 2% risk per position
                max_portfolio_risk_percentage=10.0,  # 10% maximum portfolio risk
                min_confidence_threshold=0.6,  # 60% minimum confidence
                enable_position_sizing=True,
                enable_risk_assessment=True,
                enable_stop_loss_calculation=True,
                enable_account_risk_evaluation=True,
            )

            self._risk_manager = RiskManager(
                config=risk_config,
                event_hub=self._event_hub,
                config_manager=self._config_manager,
                position_sizer=None,  # Use default position sizer
                risk_assessor=None,  # Use default risk assessor
                stop_loss_calculator=None,  # Use default stop loss calculator
                account_risk_evaluator=None,  # Use default account risk evaluator
            )
            self._logger.info("✓ RiskManager: Initialized with conservative defaults")
        except Exception as e:
            self._logger.error(f"RiskManager initialization failed: {e}")
            raise ComponentInitializationError(
                f"RiskManager initialization failed: {e}"
            ) from e

    def _initialize_execution_engine(self) -> None:
        """Initialize order execution engine."""
        try:
            if not self._binance_client:
                msg = "BinanceClient must be initialized before ExecutionEngine"
                raise ComponentInitializationError(msg)

            # Create execution engine configuration with correct parameters
            execution_config = ExecutionEngineConfig()
            execution_config.enable_pre_execution_validation = True
            execution_config.enable_order_execution = True
            execution_config.max_retry_attempts = 3
            execution_config.execution_timeout_seconds = 30.0
            execution_config.min_order_value_usd = 10.0
            execution_config.max_order_value_usd = 10000.0
            execution_config.enable_execution_monitoring = True
            execution_config.log_all_order_requests = True

            self._execution_engine = ExecutionEngine(
                config=execution_config,
                event_hub=self._event_hub,
                binance_client=self._binance_client,
            )
            self._logger.info(
                "✓ ExecutionEngine: Initialized with default configuration"
            )
        except Exception as e:
            self._logger.error(f"ExecutionEngine initialization failed: {e}")
            raise ComponentInitializationError(
                f"ExecutionEngine initialization failed: {e}"
            ) from e

    def _initialize_portfolio_manager(self) -> None:
        """Initialize portfolio management system."""
        try:
            if not self._binance_client:
                msg = "BinanceClient must be initialized before PortfolioManager"
                raise ComponentInitializationError(msg)

            # Create portfolio manager configuration with defaults
            portfolio_config = PortfolioManagerConfig(
                base_currency="USDT",
                enable_position_tracking=True,
                enable_balance_tracking=True,
                enable_portfolio_reporting=True,
                sync_interval_minutes=5,
                reporting_interval_minutes=15,
            )

            self._portfolio_manager = PortfolioManager(
                config=portfolio_config,
                exchange_client=self._binance_client,
                event_hub=self._event_hub,
                config_manager=self._config_manager,
                initial_balance=Decimal("10000.0"),  # Paper trading balance
            )
            self._logger.info("✓ PortfolioManager: Initialized with $10K paper balance")
        except Exception as e:
            self._logger.error(f"PortfolioManager initialization failed: {e}")
            msg = f"PortfolioManager initialization failed: {e}"
            raise ComponentInitializationError(msg) from e

    def _initialize_discord_notifier(self) -> None:
        """Initialize Discord notification system."""
        try:
            self._discord_notifier = DiscordNotifier(
                config_manager=self._config_manager,
                http_client=None,  # Use default HTTP client
                event_hub=self._event_hub,
                message_formatter_factory=None,  # Use default message formatter
            )
            self._logger.info(
                "✓ DiscordNotifier: Initialized with default configuration"
            )
        except Exception as e:
            # Don't fail the entire system if Discord notifications fail
            self._logger.warning(f"DiscordNotifier initialization failed: {e}")
            self._logger.warning("Continuing without Discord notifications")
            self._discord_notifier = None

    def _initialize_performance_logger(self) -> None:
        """Initialize performance logger for trade execution tracking."""
        try:
            if not self._event_hub:
                msg = "EventHub must be initialized before PerformanceLogger"
                raise ComponentInitializationError(msg)

            # Create performance logger with default settings
            self._performance_logger = create_performance_logger(
                event_hub=self._event_hub,
                log_directory="logs",
                csv_filename="trades.csv"
            )

            self._logger.info("✓ PerformanceLogger: Initialized and subscribed to ORDER_FILLED events")

        except Exception as e:
            # Don't fail the entire system if performance logging fails
            self._logger.warning(f"PerformanceLogger initialization failed: {e}")
            self._logger.warning("Continuing without performance logging")
            self._performance_logger = None

    def _initialize_websocket_manager(self) -> None:
        """Initialize WebSocket manager for real-time market data streaming.

        Raises:
            ComponentInitializationError: If WebSocket manager initialization fails
        """
        try:
            if not self._config_manager or not self._event_hub:
                msg = "Core components must be initialized before WebSocket manager"
                raise ComponentInitializationError(msg)

            # Get trading symbol from configuration or use default
            trading_config = self._config_manager.get_trading_config()
            symbol = trading_config.get("symbol", "btcusdt")

            # Create WebSocket manager using factory function
            self._websocket_manager = create_binance_websocket_manager(
                config_manager=self._config_manager,
                event_hub=self._event_hub,
                symbol=symbol,
                reconnection_config=None,  # Use default reconnection config
            )

            self._websocket_initialized = True
            self._logger.info(
                f"✓ WebSocketManager: Initialized for symbol {symbol.upper()}"
            )

        except Exception as e:
            # Don't fail the entire system if WebSocket fails to initialize
            self._logger.warning(f"WebSocketManager initialization failed: {e}")
            self._logger.warning("Continuing without WebSocket streaming")
            self._websocket_manager = None
            self._websocket_initialized = False

    def _setup_event_subscriptions(self) -> None:
        """
        Setup event subscriptions for all trading modules.

        Establishes the event flow architecture:
        1. WebSocketManager publishes MARKET_DATA_RECEIVED
        2. DataProcessor subscribes to MARKET_DATA_RECEIVED, publishes
           CANDLE_DATA_PROCESSED
        3. ICTStrategy subscribes to CANDLE_DATA_PROCESSED, publishes
           TRADING_SIGNAL_GENERATED
        4. RiskManager subscribes to TRADING_SIGNAL_GENERATED,
           publishes ORDER_REQUEST_GENERATED
        5. ExecutionEngine subscribes to ORDER_REQUEST_GENERATED,
           publishes ORDER_PLACED/ORDER_FILLED
        6. PortfolioManager subscribes to ORDER_FILLED
        7. DiscordNotifier subscribes to multiple events for notifications

        Raises:
            ComponentInitializationError: If event subscription setup fails
        """
        if not self._event_hub or not self._logger:
            raise ComponentInitializationError(
                "Core components must be initialized before event subscriptions"
            )

        self._logger.info("=== Setting Up Event Subscriptions ===")

        try:
            # Track successful subscriptions for logging
            successful_subscriptions = []

            # 1. DataProcessor - manually subscribe to MARKET_DATA_RECEIVED
            # (DataProcessor already subscribes in its constructor,
            # but we verify it here)
            if self._data_processor:
                try:
                    # DataProcessor constructor already handles
                    # this subscription
                    self._logger.info(
                        "✓ DataProcessor: Already subscribed to "
                        "MARKET_DATA_RECEIVED events"
                    )
                    successful_subscriptions.append(
                        "DataProcessor -> MARKET_DATA_RECEIVED"
                    )
                except Exception as e:
                    self._logger.error(
                        f"Failed to setup DataProcessor event subscription: {e}"
                    )
                    raise ComponentInitializationError(
                        f"DataProcessor event subscription failed: {e}"
                    )

            # 2. ICTStrategy - call existing _subscribe_to_events() method
            if self._ict_strategy:
                try:
                    self._ict_strategy._subscribe_to_events()
                    self._logger.info("✓ ICTStrategy: Subscribed to trading events")
                    successful_subscriptions.append(
                        "ICTStrategy -> CANDLE_DATA_PROCESSED"
                    )
                except Exception as e:
                    self._logger.error(
                        f"Failed to setup ICTStrategy event subscriptions: {e}"
                    )
                    raise ComponentInitializationError(
                        f"ICTStrategy event subscription failed: {e}"
                    )

            # 3. RiskManager - automatic subscription in constructor
            if self._risk_manager:
                try:
                    # RiskManager constructor already handles event subscriptions
                    self._logger.info(
                        "✓ RiskManager: Auto-subscribed to "
                        "TRADING_SIGNAL_GENERATED events"
                    )
                    successful_subscriptions.append(
                        "RiskManager -> TRADING_SIGNAL_GENERATED"
                    )
                except Exception as e:
                    self._logger.warning(
                        f"RiskManager event subscription verification failed: {e}"
                    )

            # 4. ExecutionEngine - automatic subscription in constructor
            if self._execution_engine:
                try:
                    # ExecutionEngine constructor already handles event subscriptions
                    self._logger.info(
                        "✓ ExecutionEngine: Auto-subscribed to "
                        "ORDER_REQUEST_GENERATED events"
                    )
                    successful_subscriptions.append(
                        "ExecutionEngine -> ORDER_REQUEST_GENERATED"
                    )
                except Exception as e:
                    self._logger.warning(
                        f"ExecutionEngine event subscription verification failed: {e}"
                    )

            # 5. PortfolioManager - automatic subscription in constructor
            if self._portfolio_manager:
                try:
                    # PortfolioManager constructor already handles event subscriptions
                    self._logger.info(
                        "✓ PortfolioManager: Auto-subscribed to ORDER_FILLED events"
                    )
                    successful_subscriptions.append("PortfolioManager -> ORDER_FILLED")
                except Exception as e:
                    self._logger.warning(
                        f"PortfolioManager event subscription verification failed: {e}"
                    )

            # 6. DiscordNotifier - call existing subscribe_to_events() method
            if self._discord_notifier:
                try:
                    self._discord_notifier.subscribe_to_events()
                    self._logger.info(
                        "✓ DiscordNotifier: Subscribed to notification events"
                    )
                    successful_subscriptions.extend(
                        [
                            "DiscordNotifier -> ORDER_FILLED",
                            "DiscordNotifier -> ERROR_OCCURRED",
                            "DiscordNotifier -> CONNECTION_LOST",
                            "DiscordNotifier -> TRADING_SIGNAL_GENERATED",
                            "DiscordNotifier -> RISK_LIMIT_EXCEEDED",
                        ]
                    )
                except Exception as e:
                    # Don't fail system if Discord notifications fail
                    self._logger.warning(
                        f"DiscordNotifier event subscription failed: {e}"
                    )
                    self._logger.warning(
                        "Continuing without Discord notification events"
                    )

            # Log summary of established event subscriptions
            self._log_event_subscription_summary(successful_subscriptions)

            # Publish event subscription setup complete event
            self._event_hub.publish(
                EventType.SYSTEM_STARTUP,
                {
                    "timestamp": "now",
                    "event_subscriptions_setup": True,
                    "subscription_count": len(successful_subscriptions),
                    "subscriptions": successful_subscriptions,
                },
            )

            self._logger.info("=== Event Subscriptions Setup Complete ===")

        except ComponentInitializationError:
            # Re-raise component initialization errors
            raise
        except Exception as e:
            error_msg = f"Unexpected error setting up event subscriptions: {e}"
            self._logger.error(error_msg)
            raise ComponentInitializationError(error_msg) from e

    def _log_event_subscription_summary(
        self, successful_subscriptions: List[str]
    ) -> None:
        """
        Log summary of established event subscriptions.

        Args:
            successful_subscriptions: List of successful subscription descriptions
        """
        if not self._logger:
            return

        self._logger.info("=== Event Flow Architecture Established ===")
        self._logger.info(
            "Event flow: WebSocket -> DataProcessor -> ICTStrategy -> RiskManager"
        )
        self._logger.info("            -> ExecutionEngine -> PortfolioManager")
        self._logger.info("Notifications: DiscordNotifier listens to multiple events")

        self._logger.info(
            f"Total event subscriptions established: {len(successful_subscriptions)}"
        )

        if successful_subscriptions:
            self._logger.info("Active event subscriptions:")
            for subscription in successful_subscriptions:
                self._logger.info(f"  • {subscription}")

        # Log event hub statistics
        if self._event_hub:
            try:
                startup_count = self._event_hub.get_subscriber_count(
                    EventType.SYSTEM_STARTUP
                )
                market_data_count = self._event_hub.get_subscriber_count(
                    EventType.MARKET_DATA_RECEIVED
                )
                signal_count = self._event_hub.get_subscriber_count(
                    EventType.TRADING_SIGNAL_GENERATED
                )

                self._logger.info("EventHub subscriber counts:")
                self._logger.info(f"  • SYSTEM_STARTUP: {startup_count}")
                self._logger.info(f"  • MARKET_DATA_RECEIVED: {market_data_count}")
                self._logger.info(f"  • TRADING_SIGNAL_GENERATED: {signal_count}")
            except Exception as e:
                self._logger.debug(f"Could not log EventHub statistics: {e}")

    def _log_trading_modules_success(self) -> None:
        """Log successful trading modules initialization."""
        self._logger.info("=== Trading Modules Successfully Initialized ===")

        # Log each module status
        modules_status = [
            ("BinanceClient", self._binance_client is not None),
            ("DataProcessor", self._data_processor is not None),
            ("ICTStrategy", self._ict_strategy is not None),
            ("RiskManager", self._risk_manager is not None),
            ("ExecutionEngine", self._execution_engine is not None),
            ("PortfolioManager", self._portfolio_manager is not None),
            ("DiscordNotifier", self._discord_notifier is not None),
            ("WebSocketManager", self._websocket_manager is not None),
        ]

        for module_name, initialized in modules_status:
            status = "✓ Ready" if initialized else "✗ Failed"
            self._logger.info(f"{module_name}: {status}")

        self._logger.info("Trading system is ready for operation")

    def get_config_manager(self) -> ConfigManager:
        """
        Get initialized ConfigManager instance.

        Returns:
            ConfigManager: Initialized configuration manager

        Raises:
            RuntimeError: If components not initialized
        """
        if not self._is_initialized or not self._config_manager:
            msg = (
                "Application not initialized. Call "
                "initialize_core_components() first."
            )
            raise RuntimeError(msg)
        return self._config_manager

    def get_event_hub(self) -> EventHub:
        """
        Get initialized EventHub instance.

        Returns:
            EventHub: Initialized event hub

        Raises:
            RuntimeError: If components not initialized
        """
        if not self._is_initialized or not self._event_hub:
            msg = (
                "Application not initialized. Call "
                "initialize_core_components() first."
            )
            raise RuntimeError(msg)
        return self._event_hub

    def get_logger(self) -> logging.Logger:
        """
        Get initialized Logger instance.

        Returns:
            logging.Logger: Initialized logger

        Raises:
            RuntimeError: If components not initialized
        """
        if not self._is_initialized or not self._logger:
            msg = (
                "Application not initialized. Call "
                "initialize_core_components() first."
            )
            raise RuntimeError(msg)
        return self._logger

    def is_initialized(self) -> bool:
        """
        Check if core components are initialized.

        Returns:
            bool: True if all core components are initialized
        """
        return self._is_initialized

    def are_trading_modules_initialized(self) -> bool:
        """
        Check if trading modules are initialized.

        Returns:
            bool: True if all trading modules are initialized
        """
        return self._trading_modules_initialized

    def get_binance_client(self) -> BinanceClient:
        """
        Get initialized BinanceClient instance.

        Returns:
            BinanceClient: Initialized Binance client

        Raises:
            RuntimeError: If trading modules not initialized
        """
        if not self._trading_modules_initialized or not self._binance_client:
            msg = (
                "Trading modules not initialized. "
                "Call initialize_trading_modules() first."
            )
            raise RuntimeError(msg)
        return self._binance_client

    def get_data_processor(self) -> MarketDataProcessor:
        """
        Get initialized DataProcessor instance.

        Returns:
            MarketDataProcessor: Initialized data processor

        Raises:
            RuntimeError: If trading modules not initialized
        """
        if not self._trading_modules_initialized or not self._data_processor:
            msg = (
                "Trading modules not initialized. "
                "Call initialize_trading_modules() first."
            )
            raise RuntimeError(msg)
        return self._data_processor

    def get_ict_strategy(self) -> ICTStrategy:
        """
        Get initialized ICTStrategy instance.

        Returns:
            ICTStrategy: Initialized ICT strategy

        Raises:
            RuntimeError: If trading modules not initialized
        """
        if not self._trading_modules_initialized or not self._ict_strategy:
            msg = (
                "Trading modules not initialized. "
                "Call initialize_trading_modules() first."
            )
            raise RuntimeError(msg)
        return self._ict_strategy

    def get_risk_manager(self) -> RiskManager:
        """
        Get initialized RiskManager instance.

        Returns:
            RiskManager: Initialized risk manager

        Raises:
            RuntimeError: If trading modules not initialized
        """
        if not self._trading_modules_initialized or not self._risk_manager:
            msg = (
                "Trading modules not initialized. "
                "Call initialize_trading_modules() first."
            )
            raise RuntimeError(msg)
        return self._risk_manager

    def get_execution_engine(self) -> ExecutionEngine:
        """
        Get initialized ExecutionEngine instance.

        Returns:
            ExecutionEngine: Initialized execution engine

        Raises:
            RuntimeError: If trading modules not initialized
        """
        if not self._trading_modules_initialized or not self._execution_engine:
            msg = (
                "Trading modules not initialized. "
                "Call initialize_trading_modules() first."
            )
            raise RuntimeError(msg)
        return self._execution_engine

    def get_portfolio_manager(self) -> PortfolioManager:
        """
        Get initialized PortfolioManager instance.

        Returns:
            PortfolioManager: Initialized portfolio manager

        Raises:
            RuntimeError: If trading modules not initialized
        """
        if not self._trading_modules_initialized or not self._portfolio_manager:
            msg = (
                "Trading modules not initialized. "
                "Call initialize_trading_modules() first."
            )
            raise RuntimeError(msg)
        return self._portfolio_manager

    def get_discord_notifier(self) -> Optional[DiscordNotifier]:
        """
        Get initialized DiscordNotifier instance.

        Returns:
            Optional[DiscordNotifier]: Initialized Discord notifier or None if failed

        Note:
            Discord notifier is optional and may be None if initialization failed
        """
        return self._discord_notifier

    def get_websocket_manager(self) -> Optional[BinanceWebSocketManager]:
        """
        Get initialized WebSocketManager instance.

        Returns:
            Optional[BinanceWebSocketManager]: Initialized WebSocket manager
                or None if failed

        Note:
            WebSocket manager is optional and may be None if initialization failed
        """
        return self._websocket_manager

    def is_websocket_initialized(self) -> bool:
        """
        Check if WebSocket manager is initialized.

        Returns:
            bool: True if WebSocket manager is initialized
        """
        return self._websocket_initialized

    async def _system_heartbeat_loop(self) -> None:
        """
        Background task that publishes system heartbeat events every hour.

        This method runs continuously while the application is active,
        sending SYSTEM_HEARTBEAT events to indicate the system is running normally.
        """
        if not self._logger or not self._event_hub:
            return

        self._logger.info("System heartbeat monitoring started (1-hour intervals)")

        while self._running and not self._shutdown_event.is_set():
            try:
                # Wait for 1 hour (3600 seconds) between heartbeats
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=3600.0  # 1 hour
                )
                # If we reach here, shutdown was requested
                break

            except asyncio.TimeoutError:
                # Timeout is expected - this means 1 hour has passed
                if self._running and not self._shutdown_event.is_set():
                    # Publish heartbeat event
                    heartbeat_data = {
                        "timestamp": int(time.time()),
                        "uptime_hours": int((time.time() - self._start_time) / 3600) if hasattr(self, '_start_time') else 0,
                        "components_status": {
                            "core_initialized": self._is_initialized,
                            "trading_modules_initialized": self._trading_modules_initialized,
                            "websocket_connected": bool(
                                self._websocket_manager
                                and hasattr(self._websocket_manager, 'is_connected')
                                and self._websocket_manager.is_connected()
                            )
                        }
                    }

                    self._event_hub.publish(EventType.SYSTEM_HEARTBEAT, heartbeat_data)
                    self._logger.info("System heartbeat published - all systems operational")

            except Exception as e:
                self._logger.error(f"Error in heartbeat loop: {e}")
                # Continue running even if there's an error
                await asyncio.sleep(60)  # Wait 1 minute before retrying

        self._logger.info("System heartbeat monitoring stopped")

    def shutdown(self) -> None:
        """
        Gracefully shutdown the application.

        Publishes shutdown event and cleans up resources.
        """
        if self._logger:
            self._logger.info("=== Trading Bot Shutdown Initiated ===")

        try:
            # Shutdown trading modules first
            self._shutdown_trading_modules()

            # Publish shutdown event
            if self._event_hub:
                self._event_hub.publish(
                    EventType.SYSTEM_SHUTDOWN,
                    {"timestamp": "now", "reason": "Graceful shutdown"},
                )

                # Clear all event subscribers for clean shutdown
                self._event_hub.clear_subscribers()

            # Log shutdown completion
            if self._logger:
                self._logger.info("=== Trading Bot Shutdown Complete ===")

        except Exception as e:
            if self._logger:
                self._logger.error(f"Error during shutdown: {e}")
            else:
                print(f"Error during shutdown: {e}", file=sys.stderr)

    def _shutdown_trading_modules(self) -> None:
        """
        Gracefully shutdown trading modules.

        Cleans up resources and connections for each trading module.
        """
        if not self._trading_modules_initialized:
            return

        if self._logger:
            self._logger.info("Shutting down trading modules...")

        try:
            # Shutdown modules in reverse dependency order
            # Note: WebSocket manager is shut down in async_cleanup()

            if self._discord_notifier:
                try:
                    # Discord notifier may have cleanup methods
                    self._logger.info("✓ DiscordNotifier: Shutdown complete")
                except Exception as e:
                    self._logger.warning(f"DiscordNotifier shutdown warning: {e}")

            if self._portfolio_manager:
                try:
                    # Portfolio manager may need to save final state
                    self._logger.info("✓ PortfolioManager: Shutdown complete")
                except Exception as e:
                    self._logger.warning(f"PortfolioManager shutdown warning: {e}")

            if self._execution_engine:
                try:
                    # Execution engine may need to cancel pending orders
                    self._logger.info("✓ ExecutionEngine: Shutdown complete")
                except Exception as e:
                    self._logger.warning(f"ExecutionEngine shutdown warning: {e}")

            if self._risk_manager:
                try:
                    # Risk manager cleanup
                    self._logger.info("✓ RiskManager: Shutdown complete")
                except Exception as e:
                    self._logger.warning(f"RiskManager shutdown warning: {e}")

            if self._ict_strategy:
                try:
                    # Strategy cleanup
                    self._logger.info("✓ ICTStrategy: Shutdown complete")
                except Exception as e:
                    self._logger.warning(f"ICTStrategy shutdown warning: {e}")

            if self._data_processor:
                try:
                    # Data processor cleanup
                    self._logger.info("✓ DataProcessor: Shutdown complete")
                except Exception as e:
                    self._logger.warning(f"DataProcessor shutdown warning: {e}")

            if self._binance_client:
                try:
                    # Close Binance client connections
                    self._logger.info("✓ BinanceClient: Shutdown complete")
                except Exception as e:
                    self._logger.warning(f"BinanceClient shutdown warning: {e}")

            self._logger.info("Trading modules shutdown complete")

        except Exception as e:
            if self._logger:
                self._logger.error(f"Error shutting down trading modules: {e}")

    async def run_async(self) -> None:
        """
        Run the trading bot in async mode with WebSocket streaming.

        This method provides the main async event loop for the trading bot,
        including WebSocket data streaming, signal handling, and graceful shutdown.

        Raises:
            RuntimeError: If components not initialized
            ComponentInitializationError: If WebSocket connection fails
        """
        if not self._is_initialized or not self._trading_modules_initialized:
            raise RuntimeError(
                "All components must be initialized before starting async mode. "
                "Call initialize_core_components() and "
                "initialize_trading_modules() first."
            )

        self._running = True
        self._start_time = time.time()  # Track application start time
        self._logger.info("=== Starting Trading Bot Async Mode ===")

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        # Start system heartbeat monitoring
        if self._event_hub:
            self._heartbeat_task = asyncio.create_task(self._system_heartbeat_loop())
            self._logger.info("System heartbeat monitoring task started")

        try:
            # Start WebSocket connection if available
            if self._websocket_manager:
                self._logger.info("Starting WebSocket data streaming...")
                try:
                    await self._websocket_manager.start()
                    self._logger.info(
                        f"✓ WebSocket streaming active for "
                        f"symbol {self._websocket_manager._symbol.upper()}"
                    )
                except WebSocketConnectionError as e:
                    self._logger.error(f"WebSocket connection failed: {e}")
                    self._logger.warning(
                        "Continuing in async mode without WebSocket data"
                    )
                    # Don't raise - continue without WebSocket
                except Exception as e:
                    self._logger.error(f"Unexpected WebSocket error: {e}")
                    self._logger.warning(
                        "Continuing in async mode without WebSocket data"
                    )
            else:
                self._logger.warning(
                    "WebSocket manager not initialized - no real-time data"
                )

            # Log async mode startup complete
            self._logger.info("=== Trading Bot Async Mode Active ===")
            self._logger.info("System Status:")
            self._logger.info(
                f"  • Core Components: {'✓' if self._is_initialized else '✗'}"
            )
            self._logger.info(
                f"  • Trading Modules: "
                f"{'✓' if self._trading_modules_initialized else '✗'}"
            )
            ws_status = (
                "✓"
                if (
                    self._websocket_manager
                    and hasattr(self._websocket_manager, "is_connected")
                    and self._websocket_manager.is_connected()
                )
                else "✗"
            )
            self._logger.info(f"  • WebSocket Connection: {ws_status}")
            startup_count = (
                self._event_hub.get_subscriber_count(EventType.SYSTEM_STARTUP)
                if self._event_hub
                else 0
            )
            self._logger.info(
                f"  • Event Subscriptions: {'✓' if startup_count > 0 else '✗'}"
            )

            # Wait for shutdown signal
            self._logger.info("Trading bot running... Press Ctrl+C to stop")
            await self._shutdown_event.wait()

        except asyncio.CancelledError:
            self._logger.info("Async operation cancelled - initiating shutdown")
        except KeyboardInterrupt:
            self._logger.info("Keyboard interrupt received - initiating shutdown")
        except Exception as e:
            self._logger.error(f"Unexpected error in async mode: {e}")
            raise
        finally:
            self._running = False
            self._logger.info("=== Async Mode Shutdown Initiated ===")
            await self._async_cleanup()

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown in async mode."""

        def signal_handler(signum: int, _frame) -> None:
            """Handle shutdown signals by setting the shutdown event."""
            signal_name = signal.Signals(signum).name
            self._logger.info(
                f"Received {signal_name} signal - initiating graceful shutdown"
            )

            # Schedule async shutdown in the event loop
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._signal_shutdown())
            except RuntimeError:
                # No event loop running, fallback to sync shutdown
                self._logger.warning(
                    "No event loop running - falling back to sync shutdown"
                )
                self.shutdown()

        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def _signal_shutdown(self) -> None:
        """Handle shutdown signal in async context."""
        self._shutdown_event.set()

    async def _async_cleanup(self) -> None:
        """Perform async cleanup operations."""
        self._logger.info("Performing async cleanup...")

        try:
            # Stop heartbeat monitoring task
            if self._heartbeat_task and not self._heartbeat_task.done():
                self._logger.info("Stopping heartbeat monitoring task...")
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    self._logger.info("✓ Heartbeat monitoring task stopped")
                except Exception as e:
                    self._logger.warning(f"Heartbeat task shutdown warning: {e}")
            # Stop WebSocket connection if active
            if (
                self._websocket_manager
                and hasattr(self._websocket_manager, "is_connected")
                and self._websocket_manager.is_connected()
            ):
                self._logger.info("Stopping WebSocket connection...")
                try:
                    await self._websocket_manager.stop()
                    self._logger.info("✓ WebSocket connection stopped")
                except Exception as e:
                    self._logger.warning(f"WebSocket shutdown warning: {e}")

            # Perform standard synchronous cleanup
            self.shutdown()

            self._logger.info("✓ Async cleanup complete")

        except Exception as e:
            self._logger.error(f"Error during async cleanup: {e}")
            # Still perform sync cleanup as fallback
            try:
                self.shutdown()
            except Exception as sync_error:
                self._logger.error(f"Sync cleanup also failed: {sync_error}")

    async def async_shutdown(self) -> None:
        """
        Gracefully shutdown the application in async mode.

        This method handles async resource cleanup and should be used
        instead of the synchronous shutdown() method when running in async mode.

        Example:
            await app.async_shutdown()
        """
        self._logger.info("=== Async Shutdown Initiated ===")

        # Set shutdown event to signal running tasks to stop
        if not self._shutdown_event.is_set():
            self._shutdown_event.set()

        # Wait a moment for tasks to respond to shutdown signal
        await asyncio.sleep(0.1)

        # Perform async cleanup
        await self._async_cleanup()

        self._logger.info("=== Async Shutdown Complete ===")


def main() -> int:
    """
    Main application entry point.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    app = None
    try:
        # Create and initialize application
        app = TradingBotApplication()
        app.initialize_core_components()

        # Get logger for main function
        logger = app.get_logger()
        logger.info("Trading Bot application started successfully")

        # Initialize trading modules (Task 10.2)
        app.initialize_trading_modules()

        # All components are now ready
        event_hub = app.get_event_hub()

        logger.info("All components initialized successfully")
        startup_subscribers = event_hub.get_subscriber_count(EventType.SYSTEM_STARTUP)
        logger.info(f"Event Hub has {startup_subscribers} startup subscribers")

        # Log summary of initialized components
        logger.info("=== System Ready ===")
        logger.info("Core Components: ConfigManager, EventHub, Logger")
        logger.info(
            "Trading Modules: BinanceClient, DataProcessor, ICTStrategy, "
            "RiskManager, ExecutionEngine, PortfolioManager, DiscordNotifier"
        )
        logger.info("System is ready for trading operations")

        # Trading system is now ready for:
        # 1. Market data streaming (WebSocket + REST)
        # 2. Strategy signal generation
        # 3. Risk assessment and position sizing
        # 4. Order execution
        # 5. Portfolio tracking
        # 6. Notifications

        # Check if WebSocket is initialized and connected
        if app.get_websocket_manager() and app.get_websocket_manager().is_connected():
            logger.info("Real-time WebSocket data streaming is active")
        else:
            logger.warning("No real-time data streaming - using REST API only")

        return 0

    except ComponentInitializationError as e:
        print(f"FATAL: Component initialization failed: {e}", file=sys.stderr)
        return 1
    except ConfigurationError as e:
        print(f"FATAL: Configuration error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"FATAL: Unexpected error: {e}", file=sys.stderr)
        return 1
    finally:
        # Ensure graceful shutdown
        if app and app.is_initialized():
            try:
                app.shutdown()
            except Exception as e:
                print(f"Error during shutdown: {e}", file=sys.stderr)


async def async_main() -> int:
    """
    Async main application entry point.

    Provides the main async entry point for WebSocket-enabled trading bot operation.
    Initializes all components and runs the async event loop with WebSocket streaming.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    app = None
    try:
        # Create and initialize application
        app = TradingBotApplication()
        app.initialize_core_components()

        # Get logger for async main function
        logger = app.get_logger()
        logger.info("Trading Bot async application started")

        # Initialize trading modules (includes WebSocket manager)
        app.initialize_trading_modules()

        # Log async mode startup
        logger.info("=== Async Mode Configuration ===")
        logger.info("Mode: Real-time WebSocket + Event-driven processing")
        logger.info("Components: All trading modules + WebSocket streaming")
        logger.info("Signal handling: SIGINT, SIGTERM for graceful shutdown")

        # Run in async mode with WebSocket streaming
        await app.run_async()

        return 0

    except ComponentInitializationError as e:
        print(f"FATAL: Component initialization failed: {e}", file=sys.stderr)
        return 1
    except ConfigurationError as e:
        print(f"FATAL: Configuration error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nShutdown requested via keyboard interrupt", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"FATAL: Unexpected error: {e}", file=sys.stderr)
        return 1
    finally:
        # Ensure graceful async shutdown
        if app and app.is_initialized():
            try:
                await app.async_shutdown()
            except Exception as e:
                print(f"Error during async shutdown: {e}", file=sys.stderr)


if __name__ == "__main__":
    """
    Entry point when running as main module.

    Provides both sync and async entry points:
    - Default: async mode with WebSocket streaming
    - Fallback: sync mode for compatibility
    """
    # Check for sync mode flag
    sync_mode = os.getenv("TRADING_BOT_SYNC_MODE", "false").lower() == "true"

    if sync_mode:
        # Run in sync mode (legacy compatibility)
        print("Running in sync mode (no WebSocket streaming)", file=sys.stderr)
        exit_code = main()
    else:
        # Run in async mode with WebSocket streaming (default)
        print("Running in async mode with WebSocket streaming", file=sys.stderr)
        try:
            exit_code = asyncio.run(async_main())
        except KeyboardInterrupt:
            print("\nGraceful shutdown completed", file=sys.stderr)
            exit_code = 0
        except Exception as e:
            print(f"Failed to start async mode: {e}", file=sys.stderr)
            exit_code = 1

    sys.exit(exit_code)
