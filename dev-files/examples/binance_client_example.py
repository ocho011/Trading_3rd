"""
Example usage of BinanceClient wrapper.

Demonstrates how to use the BinanceClient with ConfigManager
for trading operations following dependency injection patterns.
"""

from trading_bot.core.config_manager import ConfigManager, EnvConfigLoader
from trading_bot.market_data.binance_client import create_binance_client, BinanceError


def main() -> None:
    """
    Demonstrate BinanceClient usage with proper error handling.
    """
    try:
        # Create configuration manager with environment loader
        config_loader = EnvConfigLoader()
        config_manager = ConfigManager(config_loader)
        config_manager.load_configuration()

        # Create and initialize Binance client
        print("Initializing Binance client...")
        client = create_binance_client(config_manager)

        # Check connection status
        if client.is_connected():
            print("✓ Successfully connected to Binance")

            # Get account information
            print("\nFetching account information...")
            account_info = client.get_account_info()
            print(f"Account status: {account_info.get('status', 'Unknown')}")
            print(f"Account type: {account_info.get('accountType', 'Unknown')}")

            # Get ticker information for BTC/USDT
            print("\nFetching BTC/USDT ticker...")
            ticker = client.get_symbol_ticker("BTCUSDT")
            print(f"BTC/USDT Price: {ticker.get('price', 'N/A')}")

            # Get exchange information
            print("\nFetching exchange information...")
            exchange_info = client.get_exchange_info("BTCUSDT")
            if exchange_info:
                print(f"Symbol: {exchange_info.get('symbol', 'N/A')}")
                print(f"Status: {exchange_info.get('status', 'N/A')}")

            # Example: Place a small test order (commented out for safety)
            # WARNING: Uncomment only if you want to place real orders
            # print("\nPlacing test market order...")
            # order_result = client.place_market_order("BTCUSDT", "BUY", "0.001")
            # print(f"Order ID: {order_result.get('orderId', 'N/A')}")

        else:
            print("✗ Failed to connect to Binance")

    except BinanceError as e:
        print(f"Binance error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
