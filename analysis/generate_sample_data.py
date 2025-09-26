"""
Generate sample CSV data for performance analysis testing.

This script creates realistic sample trade and portfolio data for testing
the performance analysis system.
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any


def generate_sample_trades(num_trades: int = 100) -> List[Dict[str, Any]]:
    """
    Generate sample trade data for testing.

    Args:
        num_trades: Number of trades to generate

    Returns:
        List of trade records
    """
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT", "SOLUSDT"]
    directions = ["BUY", "SELL"]
    entry_reasons = ["signal_generated", "momentum_breakout", "reversal_signal", "dca_entry"]
    exit_reasons = ["stop_loss", "take_profit", "signal_exit", "time_exit"]

    trades = []
    base_time = datetime.now() - timedelta(days=30)

    # Starting balance for PnL calculation
    balance = 10000.0

    for i in range(num_trades):
        # Random time increment
        trade_time = base_time + timedelta(hours=random.uniform(1, 24))
        base_time = trade_time

        symbol = random.choice(symbols)
        direction = random.choice(directions)

        # Generate realistic prices based on symbol
        if symbol.startswith("BTC"):
            base_price = random.uniform(40000, 70000)
        elif symbol.startswith("ETH"):
            base_price = random.uniform(2000, 4000)
        else:
            base_price = random.uniform(0.5, 50)

        # Entry and exit prices with some spread
        entry_price = base_price
        price_change = random.uniform(-0.05, 0.05)  # -5% to +5% price change
        exit_price = entry_price * (1 + price_change)

        # Trade size (realistic for the symbol)
        if symbol.startswith("BTC"):
            size = random.uniform(0.01, 0.5)
        elif symbol.startswith("ETH"):
            size = random.uniform(0.1, 2.0)
        else:
            size = random.uniform(10, 1000)

        # Calculate PnL
        if direction == "BUY":
            pnl = (exit_price - entry_price) * size
        else:
            pnl = (entry_price - exit_price) * size

        # Add some realistic fees
        fees = (entry_price * size) * 0.001  # 0.1% fee
        pnl -= fees

        # Update balance
        balance += pnl

        trade = {
            "timestamp": trade_time.isoformat(),
            "symbol": symbol,
            "direction": direction,
            "entry_price": round(entry_price, 6),
            "exit_price": round(exit_price, 6),
            "size": round(size, 6),
            "pnl": round(pnl, 4),
            "entry_reason": random.choice(entry_reasons),
            "exit_reason": random.choice(exit_reasons),
            "order_id": f"order_{i+1:06d}",
            "execution_time": trade_time.isoformat(),
            "fees": round(fees, 4),
            "trading_mode": "paper"
        }

        trades.append(trade)

    return trades


def generate_sample_portfolio_history(num_snapshots: int = 720) -> List[Dict[str, Any]]:
    """
    Generate sample portfolio history data.

    Args:
        num_snapshots: Number of portfolio snapshots (default 720 = 30 days hourly)

    Returns:
        List of portfolio records
    """
    portfolio_history = []
    base_time = datetime.now() - timedelta(days=30)
    starting_balance = 10000.0
    current_balance = starting_balance

    for i in range(num_snapshots):
        # Hourly snapshots
        snapshot_time = base_time + timedelta(hours=i)

        # Simulate portfolio growth with some volatility
        daily_return = random.normalvariate(0.0002, 0.02)  # ~0.02% daily return with 2% volatility
        current_balance *= (1 + daily_return)

        # Random unrealized PnL
        unrealized_pnl = random.uniform(-100, 200)

        # Random number of open positions
        open_positions = random.randint(0, 5)

        # Different snapshot types
        snapshot_types = ["hourly", "daily", "manual"]
        if i % 24 == 0:
            snapshot_type = "daily"
        elif i % 168 == 0:  # Weekly
            snapshot_type = "manual"
        else:
            snapshot_type = "hourly"

        portfolio_record = {
            "timestamp": snapshot_time.isoformat(),
            "total_balance": round(current_balance, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "open_positions_count": open_positions,
            "base_currency": "USDT",
            "snapshot_type": snapshot_type
        }

        portfolio_history.append(portfolio_record)

    return portfolio_history


def save_csv_data(data: List[Dict[str, Any]], filepath: Path, fieldnames: List[str]) -> None:
    """
    Save data to CSV file.

    Args:
        data: List of records to save
        filepath: Path to save CSV file
        fieldnames: CSV column names
    """
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Generated {len(data)} records in {filepath}")


def main():
    """Generate and save sample data files."""
    # Define paths
    base_dir = Path(__file__).parent.parent
    logs_dir = base_dir / "logs"

    # Trade CSV fieldnames (from PerformanceLogger)
    trade_fieldnames = [
        "timestamp", "symbol", "direction", "entry_price", "exit_price",
        "size", "pnl", "entry_reason", "exit_reason", "order_id",
        "execution_time", "fees", "trading_mode"
    ]

    # Portfolio CSV fieldnames (from PerformanceLogger)
    portfolio_fieldnames = [
        "timestamp", "total_balance", "unrealized_pnl",
        "open_positions_count", "base_currency", "snapshot_type"
    ]

    # Generate sample data
    print("Generating sample trade data...")
    trades = generate_sample_trades(150)  # 150 trades over 30 days

    print("Generating sample portfolio history...")
    portfolio_history = generate_sample_portfolio_history(720)  # 720 hourly snapshots

    # Save to CSV files
    trades_path = logs_dir / "trades.csv"
    portfolio_path = logs_dir / "portfolio_history.csv"

    save_csv_data(trades, trades_path, trade_fieldnames)
    save_csv_data(portfolio_history, portfolio_path, portfolio_fieldnames)

    print("\nSample data generation complete!")
    print(f"Trades CSV: {trades_path}")
    print(f"Portfolio CSV: {portfolio_path}")


if __name__ == "__main__":
    main()