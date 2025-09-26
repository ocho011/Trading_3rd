#!/usr/bin/env python3
"""
Trading Bot Performance Analyzer

A comprehensive performance analysis script for the trading bot that loads
trade execution logs and portfolio snapshots to generate detailed performance
metrics and visualizations.

Usage:
    python performance_analyzer.py [--trades-csv TRADES_CSV] [--portfolio-csv PORTFOLIO_CSV] [--export]

Example:
    python performance_analyzer.py --trades-csv ../logs/trades.csv --portfolio-csv ../logs/portfolio_history.csv --export
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class PerformanceAnalyzer:
    """
    Comprehensive trading performance analyzer.

    This class provides methods to load trade and portfolio data,
    calculate performance metrics, and generate visualizations.
    """

    def __init__(self, trades_csv: str, portfolio_csv: str):
        """
        Initialize the Performance Analyzer.

        Args:
            trades_csv: Path to trades CSV file
            portfolio_csv: Path to portfolio history CSV file
        """
        self.trades_csv = trades_csv
        self.portfolio_csv = portfolio_csv
        self.risk_free_rate = 0.02  # 2% annual risk-free rate

        # Data containers
        self.trades_df = pd.DataFrame()
        self.portfolio_df = pd.DataFrame()
        self.equity_curve = pd.DataFrame()

        # Metrics containers
        self.basic_metrics = {}
        self.portfolio_metrics = {}
        self.drawdown_metrics = {}
        self.sharpe_ratio = 0.0

    def load_data(self) -> bool:
        """
        Load and preprocess both trade and portfolio data.

        Returns:
            True if data loaded successfully, False otherwise
        """
        try:
            # Load trades data
            if Path(self.trades_csv).exists():
                self.trades_df = self._load_trades_data()
                print(f"✅ Loaded {len(self.trades_df)} trades")
            else:
                print(f"❌ Trades file not found: {self.trades_csv}")
                return False

            # Load portfolio data
            if Path(self.portfolio_csv).exists():
                self.portfolio_df = self._load_portfolio_data()
                print(f"✅ Loaded {len(self.portfolio_df)} portfolio snapshots")
            else:
                print(f"❌ Portfolio file not found: {self.portfolio_csv}")
                return False

            return True

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def _load_trades_data(self) -> pd.DataFrame:
        """Load and preprocess trade data from CSV file."""
        df = pd.read_csv(self.trades_csv)

        # Convert timestamp columns to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['execution_time'] = pd.to_datetime(df['execution_time'])

        # Ensure numeric columns are properly typed
        numeric_columns = ['entry_price', 'exit_price', 'size', 'pnl', 'fees']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Add derived columns
        df['trade_value'] = df['size'] * df['entry_price']
        df['gross_pnl'] = df['pnl'] + df['fees']  # PnL before fees
        df['return_pct'] = (df['exit_price'] - df['entry_price']) / df['entry_price'] * 100
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()

        # Classify trades as winners/losers
        df['trade_result'] = df['pnl'].apply(
            lambda x: 'Win' if x > 0 else 'Loss' if x < 0 else 'Breakeven'
        )

        return df

    def _load_portfolio_data(self) -> pd.DataFrame:
        """Load and preprocess portfolio history data from CSV file."""
        df = pd.read_csv(self.portfolio_csv)

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Ensure numeric columns are properly typed
        numeric_columns = ['total_balance', 'unrealized_pnl', 'open_positions_count']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Add derived columns
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour

        # Calculate returns
        df['balance_change'] = df['total_balance'].diff()
        df['balance_change_pct'] = df['total_balance'].pct_change() * 100

        return df

    def calculate_metrics(self) -> None:
        """Calculate all performance metrics."""
        print("🔄 Calculating performance metrics...")

        self.basic_metrics = self._calculate_basic_metrics()
        self.equity_curve = self._calculate_equity_curve()
        self.drawdown_metrics = self._calculate_drawdown()
        self.sharpe_ratio = self._calculate_sharpe_ratio()
        self.portfolio_metrics = self._calculate_portfolio_metrics()

        print("✅ Performance metrics calculated")

    def _calculate_basic_metrics(self) -> Dict[str, float]:
        """Calculate basic trading metrics."""
        if self.trades_df.empty:
            return {}

        total_trades = len(self.trades_df)
        winning_trades = len(self.trades_df[self.trades_df['pnl'] > 0])
        losing_trades = len(self.trades_df[self.trades_df['pnl'] < 0])

        total_pnl = self.trades_df['pnl'].sum()
        total_fees = self.trades_df['fees'].sum()
        gross_pnl = total_pnl + total_fees

        avg_win = self.trades_df[self.trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = self.trades_df[self.trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

        largest_win = self.trades_df['pnl'].max()
        largest_loss = self.trades_df['pnl'].min()

        # Win rate
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        # Profit factor
        gross_profit = self.trades_df[self.trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(self.trades_df[self.trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Expectancy
        expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'gross_pnl': gross_pnl,
            'total_fees': total_fees,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy
        }

    def _calculate_equity_curve(self) -> pd.DataFrame:
        """Calculate equity curve from trades."""
        if self.trades_df.empty:
            return pd.DataFrame()

        # Sort trades by timestamp
        trades_sorted = self.trades_df.sort_values('timestamp').copy()

        # Calculate cumulative PnL
        trades_sorted['cumulative_pnl'] = trades_sorted['pnl'].cumsum()

        # Assume starting balance
        starting_balance = 10000
        trades_sorted['equity'] = starting_balance + trades_sorted['cumulative_pnl']

        return trades_sorted[['timestamp', 'cumulative_pnl', 'equity']]

    def _calculate_drawdown(self) -> Dict[str, float]:
        """Calculate maximum drawdown and related metrics."""
        if self.equity_curve.empty:
            return {}

        # Calculate running maximum (peak)
        equity_curve = self.equity_curve.copy()
        equity_curve['peak'] = equity_curve['equity'].cummax()

        # Calculate drawdown
        equity_curve['drawdown'] = (equity_curve['equity'] - equity_curve['peak']) / equity_curve['peak'] * 100

        max_drawdown = equity_curve['drawdown'].min()
        max_drawdown_idx = equity_curve['drawdown'].idxmin()
        max_drawdown_date = equity_curve.loc[max_drawdown_idx, 'timestamp']

        # Calculate average drawdown
        avg_drawdown = equity_curve[equity_curve['drawdown'] < 0]['drawdown'].mean()
        avg_drawdown = avg_drawdown if not pd.isna(avg_drawdown) else 0

        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_date': max_drawdown_date,
            'avg_drawdown': avg_drawdown,
            'drawdown_series': equity_curve['drawdown']
        }

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from equity curve."""
        if self.equity_curve.empty or len(self.equity_curve) < 2:
            return 0

        # Calculate daily returns
        equity_curve = self.equity_curve.copy()
        equity_curve['daily_return'] = equity_curve['equity'].pct_change()

        # Remove NaN values
        daily_returns = equity_curve['daily_return'].dropna()

        if len(daily_returns) < 2:
            return 0

        # Calculate average return and standard deviation
        avg_return = daily_returns.mean()
        std_return = daily_returns.std()

        if std_return == 0:
            return 0

        # Convert risk-free rate to daily
        daily_rf_rate = self.risk_free_rate / 365

        # Calculate Sharpe ratio
        sharpe_ratio = (avg_return - daily_rf_rate) / std_return

        # Annualize
        return sharpe_ratio * np.sqrt(365)

    def _calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate metrics from portfolio history."""
        if self.portfolio_df.empty:
            return {}

        starting_balance = self.portfolio_df['total_balance'].iloc[0]
        ending_balance = self.portfolio_df['total_balance'].iloc[-1]

        total_return = ((ending_balance - starting_balance) / starting_balance) * 100

        # Calculate time period
        start_date = self.portfolio_df['timestamp'].iloc[0]
        end_date = self.portfolio_df['timestamp'].iloc[-1]
        days = (end_date - start_date).days

        # Annualized return
        annualized_return = ((ending_balance / starting_balance) ** (365 / days) - 1) * 100 if days > 0 else 0

        # Volatility
        volatility = self.portfolio_df['balance_change_pct'].std() * np.sqrt(365) if not self.portfolio_df['balance_change_pct'].isna().all() else 0

        return {
            'starting_balance': starting_balance,
            'ending_balance': ending_balance,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'days_tracked': days
        }

    def print_performance_report(self) -> None:
        """Print comprehensive performance report."""
        print("=" * 80)
        print("                    TRADING PERFORMANCE REPORT")
        print("=" * 80)

        # Basic Trading Metrics
        print("\\n📊 BASIC TRADING METRICS")
        print("-" * 40)
        if self.basic_metrics:
            print(f"Total Trades:        {self.basic_metrics['total_trades']:,}")
            print(f"Winning Trades:      {self.basic_metrics['winning_trades']:,} ({self.basic_metrics['win_rate']:.1f}%)")
            print(f"Losing Trades:       {self.basic_metrics['losing_trades']:,}")
            print(f"Win Rate:            {self.basic_metrics['win_rate']:.1f}%")
            print(f"Profit Factor:       {self.basic_metrics['profit_factor']:.2f}")
            print(f"Expectancy:          ${self.basic_metrics['expectancy']:.2f}")

        # Financial Performance
        print("\\n💰 FINANCIAL PERFORMANCE")
        print("-" * 40)
        if self.basic_metrics:
            print(f"Total P&L:           ${self.basic_metrics['total_pnl']:.2f}")
            print(f"Gross P&L:           ${self.basic_metrics['gross_pnl']:.2f}")
            print(f"Total Fees:          ${self.basic_metrics['total_fees']:.2f}")
            print(f"Average Win:         ${self.basic_metrics['avg_win']:.2f}")
            print(f"Average Loss:        ${self.basic_metrics['avg_loss']:.2f}")
            print(f"Largest Win:         ${self.basic_metrics['largest_win']:.2f}")
            print(f"Largest Loss:        ${self.basic_metrics['largest_loss']:.2f}")

        # Portfolio Metrics
        print("\\n📈 PORTFOLIO PERFORMANCE")
        print("-" * 40)
        if self.portfolio_metrics:
            print(f"Starting Balance:    ${self.portfolio_metrics['starting_balance']:,.2f}")
            print(f"Ending Balance:      ${self.portfolio_metrics['ending_balance']:,.2f}")
            print(f"Total Return:        {self.portfolio_metrics['total_return']:.2f}%")
            print(f"Annualized Return:   {self.portfolio_metrics['annualized_return']:.2f}%")
            print(f"Volatility:          {self.portfolio_metrics['volatility']:.2f}%")
            print(f"Days Tracked:        {self.portfolio_metrics['days_tracked']}")

        # Risk Metrics
        print("\\n⚠️  RISK METRICS")
        print("-" * 40)
        if self.drawdown_metrics:
            print(f"Max Drawdown:        {self.drawdown_metrics['max_drawdown']:.2f}%")
            if 'max_drawdown_date' in self.drawdown_metrics:
                print(f"Max DD Date:         {self.drawdown_metrics['max_drawdown_date']}")
            print(f"Avg Drawdown:        {self.drawdown_metrics['avg_drawdown']:.2f}%")
        print(f"Sharpe Ratio:        {self.sharpe_ratio:.2f}")

        # Performance Rating
        print("\\n🏆 PERFORMANCE RATING")
        print("-" * 40)

        rating_score = 0
        if self.basic_metrics and self.basic_metrics['win_rate'] > 50:
            rating_score += 1
        if self.basic_metrics and self.basic_metrics['profit_factor'] > 1.5:
            rating_score += 1
        if self.portfolio_metrics and self.portfolio_metrics['total_return'] > 10:
            rating_score += 1
        if self.sharpe_ratio > 1.0:
            rating_score += 1
        if self.drawdown_metrics and self.drawdown_metrics['max_drawdown'] > -10:
            rating_score += 1

        ratings = {
            5: "🌟 EXCELLENT",
            4: "⭐ VERY GOOD",
            3: "✅ GOOD",
            2: "⚠️  FAIR",
            1: "❌ POOR",
            0: "💀 VERY POOR"
        }

        print(f"Overall Rating:      {ratings.get(rating_score, '❓ UNKNOWN')} ({rating_score}/5)")
        print("\\n" + "=" * 80)

    def generate_visualizations(self, save_plots: bool = False) -> None:
        """Generate comprehensive visualizations."""
        print("🎨 Generating visualizations...")

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Create equity curve plot
        self._plot_equity_curve(save_plots)

        # Create trade analysis plots
        self._plot_trade_analysis(save_plots)

        # Create portfolio analysis plots
        self._plot_portfolio_analysis(save_plots)

        print("✅ Visualizations generated")

    def _plot_equity_curve(self, save_plots: bool = False) -> None:
        """Create equity curve and drawdown visualization."""
        if self.equity_curve.empty:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Equity curve
        ax1.plot(self.equity_curve['timestamp'], self.equity_curve['equity'],
                linewidth=2, color='blue', label='Equity')
        ax1.set_title('Equity Curve', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Balance ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Add starting balance line
        if self.portfolio_metrics:
            ax1.axhline(y=self.portfolio_metrics['starting_balance'],
                       color='gray', linestyle='--', alpha=0.7, label='Starting Balance')
            ax1.legend()

        # Drawdown
        if 'drawdown_series' in self.drawdown_metrics:
            ax2.fill_between(self.equity_curve['timestamp'],
                           self.drawdown_metrics['drawdown_series'], 0,
                           color='red', alpha=0.3, label='Drawdown')
            ax2.plot(self.equity_curve['timestamp'],
                    self.drawdown_metrics['drawdown_series'],
                    color='red', linewidth=1)
            ax2.set_title('Drawdown', fontsize=16, fontweight='bold')
            ax2.set_ylabel('Drawdown (%)', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend()

        plt.tight_layout()

        if save_plots:
            plt.savefig('equity_curve.png', dpi=300, bbox_inches='tight')

        plt.show()

    def _plot_trade_analysis(self, save_plots: bool = False) -> None:
        """Create comprehensive trade analysis visualizations."""
        if self.trades_df.empty:
            return

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # 1. P&L Distribution
        axes[0, 0].hist(self.trades_df['pnl'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(self.trades_df['pnl'].mean(), color='red', linestyle='--',
                          label=f'Mean: ${self.trades_df["pnl"].mean():.2f}')
        axes[0, 0].set_title('P&L Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('P&L ($)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. P&L by Symbol
        symbol_pnl = self.trades_df.groupby('symbol')['pnl'].sum().sort_values(ascending=False)
        colors = ['green' if x > 0 else 'red' for x in symbol_pnl.values]
        axes[0, 1].bar(symbol_pnl.index, symbol_pnl.values, color=colors, alpha=0.7)
        axes[0, 1].set_title('P&L by Symbol', fontweight='bold')
        axes[0, 1].set_xlabel('Symbol')
        axes[0, 1].set_ylabel('Total P&L ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Trade Count by Symbol
        symbol_counts = self.trades_df['symbol'].value_counts()
        axes[0, 2].pie(symbol_counts.values, labels=symbol_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 2].set_title('Trade Distribution by Symbol', fontweight='bold')

        # 4. Win/Loss Ratio
        trade_results = self.trades_df['trade_result'].value_counts()
        colors_result = ['green', 'red', 'gray']
        axes[1, 0].pie(trade_results.values, labels=trade_results.index, autopct='%1.1f%%',
                      colors=colors_result, startangle=90)
        axes[1, 0].set_title('Win/Loss Distribution', fontweight='bold')

        # 5. Trades by Hour of Day
        hourly_trades = self.trades_df['hour'].value_counts().sort_index()
        axes[1, 1].bar(hourly_trades.index, hourly_trades.values, alpha=0.7, color='orange')
        axes[1, 1].set_title('Trades by Hour of Day', fontweight='bold')
        axes[1, 1].set_xlabel('Hour')
        axes[1, 1].set_ylabel('Number of Trades')
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Cumulative P&L over Time
        trades_sorted = self.trades_df.sort_values('timestamp')
        cumulative_pnl = trades_sorted['pnl'].cumsum()
        axes[1, 2].plot(trades_sorted['timestamp'], cumulative_pnl, linewidth=2, color='purple')
        axes[1, 2].set_title('Cumulative P&L over Time', fontweight='bold')
        axes[1, 2].set_xlabel('Date')
        axes[1, 2].set_ylabel('Cumulative P&L ($)')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_plots:
            plt.savefig('trade_analysis.png', dpi=300, bbox_inches='tight')

        plt.show()

    def _plot_portfolio_analysis(self, save_plots: bool = False) -> None:
        """Create portfolio performance visualizations."""
        if self.portfolio_df.empty:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # 1. Portfolio Balance over Time
        axes[0, 0].plot(self.portfolio_df['timestamp'], self.portfolio_df['total_balance'],
                       linewidth=2, color='blue', label='Total Balance')
        axes[0, 0].set_title('Portfolio Balance over Time', fontweight='bold')
        axes[0, 0].set_ylabel('Balance ($)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. Unrealized P&L over Time
        axes[0, 1].plot(self.portfolio_df['timestamp'], self.portfolio_df['unrealized_pnl'],
                       linewidth=2, color='orange', label='Unrealized P&L')
        axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Unrealized P&L over Time', fontweight='bold')
        axes[0, 1].set_ylabel('Unrealized P&L ($)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. Open Positions Count
        axes[1, 0].plot(self.portfolio_df['timestamp'], self.portfolio_df['open_positions_count'],
                       linewidth=2, color='green', marker='o', markersize=2, label='Open Positions')
        axes[1, 0].set_title('Open Positions over Time', fontweight='bold')
        axes[1, 0].set_ylabel('Number of Positions')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. Daily Returns Distribution
        daily_returns = self.portfolio_df['balance_change_pct'].dropna()
        if not daily_returns.empty:
            axes[1, 1].hist(daily_returns, bins=30, alpha=0.7, color='purple', edgecolor='black')
            axes[1, 1].axvline(daily_returns.mean(), color='red', linestyle='--',
                              label=f'Mean: {daily_returns.mean():.3f}%')
            axes[1, 1].set_title('Daily Returns Distribution', fontweight='bold')
            axes[1, 1].set_xlabel('Daily Return (%)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            plt.savefig('portfolio_analysis.png', dpi=300, bbox_inches='tight')

        plt.show()

    def export_results(self, output_dir: str = "../exports") -> None:
        """Export analysis results to files."""
        try:
            # Create exports directory
            export_dir = Path(output_dir)
            export_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Export metrics to JSON
            all_metrics = {
                'generated_at': datetime.now().isoformat(),
                'basic_metrics': self.basic_metrics,
                'portfolio_metrics': self.portfolio_metrics,
                'drawdown_metrics': {k: v for k, v in self.drawdown_metrics.items() if k != 'drawdown_series'},
                'sharpe_ratio': self.sharpe_ratio
            }

            metrics_file = export_dir / f"performance_metrics_{timestamp}.json"

            with open(metrics_file, 'w') as f:
                json.dump(all_metrics, f, indent=2, default=str)

            # Export equity curve
            if not self.equity_curve.empty:
                equity_file = export_dir / f"equity_curve_{timestamp}.csv"
                self.equity_curve.to_csv(equity_file, index=False)
                print(f"✅ Equity curve exported to: {equity_file}")

            print(f"✅ Performance metrics exported to: {metrics_file}")
            print(f"📁 All exports saved to: {export_dir.absolute()}")

        except Exception as e:
            print(f"❌ Error exporting results: {e}")


def main():
    """Main function to run the performance analysis."""
    parser = argparse.ArgumentParser(
        description="Trading Bot Performance Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --trades-csv ../logs/trades.csv --portfolio-csv ../logs/portfolio_history.csv
  %(prog)s --export --save-plots
        """
    )

    parser.add_argument(
        '--trades-csv',
        default='../logs/trades.csv',
        help='Path to trades CSV file (default: ../logs/trades.csv)'
    )

    parser.add_argument(
        '--portfolio-csv',
        default='../logs/portfolio_history.csv',
        help='Path to portfolio history CSV file (default: ../logs/portfolio_history.csv)'
    )

    parser.add_argument(
        '--export',
        action='store_true',
        help='Export results to files'
    )

    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save plots as PNG files'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots (useful for automated analysis)'
    )

    args = parser.parse_args()

    print("🚀 Starting Trading Bot Performance Analysis")
    print("=" * 60)

    # Initialize analyzer
    analyzer = PerformanceAnalyzer(args.trades_csv, args.portfolio_csv)

    # Load data
    if not analyzer.load_data():
        print("❌ Failed to load data. Exiting.")
        sys.exit(1)

    # Calculate metrics
    analyzer.calculate_metrics()

    # Print report
    analyzer.print_performance_report()

    # Generate visualizations
    if not args.no_plots:
        analyzer.generate_visualizations(save_plots=args.save_plots)

    # Export results
    if args.export:
        analyzer.export_results()

    print("\\n✅ Performance analysis completed successfully!")


if __name__ == "__main__":
    main()