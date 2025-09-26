# Trading Bot Performance Analysis

This directory contains comprehensive performance analysis tools for the trading bot, including data loading, metric calculation, and visualization capabilities.

## Overview

The performance analysis system processes trade execution logs and portfolio snapshots to generate detailed performance reports with key metrics and visualizations.

## Files

### Core Analysis Tools

- **`analyzer.ipynb`** - Jupyter Notebook with interactive performance analysis
- **`performance_analyzer.py`** - Standalone Python script for automated analysis
- **`generate_sample_data.py`** - Sample data generation for testing

### Generated Files

- **`../exports/`** - Exported analysis results (JSON metrics, CSV data)
- **Plot files** - Generated visualization charts (when `--save-plots` used)

## Features

### Performance Metrics

- **Basic Trading Metrics**
  - Total trades, win rate, profit factor
  - Average win/loss, largest win/loss
  - Expectancy calculation

- **Portfolio Performance**
  - Total and annualized returns
  - Volatility and risk metrics
  - Starting vs ending balance

- **Risk Analysis**
  - Maximum drawdown calculation
  - Average drawdown
  - Sharpe ratio
  - Equity curve analysis

### Visualizations

- **Equity Curve & Drawdown** - Portfolio value over time with drawdown overlay
- **Trade Analysis** - P&L distribution, symbol performance, hourly patterns
- **Portfolio Tracking** - Balance changes, unrealized P&L, position counts
- **Distribution Analysis** - Win/loss ratios, return distributions

## Usage

### Jupyter Notebook (Interactive)

```bash
# Start Jupyter
jupyter notebook analyzer.ipynb

# Follow the notebook cells step by step
# All visualizations will display inline
```

### Python Script (Automated)

```bash
# Basic analysis (console output only)
python3 performance_analyzer.py

# With custom file paths
python3 performance_analyzer.py --trades-csv ../logs/trades.csv --portfolio-csv ../logs/portfolio_history.csv

# Export results to files
python3 performance_analyzer.py --export

# Save plots as PNG files
python3 performance_analyzer.py --save-plots

# Run without displaying plots (for automation)
python3 performance_analyzer.py --no-plots --export
```

### Generate Sample Data

```bash
# Create test data for development/testing
python3 generate_sample_data.py
```

## Data Format

### Trades CSV (`trades.csv`)

Required columns:
- `timestamp` - Trade execution timestamp
- `symbol` - Trading symbol (e.g., BTCUSDT)
- `direction` - BUY or SELL
- `entry_price` - Entry price
- `exit_price` - Exit price
- `size` - Trade size/quantity
- `pnl` - Profit/Loss after fees
- `fees` - Trading fees
- `entry_reason` - Reason for trade entry
- `exit_reason` - Reason for trade exit
- `order_id` - Unique order identifier
- `execution_time` - Order execution time
- `trading_mode` - Trading mode (paper/live)

### Portfolio History CSV (`portfolio_history.csv`)

Required columns:
- `timestamp` - Snapshot timestamp
- `total_balance` - Total portfolio balance
- `unrealized_pnl` - Unrealized P&L
- `open_positions_count` - Number of open positions
- `base_currency` - Base currency (e.g., USDT)
- `snapshot_type` - Type of snapshot (hourly/daily/manual)

## Output Files

### Exported Metrics (`performance_metrics_YYYYMMDD_HHMMSS.json`)

```json
{
  "generated_at": "2025-09-27T07:33:03.777194",
  "basic_metrics": {
    "total_trades": 150,
    "win_rate": 49.33,
    "profit_factor": 0.81,
    "total_pnl": -4118.32,
    "sharpe_ratio": -0.64
  },
  "portfolio_metrics": {
    "total_return": -44.71,
    "annualized_return": -99.94,
    "max_drawdown": -62.75
  }
}
```

### Equity Curve (`equity_curve_YYYYMMDD_HHMMSS.csv`)

- Trade-by-trade equity progression
- Cumulative P&L tracking
- Portfolio balance over time

## Dependencies

### Required Python Packages

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

### System Requirements

- Python 3.8+
- 4GB+ RAM for large datasets
- Display capability for visualizations (or use `--no-plots`)

## Performance Rating System

The analyzer provides an overall performance rating (0-5) based on:

- **Win Rate** > 50% (+1 point)
- **Profit Factor** > 1.5 (+1 point)
- **Total Return** > 10% (+1 point)
- **Sharpe Ratio** > 1.0 (+1 point)
- **Max Drawdown** > -10% (+1 point)

**Ratings:**
- 🌟 EXCELLENT (5/5)
- ⭐ VERY GOOD (4/5)
- ✅ GOOD (3/5)
- ⚠️ FAIR (2/5)
- ❌ POOR (1/5)
- 💀 VERY POOR (0/5)

## Integration

### With Trading Bot

The performance logger in `trading_bot/analysis/performance_logger.py` automatically generates the required CSV files when trades are executed and portfolio snapshots are taken.

### Automated Reports

```bash
# Daily automated analysis
python3 performance_analyzer.py --export --no-plots

# Weekly detailed report with charts
python3 performance_analyzer.py --export --save-plots
```

## Troubleshooting

### Common Issues

1. **File Not Found**: Ensure CSV files exist in `../logs/` directory
2. **Import Errors**: Install required packages with pip
3. **Memory Issues**: Use `--no-plots` for large datasets
4. **Date Format**: Ensure timestamps are ISO format compatible

### Data Validation

The analyzer includes automatic data validation:
- Numeric column type checking
- Timestamp format validation
- Missing data handling
- Outlier detection warnings

## Advanced Usage

### Custom Analysis

Extend the `PerformanceAnalyzer` class for custom metrics:

```python
from performance_analyzer import PerformanceAnalyzer

class CustomAnalyzer(PerformanceAnalyzer):
    def calculate_custom_metric(self):
        # Your custom calculation
        pass
```

### Batch Processing

Process multiple trading periods:

```bash
for file in logs/trades_*.csv; do
    python3 performance_analyzer.py --trades-csv "$file" --export
done
```

## Future Enhancements

- Real-time analysis dashboard
- Advanced risk metrics (VaR, CVaR)
- Benchmark comparison
- Strategy backtesting integration
- Machine learning performance prediction

---

For questions or issues, refer to the trading bot documentation or create an issue in the project repository.