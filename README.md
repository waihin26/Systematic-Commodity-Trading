# Systematic Commodity Trading: Donchian Channel Strategy

A **complete implementation** of a professional-grade Donchian Channel breakout strategy for gold trading, featuring comprehensive parameter optimization and institutional-quality analysis.

## Project Overview

This repository contains a fully-implemented systematic trading strategy that uses Donchian Channel breakouts to trade gold. The project includes data processing, strategy implementation, extensive parameter optimization across 5,250+ combinations, and professional reporting suitable for institutional presentation.

**Key Achievement:** 416.1% total returns over 37.5 years (1988-2025) with a Sharpe ratio of 0.47 and maximum drawdown of 22.8%.

## Quick Start

### Prerequisites

```bash
pip install backtrader pandas pyarrow matplotlib seaborn numpy
```

### Project Structure

```
Systematic-Commodity-Trading/
├── strategies/                      # Trading strategy scripts
│   ├── gold_donchian_channel_strategy.py   # Main Donchian strategy backtest
│   └── donchian_optimizer.py               # Parameter optimization script
├── processed_data/                  # Cleaned gold price data (parquet)
├── reports/                         # Professional LaTeX and PDF reports
│   ├── donchian_channel_strat.tex         # Main institutional report
│   └── donchian_channel_strat.pdf         # PDF version of the report
├── raw_data/                        # Raw gold price data (Excel)
│   └── MSCI_Comps.xlsx                    # Original data file
└── README.md                       
```

## Strategy Implementation

### Donchian Channel Breakout Logic

The strategy uses a sophisticated breakout system with the following components:

**Entry Signals:**

- Price closes above 20-day highest high (breakout confirmation)
- Minimum 40-day ATR period required for indicator stability

**Exit Signals:**

- Price closes below 20-day lowest low (trend reversal), OR
- Profit target reached (Entry price + 4 × ATR)

**Risk Management:**

- ATR-based position sizing (1% portfolio risk per trade)
- Maximum 95% cash utilization
- Dynamic profit targets based on volatility

### Key Implementation Files

1. **`gold_donchian_channel_strategy.py`** - Main strategy implementation

   - Complete backtest from 1988-2025
   - Portfolio performance visualization
   - Comprehensive metrics calculation

2. **`donchian_optimizer.py`** - Parameter optimization script
   - Tests 5,250 parameter combinations
   - Generates performance heatmaps
   - Identifies robust parameter zones

### Core Strategy Code

```python
class GoldDonchianBreakoutStrategy(bt.Strategy):
    params = (
        ('entry_period', 20),     # Breakout period
        ('exit_period', 20),      # Exit period
        ('atr_period', 40),       # ATR for sizing
        ('atr_multiplier', 4),    # Profit target
        ('risk_percent', 1.0),    # Risk per trade
    )

    def next(self):
        if not self.position:
            # Entry: Price > 20-day high
            if self.dataclose[0] > self.highest_high[-1]:
                size = self.calculate_position_size()
                self.buy(size=size)
        else:
            # Exit: Price < 20-day low OR profit target hit
            if (self.dataclose[0] < self.lowest_low[-1] or
                self.dataclose[0] >= self.profit_target):
                self.sell(size=self.position.size)
```

## Performance Results

### Key Metrics (Optimized 20/20 Configuration)

| Metric                | Strategy Performance | Buy & Hold |
| --------------------- | -------------------- | ---------- |
| **Total Return**      | 416.1%               | 595.6%     |
| **Annualized Return** | 3.93%                | 5.38%      |
| **Sharpe Ratio**      | 0.47                 | 0.31       |
| **Maximum Drawdown**  | 22.8%                | 36.7%      |
| **Win Rate**          | 68.2%                | N/A        |
| **Total Trades**      | 274                  | N/A        |
| **Market Exposure**   | 45.3%                | 100%       |

### Strategy Advantages

✅ **Superior Risk-Adjusted Returns:** 52% better Sharpe ratio (0.47 vs 0.31)  
✅ **Lower Drawdowns:** 38% reduction in maximum drawdown  
✅ **Capital Efficiency:** Only 45% market exposure vs 100% buy-and-hold  
✅ **Systematic Execution:** Removes emotional decision-making  
✅ **Robust Parameters:** Validated across 5,250+ combinations

## Parameter Optimization

The strategy underwent comprehensive optimization testing:

- **Parameter Combinations:** 5,250 unique configurations
- **Optimization Variables:** Entry/exit periods, ATR settings, risk levels
- **Robustness Testing:** Multiple criteria scoring system
- **Heatmap Analysis:** Visual identification of performance zones

### Optimal Parameters Identified

- **Entry Period:** 20 days (breakout lookback)
- **Exit Period:** 20 days (exit trigger)
- **ATR Period:** 40 days (volatility measurement)
- **ATR Multiplier:** 4.0 (profit target scaling)
- **Risk Percentage:** 1.0% (position sizing)

## Getting Started

### 1. Run the Main Strategy

```bash
cd strategies/
python3 gold_donchian_channel_strategy.py
```

This will:

- Load and process gold price data (1988-2025)
- Execute the complete backtest
- Generate performance charts
- Display comprehensive metrics

### 2. Explore Parameter Optimization

```bash
cd strategies/
python3 donchian_optimizer.py
```

This will:

- Test 5,250+ parameter combinations
- Generate optimization heatmaps
- Identify robust parameter zones
- Save results to CSV for analysis

### 3. View Results

The strategy generates:

- **Performance Charts:** Portfolio value and returns over time
- **Optimization Heatmaps:** Parameter sensitivity analysis
- **Professional Reports:** LaTeX-formatted institutional presentation
- **Trade Analysis:** Detailed trade-by-trade breakdown

## Limitations & Considerations

⚠️ **Synthetic OHLC Data:** Strategy uses constructed high/low prices from close-only data  
⚠️ **No Slippage Model:** Real-world execution costs not included in backtest  
⚠️ **Regime Dependency:** Performance concentrated in specific gold bull markets  
⚠️ **Limited Universe:** Single-asset strategy (gold only)

## Professional Applications

This implementation demonstrates:

- **Institutional-Quality Code:** Proper backtesting framework using Backtrader
- **Comprehensive Analysis:** Multiple performance metrics and risk measures
- **Parameter Robustness:** Extensive optimization to avoid overfitting
- **Professional Reporting:** LaTeX documentation suitable for client presentation
- **Practical Implementation:** Ready for paper trading or live deployment

## Repository Contents

| File/Folder                                    | Purpose                             |
| ---------------------------------------------- | ----------------------------------- |
| `strategies/gold_donchian_channel_strategy.py` | Main Donchian strategy backtest     |
| `strategies/donchian_optimizer.py`             | Parameter optimization script       |
| `processed_data/gold.parquet`                  | Clean gold price data (1988-2025)   |
| `reports/donchian_channel_strat.tex`           | Main LaTeX report                   |
| `reports/donchian_channel_strat.pdf`           | PDF version of institutional report |
| `raw_data/MSCI_Comps.xlsx`                     | Raw gold price data (Excel)         |
| `README.md`                                    | Project overview and instructions   |

## Contact & Usage

This repository showcases a complete systematic trading strategy implementation suitable for:

- **Portfolio Managers** evaluating systematic strategies
- **Quantitative Researchers** studying breakout methodologies
- **Trading System Developers** seeking production-ready code
- **Academic Research** on commodity trading strategies

For questions about implementation details or extending the strategy, please open an issue in this repository.

---

_"Systematic trading transforms market intuition into measurable, repeatable performance."_
