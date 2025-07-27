# Gold Systematic Trading Strategy

A **plain-English** roadmap to go from "Excel dump" ➜ "3-month paper-trading Gold strategy"

## Project Overview

This project implements a simple momentum-based trading strategy for Gold, focusing on practicality and clear results over complexity. The goal is to build a working trading system in approximately one working day (8 hours).

## Quick Start

### Prerequisites

```bash
pip install pandas pyarrow matplotlib
```

### Project Structure

```
gold_project/
├── raw_data/          # Original Excel files and data sources
├── working/           # Processed data and intermediate files
├── results/           # Final outputs and performance reports
└── README.md         # This file
```

## Implementation Roadmap

| **Stage**                             | **What you actually do**                                    | **Time**  | **Why it matters**                                               |
| ------------------------------------- | ----------------------------------------------------------- | --------- | ---------------------------------------------------------------- |
| **1. Organise your files**            | Create folder structure and place Excel data in `raw_data/` | 30 min    | Keeps you from getting lost later                                |
| **2. Turn Excel into usable table**   | Load Excel data, clean it, and save as Parquet format       | 1 hour    | Gives you a **clean daily price table** you can reload instantly |
| **3. Build simple Gold trading rule** | **Rule:** Buy Gold when price > 50-day MA, sell when below  | 2 hours   | Proves you can turn an **idea** into concrete daily P&L          |
| **4. Validate the strategy**          | Calculate total returns and maximum drawdown                | 1 hour    | Basic pass/fail before investing more time                       |
| **5. Paper trade for 3 months**       | Run forward simulation for Apr-Jul 2025 period              | 1-2 hours | Creates a performance track-record                               |
| **6. Create summary report**          | Build 1-page PowerPoint with key metrics and chart          | 2 hours   | Clear, concrete results to present                               |

**Total time: ~1 working day (8 hours)**

## Trading Strategy Details

### Core Rule

- **Long Signal:** Gold price > 50-day moving average
- **Exit Signal:** Gold price < 50-day moving average
- **Position:** 100% allocated when long, 0% (cash) when out

### Key Code Snippet

```python
import pandas as pd

# Load and clean data
df = pd.read_excel('raw_data/prices_1988-2025.xlsx', header=0)
df = df.set_index(df.columns[0])
df = df.ffill().sort_index()
df.to_parquet('working/daily_prices.parquet')

# Generate trading signals
gold = df['GOLDS Index']
ma50 = gold.rolling(50).mean()
in_market = gold > ma50
daily_return = gold.pct_change().shift(-1)
strategy_return = daily_return * in_market.shift(1)
```

## Performance Metrics

The strategy will be evaluated on:

- **Total Return:** Cumulative percentage gain/loss
- **Maximum Drawdown:** Largest peak-to-trough decline
- **Win Rate:** Percentage of profitable trades
- **Sharpe Ratio:** Risk-adjusted returns

## Next Steps

After completing the basic implementation:

1. Add transaction costs modeling
2. Implement position sizing rules
3. Test additional technical indicators
4. Expand to other precious metals
5. Add risk management overlays

## FAQ

| Question                         | Answer                                               |
| -------------------------------- | ---------------------------------------------------- |
| **Do I need fancy libraries?**   | No. `pandas` and `matplotlib` are enough             |
| **What about trading costs?**    | Ignore for proof-of-concept; mention in final report |
| **Cash or short when not long?** | Keep in cash (return = 0) for simplicity             |
| **Position sizing?**             | Start with 100% allocation when signal is active     |

## Getting Started

1. **Create folder structure:**

   ```bash
   mkdir -p raw_data working results
   ```

2. **Install dependencies:**

   ```bash
   pip install pandas pyarrow matplotlib
   ```

3. **Place your Excel file in `raw_data/`**

4. **Run the data processing script to create `working/daily_prices.parquet`**

5. **Implement the trading rule and generate results**

## Contact

For questions or suggestions, please open an issue in this repository.

---

_"The best trading system is the one you actually implement and follow."_
