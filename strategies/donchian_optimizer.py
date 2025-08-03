"""
Donchian Channel Parameter Optimizer
===================================

This script runs a comprehensive grid search to find optimal parameters for the
Donchian Channel breakout strategy. It tests multiple combinations of:
- Entry periods (breakout lookback)
- Exit periods (exit lookback)
- ATR periods (volatility measurement)
- ATR multipliers (profit targets)
- Risk percentages (position sizing)

The results are visualized in heatmaps and tables to identify robust parameter zones.

Usage:
    python donchian_optimizer.py
"""

import backtrader as bt
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import warnings
warnings.filterwarnings('ignore')


class OptimizedDonchianStrategy(bt.Strategy):
    """Streamlined Donchian strategy for optimization"""
    
    params = (
        ('entry_period', 20),
        ('exit_period', 10), 
        ('atr_period', 20),
        ('atr_multiplier', 3.0),
        ('risk_percent', 2.0),
    )
    
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        
        # Indicators
        self.highest_high = bt.indicators.Highest(
            self.dataclose, period=self.params.entry_period
        )
        self.lowest_low = bt.indicators.Lowest(
            self.dataclose, period=self.params.exit_period
        )
        self.atr = bt.indicators.AverageTrueRange(
            self.datas[0], period=self.params.atr_period
        )
        
        # Tracking
        self.entry_price = None
        self.profit_target = None
        
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
                self.profit_target = self.entry_price + (self.params.atr_multiplier * self.atr[0])
            else:
                self.entry_price = None
                self.profit_target = None
        self.order = None
        
    def next(self):
        if len(self.data) < max(self.params.entry_period, self.params.atr_period):
            return
            
        if self.order:
            return
            
        current_price = self.dataclose[0]
        
        if not self.position:
            # Entry signal
            if current_price > self.highest_high[-1]:
                atr_value = self.atr[0]
                portfolio_value = self.broker.getvalue()
                risk_amount = portfolio_value * (self.params.risk_percent / 100)
                
                if atr_value > 0:
                    position_value = risk_amount / atr_value * current_price
                    size = int(position_value / current_price)
                    max_size = int(self.broker.getcash() * 0.95 / current_price)
                    size = min(size, max_size)
                    
                    if size > 0:
                        self.order = self.buy(size=size)
        else:
            # Exit signals
            should_exit = False
            
            if current_price < self.lowest_low[-1]:
                should_exit = True
            elif self.profit_target and current_price >= self.profit_target:
                should_exit = True
                
            if should_exit:
                self.order = self.sell(size=self.position.size)


def load_gold_data():
    """Load and prepare gold data for optimization"""
    data_path = Path("../processed_data/gold.parquet")
    
    if not data_path.exists():
        print("Error: Gold data not found. Please run extract_gold.py first.")
        return None
        
    try:
        df = pd.read_parquet(data_path)
        df.index = pd.to_datetime(df.index, format="%d-%m-%Y")
        df = df.sort_index().dropna()
        
        # Create OHLC data
        np.random.seed(42)
        noise_factor = 0.002
        
        df_bt = pd.DataFrame({
            'Open': df['Gold Index Prices'],
            'High': df['Gold Index Prices'] * (1 + np.random.uniform(0, noise_factor, len(df))),
            'Low': df['Gold Index Prices'] * (1 - np.random.uniform(0, noise_factor, len(df))),
            'Close': df['Gold Index Prices'],
            'Volume': 1000
        }, index=df.index)
        
        df_bt['High'] = np.maximum(df_bt['High'], df_bt['Close'])
        df_bt['Low'] = np.minimum(df_bt['Low'], df_bt['Close'])
        
        print(f"Loaded {len(df_bt)} rows for optimization")
        return df_bt
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def run_single_optimization(params, data):
    """Run a single backtest with given parameters"""
    try:
        cerebro = bt.Cerebro()
        cerebro.addstrategy(OptimizedDonchianStrategy, **params)
        
        data_feed = bt.feeds.PandasData(
            dataname=data,
            datetime=None,
            open='Open',
            high='High',
            low='Low', 
            close='Close',
            volume='Volume',
            openinterest=-1
        )
        cerebro.adddata(data_feed)
        
        cerebro.broker.setcash(10_000_000)
        cerebro.broker.setcommission(commission=0.00001)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        results = cerebro.run()
        strat = results[0]
        
        # Extract metrics
        final_value = cerebro.broker.getvalue()
        initial_value = 10_000_000
        total_return = (final_value - initial_value) / initial_value * 100
        
        sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        if sharpe is None:
            sharpe = 0
            
        drawdown = strat.analyzers.drawdown.get_analysis()
        max_dd = drawdown.get('max', {}).get('drawdown', 0)
        
        trades = strat.analyzers.trades.get_analysis()
        total_trades = trades.get('total', {}).get('total', 0)
        win_rate = 0
        if total_trades > 0:
            won = trades.get('won', {}).get('total', 0)
            win_rate = (won / total_trades) * 100
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'final_value': final_value
        }
        
    except Exception as e:
        print(f"Error in optimization: {e}")
        return {
            'total_return': -999,
            'sharpe_ratio': -999,
            'max_drawdown': 100,
            'total_trades': 0,
            'win_rate': 0,
            'final_value': 0
        }


def run_grid_search():
    """Run comprehensive grid search optimization"""
    print("🔍 Starting Donchian Channel Parameter Optimization...")
    print("=" * 60)
    
    # Load data
    data = load_gold_data()
    if data is None:
        return
    
    # Define parameter ranges to test
    param_ranges = {
        'entry_period': [20, 30, 40, 55, 70, 95, 120],      # Breakout periods
        'exit_period': [10, 15, 20, 25, 30],                # Exit periods  
        'atr_period': [14, 20, 25, 30, 40],                 # ATR periods
        'atr_multiplier': [2.0, 2.5, 3.0, 3.5, 4.0, 5.0],  # Profit targets
        'risk_percent': [1.0, 1.5, 2.0, 2.5, 3.0]          # Risk per trade
    }
    
    print("Parameter Ranges:")
    for param, values in param_ranges.items():
        print(f"  {param}: {values}")
    
    # Generate all combinations (this will be a lot!)
    param_combinations = list(product(*param_ranges.values()))
    total_combinations = len(param_combinations)
    
    print(f"\n📊 Testing {total_combinations:,} parameter combinations...")
    print("This may take several minutes...\n")
    
    results = []
    
    for i, combo in enumerate(param_combinations):
        params = dict(zip(param_ranges.keys(), combo))
        
        # Progress indicator
        if i % 100 == 0:
            print(f"Progress: {i:,}/{total_combinations:,} ({i/total_combinations*100:.1f}%)")
        
        # Run single optimization
        result = run_single_optimization(params, data)
        result.update(params)
        results.append(result)
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    print(f"\n✅ Optimization complete! Tested {len(df_results)} combinations")
    return df_results


def analyze_results(df_results):
    """Analyze and display optimization results"""
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS ANALYSIS")
    print("=" * 60)
    
    # Filter out failed runs
    df_clean = df_results[df_results['total_return'] > -900].copy()
    print(f"Valid results: {len(df_clean):,} out of {len(df_results):,}")
    
    if len(df_clean) == 0:
        print("❌ No valid results found!")
        return
    
    # Top 20 by different metrics
    print("\n🏆 TOP 20 COMBINATIONS BY TOTAL RETURN:")
    print("-" * 50)
    top_return = df_clean.nlargest(20, 'total_return')
    for i, row in top_return.iterrows():
        print(f"{row['total_return']:6.1f}% | Entry:{row['entry_period']:3.0f} Exit:{row['exit_period']:2.0f} "
              f"ATR:{row['atr_period']:2.0f} Mult:{row['atr_multiplier']:3.1f} Risk:{row['risk_percent']:3.1f}% "
              f"| Sharpe:{row['sharpe_ratio']:5.2f} DD:{row['max_drawdown']:5.1f}% Trades:{row['total_trades']:3.0f}")
    
    print("\n🎯 TOP 20 COMBINATIONS BY SHARPE RATIO:")
    print("-" * 50)
    top_sharpe = df_clean.nlargest(20, 'sharpe_ratio')
    for i, row in top_sharpe.iterrows():
        print(f"{row['sharpe_ratio']:5.2f} | Entry:{row['entry_period']:3.0f} Exit:{row['exit_period']:2.0f} "
              f"ATR:{row['atr_period']:2.0f} Mult:{row['atr_multiplier']:3.1f} Risk:{row['risk_percent']:3.1f}% "
              f"| Return:{row['total_return']:6.1f}% DD:{row['max_drawdown']:5.1f}% Trades:{row['total_trades']:3.0f}")
    
    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"Average Return:    {df_clean['total_return'].mean():6.1f}%")
    print(f"Median Return:     {df_clean['total_return'].median():6.1f}%") 
    print(f"Best Return:       {df_clean['total_return'].max():6.1f}%")
    print(f"Worst Return:      {df_clean['total_return'].min():6.1f}%")
    print(f"Average Sharpe:    {df_clean['sharpe_ratio'].mean():6.2f}")
    print(f"Average Drawdown:  {df_clean['max_drawdown'].mean():6.1f}%")
    
    return df_clean


def create_heatmaps(df_results):
    """Create heatmap visualizations"""
    df_clean = df_results[df_results['total_return'] > -900].copy()
    
    if len(df_clean) == 0:
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Donchian Channel Parameter Optimization Heatmaps', fontsize=16, fontweight='bold')
    
    # 1. Entry vs Exit Period (Total Return)
    pivot1 = df_clean.groupby(['entry_period', 'exit_period'])['total_return'].mean().unstack()
    sns.heatmap(pivot1, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=axes[0,0])
    axes[0,0].set_title('Total Return % by Entry vs Exit Period')
    axes[0,0].set_xlabel('Exit Period (days)')
    axes[0,0].set_ylabel('Entry Period (days)')
    
    # 2. Entry vs Exit Period (Sharpe Ratio)
    pivot2 = df_clean.groupby(['entry_period', 'exit_period'])['sharpe_ratio'].mean().unstack()
    sns.heatmap(pivot2, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=axes[0,1])
    axes[0,1].set_title('Sharpe Ratio by Entry vs Exit Period')
    axes[0,1].set_xlabel('Exit Period (days)')
    axes[0,1].set_ylabel('Entry Period (days)')
    
    # 3. ATR Period vs ATR Multiplier (Total Return)
    pivot3 = df_clean.groupby(['atr_period', 'atr_multiplier'])['total_return'].mean().unstack()
    sns.heatmap(pivot3, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=axes[1,0])
    axes[1,0].set_title('Total Return % by ATR Period vs ATR Multiplier')
    axes[1,0].set_xlabel('ATR Multiplier')
    axes[1,0].set_ylabel('ATR Period (days)')
    
    # 4. Risk vs Entry Period (Sharpe Ratio)
    pivot4 = df_clean.groupby(['risk_percent', 'entry_period'])['sharpe_ratio'].mean().unstack()
    sns.heatmap(pivot4, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=axes[1,1])
    axes[1,1].set_title('Sharpe Ratio by Risk % vs Entry Period')
    axes[1,1].set_xlabel('Entry Period (days)')
    axes[1,1].set_ylabel('Risk Percent (%)')
    
    plt.tight_layout()
    plt.savefig('donchian_optimization_heatmaps.png', dpi=300, bbox_inches='tight')
    print("\n📊 Heatmaps saved as 'donchian_optimization_heatmaps.png'")
    plt.show()


def find_robust_zones(df_results):
    """Find parameter combinations that work well across multiple criteria"""
    df_clean = df_results[df_results['total_return'] > -900].copy()
    
    if len(df_clean) == 0:
        return
    
    print("\n🔍 FINDING ROBUST PARAMETER ZONES:")
    print("=" * 50)
    
    # Define "good" criteria
    criteria = {
        'return_good': df_clean['total_return'] > df_clean['total_return'].quantile(0.75),
        'sharpe_good': df_clean['sharpe_ratio'] > df_clean['sharpe_ratio'].quantile(0.75), 
        'dd_good': df_clean['max_drawdown'] < df_clean['max_drawdown'].quantile(0.25),
        'trades_good': df_clean['total_trades'] > 20  # Minimum sample size
    }
    
    # Score each combination
    df_clean['score'] = 0
    for criterion, mask in criteria.items():
        df_clean['score'] += mask.astype(int)
        print(f"{criterion}: {mask.sum()} combinations qualify")
    
    # Find robust combinations (score >= 3 out of 4 criteria)
    robust = df_clean[df_clean['score'] >= 3].sort_values('score', ascending=False)
    
    print(f"\n🎯 ROBUST COMBINATIONS (3+ criteria met):")
    print(f"Found {len(robust)} robust parameter sets:")
    print("-" * 80)
    
    for i, row in robust.head(15).iterrows():
        print(f"Score:{row['score']}/4 | Entry:{row['entry_period']:3.0f} Exit:{row['exit_period']:2.0f} "
              f"ATR:{row['atr_period']:2.0f} Mult:{row['atr_multiplier']:3.1f} Risk:{row['risk_percent']:3.1f}% "
              f"| Ret:{row['total_return']:6.1f}% Sharpe:{row['sharpe_ratio']:5.2f} "
              f"DD:{row['max_drawdown']:5.1f}% Trades:{row['total_trades']:3.0f}")
    
    if len(robust) > 0:
        best_robust = robust.iloc[0]
        print(f"\n🏆 RECOMMENDED PARAMETERS:")
        print(f"Entry Period:    {best_robust['entry_period']:.0f} days")
        print(f"Exit Period:     {best_robust['exit_period']:.0f} days") 
        print(f"ATR Period:      {best_robust['atr_period']:.0f} days")
        print(f"ATR Multiplier:  {best_robust['atr_multiplier']:.1f}")
        print(f"Risk Percent:    {best_robust['risk_percent']:.1f}%")
        print(f"\nExpected Performance:")
        print(f"Total Return:    {best_robust['total_return']:.1f}%")
        print(f"Sharpe Ratio:    {best_robust['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:    {best_robust['max_drawdown']:.1f}%")
        print(f"Total Trades:    {best_robust['total_trades']:.0f}")
    
    return robust


def main():
    """Main optimization workflow"""
    print("🚀 DONCHIAN CHANNEL PARAMETER OPTIMIZATION")
    print("=" * 60)
    print("This will test thousands of parameter combinations to find the best settings.")
    print("Estimated time: 5-15 minutes depending on your computer.")
    print()
    
    response = input("Continue with optimization? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Optimization cancelled.")
        return
    
    # Run grid search
    results_df = run_grid_search()
    
    if results_df is None or len(results_df) == 0:
        print("❌ Optimization failed!")
        return
    
    # Analyze results
    clean_results = analyze_results(results_df)
    
    # Create visualizations
    create_heatmaps(results_df)
    
    # Find robust parameter zones
    robust_params = find_robust_zones(results_df)
    
    # Save results
    results_df.to_csv('donchian_optimization_results.csv', index=False)
    print(f"\n💾 Full results saved to 'donchian_optimization_results.csv'")
    
    print("\n✅ OPTIMIZATION COMPLETE!")
    print("Check the heatmaps and robust parameter recommendations above.")


if __name__ == "__main__":
    main()
