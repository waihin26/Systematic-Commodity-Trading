"""
Gold RSI Trading Strategy
========================

This script implements a simple RSI-based trading strategy for gold:
- Buy Signal: RSI < 40 (oversold)
- Sell Signal: RSI > 60 (overbought)
- Initial Capital: $10,000,000 USD

Features:
1. RSI calculation with 30-day period
2. Clear buy/sell signals
3. Performance metrics calculation
4. Portfolio visualization

Usage:
    python gold_rsi_strategy.py
"""

import backtrader as bt
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime



class GoldRSIStrategy(bt.Strategy):    
    params = (
        ('rsi_period', 30),      # RSI calculation period (30 days)
        ('rsi_buy', 40),         # RSI buy threshold (oversold) - more reasonable
        ('rsi_sell', 60),        # RSI sell threshold (overbought) - more reasonable
        ('printlog', False),     # Disable trade logs to reduce output
    )
    
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        
        # Track buy prices and commisions for PnL calculations
        self.buyprice = None
        self.buycomm = None
        
        # Add RSI indicator
        self.rsi = bt.indicators.RSI(
            self.datas[0].close,
            period=self.params.rsi_period
        )
        
        # Track trade statistics
        self.trade_count = 0
        self.winning_trades = 0
        self.total_pnl = 0
        self.total_commission = 0
        self.order_count = 0  # Track total number of orders
        
        self.portfolio_values = []  # Track portfolio value over time
        self.dates = []            # Track dates
        self.trade_dates = []      # Track trade dates
        self.trade_values = []     # Track portfolio value at trade times
        self.trade_types = []      # Track 'BUY' or 'SELL'
        
    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - nothing to do
            return
            
        # Check if an order has been completed
        if order.status in [order.Completed]:
            # Track total commission paid and order count
            self.total_commission += order.executed.comm
            self.order_count += 1
            
            current_date = self.datas[0].datetime.datetime(0)
            current_value = self.broker.getvalue()
            
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                
                self.trade_dates.append(current_date)
                self.trade_values.append(current_value)
                self.trade_types.append('BUY')
                
            else:  # Sell
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.trade_dates.append(current_date)
                self.trade_values.append(current_value)
                self.trade_types.append('SELL')
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            
        # Reset 
        self.order = None
        
    def notify_trade(self, trade):
        """Handle completed trade notifications"""
        if not trade.isclosed:
            return
            
        self.trade_count += 1 # Represents an entry and an exit
        pnl = trade.pnl
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
            
        self.log(f'TRADE #{self.trade_count} - PNL: ${pnl:.2f}')
        
    def log(self, txt, dt=None):
        """Logging function"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.datetime(0)
            print(f'{dt.strftime("%d-%m-%Y")}: {txt}')
            
    def next(self):
        """Main strategy logic - called for each bar"""
        
        current_date = self.datas[0].datetime.datetime(0)
        current_value = self.broker.getvalue()
        self.dates.append(current_date)
        self.portfolio_values.append(current_value)
        
        # Check if we have an order pending
        if self.order:
            return
            
        # Check if we are in the market
        if not self.position:
            # Not in market - look for buy signal
            if self.rsi[0] < self.params.rsi_buy:
                self.log(f'BUY CREATE - RSI: {self.rsi[0]:.2f}, Price: {self.dataclose[0]:.2f}')
                # Calculate position size (use 95% of available cash to avoid margin issues)
                available_cash = self.broker.getcash() * 0.95
                size = int(available_cash / self.dataclose[0])
                if size > 0:  # Only place order if we can afford at least 1 unit
                    self.order = self.buy(size=size)
                else:
                    self.log(f'Insufficient cash for purchase: ${available_cash:.2f} < ${self.dataclose[0]:.2f}')
                
        else:
            # In market holding a position - look for sell signal  
            if self.rsi[0] > self.params.rsi_sell:
                self.log(f'SELL CREATE - RSI: {self.rsi[0]:.2f}, Price: {self.dataclose[0]:.2f}')
                self.order = self.sell(size=self.position.size)


class GoldDataFeed(bt.feeds.PandasData):
    """
    Custom data feed for our gold data format
    """
    params = (
        ('datetime', 'Date'),     # Use index as datetime
        ('open', -1),           # No open price
        ('high', -1),           # No high price  
        ('low', -1),            # No low price
        ('close', 'Gold Index Prices'),  # Close price column
        ('volume', -1),         # No volume
        ('openinterest', -1),   # No open interest
    )


def load_gold_data():
    """Load the processed gold data"""
    data_path = Path("../processed_data/gold.parquet")
    
    if not data_path.exists():
        print("Error: Gold data not found. Please run extract_gold.py first.")
        return None
        
    try:
        df = pd.read_parquet(data_path)
        
        print(f"Raw data shape: {df.shape}")
        print(f"Index type: {type(df.index)}")
        print(f"Sample values: {df.head(2)}")
        
        # Convert string dates to datetime for backtrader
        df.index = pd.to_datetime(df.index, format="%d-%m-%Y")
        
        # Sort by date to ensure chronological order
        df = df.sort_index()
        
        # Remove any NaN values that might cause issues
        df = df.dropna()
        
        # Ensure we have proper OHLC structure for backtrader
        # Since we only have close prices, create dummy OHLC data
        df_bt = pd.DataFrame({
            'Open': df['Gold Index Prices'],
            'High': df['Gold Index Prices'], 
            'Low': df['Gold Index Prices'],
            'Close': df['Gold Index Prices'],
            'Volume': 0  # Dummy volume
        }, index=df.index)
        
        print(f"Loaded {len(df_bt)} rows of gold data")
        print(f"Date range: {df_bt.index.min().strftime('%d-%m-%Y')} to {df_bt.index.max().strftime('%d-%m-%Y')}")
        print(f"Price range: ${df_bt['Close'].min():.2f} - ${df_bt['Close'].max():.2f}")
        
        return df_bt
        
    except Exception as e:
        print(f"Error loading gold data: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_performance_metrics(cerebro):
    """Calculate comprehensive performance metrics"""
    # Get final portfolio value
    final_value = cerebro.broker.getvalue()
    initial_value = 10_000_000  # $10M initial capital
    
    # Basic returns
    total_return = (final_value - initial_value) / initial_value * 100
    
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    print(f"Initial Capital:     ${initial_value:,.2f}")
    print(f"Final Portfolio:     ${final_value:,.2f}")
    print(f"Total Return:        {total_return:.2f}%")
    
    # Get strategy instance to access trade statistics
    strategy = cerebro.runstrats[0][0]
    
    if hasattr(strategy, 'trade_count') and strategy.trade_count > 0:
        win_rate = (strategy.winning_trades / strategy.trade_count) * 100
        avg_return_per_trade = strategy.total_pnl / strategy.trade_count
        avg_commission_per_order = strategy.total_commission / strategy.order_count if strategy.order_count > 0 else 0
        
        print(f"Total Trades:        {strategy.trade_count}")
        print(f"Total Orders:        {strategy.order_count}")
        print(f"Winning Trades:      {strategy.winning_trades}")
        print(f"Win Rate:            {win_rate:.2f}%")
        print(f"Avg Return/Trade:    ${avg_return_per_trade:.2f}")
        print(f"Total P&L:           ${strategy.total_pnl:.2f}")
        print(f"Total Commission:    ${strategy.total_commission:.2f}")
        print(f"Avg Comm/Order:      ${avg_commission_per_order:.2f}")
        print(f"Net P&L:             ${strategy.total_pnl - strategy.total_commission:.2f}")
    else:
        print("No trades executed during backtest period")
        
    print("="*60)
    
def plot_portfolio_performance(strategy):
    """Create a comprehensive portfolio performance chart"""
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    fig.suptitle('Gold RSI Strategy - Portfolio Performance (1988-2025)', fontsize=16, fontweight='bold')
    
    # Convert dates and values to arrays for plotting
    dates = np.array(strategy.dates)
    portfolio_values = np.array(strategy.portfolio_values)
    
    # Plot 1: Portfolio Value Over Time
    ax1.plot(dates, portfolio_values, linewidth=2, color='darkblue', label='Portfolio Value')
    ax1.axhline(y=10_000_000, color='red', linestyle='--', alpha=0.7, label='Initial Capital ($10M)')
    
    # Add trade markers
    buy_dates = []
    buy_values = []
    sell_dates = []
    sell_values = []
    
    for i, trade_type in enumerate(strategy.trade_types):
        if trade_type == 'BUY':
            buy_dates.append(strategy.trade_dates[i])
            buy_values.append(strategy.trade_values[i])
        else:
            sell_dates.append(strategy.trade_dates[i])
            sell_values.append(strategy.trade_values[i])
    
    # Plot buy and sell signals
    if buy_dates:
        ax1.scatter(buy_dates, buy_values, color='green', marker='^', s=50, alpha=0.7, label=f'Buy Signals ({len(buy_dates)})')
    if sell_dates:
        ax1.scatter(sell_dates, sell_values, color='red', marker='v', s=50, alpha=0.7, label=f'Sell Signals ({len(sell_dates)})')
    
    ax1.set_ylabel('Portfolio Value (USD)', fontsize=12)
    ax1.set_title('Portfolio Value Over Time', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.YearLocator(5))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
    
    # Plot 2: Cumulative Returns
    initial_value = portfolio_values[0]
    cumulative_returns = ((portfolio_values - initial_value) / initial_value) * 100
    
    ax2.plot(dates, cumulative_returns, linewidth=2, color='darkgreen', label='Cumulative Return (%)')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.fill_between(dates, cumulative_returns, 0, alpha=0.3, color='green', where=(cumulative_returns >= 0))
    ax2.fill_between(dates, cumulative_returns, 0, alpha=0.3, color='red', where=(cumulative_returns < 0))
    
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax2.set_title('Cumulative Returns Over Time', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    ax2.xaxis.set_major_locator(mdates.YearLocator(5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_minor_locator(mdates.YearLocator(1))
    
    # Add performance statistics as text
    final_return = cumulative_returns[-1]
    max_return = np.max(cumulative_returns)
    min_return = np.min(cumulative_returns)
    
    stats_text = f"""Performance Summary:
    Final Return: {final_return:.1f}%
    Max Return: {max_return:.1f}%
    Max Drawdown: {min_return:.1f}%
    Total Trades: {strategy.trade_count}
    Win Rate: {(strategy.winning_trades/strategy.trade_count)*100:.1f}%"""
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('gold_rsi_portfolio_performance.png', dpi=300, bbox_inches='tight')
    print("\n📊 Portfolio performance chart saved as 'gold_rsi_portfolio_performance.png'")
    
    # Show the plot
    plt.show()


def main():
    print("=== Gold RSI Strategy Backtest ===")
    
    # Load gold data
    df = load_gold_data()
    if df is None:
        return
        
    # Create Cerebro engine
    cerebro = bt.Cerebro()
    
    # Add strategy
    cerebro.addstrategy(GoldRSIStrategy)
    
    # Create data feed using standard backtrader format
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,  # Use index
        open='Open',
        high='High', 
        low='Low',
        close='Close',
        volume='Volume',
        openinterest=-1
    )
    cerebro.adddata(data)
    
    # Set initial capital and commission
    initial_cash = 10_000_000  # $10 million
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% per trade
    
    # Add analyzers for detailed performance metrics
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    print(f'Starting Portfolio Value: ${cerebro.broker.getvalue():,.2f}')
    
    # Run backtest
    results = cerebro.run()
    
    print(f'Final Portfolio Value: ${cerebro.broker.getvalue():,.2f}')
    
    # Calculate and display performance metrics
    calculate_performance_metrics(cerebro)
    
    # Display analyzer results
    strat = results[0]
    
    print("\nADDITIONAL METRICS:")
    print("-" * 30)
    
    # Sharpe Ratio
    if hasattr(strat.analyzers.sharpe, 'ratio') and strat.analyzers.sharpe.ratio is not None:
        print(f'Sharpe Ratio: {strat.analyzers.sharpe.ratio:.4f}')
    
    # Maximum Drawdown
    drawdown = strat.analyzers.drawdown.get_analysis()
    if 'max' in drawdown and 'drawdown' in drawdown['max']:
        print(f'Max Drawdown: {drawdown["max"]["drawdown"]:.2f}%')
    
    strategy_instance = results[0]
    plot_portfolio_performance(strategy_instance)
    print("Strategy completed successfully!")


if __name__ == "__main__":
    main()
