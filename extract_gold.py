"""
Extract Gold Price Data
======================

This script:
1) Extracts only the "GOLDS Index" column from an Excel file
2) Fills any gaps in the data
3) Saves it as a quick-loading parquet file

Usage:
    python extract_gold.py

Notes:
    - Assumes GOLDS Index data is in column N, starting from row 8
    - Date column is assumed to be in column A, starting from row 8
"""

import pandas as pd
import os
from pathlib import Path


def find_excel_file():
    """Return the path to MSCI_Comps.xlsx if it exists in raw_data or current directory, else None"""
    # Check raw_data folder first
    raw_data_path = Path("raw_data") / "MSCI_Comps.xlsx"
    if raw_data_path.exists():
        return str(raw_data_path)
    # Then check current directory
    current_path = Path("MSCI_Comps.xlsx")
    if current_path.exists():
        return str(current_path)
    return None


def extract_gold_data(excel_file):
    """Extract GOLDS Index data from Excel file"""
    print(f"Reading data from: {excel_file}")
    
    try:
        # Read the Excel file, skiprows=7 to skip the first 7 rows, as data starts from row 8
        # usecols='A,N' to only read columns A (date) and N (GOLDS Index)
        df = pd.read_excel(
            excel_file,
            skiprows=7,  # Skip rows 0-7, data starts from row 8
            usecols="A,N",  # Only read columns A (Date) and N (GOLDS Index)
            header=None,  # No header row as we're already skipping rows
            names=["Date", "GOLDS_Index"]  # Custom column names
        )
        
        # Check if we have data
        if df.empty:
            print("Error: No data found in the Excel file")
            return None
            
        print(f"Successfully read {len(df)} rows of data")
        return df
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

def process_gold_data(df):
    if df is None:
        print("no data found")
        return None
        
    try:
        # Convert Excel dates "4/1/88", to a datetime
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        
        # Create nicely formatted dates
        dates = df["Date"].dt.strftime("%d-%m-%Y")
        
        # Create new DataFrame with desired column names
        df_new = pd.DataFrame({
            'Date': dates,
            'Gold Index Prices': df["GOLDS_Index"]
        })
        
        # Sort chronologically using original dates
        df_new = df_new.set_index('Date')
        df_filled = df_new.ffill()  # Forward fill missing values if there are any
        
        # Check for any remaining missing values
        if df_filled["Gold Index Prices"].isna().sum() > 0:
            df_filled = df_filled.bfill()
        
        # Show the date range of our data
        dates = pd.to_datetime(df_filled.index, format="%d-%m-%Y")
        first_date = dates.min().strftime("%d-%m-%Y")
        last_date = dates.max().strftime("%d-%m-%Y")
        print(f"Data ranges from {first_date} to {last_date}")
        
        return df_filled

    except Exception as e:
        print(f"Error processing data: {e}")
        return None


def save_processed_data(df, format="both"):
    """Save the processed data in both parquet and CSV formats"""
    if df is None:
        return False
        
    try:
        # Create processed_data directory if it doesn't exist
        processed_dir = Path("processed_data")
        processed_dir.mkdir(exist_ok=True)
        
        # Always save as CSV for easy Excel viewing
        csv_path = processed_dir / "gold.csv"
        df.to_csv(csv_path)
        print(f"Saved CSV data to {csv_path} (open this in Excel)")
        
        # Save in the specified format(s)
        if format.lower() in ["parquet", "both"]:
            parquet_path = processed_dir / "gold.parquet"
            df.to_parquet(parquet_path)
            print(f"Saved parquet data to {parquet_path}")
            
        if format.lower() == "pickle":
            pickle_path = processed_dir / "gold.pkl"
            df.to_pickle(pickle_path)
            print(f"Saved pickle data to {pickle_path}")
        
        return True
        
    except Exception as e:
        print(f"Error saving data: {e}")
        return False


def show_data_preview(df):
    """Show a preview of the data"""
    if df is None:
        return
    
    # Set pandas display options for better alignment
    pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.colheader_justify', 'right')
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    
    print("\n--- Data Preview ---")
    print("\nFirst 5 rows:")
    # Format the DataFrame for display
    df_display = df.copy()
    # Ensure consistent spacing
    max_date_len = max(len(str(date)) for date in df_display.index)
    date_format = '{:<' + str(max_date_len) + '}'
    df_display.index = [date_format.format(date) for date in df_display.index]
    df_display['Gold Index Prices'] = df_display['Gold Index Prices'].apply(lambda x: f"{x:>12.2f}")
    print(df_display.head().to_string(justify='left'))
    
    print("\nLast 5 rows:")
    print(df_display.tail().to_string(justify='left'))
    
    print(f"\nTotal rows: {len(df)}")
    
    # Calculate basic statistics
    print("\nBasic statistics:")
    stats = df.describe()
    print(stats.to_string(float_format=lambda x: '{:12.2f}'.format(x)))
    
    # Check for missing values
    print(f"\nMissing values: {df.isna().sum().sum()}")


def verify_parquet(file_path='processed_data/gold.parquet'):
    """Verify the contents of the saved parquet file"""
    print("\nVerifying saved parquet file:")
    print("="*50)
    try:
        df = pd.read_parquet(file_path)
        print("\nParquet file contents:")
        pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))
        print(df.head(3).to_string())
        print("\n... and last 3 rows:")
        print(df.tail(3).to_string())
        return True
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return False

def main():
    print("=== GOLDS Index Data Extraction ===")
    excel_file = find_excel_file()
    if not excel_file:
        print("Error: No Excel file found in raw_data directory or current directory")
        return
    df = extract_gold_data(excel_file)
    df_processed = process_gold_data(df)
    show_data_preview(df_processed)
    save_processed_data(df_processed, format="parquet")
    verify_parquet()
    print("\nDone! You can now use the processed gold data from processed_data/gold.parquet")


if __name__ == "__main__":
    main()
