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
    """Process the Gold data using Singapore format (DD-MM-YYYY)"""
    if df is None:
        print("no data found")
        return None
        
    try:
        # Convert dates to datetime, day/month/year
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        
        # Create a formatted date string in DD-MM-YYYY format
        df["Date_Formatted"] = df["Date"].dt.strftime("%d-%m-%Y")
        
        # Set formatted date as index
        df = df.set_index("Date_Formatted")
        
        # Sort by original date (to maintain chronological order)
        df = df.sort_values("Date")
        df = df.drop("Date", axis=1)  # Remove the original date column
        
        # Fill any gaps/NaN values
        df_filled = df.ffill()  # Forward fill
        
        # Check for any remaining NaNs
        if df_filled["GOLDS_Index"].isna().sum() > 0:
            print(f"Warning: {df_filled['GOLDS_Index'].isna().sum()} NaN values remain after forward fill")
            # Try backward fill for any remaining NaNs
            df_filled = df_filled.bfill()
        
        # Get first and last dates for display
        dates = pd.to_datetime(df_filled.index, format="%d-%m-%Y")
        first_date = dates.min().strftime("%d-%m-%Y")
        last_date = dates.max().strftime("%d-%m-%Y")
        print(f"Data ranges from {first_date} to {last_date}")
        
        return df_filled
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return None


def save_processed_data(df, format="parquet"):
    """Save the processed data in the specified format"""
    if df is None:
        return False
        
    try:
        # Create working directory if it doesn't exist
        working_dir = Path("working")
        working_dir.mkdir(exist_ok=True)
        
        if format.lower() == "parquet":
            output_path = working_dir / "gold.parquet"
            df.to_parquet(output_path)
        elif format.lower() == "pickle" or format.lower() == "pkl":
            output_path = working_dir / "gold.pkl"
            df.to_pickle(output_path)
        else:
            output_path = working_dir / "gold.csv"
            df.to_csv(output_path)
        
        print(f"Saved processed data to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error saving data: {e}")
        return False


def show_data_preview(df):
    """Show a preview of the data"""
    if df is None:
        return
        
    print("\n--- Data Preview ---")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nLast 5 rows:")
    print(df.tail())
    
    print(f"\nTotal rows: {len(df)}")
    
    # Calculate basic statistics
    print("\nBasic statistics:")
    print(df.describe())
    
    # Check for missing values
    print(f"\nMissing values: {df.isna().sum().sum()}")


def main():
    print("=== GOLDS Index Data Extraction ===")
    
    # Find Excel file
    excel_file = find_excel_file()
    if not excel_file:
        print("Error: No Excel file found in raw_data directory or current directory")
        return
    
    # Extract GOLDS Index data
    df = extract_gold_data(excel_file)
    
    # Process the data
    df_processed = process_gold_data(df)
    
    # Show preview
    show_data_preview(df_processed)
    
    # Save processed data
    save_processed_data(df_processed, format="parquet")
    
    print("\nDone! You can now use the processed gold data from working/gold.parquet")
    print("Example code to load the data:")
    print("import pandas as pd")
    print("gold_data = pd.read_parquet('working/gold.parquet')")


if __name__ == "__main__":
    main()
