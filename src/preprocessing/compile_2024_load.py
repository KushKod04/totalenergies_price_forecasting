import pandas as pd
import os
from pathlib import Path
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_monthly_folders(base_path: str) -> List[str]:
    """Get all monthly folders in the 2024 Load directory."""
    base_path = Path(base_path)
    monthly_folders = [f for f in base_path.iterdir() if f.is_dir() and "ACTUALSYSLOADWZNP6345" in f.name]
    return sorted(monthly_folders)

def read_daily_file(file_path: Path) -> pd.DataFrame:
    """Read a single daily load file."""
    try:
        df = pd.read_csv(file_path)
        # Extract date from filename (format: YYYYMMDD)
        date_str = file_path.stem.split('.')[-2]  # Get the date part from filename
        df['date'] = pd.to_datetime(date_str, format='%Y%m%d')
        return df
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return pd.DataFrame()

def process_monthly_folder(folder_path: Path) -> pd.DataFrame:
    """Process all daily files in a monthly folder."""
    daily_files = [f for f in folder_path.iterdir() if f.is_file() and f.suffix == '.csv']
    dfs = []
    
    for file_path in sorted(daily_files):
        df = read_daily_file(file_path)
        if not df.empty:
            dfs.append(df)
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def compile_2024_load_data(base_path: str = "data/raw/load/2024 Load") -> pd.DataFrame:
    """Compile all 2024 load data into a single DataFrame."""
    monthly_folders = get_monthly_folders(base_path)
    all_data = []
    
    for folder in monthly_folders:
        logger.info(f"Processing folder: {folder.name}")
        monthly_data = process_monthly_folder(folder)
        if not monthly_data.empty:
            all_data.append(monthly_data)
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Convert OperDay to datetime
        final_df['OperDay'] = pd.to_datetime(final_df['OperDay'])
        
        # Combine date and HourEnding into a single datetime column
        final_df['datetime'] = final_df.apply(
            lambda x: pd.to_datetime(f"{x['OperDay'].strftime('%Y-%m-%d')} {x['HourEnding']}"),
            axis=1
        )
        
        # Sort by datetime
        final_df = final_df.sort_values('datetime')
        
        # Save the compiled data
        output_path = Path("data/processed/2024_load_compiled.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        logger.info(f"Compiled data saved to {output_path}")
        
        return final_df
    else:
        logger.error("No data was compiled")
        return pd.DataFrame()

if __name__ == "__main__":
    df = compile_2024_load_data()
    if not df.empty:
        print("\nData Summary:")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print("\nColumns:")
        print(df.columns.tolist())
        print("\nSample data:")
        print(df.head()) 