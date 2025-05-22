import pandas as pd
import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Define file paths
WEATHER_PATH = PROJECT_ROOT / "data/processed/weather_clean.csv"
GEN_PATH = PROJECT_ROOT / "data/processed/Combined_Hourly_Gen_Without_Price.csv"
WIND_SOLAR_PATH = PROJECT_ROOT / "data/processed/wind_solar_data.csv"
SOLAR_XLSX_PATH = PROJECT_ROOT / "data/processed/solar_data_2022_to_2024.xlsx"

def compile_data():
    try:
        # Read the files
        print("Reading weather data...")
        weather = pd.read_csv(WEATHER_PATH)
        
        print("Reading generation data...")
        gen = pd.read_csv(GEN_PATH)
        
        print("Reading wind and solar data...")
        wind_solar = pd.read_csv(WIND_SOLAR_PATH)
        
        print("Reading solar xlsx data...")
        solar_xlsx = pd.read_excel(SOLAR_XLSX_PATH)

        # Start with generation data
        total = gen.copy()
        
        # Merge all datasets
        print("Merging datasets...")
        total = total.merge(weather, left_index=True, right_index=True, how='left', suffixes=('', '_weather'))
        total = total.merge(wind_solar, left_index=True, right_index=True, how='left', suffixes=('', '_windsolar'))
        total = total.merge(solar_xlsx, left_index=True, right_index=True, how='left', suffixes=('', '_solar'))

        # Convert datetime columns
        print("Processing datetime columns...")
        datetime_cols = ['Hour', 'DATE', 'Time (Hour-Ending)', 'Date']
        for col in datetime_cols:
            if col in total.columns:
                total[col] = pd.to_datetime(total[col], errors='coerce')

        # Combine datetime columns
        total['time'] = total['Hour'].combine_first(
            total['Time (Hour-Ending)']).combine_first(
            total['DATE']).combine_first(
            total['Date'])

        # Drop original datetime columns
        total.drop(columns=datetime_cols, inplace=True)

        # Reorder columns
        cols = ['time'] + [col for col in total.columns if col != 'time']
        total = total[cols]

        # Save the merged data
        output_path = PROJECT_ROOT / "data/processed/final_merged_data.csv"
        print(f"Saving merged data to: {output_path}")
        total.to_csv(output_path, index=False)
        print("Data compilation completed successfully!")
        
        return total
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        print("Please ensure all required files exist in the data/processed directory:")
        print(f"- {WEATHER_PATH}")
        print(f"- {GEN_PATH}")
        print(f"- {WIND_SOLAR_PATH}")
        print(f"- {SOLAR_XLSX_PATH}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    compile_data() 