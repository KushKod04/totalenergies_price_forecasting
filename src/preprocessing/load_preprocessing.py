import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def preprocess_load_data(file_path):
    """
    Preprocess the aggregated load data from ERCOT.
    
    Args:
        file_path (str): Path to the raw load data CSV file
        
    Returns:
        pd.DataFrame: Processed load data
    """
    # Read the data
    load_df = pd.read_csv(file_path)
    
    # Convert SCED Time Stamp to datetime
    load_df['SCED Time Stamp'] = pd.to_datetime(load_df['SCED Time Stamp'])
    
    # Extract date and time components
    load_df['Date'] = load_df['SCED Time Stamp'].dt.date
    load_df['Hour'] = load_df['SCED Time Stamp'].dt.hour
    load_df['Minute'] = load_df['SCED Time Stamp'].dt.minute
    
    # Create a proper datetime index
    load_df.set_index('SCED Time Stamp', inplace=True)
    
    return load_df

def analyze_load_data(load_df):
    """
    Perform basic analysis on the load data.
    
    Args:
        load_df (pd.DataFrame): Processed load data
    """
    # Check for missing values
    print("Missing values per column:")
    print(load_df.isnull().sum())
    
    # Check for duplicate timestamps
    print("\nNumber of duplicate timestamps:", load_df.index.duplicated().sum())
    
    # Basic statistics
    print("\nBasic statistics of AGG LOAD SUMMARY:")
    print(load_df['AGG LOAD SUMMARY'].describe())
    
    # Calculate hourly statistics
    hourly_stats = load_df.groupby('Hour')['AGG LOAD SUMMARY'].agg(['mean', 'std', 'min', 'max'])
    print("\nHourly Load Statistics:")
    print(hourly_stats)

def visualize_load_data(load_df):
    """
    Create visualizations for the load data.
    
    Args:
        load_df (pd.DataFrame): Processed load data
    """
    # Time series plot
    plt.figure(figsize=(15, 6))
    plt.plot(load_df.index, load_df['AGG LOAD SUMMARY'])
    plt.title('ERCOT System Load Over Time')
    plt.xlabel('Time')
    plt.ylabel('Load (MW)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=load_df['AGG LOAD SUMMARY'])
    plt.title('Distribution of System Load')
    plt.ylabel('Load (MW)')
    plt.show()

def save_processed_data(load_df, output_path):
    """
    Save the processed data to a CSV file.
    
    Args:
        load_df (pd.DataFrame): Processed load data
        output_path (str): Path to save the processed data
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    load_df.to_csv(output_path)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # File paths relative to project root
    input_file = os.path.join(project_root, 'data', 'raw', 'load', '2d_Agg_Load_Summary-17-APR-25.csv')
    output_file = os.path.join(project_root, 'data', 'processed', 'load', 'processed_aggregated_load.csv')
    
    # Process the data
    load_df = preprocess_load_data(input_file)
    
    # Analyze the data
    analyze_load_data(load_df)
    
    # Visualize the data
    visualize_load_data(load_df)
    
    # Save the processed data
    save_processed_data(load_df, output_file) 