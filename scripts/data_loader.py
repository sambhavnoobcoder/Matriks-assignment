"""
Data Loader Module

This module provides functions to load data from CSV files.
"""

import pandas as pd
import os

def load_data(file_path):
    """
    Load data from a CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    
    try:
        data = pd.read_csv(file_path)
        
        # Convert date columns to datetime
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
        
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def load_all_data(base_dir='../data'):
    """
    Load all data files from the data directory
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing data files
        
    Returns:
    --------
    dict
        Dictionary containing all loaded DataFrames
    """
    data_files = {
        'raw_data': os.path.join(base_dir, 'matiks_data_clean.csv'),
        'dau': os.path.join(base_dir, 'dau.csv'),
        'wau': os.path.join(base_dir, 'wau.csv'),
        'mau': os.path.join(base_dir, 'mau.csv'),
        'revenue': os.path.join(base_dir, 'revenue_analysis.csv'),
        'user_segments': os.path.join(base_dir, 'user_segments.csv'),
        'churn_analysis': os.path.join(base_dir, 'churn_analysis.csv'),
        'cohort_retention': os.path.join(base_dir, 'cohort_retention.csv'),
        'user_clusters': os.path.join(base_dir, 'user_clusters.csv'),
        'cluster_analysis': os.path.join(base_dir, 'cluster_analysis.csv'),
        'funnel_analysis': os.path.join(base_dir, 'funnel_analysis.csv')
    }
    
    data_dict = {}
    
    for key, file_path in data_files.items():
        try:
            data_dict[key] = load_data(file_path)
            
            # Special handling for cohort_retention
            if key == 'cohort_retention' and data_dict[key] is not None:
                # Convert index to datetime for cohort analysis
                if data_dict[key].index.name == 'cohort':
                    data_dict[key].index = pd.to_datetime(data_dict[key].index)
        except Exception as e:
            print(f"Could not load {key} data: {e}")
            data_dict[key] = None
    
    return data_dict

if __name__ == "__main__":
    # Test loading data
    data = load_data("../data/matiks_data_clean.csv")
    if data is not None:
        print(f"Successfully loaded data with {len(data)} rows and {len(data.columns)} columns")
        print(f"Columns: {data.columns.tolist()}")
    else:
        print("Failed to load data") 