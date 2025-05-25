import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent directory to path to import data_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_loader import load_data

def perform_cohort_analysis(df):
    """
    Perform cohort analysis based on signup date
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned data with user_id, date, and signup_date columns
    
    Returns:
    --------
    pandas.DataFrame
        Cohort analysis results
    """
    # Ensure required columns exist
    if 'user_id' not in df.columns or 'date' not in df.columns:
        raise ValueError("DataFrame must contain 'user_id' and 'date' columns")
    
    # If signup_date doesn't exist, use the earliest date for each user as signup_date
    if 'signup_date' not in df.columns:
        print("signup_date column not found. Using earliest date for each user as signup_date.")
        signup_dates = df.groupby('user_id')['date'].min().reset_index()
        signup_dates.columns = ['user_id', 'signup_date']
        df = pd.merge(df, signup_dates, on='user_id')
    
    # Extract the year and month from signup_date to create cohort groups
    df['cohort'] = df['signup_date'].dt.to_period('M')
    
    # Calculate the month offset (difference between activity month and signup month)
    df['activity_month'] = df['date'].dt.to_period('M')
    df['month_offset'] = (df['activity_month'].dt.start_time.dt.to_period('M') - 
                         df['cohort'].dt.start_time.dt.to_period('M')).apply(lambda x: x.n)
    
    # Count the number of users in each cohort and month offset
    cohort_data = df.groupby(['cohort', 'month_offset'])['user_id'].nunique().reset_index()
    
    # Pivot the data to create the cohort table
    cohort_table = cohort_data.pivot(index='cohort', columns='month_offset', values='user_id')
    
    # Calculate retention rates
    cohort_sizes = cohort_table[0]
    retention_table = cohort_table.div(cohort_sizes, axis=0) * 100
    
    return retention_table

def plot_cohort_heatmap(retention_table, output_path=None):
    """
    Create a heatmap visualization of cohort retention
    
    Parameters:
    -----------
    retention_table : pandas.DataFrame
        Cohort retention table
    output_path : str, optional
        Path to save the visualization
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    plt.figure(figsize=(15, 8))
    plt.title('Cohort Retention Analysis (%)')
    
    # Create the heatmap
    sns.heatmap(retention_table, annot=True, fmt='.1f', cmap='YlGnBu', linewidths=.5)
    
    plt.ylabel('Cohort (Signup Month)')
    plt.xlabel('Month Offset')
    
    # Save the figure if output_path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Cohort heatmap saved to {output_path}")
    
    return plt.gcf()

def perform_funnel_analysis(df, stage_columns, output_path=None):
    """
    Perform funnel analysis for user progression
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned data with stage columns
    stage_columns : list
        List of column names representing stages in the funnel
    output_path : str, optional
        Path to save the visualization
    
    Returns:
    --------
    pandas.DataFrame
        Funnel analysis results
    """
    # Ensure all stage columns exist
    for col in stage_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")
    
    # Count users at each stage
    funnel_data = {}
    for col in stage_columns:
        # Count users who have reached this stage (non-null values)
        funnel_data[col] = df[col].notna().sum()
    
    # Create a DataFrame for the funnel data
    funnel_df = pd.DataFrame(list(funnel_data.items()), columns=['Stage', 'Users'])
    
    # Calculate conversion rate
    funnel_df['Conversion_Rate'] = 100 * funnel_df['Users'] / funnel_df['Users'].iloc[0]
    
    # Calculate drop-off
    funnel_df['Drop_Off'] = funnel_df['Users'].shift(1) - funnel_df['Users']
    funnel_df['Drop_Off_Rate'] = 100 * funnel_df['Drop_Off'] / funnel_df['Users'].shift(1)
    
    # Fill NaN values for the first stage
    funnel_df.fillna({'Drop_Off': 0, 'Drop_Off_Rate': 0}, inplace=True)
    
    # Create a visualization if output_path is provided
    if output_path:
        plt.figure(figsize=(12, 6))
        
        # Create the funnel chart
        plt.bar(funnel_df['Stage'], funnel_df['Users'], color='skyblue')
        
        # Add user count labels
        for i, (_, row) in enumerate(funnel_df.iterrows()):
            plt.text(i, row['Users'], f"{row['Users']}\n({row['Conversion_Rate']:.1f}%)", 
                     ha='center', va='bottom')
        
        plt.title('User Funnel Analysis')
        plt.ylabel('Number of Users')
        plt.xlabel('Funnel Stage')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Funnel analysis chart saved to {output_path}")
    
    return funnel_df

if __name__ == "__main__":
    # Load cleaned data
    data_path = "../data/matiks_data_clean.csv"
    df = load_data(data_path)
    
    if df is not None:
        # Create output directory if it doesn't exist
        os.makedirs("../output", exist_ok=True)
        
        # Perform cohort analysis
        try:
            retention_table = perform_cohort_analysis(df)
            
            # Save retention table
            retention_table.to_csv("../data/cohort_retention.csv")
            
            # Create and save cohort heatmap
            plot_cohort_heatmap(retention_table, "../output/cohort_heatmap.png")
        except Exception as e:
            print(f"Error performing cohort analysis: {e}")
        
        # Perform funnel analysis if appropriate columns exist
        # Note: The actual column names should be adjusted based on the real data
        try:
            # This is just an example - adjust the stage columns based on actual data
            funnel_stages = ['signup_date', 'first_game', 'repeat_session']
            funnel_df = perform_funnel_analysis(df, funnel_stages, "../output/funnel_analysis.png")
            
            # Save funnel analysis
            funnel_df.to_csv("../data/funnel_analysis.csv", index=False)
        except Exception as e:
            print(f"Error performing funnel analysis: {e}") 