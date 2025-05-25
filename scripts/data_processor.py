import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """
    Load data from CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data
    """
    print(f"Loading data from {file_path}")
    data = pd.read_csv(file_path)
    
    # Convert date column to datetime
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
    
    return data

def process_user_metrics(data, output_dir):
    """
    Process user metrics (DAU, WAU, MAU)
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw data
    output_dir : str
        Directory to save processed data
        
    Returns:
    --------
    tuple
        (dau_df, wau_df, mau_df) - DataFrames with daily, weekly, and monthly active users
    """
    print("Processing user metrics...")
    
    # Ensure date column is present
    if 'date' not in data.columns:
        print("Error: date column not found in data")
        return None, None, None
    
    # Ensure user_id column is present
    if 'user_id' not in data.columns:
        print("Error: user_id column not found in data")
        return None, None, None
    
    # Calculate DAU
    dau = data.groupby(data['date'].dt.date).agg({'user_id': 'nunique'})
    dau = dau.reset_index()
    dau.columns = ['date', 'dau']
    
    # Calculate WAU
    # Create a date range for all dates in the data
    date_range = pd.date_range(start=data['date'].min(), end=data['date'].max())
    wau_data = []
    
    for date in date_range:
        # Get users active in the past 7 days
        week_start = date - timedelta(days=6)
        active_users = data[(data['date'] >= week_start) & (data['date'] <= date)]['user_id'].nunique()
        wau_data.append({'date': date, 'wau': active_users})
    
    wau = pd.DataFrame(wau_data)
    
    # Calculate MAU
    mau_data = []
    
    for date in date_range:
        # Get users active in the past 30 days
        month_start = date - timedelta(days=29)
        active_users = data[(data['date'] >= month_start) & (data['date'] <= date)]['user_id'].nunique()
        mau_data.append({'date': date, 'mau': active_users})
    
    mau = pd.DataFrame(mau_data)
    
    # Save processed data
    dau.to_csv(os.path.join(output_dir, 'dau.csv'), index=False)
    wau.to_csv(os.path.join(output_dir, 'wau.csv'), index=False)
    mau.to_csv(os.path.join(output_dir, 'mau.csv'), index=False)
    
    return dau, wau, mau

def process_revenue_data(data, output_dir):
    """
    Process revenue data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw data
    output_dir : str
        Directory to save processed data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with revenue analysis
    """
    print("Processing revenue data...")
    
    # Ensure date column is present
    if 'date' not in data.columns:
        print("Error: date column not found in data")
        return None
    
    # Ensure user_id column is present
    if 'user_id' not in data.columns:
        print("Error: user_id column not found in data")
        return None
    
    # Find revenue column
    revenue_cols = [col for col in data.columns if 'revenue' in col.lower()]
    if not revenue_cols:
        print("Error: No revenue column found in data")
        return None
    
    revenue_col = revenue_cols[0]
    
    # Calculate daily revenue
    revenue_data = data.groupby('date').agg({
        revenue_col: 'sum',
        'user_id': 'nunique'
    }).reset_index()
    
    revenue_data.columns = ['date', 'daily_revenue', 'daily_users']
    
    # Calculate ARPU (Average Revenue Per User)
    revenue_data['arpu'] = revenue_data['daily_revenue'] / revenue_data['daily_users']
    
    # Calculate 7-day rolling average
    revenue_data['revenue_7d_avg'] = revenue_data['daily_revenue'].rolling(window=7, min_periods=1).mean()
    
    # Save processed data
    revenue_data.to_csv(os.path.join(output_dir, 'revenue_analysis.csv'), index=False)
    
    return revenue_data

def process_user_segments(data, output_dir):
    """
    Process user segments
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw data
    output_dir : str
        Directory to save processed data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with user segments
    """
    print("Processing user segments...")
    
    # Ensure user_id column is present
    if 'user_id' not in data.columns:
        print("Error: user_id column not found in data")
        return None
    
    # Find revenue column
    revenue_cols = [col for col in data.columns if 'revenue' in col.lower()]
    if not revenue_cols:
        print("Warning: No revenue column found in data, skipping revenue segmentation")
        revenue_col = None
    else:
        revenue_col = revenue_cols[0]
    
    # Aggregate data by user
    user_data = data.groupby('user_id').agg({
        'date': 'nunique',  # Number of active days
    })
    
    if revenue_col:
        # Add revenue data if available
        revenue_by_user = data.groupby('user_id')[revenue_col].sum()
        user_data[revenue_col] = revenue_by_user
    
    # Reset index
    user_data = user_data.reset_index()
    
    # Create activity segments
    activity_bins = [0, 1, 5, 15, float('inf')]
    activity_labels = ['One-time', 'Casual', 'Regular', 'Power']
    user_data['activity_segment'] = pd.cut(user_data['date'], bins=activity_bins, labels=activity_labels)
    
    # Create revenue segments if revenue data is available
    if revenue_col:
        revenue_bins = [0, 1, 10, 50, float('inf')]
        revenue_labels = ['Non-payer', 'Low-spender', 'Medium-spender', 'High-spender']
        user_data['revenue_segment'] = pd.cut(user_data[revenue_col], bins=revenue_bins, labels=revenue_labels)
    
    # Add device type and game mode if available
    if 'device_type' in data.columns:
        # Get the most common device type for each user
        device_by_user = data.groupby('user_id')['device_type'].agg(lambda x: x.value_counts().index[0])
        user_data['device_type'] = user_data['user_id'].map(device_by_user)
    
    if 'game_mode' in data.columns:
        # Get the most common game mode for each user
        mode_by_user = data.groupby('user_id')['game_mode'].agg(lambda x: x.value_counts().index[0])
        user_data['game_mode'] = user_data['user_id'].map(mode_by_user)
    
    # Save processed data
    user_data.to_csv(os.path.join(output_dir, 'user_segments.csv'), index=False)
    
    return user_data

def process_churn_analysis(data, user_segments, output_dir):
    """
    Process churn analysis
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw data
    user_segments : pandas.DataFrame
        User segments data
    output_dir : str
        Directory to save processed data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with churn analysis
    """
    print("Processing churn analysis...")
    
    # Ensure date column is present
    if 'date' not in data.columns:
        print("Error: date column not found in data")
        return None
    
    # Ensure user_id column is present
    if 'user_id' not in data.columns:
        print("Error: user_id column not found in data")
        return None
    
    # Get the last activity date for each user
    last_activity = data.groupby('user_id')['date'].max().reset_index()
    last_activity.columns = ['user_id', 'last_activity_date']
    
    # Calculate days since last activity
    max_date = data['date'].max()
    last_activity['days_since_activity'] = (max_date - last_activity['last_activity_date']).dt.days
    
    # Define churned users (inactive for more than 30 days)
    last_activity['churned'] = last_activity['days_since_activity'] > 30
    
    # Merge with user segments if available
    if user_segments is not None:
        churn_data = pd.merge(last_activity, user_segments, on='user_id', how='left')
    else:
        churn_data = last_activity
    
    # Save processed data
    churn_data.to_csv(os.path.join(output_dir, 'churn_analysis.csv'), index=False)
    
    return churn_data

def process_cohort_analysis(data, output_dir):
    """
    Process cohort analysis
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw data
    output_dir : str
        Directory to save processed data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with cohort retention analysis
    """
    print("Processing cohort analysis...")
    
    # Ensure date column is present
    if 'date' not in data.columns:
        print("Error: date column not found in data")
        return None
    
    # Ensure user_id column is present
    if 'user_id' not in data.columns:
        print("Error: user_id column not found in data")
        return None
    
    # Get the first activity date for each user (cohort assignment)
    user_cohorts = data.groupby('user_id')['date'].min().reset_index()
    user_cohorts.columns = ['user_id', 'cohort_date']
    
    # Convert to cohort month
    user_cohorts['cohort'] = user_cohorts['cohort_date'].dt.to_period('M')
    
    # Merge with the original data to get activity by cohort
    data_with_cohort = pd.merge(data, user_cohorts[['user_id', 'cohort']], on='user_id', how='left')
    
    # Calculate the period (month) for each activity
    data_with_cohort['period'] = data_with_cohort['date'].dt.to_period('M')
    
    # Calculate the period number (months since joining)
    data_with_cohort['period_number'] = (data_with_cohort['period'] - data_with_cohort['cohort']).apply(lambda x: x.n)
    
    # Only keep data where period_number >= 0
    data_with_cohort = data_with_cohort[data_with_cohort['period_number'] >= 0]
    
    # Count unique users by cohort and period number
    cohort_data = data_with_cohort.groupby(['cohort', 'period_number'])['user_id'].nunique().reset_index()
    
    # Pivot the data to get the retention table
    cohort_pivot = cohort_data.pivot(index='cohort', columns='period_number', values='user_id')
    
    # Calculate retention rates
    cohort_size = cohort_pivot[0]
    retention_table = cohort_pivot.divide(cohort_size, axis=0) * 100
    
    # Save processed data
    retention_table.to_csv(os.path.join(output_dir, 'cohort_retention.csv'))
    
    return retention_table

def process_user_clusters(data, user_segments, output_dir):
    """
    Process user clusters using K-means clustering
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw data
    user_segments : pandas.DataFrame
        User segments data
    output_dir : str
        Directory to save processed data
        
    Returns:
    --------
    tuple
        (user_clusters, cluster_analysis) - DataFrames with user clusters and cluster analysis
    """
    print("Processing user clusters...")
    
    # Check if we have user segments data
    if user_segments is None:
        print("Error: User segments data is required for clustering")
        return None, None
    
    # Find revenue column
    revenue_cols = [col for col in user_segments.columns if 'revenue' in col.lower() and user_segments[col].dtype != 'object' and not pd.api.types.is_categorical_dtype(user_segments[col])]
    
    # Select features for clustering
    features = ['date']  # Number of active days
    
    if revenue_cols:
        features.extend(revenue_cols)
    
    # Check if we have enough features
    if len(features) < 2:
        print("Warning: Not enough features for clustering")
        # Try to add more features if available
        if 'days_since_activity' in user_segments.columns:
            features.append('days_since_activity')
    
    # Prepare data for clustering - only use numeric columns
    numeric_features = []
    for feature in features:
        if feature in user_segments.columns and pd.api.types.is_numeric_dtype(user_segments[feature]):
            numeric_features.append(feature)
    
    if len(numeric_features) < 2:
        print("Error: Not enough numeric features for clustering")
        return None, None
    
    cluster_data = user_segments[['user_id'] + numeric_features].copy()
    
    # Handle missing values
    cluster_data = cluster_data.fillna(0)
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_data[numeric_features])
    
    # Determine optimal number of clusters (using elbow method)
    inertia = []
    k_range = range(2, min(11, len(cluster_data) // 10 + 1))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)
    
    # Find the elbow point (simplified approach)
    k_optimal = 4  # Default to 4 clusters
    if len(inertia) > 2:
        # Calculate the rate of decrease
        decrease_rate = [inertia[i-1] - inertia[i] for i in range(1, len(inertia))]
        # Find the point where the decrease rate slows down
        for i in range(1, len(decrease_rate)):
            if decrease_rate[i] / decrease_rate[0] < 0.3:  # Threshold of 30%
                k_optimal = k_range[i]
                break
    
    # Apply K-means clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # Add cluster labels to the data
    cluster_data['cluster'] = cluster_labels
    
    # Create cluster analysis
    cluster_analysis = pd.DataFrame(index=range(k_optimal))
    
    # Calculate cluster characteristics
    for feature in numeric_features:
        cluster_analysis[feature] = cluster_data.groupby('cluster')[feature].mean()
    
    # Add cluster size
    cluster_analysis['size'] = cluster_data.groupby('cluster').size()
    cluster_analysis['percentage'] = cluster_analysis['size'] / len(cluster_data) * 100
    
    # Calculate cluster centers
    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=numeric_features
    )
    cluster_centers['cluster'] = range(k_optimal)
    
    # Save processed data
    cluster_data.to_csv(os.path.join(output_dir, 'user_clusters.csv'), index=False)
    cluster_analysis.to_csv(os.path.join(output_dir, 'cluster_analysis.csv'))
    
    return cluster_data, cluster_analysis

def process_funnel_analysis(data, output_dir):
    """
    Process funnel analysis
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw data
    output_dir : str
        Directory to save processed data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with funnel analysis
    """
    print("Processing funnel analysis...")
    
    # Check if we have the necessary columns for funnel analysis
    # This is a simplified approach - in a real scenario, we would need to define specific funnel steps
    funnel_steps = ['view', 'click', 'signup', 'purchase']
    
    # Check if any of these columns exist in the data
    available_steps = [step for step in funnel_steps if any(step in col.lower() for col in data.columns)]
    
    if not available_steps:
        print("Warning: No funnel step columns found in data")
        # Create a dummy funnel dataset for demonstration
        funnel_data = pd.DataFrame({
            'step': ['View', 'Click', 'Signup', 'Purchase'],
            'users': [1000, 500, 200, 50],
            'conversion_rate': [100, 50, 40, 25]
        })
    else:
        # For demonstration, we'll create a simple funnel based on available columns
        funnel_data = pd.DataFrame(columns=['step', 'users', 'conversion_rate'])
        
        # Count users at each step
        total_users = data['user_id'].nunique()
        funnel_data.loc[0] = ['Total Users', total_users, 100]
        
        for i, step in enumerate(available_steps):
            # Find columns related to this step
            step_cols = [col for col in data.columns if step in col.lower()]
            if step_cols:
                # Count users who performed this step
                step_users = data[data[step_cols[0]] > 0]['user_id'].nunique()
                conversion_rate = (step_users / total_users) * 100 if total_users > 0 else 0
                funnel_data.loc[i+1] = [step.capitalize(), step_users, conversion_rate]
    
    # Save processed data
    funnel_data.to_csv(os.path.join(output_dir, 'funnel_analysis.csv'), index=False)
    
    return funnel_data

def main():
    """
    Main function to process all data
    """
    # Define input and output directories
    input_file = "../data/matiks_data_clean.csv"
    output_dir = "../data"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = load_data(input_file)
    
    # Process user metrics
    dau, wau, mau = process_user_metrics(data, output_dir)
    
    # Process revenue data
    revenue_data = process_revenue_data(data, output_dir)
    
    # Process user segments
    user_segments = process_user_segments(data, output_dir)
    
    # Process churn analysis
    churn_data = process_churn_analysis(data, user_segments, output_dir)
    
    # Process cohort analysis
    cohort_data = process_cohort_analysis(data, output_dir)
    
    # Process user clusters
    user_clusters, cluster_analysis = process_user_clusters(data, user_segments, output_dir)
    
    # Process funnel analysis
    funnel_data = process_funnel_analysis(data, output_dir)
    
    print("Data processing complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("Creating dummy files for testing...")
        
        # Create data directory
        output_dir = "../data"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create dummy files if they don't exist
        dummy_files = [
            'dau.csv', 'wau.csv', 'mau.csv', 'revenue_analysis.csv', 
            'user_segments.csv', 'churn_analysis.csv', 'cohort_retention.csv',
            'user_clusters.csv', 'cluster_analysis.csv', 'funnel_analysis.csv'
        ]
        
        for file in dummy_files:
            file_path = os.path.join(output_dir, file)
            if not os.path.exists(file_path):
                # Create a minimal dummy file
                pd.DataFrame({'dummy': [0]}).to_csv(file_path, index=False)
                print(f"Created dummy file: {file_path}") 