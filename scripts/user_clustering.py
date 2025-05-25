import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import sys

# Add parent directory to path to import data_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_loader import load_data

def prepare_clustering_data(df):
    """
    Prepare data for user clustering
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned data with user_id, activity, and revenue information
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame prepared for clustering
    """
    # Ensure user_id column exists
    if 'user_id' not in df.columns:
        raise ValueError("DataFrame must contain 'user_id' column")
    
    # Create features for clustering
    cluster_features = []
    
    # 1. Activity frequency
    user_activity_count = df.groupby('user_id').size().reset_index(name='activity_count')
    cluster_features.append(user_activity_count)
    
    # 2. Revenue (if available)
    revenue_cols = [col for col in df.columns if 'revenue' in col.lower()]
    if revenue_cols:
        user_revenue = df.groupby('user_id')[revenue_cols].sum().reset_index()
        
        # If there are multiple revenue columns, add a total
        if len(revenue_cols) > 1:
            user_revenue['total_revenue'] = user_revenue[revenue_cols].sum(axis=1)
            # Keep only user_id and total_revenue
            user_revenue = user_revenue[['user_id', 'total_revenue']]
        
        cluster_features.append(user_revenue)
    
    # 3. Session duration (if available)
    if 'session_duration' in df.columns:
        user_session_duration = df.groupby('user_id')['session_duration'].mean().reset_index(name='avg_session_duration')
        cluster_features.append(user_session_duration)
    
    # 4. Days since first activity
    if 'date' in df.columns:
        user_first_activity = df.groupby('user_id')['date'].min().reset_index(name='first_activity')
        user_last_activity = df.groupby('user_id')['date'].max().reset_index(name='last_activity')
        user_activity_span = pd.merge(user_first_activity, user_last_activity, on='user_id')
        user_activity_span['activity_span_days'] = (user_activity_span['last_activity'] - user_activity_span['first_activity']).dt.days
        cluster_features.append(user_activity_span[['user_id', 'activity_span_days']])
    
    # 5. Device type distribution (if available)
    if 'device_type' in df.columns:
        device_dummies = pd.get_dummies(df[['user_id', 'device_type']], columns=['device_type'], prefix='', prefix_sep='')
        device_ratios = device_dummies.groupby('user_id').mean().reset_index()
        cluster_features.append(device_ratios)
    
    # Merge all features
    clustering_data = cluster_features[0]
    for feature_df in cluster_features[1:]:
        clustering_data = pd.merge(clustering_data, feature_df, on='user_id', how='left')
    
    # Fill missing values
    clustering_data = clustering_data.fillna(0)
    
    return clustering_data

def perform_kmeans_clustering(data, n_clusters=4):
    """
    Perform K-means clustering on user data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Prepared data for clustering
    n_clusters : int
        Number of clusters
    
    Returns:
    --------
    tuple
        (data_with_clusters, kmeans_model, scaled_features)
    """
    # Extract user_id
    user_ids = data['user_id']
    
    # Select numerical features for clustering
    features = data.drop('user_id', axis=1)
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)
    
    # Add cluster labels to the original data
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = clusters
    
    return data_with_clusters, kmeans, scaled_features

def find_optimal_clusters(scaled_features, max_clusters=10):
    """
    Find the optimal number of clusters using the elbow method
    
    Parameters:
    -----------
    scaled_features : numpy.ndarray
        Scaled features for clustering
    max_clusters : int
        Maximum number of clusters to try
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='-')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.tight_layout()
    
    return plt.gcf()

def visualize_clusters(data_with_clusters, feature1, feature2, output_path=None):
    """
    Visualize clusters using scatter plot
    
    Parameters:
    -----------
    data_with_clusters : pandas.DataFrame
        Data with cluster labels
    feature1 : str
        First feature for visualization
    feature2 : str
        Second feature for visualization
    output_path : str, optional
        Path to save the visualization
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with clusters
    sns.scatterplot(data=data_with_clusters, x=feature1, y=feature2, hue='cluster', palette='viridis', s=100, alpha=0.7)
    
    plt.title(f'User Clusters: {feature1} vs {feature2}')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.tight_layout()
    
    # Save the figure if output_path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Cluster visualization saved to {output_path}")
    
    return plt.gcf()

def analyze_clusters(data_with_clusters):
    """
    Analyze characteristics of each cluster
    
    Parameters:
    -----------
    data_with_clusters : pandas.DataFrame
        Data with cluster labels
    
    Returns:
    --------
    pandas.DataFrame
        Cluster analysis results
    """
    # Group by cluster and calculate statistics
    cluster_analysis = data_with_clusters.groupby('cluster').mean()
    
    # Count users in each cluster
    cluster_counts = data_with_clusters.groupby('cluster').size().reset_index(name='user_count')
    
    # Merge counts with analysis
    cluster_analysis = cluster_analysis.reset_index().merge(cluster_counts, on='cluster')
    
    # Calculate percentage of users in each cluster
    total_users = cluster_analysis['user_count'].sum()
    cluster_analysis['user_percentage'] = 100 * cluster_analysis['user_count'] / total_users
    
    return cluster_analysis

if __name__ == "__main__":
    # Load cleaned data
    data_path = "../data/matiks_data_clean.csv"
    df = load_data(data_path)
    
    if df is not None:
        # Create output directory if it doesn't exist
        os.makedirs("../output", exist_ok=True)
        
        try:
            # Prepare data for clustering
            clustering_data = prepare_clustering_data(df)
            
            # Find optimal number of clusters
            scaled_features = StandardScaler().fit_transform(clustering_data.drop('user_id', axis=1))
            elbow_fig = find_optimal_clusters(scaled_features)
            elbow_fig.savefig("../output/elbow_method.png", bbox_inches='tight')
            
            # Perform clustering with the chosen number of clusters
            # Note: The number of clusters (4) can be adjusted based on the elbow method results
            data_with_clusters, kmeans, _ = perform_kmeans_clustering(clustering_data, n_clusters=4)
            
            # Save clustering results
            data_with_clusters.to_csv("../data/user_clusters.csv", index=False)
            
            # Analyze clusters
            cluster_analysis = analyze_clusters(data_with_clusters)
            cluster_analysis.to_csv("../data/cluster_analysis.csv", index=False)
            
            # Visualize clusters
            # Note: The feature names should be adjusted based on the actual data
            if 'activity_count' in data_with_clusters.columns and 'total_revenue' in data_with_clusters.columns:
                visualize_clusters(data_with_clusters, 'activity_count', 'total_revenue', 
                                  "../output/user_clusters_activity_revenue.png")
            
            print("User clustering completed successfully")
        except Exception as e:
            print(f"Error performing user clustering: {e}") 