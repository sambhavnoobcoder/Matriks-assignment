import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

def display_user_clusters(data, filtered_data):
    """
    Display user clustering page
    
    Parameters:
    -----------
    data : dict
        Dictionary containing all loaded DataFrames
    filtered_data : pandas.DataFrame
        Filtered raw data based on date selection
    """
    st.markdown('<div class="sub-header">User Clusters Analysis</div>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    This section uses machine learning to identify natural groups (clusters) of users based on their behavior and characteristics,
    helping to understand different user types and their needs.
    """)
    
    # Check if we have the necessary data
    if data['user_clusters'] is not None and data['cluster_analysis'] is not None:
        # Check if we need to rename columns
        user_clusters_df = data['user_clusters'].copy()
        cluster_analysis_df = data['cluster_analysis'].copy()
        
        # Check and rename columns if needed in user_clusters
        if 'cluster' not in user_clusters_df.columns:
            # Try to find a similar column
            potential_cluster_cols = [col for col in user_clusters_df.columns if 'cluster' in col.lower()]
            if potential_cluster_cols:
                user_clusters_df['cluster'] = user_clusters_df[potential_cluster_cols[0]]
                st.info(f"Using '{potential_cluster_cols[0]}' as the cluster column.")
            else:
                st.error("No cluster column found in user clusters data. Available columns: " + 
                         ", ".join(user_clusters_df.columns))
                return
        
        # Check and rename columns in cluster_analysis
        # First make sure the index is properly set (in case it's been saved as a column)
        if 'Unnamed: 0' in cluster_analysis_df.columns:
            cluster_analysis_df = cluster_analysis_df.set_index('Unnamed: 0')
        
        # If there's no 'cluster' column, the index may be the cluster column
        if 'cluster' not in cluster_analysis_df.columns:
            # Try to create a cluster column from the index
            try:
                cluster_analysis_df['cluster'] = cluster_analysis_df.index
                st.info("Using index as the cluster column.")
            except:
                # Look for similar columns
                potential_cluster_cols = [col for col in cluster_analysis_df.columns if 'cluster' in col.lower()]
                if potential_cluster_cols:
                    cluster_analysis_df['cluster'] = cluster_analysis_df[potential_cluster_cols[0]]
                    st.info(f"Using '{potential_cluster_cols[0]}' as the cluster column.")
                else:
                    st.error("No cluster column found in cluster analysis data. Available columns: " + 
                             ", ".join(cluster_analysis_df.columns))
                    return
        
        # Check for user_count and user_percentage columns or alternatives
        if 'user_count' not in cluster_analysis_df.columns:
            # Look for alternatives like 'size' or 'count'
            size_cols = [col for col in cluster_analysis_df.columns if col in ['size', 'count', 'users']]
            if size_cols:
                cluster_analysis_df['user_count'] = cluster_analysis_df[size_cols[0]]
            else:
                # Create a default user_count based on cluster distribution
                cluster_counts = user_clusters_df['cluster'].value_counts()
                for idx, row in cluster_analysis_df.iterrows():
                    cluster_id = row['cluster']
                    if cluster_id in cluster_counts:
                        cluster_analysis_df.at[idx, 'user_count'] = cluster_counts[cluster_id]
        
        if 'user_percentage' not in cluster_analysis_df.columns:
            # Look for alternatives like 'percentage' or 'pct'
            pct_cols = [col for col in cluster_analysis_df.columns if col in ['percentage', 'pct', 'percent']]
            if pct_cols:
                cluster_analysis_df['user_percentage'] = cluster_analysis_df[pct_cols[0]]
            elif 'user_count' in cluster_analysis_df.columns:
                # Calculate percentage
                total_users = cluster_analysis_df['user_count'].sum()
                if total_users > 0:
                    cluster_analysis_df['user_percentage'] = (cluster_analysis_df['user_count'] / total_users) * 100
        
        # User clusters overview
        st.markdown('### Cluster Distribution')
        
        # Count users in each cluster
        cluster_counts = user_clusters_df['cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'User Count']
        cluster_counts['Percentage'] = (cluster_counts['User Count'] / cluster_counts['User Count'].sum() * 100).round(1)
        
        # Create pie chart for cluster distribution
        fig = px.pie(
            cluster_counts,
            values='User Count',
            names='Cluster',
            title='User Distribution by Cluster',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Cluster Characteristics", "Cluster Comparison", "Cluster Scatter Plot"])
        
        with tab1:
            # Display cluster characteristics
            st.markdown('#### Cluster Characteristics')
            
            # Remove non-feature columns for visualization
            exclude_cols = ['cluster', 'user_count', 'user_percentage', 'size', 'percentage', 'count', 'users', 'pct', 'percent']
            feature_cols = [col for col in cluster_analysis_df.columns if col not in exclude_cols]
            
            if feature_cols:
                # Let user select a cluster to analyze
                cluster_ids = sorted(cluster_analysis_df['cluster'].unique())
                selected_cluster = st.selectbox('Select Cluster to Analyze', cluster_ids)
                
                # Filter data for selected cluster
                cluster_data = cluster_analysis_df[cluster_analysis_df['cluster'] == selected_cluster]
                
                if not cluster_data.empty:
                    # Display cluster size
                    if 'user_count' in cluster_data.columns:
                        cluster_size = cluster_data['user_count'].values[0]
                    else:
                        cluster_size = "Unknown"
                        
                    if 'user_percentage' in cluster_data.columns:
                        cluster_pct = cluster_data['user_percentage'].values[0]
                    else:
                        cluster_pct = "Unknown"
                    
                    st.markdown(f"""
                    <div class="insight-box">
                    <strong>Cluster {selected_cluster} Size:</strong> {cluster_size} users ({cluster_pct}% of total)
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create a radar chart of cluster characteristics
                    if len(feature_cols) > 0:
                        # Normalize features for radar chart
                        radar_data = cluster_analysis_df[feature_cols].copy()
                        for col in feature_cols:
                            max_val = radar_data[col].max()
                            min_val = radar_data[col].min()
                            if max_val > min_val:
                                radar_data[col] = (radar_data[col] - min_val) / (max_val - min_val)
                            else:
                                radar_data[col] = 0
                        
                        # Add cluster column back
                        radar_data['cluster'] = cluster_analysis_df['cluster']
                        
                        # Create radar chart data
                        fig = go.Figure()
                        
                        # Add selected cluster
                        selected_data = radar_data[radar_data['cluster'] == selected_cluster].iloc[0]
                        selected_values = selected_data[feature_cols].values.tolist()
                        selected_values.append(selected_values[0])  # Close the loop
                        
                        # Add radar chart for selected cluster
                        fig.add_trace(go.Scatterpolar(
                            r=selected_values,
                            theta=feature_cols + [feature_cols[0]],  # Close the loop
                            fill='toself',
                            name=f'Cluster {selected_cluster}',
                            line=dict(color='rgb(31, 119, 180)', width=3)
                        ))
                        
                        # Add average cluster for comparison
                        avg_values = radar_data[feature_cols].mean().values.tolist()
                        avg_values.append(avg_values[0])  # Close the loop
                        
                        fig.add_trace(go.Scatterpolar(
                            r=avg_values,
                            theta=feature_cols + [feature_cols[0]],  # Close the loop
                            fill='toself',
                            name='Average',
                            line=dict(color='rgba(255, 0, 0, 0.5)', width=2, dash='dot')
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )
                            ),
                            title=f'Characteristics of Cluster {selected_cluster} vs Average',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Find the top features that define this cluster
                        cluster_data = cluster_analysis_df[cluster_analysis_df['cluster'] == selected_cluster].iloc[0][feature_cols]
                        avg_data = cluster_analysis_df[feature_cols].mean()
                        
                        # Calculate percentage difference from average
                        diff_pct = ((cluster_data - avg_data) / avg_data * 100).fillna(0)
                        
                        # Sort features by absolute difference
                        top_features = diff_pct.abs().sort_values(ascending=False).head(3).index.tolist()
                        
                        # Display insights
                        st.markdown(f"""
                        <div class="insight-box">
                        <strong>Key Characteristics of Cluster {selected_cluster}:</strong>
                        <ul>
                        """, unsafe_allow_html=True)
                        
                        for feature in top_features:
                            value = cluster_data[feature]
                            avg = avg_data[feature]
                            diff = diff_pct[feature]
                            comparison = "higher" if diff > 0 else "lower"
                            
                            st.markdown(f"""
                            <li><strong>{feature}:</strong> {value:.2f} ({abs(diff):.1f}% {comparison} than average)</li>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("""
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("No feature columns available for visualization.")
                else:
                    st.warning("Cluster data is empty.")
            else:
                st.warning("No feature columns available for visualization.")
        
        with tab2:
            # Cluster comparison
            st.markdown('#### Cluster Comparison')
            
            # Use the already verified cluster_analysis_df
            if 'cluster' in cluster_analysis_df.columns:
                # Remove non-feature columns for visualization
                exclude_cols = ['cluster', 'user_count', 'user_percentage', 'size', 'percentage', 'count', 'users', 'pct', 'percent']
                feature_cols = [col for col in cluster_analysis_df.columns if col not in exclude_cols]
                
                if feature_cols:
                    # Let user select a feature for comparison
                    selected_feature = st.selectbox('Select Feature for Comparison', feature_cols)
                    
                    # Create the comparison bar chart
                    fig = px.bar(
                        cluster_analysis_df,
                        x='cluster',
                        y=selected_feature,
                        title=f'Comparison of {selected_feature} Across Clusters',
                        color='cluster',
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        text_auto='.2f'
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display cluster user counts if available
                    if 'user_count' in cluster_analysis_df.columns:
                        fig = px.bar(
                            cluster_analysis_df,
                            x='cluster',
                            y='user_count',
                            title='User Count by Cluster',
                            color='cluster',
                            color_discrete_sequence=px.colors.qualitative.Bold,
                            text_auto=True
                        )
                        
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show cluster comparison table
                    st.markdown('#### Cluster Comparison Table')
                    
                    # Determine which columns to display
                    available_cols = ['cluster']
                    if 'user_count' in cluster_analysis_df.columns:
                        available_cols.append('user_count')
                    if 'user_percentage' in cluster_analysis_df.columns:
                        available_cols.append('user_percentage')
                    available_cols.extend(feature_cols)
                    
                    # Format the table for display
                    display_df = cluster_analysis_df[available_cols].copy()
                    
                    # Format percentage if available
                    if 'user_percentage' in display_df.columns:
                        display_df['user_percentage'] = display_df['user_percentage'].apply(lambda x: f"{x:.1f}%")
                    
                    # Rename columns for better display
                    display_df.columns = [col.replace('_', ' ').title() for col in display_df.columns]
                    
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.warning("No feature columns available for comparison.")
            else:
                st.warning("Cluster column not found in cluster analysis data.")
        
        with tab3:
            # Create scatter plot visualization
            st.markdown('#### Cluster Scatter Plot')
            
            # Use the already verified user_clusters_df and cluster_analysis_df
            if 'cluster' in user_clusters_df.columns and 'cluster' in cluster_analysis_df.columns:
                # Remove non-feature columns for visualization
                exclude_cols = ['cluster', 'user_count', 'user_percentage', 'size', 'percentage', 'count', 'users', 'pct', 'percent']
                feature_cols = [col for col in cluster_analysis_df.columns if col not in exclude_cols]
                
                if feature_cols:
                    # Get feature columns that exist in both dataframes
                    common_features = [col for col in feature_cols if col in user_clusters_df.columns]
                    
                    if common_features:
                        # Let user select features for scatter plot
                        x_feature = st.selectbox('Select X-axis Feature', common_features, index=0)
                        y_feature = st.selectbox('Select Y-axis Feature', common_features, index=min(1, len(common_features)-1))
                        
                        # Create scatter plot
                        fig = px.scatter(
                            user_clusters_df,
                            x=x_feature,
                            y=y_feature,
                            color='cluster',
                            title=f'User Clusters: {x_feature} vs {y_feature}',
                            color_discrete_sequence=px.colors.qualitative.Bold,
                            opacity=0.7
                        )
                        
                        # Add cluster centers if these features exist in cluster_analysis_df
                        if x_feature in cluster_analysis_df.columns and y_feature in cluster_analysis_df.columns:
                            for _, row in cluster_analysis_df.iterrows():
                                try:
                                    cluster_id = int(row['cluster'])
                                except:
                                    cluster_id = row['cluster']
                                    
                                fig.add_annotation(
                                    x=row[x_feature],
                                    y=row[y_feature],
                                    text=f"Cluster {cluster_id}",
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowsize=1,
                                    arrowwidth=2,
                                    ax=0,
                                    ay=-40
                                )
                        
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display cluster insights
                        st.markdown("""
                        <div class="insight-box">
                        <strong>Cluster Interpretation:</strong>
                        <ul>
                            <li>Distinct clusters indicate different user segments with unique behavior patterns.</li>
                            <li>The distance between clusters shows how different user groups are from each other.</li>
                            <li>The size of each cluster indicates the relative proportion of users in that segment.</li>
                            <li>Consider tailoring features and campaigns for each distinct user cluster.</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("No common feature columns found between user clusters and cluster analysis data.")
                else:
                    st.warning("No feature columns available for visualization.")
            else:
                st.warning("Cluster columns not found in data. Please ensure clustering has been performed.")
    else:
        st.warning("User clustering data is not available. Please run the data processing scripts first.") 