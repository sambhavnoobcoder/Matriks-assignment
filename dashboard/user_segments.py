import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def display_user_segments(data, filtered_data):
    """
    Display user segments page
    
    Parameters:
    -----------
    data : dict
        Dictionary containing all loaded DataFrames
    filtered_data : pandas.DataFrame
        Filtered raw data based on date selection
    """
    st.markdown('<div class="sub-header">User Segments Analysis</div>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    This section analyzes different user segments based on activity levels, revenue contribution,
    and other characteristics to understand diverse user behaviors.
    """)
    
    # Check if we have the necessary data
    if data['user_segments'] is not None:
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Activity Segments", "Revenue Segments", "Device & Game Mode Analysis"])
        
        with tab1:
            # Check if activity_segment column exists
            if 'activity_segment' in data['user_segments'].columns:
                # Count users in each activity segment
                activity_counts = data['user_segments']['activity_segment'].value_counts().reset_index()
                activity_counts.columns = ['activity_segment', 'user_count']
                
                # Ensure proper ordering
                segment_order = ['Low', 'Medium', 'High', 'Very High']
                activity_counts['activity_segment'] = pd.Categorical(
                    activity_counts['activity_segment'], 
                    categories=segment_order, 
                    ordered=True
                )
                activity_counts = activity_counts.sort_values('activity_segment')
                
                # Create pie chart
                fig = px.pie(
                    activity_counts,
                    values='user_count',
                    names='activity_segment',
                    title='User Distribution by Activity Level',
                    color='activity_segment',
                    color_discrete_map={
                        'Low': 'lightblue',
                        'Medium': 'royalblue',
                        'High': 'darkblue',
                        'Very High': 'navy'
                    }
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=500)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate and display average activity count by segment if available
                if 'activity_count' in data['user_segments'].columns:
                    avg_activity = data['user_segments'].groupby('activity_segment')['activity_count'].mean().reset_index()
                    avg_activity = avg_activity.sort_values('activity_count')
                    
                    fig = px.bar(
                        avg_activity,
                        x='activity_segment',
                        y='activity_count',
                        title='Average User Activity by Segment',
                        labels={
                            'activity_segment': 'Activity Segment',
                            'activity_count': 'Average Activity Count'
                        },
                        color='activity_count',
                        color_continuous_scale='Blues'
                    )
                    
                    fig.update_layout(height=400, coloraxis_showscale=False)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Insights based on activity segments
                    st.markdown(f"""
                    <div class="insight-box">
                    <strong>Activity Segment Insights:</strong>
                    <ul>
                        <li>The largest user segment by activity is <strong>{activity_counts.iloc[activity_counts['user_count'].argmax()]['activity_segment']}</strong> ({activity_counts['user_count'].max() / activity_counts['user_count'].sum() * 100:.1f}% of users).</li>
                        <li>The most active users (Very High segment) perform <strong>{avg_activity[avg_activity['activity_segment'] == 'Very High']['activity_count'].values[0] if 'Very High' in avg_activity['activity_segment'].values else 0:.1f}</strong> activities on average.</li>
                        <li>Consider targeted engagement strategies for each activity segment to increase overall platform usage.</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Basic insights without activity_count
                    st.markdown(f"""
                    <div class="insight-box">
                    <strong>Activity Segment Insights:</strong>
                    <ul>
                        <li>The largest user segment by activity is <strong>{activity_counts.iloc[activity_counts['user_count'].argmax()]['activity_segment']}</strong> ({activity_counts['user_count'].max() / activity_counts['user_count'].sum() * 100:.1f}% of users).</li>
                        <li>Consider targeted engagement strategies for each activity segment to increase overall platform usage.</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Activity segment data is not available.")
        
        with tab2:
            # Check if revenue_segment column exists
            if 'revenue_segment' in data['user_segments'].columns:
                # Count users in each revenue segment
                revenue_counts = data['user_segments']['revenue_segment'].value_counts().reset_index()
                revenue_counts.columns = ['revenue_segment', 'user_count']
                
                # Ensure proper ordering
                segment_order = ['Low', 'Medium', 'High', 'Very High']
                revenue_counts['revenue_segment'] = pd.Categorical(
                    revenue_counts['revenue_segment'], 
                    categories=segment_order, 
                    ordered=True
                )
                revenue_counts = revenue_counts.sort_values('revenue_segment')
                
                # Create pie chart
                fig = px.pie(
                    revenue_counts,
                    values='user_count',
                    names='revenue_segment',
                    title='User Distribution by Revenue Contribution',
                    color='revenue_segment',
                    color_discrete_map={
                        'Low': 'lightgreen',
                        'Medium': 'yellowgreen',
                        'High': 'green',
                        'Very High': 'darkgreen'
                    }
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=500)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Check if total_revenue column exists
                if 'total_revenue' in data['user_segments'].columns:
                    # Use total_revenue column
                    revenue_col = 'total_revenue'
                elif 'revenue' in data['user_segments'].columns:
                    # Use revenue column
                    revenue_col = 'revenue'
                
                if 'total_revenue' in data['user_segments'].columns or 'revenue' in data['user_segments'].columns:
                    # Calculate and display average revenue by segment
                    avg_revenue = data['user_segments'].groupby('revenue_segment')[revenue_col].mean().reset_index()
                    avg_revenue = avg_revenue.sort_values(revenue_col)
                    
                    fig = px.bar(
                        avg_revenue,
                        x='revenue_segment',
                        y=revenue_col,
                        title='Average Revenue by Segment',
                        labels={
                            'revenue_segment': 'Revenue Segment',
                            revenue_col: 'Average Revenue ($)'
                        },
                        color=revenue_col,
                        color_continuous_scale='Greens'
                    )
                    
                    fig.update_layout(height=400, coloraxis_showscale=False)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate total revenue by segment
                    total_by_segment = data['user_segments'].groupby('revenue_segment')[revenue_col].sum().reset_index()
                    total_overall = total_by_segment[revenue_col].sum()
                    total_by_segment['percentage'] = total_by_segment[revenue_col] / total_overall * 100
                    
                    # Insights based on revenue segments
                    highest_segment = total_by_segment.loc[total_by_segment['percentage'].idxmax(), 'revenue_segment']
                    highest_pct = total_by_segment['percentage'].max()
                    
                    st.markdown(f"""
                    <div class="insight-box">
                    <strong>Revenue Segment Insights:</strong>
                    <ul>
                        <li>The <strong>{highest_segment}</strong> revenue segment contributes <strong>{highest_pct:.1f}%</strong> of the total revenue.</li>
                        <li>Average revenue from Very High segment users is <strong>${avg_revenue[avg_revenue['revenue_segment'] == 'Very High'][revenue_col].values[0] if 'Very High' in avg_revenue['revenue_segment'].values else 0:.2f}</strong>.</li>
                        <li>Consider developing specialized premium features for high-revenue users while creating more entry-level monetization options for low-revenue users.</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Revenue data is not available.")
            else:
                st.warning("Revenue segment data is not available.")
        
        with tab3:
            # Check if filtered_data contains device_type or game_mode
            segment_columns = [col for col in filtered_data.columns if col in ['device_type', 'game_mode']]
            
            if segment_columns:
                # Let user select which segment to analyze
                selected_segment = st.selectbox("Select Segment to Analyze", segment_columns)
                
                if selected_segment in filtered_data.columns:
                    # Count users by segment
                    segment_counts = filtered_data.groupby(['user_id', selected_segment]).size().reset_index(name='count')
                    segment_counts = segment_counts.groupby(selected_segment)['user_id'].nunique().reset_index()
                    segment_counts.columns = [selected_segment, 'user_count']
                    
                    # Create bar chart
                    fig = px.bar(
                        segment_counts,
                        x=selected_segment,
                        y='user_count',
                        title=f'User Count by {selected_segment}',
                        labels={
                            selected_segment: selected_segment.replace('_', ' ').title(),
                            'user_count': 'Number of Users'
                        },
                        color='user_count',
                        color_continuous_scale='Viridis'
                    )
                    
                    fig.update_layout(height=400, coloraxis_showscale=False)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # If revenue data is available, show revenue by segment
                    revenue_cols = [col for col in filtered_data.columns if 'revenue' in col.lower()]
                    if revenue_cols:
                        # Aggregate revenue by segment
                        segment_revenue = filtered_data.groupby(selected_segment)[revenue_cols].sum().reset_index()
                        
                        # If there are multiple revenue columns, add a total
                        if len(revenue_cols) > 1 and 'total_revenue' not in segment_revenue.columns:
                            segment_revenue['total_revenue'] = segment_revenue[revenue_cols].sum(axis=1)
                            revenue_col = 'total_revenue'
                        else:
                            revenue_col = revenue_cols[0]
                        
                        # Calculate ARPU by segment
                        segment_arpu = segment_revenue.copy()
                        segment_arpu = pd.merge(segment_arpu, segment_counts, on=selected_segment)
                        segment_arpu['arpu'] = segment_arpu[revenue_col] / segment_arpu['user_count']
                        
                        # Create bar chart
                        fig = px.bar(
                            segment_arpu,
                            x=selected_segment,
                            y='arpu',
                            title=f'Average Revenue Per User by {selected_segment}',
                            labels={
                                selected_segment: selected_segment.replace('_', ' ').title(),
                                'arpu': 'ARPU ($)'
                            },
                            color='arpu',
                            color_continuous_scale='Viridis'
                        )
                        
                        fig.update_layout(height=400, coloraxis_showscale=False)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Insights based on segment analysis
                        top_segment_users = segment_counts.loc[segment_counts['user_count'].idxmax(), selected_segment]
                        top_segment_arpu = segment_arpu.loc[segment_arpu['arpu'].idxmax(), selected_segment]
                        
                        st.markdown(f"""
                        <div class="insight-box">
                        <strong>{selected_segment.replace('_', ' ').title()} Insights:</strong>
                        <ul>
                            <li>Most users are on <strong>{top_segment_users}</strong>.</li>
                            <li>Highest ARPU is from <strong>{top_segment_arpu}</strong> users.</li>
                            <li>{'Consider optimizing the experience and monetization strategies for these key segments.' if top_segment_users == top_segment_arpu else 'There is an opportunity to increase monetization for the most popular segment.'}</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning(f"Selected segment '{selected_segment}' not found in data.")
            else:
                st.warning("Device type and game mode data is not available.")
    else:
        st.warning("User segments data is not available. Please run the data processing scripts first.") 