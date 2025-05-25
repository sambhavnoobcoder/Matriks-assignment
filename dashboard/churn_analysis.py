import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def display_churn_analysis(data, filtered_data):
    """
    Display churn analysis page
    
    Parameters:
    -----------
    data : dict
        Dictionary containing all loaded DataFrames
    filtered_data : pandas.DataFrame
        Filtered raw data based on date selection
    """
    st.markdown('<div class="sub-header">Churn Analysis</div>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    This section analyzes user churn patterns to identify at-risk users and opportunities
    for improving retention.
    """)
    
    # Check if we have the necessary data
    if data['churn_analysis'] is not None and 'churned' in data['churn_analysis'].columns:
        # Calculate churn rate
        churn_count = data['churn_analysis']['churned'].sum()
        total_users = len(data['churn_analysis'])
        churn_rate = (churn_count / total_users) * 100 if total_users > 0 else 0
        
        # Display churn KPIs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-value">{churn_rate:.1f}%</div>
                <div class="kpi-label">Churn Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-value">{churn_count:,}</div>
                <div class="kpi-label">Churned Users</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            active_users = total_users - churn_count
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-value">{active_users:,}</div>
                <div class="kpi-label">Active Users</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Churn Overview", "Inactivity Analysis", "At-Risk Users"])
        
        with tab1:
            # Create churn distribution pie chart
            churn_dist = data['churn_analysis']['churned'].value_counts().reset_index()
            churn_dist.columns = ['Status', 'Count']
            churn_dist['Status'] = churn_dist['Status'].map({True: 'Churned', False: 'Active'})
            
            fig = px.pie(
                churn_dist,
                values='Count',
                names='Status',
                title='User Status Distribution',
                color='Status',
                color_discrete_map={
                    'Active': 'green',
                    'Churned': 'red'
                }
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # If we have user_segments data, analyze churn by segment
            if data['user_segments'] is not None and 'user_id' in data['user_segments'].columns:
                # Merge churn data with segments
                segments_with_churn = pd.merge(
                    data['churn_analysis'],
                    data['user_segments'],
                    on='user_id',
                    how='inner'
                )
                
                # Check if activity_segment exists
                if 'activity_segment' in segments_with_churn.columns:
                    # Calculate churn rate by activity segment
                    churn_by_activity = segments_with_churn.groupby('activity_segment')['churned'].mean().reset_index()
                    churn_by_activity['churn_rate'] = churn_by_activity['churned'] * 100
                    
                    # Ensure proper ordering
                    segment_order = ['Low', 'Medium', 'High', 'Very High']
                    churn_by_activity['activity_segment'] = pd.Categorical(
                        churn_by_activity['activity_segment'], 
                        categories=segment_order, 
                        ordered=True
                    )
                    churn_by_activity = churn_by_activity.sort_values('activity_segment')
                    
                    # Create bar chart
                    fig = px.bar(
                        churn_by_activity,
                        x='activity_segment',
                        y='churn_rate',
                        title='Churn Rate by Activity Segment',
                        labels={
                            'activity_segment': 'Activity Segment',
                            'churn_rate': 'Churn Rate (%)'
                        },
                        color='churn_rate',
                        color_continuous_scale='Reds_r'
                    )
                    
                    fig.update_layout(height=400, coloraxis_showscale=False)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Insights based on churn by activity segment
                    highest_churn_segment = churn_by_activity.loc[churn_by_activity['churn_rate'].idxmax(), 'activity_segment']
                    highest_churn_rate = churn_by_activity['churn_rate'].max()
                    
                    lowest_churn_segment = churn_by_activity.loc[churn_by_activity['churn_rate'].idxmin(), 'activity_segment']
                    lowest_churn_rate = churn_by_activity['churn_rate'].min()
                    
                    st.markdown(f"""
                    <div class="insight-box">
                    <strong>Churn by Activity Segment:</strong>
                    <ul>
                        <li>Highest churn rate: <strong>{highest_churn_segment}</strong> segment ({highest_churn_rate:.1f}%)</li>
                        <li>Lowest churn rate: <strong>{lowest_churn_segment}</strong> segment ({lowest_churn_rate:.1f}%)</li>
                        <li>{'Low activity users are most likely to churn, suggesting improved onboarding or early engagement could help.' if highest_churn_segment == 'Low' else 'Consider investigating why even highly active users are churning.'}</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            # Analyze days since last activity
            days_since_last = data['churn_analysis'].copy()
            
            # Check if the required column exists
            activity_col = None
            if 'days_since_last_activity' in days_since_last.columns:
                activity_col = 'days_since_last_activity'
            elif 'days_since_activity' in days_since_last.columns:
                activity_col = 'days_since_activity'
            
            if activity_col:
                # Create histogram of days since last activity
                fig = px.histogram(
                    days_since_last,
                    x=activity_col,
                    title='Distribution of Days Since Last Activity',
                    labels={
                        activity_col: 'Days Since Last Activity',
                        'count': 'Number of Users'
                    },
                    color='churned',
                    color_discrete_map={
                        True: 'red',
                        False: 'green'
                    }
                )
                
                fig.update_layout(height=400)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create user inactivity segments
                inactivity_bins = [0, 7, 14, 30, 60, 90, float('inf')]
                inactivity_labels = ['1-7 days', '8-14 days', '15-30 days', '31-60 days', '61-90 days', '90+ days']
                
                days_since_last['inactivity_segment'] = pd.cut(
                    days_since_last[activity_col],
                    bins=inactivity_bins,
                    labels=inactivity_labels,
                    right=False
                )
                
                # Count users by inactivity segment
                inactivity_counts = days_since_last.groupby('inactivity_segment').size().reset_index(name='user_count')
                
                # Create bar chart
                fig = px.bar(
                    inactivity_counts,
                    x='inactivity_segment',
                    y='user_count',
                    title='Users by Inactivity Period',
                    labels={
                        'inactivity_segment': 'Inactivity Period',
                        'user_count': 'Number of Users'
                    },
                    color='user_count',
                    color_continuous_scale='Reds'
                )
                
                fig.update_layout(height=400, coloraxis_showscale=False)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate average days since last activity
                avg_days = days_since_last[activity_col].mean()
                
                # Calculate percentage of users inactive for more than 30 days
                inactive_30_plus = days_since_last[days_since_last[activity_col] > 30].shape[0]
                inactive_30_plus_pct = (inactive_30_plus / len(days_since_last)) * 100 if len(days_since_last) > 0 else 0
                
                st.markdown(f"""
                <div class="insight-box">
                <strong>Inactivity Analysis:</strong>
                <ul>
                    <li>Average days since last activity: <strong>{avg_days:.1f} days</strong></li>
                    <li><strong>{inactive_30_plus_pct:.1f}%</strong> of users have been inactive for more than 30 days</li>
                    <li>Consider implementing re-engagement campaigns targeting users who haven't been active for 15-30 days to prevent churn.</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Required column for inactivity analysis not found. Check if 'days_since_activity' or 'days_since_last_activity' exists in your data.")
        
        with tab3:
            # Create a section for at-risk users
            st.markdown("### At-Risk User Analysis")
            
            # Define at-risk threshold
            risk_threshold = st.slider(
                "Inactivity Threshold (Days)",
                min_value=7,
                max_value=60,
                value=20,
                help="Define how many days of inactivity to consider a user at-risk of churning"
            )
            
            # Check if the required column exists
            activity_col = None
            if 'days_since_last_activity' in data['churn_analysis'].columns:
                activity_col = 'days_since_last_activity'
            elif 'days_since_activity' in data['churn_analysis'].columns:
                activity_col = 'days_since_activity'
            
            if activity_col:
                # Identify at-risk users (inactive but not yet churned)
                at_risk = data['churn_analysis'][
                    (data['churn_analysis'][activity_col] >= risk_threshold) &
                    (~data['churn_analysis']['churned'])
                ]
                
                # Count at-risk users
                at_risk_count = len(at_risk)
                at_risk_pct = (at_risk_count / total_users) * 100 if total_users > 0 else 0
                
                st.markdown(f"""
                <div class="insight-box">
                <strong>At-Risk Users:</strong>
                <ul>
                    <li><strong>{at_risk_count:,}</strong> users ({at_risk_pct:.1f}% of total) are at risk of churning</li>
                    <li>These users have been inactive for {risk_threshold}+ days but haven't reached the churn threshold yet</li>
                    <li>This group represents your best opportunity for targeted retention campaigns</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # If we have more user data, provide deeper analysis
                if filtered_data is not None and 'user_id' in filtered_data.columns:
                    # Segment columns to analyze
                    segment_columns = [col for col in filtered_data.columns if col not in ['user_id', 'date']]
                    
                    if segment_columns:
                        # Let user select which segment to analyze
                        selected_segment = st.selectbox("Analyze At-Risk Users by", segment_columns)
                        
                        if selected_segment in filtered_data.columns:
                            # Get segment distribution for at-risk users
                            at_risk_users = at_risk['user_id'].tolist()
                            
                            if at_risk_users:
                                at_risk_segments = filtered_data[filtered_data['user_id'].isin(at_risk_users)]
                                segment_dist = at_risk_segments.groupby(selected_segment).size().reset_index(name='user_count')
                                
                                # Create bar chart
                                fig = px.bar(
                                    segment_dist,
                                    x=selected_segment,
                                    y='user_count',
                                    title=f'At-Risk Users by {selected_segment}',
                                    labels={
                                        selected_segment: selected_segment.replace('_', ' ').title(),
                                        'user_count': 'Number of Users'
                                    },
                                    color='user_count',
                                    color_continuous_scale='Oranges'
                                )
                                
                                fig.update_layout(height=400, coloraxis_showscale=False)
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Identify the segment with the most at-risk users
                                top_at_risk_segment = segment_dist.loc[segment_dist['user_count'].idxmax(), selected_segment]
                                top_segment_count = segment_dist['user_count'].max()
                                top_segment_pct = (top_segment_count / at_risk_count) * 100 if at_risk_count > 0 else 0
                                
                                st.markdown(f"""
                                <div class="insight-box">
                                <strong>At-Risk User Segments:</strong>
                                <ul>
                                    <li>The segment with the most at-risk users is <strong>{top_at_risk_segment}</strong> ({top_segment_pct:.1f}% of at-risk users)</li>
                                    <li>Consider creating a targeted re-engagement campaign specifically for {top_at_risk_segment} users</li>
                                </ul>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.info("No at-risk users found with the current threshold.")
                
                # Display a sample of at-risk users
                if not at_risk.empty:
                    st.markdown("### Sample of At-Risk Users")
                    st.dataframe(at_risk.head(10), use_container_width=True)
            else:
                st.warning("Required column for at-risk analysis not found. Check if 'days_since_activity' or 'days_since_last_activity' exists in your data.")
    else:
        st.warning("Churn analysis data is not available. Please run the data processing scripts first.") 