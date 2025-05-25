import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

def display_user_metrics(data, filtered_data):
    """
    Display user metrics analysis page
    
    Parameters:
    -----------
    data : dict
        Dictionary containing all data DataFrames
    filtered_data : pandas.DataFrame
        Filtered raw data based on date range selection
    """
    st.markdown('<div class="sub-header">User Metrics Analysis</div>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    This section analyzes user activity metrics including Daily Active Users (DAU), 
    Weekly Active Users (WAU), and Monthly Active Users (MAU) to understand user engagement patterns.
    """)
    
    # Check if we have the necessary data
    if data['dau'] is None or data['wau'] is None or data['mau'] is None:
        st.warning("User metrics data is not available. Please run data processing first.")
        return
    
    # Create tabs
    tabs = st.tabs(["DAU/WAU/MAU Trends", "Stickiness", "Growth Metrics"])
    
    # DAU/WAU/MAU Trends tab
    with tabs[0]:
        st.subheader("Daily, Weekly, and Monthly Active Users")
        
        # Create a date range selector
        if 'date' in data['dau'].columns:
            min_date = data['dau']['date'].min()
            max_date = data['dau']['date'].max()
            
            # Convert to datetime if needed
            if not isinstance(min_date, datetime):
                min_date = pd.to_datetime(min_date)
            if not isinstance(max_date, datetime):
                max_date = pd.to_datetime(max_date)
            
            # Create date range selector
            date_range = st.date_input(
                "Select Date Range for Trends",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date()
            )
            
            # Filter data based on date range
            if len(date_range) == 2:
                start_date, end_date = date_range
                dau_filtered = data['dau'][(data['dau']['date'] >= pd.Timestamp(start_date)) & 
                                        (data['dau']['date'] <= pd.Timestamp(end_date))]
                wau_filtered = data['wau'][(data['wau']['date'] >= pd.Timestamp(start_date)) & 
                                        (data['wau']['date'] <= pd.Timestamp(end_date))]
                mau_filtered = data['mau'][(data['mau']['date'] >= pd.Timestamp(start_date)) & 
                                        (data['mau']['date'] <= pd.Timestamp(end_date))]
            else:
                dau_filtered = data['dau']
                wau_filtered = data['wau']
                mau_filtered = data['mau']
            
            # Create multi-line chart
            fig = go.Figure()
            
            # Add DAU line
            fig.add_trace(go.Scatter(
                x=dau_filtered['date'],
                y=dau_filtered['dau'],
                mode='lines',
                name='DAU',
                line=dict(color='blue')
            ))
            
            # Add WAU line
            fig.add_trace(go.Scatter(
                x=wau_filtered['date'],
                y=wau_filtered['wau'],
                mode='lines',
                name='WAU',
                line=dict(color='green')
            ))
            
            # Add MAU line
            fig.add_trace(go.Scatter(
                x=mau_filtered['date'],
                y=mau_filtered['mau'],
                mode='lines',
                name='MAU',
                line=dict(color='red')
            ))
            
            # Update layout
            fig.update_layout(
                title='User Activity Trends',
                xaxis_title='Date',
                yaxis_title='Active Users',
                legend_title='Metric',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate metrics
            if not dau_filtered.empty and not wau_filtered.empty and not mau_filtered.empty:
                avg_dau = dau_filtered['dau'].mean()
                avg_wau = wau_filtered['wau'].mean()
                avg_mau = mau_filtered['mau'].mean()
                
                # Display metrics in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average DAU", f"{avg_dau:.0f}")
                
                with col2:
                    st.metric("Average WAU", f"{avg_wau:.0f}")
                
                with col3:
                    st.metric("Average MAU", f"{avg_mau:.0f}")
                
                # Calculate growth metrics
                if len(dau_filtered) > 1:
                    first_dau = dau_filtered.iloc[0]['dau']
                    last_dau = dau_filtered.iloc[-1]['dau']
                    dau_growth = ((last_dau - first_dau) / first_dau) * 100 if first_dau > 0 else 0
                    
                    st.markdown(f"""
                    <div class="insight-box">
                        <b>DAU Growth:</b> {dau_growth:.1f}% over the selected period
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Date column not found in user metrics data.")
    
    # Stickiness tab
    with tabs[1]:
        st.subheader("User Stickiness")
        
        if 'date' in data['dau'].columns and 'date' in data['wau'].columns and 'date' in data['mau'].columns:
            # Calculate stickiness (DAU/MAU ratio)
            stickiness_data = pd.merge(
                data['dau'][['date', 'dau']], 
                data['mau'][['date', 'mau']], 
                on='date', 
                how='inner'
            )
            
            stickiness_data['stickiness'] = stickiness_data['dau'] / stickiness_data['mau'] * 100
            
            # Create stickiness chart
            fig = px.line(
                stickiness_data, 
                x='date', 
                y='stickiness',
                title='User Stickiness (DAU/MAU Ratio)',
                labels={'stickiness': 'Stickiness (%)', 'date': 'Date'}
            )
            
            # Add a reference line for good stickiness (20%)
            fig.add_hline(
                y=20, 
                line_dash="dash", 
                line_color="green",
                annotation_text="Good Stickiness (20%)",
                annotation_position="bottom right"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate average stickiness
            avg_stickiness = stickiness_data['stickiness'].mean()
            
            st.markdown(f"""
            <div class="insight-box">
                <b>Average Stickiness:</b> {avg_stickiness:.1f}%
                <br><br>
                <b>Interpretation:</b>
                <ul>
                    <li>Stickiness measures how frequently your users engage with your app.</li>
                    <li>A higher percentage means users return more often.</li>
                    <li>Industry benchmark for good stickiness is around 20%.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Date column not found in user metrics data.")
    
    # Growth Metrics tab
    with tabs[2]:
        st.subheader("User Growth Metrics")
        
        if 'date' in data['dau'].columns:
            # Calculate week-over-week and month-over-month growth
            dau_data = data['dau'].copy()
            
            # Convert date to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(dau_data['date']):
                dau_data['date'] = pd.to_datetime(dau_data['date'])
            
            # Add week and month columns
            dau_data['week'] = dau_data['date'].dt.isocalendar().week
            dau_data['month'] = dau_data['date'].dt.month
            dau_data['year'] = dau_data['date'].dt.year
            
            # Calculate weekly averages
            weekly_avg = dau_data.groupby(['year', 'week'])['dau'].mean().reset_index()
            weekly_avg['year_week'] = weekly_avg['year'].astype(str) + '-W' + weekly_avg['week'].astype(str)
            
            # Calculate monthly averages
            monthly_avg = dau_data.groupby(['year', 'month'])['dau'].mean().reset_index()
            monthly_avg['year_month'] = monthly_avg['year'].astype(str) + '-' + monthly_avg['month'].astype(str)
            
            # Create two columns for charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Weekly growth chart
                fig_weekly = px.bar(
                    weekly_avg, 
                    x='year_week', 
                    y='dau',
                    title='Weekly Average DAU',
                    labels={'dau': 'Average DAU', 'year_week': 'Week'}
                )
                
                st.plotly_chart(fig_weekly, use_container_width=True)
            
            with col2:
                # Monthly growth chart
                fig_monthly = px.bar(
                    monthly_avg, 
                    x='year_month', 
                    y='dau',
                    title='Monthly Average DAU',
                    labels={'dau': 'Average DAU', 'year_month': 'Month'}
                )
                
                st.plotly_chart(fig_monthly, use_container_width=True)
            
            # Calculate growth rates
            if len(weekly_avg) > 1:
                first_week_avg = weekly_avg.iloc[0]['dau']
                last_week_avg = weekly_avg.iloc[-1]['dau']
                wow_growth = ((last_week_avg - first_week_avg) / first_week_avg) * 100 if first_week_avg > 0 else 0
                
                st.markdown(f"""
                <div class="insight-box">
                    <b>Week-over-Week Growth:</b> {wow_growth:.1f}% from first to last week
                </div>
                """, unsafe_allow_html=True)
            
            if len(monthly_avg) > 1:
                first_month_avg = monthly_avg.iloc[0]['dau']
                last_month_avg = monthly_avg.iloc[-1]['dau']
                mom_growth = ((last_month_avg - first_month_avg) / first_month_avg) * 100 if first_month_avg > 0 else 0
                
                st.markdown(f"""
                <div class="insight-box">
                    <b>Month-over-Month Growth:</b> {mom_growth:.1f}% from first to last month
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Date column not found in user metrics data.") 