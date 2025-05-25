import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def display_revenue_analysis(data, filtered_data):
    """
    Display revenue analysis page
    
    Parameters:
    -----------
    data : dict
        Dictionary containing all loaded DataFrames
    filtered_data : pandas.DataFrame
        Filtered raw data based on date selection
    """
    st.markdown('<div class="sub-header">Revenue Analysis</div>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    This section analyzes revenue patterns over time, by user segment, and by other dimensions
    to identify opportunities for growth and optimization.
    """)
    
    # Check if we have the necessary data
    if data['revenue'] is not None:
        # Revenue KPIs
        # Identify revenue columns
        revenue_cols = [col for col in data['revenue'].columns if 'revenue' in col.lower() and col != 'total_revenue']
        
        if revenue_cols:
            # Calculate metrics
            total_revenue = data['revenue'][revenue_cols].sum().sum()
            avg_daily_revenue = total_revenue / data['revenue']['date'].nunique() if data['revenue']['date'].nunique() > 0 else 0
            
            # If we have user data, calculate ARPU
            if filtered_data is not None and 'user_id' in filtered_data.columns:
                total_users = filtered_data['user_id'].nunique()
                arpu = total_revenue / total_users if total_users > 0 else 0
            else:
                arpu = 0
            
            # Create KPI metrics display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-value">${total_revenue:,.2f}</div>
                    <div class="kpi-label">Total Revenue</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-value">${avg_daily_revenue:,.2f}</div>
                    <div class="kpi-label">Average Daily Revenue</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-value">${arpu:,.2f}</div>
                    <div class="kpi-label">Average Revenue Per User</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Revenue Trends", "Revenue by Segment", "Revenue Distribution"])
            
            with tab1:
                # Plot revenue trends over time
                if 'revenue_analysis' in data and data['revenue_analysis'] is not None:
                    rev_data = data['revenue_analysis'].copy()
                    
                    # Check if required columns exist
                    required_cols = ['date', 'daily_revenue', 'revenue_7d_avg']
                    missing_cols = [col for col in required_cols if col not in rev_data.columns]
                    
                    if not missing_cols:
                        # Convert date to datetime if it's not already
                        if not pd.api.types.is_datetime64_any_dtype(rev_data['date']):
                            rev_data['date'] = pd.to_datetime(rev_data['date'])
                        
                        # Create a subplot with a shared x-axis
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        # Add bar chart for daily revenue
                        fig.add_trace(
                            go.Bar(
                                x=rev_data['date'],
                                y=rev_data['daily_revenue'],
                                name='Daily Revenue',
                                marker_color='royalblue'
                            ),
                            secondary_y=False
                        )
                        
                        # Add line chart for 7-day rolling average
                        fig.add_trace(
                            go.Scatter(
                                x=rev_data['date'],
                                y=rev_data['revenue_7d_avg'],
                                name='7-day Rolling Average',
                                line=dict(color='red', width=2)
                            ),
                            secondary_y=False
                        )
                        
                        # Set axis titles
                        fig.update_xaxes(title_text='Date')
                        fig.update_yaxes(title_text='Revenue ($)', secondary_y=False)
                        
                        # Set layout
                        fig.update_layout(
                            title='Revenue Trends Over Time',
                            hovermode='x unified',
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate growth metrics
                        if len(rev_data) >= 14:  # Only if we have at least 2 weeks of data
                            # Calculate first and last week average revenue
                            first_week = rev_data.head(7)['daily_revenue'].mean()
                            last_week = rev_data.tail(7)['daily_revenue'].mean()
                            
                            # Calculate growth percentage
                            growth_pct = ((last_week - first_week) / first_week) * 100 if first_week > 0 else 0
                            
                            st.markdown(f"""
                            <div class="insight-box">
                            <strong>Revenue Growth Insights:</strong>
                            <ul>
                                <li>First week average revenue: <strong>${first_week:.2f}</strong></li>
                                <li>Last week average revenue: <strong>${last_week:.2f}</strong></li>
                                <li>Overall growth: <strong>{growth_pct:.1f}%</strong> {'(increasing ↑)' if growth_pct > 0 else '(decreasing ↓)' if growth_pct < 0 else '(stable →)'}</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning(f"Required columns missing for revenue trends: {', '.join(missing_cols)}")
                else:
                    # If revenue_analysis data is not available, check if we can create a simple chart from filtered_data
                    if filtered_data is not None and 'revenue' in filtered_data.columns:
                        st.info("Revenue analysis data is not available in preprocessed format. Creating a basic chart from raw data.")
                        
                        # Aggregate daily revenue
                        daily_rev = filtered_data.groupby('date')['revenue'].sum().reset_index()
                        
                        # Create a simple bar chart
                        fig = px.bar(
                            daily_rev,
                            x='date',
                            y='revenue',
                            title='Daily Revenue',
                            labels={
                                'date': 'Date',
                                'revenue': 'Revenue ($)'
                            }
                        )
                        
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate simple growth metrics
                        if len(daily_rev) >= 14:  # Only if we have at least 2 weeks of data
                            # Calculate first and last week average revenue
                            first_week = daily_rev.head(7)['revenue'].mean()
                            last_week = daily_rev.tail(7)['revenue'].mean()
                            
                            # Calculate growth percentage
                            growth_pct = ((last_week - first_week) / first_week) * 100 if first_week > 0 else 0
                            
                            st.markdown(f"""
                            <div class="insight-box">
                            <strong>Revenue Growth Insights:</strong>
                            <ul>
                                <li>First week average revenue: <strong>${first_week:.2f}</strong></li>
                                <li>Last week average revenue: <strong>${last_week:.2f}</strong></li>
                                <li>Overall growth: <strong>{growth_pct:.1f}%</strong> {'(increasing ↑)' if growth_pct > 0 else '(decreasing ↓)' if growth_pct < 0 else '(stable →)'}</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("Revenue analysis data is not available.")
            
            with tab2:
                # Check if we have user segment data
                if filtered_data is not None and 'user_id' in filtered_data.columns:
                    # Define segments to analyze (device_type, game_mode, etc.)
                    segment_options = [col for col in filtered_data.columns if col not in ['user_id', 'date']]
                    
                    if segment_options:
                        # Let user select segment
                        selected_segment = st.selectbox("Select Segment", segment_options)
                        
                        if selected_segment in filtered_data.columns:
                            # Check if any revenue column exists in filtered_data
                            filtered_revenue_cols = []
                            
                            # First check for specific revenue columns
                            if revenue_cols:
                                filtered_revenue_cols = [col for col in revenue_cols if col in filtered_data.columns]
                            
                            # If none found, check for generic 'revenue' column
                            if not filtered_revenue_cols and 'revenue' in filtered_data.columns:
                                filtered_revenue_cols = ['revenue']
                            
                            if filtered_revenue_cols:
                                # Aggregate revenue by segment
                                segment_revenue = filtered_data.groupby(selected_segment)[filtered_revenue_cols].sum().reset_index()
                                
                                # If there are multiple revenue columns, add a total
                                if len(filtered_revenue_cols) > 1 and 'total_revenue' not in segment_revenue.columns:
                                    segment_revenue['total_revenue'] = segment_revenue[filtered_revenue_cols].sum(axis=1)
                                
                                # Plot the data
                                fig = px.bar(
                                    segment_revenue,
                                    x=selected_segment,
                                    y='total_revenue' if 'total_revenue' in segment_revenue.columns else filtered_revenue_cols[0],
                                    title=f'Revenue by {selected_segment}',
                                    labels={
                                        selected_segment: selected_segment.replace('_', ' ').title(),
                                        'total_revenue': 'Total Revenue ($)',
                                        filtered_revenue_cols[0]: 'Revenue ($)'
                                    },
                                    color='total_revenue' if 'total_revenue' in segment_revenue.columns else filtered_revenue_cols[0],
                                    color_continuous_scale='Viridis'
                                )
                                
                                fig.update_layout(height=500, coloraxis_showscale=False)
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Identify top and bottom segments
                                if not segment_revenue.empty:
                                    y_col = 'total_revenue' if 'total_revenue' in segment_revenue.columns else filtered_revenue_cols[0]
                                    top_segment = segment_revenue.loc[segment_revenue[y_col].idxmax(), selected_segment]
                                    bottom_segment = segment_revenue.loc[segment_revenue[y_col].idxmin(), selected_segment]
                                    
                                    st.markdown(f"""
                                    <div class="insight-box">
                                    <strong>Segment Analysis:</strong>
                                    <ul>
                                        <li>Highest revenue segment: <strong>{top_segment}</strong></li>
                                        <li>Lowest revenue segment: <strong>{bottom_segment}</strong></li>
                                        <li>Understanding these differences can help optimize monetization strategies for specific segments.</li>
                                    </ul>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.warning(f"No revenue columns found in filtered data. Available columns: {', '.join(filtered_data.columns)}")
                        else:
                            st.warning(f"Selected segment '{selected_segment}' not found in data.")
                    else:
                        st.warning("No segment columns found in data.")
                else:
                    st.warning("User segment data is not available.")
            
            with tab3:
                # Revenue distribution analysis (Pareto principle check)
                if filtered_data is not None and 'user_id' in filtered_data.columns:
                    # Check if any revenue column exists in filtered_data
                    filtered_revenue_cols = []
                    
                    # First check for specific revenue columns
                    if revenue_cols:
                        filtered_revenue_cols = [col for col in revenue_cols if col in filtered_data.columns]
                    
                    # If none found, check for generic 'revenue' column
                    if not filtered_revenue_cols and 'revenue' in filtered_data.columns:
                        filtered_revenue_cols = ['revenue']
                    
                    if filtered_revenue_cols:
                        # Aggregate revenue by user
                        user_revenue = filtered_data.groupby('user_id')[filtered_revenue_cols].sum().reset_index()
                        
                        # If there are multiple revenue columns, add a total
                        if len(filtered_revenue_cols) > 1 and 'total_revenue' not in user_revenue.columns:
                            user_revenue['total_revenue'] = user_revenue[filtered_revenue_cols].sum(axis=1)
                        
                        # Sort by revenue
                        revenue_col = 'total_revenue' if 'total_revenue' in user_revenue.columns else filtered_revenue_cols[0]
                        user_revenue = user_revenue.sort_values(by=revenue_col, ascending=False)
                        
                        # Calculate cumulative percentage
                        total = user_revenue[revenue_col].sum()
                        user_revenue['cum_percentage'] = user_revenue[revenue_col].cumsum() / total * 100 if total > 0 else 0
                        
                        # Calculate percentile
                        user_revenue['percentile'] = np.arange(1, len(user_revenue) + 1) / len(user_revenue) * 100
                        
                        # Create plot
                        fig = go.Figure()
                        
                        fig.add_trace(
                            go.Scatter(
                                x=user_revenue['percentile'],
                                y=user_revenue['cum_percentage'],
                                name='Revenue Distribution',
                                line=dict(color='blue', width=2)
                            )
                        )
                        
                        # Add reference line for perfect equality
                        fig.add_trace(
                            go.Scatter(
                                x=[0, 100],
                                y=[0, 100],
                                name='Equal Distribution',
                                line=dict(color='red', width=2, dash='dash')
                            )
                        )
                        
                        # Find the percentage of users that generate 80% of revenue (Pareto)
                        users_for_80pct = user_revenue[user_revenue['cum_percentage'] <= 80].shape[0]
                        pct_users_for_80pct = (users_for_80pct / len(user_revenue)) * 100 if len(user_revenue) > 0 else 0
                        
                        # Add reference lines for 80/20 rule
                        fig.add_shape(
                            type="line",
                            x0=0, y0=80, x1=pct_users_for_80pct, y1=80,
                            line=dict(color="green", width=2, dash="dot")
                        )
                        
                        fig.add_shape(
                            type="line",
                            x0=pct_users_for_80pct, y0=0, x1=pct_users_for_80pct, y1=80,
                            line=dict(color="green", width=2, dash="dot")
                        )
                        
                        fig.update_layout(
                            title='Revenue Distribution (Lorenz Curve)',
                            xaxis_title='Percentage of Users (%)',
                            yaxis_title='Cumulative Percentage of Revenue (%)',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown(f"""
                        <div class="insight-box">
                        <strong>Revenue Distribution Analysis:</strong>
                        <ul>
                            <li>Top <strong>{pct_users_for_80pct:.1f}%</strong> of users generate 80% of revenue.</li>
                            <li>This indicates {'a high level of' if pct_users_for_80pct < 20 else 'moderate'} revenue concentration.</li>
                            <li>{'Consider strategies to increase monetization from the wider user base.' if pct_users_for_80pct < 20 else 'Revenue is relatively well distributed across the user base.'}</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Top spenders analysis
                        top_n = min(10, len(user_revenue))
                        st.markdown(f"<div class='sub-header'>Top {top_n} Spenders</div>", unsafe_allow_html=True)
                        
                        # Format revenue column to display as currency
                        formatted_revenue = user_revenue.head(top_n).copy()
                        formatted_revenue[revenue_col] = formatted_revenue[revenue_col].apply(lambda x: f"${x:,.2f}")
                        
                        # Display table
                        st.dataframe(
                            formatted_revenue[['user_id', revenue_col]].rename(
                                columns={revenue_col: 'Total Revenue'}
                            ),
                            use_container_width=True
                        )
                    else:
                        st.warning(f"No revenue columns found in filtered data. Available columns: {', '.join(filtered_data.columns)}")
                else:
                    st.warning("User revenue data is not available.")
        else:
            st.warning("No revenue columns found in data.")
    else:
        st.warning("Revenue data is not available. Please run the data processing scripts first.") 