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

def display_cohort_analysis(data, filtered_data):
    """
    Display cohort analysis page
    
    Parameters:
    -----------
    data : dict
        Dictionary containing all loaded DataFrames
    filtered_data : pandas.DataFrame
        Filtered raw data based on date selection
    """
    st.markdown('<div class="sub-header">Cohort Analysis</div>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    This section analyzes user retention patterns based on when they first joined (cohort),
    helping to understand how well users are retained over time.
    """)
    
    # Check if we have the necessary data
    if data['cohort_retention'] is not None:
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Retention Heatmap", "Retention Curves"])
        
        with tab1:
            # Display the cohort retention heatmap
            # Convert the dataframe to a format suitable for heatmap
            retention_data = data['cohort_retention'].copy()
            
            # Check if the data contains non-numeric values and try to convert them
            for col in retention_data.columns:
                try:
                    retention_data[col] = pd.to_numeric(retention_data[col], errors='coerce')
                except:
                    st.warning(f"Column {col} contains non-numeric values that couldn't be converted")
            
            # Check if the index is already a datetime
            if not isinstance(retention_data.index, pd.DatetimeIndex):
                # Try to convert the index to datetime if it's not already
                try:
                    retention_data.index = pd.to_datetime(retention_data.index)
                except:
                    pass
            
            # Format the index to show as month-year
            if isinstance(retention_data.index, pd.DatetimeIndex):
                cohort_labels = retention_data.index.strftime('%b %Y')
            else:
                cohort_labels = retention_data.index
            
            # Get the column values (month offsets)
            month_offsets = retention_data.columns.astype(str)
            
            # Create the heatmap data
            heatmap_data = []
            for i, cohort in enumerate(cohort_labels):
                for j, offset in enumerate(month_offsets):
                    if j < len(retention_data.columns) and i < len(retention_data):
                        value = retention_data.iloc[i, j]
                        # Use pd.isna() instead of np.isnan() to handle all types of missing values
                        if not pd.isna(value):  # Skip NaN values
                            heatmap_data.append({
                                'Cohort': cohort,
                                'Month Offset': f"Month {offset}",
                                'Retention Rate (%)': value
                            })
            
            # Convert to DataFrame for plotting
            heatmap_df = pd.DataFrame(heatmap_data)
            
            # Create the heatmap
            if not heatmap_df.empty:
                fig = px.imshow(
                    retention_data.values,
                    labels=dict(x="Month Offset", y="Cohort", color="Retention Rate (%)"),
                    x=month_offsets,
                    y=cohort_labels,
                    color_continuous_scale='YlGnBu',
                    aspect="auto",
                    title="Cohort Retention Analysis"
                )
                
                # Add text annotations
                for i in range(len(cohort_labels)):
                    for j in range(len(month_offsets)):
                        if j < retention_data.shape[1] and i < retention_data.shape[0]:
                            value = retention_data.iloc[i, j]
                            if not pd.isna(value):
                                try:
                                    # Try to convert value to float
                                    float_value = float(value)
                                    text = f"{float_value:.1f}%"
                                except (ValueError, TypeError):
                                    # If conversion fails, just use the value as is
                                    text = f"{value}%"
                                    
                                fig.add_annotation(
                                    x=j,
                                    y=i,
                                    text=text,
                                    showarrow=False,
                                    font=dict(color="black")  # Simplify by using black for all text
                                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate average retention rates
                try:
                    avg_retention = retention_data.mean(axis=0, numeric_only=True)
                    
                    # Display insights
                    if len(avg_retention) > 1:
                        m1_idx = 1 if 1 in avg_retention.index else avg_retention.index[1]
                        m1_retention = avg_retention[m1_idx]
                        
                        st.markdown(f"""
                        <div class="insight-box">
                        <strong>Cohort Retention Insights:</strong>
                        <ul>
                            <li>Average Month 1 retention rate: <strong>{m1_retention:.1f}%</strong></li>
                            <li>Newer cohorts show {'better' if retention_data.iloc[-1, 1] > m1_retention else 'worse'} retention compared to older cohorts.</li>
                            <li>The largest drop in retention typically occurs between Month 0 and Month 1, indicating the critical period for user engagement.</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Could not calculate average retention: {e}")
            else:
                st.warning("No data available for cohort heatmap visualization.")
        
        with tab2:
            # Create retention curves
            if not retention_data.empty:
                # Average retention by month offset
                try:
                    avg_retention = retention_data.mean(axis=0, numeric_only=True)
                    
                    # Create a DataFrame for the average retention curve
                    avg_curve = pd.DataFrame({
                        'Month Offset': avg_retention.index,
                        'Retention Rate (%)': avg_retention.values,
                        'Cohort': 'Average'
                    })
                    
                    # Create individual curves for each cohort
                    cohort_curves = []
                    for i, cohort in enumerate(cohort_labels):
                        if i < len(retention_data):
                            cohort_data = retention_data.iloc[i]
                            # Convert to numeric, replace non-numeric with NaN
                            cohort_values = pd.to_numeric(cohort_data.values, errors='coerce')
                            curve_data = pd.DataFrame({
                                'Month Offset': cohort_data.index,
                                'Retention Rate (%)': cohort_values,
                                'Cohort': cohort
                            })
                            cohort_curves.append(curve_data)
                    
                    # Combine all curves
                    all_curves = pd.concat([avg_curve] + cohort_curves, ignore_index=True)
                    
                    # Create plot
                    fig = px.line(
                        all_curves,
                        x='Month Offset',
                        y='Retention Rate (%)',
                        color='Cohort',
                        title='Retention Curves by Cohort',
                        labels={'Month Offset': 'Month Offset', 'Retention Rate (%)': 'Retention Rate (%)'}
                    )
                    
                    # Highlight the average curve
                    fig.update_traces(
                        line=dict(width=1),
                        selector=dict(name='Average')
                    )
                    
                    fig.update_traces(
                        line=dict(width=3, dash='solid'),
                        selector=dict(name='Average')
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate and display cohort performance
                    # Find the best and worst performing cohorts at Month 1
                    month1_retention = {}
                    for i, cohort in enumerate(cohort_labels):
                        if i < len(retention_data) and 1 in retention_data.columns:
                            value = retention_data.iloc[i, 1]
                            try:
                                # Try to convert to float
                                float_value = float(value)
                                month1_retention[cohort] = float_value
                            except (ValueError, TypeError):
                                # Skip if value can't be converted to float
                                continue
                    
                    if month1_retention:
                        best_cohort = max(month1_retention.items(), key=lambda x: x[1])
                        worst_cohort = min(month1_retention.items(), key=lambda x: x[1])
                        
                        st.markdown(f"""
                        <div class="insight-box">
                        <strong>Cohort Performance:</strong>
                        <ul>
                            <li>Best performing cohort: <strong>{best_cohort[0]}</strong> with {best_cohort[1]:.1f}% Month 1 retention</li>
                            <li>Worst performing cohort: <strong>{worst_cohort[0]}</strong> with {worst_cohort[1]:.1f}% Month 1 retention</li>
                            <li>This suggests that {'recent improvements have been effective' if best_cohort[0] in cohort_labels[-3:] else 'there is room for improvement in recent cohorts'}.</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Could not calculate average retention: {e}")
            else:
                st.warning("No data available for retention curves visualization.")
            
            # Funnel Analysis (if available)
            if data.get('funnel_analysis') is not None:
                st.markdown('<div class="sub-header">Funnel Analysis</div>', unsafe_allow_html=True)
                
                funnel_data = data['funnel_analysis']
                
                # Map actual column names to expected column names if needed
                column_mapping = {
                    'step': 'Stage',
                    'users': 'Users',
                    'conversion_rate': 'Conversion_Rate'
                }
                
                # Create a copy with mapped column names
                funnel_plot_data = funnel_data.copy()
                for old_col, new_col in column_mapping.items():
                    if old_col in funnel_plot_data.columns and new_col not in funnel_plot_data.columns:
                        funnel_plot_data[new_col] = funnel_plot_data[old_col]
                
                # Check if required columns exist (either original or mapped)
                required_cols = ['Stage', 'Users']
                alternative_cols = ['step', 'users']
                
                # Check if either the expected columns or their alternatives exist
                missing_cols = []
                for i, col in enumerate(required_cols):
                    if col not in funnel_plot_data.columns and alternative_cols[i] not in funnel_plot_data.columns:
                        missing_cols.append(f"{col} (or {alternative_cols[i]})")
                
                if not missing_cols:
                    # Use mapped columns for visualization
                    stage_col = 'Stage' if 'Stage' in funnel_plot_data.columns else 'step'
                    users_col = 'Users' if 'Users' in funnel_plot_data.columns else 'users'
                    
                    # Create funnel visualization
                    fig = go.Figure(go.Funnel(
                        y=funnel_plot_data[stage_col],
                        x=funnel_plot_data[users_col],
                        textposition="inside",
                        textinfo="value+percent initial",
                        opacity=0.8,
                        marker={"color": ["royalblue", "darkblue", "blue", "navy"],
                                "line": {"width": [2, 2, 2, 2], "color": ["wheat", "wheat", "wheat", "wheat"]}},
                        connector={"line": {"color": "royalblue", "dash": "solid", "width": 3}}
                    ))
                    
                    fig.update_layout(
                        title="User Progression Funnel",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Map conversion and drop-off columns
                    conversion_col = 'Conversion_Rate' if 'Conversion_Rate' in funnel_plot_data.columns else 'conversion_rate'
                    
                    # Calculate drop-off rates if not available
                    if 'Drop_Off_Rate' not in funnel_plot_data.columns and 'drop_off_rate' not in funnel_plot_data.columns:
                        # Calculate drop-off rates from users numbers
                        users = funnel_plot_data[users_col].values
                        drop_off = np.zeros_like(users)
                        for i in range(1, len(users)):
                            drop_off[i] = ((users[i-1] - users[i]) / users[i-1]) * 100 if users[i-1] > 0 else 0
                        funnel_plot_data['Drop_Off_Rate'] = drop_off
                    
                    drop_off_col = 'Drop_Off_Rate' if 'Drop_Off_Rate' in funnel_plot_data.columns else 'drop_off_rate'
                    
                    # Display conversion rates if available
                    if conversion_col in funnel_plot_data.columns:
                        # Find the step with the highest drop-off
                        if drop_off_col in funnel_plot_data.columns:
                            highest_dropoff_idx = funnel_plot_data[drop_off_col].idxmax() if not funnel_plot_data[drop_off_col].isna().all() else None
                            
                            if highest_dropoff_idx is not None and highest_dropoff_idx > 0:
                                highest_dropoff_stage = funnel_plot_data.iloc[highest_dropoff_idx][stage_col]
                                highest_dropoff_rate = funnel_plot_data.iloc[highest_dropoff_idx][drop_off_col]
                                previous_stage = funnel_plot_data.iloc[highest_dropoff_idx - 1][stage_col]
                                
                                st.markdown(f"""
                                <div class="insight-box">
                                <strong>Funnel Analysis Insights:</strong>
                                <ul>
                                    <li>The highest drop-off occurs between <strong>{previous_stage}</strong> and <strong>{highest_dropoff_stage}</strong> with a {highest_dropoff_rate:.1f}% drop-off rate.</li>
                                    <li>Overall conversion rate from first to last step: <strong>{funnel_plot_data.iloc[-1][conversion_col]:.1f}%</strong></li>
                                    <li>Focus on improving the transition between {previous_stage} and {highest_dropoff_stage} to increase overall conversion.</li>
                                </ul>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.warning(f"Funnel data missing required columns: {', '.join(missing_cols)}. Available columns: {', '.join(funnel_data.columns)}")
    else:
        st.warning("Cohort analysis data is not available. Please run the data processing scripts first.") 