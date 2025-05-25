import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path to import data loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_loader import load_data

# Import page modules
from dashboard.user_metrics import display_user_metrics
from dashboard.revenue_analysis import display_revenue_analysis
from dashboard.user_segments import display_user_segments
from dashboard.churn_analysis import display_churn_analysis
from dashboard.cohort_analysis import display_cohort_analysis
from dashboard.user_clusters import display_user_clusters

# Set page configuration
st.set_page_config(
    page_title="Matiks Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    /* Dark theme - Main background */
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    
    /* Text */
    p, li, div {
        color: white !important;
    }
    
    /* Main header */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
        margin-bottom: 1rem;
    }
    
    /* Sub-header for section titles */
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #4c8bf5;
        color: white !important;
        margin-top: 2rem;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #262730;
    }
    
    /* Cards and boxes */
    .stBlock, div.stBlock > div {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    /* Insight boxes */
    .insight-box {
        background-color: #262730;
        border-left: 4px solid #4c8bf5;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
        color: white;
    }
    
    /* KPI boxes */
    .kpi-box {
        background-color: #262730;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
        text-align: center;
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
        color: #4c8bf5 !important;
    }
    
    .kpi-label {
        font-size: 1rem;
        color: #e0e0e0 !important;
    }
    
    /* Tables */
    .dataframe {
        color: white !important;
    }
    
    /* Make sure text in widgets is visible */
    .stSelectbox label, .stSlider label, .stDateInput label, .stMultiselect label {
        color: white !important;
    }
    
    /* Custom styling for KPI metrics */
    .metric-container {
        background-color: #262730;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_all_data():
    """
    Load all the necessary data files
    
    Returns:
    --------
    dict
        Dictionary containing all loaded DataFrames
    """
    data_files = {
        'raw_data': '../data/matiks_data_clean.csv',
        'dau': '../data/dau.csv',
        'wau': '../data/wau.csv',
        'mau': '../data/mau.csv',
        'revenue': '../data/revenue_analysis.csv',
        'user_segments': '../data/user_segments.csv',
        'churn_analysis': '../data/churn_analysis.csv',
        'cohort_retention': '../data/cohort_retention.csv',
        'user_clusters': '../data/user_clusters.csv',
        'cluster_analysis': '../data/cluster_analysis.csv',
        'funnel_analysis': '../data/funnel_analysis.csv'
    }
    
    data_dict = {}
    
    for key, file_path in data_files.items():
        try:
            data_dict[key] = pd.read_csv(file_path)
            
            # Convert date columns to datetime
            if key in ['raw_data', 'dau', 'wau', 'mau', 'revenue']:
                if 'date' in data_dict[key].columns:
                    data_dict[key]['date'] = pd.to_datetime(data_dict[key]['date'])
            
            if key == 'churn_analysis':
                if 'last_activity_date' in data_dict[key].columns:
                    data_dict[key]['last_activity_date'] = pd.to_datetime(data_dict[key]['last_activity_date'])
            
            if key == 'cohort_retention':
                # Convert index to datetime for cohort analysis
                if data_dict[key].index.name == 'cohort':
                    data_dict[key].index = pd.to_datetime(data_dict[key].index)
        except Exception as e:
            st.warning(f"Could not load {key} data: {e}")
            data_dict[key] = None
    
    return data_dict

def main():
    """
    Main function to run the dashboard
    """
    # Header
    st.markdown('<div class="main-header">Matiks Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Load data
    data = load_all_data()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "User Metrics", "Revenue Analysis", "User Segments", "Churn Analysis", "Cohort Analysis", "User Clusters"]
    )
    
    # Date filter in sidebar (if raw data is available)
    if data['raw_data'] is not None and 'date' in data['raw_data'].columns:
        min_date = data['raw_data']['date'].min().date()
        max_date = data['raw_data']['date'].max().date()
        
        st.sidebar.markdown("---")
        st.sidebar.title("Date Filter")
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Apply date filter to raw data
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_data = data['raw_data'][
                (data['raw_data']['date'].dt.date >= start_date) & 
                (data['raw_data']['date'].dt.date <= end_date)
            ]
        else:
            filtered_data = data['raw_data']
    else:
        filtered_data = data['raw_data']
    
    # Display appropriate page based on selection
    if page == "Overview":
        display_overview(data, filtered_data)
    elif page == "User Metrics":
        display_user_metrics(data, filtered_data)
    elif page == "Revenue Analysis":
        display_revenue_analysis(data, filtered_data)
    elif page == "User Segments":
        display_user_segments(data, filtered_data)
    elif page == "Churn Analysis":
        display_churn_analysis(data, filtered_data)
    elif page == "Cohort Analysis":
        display_cohort_analysis(data, filtered_data)
    elif page == "User Clusters":
        display_user_clusters(data, filtered_data)

# Function to display the Overview page
def display_overview(data, filtered_data):
    st.markdown('<div class="sub-header">Dashboard Overview</div>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    This dashboard provides comprehensive analytics for Matiks user behavior and revenue data.
    Use the navigation panel on the left to explore different aspects of the data.
    """)
    
    # Create overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Check if we have the necessary data
    if filtered_data is not None:
        with col1:
            total_users = filtered_data['user_id'].nunique() if 'user_id' in filtered_data.columns else 0
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-value">{total_users:,}</div>
                <div class="kpi-label">Total Users</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if 'date' in filtered_data.columns:
                date_range = (filtered_data['date'].max() - filtered_data['date'].min()).days
                date_range = max(1, date_range)  # Avoid division by zero
            else:
                date_range = 1
            
            active_days = filtered_data['date'].dt.date.nunique() if 'date' in filtered_data.columns else 0
            activity_rate = (active_days / date_range) * 100 if date_range > 0 else 0
            
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-value">{activity_rate:.1f}%</div>
                <div class="kpi-label">Activity Rate</div>
            </div>
            """, unsafe_allow_html=True)
    
        with col3:
            # Revenue KPI
            revenue_cols = [col for col in filtered_data.columns if 'revenue' in col.lower()]
            if revenue_cols:
                total_revenue = filtered_data[revenue_cols].sum().sum()
                st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-value">${total_revenue:,.2f}</div>
                    <div class="kpi-label">Total Revenue</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="kpi-box">
                    <div class="kpi-value">N/A</div>
                    <div class="kpi-label">Total Revenue</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            # Check if we have the data for churn analysis
            if data['churn_analysis'] is not None and 'churned' in data['churn_analysis'].columns:
                churn_rate = (data['churn_analysis']['churned'].sum() / len(data['churn_analysis'])) * 100
                st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-value">{churn_rate:.1f}%</div>
                    <div class="kpi-label">Churn Rate</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="kpi-box">
                    <div class="kpi-value">N/A</div>
                    <div class="kpi-label">Churn Rate</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Display sample of data if available
    if filtered_data is not None and not filtered_data.empty:
        st.markdown('<div class="sub-header">Sample Data</div>', unsafe_allow_html=True)
        st.dataframe(filtered_data.head(5), use_container_width=True)
    
    # Display a basic overview of what's available in the dashboard
    st.markdown('<div class="sub-header">Dashboard Sections</div>', unsafe_allow_html=True)
    
    st.markdown("""
    * **User Metrics**: Analyze DAU, WAU, and MAU trends
    * **Revenue Analysis**: Track revenue patterns over time and by segment
    * **User Segments**: Understand different user groups based on behavior
    * **Churn Analysis**: Identify and analyze user churn patterns
    * **Cohort Analysis**: Track retention rates for different user cohorts
    * **User Clusters**: Discover natural segments in user data using clustering
    """)

if __name__ == "__main__":
    main() 