#!/usr/bin/env python3
"""
Matiks Analytics Dashboard Runner Script

This script provides commands to process data and run the Streamlit dashboard.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_data_processing():
    """Run the data processing script to prepare all data for the dashboard"""
    print("Running data processing...")
    script_path = os.path.join("scripts", "data_processor.py")
    
    try:
        subprocess.run([sys.executable, script_path], check=True)
        print("Data processing completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running data processing: {e}")
        return False

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("Starting Matiks Analytics Dashboard...")
    dashboard_path = os.path.join("dashboard", "app.py")
    
    try:
        subprocess.run(["streamlit", "run", dashboard_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running dashboard: {e}")
        return False
    except FileNotFoundError:
        print("Streamlit not found. Please make sure Streamlit is installed.")
        print("You can install it with: pip install streamlit")
        return False

def check_data_files():
    """Check if necessary data files exist"""
    data_dir = "data"
    required_files = [
        "matiks_data_clean.csv",
        "dau.csv",
        "wau.csv",
        "mau.csv",
        "revenue_analysis.csv",
        "user_segments.csv",
        "churn_analysis.csv",
        "cohort_retention.csv",
        "user_clusters.csv",
        "cluster_analysis.csv",
        "funnel_analysis.csv"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    return missing_files

def main():
    """Main function to parse arguments and run commands"""
    parser = argparse.ArgumentParser(description="Matiks Analytics Dashboard Runner")
    parser.add_argument("--process-data", action="store_true", help="Run data processing before starting dashboard")
    parser.add_argument("--dashboard-only", action="store_true", help="Run only the dashboard without data processing")
    
    args = parser.parse_args()
    
    # Ensure we're in the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Check if data directory exists
    if not os.path.exists("data"):
        print("Creating data directory...")
        os.makedirs("data")
    
    # Process data if requested or if files are missing
    if args.dashboard_only:
        missing_files = check_data_files()
        if missing_files:
            print(f"Warning: Some required data files are missing: {', '.join(missing_files)}")
            print("The dashboard may not work correctly without these files.")
            response = input("Do you want to run data processing first? (y/n): ")
            if response.lower() == 'y':
                run_data_processing()
    elif args.process_data or check_data_files():
        run_data_processing()
    
    # Run the dashboard
    run_dashboard()

if __name__ == "__main__":
    main() 