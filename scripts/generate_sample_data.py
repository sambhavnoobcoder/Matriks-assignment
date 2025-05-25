#!/usr/bin/env python3
"""
Sample Data Generator for Matiks Analytics Dashboard

This script generates sample data for testing the Matiks Analytics Dashboard.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random

def generate_sample_data(num_users=1000, days=90, output_path='../data/matiks_data_clean.csv'):
    """
    Generate sample data for testing
    
    Parameters:
    -----------
    num_users : int
        Number of users to generate
    days : int
        Number of days of data to generate
    output_path : str
        Path to save the generated data
    """
    print(f"Generating sample data with {num_users} users over {days} days...")
    
    # Create user IDs
    user_ids = [f"user_{i}" for i in range(1, num_users + 1)]
    
    # Create date range
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Create device types and game modes
    device_types = ['Mobile', 'Desktop', 'Tablet']
    game_modes = ['Casual', 'Ranked', 'Tournament', 'Practice']
    
    # Create data rows
    data_rows = []
    
    # Assign user segments and activity patterns
    user_segments = {}
    for user_id in user_ids:
        # Randomly assign activity level (affects how often they appear in the data)
        activity_level = random.choices(['high', 'medium', 'low', 'churned'], weights=[0.1, 0.3, 0.4, 0.2])[0]
        
        # Randomly assign spending level (affects revenue)
        spending_level = random.choices(['high', 'medium', 'low', 'none'], weights=[0.05, 0.15, 0.3, 0.5])[0]
        
        # Randomly assign preferred device and game mode
        preferred_device = random.choice(device_types)
        preferred_game_mode = random.choice(game_modes)
        
        # Store user segment info
        user_segments[user_id] = {
            'activity_level': activity_level,
            'spending_level': spending_level,
            'preferred_device': preferred_device,
            'preferred_game_mode': preferred_game_mode
        }
    
    # Generate activity data
    for date in date_range:
        # Determine which users are active on this date
        active_users = []
        
        for user_id in user_ids:
            segment = user_segments[user_id]
            
            # Probability of being active based on activity level
            if segment['activity_level'] == 'high':
                prob = 0.8
            elif segment['activity_level'] == 'medium':
                prob = 0.5
            elif segment['activity_level'] == 'low':
                prob = 0.2
            else:  # churned
                # Churned users are active at the beginning but stop after a while
                days_since_start = (date - start_date).days
                if days_since_start < days // 3:
                    prob = 0.3
                else:
                    prob = 0.01
            
            # Check if user is active on this date
            if random.random() < prob:
                active_users.append(user_id)
        
        # Generate activity data for active users
        for user_id in active_users:
            segment = user_segments[user_id]
            
            # Number of activities for this user on this date
            if segment['activity_level'] == 'high':
                num_activities = random.randint(3, 10)
            elif segment['activity_level'] == 'medium':
                num_activities = random.randint(2, 5)
            else:
                num_activities = random.randint(1, 3)
            
            # Generate revenue based on spending level
            if segment['spending_level'] == 'high':
                revenue = round(random.uniform(10, 50), 2) if random.random() < 0.3 else 0
            elif segment['spending_level'] == 'medium':
                revenue = round(random.uniform(5, 15), 2) if random.random() < 0.2 else 0
            elif segment['spending_level'] == 'low':
                revenue = round(random.uniform(1, 5), 2) if random.random() < 0.1 else 0
            else:
                revenue = 0
            
            # Determine device type (mostly preferred, sometimes others)
            if random.random() < 0.8:
                device = segment['preferred_device']
            else:
                device = random.choice([d for d in device_types if d != segment['preferred_device']])
            
            # Determine game mode (mostly preferred, sometimes others)
            if random.random() < 0.7:
                game_mode = segment['preferred_game_mode']
            else:
                game_mode = random.choice([m for m in game_modes if m != segment['preferred_game_mode']])
            
            # Add row to data
            data_rows.append({
                'user_id': user_id,
                'date': date,
                'activity_count': num_activities,
                'revenue': revenue,
                'device_type': device,
                'game_mode': game_mode
            })
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Sample data generated and saved to {output_path}")
    print(f"Total records: {len(df)}")
    
    return df

if __name__ == "__main__":
    # Generate sample data
    generate_sample_data() 