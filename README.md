# Matiks Data Analysis Project

This project contains a comprehensive data analysis and dashboard solution for Matiks user behavior and revenue data. The dashboard provides insights into user metrics, revenue patterns, user segmentation, churn analysis, cohort analysis, and user clustering.

## Project Structure

```
Matiks-assignment/
├── data/                    # Directory for data files
│   └── matiks_data_clean.csv # Clean data file
├── dashboard/               # Dashboard components
│   ├── app.py               # Main dashboard application
│   ├── user_metrics.py      # User metrics page
│   ├── revenue_analysis.py  # Revenue analysis page
│   ├── user_segments.py     # User segments page
│   ├── churn_analysis.py    # Churn analysis page
│   ├── cohort_analysis.py   # Cohort analysis page
│   └── user_clusters.py     # User clusters page
├── scripts/                 # Data processing scripts
│   ├── data_loader.py       # Module for loading data
│   └── data_processor.py    # Data processing and preparation
├── run_dashboard.py         # Script to run the dashboard
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Features

- **User Metrics Analysis**: Track DAU, WAU, MAU and user growth metrics
- **Revenue Analysis**: Analyze revenue trends, ARPU, and revenue distribution
- **User Segmentation**: Segment users based on activity and revenue
- **Churn Analysis**: Identify at-risk users and analyze churn patterns
- **Cohort Analysis**: Track retention rates across user cohorts
- **User Clustering**: Use machine learning to discover natural user segments

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Matiks-assignment.git
cd Matiks-assignment
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Dashboard

To run the dashboard with automatic data processing:

```bash
python run_dashboard.py
```

This will:
1. Check if the necessary data files exist
2. Run the data processing if needed
3. Start the Streamlit dashboard

### Command Line Options

- Run data processing only:
```bash
python run_dashboard.py --process-data
```

- Run dashboard without data processing:
```bash
python run_dashboard.py --dashboard-only
```

### Accessing the Dashboard

Once running, the dashboard will be available in your web browser at:
```
http://localhost:8501
```

## Data Requirements

The dashboard expects a CSV file with the following structure:

- `user_id`: Unique identifier for each user
- `date`: Date of user activity
- `revenue`: Revenue generated (if available)
- Other optional columns: device_type, game_mode, etc.

Place your data file in the `data/` directory with the name `matiks_data_clean.csv`.

## Development

### Adding New Features

To add new analyses to the dashboard:

1. Create a new Python file in the `dashboard/` directory
2. Implement a display function that accepts `data` and `filtered_data` parameters
3. Add the page to the navigation in `dashboard/app.py`

### Data Processing

To modify data processing:

1. Edit the `scripts/data_processor.py` file
2. Add new processing functions as needed
3. Update the `main()` function to include your new processing steps

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- plotly
- streamlit
- scikit-learn
- scipy

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Matiks for providing the project requirements and data structure
- Streamlit for the interactive dashboard framework

## Demo

A video demonstration of the dashboard is available in this repository. You can view it to see the dashboard in action:

- [Watch Demo Video](demo.mov)

The demo showcases:
- The interactive dashboard with dark theme
- Navigation between different analytical pages
- Data visualization capabilities
- Insights and recommendations based on data analysis 