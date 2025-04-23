# Income and Wealth Inequality Analysis

This Streamlit application explores the relationship between economic development and inequality across countries and over time. The analysis uses data from the World Inequality Database (WID) to examine patterns in income inequality, wealth inequality, and their relationship with economic development.

## Features

- **Trends Over Time**: Visualize how income inequality and per-adult income have changed over time
- **Cross-Sectional Analysis**: Examine the relationship between per-adult income and income inequality for specific years
- **Regression Analysis**: View OLS regression results for the relationship between income and inequality
- **Boxplot Analysis**: Compare inequality distributions across different income levels

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app with:
```
streamlit run app.py
```

The app will open in your default web browser.

## Data

The application uses data from the World Inequality Database (WID), which is included in the `WID_data` directory.

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Statsmodels
- SciPy
- Plotly
