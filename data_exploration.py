# World Inequality Database Analysis
# CS 328 Writing Assignment - 2025

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Set visualization styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create output directories if they don't exist
os.makedirs('figures', exist_ok=True)
os.makedirs('output', exist_ok=True)

# Function to read WID CSV files with proper parameters
def read_wid_csv(file_path):
    """
    Read WID CSV files using the semicolon separator as specified in documentation.
    """
    try:
        return pd.read_csv(file_path, sep=';', encoding='utf-8')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Function to list available files in the WID data directory
def list_wid_files(directory='wid_all_data'):
    """List and categorize files in the WID data directory."""
    if not os.path.exists(directory):
        print(f"Directory {directory} not found")
        return None
    
    all_files = os.listdir(directory)
    
    # Categorize files
    country_file = [f for f in all_files if f == 'WID_countries.csv']
    data_files = sorted([f for f in all_files if f.startswith('WID_data_')])
    metadata_files = sorted([f for f in all_files if f.startswith('WID_metadata_')])
    other_files = [f for f in all_files if f not in country_file + data_files + metadata_files]
    
    # Create a summary dictionary
    file_summary = {
        'country_file': country_file,
        'data_files': data_files,
        'metadata_files': metadata_files,
        'other_files': other_files,
        'total_files': len(all_files),
        'total_countries': len(data_files)
    }
    
    return file_summary

# Explore available countries and their metadata
def explore_countries(directory='wid_all_data'):
    """Load and explore country data from WID_countries.csv."""
    countries_path = os.path.join(directory, 'WID_countries.csv')
    
    if not os.path.exists(countries_path):
        print(f"Country file not found at {countries_path}")
        return None
    
    countries_df = read_wid_csv(countries_path)
    
    if countries_df is not None:
        # Create a summary of regions
        region_counts = countries_df['region'].value_counts()
        region2_counts = countries_df['region2'].value_counts()
        
        # Filter actual countries (2-letter codes) from regions/aggregates
        countries_only = countries_df[countries_df['alpha2'].str.len() == 2]
        
        # Create a country summary
        country_summary = {
            'total_entries': len(countries_df),
            'country_count': len(countries_only),
            'regions': region_counts.to_dict(),
            'subregions': region2_counts.to_dict()
        }
        
        return {
            'countries_df': countries_df,
            'summary': country_summary
        }
    
    return None

# Function to explore the structure of a single country's data file
def explore_country_data(country_code, directory='wid_all_data'):
    """
    Explore the data structure for a single country.
    
    Args:
        country_code (str): Two-letter country code (e.g., 'US', 'FR')
        directory (str): Path to the WID data directory
    
    Returns:
        dict: Summary information about the country's data
    """
    data_path = os.path.join(directory, f'WID_data_{country_code}.csv')
    metadata_path = os.path.join(directory, f'WID_metadata_{country_code}.csv')
    
    if not os.path.exists(data_path) or not os.path.exists(metadata_path):
        print(f"Data or metadata file for {country_code} not found")
        return None
    
    # Load data and metadata
    data_df = read_wid_csv(data_path)
    metadata_df = read_wid_csv(metadata_path)
    
    if data_df is None or metadata_df is None:
        return None
    
    # Create data summary
    data_summary = {
        'rows': len(data_df),
        'variables': data_df['variable'].nunique(),
        'variable_list': sorted(data_df['variable'].unique()),
        'percentiles': data_df['percentile'].nunique(),
        'percentile_list': sorted(data_df['percentile'].unique()),
        'years': {
            'min': data_df['year'].min(),
            'max': data_df['year'].max(),
            'count': data_df['year'].nunique()
        }
    }
    
    # Create metadata summary
    metadata_summary = {
        'rows': len(metadata_df),
        'unique_variables': metadata_df['variable'].nunique(),
        'variable_list': sorted(metadata_df['variable'].unique())
    }
    
    return {
        'data_df': data_df,
        'metadata_df': metadata_df,
        'data_summary': data_summary,
        'metadata_summary': metadata_summary
    }

# Function to extract variable descriptions from metadata
def get_variable_descriptions(metadata_df):
    """
    Extract unique variable descriptions from metadata.
    
    Args:
        metadata_df (pd.DataFrame): Metadata dataframe
    
    Returns:
        pd.DataFrame: Dataframe with variable codes and descriptions
    """
    if metadata_df is None:
        return None
    
    # Check if required columns exist
    required_columns = ['variable', 'age', 'pop', 'shortname', 'simpledes', 'technicaldes', 'longtype', 'shortpop', 'longpop', 'shortage', 'longage', 'unit']
    if not all(col in metadata_df.columns for col in required_columns):
        print(f"Metadata is missing required columns. Available columns: {metadata_df.columns.tolist()}")
        return None
    
    # Extract unique variable descriptions
    var_descriptions = metadata_df[required_columns].drop_duplicates()
    print(f"Variable Descriptions DataFrame:\n{var_descriptions.sort_values('variable').reset_index(drop=True).head()}")
    return var_descriptions.sort_values('variable').reset_index(drop=True)

# Function to examine variable availability across countries
def compare_variable_availability(country_list, directory='wid_all_data'):
    """
    Compare which variables are available across multiple countries.
    
    Args:
        country_list (list): List of country codes to compare
        directory (str): Path to WID data directory
    
    Returns:
        pd.DataFrame: Data frame showing variable availability by country
    """
    availability_data = []
    
    for country in country_list:
        data_path = os.path.join(directory, f'WID_data_{country}.csv')
        
        if os.path.exists(data_path):
            data_df = read_wid_csv(data_path)
            
            if data_df is not None:
                variables = data_df['variable'].unique()
                
                for var in variables:
                    # Check year range for this variable
                    var_data = data_df[data_df['variable'] == var]
                    year_min = var_data['year'].min()
                    year_max = var_data['year'].max()
                    
                    availability_data.append({
                        'country': country,
                        'variable': var,
                        'available': True,
                        'year_min': year_min,
                        'year_max': year_max,
                        'year_count': var_data['year'].nunique()
                    })
    
    # Convert to DataFrame
    availability_df = pd.DataFrame(availability_data)
    
    print(f"Availability DataFrame:\n{availability_df.head()}")
    # Create a pivot table of availability
    if not availability_df.empty:
        pivot_df = pd.pivot_table(
            availability_df, 
            values='available',
            index='variable',
            columns='country',
            aggfunc=lambda x: True if len(x) > 0 else False,
            fill_value=False
        )
        
        print(f"Pivot Table:\n{pivot_df.head()}")
        # Add a total count column
        pivot_df['total_countries'] = pivot_df.sum(axis=1)
        
        # Sort by availability
        pivot_df = pivot_df.sort_values('total_countries', ascending=False)
        
        return pivot_df
    
    return None

# Main execution to explore the dataset
def explore_dataset(directory='wid_all_data'):
    """Main function to explore the WID dataset structure."""
    print("Exploring WID dataset structure...")
    
    # List available files
    files = list_wid_files(directory)
    if files:
        print(f"Total files: {files['total_files']}")
        print(f"Country files: {len(files['data_files'])}")
        
        # Show some example countries
        if files['data_files']:
            print("Example countries:", [f.replace('WID_data_', '').replace('.csv', '') 
                                         for f in files['data_files'][:10]])
    
    # Explore countries metadata
    countries_info = explore_countries(directory)
    if countries_info:
        countries_df = countries_info['countries_df']
        print(f"\nTotal countries/regions: {len(countries_df)}")
        
        # Display regions
        print("\nWorld regions:")
        for region, count in countries_info['summary']['regions'].items():
            print(f"  {region}: {count} entries")
    
    # Explore a sample country
    sample_country = 'US'  # United States as example
    country_info = explore_country_data(sample_country, directory)
    
    if country_info:
        print(f"\nSample data for {sample_country}:")
        # print(f"  Data Summary: {country_info['data_summary']}")
        # print(f"  Metadata Summary: {country_info['metadata_summary']}")
        print(f"  Rows: {country_info['data_summary']['rows']}")
        print(f"  Unique variables: {country_info['data_summary']['variables']}")
        print(f"  Year range: {country_info['data_summary']['years']['min']} - {country_info['data_summary']['years']['max']}")
        
        # Show some variable descriptions
        var_desc = get_variable_descriptions(country_info['metadata_df'])
        if var_desc is not None and len(var_desc) > 0:
            print("\nSample variable descriptions:")
            for _, row in var_desc.head(5).iterrows():
                print(f"  {row['variable']}: {row['longtype']} ({row['unit']})")
    
    return {
        'files': files,
        'countries_info': countries_info,
        'sample_country_info': country_info
    }

# Run the exploration if executed as a script
if __name__ == "__main__":
    explore_result = explore_dataset()
    print("\nExploration complete!")