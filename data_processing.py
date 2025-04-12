# World Inequality Database - Data Processing and Feature Engineering

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Import functions from our exploration module
from data_exploration import read_wid_csv, explore_countries, explore_country_data, compare_variable_availability

# Key variables we'll be using based on what's available in the dataset
# We found these variables in our exploration:
# - Income variables: 'adiincf992', 'adiinci992', 'aptincf992', 'gfaincj992'
# - Wealth variables: 'bhwealj992'
# - Share variables: 'spllinf992'

# Define variables to try
INCOME_VARIABLES = ['aptincf992', 'adiinci992', 'aptinci992', 'gfaincj992'] 
WEALTH_VARIABLES = ['bhwealj992']
SHARE_VARIABLES = ['spllinf992']

# Define percentiles of interest (based on what's available in your dataset)
TOP_PERCENTILES = ['p90p100', 'p99p100', 'p99.9p100']  # Top 10%, Top 1%, Top 0.1%
BOTTOM_PERCENTILES = ['p0p50', 'p0p90']  # Bottom 50%, Bottom 90%
MIDDLE_PERCENTILES = ['p50p90']  # Middle 40%

# Countries to include in our analysis
# We'll select a diverse set of countries from different regions and development levels
COUNTRIES_TO_ANALYZE = [
    # High-income countries
    'US',   # United States
    'FR',   # France
    'DE',   # Germany
    'GB',   # United Kingdom
    'JP',   # Japan
    
    # Upper-middle income countries
    'BR',   # Brazil
    'CN',   # China
    'RU',   # Russia
    'ZA',   # South Africa
    
    # Lower-middle and low-income countries
    'IN',   # India
    'ID',   # Indonesia
    'NG',   # Nigeria
    'EG'    # Egypt
]

# Function to load country data with selected variables
def load_country_data(country_code, directory='wid_all_data'):
    """
    Load specific inequality variables for a given country.
    
    Args:
        country_code (str): Two-letter country code
        directory (str): Path to WID data directory
    
    Returns:
        tuple: (data_df, metadata_df) for the country
    """
    data_path = os.path.join(directory, f'WID_data_{country_code}.csv')
    metadata_path = os.path.join(directory, f'WID_metadata_{country_code}.csv')
    
    if not os.path.exists(data_path) or not os.path.exists(metadata_path):
        print(f"Data or metadata not found for {country_code}")
        return None, None
    
    data_df = read_wid_csv(data_path)
    metadata_df = read_wid_csv(metadata_path)
    
    return data_df, metadata_df

# Function to create a dataset for a specific inequality metric
def create_inequality_dataset(countries, variable_codes, percentiles, directory='wid_all_data'):
    """
    Create a dataset comparing specific inequality variables across countries.
    Will try each variable code in the list until one works.
    
    Args:
        countries (list): List of country codes
        variable_codes (list or str): WID variable code(s) to try
        percentiles (list): List of percentile codes (e.g., ['p99p100', 'p0p50'])
        directory (str): Path to WID data directory
    
    Returns:
        pd.DataFrame: Combined dataset with inequality data
    """
    # Convert single variable code to list for consistent processing
    if isinstance(variable_codes, str):
        variable_codes = [variable_codes]
    
    # Load country information for names
    countries_info = explore_countries(directory)
    if countries_info is None:
        print("Could not load country information")
        return None
    
    countries_df = countries_info['countries_df']
    country_name_map = dict(zip(countries_df['alpha2'], countries_df['shortname']))
    
    # Try each variable code until we find one that works
    for variable_code in variable_codes:
        print(f"Trying variable code: {variable_code}")
        combined_df = pd.DataFrame()
        
        for country in countries:
            data_df, metadata_df = load_country_data(country, directory)
            
            if data_df is None:
                print(f"  Skipping {country} - could not load data")
                continue
            
            # Filter for the requested variable and percentiles
            filtered_df = data_df[(data_df['variable'] == variable_code) & 
                                (data_df['percentile'].isin(percentiles))]
            
            if filtered_df.empty:
                print(f"  No data for {variable_code} with percentiles {percentiles} in {country}")
                continue
            
            # Add country name
            filtered_df['country_code'] = country
            filtered_df['country_name'] = country_name_map.get(country, country)
            
            # Append to combined dataset
            combined_df = pd.concat([combined_df, filtered_df])
            print(f"  Found data for {country}: {len(filtered_df)} rows")
        
        if not combined_df.empty:
            print(f"Successfully found data for variable {variable_code}")
            return combined_df
    
    print(f"No data found for any of these variables: {variable_codes} across specified countries and percentiles")
    return None

# Function to create a comparative dataset of income/wealth distribution over time
def create_time_series_dataset(variable_codes, percentile, countries=COUNTRIES_TO_ANALYZE, directory='wid_all_data'):
    """
    Create a dataset of inequality metrics over time for multiple countries.
    Will try multiple variable codes until one works.
    
    Args:
        variable_codes (list or str): WID variable code(s) to try
        percentile (str): Percentile code
        countries (list): List of country codes
        directory (str): Path to WID data directory
    
    Returns:
        pd.DataFrame: Time series data for the specified variable and percentile
    """
    # Convert single variable code to list for consistent processing
    if isinstance(variable_codes, str):
        variable_codes = [variable_codes]
    
    # Get variable description (try to get from first variable code, but not critical)
    variable_desc = None
    try:
        sample_country = countries[0]
        _, metadata_df = load_country_data(sample_country, directory)
        
        if metadata_df is not None:
            for var_code in variable_codes:
                var_info = metadata_df[metadata_df['variable'] == var_code]
                if not var_info.empty:
                    variable_desc = var_info.iloc[0]['simpledes']
                    break
    except:
        pass
    
    # Try to create dataset with any of the provided variable codes
    dataset = create_inequality_dataset(countries, variable_codes, [percentile], directory)
    
    if dataset is not None:
        # Pivot to have years as columns and countries as rows for easier plotting
        dataset = dataset.sort_values(['country_name', 'year'])
        
        # Add metadata
        dataset.attrs['variable_code'] = dataset['variable'].iloc[0]  # Use the actual variable code that worked
        dataset.attrs['variable_desc'] = variable_desc
        dataset.attrs['percentile'] = percentile
    
    return dataset

# Function to create a dataset for GDP per capita
def create_gdp_dataset(countries=COUNTRIES_TO_ANALYZE, directory='wid_all_data'):
    """
    Create a dataset of GDP per capita for comparison with inequality metrics.
    Using national income per adult as a proxy.
    
    Args:
        countries (list): List of country codes
        directory (str): Path to WID data directory
    
    Returns:
        pd.DataFrame: GDP per capita data
    """
    # Try several possible GDP/income per adult variables
    gdp_variables = ['anninc992i', 'aptinc992i', 'adiinci992', 'adiincf992']
    
    # We don't need a percentile for this aggregate measure, but WID still requires one
    # p0p100 represents the entire population
    gdp_data = create_inequality_dataset(countries, gdp_variables, ['p0p100'], directory)
    
    if gdp_data is not None:
        # Add variable description
        var_code = gdp_data['variable'].iloc[0]
        gdp_data.attrs['variable_desc'] = f'Income per Adult ({var_code})'
        
        # Convert to common currency (USD) using most recent PPP rates
        # This would require additional implementation to get PPP conversion rates
        # For simplicity, we'll leave the values in local currency
    
    return gdp_data

# Function to create a cross-sectional dataset with the available variables
def create_cross_sectional_dataset(countries=COUNTRIES_TO_ANALYZE, year=2020, directory='wid_all_data'):
    """
    Create a cross-sectional dataset combining multiple inequality metrics for a specific year.
    Adapts to find available variables in the dataset.
    
    Args:
        countries (list): List of country codes
        year (int): Reference year for the cross-section
        directory (str): Path to WID data directory
    
    Returns:
        pd.DataFrame: Combined dataset with multiple inequality metrics
    """
    print("Building cross-sectional dataset with available metrics...")
    
    # Initialize results dataframe with country codes
    countries_info = explore_countries(directory)
    country_name_map = {}
    
    if countries_info is not None and 'countries_df' in countries_info:
        countries_df = countries_info['countries_df']
        country_name_map = dict(zip(countries_df['alpha2'], countries_df['shortname']))
    
    # Create the base dataframe with country information
    result_df = pd.DataFrame({
        'country_code': countries,
        'country_name': [country_name_map.get(c, c) for c in countries]
    })
    
    # Identify what variables we have available from our time series datasets
    available_datasets = []
    
    # Try loading income variables for different percentiles
    for percentile in TOP_PERCENTILES + BOTTOM_PERCENTILES:
        for var_prefix in ['a', 's']:  # Try both average and share variables
            for var_type in INCOME_VARIABLES:
                # Only use the variable type part (e.g., 'ptincf992' from 'aptincf992')
                var_base = var_type[1:] if var_type.startswith('a') or var_type.startswith('s') else var_type
                test_var = f"{var_prefix}{var_base}"
                
                dataset = create_inequality_dataset(countries, [test_var], [percentile], directory)
                if dataset is not None and not dataset.empty:
                    metric_name = f"{var_prefix}_{var_base}_{percentile}"
                    available_datasets.append({
                        'name': metric_name,
                        'dataset': dataset,
                        'variable': test_var,
                        'percentile': percentile
                    })
                    print(f"  Found data for {test_var} with percentile {percentile}")
    
    # Try loading wealth variables
    for percentile in TOP_PERCENTILES + BOTTOM_PERCENTILES:
        for wealth_var in WEALTH_VARIABLES:
            dataset = create_inequality_dataset(countries, [wealth_var], [percentile], directory)
            if dataset is not None and not dataset.empty:
                metric_name = f"{wealth_var}_{percentile}"
                available_datasets.append({
                    'name': metric_name,
                    'dataset': dataset,
                    'variable': wealth_var,
                    'percentile': percentile
                })
                print(f"  Found data for {wealth_var} with percentile {percentile}")
    
    # Try loading GDP or income per adult variables
    gdp_data = create_gdp_dataset(countries, directory)
    if gdp_data is not None and not gdp_data.empty:
        var_code = gdp_data['variable'].iloc[0]
        available_datasets.append({
            'name': f"{var_code}_per_adult",
            'dataset': gdp_data,
            'variable': var_code,
            'percentile': 'p0p100'
        })
        print(f"  Found GDP/income per adult data: {var_code}")
    
    # Extract values for the reference year (or closest available)
    for dataset_info in available_datasets:
        df = dataset_info['dataset']
        name = dataset_info['name']
        
        # Initialize new columns with NaN
        result_df[f"{name}_value"] = np.nan
        result_df[f"{name}_year"] = np.nan
        
        # Process each country
        for country in result_df['country_code']:
            country_data = df[df['country_code'] == country]
            
            if not country_data.empty:
                # Try to get the exact year first
                year_data = country_data[country_data['year'] == year]
                
                # If exact year not available, find closest year
                if year_data.empty:
                    available_years = country_data['year'].unique()
                    if len(available_years) > 0:
                        closest_year = available_years[np.abs(available_years - year).argmin()]
                        year_data = country_data[country_data['year'] == closest_year]
                
                # If we found data, add it to the result
                if not year_data.empty:
                    row_idx = result_df[result_df['country_code'] == country].index[0]
                    result_df.loc[row_idx, f"{name}_value"] = year_data['value'].iloc[0]
                    result_df.loc[row_idx, f"{name}_year"] = year_data['year'].iloc[0]
    
    # Add region information if available
    if countries_info is not None and 'countries_df' in countries_info:
        countries_df = countries_info['countries_df']
        
        for idx, row in result_df.iterrows():
            country_info = countries_df[countries_df['alpha2'] == row['country_code']]
            if not country_info.empty:
                result_df.loc[idx, 'region'] = country_info['region'].iloc[0]
                result_df.loc[idx, 'region2'] = country_info['region2'].iloc[0]
    
    # Display summary of metrics found
    value_cols = [col for col in result_df.columns if col.endswith('_value')]
    print(f"Created cross-sectional dataset with {len(value_cols)} metrics for {len(result_df)} countries")
    
    return result_df

# Function to create a dataset comparing changes in inequality over time
def create_inequality_change_dataset(countries=COUNTRIES_TO_ANALYZE, 
                                     variable_codes=INCOME_VARIABLES,
                                     percentile='p99p100', 
                                     start_year=1980, 
                                     end_year=2020,
                                     directory='wid_all_data'):
    """
    Create a dataset showing changes in inequality metrics over time.
    
    Args:
        countries (list): List of country codes
        variable_codes (list or str): WID variable code(s) to try
        percentile (str): Percentile code
        start_year (int): Starting year for change calculation
        end_year (int): Ending year for change calculation
        directory (str): Path to WID data directory
    
    Returns:
        pd.DataFrame: Dataset with inequality changes
    """
    # Get the time series data
    time_series = create_time_series_dataset(variable_codes, percentile, countries, directory)
    
    if time_series is None:
        return None
    
    # Calculate changes
    change_data = []
    
    # Group by country
    for country, group in time_series.groupby('country_code'):
        group = group.sort_values('year')
        country_name = group['country_name'].iloc[0]
        
        # Try to get values for exact years
        start_data = group[group['year'] == start_year]
        end_data = group[group['year'] == end_year]
        
        # If exact years not available, find closest years
        if start_data.empty:
            available_years = group['year'].unique()
            closest_start = available_years[np.abs(available_years - start_year).argmin()]
            start_data = group[group['year'] == closest_start]
        
        if end_data.empty:
            available_years = group['year'].unique()
            closest_end = available_years[np.abs(available_years - end_year).argmin()]
            end_data = group[group['year'] == closest_end]
        
        # Skip if we don't have data for both periods
        if start_data.empty or end_data.empty:
            print(f"Insufficient data for {country} to calculate changes")
            continue
        
        # Calculate changes
        start_value = start_data['value'].iloc[0]
        end_value = end_data['value'].iloc[0]
        actual_start_year = start_data['year'].iloc[0]
        actual_end_year = end_data['year'].iloc[0]
        
        absolute_change = end_value - start_value
        percent_change = (absolute_change / start_value) * 100 if start_value != 0 else np.nan
        
        change_data.append({
            'country_code': country,
            'country_name': country_name,
            'start_year': actual_start_year,
            'end_year': actual_end_year,
            'start_value': start_value,
            'end_value': end_value,
            'absolute_change': absolute_change,
            'percent_change': percent_change
        })
    
    # Convert to DataFrame
    change_df = pd.DataFrame(change_data)
    
    if change_df.empty:
        return None
    
    # Add metadata
    change_df.attrs['variable_code'] = time_series.attrs.get('variable_code', '')
    change_df.attrs['variable_desc'] = time_series.attrs.get('variable_desc', '')
    change_df.attrs['percentile'] = percentile
    
    return change_df

# Function to combine income and wealth inequality data for correlation analysis
def create_correlation_dataset(countries=COUNTRIES_TO_ANALYZE, reference_year=2020, directory='wid_all_data'):
    """
    Create a dataset to analyze correlations between income and wealth inequality.
    
    Args:
        countries (list): List of country codes
        reference_year (int): Reference year for the cross-section
        directory (str): Path to WID data directory
    
    Returns:
        pd.DataFrame: Dataset with income and wealth inequality metrics
    """
    # Get cross-sectional data
    cross_section = create_cross_sectional_dataset(countries, reference_year, directory)
    
    if cross_section is None or cross_section.empty:
        print("Could not create cross-sectional dataset for correlation analysis")
        return None
    
    # Create metrics for correlation analysis
    corr_metrics = [
        ('top1_income_share', 'top1_wealth_share'),
        ('top10_income_share', 'top10_wealth_share'),
        ('bottom50_income_share', 'bottom50_wealth_share'),
        ('gdp_per_adult', 'top1_income_share'),
        ('gdp_per_adult', 'top1_wealth_share')
    ]
    
    # Calculate correlations
    correlations = {}
    
    for x_var, y_var in corr_metrics:
        if x_var in cross_section.columns and y_var in cross_section.columns:
            # Filter out NaN values
            valid_data = cross_section[[x_var, y_var]].dropna()
            
            if len(valid_data) >= 5:  # Require at least 5 countries for meaningful correlation
                corr, p_value = stats.pearsonr(valid_data[x_var], valid_data[y_var])
                correlations[f'{x_var}_vs_{y_var}'] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'n': len(valid_data)
                }
    
    # Add correlations to dataset attributes
    cross_section.attrs['correlations'] = correlations
    
    return cross_section

# Main function to prepare all datasets
def prepare_all_datasets(directory='wid_all_data'):
    """
    Prepare all datasets needed for our inequality analysis.
    
    Args:
        directory (str): Path to WID data directory
    
    Returns:
        dict: Dictionary of prepared datasets
    """
    print("Preparing inequality datasets...")
    
    datasets = {}
    
    # 1. Time series of top 1% income share
    print("Creating top 1% income share time series...")
    datasets['top1_income_time'] = create_time_series_dataset(
        INCOME_VARIABLES, 'p99p100', COUNTRIES_TO_ANALYZE, directory)
    
    # 2. Time series of top 10% income share 
    print("Creating top 10% income share time series...")
    datasets['top10_income_time'] = create_time_series_dataset(
        INCOME_VARIABLES, 'p90p100', COUNTRIES_TO_ANALYZE, directory)
    
    # 3. Time series of bottom 50% income share
    print("Creating bottom 50% income share time series...")
    datasets['bottom50_income_time'] = create_time_series_dataset(
        INCOME_VARIABLES, 'p0p50', COUNTRIES_TO_ANALYZE, directory)
    
    # 4. Time series of top 1% wealth share
    print("Creating top 1% wealth share time series...")
    datasets['top1_wealth_time'] = create_time_series_dataset(
        WEALTH_VARIABLES, 'p99p100', COUNTRIES_TO_ANALYZE, directory)
    
    # 5. Try to create a dataset with share variables if available
    print("Creating share variable time series if available...")
    if SHARE_VARIABLES:
        for percentile in TOP_PERCENTILES + BOTTOM_PERCENTILES:
            dataset_name = f"share_{percentile}_time"
            datasets[dataset_name] = create_time_series_dataset(
                SHARE_VARIABLES, percentile, COUNTRIES_TO_ANALYZE, directory)
    
    # 6. Cross-sectional data for most recent year
    print("Creating cross-sectional dataset...")
    datasets['cross_section'] = create_cross_sectional_dataset(
        COUNTRIES_TO_ANALYZE, 2020, directory)
    
    # 7. Changes in income inequality (1980-2020)
    print("Creating income inequality change dataset...")
    datasets['income_change'] = create_inequality_change_dataset(
        COUNTRIES_TO_ANALYZE, INCOME_VARIABLES, 'p99p100', 1980, 2020, directory)
    
    # 8. Changes in wealth inequality (1980-2020)
    print("Creating wealth inequality change dataset...")
    datasets['wealth_change'] = create_inequality_change_dataset(
        COUNTRIES_TO_ANALYZE, WEALTH_VARIABLES, 'p99p100', 1980, 2020, directory)
    
    # 9. Correlation dataset
    print("Creating correlation dataset...")
    datasets['correlation'] = create_correlation_dataset(
        COUNTRIES_TO_ANALYZE, 2020, directory)
    
    # Save datasets to CSV files
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    for name, df in datasets.items():
        if df is not None and not df.empty:
            df.to_csv(os.path.join(output_dir, f'{name}.csv'), index=False)
            print(f"Saved {name}.csv")
    
    print("Dataset preparation complete!")
    return datasets

# Run the data preparation if executed as a script
if __name__ == "__main__":
    # prepared_data = prepare_all_datasets()
    
    dataset = create_inequality_dataset(countries=['US', 'FR'], 
                                        variable_codes=INCOME_VARIABLES, 
                                        percentiles=['p99p100'], 
                                        directory='wid_all_data')
    
    print(compare_variable_availability())
    
    print(dataset.head())