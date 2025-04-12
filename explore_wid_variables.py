import os
import pandas as pd

def explore_variables(country_code='US', directory='wid_all_data'):
    data_path = os.path.join(directory, f'WID_data_{country_code}.csv')
    
    if not os.path.exists(data_path):
        print(f"Data file for {country_code} not found")
        return
    
    # Load a sample of data to see what's available
    data_df = pd.read_csv(data_path, sep=';', nrows=5000)
    
    # Get unique variables
    variables = sorted(data_df['variable'].unique())
    
    # Look for income and wealth related variables
    income_vars = [v for v in variables if 'inc' in v]
    wealth_vars = [v for v in variables if 'weal' in v]
    share_vars = [v for v in variables if v.startswith('s')]
    
    print(f"Sample income variables: {income_vars[:10]}")
    print(f"Sample wealth variables: {wealth_vars[:10]}")
    print(f"Sample share variables: {share_vars[:10]}")
    
    # Get sample percentiles
    percentiles = sorted(data_df['percentile'].unique())
    print(f"Sample percentiles: {percentiles[:10]}")
    
    return variables, percentiles

# Run exploration
variables, percentiles = explore_variables('US')