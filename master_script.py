# World Inequality Database Analysis - Main Script

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Import our custom modules
# Note: In a real execution, ensure these modules are in the correct path
from data_exploration import explore_dataset
from data_processing import prepare_all_datasets, COUNTRIES_TO_ANALYZE
from data_analysis import create_all_visualizations

# Set up output directories
os.makedirs('output', exist_ok=True)
os.makedirs('figures', exist_ok=True)

def run_full_analysis(directory='wid_all_data'):
    """
    Run the full inequality analysis pipeline.
    
    Args:
        directory (str): Path to WID data directory
    
    Returns:
        dict: Results from each step of the pipeline
    """
    print("=" * 80)
    print("WORLD INEQUALITY DATABASE ANALYSIS")
    print("=" * 80)
    print(f"Starting analysis at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    
    results = {}
    
    # Step 1: Explore the dataset structure
    print("\nSTEP 1: EXPLORING DATASET STRUCTURE")
    print("-" * 80)
    exploration_results = explore_dataset(directory)
    results['exploration'] = exploration_results
    
    # Step 2: Process and prepare datasets
    print("\nSTEP 2: PROCESSING AND PREPARING DATASETS")
    print("-" * 80)
    prepared_datasets = prepare_all_datasets(directory)
    results['datasets'] = prepared_datasets
    
    # Step 3: Create visualizations
    print("\nSTEP 3: CREATING VISUALIZATIONS")
    print("-" * 80)
    visualizations = create_all_visualizations(prepared_datasets, directory)
    results['visualizations'] = visualizations
    
    # Step 4: Generate summary statistics
    print("\nSTEP 4: GENERATING SUMMARY STATISTICS")
    print("-" * 80)
    summary_stats = generate_summary_statistics(prepared_datasets)
    results['summary_stats'] = summary_stats
    
    # Step 5: Test hypotheses
    print("\nSTEP 5: TESTING HYPOTHESES")
    print("-" * 80)
    hypothesis_results = test_hypotheses(prepared_datasets)
    results['hypothesis_tests'] = hypothesis_results
    
    print("\n" + "=" * 80)
    print(f"Analysis completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return results

def generate_summary_statistics(datasets):
    """
    Generate summary statistics for our datasets.
    
    Args:
        datasets (dict): Dictionary of prepared datasets
    
    Returns:
        dict: Dictionary of summary statistics
    """
    summary = {}
    
    # Countries included in analysis
    summary['countries'] = COUNTRIES_TO_ANALYZE
    
    # Summary statistics for cross-sectional data
    if 'cross_section' in datasets and datasets['cross_section'] is not None:
        cross_section = datasets['cross_section']
        
        # Calculate summary statistics for key inequality metrics
        metrics = ['top1_income_share', 'top10_income_share', 'bottom50_income_share', 
                   'top1_wealth_share', 'top10_wealth_share', 'bottom50_wealth_share']
        
        metric_stats = {}
        for metric in metrics:
            if metric in cross_section.columns:
                valid_data = cross_section[metric].dropna()
                
                if not valid_data.empty:
                    metric_stats[metric] = {
                        'mean': valid_data.mean(),
                        'median': valid_data.median(),
                        'min': valid_data.min(),
                        'min_country': cross_section.loc[valid_data.idxmin(), 'country_name'],
                        'max': valid_data.max(),
                        'max_country': cross_section.loc[valid_data.idxmax(), 'country_name'],
                        'std_dev': valid_data.std(),
                        'count': len(valid_data)
                    }
        
        summary['metric_stats'] = metric_stats
        
        # Summarize by region if region data is available
        if 'region' in cross_section.columns:
            region_stats = cross_section.groupby('region')[metrics].mean().to_dict()
            summary['region_stats'] = region_stats
    
    # Changes in inequality over time
    if 'income_change' in datasets and datasets['income_change'] is not None:
        income_change = datasets['income_change']
        
        # Calculate average changes
        summary['avg_income_change'] = {
            'abs_mean': income_change['absolute_change'].mean(),
            'pct_mean': income_change['percent_change'].mean(),
            'countries_increasing': sum(income_change['absolute_change'] > 0),
            'countries_decreasing': sum(income_change['absolute_change'] < 0),
            'largest_increase': income_change.loc[income_change['percent_change'].idxmax(), 'country_name'],
            'largest_decrease': income_change.loc[income_change['percent_change'].idxmin(), 'country_name']
        }
    
    if 'wealth_change' in datasets and datasets['wealth_change'] is not None:
        wealth_change = datasets['wealth_change']
        
        # Calculate average changes
        summary['avg_wealth_change'] = {
            'abs_mean': wealth_change['absolute_change'].mean(),
            'pct_mean': wealth_change['percent_change'].mean(),
            'countries_increasing': sum(wealth_change['absolute_change'] > 0),
            'countries_decreasing': sum(wealth_change['absolute_change'] < 0),
            'largest_increase': wealth_change.loc[wealth_change['percent_change'].idxmax(), 'country_name'],
            'largest_decrease': wealth_change.loc[wealth_change['percent_change'].idxmin(), 'country_name']
        }
    
    # Print summary statistics
    print("Summary Statistics:")
    print(f"  Countries analyzed: {len(summary['countries'])}")
    
    if 'metric_stats' in summary:
        print("\n  Inequality Metrics (Latest Available Year):")
        for metric, stats in summary['metric_stats'].items():
            print(f"    {metric.replace('_', ' ').title()}:")
            print(f"      Mean: {stats['mean']:.2%}")
            print(f"      Range: {stats['min']:.2%} ({stats['min_country']}) to {stats['max']:.2%} ({stats['max_country']})")
    
    if 'avg_income_change' in summary:
        print("\n  Income Inequality Changes (1980-2020):")
        inc_stats = summary['avg_income_change']
        print(f"    Average percent change: {inc_stats['pct_mean']:.2f}%")
        print(f"    Countries with increasing inequality: {inc_stats['countries_increasing']}")
        print(f"    Countries with decreasing inequality: {inc_stats['countries_decreasing']}")
        print(f"    Largest increase: {inc_stats['largest_increase']}")
        print(f"    Largest decrease: {inc_stats['largest_decrease']}")
    
    # Save summary statistics to CSV
    with open('output/summary_statistics.txt', 'w') as f:
        f.write("WORLD INEQUALITY DATABASE ANALYSIS\n")
        f.write("Summary Statistics\n\n")
        
        f.write(f"Countries analyzed ({len(summary['countries'])}):\n")
        f.write(", ".join(summary['countries']) + "\n\n")
        
        if 'metric_stats' in summary:
            f.write("Inequality Metrics (Latest Available Year):\n")
            for metric, stats in summary['metric_stats'].items():
                f.write(f"  {metric.replace('_', ' ').title()}:\n")
                f.write(f"    Mean: {stats['mean']:.2%}\n")
                f.write(f"    Median: {stats['median']:.2%}\n")
                f.write(f"    Range: {stats['min']:.2%} ({stats['min_country']}) to {stats['max']:.2%} ({stats['max_country']})\n")
                f.write(f"    Standard Deviation: {stats['std_dev']:.2%}\n")
                f.write(f"    Countries with data: {stats['count']}\n\n")
        
        if 'region_stats' in summary:
            f.write("Regional Averages:\n")
            for region, metrics in summary['region_stats'].items():
                f.write(f"  {region}:\n")
                for metric, value in metrics.items():
                    if not np.isnan(value):
                        f.write(f"    {metric.replace('_', ' ').title()}: {value:.2%}\n")
                f.write("\n")
        
        if 'avg_income_change' in summary:
            f.write("Income Inequality Changes (1980-2020):\n")
            inc_stats = summary['avg_income_change']
            f.write(f"  Average percent change: {inc_stats['pct_mean']:.2f}%\n")
            f.write(f"  Countries with increasing inequality: {inc_stats['countries_increasing']}\n")
            f.write(f"  Countries with decreasing inequality: {inc_stats['countries_decreasing']}\n")
            f.write(f"  Largest increase: {inc_stats['largest_increase']}\n")
            f.write(f"  Largest decrease: {inc_stats['largest_decrease']}\n\n")
        
        if 'avg_wealth_change' in summary:
            f.write("Wealth Inequality Changes (1980-2020):\n")
            wealth_stats = summary['avg_wealth_change']
            f.write(f"  Average percent change: {wealth_stats['pct_mean']:.2f}%\n")
            f.write(f"  Countries with increasing inequality: {wealth_stats['countries_increasing']}\n")
            f.write(f"  Countries with decreasing inequality: {wealth_stats['countries_decreasing']}\n")
            f.write(f"  Largest increase: {wealth_stats['largest_increase']}\n")
            f.write(f"  Largest decrease: {wealth_stats['largest_decrease']}\n")
    
    print(f"  Summary statistics saved to 'output/summary_statistics.txt'")
    
    return summary

def test_hypotheses(datasets):
    """
    Test our hypotheses about inequality patterns.
    
    Args:
        datasets (dict): Dictionary of prepared datasets
    
    Returns:
        dict: Results of hypothesis tests
    """
    results = {}
    
    # Hypothesis 1: Countries with similar economic development levels show significantly different inequality patterns
    print("\nHypothesis 1: Countries with similar economic development levels show significantly different inequality patterns")
    
    if 'cross_section' in datasets and datasets['cross_section'] is not None:
        cross_section = datasets['cross_section']
        
        # First, identify which columns we can use for testing this hypothesis
        # Look for GDP-like variables (per capita/adult income metrics)
        gdp_cols = [col for col in cross_section.columns 
                   if any(term in col for term in ['gdp', 'income', 'inc']) 
                   and '_value' in col]
        
        # Look for inequality metrics
        inequality_cols = [col for col in cross_section.columns 
                          if any(term in col for term in ['top', 'share', 'gini'])
                          and '_value' in col]
        
        if gdp_cols and inequality_cols:
            # Use the first available GDP and inequality metrics
            gdp_col = gdp_cols[0]
            inequality_col = inequality_cols[0]
            
            print(f"  Using {gdp_col} as economic development indicator")
            print(f"  Using {inequality_col} as inequality indicator")
            
            # Make sure we have enough data points
            valid_data = cross_section[[gdp_col, inequality_col]].dropna()
            
            if len(valid_data) >= 6:  # Need at least 6 countries to make 3 groups of 2
                # Create GDP quantiles to group countries by development level
                # Use rank() to handle duplicate values, then use qcut on the ranks
                try:
                    # Calculate ranks (handling ties)
                    ranks = valid_data[gdp_col].rank(method='first')
                    
                    # Determine number of quantiles based on data points
                    n_quantiles = min(3, len(valid_data) // 2)  # At least 2 countries per group
                    
                    # Create quantiles based on ranks
                    valid_data['gdp_quantile'] = pd.qcut(
                        ranks, 
                        q=n_quantiles, 
                        labels=[f'Group {i+1}' for i in range(n_quantiles)]
                    )
                    
                    # Calculate inequality variation within each GDP group
                    gdp_group_stats = valid_data.groupby('gdp_quantile')[inequality_col].agg(['mean', 'std', 'min', 'max', 'count'])
                    
                    # Calculate coefficient of variation (std/mean) to compare variation across groups
                    gdp_group_stats['cv'] = gdp_group_stats['std'] / gdp_group_stats['mean']
                    
                    # Calculate range as a percentage of mean
                    gdp_group_stats['range_pct_of_mean'] = (gdp_group_stats['max'] - gdp_group_stats['min']) / gdp_group_stats['mean'] * 100
                    
                    results['gdp_group_stats'] = gdp_group_stats.to_dict()
                    
                    # Find examples of similar GDP but different inequality
                    # For each GDP group, find country pairs with the largest inequality differences
                    for gdp_group in gdp_group_stats.index:
                        group_data = valid_data[valid_data['gdp_quantile'] == gdp_group]
                        
                        if len(group_data) >= 2:
                            # Find min and max inequality in this group
                            min_idx = group_data[inequality_col].idxmin()
                            max_idx = group_data[inequality_col].idxmax()
                            
                            # Get country names if available, otherwise use codes
                            low_country = (cross_section.loc[min_idx, 'country_name'] 
                                          if 'country_name' in cross_section.columns 
                                          else cross_section.loc[min_idx, 'country_code'])
                            high_country = (cross_section.loc[max_idx, 'country_name'] 
                                          if 'country_name' in cross_section.columns 
                                          else cross_section.loc[max_idx, 'country_code'])
                            
                            results[f'{gdp_group}_gdp_contrast'] = {
                                'low_inequality_country': low_country,
                                'low_inequality_value': group_data.loc[min_idx, inequality_col],
                                'high_inequality_country': high_country,
                                'high_inequality_value': group_data.loc[max_idx, inequality_col],
                                'difference_ratio': group_data.loc[max_idx, inequality_col] / group_data.loc[min_idx, inequality_col]
                            }
                    
                    # Print the results
                    print("  Results:")
                    print(f"  Variation in {inequality_col} within GDP Groups:")
                    for gdp_group, stats in gdp_group_stats.iterrows():
                        print(f"    {gdp_group} (n={stats['count']:.0f}):")
                        print(f"      Mean: {stats['mean']:.4f}")
                        print(f"      Range: {stats['min']:.4f} to {stats['max']:.4f}")
                        print(f"      Coefficient of Variation: {stats['cv']:.2f}")
                        print(f"      Range as % of Mean: {stats['range_pct_of_mean']:.1f}%")
                    
                    print("\n  Examples of Similar GDP but Different Inequality:")
                    for gdp_group in gdp_group_stats.index:
                        if f'{gdp_group}_gdp_contrast' in results:
                            contrast = results[f'{gdp_group}_gdp_contrast']
                            print(f"    {gdp_group}:")
                            print(f"      {contrast['low_inequality_country']} ({contrast['low_inequality_value']:.4f}) vs. "
                                f"{contrast['high_inequality_country']} ({contrast['high_inequality_value']:.4f})")
                            print(f"      Difference ratio: {contrast['difference_ratio']:.1f}x")
                    
                    # Final assessment
                    print("\n  Assessment:")
                    if any(stats['cv'] > 0.2 for _, stats in gdp_group_stats.iterrows()):
                        print("  ✓ SUPPORTED: Significant variation in inequality exists within similar GDP levels")
                        results['hypothesis1_supported'] = True
                    else:
                        print("  ✗ NOT SUPPORTED: Variation in inequality within GDP levels is limited")
                        results['hypothesis1_supported'] = False
                except Exception as e:
                    print(f"  Error during hypothesis testing: {e}")
                    results['hypothesis1_supported'] = None
            else:
                print(f"  Insufficient data points ({len(valid_data)}) to test hypothesis")
                results['hypothesis1_supported'] = None
        else:
            print("  Cannot test hypothesis - missing GDP or inequality metrics")
            if gdp_cols:
                print(f"  Available GDP/income metrics: {gdp_cols}")
            if inequality_cols:
                print(f"  Available inequality metrics: {inequality_cols}")
            results['hypothesis1_supported'] = None
    else:
        print("  Cannot test hypothesis - missing cross-sectional dataset")
        results['hypothesis1_supported'] = None
    
    # Hypothesis 2: The relationship between top income shares and bottom 50% prosperity varies across countries
    print("\nHypothesis 2: The relationship between top income shares and bottom 50% prosperity varies across countries")
    
    if 'cross_section' in datasets and datasets['cross_section'] is not None:
        cross_section = datasets['cross_section']
        
        # Check if we have any metrics that could be used for this hypothesis
        # Looking for top income and bottom income metrics
        top_cols = [col for col in cross_section.columns if any(x in col for x in ['top', 'p90p100', 'p99p100'])]
        bottom_cols = [col for col in cross_section.columns if any(x in col for x in ['bottom', 'p0p50'])]
        
        if top_cols and bottom_cols:
            # Use the first available metrics
            top_col = top_cols[0]
            bottom_col = bottom_cols[0]
            
            print(f"  Using {top_col} and {bottom_col} to test hypothesis")
            
            # Make sure we have enough data points
            valid_data = cross_section[[top_col, bottom_col, 'country_code']].dropna()
            
            if len(valid_data) >= 3:  # Need at least 3 countries for meaningful comparison
                try:
                    # Calculate the ratio of top to bottom metrics
                    valid_data['top_bottom_ratio'] = valid_data[top_col] / valid_data[bottom_col]
                    
                    # Calculate summary statistics for the ratio
                    ratio_stats = valid_data['top_bottom_ratio'].agg(['mean', 'median', 'std', 'min', 'max']).to_dict()
                    
                    # Find countries with highest and lowest ratios
                    min_idx = valid_data['top_bottom_ratio'].idxmin()
                    max_idx = valid_data['top_bottom_ratio'].idxmax()
                    
                    # Get country codes (or names if available)
                    min_country = valid_data.loc[min_idx, 'country_code']
                    max_country = valid_data.loc[max_idx, 'country_code']
                    
                    # Try to get country names if available
                    if 'country_name' in valid_data.columns:
                        min_country = valid_data.loc[min_idx, 'country_name']
                        max_country = valid_data.loc[max_idx, 'country_name']
                    
                    # Calculate coefficient of variation
                    ratio_stats['cv'] = ratio_stats['std'] / ratio_stats['mean']
                    
                    results['top_bottom_ratio'] = {
                        'stats': ratio_stats,
                        'min_country': min_country,
                        'min_value': valid_data.loc[min_idx, 'top_bottom_ratio'],
                        'max_country': max_country,
                        'max_value': valid_data.loc[max_idx, 'top_bottom_ratio'],
                        'max_min_ratio': valid_data.loc[max_idx, 'top_bottom_ratio'] / valid_data.loc[min_idx, 'top_bottom_ratio']
                    }
                    
                    # Print the results
                    print("  Results:")
                    print(f"  Ratio of {top_col} to {bottom_col}:")
                    print(f"    Mean Ratio: {ratio_stats['mean']:.2f}")
                    print(f"    Range: {ratio_stats['min']:.2f} ({min_country}) to "
                          f"{ratio_stats['max']:.2f} ({max_country})")
                    print(f"    Coefficient of Variation: {ratio_stats['cv']:.2f}")
                    print(f"    Max/Min Ratio: {results['top_bottom_ratio']['max_min_ratio']:.1f}x")
                    
                    # Final assessment
                    print("\n  Assessment:")
                    if ratio_stats['cv'] > 0.3 or results['top_bottom_ratio']['max_min_ratio'] > 2.0:
                        print("  ✓ SUPPORTED: The relationship between top and bottom income shares varies substantially across countries")
                        results['hypothesis2_supported'] = True
                    else:
                        print("  ✗ NOT SUPPORTED: The relationship between top and bottom income shares is relatively consistent")
                        results['hypothesis2_supported'] = False
                except Exception as e:
                    print(f"  Error testing hypothesis: {e}")
                    results['hypothesis2_supported'] = None
            else:
                print(f"  Insufficient data points ({len(valid_data)}) to test hypothesis")
                results['hypothesis2_supported'] = None
        else:
            print("  Cannot test hypothesis - missing top or bottom income share data")
            print(f"  Available top metrics: {top_cols}")
            print(f"  Available bottom metrics: {bottom_cols}")
            results['hypothesis2_supported'] = None
    else:
        print("  Cannot test hypothesis - missing cross-sectional dataset")
        results['hypothesis2_supported'] = None
    
    # Hypothesis 3: Changes in wealth inequality don't always move in tandem with changes in income inequality
    print("\nHypothesis 3: Changes in wealth inequality don't always move in tandem with changes in income inequality")
    
    if all(k in datasets and datasets[k] is not None and not datasets[k].empty for k in ['income_change', 'wealth_change']):
        income_change = datasets['income_change']
        wealth_change = datasets['wealth_change']
        
        # Merge the datasets to compare changes
        if 'country_code' in income_change.columns and 'country_code' in wealth_change.columns:
            common_countries = set(income_change['country_code']).intersection(set(wealth_change['country_code']))
            
            if common_countries:
                try:
                    # Create a merged dataset for common countries
                    income_subset = income_change[income_change['country_code'].isin(common_countries)]
                    wealth_subset = wealth_change[wealth_change['country_code'].isin(common_countries)]
                    
                    # Create merged dataframe
                    merged = pd.merge(
                        income_subset[['country_code', 'percent_change']],
                        wealth_subset[['country_code', 'percent_change']],
                        on='country_code',
                        suffixes=('_income', '_wealth')
                    )
                    
                    # Add country names if available
                    if 'country_name' in income_subset.columns:
                        merged = pd.merge(
                            merged,
                            income_subset[['country_code', 'country_name']].drop_duplicates(),
                            on='country_code'
                        )
                    
                    # Calculate correlation between income and wealth changes
                    if len(merged) >= 3:  # Need at least 3 points for meaningful correlation
                        correlation = np.corrcoef(merged['percent_change_income'], merged['percent_change_wealth'])[0, 1]
                        p_value = stats.pearsonr(merged['percent_change_income'], merged['percent_change_wealth'])[1] if len(merged) > 2 else 1.0
                    else:
                        correlation = np.nan
                        p_value = np.nan
                    
                    # Count countries where income and wealth inequality moved in different directions
                    different_directions = sum((merged['percent_change_income'] > 0) != (merged['percent_change_wealth'] > 0))
                    
                    # Find the country with the largest divergence (if there are enough countries)
                    if len(merged) >= 2:
                        merged['direction_diff'] = np.abs(merged['percent_change_income'] - merged['percent_change_wealth'])
                        divergence_idx = merged['direction_diff'].idxmax()
                        
                        # Determine country name
                        if 'country_name' in merged.columns:
                            divergence_country = merged.loc[divergence_idx, 'country_name']
                        else:
                            divergence_country = merged.loc[divergence_idx, 'country_code']
                        
                        results['income_wealth_changes'] = {
                            'correlation': correlation,
                            'p_value': p_value,
                            'countries_different_directions': different_directions,
                            'percent_different_directions': different_directions / len(merged) * 100 if len(merged) > 0 else 0,
                            'largest_divergence_country': divergence_country,
                            'largest_divergence_income': merged.loc[divergence_idx, 'percent_change_income'],
                            'largest_divergence_wealth': merged.loc[divergence_idx, 'percent_change_wealth']
                        }
                    else:
                        results['income_wealth_changes'] = {
                            'correlation': correlation,
                            'p_value': p_value,
                            'countries_different_directions': different_directions,
                            'percent_different_directions': different_directions / len(merged) * 100 if len(merged) > 0 else 0
                        }
                    
                    # Print the results
                    print("  Results:")
                    if not np.isnan(correlation):
                        print(f"    Correlation between income and wealth inequality changes: {correlation:.2f}")
                    else:
                        print("    Correlation could not be calculated (insufficient data)")
                        
                    print(f"    Countries with different directions: {different_directions} out of {len(merged)} ({different_directions/len(merged)*100:.1f}% if len(merged) > 0 else 0)")
                    
                    if len(merged) >= 2:
                        divergence_country = results['income_wealth_changes']['largest_divergence_country']
                        divergence_income = results['income_wealth_changes']['largest_divergence_income']
                        divergence_wealth = results['income_wealth_changes']['largest_divergence_wealth']
                        
                        print(f"    Largest divergence: {divergence_country}")
                        print(f"      Income change: {divergence_income:.1f}%")
                        print(f"      Wealth change: {divergence_wealth:.1f}%")
                    
                    # Final assessment
                    print("\n  Assessment:")
                    if (not np.isnan(correlation) and correlation < 0.7) or different_directions >= max(1, len(merged) // 4):
                        print("  ✓ SUPPORTED: Wealth and income inequality often move independently")
                        results['hypothesis3_supported'] = True
                    elif np.isnan(correlation) or len(merged) < 3:
                        print("  ? INCONCLUSIVE: Insufficient data to test hypothesis")
                        results['hypothesis3_supported'] = None
                    else:
                        print("  ✗ NOT SUPPORTED: Wealth and income inequality generally move together")
                        results['hypothesis3_supported'] = False
                except Exception as e:
                    print(f"  Error testing hypothesis: {e}")
                    print("  ? INCONCLUSIVE: Encountered error during analysis")
                    results['hypothesis3_supported'] = None
            else:
                print("  Cannot test hypothesis - no common countries between income and wealth change datasets")
                results['hypothesis3_supported'] = None
        else:
            print("  Cannot test hypothesis - country_code column missing in one or both datasets")
            results['hypothesis3_supported'] = None
    else:
        print("  Cannot test hypothesis - missing income or wealth change datasets")
        results['hypothesis3_supported'] = None
    
    # Save hypothesis test results to file
    with open('output/hypothesis_tests.txt', 'w') as f:
        f.write("WORLD INEQUALITY DATABASE ANALYSIS\n")
        f.write("Hypothesis Tests\n\n")
        
        # Hypothesis 1
        f.write("Hypothesis 1: Countries with similar economic development levels show significantly different inequality patterns\n")
        if 'hypothesis1_supported' in results:
            if results['hypothesis1_supported'] is True:
                f.write("  RESULT: SUPPORTED\n\n")
            elif results['hypothesis1_supported'] is False:
                f.write("  RESULT: NOT SUPPORTED\n\n")
            else:
                f.write("  RESULT: INCONCLUSIVE (Insufficient data)\n\n")
            
            if 'gdp_group_stats' in results:
                f.write("  Variation in Top 1% Income Share within GDP Groups:\n")
                for gdp_group, stats in results['gdp_group_stats'].items():
                    f.write(f"    {gdp_group} GDP Group (n={stats['count']:.0f}):\n")
                    f.write(f"      Mean: {stats['mean']:.2%}\n")
                    f.write(f"      Range: {stats['min']:.2%} to {stats['max']:.2%}\n")
                    f.write(f"      Coefficient of Variation: {stats['cv']:.2f}\n")
                    f.write(f"      Range as % of Mean: {stats['range_pct_of_mean']:.1f}%\n\n")
        else:
            f.write("  RESULT: INCONCLUSIVE (Insufficient data)\n\n")
        
        # Hypothesis 2
        f.write("Hypothesis 2: The relationship between top income shares and bottom 50% prosperity varies across countries\n")
        if 'hypothesis2_supported' in results:
            if results['hypothesis2_supported'] is True:
                f.write("  RESULT: SUPPORTED\n\n")
            elif results['hypothesis2_supported'] is False:
                f.write("  RESULT: NOT SUPPORTED\n\n")
            else:
                f.write("  RESULT: INCONCLUSIVE (Insufficient data)\n\n")
            
            if 'top10_bottom50_ratio' in results:
                ratio_data = results['top10_bottom50_ratio']
                f.write("  Ratio of Top 10% to Bottom 50% Income Share:\n")
                f.write(f"    Mean Ratio: {ratio_data['stats']['mean']:.2f}\n")
                f.write(f"    Range: {ratio_data['stats']['min']:.2f} ({ratio_data['min_country']}) to "
                      f"{ratio_data['stats']['max']:.2f} ({ratio_data['max_country']})\n")
                f.write(f"    Coefficient of Variation: {ratio_data['stats']['cv']:.2f}\n")
                f.write(f"    Max/Min Ratio: {ratio_data['max_min_ratio']:.1f}x\n\n")
        else:
            f.write("  RESULT: INCONCLUSIVE (Insufficient data)\n\n")
        
        # Hypothesis 3
        f.write("Hypothesis 3: Changes in wealth inequality don't always move in tandem with changes in income inequality\n")
        if 'hypothesis3_supported' in results:
            if results['hypothesis3_supported'] is True:
                f.write("  RESULT: SUPPORTED\n\n")
            elif results['hypothesis3_supported'] is False:
                f.write("  RESULT: NOT SUPPORTED\n\n")
            else:
                f.write("  RESULT: INCONCLUSIVE (Insufficient data)\n\n")
            
            if 'income_wealth_changes' in results:
                change_data = results['income_wealth_changes']
                f.write("  Comparison of Income and Wealth Inequality Changes:\n")
                f.write(f"    Correlation: {change_data['correlation']:.2f} (p = {change_data['p_value']:.3f})\n")
                f.write(f"    Countries with different directions: {change_data['countries_different_directions']} "
                      f"({change_data['percent_different_directions']:.1f}%)\n")
                f.write(f"    Largest divergence: {change_data['largest_divergence_country']}\n")
                f.write(f"      Income change: {change_data['largest_divergence_income']:.1f}%\n")
                f.write(f"      Wealth change: {change_data['largest_divergence_wealth']:.1f}%\n")
        else:
            f.write("  RESULT: INCONCLUSIVE (Insufficient data)\n\n")
    
    print(f"  Hypothesis test results saved to 'output/hypothesis_tests.txt'")
    
    return results

# Run the full analysis if executed as a script
if __name__ == "__main__":
    analysis_results = run_full_analysis()
    print("\nFull analysis complete! Results are available in the 'output' and 'figures' directories.")