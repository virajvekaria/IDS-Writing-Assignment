# World Inequality Database - Analysis and Visualization

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mtick
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib as mpl

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Set visualization styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create output directories if they don't exist
os.makedirs('figures', exist_ok=True)

# Import from our data processing module
from data_processing import (
    COUNTRIES_TO_ANALYZE,
    TOP_PERCENTILES,
    BOTTOM_PERCENTILES,
    MIDDLE_PERCENTILES,
    create_time_series_dataset,
    create_cross_sectional_dataset,
    create_inequality_change_dataset,
    create_correlation_dataset
)

# Define custom color schemes
# Creating a custom colormap for inequality visualization (green to red)
inequality_colors = ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59', '#d73027']
diverging_cmap = LinearSegmentedColormap.from_list('inequality_cmap', inequality_colors)
mpl.colormaps.register(name='inequality', cmap=diverging_cmap)

# Helper function to format percentages in plots
def percentage_formatter(x, pos):
    """Format values as percentages in plots."""
    return f'{100*x:.1f}%'

# Function to visualize inequality time trends
def plot_inequality_time_trends(dataset, title, y_label, output_file=None):
    """
    Create a line plot showing inequality trends over time for multiple countries.
    
    Args:
        dataset (pd.DataFrame): Time series dataset
        title (str): Plot title
        y_label (str): Y-axis label
        output_file (str): Output file path for saving the figure
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if dataset is None or dataset.empty:
        print("No data available for time trends plot")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each country as a line
    for country, group in dataset.groupby('country_name'):
        ax.plot(group['year'], group['value'], linewidth=2, label=country)
    
    # Format axes and labels
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(percentage_formatter))
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Tight layout to fit everything
    plt.tight_layout()
    
    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Saved figure to {output_file}")
    
    return fig

# Function to create faceted time trends for multiple inequality metrics
def plot_faceted_time_trends(datasets, titles, output_file=None):
    """
    Create a faceted plot with multiple inequality trends.
    
    Args:
        datasets (list): List of time series datasets
        titles (list): List of subplot titles
        output_file (str): Output file path for saving the figure
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if not datasets or any(d is None or d.empty for d in datasets):
        print("One or more datasets are missing for faceted plot")
        return None
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(datasets), 1, figsize=(14, 5 * len(datasets)))
    
    # If only one dataset, axes won't be an array
    if len(datasets) == 1:
        axes = [axes]
    
    # Plot each dataset in a separate subplot
    for i, (dataset, title) in enumerate(zip(datasets, titles)):
        ax = axes[i]
        
        # Plot each country as a line
        for country, group in dataset.groupby('country_name'):
            ax.plot(group['year'], group['value'], linewidth=2, label=country)
        
        # Format axes and labels
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Share of Total', fontsize=12)
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(percentage_formatter))
        ax.grid(True, alpha=0.3)
        
        # Only add legend to the first subplot
        if i == 0:
            ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Tight layout to fit everything
    plt.tight_layout()
    
    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Saved figure to {output_file}")
    
    return fig

# Function to visualize inequality changes over time
def plot_inequality_changes(change_dataset, title, output_file=None):
    """
    Create a horizontal bar chart showing changes in inequality metrics.
    
    Args:
        change_dataset (pd.DataFrame): Dataset with inequality changes
        title (str): Plot title
        output_file (str): Output file path for saving the figure
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if change_dataset is None or change_dataset.empty:
        print("No data available for inequality changes plot")
        return None
    
    # Sort by percent change
    sorted_df = change_dataset.sort_values('percent_change')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot horizontal bars
    bars = ax.barh(sorted_df['country_name'], sorted_df['percent_change'])
    
    # Color bars based on direction of change
    for i, bar in enumerate(bars):
        bar.set_color('#d73027' if sorted_df.iloc[i]['percent_change'] > 0 else '#1a9850')
    
    # Add a vertical line at zero
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add text labels for the percentage changes
    for i, value in enumerate(sorted_df['percent_change']):
        ax.text(value + (5 if value >= 0 else -5), 
                i, 
                f"{value:.1f}%", 
                va='center', 
                ha='left' if value >= 0 else 'right',
                fontsize=10)
    
    # Format axes and labels
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Percent Change (%)', fontsize=14)
    ax.set_ylabel('Country', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    # Add year range to subtitle
    start_year = sorted_df['start_year'].min()
    end_year = sorted_df['end_year'].max()
    plt.figtext(0.5, 0.01, f"Change from {start_year} to {end_year}", 
                ha='center', fontsize=12)
    
    # Tight layout to fit everything
    plt.tight_layout()
    
    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Saved figure to {output_file}")
    
    return fig

# Function to create a scatter plot comparing two inequality metrics
def plot_inequality_scatter(dataset, x_var, y_var, x_label, y_label, title, output_file=None):
    """
    Create a scatter plot comparing two inequality metrics across countries.
    
    Args:
        dataset (pd.DataFrame): Cross-sectional dataset
        x_var (str): Column name for x-axis
        y_var (str): Column name for y-axis
        x_label (str): X-axis label
        y_label (str): Y-axis label
        title (str): Plot title
        output_file (str): Output file path for saving the figure
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Check if trying to plot a variable against itself
    if x_var == y_var:
        print(f"Cannot create scatter plot of {x_var} against itself. Skipping.")
        return None
    
    if dataset is None or dataset.empty:
        print("No data available for scatter plot")
        return None
    
    # Filter out countries with missing data for these variables
    valid_data = dataset[[x_var, y_var, 'country_name', 'region']].dropna()
    
    if len(valid_data) < 5:
        print(f"Insufficient data for scatter plot of {x_var} vs {y_var}")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create region codes with correct length
    region_codes = pd.factorize(valid_data['region'])[0]
    
    # Add scatter points, colored by region - ensure array dimensions match
    scatter = ax.scatter(valid_data[x_var], valid_data[y_var], 
                         c=region_codes, 
                         s=100, alpha=0.7, cmap='viridis')
    
    # Add country labels
    for i, row in valid_data.iterrows():
        ax.annotate(row['country_name'], 
                   (row[x_var], row[y_var]),
                   xytext=(5, 5), 
                   textcoords='offset points',
                   fontsize=10)
    
    # Add correlation line
    if len(valid_data) >= 5:
        m, b = np.polyfit(valid_data[x_var], valid_data[y_var], 1)
        ax.plot(valid_data[x_var], m*valid_data[x_var] + b, color='red', linestyle='--', alpha=0.7)
        
        # Calculate and display correlation coefficient
        corr, p_value = stats.pearsonr(valid_data[x_var], valid_data[y_var])
        ax.text(0.05, 0.95, f"r = {corr:.2f} (p = {p_value:.3f})", 
                transform=ax.transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Format axes as percentages if appropriate
    if 'share' in x_var:
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(percentage_formatter))
    if 'share' in y_var:
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(percentage_formatter))
    
    # Add legend for regions
    unique_regions = sorted(valid_data['region'].unique())
    legend = ax.legend(handles=scatter.legend_elements()[0], 
                      labels=unique_regions,
                      title="Region",
                      loc='upper left',
                      bbox_to_anchor=(1.05, 1))
    
    # Format axes and labels
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Tight layout to fit everything
    plt.tight_layout()
    
    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Saved figure to {output_file}")
    
    return fig

# Function to create a clustered bar chart comparing inequality metrics
def plot_inequality_bar_chart(dataset, metrics, countries=None, title=None, output_file=None):
    """
    Create a bar chart comparing multiple inequality metrics across countries.
    
    Args:
        dataset (pd.DataFrame): Cross-sectional dataset
        metrics (list): List of column names for metrics to compare
        countries (list): List of countries to include (or None for all)
        title (str): Plot title
        output_file (str): Output file path for saving the figure
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if dataset is None or dataset.empty:
        print("No data available for bar chart")
        return None
    
    # Filter to specified countries if provided
    if countries:
        filtered_data = dataset[dataset['country_code'].isin(countries)]
    else:
        filtered_data = dataset
    
    # Drop rows with missing values for any of the metrics
    valid_data = filtered_data[['country_name'] + metrics].dropna()
    
    if valid_data.empty:
        print("No valid data available for bar chart after filtering")
        return None
    
    # Sort by the first metric
    sorted_data = valid_data.sort_values(metrics[0], ascending=False)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set width of bars
    bar_width = 0.7 / len(metrics)
    
    # Set positions for bars
    positions = np.arange(len(sorted_data))
    
    # Create bars for each metric
    for i, metric in enumerate(metrics):
        offset = (i - len(metrics)/2 + 0.5) * bar_width
        bars = ax.bar(positions + offset, sorted_data[metric], 
                     width=bar_width, 
                     label=metric.replace('_', ' ').title())
    
    # Add country names as x-tick labels
    ax.set_xticks(positions)
    ax.set_xticklabels(sorted_data['country_name'], rotation=45, ha='right')
    
    # Format y-axis as percentages if all metrics are shares
    if all('share' in metric for metric in metrics):
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(percentage_formatter))
    
    # Add legend, title, and grid
    ax.legend(fontsize=12)
    if title:
        ax.set_title(title, fontsize=16)
    ax.grid(axis='y', alpha=0.3)
    
    # Tight layout to fit everything
    plt.tight_layout()
    
    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Saved figure to {output_file}")
    
    return fig

# Function to create a correlation matrix of inequality metrics
def plot_correlation_matrix(dataset, metrics, title='Correlation Between Inequality Metrics', output_file=None):
    """
    Create a heatmap showing correlations between inequality metrics.
    
    Args:
        dataset (pd.DataFrame): Cross-sectional dataset
        metrics (list): List of column names for metrics to correlate
        title (str): Plot title
        output_file (str): Output file path for saving the figure
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if dataset is None or dataset.empty:
        print("No data available for correlation matrix")
        return None
    
    # Extract metrics and calculate correlation matrix
    valid_data = dataset[metrics].dropna()
    
    if len(valid_data) < 5:
        print("Insufficient data for correlation matrix")
        return None
    
    corr_matrix = valid_data.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, 
                center=0, fmt='.2f', linewidths=0.5, ax=ax)
    
    # Clean up metric names for display
    clean_labels = [m.replace('_', ' ').title() for m in metrics]
    ax.set_xticklabels(clean_labels, rotation=45, ha='right')
    ax.set_yticklabels(clean_labels, rotation=0)
    
    # Add title
    ax.set_title(title, fontsize=16)
    
    # Tight layout to fit everything
    plt.tight_layout()
    
    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Saved figure to {output_file}")
    
    return fig

# Function to create a clustered heatmap to identify inequality regimes
def plot_inequality_regimes(dataset, metrics, n_clusters=3, title='Inequality Regimes Across Countries', output_file=None):
    """
    Create a clustered heatmap to identify inequality regimes across countries.
    
    Args:
        dataset (pd.DataFrame): Cross-sectional dataset
        metrics (list): List of column names for metrics to use in clustering
        n_clusters (int): Number of clusters to identify
        title (str): Plot title
        output_file (str): Output file path for saving the figure
    
    Returns:
        tuple: (matplotlib.figure.Figure, pd.DataFrame) The figure and cluster assignments
    """
    if dataset is None or dataset.empty:
        print("No data available for inequality regimes analysis")
        return None, None
    
    # Extract metrics and drop rows with missing values
    metrics_data = dataset[['country_name'] + metrics].dropna()
    
    if len(metrics_data) < n_clusters * 2:
        print(f"Insufficient data for clustering into {n_clusters} regimes")
        return None, None
    
    # Standardize the data for clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(metrics_data[metrics])
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Add cluster assignments to the data
    metrics_data['cluster'] = clusters
    
    # Sort data by cluster and then by the first metric within each cluster
    sorted_data = metrics_data.sort_values(['cluster', metrics[0]])
    
    # Create a heatmap of the metrics by country, organized by cluster
    fig, ax = plt.subplots(figsize=(12, len(metrics_data) * 0.4 + 2))
    
    # Prepare data for heatmap
    heatmap_data = sorted_data[metrics].copy()
    
    # Scale data to [0, 1] for each metric for better visualization
    for col in metrics:
        min_val = heatmap_data[col].min()
        max_val = heatmap_data[col].max()
        if max_val > min_val:  # Avoid division by zero
            heatmap_data[col] = (heatmap_data[col] - min_val) / (max_val - min_val)
    
    # Create heatmap
    sns.heatmap(heatmap_data, cmap='viridis', linewidths=0.5, ax=ax)
    
    # Clean up metric names for display
    clean_labels = [m.replace('_', ' ').title() for m in metrics]
    ax.set_xticklabels(clean_labels, rotation=45, ha='right')
    
    # Use country names as y-tick labels
    ax.set_yticklabels(sorted_data['country_name'], rotation=0)
    
    # Add cluster boundaries
    cluster_boundaries = np.where(np.diff(sorted_data['cluster']) != 0)[0]
    for boundary in cluster_boundaries:
        ax.axhline(y=boundary + 1, color='white', linewidth=2)
    
    # Add cluster labels
    for cluster in range(n_clusters):
        cluster_mask = sorted_data['cluster'] == cluster
        if np.any(cluster_mask):
            start_idx = np.where(cluster_mask)[0][0]
            size = np.sum(cluster_mask)
            ax.text(-0.1, start_idx + size/2, f"Regime {cluster+1}", 
                   verticalalignment='center', horizontalalignment='right',
                   rotation=90, fontsize=12, fontweight='bold')
    
    # Add title
    ax.set_title(title, fontsize=16)
    
    # Tight layout to fit everything
    plt.tight_layout()
    
    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Saved figure to {output_file}")
    
    return fig, sorted_data

# Function to create a Plotly interactive map visualization of inequality
def create_interactive_inequality_map(dataset, metric, title=None, output_file=None):
    """
    Create an interactive map showing a global inequality metric.
    
    Args:
        dataset (pd.DataFrame): Cross-sectional dataset
        metric (str): Column name for the metric to visualize
        title (str): Plot title
        output_file (str): Output file path for saving the HTML file
    
    Returns:
        plotly.graph_objects.Figure: The interactive figure
    """
    if dataset is None or dataset.empty:
        print("No data available for interactive map")
        return None
    
    # Filter out missing values for the metric
    valid_data = dataset[['country_code', 'country_name', metric]].dropna()
    
    if valid_data.empty:
        print(f"No valid data for {metric} in the dataset")
        return None
    
    # Create the map
    fig = px.choropleth(
        valid_data,
        locations='country_code',
        color=metric,
        hover_name='country_name',
        color_continuous_scale=px.colors.diverging.RdYlGn_r if 'share' in metric else px.colors.sequential.Viridis,
        labels={metric: metric.replace('_', ' ').title()},
        locationmode='ISO-3'
    )
    
    # Update layout
    fig.update_layout(
        title=title if title else f"Global Distribution of {metric.replace('_', ' ').title()}",
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        ),
        coloraxis_colorbar=dict(
            title=metric.replace('_', ' ').title()
        )
    )
    
    # Save as HTML if output file is specified
    if output_file:
        fig.write_html(output_file)
        print(f"Saved interactive map to {output_file}")
    
    return fig

# Function to run all visualizations
def create_all_visualizations(datasets, directory='wid_all_data'):
    """
    Create all visualizations for the inequality analysis.
    
    Args:
        datasets (dict): Dictionary of prepared datasets
        directory (str): Path to WID data directory
    
    Returns:
        dict: Dictionary of created figures
    """
    print("Creating visualizations...")
    figures_dir = 'figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    figures = {}
    
    # Create visualizations for all time series datasets
    for name, dataset in datasets.items():
        if name.endswith('_time') and dataset is not None and not dataset.empty:
            # Skip datasets with only one country (not interesting for comparison)
            if dataset['country_code'].nunique() <= 1:
                print(f"Skipping {name} - only one country available")
                continue
                
            # Create a time trend plot
            print(f"Creating time trends plot for {name}...")
            
            # Determine appropriate title and y-label
            if 'income' in name:
                title = f"{name.replace('_time', '').replace('_', ' ').title()} Trends"
                y_label = "Income Value"
            elif 'wealth' in name:
                title = f"{name.replace('_time', '').replace('_', ' ').title()} Trends"
                y_label = "Wealth Value"
            elif 'share' in name:
                title = f"{name.replace('_time', '').replace('_', ' ').title()} Trends"
                y_label = "Share Value"
            else:
                title = f"{name.replace('_time', '').replace('_', ' ').title()} Trends"
                y_label = "Value"
            
            # Create the plot
            figures[f"{name}_trends"] = plot_inequality_time_trends(
                dataset,
                title,
                y_label,
                os.path.join(figures_dir, f'{name}_trends.png')
            )
    
    # Create faceted time trend plots for related metrics if available
    income_percentiles = [d for d in datasets if d.startswith('top') and 'income' in d and d.endswith('_time')]
    if len(income_percentiles) >= 2:
        print("Creating faceted income trends plot...")
        income_datasets = [datasets[d] for d in income_percentiles if datasets[d] is not None and not datasets[d].empty]
        income_titles = [d.replace('_time', '').replace('_', ' ').title() for d in income_percentiles]
        
        if income_datasets:
            figures['faceted_income_trends'] = plot_faceted_time_trends(
                income_datasets,
                income_titles,
                os.path.join(figures_dir, 'faceted_income_trends.png')
            )
    
    # Create inequality change plots
    for change_name in ['income_change', 'wealth_change']:
        if change_name in datasets and datasets[change_name] is not None and not datasets[change_name].empty:
            print(f"Creating {change_name} plot...")
            
            # Determine title based on the actual variable that worked
            variable_code = datasets[change_name].attrs.get('variable_code', '')
            percentile = datasets[change_name].attrs.get('percentile', '')
            
            title = f"Changes in {variable_code} ({percentile})"
            
            figures[change_name] = plot_inequality_changes(
                datasets[change_name],
                title,
                os.path.join(figures_dir, f'{change_name}.png')
            )
    
    # Try to create scatter plots with whatever metrics are available in cross_section
    if 'cross_section' in datasets and datasets['cross_section'] is not None:
        cross_section = datasets['cross_section']
        
        # Find all value columns
        value_cols = [col for col in cross_section.columns if col.endswith('_value')]
        
        # Look for GDP/income per adult metrics
        gdp_cols = [col for col in value_cols if any(term in col for term in ['gdp', 'nninc', 'ptinc', 'diinc'])]
        
        # Look for inequality metrics
        inequality_cols = [col for col in value_cols if any(term in col for term in ['top', 'p90p100', 'p99p100'])]
        
        # If we have both GDP and inequality metrics, create scatter plots
        if gdp_cols and inequality_cols:
            for gdp_col in gdp_cols[:1]:  # Just use the first GDP metric
                for ineq_col in inequality_cols[:2]:  # Use the first couple inequality metrics
                    # Skip if trying to plot a variable against itself
                    if gdp_col == ineq_col:
                        print(f"Skipping scatter plot of {gdp_col} against itself")
                        continue
                        
                    scatter_name = f"{gdp_col.replace('_value', '')}_vs_{ineq_col.replace('_value', '')}"
                    print(f"Creating scatter plot: {scatter_name}...")
                    
                    figures[scatter_name] = plot_inequality_scatter(
                        cross_section,
                        gdp_col,
                        ineq_col,
                        gdp_col.replace('_value', '').replace('_', ' ').title(),
                        ineq_col.replace('_value', '').replace('_', ' ').title(),
                        f"Relationship Between {gdp_col.replace('_value', '').replace('_', ' ').title()} and {ineq_col.replace('_value', '').replace('_', ' ').title()}",
                        os.path.join(figures_dir, f'{scatter_name}.png')
                    )
    
    # Try to create a bar chart comparing different metrics
    if 'cross_section' in datasets and datasets['cross_section'] is not None:
        cross_section = datasets['cross_section']
        
        # Find value columns that belong to the same category
        income_cols = [col for col in cross_section.columns if col.endswith('_value') and 'income' in col]
        wealth_cols = [col for col in cross_section.columns if col.endswith('_value') and 'wealth' in col]
        share_cols = [col for col in cross_section.columns if col.endswith('_value') and ('share' in col or any(p in col for p in ['p0p50', 'p90p100', 'p99p100']))]
        
        # Use whichever category has the most metrics
        if len(income_cols) >= 2:
            metrics_to_use = income_cols[:3]  # Use up to 3 metrics
            category = "Income"
        elif len(wealth_cols) >= 2:
            metrics_to_use = wealth_cols[:3]
            category = "Wealth"
        elif len(share_cols) >= 2:
            metrics_to_use = share_cols[:3]
            category = "Share"
        else:
            metrics_to_use = []
        
        if metrics_to_use:
            print(f"Creating {category} metrics comparison bar chart...")
            figures[f'{category.lower()}_metrics_bar'] = plot_inequality_bar_chart(
                cross_section,
                metrics_to_use,
                title=f"{category} Metrics Across Countries",
                output_file=os.path.join(figures_dir, f'{category.lower()}_metrics_bar.png')
            )
    
    # Create a correlation matrix if we have multiple metrics
    if 'cross_section' in datasets and datasets['cross_section'] is not None:
        cross_section = datasets['cross_section']
        
        # Find all value columns
        value_cols = [col for col in cross_section.columns if col.endswith('_value')]
        
        if len(value_cols) >= 3:
            # Check if we have enough countries with data for these metrics
            valid_data = cross_section[value_cols].dropna()
            
            if len(valid_data) >= 3:
                print("Creating correlation matrix of inequality metrics...")
                figures['correlation_matrix'] = plot_correlation_matrix(
                    cross_section,
                    value_cols,
                    "Correlation Between Inequality Metrics",
                    os.path.join(figures_dir, 'correlation_matrix.png')
                )
    
    # Try to identify inequality regimes if we have enough metrics
    if 'cross_section' in datasets and datasets['cross_section'] is not None:
        cross_section = datasets['cross_section']
        
        # Find metrics that could be used for clustering
        value_cols = [col for col in cross_section.columns if col.endswith('_value')]
        
        if len(value_cols) >= 2:
            # We need at least 3 countries with complete data
            cluster_data = cross_section[['country_code', 'country_name'] + value_cols].dropna()
            
            if len(cluster_data) >= 6:  # Need at least 6 countries for 3 clusters of 2
                print("Creating inequality regimes clustered heatmap...")
                figures['inequality_regimes'], cluster_assignments = plot_inequality_regimes(
                    cross_section,
                    value_cols[:3],  # Use up to 3 metrics for clustering
                    n_clusters=min(3, len(cluster_data) // 2),  # Ensure at least 2 countries per cluster
                    title="Identifying Inequality Regimes Across Countries",
                    output_file=os.path.join(figures_dir, 'inequality_regimes.png')
                )
                
                # Save cluster assignments if available
                if cluster_assignments is not None:
                    cluster_assignments.to_csv(os.path.join('output', 'inequality_regimes.csv'), index=False)
    
    print("Visualization creation complete!")
    return figures

# Run the visualization creation if executed as a script
if __name__ == "__main__":
    # Check if datasets exist in output directory
    datasets = {}
    for dataset_name in ['top1_income_time', 'top10_income_time', 'bottom50_income_time', 
                         'top1_wealth_time', 'cross_section', 'income_change', 
                         'wealth_change', 'correlation']:
        csv_path = os.path.join('output', f'{dataset_name}.csv')
        if os.path.exists(csv_path):
            datasets[dataset_name] = pd.read_csv(csv_path)
            print(f"Loaded {dataset_name} from {csv_path}")
    
    if not datasets:
        print("No datasets found in output directory. Please run data_processing.py first.")
    else:
        figures = create_all_visualizations(datasets)
        print(f"Created {len(figures)} visualizations in the 'figures' directory")