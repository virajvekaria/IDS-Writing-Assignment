import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib as mpl
from linearmodels.panel import PanelOLS, RandomEffects, compare
import warnings

# Set page configuration
st.set_page_config(
    page_title="Income and Wealth Inequality Analysis",
    page_icon="📊",
    layout="wide"
)

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Set visualization styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Title and introduction
st.title("Income and Wealth Inequality Analysis")
st.markdown("""
This application explores the relationship between economic development and inequality across countries and over time.
The analysis uses data from the World Inequality Database (WID) to examine patterns in income inequality,
wealth inequality, and their relationship with economic development.
""")

# Load data
@st.cache_data
def load_data():
    # File paths
    file_income = "WID_data/WID_Data_Average_National_Income.csv"
    file_national_income = "WID_data/WID_Data_National_Income.csv"
    file_net_personal_wealth = "WID_data/WID_Data_Net_Personal_Wealth.csv"
    file_wealth_inequality = "WID_data/WID_Data_Wealth_Gini_Coeff.csv"
    file_income_inequality = "WID_data/WID_Data_Income_Gini_Coeff.csv"
    file_wealth_to_income = "WID_data/WID_Data_Net_National_Wealth_To_Net_National_Income_Ratio.csv"
    file_GDP = "WID_data/WID_Data_GDP.csv"
    file_population = "WID_data/WID_Data_Population.csv"

    try:
        # Try to load the merged dataset if it exists
        df_merged = pd.read_csv("merged_dataset.csv")
        st.success("Loaded pre-processed merged dataset.")
        return df_merged
    except:
        st.info("Processing raw data files...")

        # Read the datasets
        df_income_wealth = pd.read_csv(file_income, header=1, sep=';')
        df_wealth_inequality = pd.read_csv(file_wealth_inequality, header=1, sep=';')
        df_income_inequality = pd.read_csv(file_income_inequality, header=1, sep=';')
        df_wealth_to_income = pd.read_csv(file_wealth_to_income, header=1, sep=';')
        df_GDP = pd.read_csv(file_GDP, header=1, sep=';')
        df_population = pd.read_csv(file_population, header=1, sep=';')
        df_net_wealth = pd.read_csv(file_net_personal_wealth, header=1, sep=';')
        df_national_income = pd.read_csv(file_national_income, header=1, sep=';')

        # Define a reshaping function
        def reshape_wid(df, value_column_name):
            df_long = df.melt(id_vars=['Percentile', 'Year'], var_name='Country', value_name=value_column_name)
            df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')
            df_long['Country'] = df_long['Country'].str.strip()

            # Prefer 'pall', else take 'p90p100'
            if 'pall' in df_long['Percentile'].unique():
                df_long = df_long[df_long['Percentile'] == 'pall']
            else:
                df_long = df_long[df_long['Percentile'] == 'p90p100']

            return df_long.drop(columns=['Percentile'])

        # Reshape and clean all datasets
        df_income_long = reshape_wid(df_income_wealth, 'PerAdultIncome')
        df_wealth_ineq_long = reshape_wid(df_wealth_inequality, 'WealthInequality')
        df_income_ineq_long = reshape_wid(df_income_inequality, 'IncomeInequality')
        df_wealth_to_income_long = reshape_wid(df_wealth_to_income, 'WealthToIncomeRatio')
        df_GDP_long = reshape_wid(df_GDP, 'GDP')
        df_population_long = reshape_wid(df_population, 'Population')
        df_net_wealth_long = reshape_wid(df_net_wealth, 'NetPersonalWealth')
        df_national_income_long = reshape_wid(df_national_income, 'NationalIncome')

        # Merge all datasets on ['Country', 'Year']
        dfs_to_merge = [
            df_income_long,
            df_wealth_ineq_long,
            df_income_ineq_long,
            df_wealth_to_income_long,
            df_GDP_long,
            df_population_long,
            df_net_wealth_long,
            df_national_income_long
        ]

        from functools import reduce
        df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Country', 'Year'], how='inner'), dfs_to_merge)

        # Save the merged dataset for future use
        df_merged.to_csv("merged_dataset.csv", index=False)

        return df_merged

# Load the data
df_merged = load_data()

# Display basic information about the dataset
st.subheader("Dataset Overview")
col1, col2 = st.columns(2)
with col1:
    st.write(f"Number of countries: {df_merged['Country'].nunique()}")
    st.write(f"Time period: {df_merged['Year'].min()} - {df_merged['Year'].max()}")
with col2:
    st.write(f"Total observations: {len(df_merged)}")
    st.write(f"Variables: {', '.join(df_merged.columns.tolist())}")

# Show a sample of the data
with st.expander("View sample data"):
    st.dataframe(df_merged.head())

# Data Analysis
st.markdown("---")

# Trends Over Time
st.header("Trends Over Time")
st.markdown("""
This section shows how income inequality and per-adult income have changed over time.
The charts display average values across all countries in the dataset.
""")

# 1. Average Income Inequality over time
avg_inequality = df_merged.groupby("Year")["IncomeInequality"].mean()

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(avg_inequality.index, avg_inequality.values, marker='o')
ax1.set_xlabel("Year")
ax1.set_ylabel("Average Income Inequality")
ax1.set_title("Average Income Inequality Over Time")
ax1.grid(True)
st.pyplot(fig1)

# 2. Average Per-Adult Income over time
avg_income = df_merged.groupby("Year")["PerAdultIncome"].mean()

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(avg_income.index, avg_income.values, marker='o', color='orange')
ax2.set_xlabel("Year")
ax2.set_ylabel("Average Per-Adult Income")
ax2.set_title("Average Per-Adult Income Over Time")
ax2.grid(True)
st.pyplot(fig2)

# Cross-Sectional Analysis
st.markdown("---")
st.header("Cross-Sectional Analysis")
st.markdown("""
This section examines the relationship between per-adult income and income inequality for a specific year.
Use the slider to select a year for analysis.
""")

# Year selection slider
year = st.slider(
    "Select Year",
    min_value=int(df_merged['Year'].min()),
    max_value=int(df_merged['Year'].max()),
    value=2020,
    step=1
)

# Filter data for selected year
df_year = df_merged[df_merged['Year'] == year].copy()

if df_year.empty:
    st.warning(f"No data available for year {year}.")
else:
    # Scale the 'Population' column to determine dot sizes
    population = df_year['Population']
    marker_sizes = (population / population.max()) * 300  # adjust scaling factor as needed

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df_year['PerAdultIncome'], df_year['IncomeInequality'],
                s=marker_sizes, alpha=0.7, edgecolor='k')

    # Compute the regression line if more than one point exists
    x = df_year['PerAdultIncome']
    y = df_year['IncomeInequality']
    if len(x) > 1:
        sorted_idx = np.argsort(x)
        x_sorted = x.iloc[sorted_idx]
        # Fit a line using np.polyfit
        m, b = np.polyfit(x, y, 1)
        y_fit = m * x_sorted + b
        ax.plot(x_sorted, y_fit, color='red', linewidth=2, label='Best Fit Line')
        # Calculate Pearson correlation coefficient
        corr_h1 = x.corr(y)
        annotation_text = f'Pearson Corr: {corr_h1:.2f}'
    else:
        annotation_text = "Not enough data for regression"

    # Annotate the plot with the Pearson correlation
    ax.text(0.05, 0.95, annotation_text, transform=ax.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Set labels and title
    ax.set_xlabel(f"Per-Adult National Income ({year})", fontsize=14)
    ax.set_ylabel(f"Income Inequality ({year})", fontsize=14)
    ax.set_title(f"Income Inequality vs. Per-Adult Income ({year})", fontsize=16)

    # Enhance legend and grid appearance
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    # Print the Pearson correlation
    if len(x) > 1:
        st.write(f"Correlation between Per-Adult Income and Income Inequality in {year}: {corr_h1:.4f}")
    else:
        st.write("Not enough data points to compute Pearson correlation.")

# Regression Analysis
st.markdown("---")
st.header("Regression Analysis")
st.markdown("""
This section shows the results of an OLS regression analysis for a specific year.
The regression model examines the relationship between per-adult income (independent variable)
and income inequality (dependent variable).
""")

# Year selection slider for regression analysis
year_reg = st.slider(
    "Select Year for Regression",
    min_value=int(df_merged['Year'].min()),
    max_value=int(df_merged['Year'].max()),
    value=2020,
    step=1
)

# Filter for the selected year
df_year_reg = df_merged[df_merged['Year'] == year_reg]

if df_year_reg.empty or len(df_year_reg) < 2:
    st.warning(f"Not enough data to run regression for year {year_reg}.")
else:
    X = sm.add_constant(df_year_reg['PerAdultIncome'])
    y = df_year_reg['IncomeInequality']
    model = sm.OLS(y, X).fit()

    # Display regression results
    st.subheader(f"OLS Regression Summary for Year {year_reg}")

    # Create a more readable summary
    results_summary = pd.DataFrame({
        'Variable': ['Constant', 'PerAdultIncome'],
        'Coefficient': [model.params[0], model.params[1]],
        'Std Error': [model.bse[0], model.bse[1]],
        't-value': [model.tvalues[0], model.tvalues[1]],
        'p-value': [model.pvalues[0], model.pvalues[1]]
    })

    st.dataframe(results_summary)

    # Display model statistics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R-squared", f"{model.rsquared:.4f}")
        st.metric("Adjusted R-squared", f"{model.rsquared_adj:.4f}")
    with col2:
        st.metric("F-statistic", f"{model.fvalue:.4f}")
        st.metric("Prob (F-statistic)", f"{model.f_pvalue:.4f}")

    # Plot the regression line
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_year_reg['PerAdultIncome'], df_year_reg['IncomeInequality'], alpha=0.7)

    # Add regression line
    x_range = np.linspace(df_year_reg['PerAdultIncome'].min(), df_year_reg['PerAdultIncome'].max(), 100)
    X_pred = sm.add_constant(x_range)
    y_pred = model.predict(X_pred)
    ax.plot(x_range, y_pred, 'r-', label='Regression Line')

    # Add confidence intervals
    from statsmodels.sandbox.regression.predstd import wls_prediction_std
    _, lower, upper = wls_prediction_std(model, X_pred)
    ax.fill_between(x_range, lower, upper, color='red', alpha=0.1, label='95% Confidence Interval')

    ax.set_xlabel('Per-Adult Income')
    ax.set_ylabel('Income Inequality')
    ax.set_title(f'Regression Analysis for Year {year_reg}')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# Boxplot Analysis
st.markdown("---")
st.header("Boxplot Analysis")
st.markdown("""
This section groups countries into income bins (Low, Medium, High) based on per-adult income
and shows the distribution of income inequality within each bin.
""")

# Year selection slider for boxplot
year_box = st.slider(
    "Select Year for Boxplot",
    min_value=int(df_merged['Year'].min()),
    max_value=int(df_merged['Year'].max()),
    value=2020,
    step=1
)

# Filter data for selected year
df_year_box = df_merged[df_merged['Year'] == year_box].copy()

if df_year_box.empty:
    st.warning(f"No data available for year {year_box}.")
else:
    # Divide countries into bins based on PerAdultIncome
    df_year_box['IncomeBin'] = pd.qcut(df_year_box['PerAdultIncome'], q=3, labels=['Low', 'Medium', 'High'])

    # Compute summary stats
    group_summary = df_year_box.groupby('IncomeBin')['IncomeInequality'].describe()
    st.subheader(f"Summary statistics of Income Inequality by Income Bin ({year_box})")
    st.dataframe(group_summary)

    # Plot boxplot
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))
    df_year_box.boxplot(column='IncomeInequality', by='IncomeBin', grid=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='red'),
                    whiskerprops=dict(color='blue'),
                    capprops=dict(color='blue'),
                    ax=ax)

    plt.xlabel("Economic Development Level (Per-Adult Income Bin)", fontsize=14)
    plt.ylabel(f"Income Inequality ({year_box})", fontsize=14)
    plt.title(f"Income Inequality for Countries with Similar Economic Development ({year_box})")
    plt.suptitle("")  # Remove automatic supertitle
    plt.tight_layout()
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Data source: World Inequality Database (WID)")
