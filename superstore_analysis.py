import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ========================================
# PHASE 1: BUSINESS CONTEXT & PROBLEM FRAMING
# ========================================

print("="*80)
print("GLOBAL SUPERSTORE - MARKET EXPANSION & ROI OPTIMIZATION ANALYSIS")
print("="*80)
print("\nPROBLEM STATEMENT:")
print("Which countries, categories, and segments should we prioritize for")
print("expansion to maximize profit and stability?")
print("\nKEY PERFORMANCE INDICATORS (KPIs):")
print("1. Profit Margin (%)")
print("2. Year-over-Year Growth Rate")
print("3. Repeat Order Frequency")
print("4. Sales Stability (Low Volatility)")
print("5. ROI Potential")
print("\nSTAKEHOLDER MAPPING:")
print("- CEO: Market Entry Strategy")
print("- CMO: Marketing Budget Allocation")
print("- CFO: ROI Predictability & Risk Assessment")
print("="*80)

# ========================================
# PHASE 2: DATA AUDIT AND CLEANING
# ========================================

print("\n\nPHASE 2: DATA AUDIT AND CLEANING")
print("-"*80)

# Load the dataset
df = pd.read_csv('Global_Superstore2.csv', encoding='latin-1')

print(f"Original dataset shape: {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

# Drop duplicates if any
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape}")

# Handle missing values - dropping rows with missing critical fields
print("\nHandling missing values...")
df = df.dropna(subset=['Order ID', 'Sales', 'Profit', 'Quantity'])

# Convert date columns to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y', errors='coerce')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d-%m-%Y', errors='coerce')

# Create time-based features
df['Year'] = df['Order Date'].dt.year
df['Quarter'] = df['Order Date'].dt.quarter
df['Month'] = df['Order Date'].dt.month
df['Day_of_Week'] = df['Order Date'].dt.dayofweek

# Create normalized metrics
df['Profit_Margin'] = (df['Profit'] / df['Sales']) * 100
df['Discount_Pct'] = df['Discount'] * 100
df['Shipping_Cost_Pct'] = (df['Shipping Cost'] / df['Sales']) * 100
df['Profit_per_Order'] = df['Profit']
df['Sales_per_Order'] = df['Sales']

# Clean up infinite and NaN values that might have been created
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=['Profit_Margin', 'Discount_Pct'])

print(f"\nCleaned dataset shape: {df.shape}")
print(f"Date range: {df['Order Date'].min()} to {df['Order Date'].max()}")
print(f"Years covered: {sorted(df['Year'].unique())}")

# ========================================
# PHASE 3: EXPLORATORY DATA ANALYSIS (EDA)
# ========================================

print("\n\nPHASE 3: EXPLORATORY DATA ANALYSIS")
print("-"*80)

# 3.1 Geographic Analysis
print("\n3.1 GEOGRAPHIC ANALYSIS")
print("-"*40)

geo_analysis = df.groupby('Country').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'count',
    'Profit_Margin': 'mean'
}).round(2)

geo_analysis.columns = ['Total_Sales', 'Total_Profit', 'Order_Count', 'Avg_Profit_Margin']
geo_analysis = geo_analysis.sort_values('Total_Profit', ascending=False)

print("\nTop 10 Countries by Profit:")
print(geo_analysis.head(10))

print("\nBottom 10 Countries by Profit (Potential Issues):")
print(geo_analysis.tail(10))

# 3.2 Segment Analysis
print("\n3.2 SEGMENT ANALYSIS")
print("-"*40)

segment_analysis = df.groupby('Segment').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'count',
    'Profit_Margin': 'mean'
}).round(2)

segment_analysis.columns = ['Total_Sales', 'Total_Profit', 'Order_Count', 'Avg_Profit_Margin']
print(segment_analysis)

# 3.3 Category Analysis
print("\n3.3 CATEGORY ANALYSIS")
print("-"*40)

category_analysis = df.groupby('Category').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'count',
    'Profit_Margin': 'mean',
    'Discount_Pct': 'mean'
}).round(2)

category_analysis.columns = ['Total_Sales', 'Total_Profit', 'Order_Count', 'Avg_Profit_Margin', 'Avg_Discount']
print(category_analysis)

# Sub-category analysis
subcategory_analysis = df.groupby('Sub-Category').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Profit_Margin': 'mean'
}).round(2)

subcategory_analysis.columns = ['Total_Sales', 'Total_Profit', 'Avg_Profit_Margin']
subcategory_analysis = subcategory_analysis.sort_values('Total_Profit', ascending=False)

print("\nTop 10 Sub-Categories by Profit:")
print(subcategory_analysis.head(10))

print("\nWorst 10 Sub-Categories by Profit:")
print(subcategory_analysis.tail(10))

# 3.4 Seasonality Analysis
print("\n3.4 SEASONALITY ANALYSIS")
print("-"*40)

monthly_trends = df.groupby(['Year', 'Month']).agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).round(2)

print("\nMonthly trends (last 12 months):")
print(monthly_trends.tail(12))

quarterly_trends = df.groupby(['Year', 'Quarter']).agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Profit_Margin': 'mean'
}).round(2)

print("\nQuarterly trends:")
print(quarterly_trends)

# Calculate volatility (coefficient of variation)
monthly_sales = df.groupby(['Country', 'Year', 'Month'])['Sales'].sum().reset_index()
volatility = monthly_sales.groupby('Country')['Sales'].agg(['std', 'mean'])
volatility['CV'] = (volatility['std'] / volatility['mean']) * 100
volatility = volatility.sort_values('CV')

print("\nTop 10 Most Stable Countries (Low Volatility):")
print(volatility.head(10))

# ========================================
# PHASE 4: MARKET ATTRACTIVENESS INDEX (MAI)
# ========================================

print("\n\nPHASE 4: MARKET ATTRACTIVENESS INDEX (MAI)")
print("-"*80)

# Calculate MAI for Country-Category combinations
mai_data = []

for country in df['Country'].unique():
    for category in df['Category'].unique():
        subset = df[(df['Country'] == country) & (df['Category'] == category)]
        
        if len(subset) < 5:  # Skip if too few orders
            continue
        
        # 1. Calculate Growth Rate (YoY)
        yearly_sales = subset.groupby('Year')['Sales'].sum()
        if len(yearly_sales) >= 2:
            growth_rate = ((yearly_sales.iloc[-1] - yearly_sales.iloc[0]) / yearly_sales.iloc[0]) * 100
        else:
            growth_rate = 0
        
        # 2. Calculate Profit Margin
        profit_margin = (subset['Profit'].sum() / subset['Sales'].sum()) * 100
        
        # 3. Calculate Stability (inverse of CV)
        monthly_sales_subset = subset.groupby(['Year', 'Month'])['Sales'].sum()
        if len(monthly_sales_subset) > 1:
            cv = (monthly_sales_subset.std() / monthly_sales_subset.mean()) * 100
            stability = 1 / (1 + cv)  # Normalize
        else:
            stability = 0
        
        # 4. Calculate Demand Volume (normalized order count)
        demand_volume = len(subset)
        
        # 5. Calculate Repeat Order Proxy (unique customers vs total orders)
        unique_customers = subset['Customer ID'].nunique()
        repeat_order_rate = (1 - unique_customers / len(subset)) if len(subset) > 0 else 0
        
        mai_data.append({
            'Country': country,
            'Category': category,
            'Growth_Rate': growth_rate,
            'Profit_Margin': profit_margin,
            'Stability': stability,
            'Demand_Volume': demand_volume,
            'Repeat_Order_Rate': repeat_order_rate,
            'Total_Sales': subset['Sales'].sum(),
            'Total_Profit': subset['Profit'].sum()
        })

mai_df = pd.DataFrame(mai_data)

# Normalize the metrics (0-1 scale)
scaler = StandardScaler()

metrics_to_normalize = ['Growth_Rate', 'Profit_Margin', 'Stability', 'Demand_Volume', 'Repeat_Order_Rate']

# Handle any inf or nan values
mai_df = mai_df.replace([np.inf, -np.inf], np.nan)
mai_df = mai_df.dropna(subset=metrics_to_normalize)

# Normalize
mai_df[metrics_to_normalize] = scaler.fit_transform(mai_df[metrics_to_normalize])

# Calculate MAI with weights
weights = {
    'Profit_Margin': 0.30,
    'Growth_Rate': 0.25,
    'Stability': 0.25,
    'Demand_Volume': 0.15,
    'Repeat_Order_Rate': 0.05
}

mai_df['MAI_Score'] = (
    mai_df['Profit_Margin'] * weights['Profit_Margin'] +
    mai_df['Growth_Rate'] * weights['Growth_Rate'] +
    mai_df['Stability'] * weights['Stability'] +
    mai_df['Demand_Volume'] * weights['Demand_Volume'] +
    mai_df['Repeat_Order_Rate'] * weights['Repeat_Order_Rate']
)

# Rank the opportunities
mai_df = mai_df.sort_values('MAI_Score', ascending=False)

print("\nTOP 15 MARKET EXPANSION OPPORTUNITIES (by MAI Score):")
print("="*80)
top_15 = mai_df.head(15)[['Country', 'Category', 'MAI_Score', 'Total_Sales', 'Total_Profit']].reset_index(drop=True)
top_15.index = top_15.index + 1
print(top_15)

print("\nBOTTOM 10 MARKETS (by MAI Score) - POTENTIAL EXIT/RESTRUCTURE:")
print("="*80)
bottom_10 = mai_df.tail(10)[['Country', 'Category', 'MAI_Score', 'Total_Sales', 'Total_Profit']].reset_index(drop=True)
print(bottom_10)

# ========================================
# PHASE 5: PREDICTIVE MODELING FOR FUTURE ROI
# ========================================

print("\n\nPHASE 5: PREDICTIVE MODELING FOR FUTURE ROI")
print("-"*80)

# Prepare data for regression
modeling_df = df.copy()

# Create dummy variables for categorical features
modeling_df = pd.get_dummies(modeling_df, columns=['Category', 'Segment', 'Market'], drop_first=True)

# Select features for the model
feature_cols = ['Sales', 'Quantity', 'Discount', 'Shipping Cost'] + \
               [col for col in modeling_df.columns if col.startswith('Category_') or 
                col.startswith('Segment_') or col.startswith('Market_')]

X = modeling_df[feature_cols]
y = modeling_df['Profit']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\nMODEL PERFORMANCE:")
print(f"Training R² Score: {train_r2:.4f}")
print(f"Testing R² Score: {test_r2:.4f}")
print(f"Training RMSE: ${train_rmse:.2f}")
print(f"Testing RMSE: ${test_rmse:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)

print("\nTOP 10 MOST IMPORTANT FEATURES (by coefficient):")
print(feature_importance.head(10))

print("\nBOTTOM 10 FEATURES (negative impact on profit):")
print(feature_importance.tail(10))

# ========================================
# PHASE 6: SEGMENTATION USING CLUSTERING
# ========================================

print("\n\nPHASE 6: MARKET SEGMENTATION USING CLUSTERING")
print("-"*80)

# Prepare data for clustering at country level
cluster_data = df.groupby('Country').agg({
    'Profit': 'mean',
    'Discount': 'mean',
    'Order ID': 'count',
    'Sales': ['std', 'mean']
}).reset_index()

cluster_data.columns = ['Country', 'Avg_Profit', 'Avg_Discount', 'Order_Frequency', 'Sales_Std', 'Sales_Mean']
cluster_data['Volatility'] = cluster_data['Sales_Std'] / cluster_data['Sales_Mean']
cluster_data = cluster_data.dropna()

# Select features for clustering
cluster_features = ['Avg_Profit', 'Avg_Discount', 'Order_Frequency', 'Volatility']
X_cluster = cluster_data[cluster_features]

# Standardize
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

# Determine optimal number of clusters using elbow method
inertias = []
K_range = range(2, 8)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster_scaled)
    inertias.append(kmeans.inertia_)

# Use 4 clusters (good balance)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_data['Cluster'] = kmeans.fit_predict(X_cluster_scaled)

# Analyze clusters
print(f"\nIDENTIFIED {optimal_k} MARKET ARCHETYPES:")
print("="*80)

for i in range(optimal_k):
    cluster_subset = cluster_data[cluster_data['Cluster'] == i]
    print(f"\nCLUSTER {i+1}: (n={len(cluster_subset)} countries)")
    print(f"  Avg Profit: ${cluster_subset['Avg_Profit'].mean():.2f}")
    print(f"  Avg Discount: {cluster_subset['Avg_Discount'].mean()*100:.1f}%")
    print(f"  Avg Order Frequency: {cluster_subset['Order_Frequency'].mean():.0f} orders")
    print(f"  Avg Volatility: {cluster_subset['Volatility'].mean():.2f}")
    print(f"  Countries: {', '.join(cluster_subset['Country'].head(5).tolist())}")
    
    # Assign business narrative
    avg_profit = cluster_subset['Avg_Profit'].mean()
    avg_volatility = cluster_subset['Volatility'].mean()
    
    if avg_profit > cluster_data['Avg_Profit'].median() and avg_volatility < cluster_data['Volatility'].median():
        archetype = "STABLE HIGH-PERFORMERS (Cash Cows)"
    elif avg_profit > cluster_data['Avg_Profit'].median() and avg_volatility > cluster_data['Volatility'].median():
        archetype = "HIGH-GROWTH HIGH-RISK (Growth Bets)"
    elif avg_profit < cluster_data['Avg_Profit'].median() and avg_volatility < cluster_data['Volatility'].median():
        archetype = "STABLE LOW-MARGIN (Efficiency Plays)"
    else:
        archetype = "LOW-MARGIN HIGH-VOLATILITY (Restructure/Exit)"
    
    print(f"  >>> STRATEGIC ARCHETYPE: {archetype}")

# ========================================
# PHASE 7: SIMULATION MODEL - BUDGET vs ROI
# ========================================

print("\n\nPHASE 7: ROI SIMULATION MODEL")
print("-"*80)

def simulate_roi(country, category, marketing_spend, growth_multiplier=1.2):
    """
    Simulate ROI based on historical performance and marketing investment
    
    Parameters:
    - country: Target country
    - category: Target category
    - marketing_spend: Marketing budget ($)
    - growth_multiplier: Expected growth from marketing (default 1.2 = 20% growth)
    
    Returns:
    - Dictionary with projected metrics
    """
    
    # Get historical data
    historical = df[(df['Country'] == country) & (df['Category'] == category)]
    
    if len(historical) == 0:
        return {
            'Status': 'No historical data',
            'Projected_Sales': 0,
            'Projected_Profit': 0,
            'ROI': 0
        }
    
    # Calculate baseline metrics
    baseline_sales = historical['Sales'].sum()
    baseline_profit = historical['Profit'].sum()
    baseline_margin = (baseline_profit / baseline_sales) * 100 if baseline_sales > 0 else 0
    
    # Project sales with growth
    projected_sales = baseline_sales * growth_multiplier
    
    # Project profit (accounting for marketing spend)
    projected_profit = (projected_sales * baseline_margin / 100) - marketing_spend
    
    # Calculate ROI
    roi = ((projected_profit - baseline_profit) / marketing_spend * 100) if marketing_spend > 0 else 0
    
    return {
        'Country': country,
        'Category': category,
        'Baseline_Sales': baseline_sales,
        'Baseline_Profit': baseline_profit,
        'Baseline_Margin': baseline_margin,
        'Marketing_Spend': marketing_spend,
        'Projected_Sales': projected_sales,
        'Projected_Profit': projected_profit,
        'Net_Profit_Increase': projected_profit - baseline_profit,
        'ROI_Percent': roi
    }

# Example simulations
print("\nEXAMPLE ROI SIMULATIONS:")
print("="*80)

test_scenarios = [
    ('United States', 'Technology', 50000),
    ('United Kingdom', 'Furniture', 30000),
    ('Australia', 'Office Supplies', 20000),
]

for country, category, budget in test_scenarios:
    result = simulate_roi(country, category, budget)
    print(f"\nScenario: {country} - {category} (Budget: ${budget:,})")
    print(f"  Baseline Sales: ${result['Baseline_Sales']:,.2f}")
    print(f"  Projected Sales: ${result['Projected_Sales']:,.2f}")
    print(f"  Baseline Profit: ${result['Baseline_Profit']:,.2f}")
    print(f"  Projected Profit: ${result['Projected_Profit']:,.2f}")
    print(f"  Net Profit Increase: ${result['Net_Profit_Increase']:,.2f}")
    print(f"  ROI: {result['ROI_Percent']:.1f}%")

# ========================================
# PHASE 8: SUMMARY & RECOMMENDATIONS
# ========================================

print("\n\n" + "="*80)
print("EXECUTIVE SUMMARY & STRATEGIC RECOMMENDATIONS")
print("="*80)

print("\n1. TOP EXPANSION OPPORTUNITIES:")
print("-"*40)
top_3_mai = mai_df.head(3)
for idx, row in top_3_mai.iterrows():
    print(f"\n  {row['Country']} - {row['Category']}")
    print(f"    MAI Score: {row['MAI_Score']:.4f}")
    print(f"    Total Profit: ${row['Total_Profit']:,.2f}")
    print(f"    Action: INVEST AGGRESSIVELY")

print("\n\n2. MARKETS REQUIRING ATTENTION:")
print("-"*40)
bottom_3_mai = mai_df.tail(3)
for idx, row in bottom_3_mai.iterrows():
    print(f"\n  {row['Country']} - {row['Category']}")
    print(f"    MAI Score: {row['MAI_Score']:.4f}")
    print(f"    Total Profit: ${row['Total_Profit']:,.2f}")
    print(f"    Action: RESTRUCTURE OR EXIT")

print("\n\n3. CATEGORY INSIGHTS:")
print("-"*40)
for idx, row in category_analysis.iterrows():
    print(f"\n  {idx}:")
    print(f"    Total Profit: ${row['Total_Profit']:,.2f}")
    print(f"    Avg Margin: {row['Avg_Profit_Margin']:.2f}%")
    if row['Avg_Profit_Margin'] > 10:
        print(f"    Recommendation: HIGH-MARGIN FOCUS")
    elif row['Avg_Profit_Margin'] < 0:
        print(f"    Recommendation: REVIEW PRICING & COSTS")
    else:
        print(f"    Recommendation: OPTIMIZE OPERATIONS")

print("\n\n4. SEGMENT STRATEGY:")
print("-"*40)
for idx, row in segment_analysis.iterrows():
    print(f"\n  {idx}:")
    print(f"    Total Profit: ${row['Total_Profit']:,.2f}")
    print(f"    Order Count: {row['Order_Count']:,.0f}")
    print(f"    Avg Margin: {row['Avg_Profit_Margin']:.2f}%")

print("\n\n5. MODEL INSIGHTS:")
print("-"*40)
print(f"  Predictive Model R²: {test_r2:.4f}")
print(f"  This model can explain {test_r2*100:.1f}% of profit variance")
print(f"  Key Drivers: Sales volume, Category mix, Market geography")

print("\n\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nNext Steps:")
print("1. Review top 15 MAI opportunities for detailed market entry planning")
print("2. Conduct sensitivity analysis on ROI projections")
print("3. Develop go-to-market strategy for high-MAI Country-Category pairs")
print("4. Set up monitoring dashboard for KPI tracking")
print("5. Prepare executive presentation with visualizations")
print("\nNote: For interactive dashboard, run the Streamlit app (separate file)")
print("="*80)