"""
Single-file Streamlit app + helper functions to implement the 8-phase roadmap
for Global_Superstore2.csv. Modular, commented, and ready to run.

Run:
    pip install -r requirements.txt
    streamlit run Global_Superstore2_analysis.py

Requirements (suggested):
    pandas numpy scikit-learn matplotlib seaborn plotly streamlit

Notes:
- The app focuses on reproducible data cleaning, EDA, MAI calculation,
  predictive regression, clustering, a simple ROI simulator, and an
  interactive Streamlit dashboard with an "Action" field per market.
- Modify weights, thresholds, or visual styles inside the config section.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
from io import BytesIO
import base64

# --------------------------
# Configuration
# --------------------------
MAI_WEIGHTS = {
    'profit_margin': 0.30,
    'growth_rate': 0.25,
    'stability': 0.25,
    'demand_volume': 0.20
}

RANDOM_STATE = 42

# --------------------------
# Phase 1: Business Framing
# --------------------------
PHASE1_TEXT = {
    'problem_statement': (
        'Which countries, categories, and segments should we prioritize for expansion '
        'to maximize profit and stability?'
    ),
    'kpis': [
        'profit_margin', 'growth_rate', 'repeat_order_share', 'sales_volatility', 'roi_potential'
    ],
    'stakeholders': ['CEO (market entry)', 'CMO (marketing allocation)', 'CFO (ROI predictability)']
}

# --------------------------
# Utility functions
# --------------------------

def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    return df


def clean_data(df):
    # Normalize column names
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Parse dates
    for col in ['Order Date', 'Ship Date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Drop exact duplicates
    df = df.drop_duplicates()

    # Create order-level master table (1 row per Order ID)
    # Some orders contain multiple lines; aggregate to order-level for many analyses
    agg_funcs = {
        'Sales': 'sum',
        'Quantity': 'sum',
        'Discount': 'mean',
        'Profit': 'sum',
        'Shipping Cost': 'sum',
        'Order Priority': lambda x: x.mode().iloc[0] if len(x.mode())>0 else x.iloc[0],
        'Customer ID': 'first',
        'Customer Name': 'first',
        'Country': 'first',
        'Region': 'first',
        'Market': 'first',
        'Category': lambda x: ','.join(sorted(set(x.dropna())))
    }

    order_level = df.groupby('Order ID').agg(agg_funcs).reset_index()

    # Feature engineering
    order_level['profit_margin'] = order_level['Profit'] / order_level['Sales'].replace({0: np.nan})
    order_level['discount_pct'] = order_level['Discount']
    order_level['shipping_cost_pct'] = order_level['Shipping Cost'] / order_level['Sales'].replace({0: np.nan})

    # Extract date features from original rows by mapping Order ID -> earliest Order Date
    order_dates = df.groupby('Order ID')['Order Date'].min().reset_index().rename(columns={'Order Date': 'order_date'})
    order_level = order_level.merge(order_dates, on='Order ID', how='left')
    order_level['year'] = order_level['order_date'].dt.year
    order_level['month'] = order_level['order_date'].dt.month
    order_level['quarter'] = order_level['order_date'].dt.to_period('Q')

    # Clean numeric columns
    numeric_cols = ['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping Cost']
    for c in numeric_cols:
        if c in order_level.columns:
            order_level[c] = pd.to_numeric(order_level[c], errors='coerce')

    # Fill or mark missing values
    order_level = order_level.dropna(subset=['Sales', 'Profit', 'order_date'])

    return order_level

# --------------------------
# Phase 3: EDA helpers
# --------------------------

def compute_country_metrics(df):
    g = df.groupby('Country').agg(
        sales=('Sales', 'sum'),
        profit=('Profit', 'sum'),
        orders=('Order ID', 'nunique'),
    )
    g['profit_margin'] = g['profit'] / g['sales'].replace({0: np.nan})
    return g.reset_index()


def monthly_trends(df, country=None):
    temp = df.copy()
    if country:
        temp = temp[temp['Country'] == country]
    monthly = temp.groupby(pd.Grouper(key='order_date', freq='M')).agg(sales=('Sales','sum'), profit=('Profit','sum'))
    monthly['profit_margin'] = monthly['profit'] / monthly['sales'].replace({0: np.nan})
    monthly = monthly.reset_index()
    return monthly

# --------------------------
# Phase 4: MAI calculation
# --------------------------

def compute_submetrics(df, group_by=['Region','Country','Category']):
    # Group-level metrics for MAI
    g = df.groupby(group_by).agg(
        sales_sum=('Sales', 'sum'),
        profit_sum=('Profit', 'sum'),
        sales_std=('Sales', 'std'),
        sales_mean=('Sales', 'mean'),
        orders=('Order ID','nunique')
    ).reset_index()
    g['profit_margin'] = g['profit_sum'] / g['sales_sum'].replace({0: np.nan})

    # growth rate: year-on-year sales growth using mean year if available
    sales_by_year = df.copy()
    sales_by_year['year'] = sales_by_year['order_date'].dt.year
    growth = sales_by_year.groupby(group_by + ['year']).agg(year_sales=('Sales','sum')).reset_index()

    # compute simple growth: (last - first)/first
    def compute_growth(group):
        group = group.sort_values('year')
        if len(group) < 2:
            return 0.0
        first = group['year_sales'].iloc[0]
        last = group['year_sales'].iloc[-1]
        if first == 0:
            return np.nan
        return (last - first) / abs(first)

    growth_scores = growth.groupby(group_by).apply(compute_growth).reset_index().rename(columns={0:'growth_rate'})
    g = g.merge(growth_scores, on=group_by, how='left')

    # stability: inverse of coefficient of variation
    g['stability'] = g['sales_mean'] / g['sales_std'].replace({0:np.nan})
    g['stability'] = g['stability'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # demand volume: normalized sales_sum
    g['demand_volume'] = g['sales_sum']

    return g


def normalize_series(s):
    # min-max normalization handling nan
    s = s.copy().astype(float)
    s_min = np.nanmin(s)
    s_max = np.nanmax(s)
    if np.isnan(s_min) or np.isnan(s_max) or s_max == s_min:
        return pd.Series([0.0]*len(s), index=s.index)
    return (s - s_min) / (s_max - s_min)


def compute_MAI(g, weights=MAI_WEIGHTS):
    # g must contain columns: profit_margin, growth_rate, stability, demand_volume
    g = g.copy()
    g['w_profit'] = normalize_series(g['profit_margin']) * weights['profit_margin']
    g['w_growth'] = normalize_series(g['growth_rate'].fillna(0)) * weights['growth_rate']
    g['w_stability'] = normalize_series(g['stability'].fillna(0)) * weights['stability']
    g['w_volume'] = normalize_series(g['demand_volume'].fillna(0)) * weights['demand_volume']
    g['MAI_score'] = g[['w_profit','w_growth','w_stability','w_volume']].sum(axis=1)
    g = g.sort_values('MAI_score', ascending=False)
    return g

# --------------------------
# Phase 5: Predictive modeling
# --------------------------

def train_profit_model(df, features=None, test_size=0.2):
    df = df.copy()
    if features is None:
        features = ['Sales','Quantity','discount_pct','shipping_cost_pct','Region','Category']

    # dropna for target
    df = df.dropna(subset=['Profit'])

    X = df[features]
    y = df['Profit']

    # simple preprocessing: numeric and categorical
    numeric_feats = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_feats = X.select_dtypes(include=['object','category']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_feats),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_feats)
    ])

    model = Pipeline(steps=[('pre', preprocessor), ('rf', RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)

    return model, {'r2': r2, 'rmse': rmse}

# --------------------------
# Phase 6: Clustering
# --------------------------

def cluster_markets(g, n_clusters=3, method='kmeans'):
    # g is group-level metrics table. Use normalized features
    X = g[['profit_margin','orders','sales_sum']].fillna(0)
    X_scaled = (X - X.mean()) / X.std()
    if method == 'kmeans':
        km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
    else:
        gm = GaussianMixture(n_components=n_clusters, random_state=RANDOM_STATE)
        labels = gm.fit_predict(X_scaled)
    g['cluster'] = labels
    return g

# --------------------------
# Phase 7: Simple ROI simulator
# --------------------------

def roi_simulator(base_row, marketing_spend, elasticity=0.0005):
    """
    Simple simulator. Expected additional sales = marketing_spend * elasticity.
    elasticity is a tunable parameter. ROI = (incremental_profit - marketing_spend)/marketing_spend
    base_row should contain sales_sum and profit_margin
    """
    base_sales = base_row['sales_sum']
    base_margin = base_row['profit_margin'] if pd.notna(base_row.get('profit_margin')) else 0.05
    incremental_sales = marketing_spend * elasticity
    incremental_profit = incremental_sales * base_margin
    roi = (incremental_profit - marketing_spend) / (marketing_spend + 1e-9)
    return {'incremental_sales': incremental_sales, 'incremental_profit': incremental_profit, 'roi': roi}

# --------------------------
# Phase 8: Streamlit dashboard
# --------------------------

def to_excel_bytes(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return buffer.getvalue()


def app():
    st.set_page_config(layout='wide', page_title='Market Expansion Intelligence')

    st.title('Market Expansion Intelligence Dashboard')
    st.markdown('''
    Roadmap implemented: Data cleaning, EDA, MAI, predictive model, clustering, simulator, and dashboard.
    ''')

    # Upload dataset
    uploaded = st.file_uploader('Upload Global_Superstore2.csv', type=['csv'])
    if uploaded is None:
        st.info('Upload the CSV to enable analyses. Example dataset: Global_Superstore2.csv')
        st.markdown('Phase 1 problem statement:')
        st.write(PHASE1_TEXT)
        return

    df_raw = pd.read_csv(uploaded, low_memory=False)
    with st.expander('Raw data preview'):
        st.dataframe(df_raw.head())

    # Cleaning
    if st.button('Run cleaning and feature engineering'):
        df = clean_data(df_raw)
        st.success('Cleaned and aggregated to order-level.')
    else:
        st.warning('Press the button above to run cleaning.')
        return

    st.sidebar.header('Filters')
    countries = sorted(df['Country'].dropna().unique().tolist())
    sel_country = st.sidebar.selectbox('Country', options=['All'] + countries)

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    total_sales = df['Sales'].sum()
    total_profit = df['Profit'].sum()
    avg_margin = (total_profit / total_sales) if total_sales!=0 else 0
    total_orders = df['Order ID'].nunique()

    col1.metric('Total Sales', f'{total_sales:,.0f}')
    col2.metric('Total Profit', f'{total_profit:,.0f}')
    col3.metric('Avg Profit Margin', f'{avg_margin:.2%}')
    col4.metric('Orders', f'{total_orders}')

    # Compute group metrics
    g = compute_submetrics(df, group_by=['Region','Country','Category'])
    mai = compute_MAI(g)

    # Tabs
    tab_overview, tab_diag, tab_sim, tab_models = st.tabs(['Overview','Diagnostics','Simulation','Models'])

    with tab_overview:
        st.subheader('Top MAI opportunities')
        top_n = st.slider('Top N', min_value=5, max_value=50, value=10)
        st.dataframe(mai.head(top_n))

        st.subheader('Actionable table')
        actions = st.text_area('Write global action notes (saved locally while app runs)', value='')
        st.markdown('Download MAI with your notes:')
        maix = mai.copy()
        maix['notes'] = actions
        b = to_excel_bytes(maix.head(200))
        st.download_button('Download MAI Excel', data=b, file_name='MAI_with_actions.xlsx')

    with tab_diag:
        st.subheader('Geographic diagnostics')
        country_metrics = compute_country_metrics(df)
        st.bar_chart(country_metrics.set_index('Country')['profit_margin'].sort_values(ascending=False).head(20))

        st.subheader('Seasonality example')
        if sel_country != 'All':
            monthly = monthly_trends(df, country=sel_country)
            st.line_chart(monthly.set_index('order_date')['sales'])
        else:
            monthly = monthly_trends(df)
            st.line_chart(monthly.set_index('order_date')['sales'])

        st.subheader('Category-level margins')
        cat = df.groupby('Category').agg(sales=('Sales','sum'), profit=('Profit','sum')).reset_index()
        cat['profit_margin'] = cat['profit']/cat['sales'].replace({0:np.nan})
        st.dataframe(cat.sort_values('profit_margin', ascending=False))

    with tab_sim:
        st.subheader('ROI Simulator')
        selected_idx = st.selectbox('Select market row from MAI (top 100)', options=mai.head(100).index.tolist())
        base_row = mai.loc[selected_idx]
        st.write(base_row[['Region','Country','Category','sales_sum','profit_margin']])
        marketing = st.number_input('Marketing spend (local currency)', value=100000.0, step=1000.0)
        elasticity = st.slider('Elasticity (incremental sales per unit spend)', min_value=0.00001, max_value=0.01, value=0.0005, step=0.00001)
        sim = roi_simulator(base_row, marketing, elasticity=elasticity)
        st.metric('Projected ROI', f'{sim["roi"]:.2%}')
        st.write(sim)

    with tab_models:
        st.subheader('Train Profit Regression Model')
        if st.button('Train model'):
            model, metrics = train_profit_model(df)
            st.write('Model metrics', metrics)
            st.write('Use the model to predict profit for a custom input:')
            # small input form
            sales_i = st.number_input('Sales', value=1000.0)
            qty_i = st.number_input('Quantity', value=1)
            disc_i = st.number_input('Discount %', value=0.0)
            ship_pct_i = st.number_input('Shipping cost %', value=0.0)
            region_i = st.selectbox('Region', options=sorted(df['Region'].dropna().unique().tolist()))
            category_i = st.selectbox('Category', options=sorted(df['Category'].dropna().unique().tolist()))
            sample = pd.DataFrame([{
                'Sales': sales_i,
                'Quantity': qty_i,
                'discount_pct': disc_i,
                'shipping_cost_pct': ship_pct_i,
                'Region': region_i,
                'Category': category_i
            }])
            pred = model.predict(sample)[0]
            st.metric('Predicted Profit', f'{pred:.2f}')

        st.subheader('Clustering markets')
        n_clusters = st.slider('Number of clusters', min_value=2, max_value=6, value=3)
        clustered = cluster_markets(mai, n_clusters=n_clusters)
        st.dataframe(clustered.sort_values('cluster').head(200))

    st.markdown('---')
    st.caption('This app is a template. Tune preprocessing and modelling choices for production use.')


if __name__ == '__main__':
    app()
