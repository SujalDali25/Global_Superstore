import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Global Superstore - Market Expansion Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
  .action-box {
        background-color: #d4c45c; /* darker yellow */
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #ffc107;
        color: white; /* ensures text remains readable */
    }
    .recommendation {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_and_clean_data():
    """Load and clean the dataset"""
    df = pd.read_csv('Global_Superstore2.csv', encoding='latin-1')
    
    # Data cleaning
    df = df.drop_duplicates()
    df = df.dropna(subset=['Order ID', 'Sales', 'Profit', 'Quantity'])
    
    # Date conversion
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y', errors='coerce')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d-%m-%Y', errors='coerce')
    
    # Create features
    df['Year'] = df['Order Date'].dt.year
    df['Quarter'] = df['Order Date'].dt.quarter
    df['Month'] = df['Order Date'].dt.month
    df['Profit_Margin'] = (df['Profit'] / df['Sales']) * 100
    df['Discount_Pct'] = df['Discount'] * 100
    df['Shipping_Cost_Pct'] = (df['Shipping Cost'] / df['Sales']) * 100
    
    # Clean infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['Profit_Margin', 'Discount_Pct'])
    
    return df

@st.cache_data
def calculate_mai(df):
    """Calculate Market Attractiveness Index"""
    mai_data = []
    
    for country in df['Country'].unique():
        for category in df['Category'].unique():
            subset = df[(df['Country'] == country) & (df['Category'] == category)]
            
            if len(subset) < 5:
                continue
            
            # Growth rate
            yearly_sales = subset.groupby('Year')['Sales'].sum()
            if len(yearly_sales) >= 2:
                growth_rate = ((yearly_sales.iloc[-1] - yearly_sales.iloc[0]) / yearly_sales.iloc[0]) * 100
            else:
                growth_rate = 0
            
            # Profit margin
            profit_margin = (subset['Profit'].sum() / subset['Sales'].sum()) * 100
            
            # Stability
            monthly_sales_subset = subset.groupby(['Year', 'Month'])['Sales'].sum()
            if len(monthly_sales_subset) > 1:
                cv = (monthly_sales_subset.std() / monthly_sales_subset.mean()) * 100
                stability = 1 / (1 + cv)
            else:
                stability = 0
            
            # Demand volume
            demand_volume = len(subset)
            
            # Repeat order rate
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
                'Total_Profit': subset['Profit'].sum(),
                'Order_Count': len(subset)
            })
    
    mai_df = pd.DataFrame(mai_data)
    
    # Normalize and calculate MAI
    scaler = StandardScaler()
    metrics = ['Growth_Rate', 'Profit_Margin', 'Stability', 'Demand_Volume', 'Repeat_Order_Rate']
    
    mai_df = mai_df.replace([np.inf, -np.inf], np.nan)
    mai_df = mai_df.dropna(subset=metrics)
    
    mai_df[metrics] = scaler.fit_transform(mai_df[metrics])
    
    weights = {'Profit_Margin': 0.30, 'Growth_Rate': 0.25, 'Stability': 0.25, 
               'Demand_Volume': 0.15, 'Repeat_Order_Rate': 0.05}
    
    mai_df['MAI_Score'] = sum(mai_df[metric] * weight for metric, weight in weights.items())
    mai_df = mai_df.sort_values('MAI_Score', ascending=False)
    
    return mai_df

def simulate_roi(df, country, category, marketing_spend, growth_multiplier=1.2):
    """Simulate ROI for a given market and investment"""
    historical = df[(df['Country'] == country) & (df['Category'] == category)]
    
    if len(historical) == 0:
        return None
    
    baseline_sales = historical['Sales'].sum()
    baseline_profit = historical['Profit'].sum()
    baseline_margin = (baseline_profit / baseline_sales) * 100 if baseline_sales > 0 else 0
    
    projected_sales = baseline_sales * growth_multiplier
    projected_profit = (projected_sales * baseline_margin / 100) - marketing_spend
    roi = ((projected_profit - baseline_profit) / marketing_spend * 100) if marketing_spend > 0 else 0
    
    return {
        'Baseline_Sales': baseline_sales,
        'Baseline_Profit': baseline_profit,
        'Baseline_Margin': baseline_margin,
        'Projected_Sales': projected_sales,
        'Projected_Profit': projected_profit,
        'Net_Profit_Increase': projected_profit - baseline_profit,
        'ROI_Percent': roi
    }

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🌍 Global Superstore - Market Expansion Command Center</div>', 
                unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading data...'):
        df = load_and_clean_data()
        mai_df = calculate_mai(df)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select View", 
                            ["📊 Executive Overview", 
                             "🔍 Diagnostic Deep Dive", 
                             "🎯 ROI Simulator",
                             "📝 Strategic Action Plan"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters")
    
    years = sorted(df['Year'].unique())
    selected_years = st.sidebar.multiselect("Year", years, default=years)
    
    markets = sorted(df['Market'].unique())
    selected_markets = st.sidebar.multiselect("Market", markets, default=markets)
    
    # Filter data
    filtered_df = df[(df['Year'].isin(selected_years)) & (df['Market'].isin(selected_markets))]
    
    # ==========================================
    # PAGE 1: EXECUTIVE OVERVIEW
    # ==========================================
    if page == "📊 Executive Overview":
        st.header("Executive Dashboard")
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sales = filtered_df['Sales'].sum()
            st.metric("Total Sales", f"${total_sales:,.0f}")
        
        with col2:
            total_profit = filtered_df['Profit'].sum()
            st.metric("Total Profit", f"${total_profit:,.0f}")
        
        with col3:
            avg_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
            st.metric("Profit Margin", f"{avg_margin:.2f}%")
        
        with col4:
            total_orders = len(filtered_df)
            st.metric("Total Orders", f"{total_orders:,}")
        
        st.markdown("---")
        
        # Two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 Countries by Profit")
            country_profit = filtered_df.groupby('Country')['Profit'].sum().sort_values(ascending=False).head(10)
            fig = px.bar(country_profit, x=country_profit.values, y=country_profit.index, 
                        orientation='h', labels={'x': 'Profit ($)', 'y': 'Country'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Category Performance")
            category_data = filtered_df.groupby('Category').agg({
                'Sales': 'sum',
                'Profit': 'sum'
            }).reset_index()
            fig = px.scatter(category_data, x='Sales', y='Profit', size='Profit', 
                           color='Category', hover_data=['Category'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # MAI Top Opportunities
        st.markdown("---")
        st.subheader("🎯 Top 10 Market Expansion Opportunities (MAI Score)")
        
        top_10_mai = mai_df.head(10)[['Country', 'Category', 'MAI_Score', 'Total_Sales', 
                                       'Total_Profit', 'Order_Count']].copy()
        top_10_mai['Total_Sales'] = top_10_mai['Total_Sales'].apply(lambda x: f"${x:,.0f}")
        top_10_mai['Total_Profit'] = top_10_mai['Total_Profit'].apply(lambda x: f"${x:,.0f}")
        top_10_mai['MAI_Score'] = top_10_mai['MAI_Score'].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(top_10_mai, use_container_width=True, hide_index=True)
        
        # Insights
        st.markdown("---")
        st.subheader("💡 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("**📈 Growth Markets**")
            top_country = mai_df.iloc[0]
            st.write(f"**{top_country['Country']} - {top_country['Category']}** shows the highest MAI score")
            st.write(f"Total Profit: ${top_country['Total_Profit']:,.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("**⚠️ Attention Needed**")
            bottom_profit = filtered_df.groupby('Country')['Profit'].sum().sort_values().head(1)
            st.write(f"**{bottom_profit.index[0]}** has lowest profit: ${bottom_profit.values[0]:,.2f}")
            st.write("Consider restructuring or exit strategy")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # ==========================================
    # PAGE 2: DIAGNOSTIC DEEP DIVE
    # ==========================================
    elif page == "🔍 Diagnostic Deep Dive":
        st.header("Diagnostic Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📍 Geographic", "👥 Segment", "📦 Category", "📅 Temporal"])
        
        with tab1:
            st.subheader("Geographic Performance Analysis")
            
            geo_data = filtered_df.groupby('Country').agg({
                'Sales': 'sum',
                'Profit': 'sum',
                'Order ID': 'count',
                'Profit_Margin': 'mean'
            }).reset_index()
            geo_data.columns = ['Country', 'Sales', 'Profit', 'Orders', 'Avg_Margin']
            geo_data = geo_data.sort_values('Profit', ascending=False).head(20)
            
            # Heatmap
            fig = px.treemap(geo_data, path=['Country'], values='Sales', 
                           color='Profit', hover_data=['Orders', 'Avg_Margin'],
                           color_continuous_scale='RdYlGn',
                           title='Sales Distribution by Country (sized by Sales, colored by Profit)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot
            fig2 = px.scatter(geo_data, x='Sales', y='Profit', size='Orders', 
                            color='Avg_Margin', hover_data=['Country'],
                            title='Sales vs Profit by Country',
                            labels={'Sales': 'Total Sales ($)', 'Profit': 'Total Profit ($)'})
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            st.subheader("Customer Segment Analysis")
            
            segment_data = filtered_df.groupby('Segment').agg({
                'Sales': 'sum',
                'Profit': 'sum',
                'Order ID': 'count',
                'Profit_Margin': 'mean',
                'Discount_Pct': 'mean'
            }).reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(segment_data, values='Sales', names='Segment', 
                           title='Sales Distribution by Segment')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(segment_data, x='Segment', y='Profit', 
                           color='Segment', title='Profit by Segment')
                st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(segment_data, use_container_width=True, hide_index=True)
        
        with tab3:
            st.subheader("Category & Sub-Category Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                category_data = filtered_df.groupby('Category').agg({
                    'Profit': 'sum',
                    'Sales': 'sum',
                    'Profit_Margin': 'mean'
                }).reset_index()
                
                fig = px.bar(category_data, x='Category', y='Profit', 
                           color='Profit_Margin', title='Category Profitability',
                           color_continuous_scale='RdYlGn')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                subcategory_data = filtered_df.groupby('Sub-Category').agg({
                    'Profit': 'sum'
                }).reset_index().sort_values('Profit', ascending=False).head(10)
                
                fig = px.bar(subcategory_data, x='Profit', y='Sub-Category', 
                           orientation='h', title='Top 10 Sub-Categories by Profit')
                st.plotly_chart(fig, use_container_width=True)
            
            # Worst performers
            st.subheader("⚠️ Worst Performing Sub-Categories")
            worst_subcategories = filtered_df.groupby('Sub-Category')['Profit'].sum().sort_values().head(10)
            fig = px.bar(worst_subcategories, x=worst_subcategories.values, 
                       y=worst_subcategories.index, orientation='h',
                       title='Bottom 10 Sub-Categories (Consider Review)',
                       color=worst_subcategories.values, color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Temporal Trends & Seasonality")
            
            # Monthly trends
            monthly_data = filtered_df.groupby(['Year', 'Month']).agg({
                'Sales': 'sum',
                'Profit': 'sum'
            }).reset_index()
            monthly_data['Date'] = pd.to_datetime(monthly_data[['Year', 'Month']].assign(Day=1))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly_data['Date'], y=monthly_data['Sales'],
                                    mode='lines+markers', name='Sales'))
            fig.add_trace(go.Scatter(x=monthly_data['Date'], y=monthly_data['Profit'],
                                    mode='lines+markers', name='Profit'))
            fig.update_layout(title='Monthly Sales & Profit Trends', 
                            xaxis_title='Date', yaxis_title='Amount ($)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Quarterly performance
            quarterly_data = filtered_df.groupby(['Year', 'Quarter']).agg({
                'Sales': 'sum',
                'Profit': 'sum',
                'Profit_Margin': 'mean'
            }).reset_index()
            quarterly_data['Period'] = quarterly_data['Year'].astype(str) + '-Q' + quarterly_data['Quarter'].astype(str)
            
            fig = px.bar(quarterly_data, x='Period', y='Profit', 
                       color='Profit_Margin', title='Quarterly Profit Performance',
                       color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================
    # PAGE 3: ROI SIMULATOR
    # ==========================================
    elif page == "🎯 ROI Simulator":
        st.header("ROI Simulation & Investment Calculator")
        
        st.markdown("""
        <div class="insight-box">
        <strong>🎯 Purpose:</strong> Simulate expected ROI for different market-category combinations 
        with varying marketing investments.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Simulation Parameters")
            
            selected_country = st.selectbox("Select Country", sorted(df['Country'].unique()))
            selected_category = st.selectbox("Select Category", sorted(df['Category'].unique()))
            
            marketing_budget = st.slider("Marketing Budget ($)", 
                                        min_value=1000, max_value=100000, 
                                        value=25000, step=1000)
            
            growth_rate = st.slider("Expected Growth Multiplier", 
                                   min_value=1.0, max_value=2.0, 
                                   value=1.2, step=0.05)
            
            simulate_button = st.button("🚀 Run Simulation", type="primary")
        
        with col2:
            st.subheader("Simulation Results")
            
            if simulate_button:
                result = simulate_roi(df, selected_country, selected_category, 
                                    marketing_budget, growth_rate)
                
                if result:
                    st.markdown('<div class="recommendation">', unsafe_allow_html=True)
                    st.markdown(f"### {selected_country} - {selected_category}")
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.metric("Baseline Sales", f"${result['Baseline_Sales']:,.2f}")
                        st.metric("Projected Sales", f"${result['Projected_Sales']:,.2f}")
                        st.metric("Baseline Profit", f"${result['Baseline_Profit']:,.2f}")
                    
                    with col_b:
                        st.metric("Projected Profit", f"${result['Projected_Profit']:,.2f}")
                        st.metric("Net Profit Increase", f"${result['Net_Profit_Increase']:,.2f}")
                        st.metric("ROI", f"{result['ROI_Percent']:.1f}%", 
                                delta=f"{result['ROI_Percent']:.1f}%")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Recommendation
                    if result['ROI_Percent'] > 50:
                        st.success("✅ **STRONG INVESTMENT OPPORTUNITY** - High expected ROI")
                    elif result['ROI_Percent'] > 20:
                        st.info("💼 **MODERATE OPPORTUNITY** - Reasonable ROI expected")
                    else:
                        st.warning("⚠️ **PROCEED WITH CAUTION** - Lower ROI expected")
                else:
                    st.error("No historical data available for this combination")
        
        # Comparison table
        st.markdown("---")
        st.subheader("Compare Multiple Scenarios")
        
        if st.checkbox("Show Top 10 Opportunities Simulation"):
            comparison_data = []
            top_10 = mai_df.head(10)
            
            for _, row in top_10.iterrows():
                result = simulate_roi(df, row['Country'], row['Category'], 30000, 1.2)
                if result:
                    comparison_data.append({
                        'Country': row['Country'],
                        'Category': row['Category'],
                        'MAI Score': row['MAI_Score'],
                        'Baseline Profit': result['Baseline_Profit'],
                        'Projected Profit': result['Projected_Profit'],
                        'ROI (%)': result['ROI_Percent']
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('ROI (%)', ascending=False)
            
            st.dataframe(comparison_df.style.format({
                'MAI Score': '{:.4f}',
                'Baseline Profit': '${:,.2f}',
                'Projected Profit': '${:,.2f}',
                'ROI (%)': '{:.1f}%'
            }), use_container_width=True)
    
    # ==========================================
    # PAGE 4: STRATEGIC ACTION PLAN
    # ==========================================
    elif page == "📝 Strategic Action Plan":
        st.header("Strategic Action Planning Workspace")
        
        st.markdown("""
        <div class="action-box">
        <strong>📋 Purpose:</strong> Document strategic actions and decisions for each market opportunity.
        Use this space to capture your recommendations for executive presentation.
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize session state for actions
        if 'actions' not in st.session_state:
            st.session_state.actions = {}
        
        # Top opportunities
        st.subheader("Top 10 Expansion Opportunities")
        
        top_10 = mai_df.head(10)
        
        for idx, row in top_10.iterrows():
            with st.expander(f"#{top_10.index.get_loc(idx) + 1}: {row['Country']} - {row['Category']} (MAI: {row['MAI_Score']:.4f})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Key Metrics:**")
                    st.write(f"- Total Sales: ${row['Total_Sales']:,.2f}")
                    st.write(f"- Total Profit: ${row['Total_Profit']:,.2f}")
                    st.write(f"- Order Count: {row['Order_Count']:,}")
                    
                    # Action input
                    key = f"{row['Country']}_{row['Category']}"
                    action_text = st.text_area(
                        "Strategic Action / Recommendation:",
                        value=st.session_state.actions.get(key, ""),
                        key=f"action_{idx}",
                        height=100
                    )
                    st.session_state.actions[key] = action_text
                
                with col2:
                    priority = st.selectbox(
                        "Priority Level:",
                        ["🔴 High", "🟡 Medium", "🟢 Low"],
                        key=f"priority_{idx}"
                    )
                    
                    timeline = st.selectbox(
                        "Timeline:",
                        ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024", "2025"],
                        key=f"timeline_{idx}"
                    )
                    
                    owner = st.text_input(
                        "Action Owner:",
                        key=f"owner_{idx}"
                    )
        
        # Summary section
        st.markdown("---")
        st.subheader("📊 Action Plan Summary")
        
        if st.button("📄 Generate Executive Summary"):
            st.markdown('<div class="recommendation">', unsafe_allow_html=True)
            st.markdown("### Executive Action Plan Summary")
            st.markdown(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            st.markdown(f"**Total Opportunities Identified:** {len(top_10)}")
            
            actions_documented = sum(1 for v in st.session_state.actions.values() if v.strip())
            st.markdown(f"**Actions Documented:** {actions_documented}/{len(top_10)}")
            
            st.markdown("#### Top 3 Priority Markets:")
            for i in range(min(3, len(top_10))):
                row = top_10.iloc[i]
                key = f"{row['Country']}_{row['Category']}"
                action = st.session_state.actions.get(key, "Not documented")
                st.markdown(f"{i+1}. **{row['Country']} - {row['Category']}**")
                st.markdown(f"   - MAI Score: {row['MAI_Score']:.4f}")
                st.markdown(f"   - Action: {action[:100]}..." if len(action) > 100 else f"   - Action: {action}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.info("💡 Tip: Screenshot this summary or export to PDF for your presentation")
        
        # Export functionality note
        st.markdown("---")
        st.markdown("""
        <div class="insight-box">
        <strong>💾 Export Options:</strong><br>
        - Use browser's Print to PDF feature to save this page<br>
        - Screenshot the summary section for quick sharing<br>
        - Copy action text for inclusion in presentation decks
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()