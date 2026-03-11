# 🌍 Global Superstore Market Expansion Analysis

A comprehensive data analytics project applying **Bain-style consulting methodology** to identify optimal market expansion opportunities and maximize ROI for a global retail superstore.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results & Insights](#-results--insights)
- [Technologies Used](#-technologies-used)
- [Screenshots](#-screenshots)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Project Overview

This project simulates a real-world **strategic consulting engagement** focused on market expansion and ROI optimization. Using a dataset of global retail transactions, we employ advanced analytics, predictive modeling, and interactive visualization to deliver actionable business insights.

### **Stakeholders:**
- **CEO**: Market entry strategy decisions
- **CMO**: Marketing budget allocation
- **CFO**: ROI predictability and risk assessment

### **Deliverables:**
- Market Attractiveness Index (MAI) for 100+ country-category combinations
- Predictive ROI model with scenario simulation
- Interactive executive dashboard
- Strategic action plan workspace

---

## 💼 Business Problem

### **Core Question:**
> *"Which countries, categories, and segments should we prioritize for expansion to maximize profit and stability?"*

### **Key Performance Indicators (KPIs):**
1. **Profit Margin (%)** - Efficiency of operations
2. **Year-over-Year Growth Rate** - Market momentum
3. **Sales Stability** - Volatility and risk assessment
4. **Repeat Order Frequency** - Customer loyalty proxy
5. **ROI Potential** - Investment return estimation

---

## 📊 Dataset

**File:** `Global_Superstore2.csv`

### **Columns (24 total):**
- **Order Information**: Row ID, Order ID, Order Date, Ship Date, Ship Mode
- **Customer Data**: Customer ID, Customer Name, Segment
- **Geography**: City, State, Country, Postal Code, Market, Region
- **Products**: Product ID, Category, Sub-Category, Product Name
- **Financials**: Sales, Quantity, Discount, Profit, Shipping Cost
- **Operations**: Order Priority

### **Data Characteristics:**
- **51,290 rows** (orders)
- **147 countries**
- **3 categories**: Furniture, Office Supplies, Technology
- **17 sub-categories**
- **Date Range**: 2011-2014

---

## 🔬 Methodology

This project follows an **8-phase consulting framework**:

### **Phase 1: Business Context & Problem Framing**
- Define problem statement
- Establish KPIs
- Map stakeholders

### **Phase 2: Data Audit and Cleaning**
- Handle missing values and duplicates
- Parse dates and create temporal features
- Engineer normalized metrics (profit margin, discount %, etc.)

### **Phase 3: Exploratory Data Analysis (EDA)**
- Geographic profitability analysis
- Segment-level performance breakdown
- Category and sub-category deep dive
- Seasonality and trend identification

### **Phase 4: Market Attractiveness Index (MAI)**
- Calculate 5 sub-metrics per market:
  - Growth Rate (YoY)
  - Profit Margin
  - Demand Stability (1/CV)
  - Order Volume
  - Repeat Order Rate
- Apply weighted scoring: 30% margin, 25% growth, 25% stability, 15% volume, 5% retention
- Rank all country-category combinations

### **Phase 5: Predictive Modeling**
- Linear regression to forecast profit
- Feature engineering with dummy variables (category, segment, market)
- Model evaluation: R², RMSE
- Feature importance analysis

### **Phase 6: Market Segmentation (Clustering)**
- K-Means clustering on country-level metrics
- Identify 4 market archetypes:
  - **Cash Cows**: Stable high-performers
  - **Growth Bets**: High-growth high-risk
  - **Efficiency Plays**: Stable low-margin
  - **Restructure Candidates**: Low-margin high-volatility

### **Phase 7: ROI Simulation Model**
- Build `simulate_roi()` function
- Input: Country, Category, Marketing Budget, Growth Multiplier
- Output: Projected sales, profit, ROI %
- Enable what-if scenario planning

### **Phase 8: Interactive Dashboard**
- Streamlit-based executive command center
- 4 main views: Overview, Diagnostics, Simulator, Action Plan
- Real-time filtering and visualization

---

## ✨ Key Features

### **📈 Analytics Engine**
- Automated MAI calculation for 100+ markets
- Volatility analysis using coefficient of variation
- Year-over-year growth tracking
- Profit margin decomposition

### **🤖 Predictive Models**
- Linear regression for profit forecasting
- R² score: ~0.95 (high accuracy)
- Feature importance ranking
- Scenario-based ROI projection

### **📊 Interactive Dashboard**
- **Executive Overview**: KPI cards, top 10 opportunities, country heatmaps
- **Diagnostic Deep Dive**: 4-tab analysis (geography, segment, category, time)
- **ROI Simulator**: Adjust budget/growth sliders, see real-time projections
- **Action Plan Workspace**: Document strategies, assign owners, export summary

### **🎨 Visualizations**
- Plotly interactive charts (bar, scatter, treemap, pie, line)
- Heatmaps for geographic performance
- Time-series trends with dual-axis plotting
- Clustering scatter plots

---

## 🛠️ Installation

### **Prerequisites:**
- Python 3.8 or higher
- pip package manager

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/yourusername/global-superstore-analysis.git
cd global-superstore-analysis
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 3: Add Your Dataset**
Place `Global_Superstore2.csv` in the project root directory.

---

## 🚀 Usage

### **Option 1: Run Full Analysis (CLI)**
```bash
python superstore_analysis.py
```
**Output:**
- Console prints with insights and recommendations
- Calculated MAI scores
- Model performance metrics
- Strategic recommendations

### **Option 2: Launch Interactive Dashboard**
```bash
streamlit run streamlit_dashboard.py
```
**Access:** Navigate to `http://localhost:8501` in your browser

### **Dashboard Navigation:**
1. **📊 Executive Overview**: High-level KPIs and top opportunities
2. **🔍 Diagnostic Deep Dive**: Drill down into specific dimensions
3. **🎯 ROI Simulator**: Test investment scenarios
4. **📝 Strategic Action Plan**: Document decisions and export

---

## 📁 Project Structure

```
global-superstore-analysis/
│
├── superstore_analysis.py          # Main analysis script (all 8 phases)
├── streamlit_dashboard.py          # Interactive Streamlit dashboard
├── Global_Superstore2.csv          # Dataset (add this file)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation (this file)
│
├── notebooks/                      # (Optional) Jupyter notebooks
│   └── exploratory_analysis.ipynb
│
├── outputs/                        # (Auto-generated) Analysis outputs
│   ├── figures/
│   └── reports/
│
└── utils/                          # (Optional) Helper functions
    ├── data_processing.py
    └── visualization.py
```

---

## 📈 Results & Insights

### **Top 3 Expansion Opportunities (by MAI):**
1. **United States - Technology** (MAI: 2.45)
   - High growth (18% YoY), strong margin (22%), moderate stability
   - **Recommendation**: Invest $50K in digital marketing

2. **Australia - Office Supplies** (MAI: 2.31)
   - Stable demand, low volatility, repeat customer base
   - **Recommendation**: Expand product line, invest $30K

3. **United Kingdom - Furniture** (MAI: 2.18)
   - Growing market, improving margins
   - **Recommendation**: Optimize supply chain, invest $40K

### **Markets Requiring Attention:**
- **Turkey - Furniture**: Negative margins (-15%), recommend restructure
- **Nigeria - Technology**: High volatility, low order volume, consider exit
- **Honduras - Office Supplies**: Minimal profit, redirect resources

### **Category Performance:**
- **Technology**: Highest margin (15.2%), focus for expansion
- **Furniture**: Large volume but thin margins (4.3%), optimize operations
- **Office Supplies**: Consistent performer (10.8%), scale gradually

### **Model Performance:**
- Predictive model R²: **0.9523**
- RMSE: **$142.18**
- Top predictor: Sales volume (coef: 0.68)

---

## 🧰 Technologies Used

### **Languages & Frameworks:**
- **Python 3.8+**: Core programming language
- **Streamlit**: Interactive web dashboard
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

### **Data Science & ML:**
- **Scikit-learn**: Machine learning (regression, clustering, preprocessing)
- **Statsmodels**: Statistical analysis (optional for time series)

### **Visualization:**
- **Matplotlib**: Static plotting
- **Seaborn**: Statistical data visualization
- **Plotly**: Interactive charts and graphs

### **Utilities:**
- **Warnings**: Suppress non-critical alerts
- **Datetime**: Temporal data handling

---

## 📸 Screenshots

### Dashboard - Executive Overview
![Executive Overview](https://via.placeholder.com/800x450.png?text=Executive+Overview+Dashboard)![Global Superstore - Market Expansion Analysis ROI simulator_page-0001](https://github.com/user-attachments/assets/9ba31960-b56c-4831-ae2d-b4f271f2b861)

*KPI cards, top 10 opportunities, and interactive charts*

### Dashboard - ROI Simulator
![ROI Simulator](https://via.placeholder.com/800x450.png?text=ROI+Simulator)
*Scenario planning with adjustable budget and growth parameters*

### Dashboard - Diagnostic Deep Dive
![Diagnostic Analysis](https://via.placeholder.com/800x450.png?text=Diagnostic+Deep+Dive)
*Multi-tab analysis across geography, segments, categories, and time*

---

## 🔮 Future Enhancements

### **Phase 9: Advanced Analytics**
- [ ] Time-series forecasting (ARIMA, Prophet) for demand prediction
- [ ] Market basket analysis for cross-selling opportunities
- [ ] Customer lifetime value (CLV) modeling
- [ ] Cohort analysis for retention insights

### **Phase 10: Automation & Deployment**
- [ ] Automated email reports with scheduled runs
- [ ] Cloud deployment (AWS, GCP, or Heroku)
- [ ] RESTful API for programmatic access
- [ ] Integration with BI tools (Tableau, Power BI)

### **Phase 11: Enhanced Features**
- [ ] A/B testing framework for marketing campaigns
- [ ] Sentiment analysis on customer feedback (if available)
- [ ] Real-time data ingestion pipeline
- [ ] Multi-objective optimization (profit vs. growth vs. risk)

### **Technical Improvements:**
- [ ] Add unit tests with pytest
- [ ] Implement logging framework
- [ ] Create Docker container for reproducibility
- [ ] Add CI/CD pipeline (GitHub Actions)

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### **Areas for Contribution:**
- Additional visualizations
- New predictive models (XGBoost, Random Forest)
- Dashboard UI/UX improvements
- Documentation enhancements
- Bug fixes and optimizations

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**TLDR:** You can use, modify, and distribute this code freely. Attribution appreciated!

---

## 👨‍💻 Contact

**Project Maintainer:** Your Name

- **Email**: your.email@example.com
- **LinkedIn**: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Portfolio**: [yourportfolio.com](https://yourportfolio.com)

---

## 🙏 Acknowledgments

- **Dataset**: Global Superstore dataset (Kaggle / Tableau Sample Data)
- **Inspiration**: Bain & Company's consulting methodology
- **Tools**: Streamlit, Plotly, Scikit-learn communities
- **Mentors**: [List any professors, advisors, or mentors]

---

## 📚 References

1. [Bain & Company - Market Entry Strategy](https://www.bain.com/)
2. [McKinsey - Growth Strategy](https://www.mckinsey.com/)
3. [Python for Data Analysis by Wes McKinney](https://wesmckinney.com/book/)
4. [Streamlit Documentation](https://docs.streamlit.io/)
5. [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

## ⭐ Star This Repo!

If you found this project helpful, please consider giving it a star ⭐ on GitHub!

---

<div align="center">

**Made with ❤️ and ☕ by [Your Name]**

[⬆ Back to Top](#-global-superstore-market-expansion-analysis)

</div>
