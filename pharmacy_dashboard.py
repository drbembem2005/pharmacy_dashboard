# Standard library imports
import calendar
import datetime
from datetime import timedelta
import io
import streamlit as st

# Data manipulation imports
import pandas as pd
import numpy as np

# Visualization imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ML imports
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Custom color scheme
COLOR_PALETTE = {
    'primary': '#2E86C1',
    'secondary': '#28B463',
    'accent': '#E74C3C',
    'neutral': '#566573',
    'background': '#F8F9F9'
}

# Set page configuration
st.set_page_config(
    page_title="Pharmacy Analytics Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #F8F9F9;
    }
    .chart-container {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    h1, h2, h3 {
        color: #2E86C1;
    }
    </style>
""", unsafe_allow_html=True)

# Display main header with custom styling
st.markdown("<h1 style='text-align: center; color: #2E86C1; padding: 20px;'>Pharmacy Analytics Dashboard</h1>", unsafe_allow_html=True)

# Load Data
@st.cache_data(ttl=3600)
def load_data():
    file_path = "pharmacy.xlsx"
    xls = pd.ExcelFile(file_path)
    
    # Load all sheets
    lists_df = pd.read_excel(xls, sheet_name="lists")
    daily_income_df = pd.read_excel(xls, sheet_name="Daily Income")
    inventory_purchases_df = pd.read_excel(xls, sheet_name="Inventory Purchases")
    expenses_df = pd.read_excel(xls, sheet_name="Expenses")
    
    # Clean up column names
    lists_df.columns = lists_df.columns.str.strip().str.lower()
    inventory_purchases_df.columns = inventory_purchases_df.columns.str.strip()
    expenses_df.columns = expenses_df.columns.str.strip()
    
    # Data preprocessing
    for df, date_col in [(daily_income_df, "Date"), 
                        (inventory_purchases_df, "Date"), 
                        (expenses_df, "Date")]:
        # Convert dates
        df["date"] = pd.to_datetime(df[date_col], errors='coerce')
        # Fill numeric columns with 0
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
    
    # Calculate derived columns for Daily Income
    daily_income_df["net_income"] = daily_income_df["Total"] - daily_income_df["5%"]
    
    # Remove rows with null dates
    daily_income_df = daily_income_df.dropna(subset=["Date"])
    inventory_purchases_df = inventory_purchases_df.dropna(subset=["Date"])
    expenses_df = expenses_df.dropna(subset=["Date"])
    
    return {
        "lists": lists_df,
        "daily_income": daily_income_df,
        "inventory_purchases": inventory_purchases_df,
        "expenses": expenses_df
    }

# Load data
try:
    data = load_data()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# ====================== SIDEBAR FILTERS ======================
with st.sidebar:
    st.markdown("### 🔍 Dashboard Filters")
    st.divider()
    
    # Date range filter
    st.markdown("#### 📅 Date Range")
    date_preset = st.selectbox(
        "Quick Select",
        ["Custom", "Last 7 Days", "Last 30 Days", "Last 90 Days", "Year to Date", "All Time"],
        key="date_preset"
    )
    
    if date_preset == "Custom":
        date_range = st.date_input(
            "Select Date Range",
            value=(data["daily_income"]["date"].min().date(), 
                   data["daily_income"]["date"].max().date()),
            min_value=data["daily_income"]["date"].min().date(),
            max_value=data["daily_income"]["date"].max().date()
        )
        start_date, end_date = date_range if len(date_range) == 2 else (
            data["daily_income"]["date"].min().date(),
            data["daily_income"]["date"].max().date()
        )
    else:
        end_date = data["daily_income"]["date"].max().date()
        start_date = {
            "Last 7 Days": end_date - timedelta(days=7),
            "Last 30 Days": end_date - timedelta(days=30),
            "Last 90 Days": end_date - timedelta(days=90),
            "Year to Date": datetime.date(end_date.year, 1, 1),
            "All Time": data["daily_income"]["date"].min().date()
        }[date_preset]

    # Filter data based on date range
    filtered_data = {
        "daily_income": data["daily_income"][
            (data["daily_income"]["date"].dt.date >= start_date) & 
            (data["daily_income"]["date"].dt.date <= end_date)
        ].copy(),
        "inventory": data["inventory_purchases"][
            (data["inventory_purchases"]["date"].dt.date >= start_date) & 
            (data["inventory_purchases"]["date"].dt.date <= end_date)
        ].copy(),
        "expenses": data["expenses"][
            (data["expenses"]["date"].dt.date >= start_date) & 
            (data["expenses"]["date"].dt.date <= end_date)
        ].copy()
    }
    
    # Additional filters
    st.markdown("#### 📦 Inventory")
    inventory_types = sorted(data["inventory_purchases"]["Inventory Type"].unique())
    selected_types = st.multiselect("Inventory Types", inventory_types, default=inventory_types)
    
    st.markdown("#### 🏢 Companies")
    companies = sorted(data["inventory_purchases"]["Invoice Company"].unique())
    selected_companies = st.multiselect("Companies", companies, default=companies)
    
    st.markdown("#### 💰 Expenses")
    expense_types = sorted(data["expenses"]["Expense Type"].unique())
    # Set default to all expense types
    selected_expenses = st.multiselect("Expense Types", expense_types, default=expense_types)
    
    # Apply additional filters
    filtered_data["inventory"] = filtered_data["inventory"][
        filtered_data["inventory"]["Inventory Type"].isin(selected_types) &
        filtered_data["inventory"]["Invoice Company"].isin(selected_companies)
    ]
    # Only filter expenses if specific types are selected
    if selected_expenses:
        filtered_data["expenses"] = filtered_data["expenses"][
            filtered_data["expenses"]["Expense Type"].isin(selected_expenses)
        ]

# ====================== MAIN DASHBOARD CONTENT ======================

# Main Dashboard Tabs
tab_overview, tab_revenue, tab_inventory, tab_expenses, tab_analytics, tab_ml, tab_search = st.tabs([
    "📊 Overview",
    "💰 Revenue",
    "📦 Inventory",
    "💸 Expenses",
    "📈 Analytics",
    "🤖 ML & Predictions",
    "🔍 Search & Reports"
])

# Overview Tab
with tab_overview:
    st.markdown("### 📊 Key Performance Indicators")
    kpi_cols = st.columns(5)

    # Calculate KPIs
    total_income = filtered_data["daily_income"]["Total"].sum()
    total_expenses = filtered_data["expenses"]["Expense Amount"].sum()
    total_purchases = filtered_data["inventory"]["Invoice Amount"].sum()
    net_profit = total_income - total_expenses - total_purchases
    money_deficit = filtered_data["daily_income"]["deficit"].sum()

    # Display KPIs
    with kpi_cols[0]:
        st.metric("Total Revenue", f"EGP {total_income:,.2f}", 
                  delta=f"{(total_income/total_income*100 if total_income > 0 else 0):.1f}% of Total")

    with kpi_cols[1]:
        st.metric("Total Expenses", f"EGP {total_expenses:,.2f}",
                  delta=f"{(total_expenses/total_income*100 if total_income > 0 else 0):.1f}% of Revenue")

    with kpi_cols[2]:
        st.metric("Total Purchases", f"EGP {total_purchases:,.2f}",
                  delta=f"{(total_purchases/total_income*100 if total_income > 0 else 0):.1f}% of Revenue")

    with kpi_cols[3]:
        st.metric("Net Profit", f"EGP {net_profit:,.2f}",
                  delta=f"{(net_profit/total_income*100 if total_income > 0 else 0):.1f}% Margin")

    with kpi_cols[4]:
        st.metric("Money Deficit", f"EGP {money_deficit:,.2f}",
                  delta=f"{(money_deficit/total_income*100 if total_income > 0 else 0):.1f}% of Revenue")

    # Financial Health Indicators
    st.markdown("### 📈 Financial Health Indicators")
    health_cols = st.columns(3)

    with health_cols[0]:
        profit_margin = (net_profit / total_income * 100) if total_income > 0 else 0
        fig_margin = go.Figure(go.Indicator(
            mode="gauge+number",
            value=profit_margin,
            title={"text": "Profit Margin (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": COLOR_PALETTE["primary"]},
                "steps": [
                    {"range": [0, 30], "color": "lightgray"},
                    {"range": [30, 70], "color": "gray"}
                ]
            }
        ))
        st.plotly_chart(fig_margin, use_container_width=True)

    with health_cols[1]:
        expense_ratio = (total_expenses / total_income * 100) if total_income > 0 else 0
        fig_expense_ratio = go.Figure(go.Indicator(
            mode="gauge+number",
            value=expense_ratio,
            title={"text": "Expense Ratio (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": COLOR_PALETTE["secondary"]},
                "steps": [
                    {"range": [0, 30], "color": "lightgray"},
                    {"range": [30, 70], "color": "gray"}
                ]
            }
        ))
        st.plotly_chart(fig_expense_ratio, use_container_width=True)

    with health_cols[2]:
        inventory_turnover = total_income / total_purchases if total_purchases > 0 else 0
        fig_turnover = go.Figure(go.Indicator(
            mode="gauge+number",
            value=inventory_turnover,
            title={"text": "Inventory Turnover Ratio"},
            gauge={
                "axis": {"range": [0, 10]},
                "bar": {"color": COLOR_PALETTE["accent"]},
                "steps": [
                    {"range": [0, 3], "color": "lightgray"},
                    {"range": [3, 7], "color": "gray"}
                ]
            }
        ))
        st.plotly_chart(fig_turnover, use_container_width=True)

    # Additional KPIs
    st.markdown("### 📊 Additional Performance Metrics")
    more_kpi_cols = st.columns(5)
    
    # Calculate more KPIs
    daily_profit = net_profit / len(filtered_data["daily_income"]) if filtered_data["daily_income"].shape[0] > 0 else 0
    expense_per_revenue = total_expenses / total_income if total_income > 0 else 0
    cash_ratio = filtered_data["daily_income"]["cash"].sum() / total_income if total_income > 0 else 0
    visa_ratio = filtered_data["daily_income"]["visa"].sum() / total_income if total_income > 0 else 0
    due_ratio = filtered_data["daily_income"]["due amount"].sum() / total_income if total_income > 0 else 0

    with more_kpi_cols[0]:
        st.metric("Daily Average Profit", f"EGP {daily_profit:,.2f}")
    
    with more_kpi_cols[1]:
        st.metric("Expense to Revenue", f"{expense_per_revenue:.2%}")
    
    with more_kpi_cols[2]:
        st.metric("Cash Ratio", f"{cash_ratio:.2%}")
    
    with more_kpi_cols[3]:
        st.metric("Visa Ratio", f"{visa_ratio:.2%}")
    
    with more_kpi_cols[4]:
        st.metric("Due Amount Ratio", f"{due_ratio:.2%}")

    # Add Rolling Averages
    st.markdown("### 📈 Rolling Averages")
    rolling_cols = st.columns(2)

    with rolling_cols[0]:
        # 7-day rolling average
        daily_income_df = filtered_data["daily_income"].set_index("date")
        rolling_7 = daily_income_df["Total"].rolling(7).mean()
        
        fig_rolling_7 = go.Figure()
        fig_rolling_7.add_trace(go.Scatter(
            x=rolling_7.index,
            y=rolling_7.values,
            name="7-day Average",
            line=dict(color=COLOR_PALETTE["primary"])
        ))
        fig_rolling_7.update_layout(
            title="7-Day Rolling Average Revenue",
            xaxis_title="Date",
            yaxis_title="Amount (EGP)",
            template="plotly_white"
        )
        st.plotly_chart(fig_rolling_7, use_container_width=True)

    with rolling_cols[1]:
        # 30-day rolling average
        rolling_30 = daily_income_df["Total"].rolling(30).mean()
        
        fig_rolling_30 = go.Figure()
        fig_rolling_30.add_trace(go.Scatter(
            x=rolling_30.index,
            y=rolling_30.values,
            name="30-day Average",
            line=dict(color=COLOR_PALETTE["secondary"])
        ))
        fig_rolling_30.update_layout(
            title="30-Day Rolling Average Revenue",
            xaxis_title="Date",
            yaxis_title="Amount (EGP)",
            template="plotly_white"
        )
        st.plotly_chart(fig_rolling_30, use_container_width=True)

    # Add Comparative Analysis
    st.markdown("### 📊 Comparative Analysis")
    
    # Create weekly comparison
    weekly_comparison = filtered_data["daily_income"].copy()
    weekly_comparison["week"] = weekly_comparison["date"].dt.isocalendar().week
    weekly_comparison["year"] = weekly_comparison["date"].dt.year
    
    weekly_totals = weekly_comparison.groupby(["year", "week"])["Total"].sum().reset_index()
    
    fig_weekly_comp = go.Figure()
    
    for year in weekly_totals["year"].unique():
        year_data = weekly_totals[weekly_totals["year"] == year]
        fig_weekly_comp.add_trace(go.Scatter(
            x=year_data["week"],
            y=year_data["Total"],
            name=str(year),
            mode="lines+markers"
        ))
    
    fig_weekly_comp.update_layout(
        title="Weekly Revenue Comparison by Year",
        xaxis_title="Week Number",
        yaxis_title="Revenue (EGP)",
        template="plotly_white"
    )
    st.plotly_chart(fig_weekly_comp, use_container_width=True)

# Revenue Tab
with tab_revenue:
    st.markdown("### 💰 Revenue Analysis")
    
    # Revenue KPIs
    st.markdown("#### 📊 Revenue KPIs")
    revenue_kpi_cols = st.columns(4)
    
    with revenue_kpi_cols[0]:
        total_revenue = filtered_data["daily_income"]["Total"].sum()
        st.metric("Total Revenue", f"EGP {total_revenue:,.2f}")
    
    with revenue_kpi_cols[1]:
        avg_daily_revenue = filtered_data["daily_income"]["Total"].mean()
        st.metric("Average Daily Revenue", f"EGP {avg_daily_revenue:,.2f}")
    
    with revenue_kpi_cols[2]:
        cash_percentage = (filtered_data["daily_income"]["cash"].sum() / total_revenue * 100) if total_revenue > 0 else 0
        st.metric("Cash Revenue %", f"{cash_percentage:.1f}%")
    
    with revenue_kpi_cols[3]:
        due_amount = filtered_data["daily_income"]["due amount"].sum()
        st.metric("Due Amount", f"EGP {due_amount:,.2f}")
    
    # Daily Revenue Trend
    daily_revenue = filtered_data["daily_income"].groupby("date").agg({
        "Total": "sum",
        "Gross Income_sys": "sum",
        "net_income": "sum"
    }).reset_index()

    fig_revenue = go.Figure()
    fig_revenue.add_trace(go.Scatter(
        x=daily_revenue["date"],
        y=daily_revenue["Total"],
        name="Actual Revenue",
        line=dict(color=COLOR_PALETTE["primary"])
    ))
    fig_revenue.add_trace(go.Scatter(
        x=daily_revenue["date"],
        y=daily_revenue["Gross Income_sys"],
        name="System Revenue",
        line=dict(color=COLOR_PALETTE["secondary"])
    ))
    fig_revenue.add_trace(go.Scatter(
        x=daily_revenue["date"],
        y=daily_revenue["net_income"],
        name="Net Revenue",
        line=dict(color=COLOR_PALETTE["accent"])
    ))
    fig_revenue.update_layout(
        title="Daily Revenue Trend",
        xaxis_title="Date",
        yaxis_title="Amount (EGP)",
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig_revenue, use_container_width=True)

    # Payment Method Distribution
    payment_data = filtered_data["daily_income"][["cash", "visa", "due amount"]].sum()
    fig_payment = go.Figure(data=[go.Pie(
        labels=payment_data.index,
        values=payment_data.values,
        hole=.4,
        marker=dict(colors=[COLOR_PALETTE["primary"], 
                          COLOR_PALETTE["secondary"],
                          COLOR_PALETTE["accent"]])
    )])
    fig_payment.update_layout(
        title="Payment Method Distribution",
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig_payment, use_container_width=True)

    # Add Revenue Breakdown Analysis
    st.markdown("### 💹 Revenue Breakdown Analysis")
    
    # Revenue by Day of Week
    daily_revenue = filtered_data["daily_income"].copy()
    daily_revenue["day_of_week"] = daily_revenue["date"].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_revenue["day_of_week"] = pd.Categorical(daily_revenue["day_of_week"], categories=day_order, ordered=True)
    
    dow_stats = daily_revenue.groupby("day_of_week").agg({
        "Total": ["mean", "std", "count"]
    }).round(2)
    
    dow_stats.columns = ["Average", "Std Dev", "Count"]
    dow_stats = dow_stats.reset_index()
    
    fig_dow = go.Figure()
    fig_dow.add_trace(go.Bar(
        x=dow_stats["day_of_week"],
        y=dow_stats["Average"],
        error_y=dict(type="data", array=dow_stats["Std Dev"]),
        name="Daily Average"
    ))
    
    fig_dow.update_layout(
        title="Revenue by Day of Week (with Standard Deviation)",
        xaxis_title="Day of Week",
        yaxis_title="Average Revenue (EGP)",
        template="plotly_white"
    )
    st.plotly_chart(fig_dow, use_container_width=True)

# Inventory Tab
with tab_inventory:
    st.markdown("### 📦 Inventory Analysis")
    
    # Inventory KPIs
    st.markdown("#### 📊 Inventory KPIs")
    inventory_kpi_cols = st.columns(4)
    
    with inventory_kpi_cols[0]:
        total_purchases = filtered_data["inventory"]["Invoice Amount"].sum()
        st.metric("Total Purchases", f"EGP {total_purchases:,.2f}")
    
    with inventory_kpi_cols[1]:
        avg_purchase = filtered_data["inventory"]["Invoice Amount"].mean()
        st.metric("Average Purchase", f"EGP {avg_purchase:,.2f}")
    
    with inventory_kpi_cols[2]:
        num_purchases = len(filtered_data["inventory"])
        st.metric("Number of Purchases", f"{num_purchases:,}")
    
    with inventory_kpi_cols[3]:
        inventory_types = filtered_data["inventory"]["Inventory Type"].nunique()
        st.metric("Inventory Types", f"{inventory_types}")
    
    # Inventory by Type
    inventory_by_type = filtered_data["inventory"].groupby("Inventory Type")["Invoice Amount"].sum()
    fig_inventory = go.Figure(data=[go.Bar(
        x=inventory_by_type.index,
        y=inventory_by_type.values,
        marker_color=COLOR_PALETTE["primary"]
    )])
    fig_inventory.update_layout(
        title="Purchases by Inventory Type",
        xaxis_title="Inventory Type",
        yaxis_title="Amount (EGP)",
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig_inventory, use_container_width=True)

    # Monthly Inventory Trend
    monthly_inventory = filtered_data["inventory"].groupby(
        filtered_data["inventory"]["date"].dt.to_period("M")
    )["Invoice Amount"].sum()

    fig_monthly_inv = go.Figure(data=[go.Scatter(
        x=monthly_inventory.index.astype(str),
        y=monthly_inventory.values,
        mode='lines+markers',
        line=dict(color=COLOR_PALETTE["primary"])
    )])
    fig_monthly_inv.update_layout(
        title="Monthly Inventory Purchases Trend",
        xaxis_title="Month",
        yaxis_title="Amount (EGP)",
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig_monthly_inv, use_container_width=True)

# Expenses Tab
with tab_expenses:
    st.markdown("### 💸 Expense Analysis")
    
    # Expense KPIs
    st.markdown("#### 📊 Expense KPIs")
    expense_kpi_cols = st.columns(4)
    
    with expense_kpi_cols[0]:
        total_expenses = filtered_data["expenses"]["Expense Amount"].sum()
        st.metric("Total Expenses", f"EGP {total_expenses:,.2f}")
    
    with expense_kpi_cols[1]:
        avg_expense = filtered_data["expenses"]["Expense Amount"].mean()
        st.metric("Average Expense", f"EGP {avg_expense:,.2f}")
    
    with expense_kpi_cols[2]:
        expense_types = filtered_data["expenses"]["Expense Type"].nunique()
        st.metric("Expense Types", f"{expense_types}")
    
    with expense_kpi_cols[3]:
        expense_ratio = (total_expenses / total_income * 100) if total_income > 0 else 0
        st.metric("Expense Ratio", f"{expense_ratio:.1f}%")
    
    # Expense Distribution
    expense_dist = filtered_data["expenses"].groupby("Expense Type")["Expense Amount"].sum().sort_values(ascending=True)
    colors = px.colors.qualitative.Set3[:len(expense_dist)]

    fig_expense = go.Figure(data=[go.Pie(
        labels=expense_dist.index,
        values=expense_dist.values,
        hole=.4,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='outside'
    )])
    fig_expense.update_layout(
        title="Expense Distribution by Type",
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400
    )
    st.plotly_chart(fig_expense, use_container_width=True)

    # Monthly Expense Trend by Type
    monthly_expenses = filtered_data["expenses"].groupby([
        filtered_data["expenses"]["date"].dt.to_period("M"),
        "Expense Type"
    ])["Expense Amount"].sum().reset_index()
    monthly_expenses["date"] = monthly_expenses["date"].astype(str)

    fig_monthly_exp = px.bar(
        monthly_expenses,
        x="date",
        y="Expense Amount",
        color="Expense Type",
        title="Monthly Expense Trend by Type",
        labels={"date": "Month", "Expense Amount": "Amount (EGP)"},
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_monthly_exp.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="Amount (EGP)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400
    )
    st.plotly_chart(fig_monthly_exp, use_container_width=True)

# Analytics Tab
with tab_analytics:
    st.markdown("### 📈 Advanced Analytics")
    
    # Analytics KPIs
    st.markdown("#### 📊 Analytics KPIs")
    analytics_kpi_cols = st.columns(4)
    
    with analytics_kpi_cols[0]:
        profit_margin = (net_profit / total_income * 100) if total_income > 0 else 0
        st.metric("Profit Margin", f"{profit_margin:.1f}%")
    
    with analytics_kpi_cols[1]:
        inventory_turnover = total_income / total_purchases if total_purchases > 0 else 0
        st.metric("Inventory Turnover", f"{inventory_turnover:.2f}x")
    
    with analytics_kpi_cols[2]:
        avg_transaction = filtered_data["daily_income"]["Total"].mean()
        st.metric("Avg Transaction", f"EGP {avg_transaction:,.2f}")
    
    with analytics_kpi_cols[3]:
        data_points = len(filtered_data["daily_income"]) + len(filtered_data["inventory"]) + len(filtered_data["expenses"])
        st.metric("Total Data Points", f"{data_points:,}")
    
    # Data Tables
    st.markdown("#### 📋 Detailed Data Tables")
    tab1, tab2, tab3 = st.tabs(["Daily Income", "Inventory", "Expenses"])
    
    with tab1:
        st.dataframe(
            filtered_data["daily_income"].sort_values("date", ascending=False),
            use_container_width=True
        )
    
    with tab2:
        st.dataframe(
            filtered_data["inventory"].sort_values("date", ascending=False),
            use_container_width=True
        )
    
    with tab3:
        st.dataframe(
            filtered_data["expenses"].sort_values("date", ascending=False),
            use_container_width=True
        )

# ML & Predictions Tab
with tab_ml:
    st.markdown("### 🤖 Machine Learning & Predictions")
    
    # Revenue Forecasting
    st.markdown("#### 📈 Revenue Forecasting")
    
    # Prepare data for Prophet
    daily_revenue = filtered_data["daily_income"].groupby("date")["Total"].sum().reset_index()
    daily_revenue.columns = ['ds', 'y']  # Prophet requires these column names
    
    # Create and fit Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05
    )
    
    # Fit the model
    with st.spinner("Training forecasting model..."):
        model.fit(daily_revenue)
        
        # Create future dates for prediction
        future_dates = model.make_future_dataframe(periods=30)
        forecast = model.predict(future_dates)
        
        # Plot the forecast
        fig_forecast = go.Figure()
        
        # Add actual values
        fig_forecast.add_trace(go.Scatter(
            x=daily_revenue['ds'],
            y=daily_revenue['y'],
            name='Actual Revenue',
            line=dict(color=COLOR_PALETTE["primary"])
        ))
        
        # Add forecast
        fig_forecast.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            name='Forecast',
            line=dict(color=COLOR_PALETTE["secondary"])
        ))
        
        # Add confidence interval
        fig_forecast.add_trace(go.Scatter(
            x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
            y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(46, 134, 193, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))
        
        fig_forecast.update_layout(
            title="30-Day Revenue Forecast",
            xaxis_title="Date",
            yaxis_title="Revenue (EGP)",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Display forecast metrics
        st.markdown("#### 📊 Forecast Metrics")
        metric_cols = st.columns(3)
        
        with metric_cols[0]:
            last_actual = daily_revenue['y'].iloc[-1]
            forecast_value = forecast['yhat'].iloc[-1]
            growth_rate = ((forecast_value - last_actual) / last_actual) * 100
            st.metric(
                "30-Day Growth Rate",
                f"{growth_rate:.1f}%",
                delta=f"EGP {forecast_value - last_actual:,.2f}"
            )
        
        with metric_cols[1]:
            weekly_avg = daily_revenue['y'].tail(7).mean()
            forecast_weekly_avg = forecast['yhat'].tail(7).mean()
            st.metric(
                "Weekly Average Forecast",
                f"EGP {forecast_weekly_avg:,.2f}",
                delta=f"{((forecast_weekly_avg - weekly_avg) / weekly_avg * 100):.1f}%"
            )
        
        with metric_cols[2]:
            confidence = (forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]) / forecast['yhat'].iloc[-1] * 100
            st.metric(
                "Forecast Confidence",
                f"{100 - confidence:.1f}%",
                delta="Interval Width"
            )
    
    # Trend Analysis
    st.markdown("#### 📊 Trend Analysis")
    trend_cols = st.columns(2)
    
    with trend_cols[0]:
        # Weekly patterns
        daily_revenue['weekday'] = daily_revenue['ds'].dt.day_name()
        weekly_patterns = daily_revenue.groupby('weekday')['y'].mean()
        
        fig_weekly = go.Figure(data=[go.Bar(
            x=weekly_patterns.index,
            y=weekly_patterns.values,
            marker_color=COLOR_PALETTE["primary"]
        )])
        fig_weekly.update_layout(
            title="Average Revenue by Day of Week",
            xaxis_title="Day of Week",
            yaxis_title="Average Revenue (EGP)",
            template="plotly_white",
            height=300
        )
        st.plotly_chart(fig_weekly, use_container_width=True)
    
    with trend_cols[1]:
        # Monthly patterns
        daily_revenue['month'] = daily_revenue['ds'].dt.month_name()
        monthly_patterns = daily_revenue.groupby('month')['y'].mean()
        
        fig_monthly = go.Figure(data=[go.Bar(
            x=monthly_patterns.index,
            y=monthly_patterns.values,
            marker_color=COLOR_PALETTE["secondary"]
        )])
        fig_monthly.update_layout(
            title="Average Revenue by Month",
            xaxis_title="Month",
            yaxis_title="Average Revenue (EGP)",
            template="plotly_white",
            height=300
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Anomaly Detection
    st.markdown("#### 🔍 Anomaly Detection")
    
    # Calculate z-scores for anomaly detection
    daily_revenue['z_score'] = (daily_revenue['y'] - daily_revenue['y'].mean()) / daily_revenue['y'].std()
    anomalies = daily_revenue[abs(daily_revenue['z_score']) > 2]
    
    fig_anomalies = go.Figure()
    fig_anomalies.add_trace(go.Scatter(
        x=daily_revenue['ds'],
        y=daily_revenue['y'],
        name='Normal',
        line=dict(color=COLOR_PALETTE["primary"])
    ))
    fig_anomalies.add_trace(go.Scatter(
        x=anomalies['ds'],
        y=anomalies['y'],
        mode='markers',
        name='Anomalies',
        marker=dict(color=COLOR_PALETTE["accent"], size=10)
    ))
    fig_anomalies.update_layout(
        title="Revenue Anomalies Detection",
        xaxis_title="Date",
        yaxis_title="Revenue (EGP)",
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig_anomalies, use_container_width=True)
    
    if not anomalies.empty:
        st.markdown("##### 📝 Detected Anomalies")
        anomaly_data = anomalies[['ds', 'y', 'z_score']].copy()
        anomaly_data['ds'] = anomaly_data['ds'].dt.strftime('%Y-%m-%d')
        anomaly_data.columns = ['Date', 'Revenue', 'Z-Score']
        st.dataframe(anomaly_data, use_container_width=True)
    else:
        st.info("No significant anomalies detected in the selected period.")

# Search & Reports Tab
with tab_search:
    st.markdown("### 🔍 Inventory Purchase Search & Reports")
    
    # Search KPIs
    st.markdown("#### 📊 Search Results Summary")
    search_kpi_cols = st.columns(4)
    
    with search_kpi_cols[0]:
        total_purchases = len(filtered_data["inventory"])
        st.metric("Total Purchases", f"{total_purchases:,}")
    
    with search_kpi_cols[1]:
        total_amount = filtered_data["inventory"]["Invoice Amount"].sum()
        st.metric("Total Amount", f"EGP {total_amount:,.2f}")
    
    with search_kpi_cols[2]:
        avg_purchase = filtered_data["inventory"]["Invoice Amount"].mean()
        st.metric("Average Purchase", f"EGP {avg_purchase:,.2f}")
    
    with search_kpi_cols[3]:
        unique_companies = filtered_data["inventory"]["Invoice Company"].nunique()
        st.metric("Unique Companies", f"{unique_companies}")
    
    # Search Filters
    st.markdown("#### 🔎 Search Filters")
    search_cols = st.columns(5)
    
    with search_cols[0]:
        invoice_id = st.text_input(
            "Invoice ID",
            placeholder="Enter Invoice ID",
            help="Search by specific invoice ID"
        )
    
    with search_cols[1]:
        search_company = st.selectbox(
            "Company",
            ["All"] + sorted(filtered_data["inventory"]["Invoice Company"].unique().tolist())
        )
    
    with search_cols[2]:
        search_type = st.selectbox(
            "Inventory Type",
            ["All"] + sorted(filtered_data["inventory"]["Inventory Type"].unique().tolist())
        )
    
    with search_cols[3]:
        min_amount = st.number_input(
            "Min Amount (EGP)",
            min_value=0.0,
            max_value=float(filtered_data["inventory"]["Invoice Amount"].max()),
            value=0.0
        )
    
    with search_cols[4]:
        max_amount = st.number_input(
            "Max Amount (EGP)",
            min_value=0.0,
            max_value=float(filtered_data["inventory"]["Invoice Amount"].max()),
            value=float(filtered_data["inventory"]["Invoice Amount"].max())
        )
    
    # Apply filters
    search_results = filtered_data["inventory"].copy()
    
    # Apply Invoice ID filter if provided
    if invoice_id:
        search_results = search_results[
            search_results["Invoice ID"].astype(str).str.contains(invoice_id, case=False, na=False)
        ]
    
    if search_company != "All":
        search_results = search_results[search_results["Invoice Company"] == search_company]
    
    if search_type != "All":
        search_results = search_results[search_results["Inventory Type"] == search_type]
    
    search_results = search_results[
        (search_results["Invoice Amount"] >= min_amount) &
        (search_results["Invoice Amount"] <= max_amount)
    ]
    
    # Display search results
    st.markdown("#### 📋 Search Results")
    
    # Results summary
    results_cols = st.columns(3)
    
    with results_cols[0]:
        st.metric(
            "Filtered Purchases",
            f"{len(search_results):,}",
            delta=f"{len(search_results) - len(filtered_data['inventory']):,}"
        )
    
    with results_cols[1]:
        filtered_amount = search_results["Invoice Amount"].sum()
        st.metric(
            "Filtered Amount",
            f"EGP {filtered_amount:,.2f}",
            delta=f"EGP {filtered_amount - total_amount:,.2f}"
        )
    
    with results_cols[2]:
        filtered_avg = search_results["Invoice Amount"].mean()
        st.metric(
            "Filtered Average",
            f"EGP {filtered_avg:,.2f}",
            delta=f"EGP {filtered_avg - avg_purchase:,.2f}"
        )
    
    # Detailed results table
    st.dataframe(
        search_results.sort_values("date", ascending=False),
        use_container_width=True
    )
    
    # Purchase Analysis
    st.markdown("#### 📊 Purchase Analysis")
    analysis_cols = st.columns(2)
    
    with analysis_cols[0]:
        # Company distribution
        company_dist = search_results.groupby("Invoice Company")["Invoice Amount"].sum().sort_values(ascending=True)
        fig_company = go.Figure(data=[go.Bar(
            x=company_dist.values,
            y=company_dist.index,
            orientation='h',
            marker_color=COLOR_PALETTE["primary"]
        )])
        fig_company.update_layout(
            title="Purchases by Company",
            xaxis_title="Amount (EGP)",
            yaxis_title="Company",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_company, use_container_width=True)
    
    with analysis_cols[1]:
        # Monthly trend
        monthly_purchases = search_results.groupby(
            search_results["date"].dt.to_period("M")
        )["Invoice Amount"].sum()
        
        fig_monthly = go.Figure(data=[go.Scatter(
            x=monthly_purchases.index.astype(str),
            y=monthly_purchases.values,
            mode='lines+markers',
            line=dict(color=COLOR_PALETTE["secondary"])
        )])
        fig_monthly.update_layout(
            title="Monthly Purchase Trend",
            xaxis_title="Month",
            yaxis_title="Amount (EGP)",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Export functionality
    st.markdown("#### 📥 Export Results")
    export_cols = st.columns(2)
    
    with export_cols[0]:
        if st.button("Export to Excel"):
            # Create Excel writer
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Write search results
                search_results.to_excel(writer, sheet_name='Search Results', index=False)
                
                # Write summary
                summary_data = pd.DataFrame({
                    'Metric': ['Total Purchases', 'Total Amount', 'Average Purchase', 'Unique Companies'],
                    'Value': [len(search_results), filtered_amount, filtered_avg, search_results["Invoice Company"].nunique()]
                })
                summary_data.to_excel(writer, sheet_name='Summary', index=False)
            
            # Create download button
            output.seek(0)
            st.download_button(
                label="Download Excel Report",
                data=output,
                file_name=f"inventory_purchase_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with export_cols[1]:
        if st.button("Generate PDF Report"):
            st.info("PDF report generation will be implemented in the next version.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Pharmacy Analytics Dashboard • Updated: "
    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>", 
    unsafe_allow_html=True
)
