import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import requests
import os

# Load Final Sales Data
file_path = "Final_Nike_Sales_Data__1990-1995_.csv"
df_sales = pd.read_csv(file_path)

df_sales["Date"] = pd.to_datetime(df_sales["Year"].astype(str) + "-" + df_sales["Month"], format="%Y-%B")
df_sales["Quarter"] = df_sales["Date"].dt.to_period("Q").astype(str)

monthly_sales = df_sales.groupby("Date").agg({
    "Revenue_USD": "sum",
    "Units_Sold": "sum",
    "Retail_Price": "mean",
    "Online_Sales_Percentage": "mean"
}).reset_index()

quarterly_sales = df_sales.groupby("Quarter").agg({
    "Revenue_USD": "sum",
    "Units_Sold": "sum",
    "Retail_Price": "mean",
    "Online_Sales_Percentage": "mean"
}).reset_index()

# Creating lag features & rolling statistics
def create_features(df, lag=4):
    for i in range(1, lag+1):
        df[f"Revenue_Lag_{i}"] = df["Revenue_USD"].shift(i)
    df["Rolling_Mean_3"] = df["Revenue_USD"].rolling(window=3).mean()
    df["Rolling_Std_3"] = df["Revenue_USD"].rolling(window=3).std()
    df["Exp_Smooth"] = df["Revenue_USD"].ewm(span=3, adjust=False).mean()
    return df.dropna()

monthly_sales = create_features(monthly_sales)
quarterly_sales = create_features(quarterly_sales)

# Streamlit Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Select Page", ["📜 Project Summary", "📊 Sales Data Insights", "📈 Sales Forecasting Model", "🏃 Running Gait & Shoe Recommendation"])



if page == "📜 Project Summary":
    st.title("📜 Project Summary")
    st.write("""
    ### Project Overview  
This project focuses on analyzing sales data, forecasting revenue trends, and improving product recommendations using data-driven insights and machine learning.

### Key Features  

✅ **Sales Data Insights**  
- Visualizes revenue trends by month, quarter, and product category.  
- Identifies high-performing regions and product lines.  

✅ **Sales Forecasting Model**  
- Implements an XGBoost-based model to predict future revenue trends.  
- Helps in strategic decision-making with accurate sales projections.  

✅ **Gait Analysis & Shoe Recommendation**  
- Uses stride length and speed to recommend the best running shoes.  
- Enhances user experience with personalized product suggestions.  
    """)
    st.markdown("<h4 style='text-align: center; color: grey;'>Done by Isha Deshmukh</h4>", unsafe_allow_html=True)
    
    # Cookie to Track Views
    if 'view_count' not in st.session_state:
        st.session_state['view_count'] = 0
    st.session_state['view_count'] += 1
    # Unique Visitor Tracking using Session State
if 'user_views' not in st.session_state:
    st.session_state['user_views'] = {}

# Assign a unique identifier for each user (basic hashing using IP)
try:
    user_ip = requests.get('https://api64.ipify.org').text  # Fetch external IP
    if user_ip not in st.session_state['user_views']:
        st.session_state['user_views'][user_ip] = True
except Exception:
    user_ip = "Unknown"

# Admin Access Toggle (Ensure Only You Can See Visitor Count)
ADMIN_SECRET = "my_secret_code"  # Change this to a unique code only you know
admin_access = st.sidebar.text_input("🔒 Admin Access (Enter Code)", type="password")

if admin_access == ADMIN_SECRET:
    st.write(f"🔒 **Total Unique Visitors:** {len(st.session_state['user_views'])}")

    # Store view count only for admin tracking
    if 'user_views' not in st.session_state:
        st.session_state['user_views'] = []
    
    user_id = st.experimental_user.hash if hasattr(st.experimental_user, 'hash') else np.random.randint(1000000)
    if user_id not in st.session_state['user_views']:
        st.session_state['user_views'].append(user_id)
    
    # Only admin sees the total views
    # Admin-only view tracking removed to prevent missing secrets error.
    if st.session_state.get('admin_access', False):
        st.write(f"🔒 Total unique views: {len(st.session_state['user_views'])}")

elif page == "📊 Sales Data Insights":
    st.title("📊 Sales Data Insights")
    
    # Monthly Revenue Visualization
    fig1 = px.line(monthly_sales, x="Date", y="Revenue_USD", title="📈 Monthly Revenue Trend", markers=True, color_discrete_sequence=["blue"])
    st.plotly_chart(fig1)
    
    # Quarterly Revenue Visualization
    fig2 = px.line(quarterly_sales, x="Quarter", y="Revenue_USD", title="📉 Quarterly Revenue Trend", markers=True, color_discrete_sequence=["red"])
    st.plotly_chart(fig2)
    
    # Revenue by Region
    region_revenue = df_sales.groupby("Region")["Revenue_USD"].sum().reset_index().sort_values(by="Revenue_USD", ascending=False)
    fig3 = px.bar(region_revenue, x="Region", y="Revenue_USD", title="🌍 Revenue by Region", color="Revenue_USD", color_continuous_scale="Blues")
    st.plotly_chart(fig3)
    
    # Revenue by Product Line
    product_revenue = df_sales.groupby("Product_Line")["Revenue_USD"].sum().reset_index().sort_values(by="Revenue_USD", ascending=False)
    fig4 = px.bar(product_revenue, x="Product_Line", y="Revenue_USD", title="🔥 Revenue by Product Line", color="Revenue_USD", color_continuous_scale="Teal")
    st.plotly_chart(fig4)
    
    # Revenue by Region and Tier
    region_tier_revenue = df_sales.groupby(["Region", "Price_Tier"]).agg({"Revenue_USD": "sum"}).reset_index()
    fig5 = px.bar(region_tier_revenue, x="Region", y="Revenue_USD", color="Price_Tier",
                   title="📌 Revenue by Region & Tier", barmode="group",
                   color_discrete_sequence=["#4682B4", "#5F9EA0", "#B0C4DE"])
    st.plotly_chart(fig5)

elif page == "📈 Sales Forecasting Model":
    st.title("📈 Sales Forecasting Model")
    
    # Prepare Data for XGBoost
    feature_cols = ["Units_Sold", "Retail_Price", "Online_Sales_Percentage", "Rolling_Mean_3", "Rolling_Std_3", "Exp_Smooth"]
    target_col = "Revenue_USD"
    
    X = monthly_sales[feature_cols]
    y = monthly_sales[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train XGBoost Model
    xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Forecast Monthly Sales
    future_dates = pd.date_range(start=monthly_sales["Date"].iloc[-1] + pd.DateOffset(months=1), periods=12, freq='M')
    future_forecast = xgb_model.predict(X.iloc[-12:])
    future_forecast = np.concatenate(([monthly_sales["Revenue_USD"].iloc[-1]], future_forecast))
    forecast_df = pd.DataFrame({"Date": [monthly_sales["Date"].iloc[-1]] + list(future_dates), "Revenue": future_forecast})
    
    fig6 = px.line(title="📈 Monthly Revenue Forecast")
    fig6.add_scatter(x=monthly_sales["Date"], y=monthly_sales["Revenue_USD"], mode='lines', name='Historical', line=dict(color='blue'))
    fig6.add_scatter(x=forecast_df["Date"], y=forecast_df["Revenue"], mode='lines', name='Forecast', line=dict(color='red'))
    st.plotly_chart(fig6, key='monthly_forecast')
    
    # Display Numeric Monthly Trends
    st.write("### Monthly Sales Data")
    st.dataframe(monthly_sales[['Date', 'Revenue_USD']].tail(12))
    
    # Forecast Quarterly Sales
    future_quarter_dates = pd.date_range(start=pd.to_datetime(quarterly_sales["Quarter"].iloc[-1]) + pd.DateOffset(months=3), periods=4, freq='Q')
    future_quarter_forecast = xgb_model.predict(X.iloc[-4:])
    future_quarter_forecast = np.concatenate(([quarterly_sales["Revenue_USD"].iloc[-1]], future_quarter_forecast))
    forecast_quarter_df = pd.DataFrame({"Quarter": [quarterly_sales["Quarter"].iloc[-1]] + list(future_quarter_dates), "Revenue": future_quarter_forecast})
    
    fig7 = px.line(title="📉 Quarterly Revenue Forecast")
    fig7.add_scatter(x=quarterly_sales["Quarter"], y=quarterly_sales["Revenue_USD"], mode='lines', name='Historical', line=dict(color='blue'))
    fig7.add_scatter(x=forecast_quarter_df["Quarter"], y=forecast_quarter_df["Revenue"], mode='lines', name='Forecast', line=dict(color='red'))
    st.plotly_chart(fig7, key='quarterly_forecast')
    
    # Display Numeric Quarterly Trends
    st.write("### Quarterly Sales Data")
    st.dataframe(quarterly_sales[['Quarter', 'Revenue_USD']].tail(4))
    
    # Model Performance Metrics
    xgb_predictions = xgb_model.predict(X_test)
    mae = mean_absolute_error(y_test, xgb_predictions)
    rmse = mean_squared_error(y_test, xgb_predictions) ** 0.5
    st.write(f"### 🏆 Model Performance: MAE = {mae:.2f}, RMSE = {rmse:.2f}")
    
    
elif page == "🏃 Running Gait & Shoe Recommendation":
    st.title("🏃 Running Gait & Shoe Recommendation")
    
    # User Inputs
    stride_length = st.slider("Select Stride Length (meters)", 0.8, 1.5, 1.1)
    running_speed = st.slider("Select Running Speed (km/h)", 3, 15, 8)
    
    # Shoe Recommendation Logic
    if stride_length < 1.0:
        if running_speed < 6:
            recommended_shoe = "Nike Pegasus"
        else:
            recommended_shoe = "Nike Zoom Fly"
    elif 1.0 <= stride_length <= 1.3:
        if running_speed < 6:
            recommended_shoe = "Nike Infinity Run"
        else:
            recommended_shoe = "Nike Vaporfly"
    else:
        if running_speed < 6:
            recommended_shoe = "Nike React Infinity"
        else:
            recommended_shoe = "Nike Alphafly"
    
    # Display Recommendation
    st.write(f"### 🏆 Recommended Shoe: {recommended_shoe}")
    
    # Simulated Gait Data Visualization
    gait_data = pd.DataFrame({
        "Stride_Length": np.random.normal(stride_length, 0.1, 100),
        "Running_Speed": np.random.normal(running_speed, 1, 100)
    })
    fig_gait = px.scatter(gait_data, x="Stride_Length", y="Running_Speed", title="👟 Running Gait Analysis", color_discrete_sequence=["purple"])
    st.plotly_chart(fig_gait)
