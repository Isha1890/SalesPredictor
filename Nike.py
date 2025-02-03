import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load Data
@st.cache
def load_data():
    df = pd.read_csv("nike_sales_2024.csv")  # Ensure dataset is in the same directory
    return df

df = load_data()

# App Title
st.title("Nike Digital Innovation & Sales Dashboard")

# Sidebar
st.sidebar.header("Filters")
selected_region = st.sidebar.selectbox("Select Region", df["Region"].unique())
selected_category = st.sidebar.selectbox("Select Product Category", df["Main_Category"].unique())

# Filtered Data
filtered_df = df[(df["Region"] == selected_region) & (df["Main_Category"] == selected_category)]

# Show Data
st.write(f"### Sales Data for {selected_region} - {selected_category}")
st.dataframe(filtered_df)

# Visualization: Sales by Price Tier
st.write("### Revenue by Price Tier")
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=filtered_df, x="Price_Tier", y="Revenue_USD", ax=ax)
st.pyplot(fig)

# Online Sales Analysis
st.write("### Online Sales Percentage by Region")
region_online_sales = df.groupby("Region")["Online_Sales_Percentage"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=region_online_sales.index, y=region_online_sales.values, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Clustering for Digital Innovation
st.write("### AI-Driven Innovation Suggestions")
X = df[["Revenue_USD", "Online_Sales_Percentage", "Retail_Price"]]
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

# Display Clustering Results
st.write("Product Segments for Digital Innovation:")
st.dataframe(df.groupby("Cluster")[["Revenue_USD", "Online_Sales_Percentage", "Retail_Price"]].mean())

# Innovation Strategy Suggestions
st.write("### Suggested Digital Strategies")
if selected_region in ["Greater China", "America"]:
    st.write("- **Increase digital marketing efforts** to drive more online sales.")
    st.write("- **AI-based personalized shopping experience** to attract premium users.")
elif selected_region in ["India", "Japan", "South Korea"]:
    st.write("- **Enhance digital payment integrations** for seamless transactions.")
    st.write("- **Invest in localized digital marketing** to boost engagement.")

st.write("Nike can leverage AI-driven insights for better decision-making and innovation!")

# Footer
st.sidebar.write("Developed by Isha Deshmukh for Nike Interview")
