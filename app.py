import streamlit as st
import pandas as pd
import plotly.express as px

# Load Data
df = pd.read_csv("data/SampleSuperstore.csv", encoding="latin-1")
forecast = pd.read_csv("outputs/sales_forecast.csv")
segments = pd.read_csv("outputs/customer_segments.csv")

# Sidebar filters
st.sidebar.header("Filters")
region_filter = st.sidebar.multiselect(
    "Select Region", df["Region"].unique(), default=df["Region"].unique()
)

# Apply filters
df_filtered = df[df["Region"].isin(region_filter)]

# Title
st.title("Retail Sales Analytics Dashboard 🛒")

# Total Sales KPI
total_sales = df_filtered["Sales"].sum()
st.metric("Total Sales", f"${total_sales:,.2f}")

# Bar Chart: Sales by Category
category_sales = df_filtered.groupby("Category")["Sales"].sum().reset_index()
fig1 = px.bar(
    category_sales, x="Category", y="Sales", title="Sales by Category", color="Category"
)
st.plotly_chart(fig1)

# Time Series: Monthly Sales
df_filtered["Order Date"] = pd.to_datetime(df_filtered["Order Date"])
monthly = (
    df_filtered.groupby(df_filtered["Order Date"].dt.to_period("M"))["Sales"]
    .sum()
    .reset_index()
)
monthly["Order Date"] = monthly["Order Date"].astype(str)
fig2 = px.line(monthly, x="Order Date", y="Sales", title="Monthly Sales Trend")
st.plotly_chart(fig2)

# Forecast Chart
fig3 = px.line(forecast, x="ds", y="yhat", title="Sales Forecast (Next 90 Days)")
st.plotly_chart(fig3)

# Customer Segments
fig4 = px.scatter(
    segments,
    x="Recency",
    y="Monetary",
    color="Cluster",
    size="Frequency",
    title="Customer Segmentation (RFM)",
    hover_data=["Customer"],
)
st.plotly_chart(fig4)

# Footer
st.markdown("---")
st.markdown("Built with Python, Streamlit, Plotly, Prophet, and KMeans.")
