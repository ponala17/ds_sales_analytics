import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

FILE_PATH = "../data/SampleSuperstore.csv"

# load dataset
df = pd.read_csv(FILE_PATH, encoding="latin-1")

# Check shape and columns
print("Shape of dataset:", df.shape)
print("\nColumn names:", df.columns.tolist())


# Peek at the data -- it gives first 5 rows
df.head()

# Data types
df.dtypes

# Missing values
df.isnull().sum()

df.rename(columns={"Sales": "sales_amount", "Profit": "profit_amount"}, inplace=True)

# Clean column names (remove whitespace)
df.columns = (
    df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_").str.lower()
)

# Drop unnecessary columns
df.drop(columns=["postal_code"], inplace=True)

# Convert date columns
df["order_date"] = pd.to_datetime(df["order_date"])
df["ship_date"] = pd.to_datetime(df["ship_date"])

# Create new time features
df["order_month"] = df["order_date"].dt.to_period("M")


print("\nColumn names:", df.columns.tolist())
# Column names: ['row_id', 'order_id', 'order_date', 'ship_date', 'ship_mode', 'customer_id', 'customer_name', 'segment', 'country',
#                'city', 'state', 'region', 'product_id', 'category', 'sub_category', 'product_name', 'sales_amount', 'quantity',
#                'discount', 'profit_amount', 'order_month']


# category_group = (
#     df.groupby("category")[["sales_amount", "profit_amount", "quantity"]]
#     .sum()
#     .reset_index()
# )

# # Bar plot using seaborn
# plt.figure(figsize=(10, 5))
# sns.barplot(x="category", y="sales_amount", data=category_group, palette="Set3")
# plt.title("Total Sales by Category")
# plt.ylabel("Sales")
# plt.show()

# # Profit
# plt.figure(figsize=(10, 5))
# sns.barplot(x="category", y="profit_amount", data=category_group, palette="Set2")
# plt.title("Total Profit by Category")
# plt.ylabel("Profit")
# plt.show()


# subcat_profit = df.groupby("sub_category")["profit_amount"].sum().sort_values()
# plt.figure(figsize=(12, 6))
# subcat_profit.plot(kind="barh", color="coral")
# plt.title("Profit by Sub-Category")
# plt.xlabel("Profit")
# plt.show()


# region_perf = (
#     df.groupby("region")[["sales_amount", "profit_amount"]].sum().reset_index()
# )

# # Dual axis barplot using matplotlib
# fig, ax1 = plt.subplots(figsize=(10, 6))

# ax2 = ax1.twinx()
# sns.barplot(x="region", y="sales_amount", data=region_perf, ax=ax1, color="skyblue")
# sns.lineplot(
#     x="region", y="profit_amount", data=region_perf, ax=ax2, color="red", marker="o"
# )

# ax1.set_ylabel("Sales", color="skyblue")
# ax2.set_ylabel("Profit", color="red")
# plt.title("Region-wise Sales and Profit")
# plt.show()


# monthly_sales = df.groupby("order_month")["sales_amount"].sum()
# monthly_sales.index = monthly_sales.index.to_timestamp()

# plt.figure(figsize=(14, 6))
# monthly_sales.plot(color="teal")
# plt.title("Monthly Sales Trend")
# plt.ylabel("Sales")
# plt.xlabel("Month")
# plt.grid(True)
# plt.show()


# plt.figure(figsize=(8, 6))
# sns.scatterplot(x="discount", y="profit_amount", data=df, hue="category", alpha=0.6)
# plt.title("Discount vs Profit")
# plt.show()

# df["shipping_delay"] = (df["ship_date"] - df["order_date"]).dt.days

# plt.figure(figsize=(8, 5))
# sns.histplot(df["shipping_delay"], bins=6, kde=True, color="orange")
# plt.title("Distribution of Shipping Delay (in Days)")
# plt.xlabel("Days")
# plt.show()


# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     df[["sales_amount", "profit_amount", "discount", "quantity"]].corr(),
#     annot=True,
#     cmap="coolwarm",
# )
# plt.title("Correlation Between Numeric Variables")
# plt.show()


# Group daily sales
daily_sales = df.groupby("order_date")["sales_amount"].sum().reset_index()

# Rename columns for Prophet
daily_sales.columns = ["ds", "y"]

# Sort by date
daily_sales = daily_sales.sort_values("ds")
daily_sales.head()


# Initialize and Train the Prophet Model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode="additive",
)

model.fit(daily_sales)


# Create Future Dataframe
# Forecast 90 days into the future
future = model.make_future_dataframe(periods=90)

# Make forecast
forecast = model.predict(future)
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()


# Plot forecast
model.plot(forecast)
plt.title("Retail Sales Forecast - Next 90 Days")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

# Visualize Components (Trend, Seasonality)
model.plot_components(forecast)
plt.show()


# Evaluate Model Accuracy (Optional but Advanced)
# Historical forecast with 3-month horizon
df_cv = cross_validation(model, initial="365 days", period="30 days", horizon="90 days")
df_p = performance_metrics(df_cv)
df_p[["horizon", "mape", "rmse", "mae"]]


# Export Forecast to CSV (for Tableau/BI Tools)
forecast[["ds", "yhat"]].to_csv("../outputs/sales_forecast.csv", index=False)


# Phase 4: Customer Segmentation with RFM + KMeans

# | RFM Metric    | Definition                              |
# | ------------- | --------------------------------------- |
# | **Recency**   | Days since the customer's last purchase |
# | **Frequency** | Total number of purchases made          |
# | **Monetary**  | Total amount the customer spent         |


# Prepare the RFM Data

import pandas as pd

# Set reference date for Recency (usually the day after the last transaction)
latest_date = df["order_date"].max() + pd.Timedelta(days=1)

# Create RFM table
rfm = (
    df.groupby("customer_name")
    .agg(
        {
            "order_date": lambda x: (latest_date - x.max()).days,  # Recency
            "order_id": "nunique",  # Frequency
            "sales_amount": "sum",  # Monetary
        }
    )
    .reset_index()
)

# Rename columns
rfm.columns = ["Customer", "Recency", "Frequency", "Monetary"]
rfm.head()


# Scale the RFM Values

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])


# Apply KMeans Clustering
# Try 4 clusters (can experiment with 3-6)
kmeans = KMeans(n_clusters=4, random_state=42)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

rfm.head()

# Visualize the Segments
plt.figure(figsize=(10, 6))
sns.scatterplot(data=rfm, x="Recency", y="Monetary", hue="Cluster", palette="Set2")
plt.title("Customer Segments by Recency and Monetary Value")
plt.show()


# Analyze Cluster Profiles
# Mean RFM by cluster
cluster_summary = (
    rfm.groupby("Cluster")
    .agg(
        {
            "Recency": "mean",
            "Frequency": "mean",
            "Monetary": "mean",
            "Customer": "count",
        }
    )
    .rename(columns={"Customer": "CustomerCount"})
    .reset_index()
)

print(cluster_summary, "------")

rfm.to_csv("../outputs/customer_segments.csv", index=False)
