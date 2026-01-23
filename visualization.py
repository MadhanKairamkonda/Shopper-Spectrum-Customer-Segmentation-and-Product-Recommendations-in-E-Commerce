# ------------------------------------------------------------
# STEP 3: Exploratory Data Analysis (EDA) - SINGLE CELL
# ------------------------------------------------------------


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Load and clean the dataset (copied from preprocess.py)
df = pd.read_csv("online_retail.csv")
clean_df = df.copy()
clean_df = clean_df.dropna(subset=["CustomerID"])
clean_df = clean_df[~clean_df["InvoiceNo"].astype(str).str.startswith("C")]
clean_df = clean_df[clean_df["Quantity"] > 0]
clean_df = clean_df[clean_df["UnitPrice"] > 0]
clean_df = clean_df.drop_duplicates()
clean_df["InvoiceDate"] = pd.to_datetime(clean_df["InvoiceDate"])

# -------------------------------
# 3.1 Transaction Volume by Country
# -------------------------------
country_transactions = clean_df["Country"].value_counts().head(10)

plt.figure(figsize=(8,4))
country_transactions.plot(kind="bar")
plt.title("Top 10 Countries by Transaction Volume")
plt.xlabel("Country")
plt.ylabel("Number of Transactions")
plt.show()

# -------------------------------
# 3.2 Top-Selling Products
# -------------------------------
top_products = (
    clean_df.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(8,4))
top_products.plot(kind="barh")
plt.title("Top 10 Selling Products by Quantity")
plt.xlabel("Total Quantity Sold")
plt.ylabel("Product")
plt.show()

# -------------------------------
# 3.3 Purchase Trends Over Time
# -------------------------------
clean_df["InvoiceMonth"] = clean_df["InvoiceDate"].dt.to_period("M")
monthly_trend = clean_df.groupby("InvoiceMonth")["InvoiceNo"].nunique()

plt.figure(figsize=(10,4))
monthly_trend.plot()
plt.title("Monthly Purchase Trend")
plt.xlabel("Month")
plt.ylabel("Number of Transactions")
plt.show()

# -------------------------------
# 3.4 Monetary Distribution per Transaction
# -------------------------------
clean_df["TransactionValue"] = clean_df["Quantity"] * clean_df["UnitPrice"]

plt.figure(figsize=(8,4))
plt.hist(clean_df["TransactionValue"], bins=50)
plt.title("Transaction Monetary Value Distribution")
plt.xlabel("Transaction Value")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# 3.5 Monetary Distribution per Customer
# -------------------------------
customer_monetary = clean_df.groupby("CustomerID")["TransactionValue"].sum()

plt.figure(figsize=(8,4))
plt.hist(customer_monetary, bins=50)
plt.title("Customer Monetary Value Distribution")
plt.xlabel("Total Spend per Customer")
plt.ylabel("Number of Customers")
plt.show()

# -------------------------------
# 3.6 RFM Distributions (Preview)
# -------------------------------
customer_frequency = clean_df.groupby("CustomerID")["InvoiceNo"].nunique()

reference_date = clean_df["InvoiceDate"].max() + pd.Timedelta(days=1)
customer_recency = (
    clean_df.groupby("CustomerID")["InvoiceDate"]
    .max()
    .apply(lambda x: (reference_date - x).days)
)

# -------------------------------
# 3.7 Elbow Curve for Cluster Selection
# -------------------------------
rfm_temp = pd.DataFrame({
    "Recency": customer_recency,
    "Frequency": customer_frequency,
    "Monetary": customer_monetary
})

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_temp)

inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(2,11), inertia, marker="o")
plt.title("Elbow Curve for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()

# -------------------------------
# 3.8 Customer Cluster Profiles (Preview)
# -------------------------------
kmeans = KMeans(n_clusters=4, random_state=42)
rfm_temp["Cluster"] = kmeans.fit_predict(rfm_scaled)

cluster_profiles = rfm_temp.groupby("Cluster").mean()
print(cluster_profiles)

# -------------------------------
# 3.9 Product Recommendation Similarity Matrix
# -------------------------------
product_matrix = clean_df.pivot_table(
    index="CustomerID",
    columns="StockCode",
    values="Quantity",
    fill_value=0
)

product_similarity = cosine_similarity(product_matrix.T)
product_similarity_df = pd.DataFrame(
    product_similarity,
    index=product_matrix.columns,
    columns=product_matrix.columns
)

top_items = product_matrix.sum().sort_values(ascending=False).head(10).index

plt.figure(figsize=(8,6))
sns.heatmap(
    product_similarity_df.loc[top_items, top_items],
    cmap="coolwarm"
)
plt.title("Product Similarity Heatmap (Top Products)")
plt.show()
