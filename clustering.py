# ------------------------------------------------------------
# STEP 4: RFM Feature Engineering, Clustering & Recommendation
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# -------------------------------
# 1️⃣ RFM FEATURE ENGINEERING
# -------------------------------

## --- Load and clean the data (copied from preprocess.py) ---
df = pd.read_csv("online_retail.csv")
clean_df = df.copy()
clean_df = clean_df.dropna(subset=["CustomerID"])
clean_df = clean_df[~clean_df["InvoiceNo"].astype(str).str.startswith("C")]
clean_df = clean_df[clean_df["Quantity"] > 0]
clean_df = clean_df[clean_df["UnitPrice"] > 0]
clean_df = clean_df.drop_duplicates()
clean_df["InvoiceDate"] = pd.to_datetime(clean_df["InvoiceDate"])

# Add TransactionValue column
clean_df["TransactionValue"] = clean_df["Quantity"] * clean_df["UnitPrice"]

# Reference date for recency
reference_date = clean_df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = clean_df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (reference_date - x.max()).days,
    "InvoiceNo": "nunique",
    "TransactionValue": "sum"
}).reset_index()

rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

# -------------------------------
# 2️⃣ STANDARDIZATION
# -------------------------------

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

# -------------------------------
# 3️⃣ CLUSTER SELECTION (Elbow + Silhouette)
# -------------------------------

inertia = []
sil_scores = []

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(rfm_scaled)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(rfm_scaled, labels))

# Elbow Plot
plt.figure(figsize=(6,4))
plt.plot(range(2,9), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# Silhouette Plot
plt.figure(figsize=(6,4))
plt.plot(range(2,9), sil_scores, marker='o')
plt.title("Silhouette Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Score")
plt.show()

# -------------------------------
# 4️⃣ RUN FINAL KMEANS
# -------------------------------

kmeans = KMeans(n_clusters=4, random_state=42)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

# -------------------------------
# 5️⃣ CLUSTER INTERPRETATION
# -------------------------------

cluster_summary = rfm.groupby("Cluster").mean()

print(cluster_summary)

# Business-friendly labels
cluster_labels = {
    0: "Regular",
    1: "Occasional",
    2: "High-Value",
    3: "At-Risk"
}

rfm["Segment"] = rfm["Cluster"].map(cluster_labels)

# -------------------------------
# 6️⃣ CLUSTER VISUALIZATION (2D)
# -------------------------------

plt.figure(figsize=(8,5))
sns.scatterplot(
    x=rfm["Recency"],
    y=rfm["Monetary"],
    hue=rfm["Segment"],
    palette="Set2"
)
plt.title("Customer Segments (Recency vs Monetary)")
plt.show()

# -------------------------------
# 7️⃣ SAVE MODEL FOR STREAMLIT
# -------------------------------

joblib.dump(kmeans, "rfm_kmeans_model.pkl")
joblib.dump(scaler, "rfm_scaler.pkl")

# -------------------------------
# 📌 RECOMMENDATION SYSTEM
# -------------------------------

# Customer–Product matrix
product_matrix = clean_df.pivot_table(
    index="CustomerID",
    columns="Description",
    values="Quantity",
    fill_value=0
)

# Cosine similarity
product_similarity = cosine_similarity(product_matrix.T)

product_sim_df = pd.DataFrame(
    product_similarity,
    index=product_matrix.columns,
    columns=product_matrix.columns
)

# Function to get top-5 recommendations
def recommend_products(product_name, top_n=5):
    """
    Returns the top_n most similar products to the given product_name
    using item-based collaborative filtering (cosine similarity).
    """
    if product_name not in product_sim_df.columns:
        return "Product not found"
    scores = product_sim_df[product_name].sort_values(ascending=False)
    recommendations = scores.iloc[1:top_n+1]
    # Format output for readability
    result = []
    for prod, score in recommendations.items():
        result.append(f"{prod} (similarity: {score:.2f})")
    return result

# Example
# Use a sample product name from the dataset for demonstration

# Prompt user for a product name
if len(product_matrix.columns) > 0:
    print("Enter a product name for recommendations (or press Enter to use a sample):")
    user_input = input().strip()
    if not user_input:
        user_input = product_matrix.columns[0]
    print(f"Top 5 recommendations for '{user_input}':")
    recommendations = recommend_products(user_input)
    if isinstance(recommendations, list):
        for rec in recommendations:
            print(rec)
    else:
        print(recommendations)
else:
    print("No products found in the dataset.")
