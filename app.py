import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------
# LOAD MODELS & DATA
# ------------------------------------

# Load trained clustering model & scaler
kmeans = joblib.load("rfm_kmeans_model.pkl")
scaler = joblib.load("rfm_scaler.pkl")

# Load cleaned dataset
clean_df = pd.read_csv("online_retail.csv")
clean_df = clean_df.dropna(subset=["CustomerID"])
clean_df = clean_df[~clean_df["InvoiceNo"].astype(str).str.startswith("C")]
clean_df = clean_df[clean_df["Quantity"] > 0]
clean_df = clean_df[clean_df["UnitPrice"] > 0]
clean_df["InvoiceDate"] = pd.to_datetime(clean_df["InvoiceDate"])
clean_df["TransactionValue"] = clean_df["Quantity"] * clean_df["UnitPrice"]

# ------------------------------------
# PRODUCT RECOMMENDATION SETUP
# ------------------------------------

product_matrix = clean_df.pivot_table(
    index="CustomerID",
    columns="Description",
    values="Quantity",
    fill_value=0
)

product_similarity = cosine_similarity(product_matrix.T)

product_sim_df = pd.DataFrame(
    product_similarity,
    index=product_matrix.columns,
    columns=product_matrix.columns
)

def recommend_products(product_name, top_n=5):
    if product_name not in product_sim_df.columns:
        return None
    scores = product_sim_df[product_name].sort_values(ascending=False)
    return scores.iloc[1:top_n+1]

# ------------------------------------
# CLUSTER LABEL MAPPING
# ------------------------------------

cluster_labels = {
    0: "Regular",
    1: "Occasional",
    2: "High-Value",
    3: "At-Risk"
}

# ------------------------------------
# STREAMLIT UI
# ------------------------------------

st.set_page_config(page_title="Shopper Spectrum", layout="wide")
st.title("🛍️ Shopper Spectrum – Customer Intelligence App")

# Tabs
tab1, tab2 = st.tabs(["🎯 Product Recommendation", "📊 Customer Segmentation"])

# ------------------------------------
# 🎯 TAB 1: PRODUCT RECOMMENDATION
# ------------------------------------

with tab1:
    st.subheader("🔍 Product Recommendation System")

    product_input = st.text_input(
        "Enter Product Name",
        placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER"
    )

    if st.button("Get Recommendations"):
        recommendations = recommend_products(product_input)

        if recommendations is None:
            st.error("❌ Product not found in dataset.")
        else:
            st.success("✅ Top 5 Similar Products")
            for product, score in recommendations.items():
                st.write(f"🟢 **{product}**  (Similarity: {score:.2f})")

# ------------------------------------
# 📊 TAB 2: CUSTOMER SEGMENTATION
# ------------------------------------

with tab2:
    st.subheader("👤 Customer Segmentation Predictor")

    recency = st.number_input("Recency (days)", min_value=0, step=1)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0, step=1)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, step=10.0)

    if st.button("Predict Cluster"):
        input_data = pd.DataFrame([[recency, frequency, monetary]],
                                  columns=["Recency", "Frequency", "Monetary"])

        scaled_input = scaler.transform(input_data)
        cluster = kmeans.predict(scaled_input)[0]
        segment = cluster_labels[cluster]

        st.success(f"🎯 Predicted Customer Segment: **{segment}**")


#  To run this app, use the command: streamlit run app.py