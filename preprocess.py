# ---------------------------------------------------
# STEP 2: Data Preprocessing & Cleaning
# ---------------------------------------------------


import pandas as pd
df = pd.read_csv("online_retail.csv")

# Make a copy to preserve raw data
clean_df = df.copy()

# Initial shape
print("Initial dataset shape:", clean_df.shape)

# 1. Remove missing CustomerID
clean_df = clean_df.dropna(subset=["CustomerID"])

# 2. Remove cancelled invoices
clean_df = clean_df[~clean_df["InvoiceNo"].astype(str).str.startswith("C")]

# 3. Remove invalid quantities
clean_df = clean_df[clean_df["Quantity"] > 0]

# 4. Remove invalid unit prices
clean_df = clean_df[clean_df["UnitPrice"] > 0]

# 5. Remove duplicate records
clean_df = clean_df.drop_duplicates()

# Convert InvoiceDate to datetime
clean_df["InvoiceDate"] = pd.to_datetime(clean_df["InvoiceDate"])

# Final shape
print("Final dataset shape after cleaning:", clean_df.shape)

# Display cleaned sample
print(clean_df.head())
