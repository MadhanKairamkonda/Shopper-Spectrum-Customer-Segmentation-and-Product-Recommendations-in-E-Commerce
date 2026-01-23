# ---------------------------------------------
# STEP 1: Dataset Loading & Initial Inspection
# ---------------------------------------------

import pandas as pd

# 1. Load the dataset
df = pd.read_csv("online_retail.csv")

print("Dataset loaded successfully!\n")

# 2. Display first 5 rows
print("First 5 rows of the dataset:")
print(df)

# 3. Dataset shape
print("\nDataset Shape (Rows, Columns):")
print(df.shape)

# 4. Column names
print("\nColumn Names:")
print(df.columns.tolist())

# 5. Dataset information (data types & non-null counts)
print("\nDataset Info:")
df.info()

# 6. Missing values summary
print("\nMissing Values Count:")
print(df.isnull().sum())

# 7. Duplicate records count
print("\nNumber of Duplicate Records:")
print(df.duplicated().sum())

# 8. Business sanity checks
# Cancelled invoices
cancelled_invoices = df[df["InvoiceNo"].astype(str).str.startswith("C")].shape[0]

# Invalid quantity
invalid_quantity = df[df["Quantity"] <= 0].shape[0]

# Invalid price
invalid_price = df[df["UnitPrice"] <= 0].shape[0]

print("\nBusiness Sanity Checks:")
print(f"Cancelled Invoices: {cancelled_invoices}")
print(f"Invalid Quantity (<=0): {invalid_quantity}")
print(f"Invalid Unit Price (<=0): {invalid_price}")
