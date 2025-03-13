import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the datasets
df_2009_2010 = pd.read_csv("online_retail_2009_2010.csv")
df_2010_2011 = pd.read_csv("online_retail_2010_2011.csv")

# Combine into one DataFrame
df = pd.concat([df_2009_2010, df_2010_2011], ignore_index=True)

# Explore the data
print("Dataset Info:")
print(df.info())
print("\nFirst 5 Rows:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())

# Clean the data
print("\nCleaning the data...")
# 1. Remove rows with missing CustomerID
df = df.dropna(subset=['Customer ID'])

# 2. Remove rows with negative Quantity or UnitPrice
df = df[(df['Quantity'].astype(float) > 0) & (df['Price'] > 0)]

# 3. Remove duplicates
df = df.drop_duplicates()

# 4. Convert CustomerID to integer
df['Customer ID'] = df['Customer ID'].astype(int)

# 5. Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Check the cleaned data
print("\nCleaned Dataset Info:")
print(df.info())
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Create a user-item interaction matrix
print("\nCreating user-item interaction matrix...")
user_item_matrix = df.pivot_table(
    index='Customer ID',
    columns='StockCode',
    values='Quantity',
    aggfunc='sum',
    fill_value=0
)

# Save the cleaned data and matrix
df.to_csv("cleaned_online_retail.csv", index=False)
user_item_matrix.to_csv("user_item_matrix.csv")

print("\nPreprocessing complete! Saved cleaned data and user-item matrix.")