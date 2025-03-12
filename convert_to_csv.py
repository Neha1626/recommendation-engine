import pandas as pd

# Read the Excel file
df = pd.read_excel("online_retail_II.xlsx")

# Split into two smaller files based on years (to reduce size)
df_2009_2010 = df[df['InvoiceDate'].dt.year <= 2010]
df_2010_2011 = df[df['InvoiceDate'].dt.year > 2010]

# Save as CSV
df_2009_2010.to_csv("online_retail_2009_2010.csv", index=False)
df_2010_2011.to_csv("online_retail_2010_2011.csv", index=False)

print("Converted to CSV and split into two files!")
