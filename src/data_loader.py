import pandas as pd
import numpy as np

# Define file paths
input_file_path = "C:/Users/Admin/OneDrive - Aayush/Desktop/pricing_model/data/oneupdated_data_with_random_values.csv"
output_file_path = "C:/Users/Admin/OneDrive - Aayush/Desktop/pricing_model/data/oneupdated_data_with_random_values.csv"

# Read the CSV file
df = pd.read_csv(input_file_path, dtype={'product_id': 'str'})

# Check the columns in the dataset
print("Columns in the dataset:", df.columns)

# Preprocess 'product_id' column to convert it to numeric (int) using the numeric portion
df['product_id'] = df['product_id'].str.extract(r'(\d+)')  # Extract numeric portion from the 'product_id'
df['product_id'] = pd.to_numeric(df['product_id'], errors='coerce')  # Convert to numeric, NaNs for non-numeric IDs
df['product_id'].fillna(-1, inplace=True)  # Fill NaN values with -1 (or any placeholder)

# Generate random price adjustments between 10% and 15% of the current price
df['price_adjustment'] = np.random.uniform(0.10, 0.15, size=len(df)) * df['current_price']

# Optionally, print a sample of the updated DataFrame to verify
print(df[['product_id', 'current_price', 'price_adjustment']].head())

# Save the updated DataFrame back to the file
df.to_csv(output_file_path, index=False)

print(f"Updated data with random price_adjustment saved successfully to: {output_file_path}")
