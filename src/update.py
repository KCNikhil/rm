import pandas as pd
import json

# Define file paths
csv_file = r"C:\Users\Admin\OneDrive - Aayush\Desktop\pricing_model\data\aggregated_output_updated_copy.csv"
json_file = r"C:\Users\Admin\OneDrive - Aayush\Desktop\pricing_model\data\pricing_data.json"

# Load CSV into Pandas DataFrame
df = pd.read_csv(csv_file)

# Convert DataFrame to JSON
json_data = df.to_json(orient="records", indent=4)

# Save JSON to a file
with open(json_file, "w") as f:
    f.write(json_data)

print(f"CSV successfully converted to JSON and saved at: {json_file}")
