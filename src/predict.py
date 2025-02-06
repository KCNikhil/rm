import pandas as pd
import joblib
import os
from pathlib import Path

def main():
    # Configuration using your updated files
    input_csv = r"C:\Users\Admin\OneDrive - Aayush\Desktop\pricing_model\data\aggregated_output.csv"
    model_dir = r"C:\Users\Admin\OneDrive - Aayush\Desktop\pricing_model\models"
    output_dir = Path(input_csv).parent

    try:
        # Load model and scaler with the correct file names
        model = joblib.load(os.path.join(model_dir, "ridge_model.pkl"))
        scaler = joblib.load(os.path.join(model_dir, "ridge_scaler.pkl"))
        
        # Load input data
        df = pd.read_csv(input_csv)
        required_columns = [
            'cost_price', 'current_price', 'customer_rating',
            'discount', 'sales_volume', 'profit_margin', 'price_ratio'
        ]
        
        # Validate and prepare data
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing columns: {missing}")

        X = df[required_columns]
        X_scaled = scaler.transform(X)

        # Make predictions using the model
        predictions = model.predict(X_scaled)
        
        # Replace the 'price_adjustment' column with the predictions
        df['price_adjustment'] = predictions

        # Save results to a new CSV file
        output_path = output_dir / f"{Path(input_csv).stem}_updated.csv"
        df.to_csv(output_path, index=False)
        
        print(f"✅ Updated file saved to:\n{output_path}")
        print("\nFirst 5 updated rows:")
        print(df[['cost_price', 'current_price', 'price_adjustment']].head())

    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        print("Verify these files exist in the models directory:")
        print("- ridge_model.pkl")
        print("- ridge_scaler.pkl")
        exit(1)

if __name__ == "__main__":
    main()
