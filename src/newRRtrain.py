import os
import sqlite3
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, validation_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# Optionally suppress specific warnings if you're confident the data format is consistent
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ---------------------------
# Database Helper Functions
# ---------------------------
def connect_db(db_path):
    """Connect to the SQLite database."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at: {db_path}")
    return sqlite3.connect(db_path)

def create_pricing_table(conn):
    """Create a pricing table if it does not exist."""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS pricing (
        product_id INTEGER PRIMARY KEY,
        cost_price REAL,
        current_price REAL,
        customer_rating REAL,
        discount REAL,
        sales_volume REAL,
        profit_margin REAL,
        price_ratio REAL,
        price_adjustment REAL
    );
    """
    conn.execute(create_table_query)
    conn.commit()

def create_artifacts_table(conn):
    """Create a table to store model artifacts (BLOBs)."""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS artifacts (
        artifact_name TEXT PRIMARY KEY,
        artifact_data BLOB
    );
    """
    conn.execute(create_table_query)
    conn.commit()

def insert_or_update_pricing_data(conn, df):
    """
    Insert data from a DataFrame into the pricing table.
    If a product_id already exists, update its record.
    """
    cursor = conn.cursor()
    for index, row in df.iterrows():
        cursor.execute("""
            INSERT OR REPLACE INTO pricing (
                product_id, cost_price, current_price, customer_rating,
                discount, sales_volume, profit_margin, price_ratio, price_adjustment
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            int(row['product_id']),
            row['cost_price'],
            row['current_price'],
            row['customer_rating'],
            row['discount'],
            row['sales_volume'],
            row['profit_margin'],
            row['price_ratio'],
            row.get('price_adjustment', 0.0)  # defaulting to 0.0 if not provided
        ))
    conn.commit()

def fetch_pricing_data(conn):
    """Fetch all pricing data from the database into a DataFrame."""
    df = pd.read_sql_query("SELECT * FROM pricing;", conn)
    return df

def update_price_adjustment(conn, product_id, predicted_adjustment):
    """Update the price_adjustment for a given product_id."""
    conn.execute("""
        UPDATE pricing SET price_adjustment = ? WHERE product_id = ?;
    """, (predicted_adjustment, product_id))
    conn.commit()

def save_artifact_to_db(conn, artifact_name, artifact_obj):
    """
    Serialize an artifact (model, scaler, etc.) and store it in the artifacts table.
    """
    artifact_data = pickle.dumps(artifact_obj)
    conn.execute("""
        INSERT OR REPLACE INTO artifacts (artifact_name, artifact_data)
        VALUES (?, ?);
    """, (artifact_name, sqlite3.Binary(artifact_data)))
    conn.commit()
    print(f"✅ Saved {artifact_name} to the database.")

def load_artifact_from_db(conn, artifact_name):
    """Load and deserialize an artifact from the artifacts table."""
    cursor = conn.cursor()
    cursor.execute("SELECT artifact_data FROM artifacts WHERE artifact_name = ?;", (artifact_name,))
    row = cursor.fetchone()
    if row is None:
        raise ValueError(f"Artifact {artifact_name} not found in database.")
    artifact_obj = pickle.loads(row[0])
    print(f"✅ Loaded {artifact_name} from the database.")
    return artifact_obj

# ---------------------------
# Model Training and Prediction
# ---------------------------
def train_model(df, config):
    """
    Train the Ridge regression model using data from the DataFrame.
    Returns the trained model, scaler, and metrics.
    """
    # Remove rows missing the target value
    initial_count = len(df)
    df = df.dropna(subset=[config["target"]])
    if len(df) < initial_count:
        print(f"⚠️ Removed {initial_count - len(df)} rows with missing target values.")

    # Prepare features and target ensuring a DataFrame is used for X with explicit column names
    X = df[config["features"]].copy()
    y = df[config["target"]]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config["model_params"]["random_state"]
    )
    print("✅ Data split completed.")

    # Scale features while retaining the DataFrame structure
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Convert the scaled array back into a DataFrame with the same columns
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # Hyperparameter tuning using validation_curve
    print("🔍 Tuning hyperparameters...")
    train_scores, val_scores = validation_curve(
        Ridge(random_state=config["model_params"]["random_state"]),
        X_train_scaled, y_train,
        param_name="alpha",
        param_range=config["model_params"]["alpha_range"],
        cv=config["model_params"]["cv_folds"],
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )
    mean_val_scores = -np.mean(val_scores, axis=1)
    best_alpha_idx = np.argmin(mean_val_scores)
    best_alpha = config["model_params"]["alpha_range"][best_alpha_idx]
    print(f"🏆 Best alpha selected: {best_alpha:.4f}")

    # Train the final model
    model = Ridge(alpha=best_alpha, random_state=config["model_params"]["random_state"])
    model.fit(X_train_scaled, y_train)
    print("✅ Model training completed.")

    # Evaluate the model
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    metrics = {
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "cv_mse": np.mean(-cross_val_score(
            model, X_train_scaled, y_train,
            cv=config["model_params"]["cv_folds"],
            scoring="neg_mean_squared_error"
        ))
    }
    print("\n📊 Model Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return model, scaler, metrics

# ---------------------------
# Main Integrated Pipeline
# ---------------------------
def main():
    # Configuration settings
    config = {
        "paths": {
            # Path for training data (JSON file)
            "data": r"C:\Users\Admin\OneDrive - Aayush\Desktop\pricing_model\data\pricing_data.json",
            # SQLite database file path
            "db": r"C:\Users\Admin\OneDrive - Aayush\Desktop\pricing_model\data\db.sqlite3"
        },
        "model_params": {
            "alpha_range": np.logspace(-3, 3, 13),
            "cv_folds": 5,
            "random_state": 42
        },
        "features": [
            "cost_price", "current_price", "customer_rating",
            "discount", "sales_volume", "profit_margin", "price_ratio"
        ],
        "target": "price_adjustment"
    }

    # Step 1: Load training data from JSON
    if not os.path.exists(config["paths"]["data"]):
        raise FileNotFoundError(f"Data file not found: {config['paths']['data']}")
    df_json = pd.read_json(config["paths"]["data"])
    print("✅ Data loaded from JSON.")

    # Step 2: Connect to the database and ensure tables exist
    conn = connect_db(config["paths"]["db"])
    create_pricing_table(conn)
    create_artifacts_table(conn)
    print("✅ Database tables are ready.")

    # Step 3: Insert or update pricing data into the database
    insert_or_update_pricing_data(conn, df_json)
    print("✅ Pricing data inserted/updated into the database.")

    # Step 4: Fetch data from the pricing table for training
    df_db = fetch_pricing_data(conn)
    print("✅ Fetched data from database for training.")

    # Step 5: Train the model using data from the database
    model, scaler, metrics = train_model(df_db, config)

    # Compute and store feature importance (optional)
    coefficients = pd.DataFrame({
        "Feature": config["features"],
        "Importance": model.coef_
    }).sort_values("Importance", ascending=False)

    # Step 6: Save the model, scaler, and related artifacts into the database
    save_artifact_to_db(conn, "ridge_model.pkl", model)
    save_artifact_to_db(conn, "ridge_scaler.pkl", scaler)
    save_artifact_to_db(conn, "ridge_feature_importance.csv", coefficients)
    save_artifact_to_db(conn, "ridge_performance_metrics.csv", metrics)
    
    # Step 7: Make predictions on all pricing data and update the pricing table
    # Ensure that the features are provided as a DataFrame with the correct column names.
    X_full = df_db[config["features"]].copy()
    X_full_scaled = scaler.transform(X_full)
    # Convert scaled predictions back into a DataFrame to maintain consistency (if needed)
    X_full_scaled = pd.DataFrame(X_full_scaled, columns=X_full.columns, index=X_full.index)
    predictions = model.predict(X_full_scaled)
    for idx, row in df_db.iterrows():
        product_id = int(row["product_id"])
        predicted_adjustment = float(predictions[idx])
        update_price_adjustment(conn, product_id, predicted_adjustment)
    print("✅ Pricing table updated with new price_adjustment predictions.")

    # (Optional) Load the saved model and scaler from the DB and run a test prediction:
    loaded_model = load_artifact_from_db(conn, "ridge_model.pkl")
    loaded_scaler = load_artifact_from_db(conn, "ridge_scaler.pkl")
    # Use a DataFrame for the sample to maintain feature names
    sample = X_full.iloc[[0]]
    sample_scaled = loaded_scaler.transform(sample)
    sample_scaled = pd.DataFrame(sample_scaled, columns=sample.columns, index=sample.index)
    sample_prediction = loaded_model.predict(sample_scaled)
    print(f"Sample prediction for product_id {df_db.iloc[0]['product_id']}: {sample_prediction[0]:.4f}")

    # Close the database connection
    conn.close()
    print("🔒 Database connection closed.")

if __name__ == "__main__":
    main()
