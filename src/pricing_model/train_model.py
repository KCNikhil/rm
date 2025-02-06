# A script dedicated to training your model, saving artifacts to the database, and updating predictions. Run this separately when you want to retrain.

import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.model_selection import train_test_split, validation_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

from db_utils import connect_db, fetch_pricing_data, save_artifact_to_db, update_price_adjustment
from config import CONFIG

# Optionally suppress specific warnings if the data format is consistent
warnings.filterwarnings("ignore", message="X does not have valid feature names")

def train_and_save_model():
    db_path = CONFIG["paths"]["db"]
    conn = connect_db(db_path)

    df = fetch_pricing_data(conn)
    print("✅ Fetched data from database for training.")

    # Remove rows missing the target
    initial_count = len(df)
    df = df.dropna(subset=[CONFIG["target"]])
    if len(df) < initial_count:
        print(f"⚠️ Removed {initial_count - len(df)} rows with missing target values.")

    X = df[CONFIG["features"]].copy()
    y = df[CONFIG["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=CONFIG["model_params"]["random_state"]
    )
    print("✅ Data split completed.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Optionally, convert back to DataFrame to retain column names:
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    print("🔍 Tuning hyperparameters...")
    train_scores, val_scores = validation_curve(
        Ridge(random_state=CONFIG["model_params"]["random_state"]),
        X_train_scaled, y_train,
        param_name="alpha",
        param_range=CONFIG["model_params"]["alpha_range"],
        cv=CONFIG["model_params"]["cv_folds"],
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )
    mean_val_scores = -np.mean(val_scores, axis=1)
    best_alpha_idx = np.argmin(mean_val_scores)
    best_alpha = CONFIG["model_params"]["alpha_range"][best_alpha_idx]
    print(f"🏆 Best alpha selected: {best_alpha:.4f}")

    model = Ridge(alpha=best_alpha, random_state=CONFIG["model_params"]["random_state"])
    model.fit(X_train_scaled, y_train)
    print("✅ Model training completed.")

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    metrics = {
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "cv_mse": np.mean(-cross_val_score(
            model, X_train_scaled, y_train,
            cv=CONFIG["model_params"]["cv_folds"],
            scoring="neg_mean_squared_error"
        ))
    }
    print("📊 Model Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Optional: compute feature importance
    coefficients = pd.DataFrame({
        "Feature": CONFIG["features"],
        "Importance": model.coef_
    }).sort_values("Importance", ascending=False)

    # Save model artifacts into the database
    save_artifact_to_db(conn, "ridge_model.pkl", model)
    save_artifact_to_db(conn, "ridge_scaler.pkl", scaler)
    save_artifact_to_db(conn, "ridge_feature_importance.csv", coefficients)
    save_artifact_to_db(conn, "ridge_performance_metrics.csv", metrics)

    # Make predictions on all pricing data and update the pricing table
    X_full = df[CONFIG["features"]].copy()
    X_full_scaled = scaler.transform(X_full)
    predictions = model.predict(X_full_scaled)
    for idx, row in df.iterrows():
        product_id = int(row["product_id"])
        predicted_adjustment = float(predictions[idx])
        update_price_adjustment(conn, product_id, predicted_adjustment)
    print("✅ Pricing table updated with new price_adjustment predictions.")

    conn.close()

if __name__ == "__main__":
    train_and_save_model()
