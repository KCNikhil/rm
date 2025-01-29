# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before other imports
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # Configuration
    config = {
        "input_csv": os.path.join("data", "oneupdated_data_with_random_values.csv"),
        "output_dir": "Curves",
        "model_dir": "models",
        "features": [
            'cost_price', 'current_price', 'customer_rating', 
            'discount', 'sales_volume', 'profit_margin', 'price_ratio'
        ],
        "target": 'price_adjustment',
        "params": {
            "train_sizes": np.linspace(0.1, 1.0, 10),
            "random_state": 42
        }
    }

    try:
        # Create directories if they don't exist
        os.makedirs(config["output_dir"], exist_ok=True)
        os.makedirs(config["model_dir"], exist_ok=True)

        # Load and validate data
        if not os.path.exists(config["input_csv"]):
            raise FileNotFoundError(f"Input file not found: {config['input_csv']}")

        df = pd.read_csv(config["input_csv"])
        print("✅ Data loaded successfully")

        # Data cleaning
        initial_count = len(df)
        df = df.dropna(subset=[config["target"]])
        cleaned_count = len(df)
        if cleaned_count < initial_count:
            print(f"⚠️  Removed {initial_count - cleaned_count} rows with missing target values")

        # Prepare features and target
        X = df[config["features"]]
        y = df[config["target"]]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=config["params"]["random_state"]
        )
        print("✅ Data split completed")

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model training
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        print("✅ Model training completed")

        # Generate predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # Calculate metrics
        metrics = {
            "train_mse": mean_squared_error(y_train, y_train_pred),
            "test_mse": mean_squared_error(y_test, y_test_pred),
            "train_r2": r2_score(y_train, y_train_pred),
            "test_r2": r2_score(y_test, y_test_pred)
        }

        # Generate learning curves
        train_sizes, train_scores, val_scores = learning_curve(
            model,
            X_train_scaled,
            y_train,
            train_sizes=config["params"]["train_sizes"],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        # Save artifacts
        def save_artifact(obj, filename):
            try:
                path = os.path.join(config["model_dir"], filename)
                joblib.dump(obj, path) if filename.endswith('.pkl') else obj.to_csv(path, index=False)
                print(f"✅ Saved {filename}")
            except Exception as e:
                print(f"⚠️  Failed to save {filename}: {str(e)}")

        save_artifact(model, "multiple_regression_model.pkl")
        save_artifact(scaler, "multiple_regression_scaler.pkl")
        save_artifact(pd.DataFrame({
            'Dataset': ['Train', 'Test'],
            'MSE': [metrics["train_mse"], metrics["test_mse"]],
            'R²': [metrics["train_r2"], metrics["test_r2"]]
        }), "multiple_regression_performance.csv")

        # Plotting functions
        def save_plot(figure_func, filename):
            try:
                plt.figure(figsize=(10, 6))
                figure_func()
                plt.tight_layout()
                plt.savefig(os.path.join(config["output_dir"], filename))
                plt.close()
                print(f"✅ Saved plot: {filename}")
            except Exception as e:
                print(f"⚠️  Failed to save {filename}: {str(e)}")

        # 1. Actual vs Predicted Plot
        save_plot(lambda: (
            plt.scatter(y_test, y_test_pred, alpha=0.5, color='blue'),
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2),
            plt.title('Multiple Regression: Actual vs Predicted Values'),
            plt.xlabel('Actual Values'),
            plt.ylabel('Predicted Values'),
            plt.grid(True)
        ), "multiple_actual_vs_predicted.png")

        # 2. Learning Curve
        save_plot(lambda: (
            plt.plot(train_sizes, -np.mean(train_scores, axis=1), 'o-', label='Training MSE'),
            plt.plot(train_sizes, -np.mean(val_scores, axis=1), 'o-', label='Validation MSE'),
            plt.title('Multiple Regression Learning Curve'),
            plt.xlabel('Training Set Size'),
            plt.ylabel('Mean Squared Error'),
            plt.legend(),
            plt.grid(True)
        ), "multiple_learning_curve.png")

        # 3. Residual Plot
        save_plot(lambda: (
            plt.scatter(y_test_pred, y_test - y_test_pred, alpha=0.5, color='green'),
            plt.axhline(0, color='r', linestyle='--'),
            plt.title('Multiple Regression Residual Analysis'),
            plt.xlabel('Predicted Values'),
            plt.ylabel('Residuals'),
            plt.grid(True)
        ), "multiple_residual_plot.png")

        # 4. Validation Curve (Dummy implementation - linear regression has no hyperparameters)
        save_plot(lambda: (
            plt.text(0.4, 0.5, 'No hyperparameters to validate\nin Multiple Regression', 
                    ha='center', va='center', fontsize=12),
            plt.title('Multiple Regression Validation Curve'),
            plt.axis('off')
        ), "multiple_validation_loss_curve.png")

        # Final report
        print("\n📊 Multiple Regression Final Metrics:")
        print(f"Training MSE: {metrics['train_mse']:.4f}")
        print(f"Testing MSE:  {metrics['test_mse']:.4f}")
        print(f"Training R²:  {metrics['train_r2']:.4f}")
        print(f"Testing R²:   {metrics['test_r2']:.4f}")

    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()