import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.model_selection import train_test_split, validation_curve, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # Configuration
    config = {
        "paths": {
            # Corrected data path (removed extra quotes)
            "data": r"C:\Users\Admin\OneDrive - Aayush\Desktop\pricing_model\data\aggregated_output_updated_copy.csv",
            "output_dir": r'C:\Users\Admin\OneDrive - Aayush\Desktop\pricing_model\Curves',
            "model_dir": r'C:\Users\Admin\OneDrive - Aayush\Desktop\pricing_model\models'
        },
        "model_params": {
            "alpha_range": np.logspace(-3, 3, 13),
            "cv_folds": 5,
            "random_state": 42,
            "learning_curve_sizes": np.linspace(0.1, 1.0, 10)
        },
        "features": [
            'cost_price', 'current_price', 'customer_rating',
            'discount', 'sales_volume', 'profit_margin', 'price_ratio'
        ],
        "target": 'price_adjustment'
    }

    try:
        # Create directories if they do not exist
        os.makedirs(config["paths"]["output_dir"], exist_ok=True)
        os.makedirs(config["paths"]["model_dir"], exist_ok=True)
        print("✅ Directories created successfully")

        # Load and validate data
        if not os.path.exists(config["paths"]["data"]):
            raise FileNotFoundError(f"Data file not found: {config['paths']['data']}")
            
        df = pd.read_csv(config["paths"]["data"])
        print("✅ Data loaded successfully")

        # Data cleaning: remove rows with missing target values
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
            random_state=config["model_params"]["random_state"]
        )
        print("✅ Data split completed")

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model training with validation curve for hyperparameter tuning
        print("🔍 Tuning hyperparameters...")
        train_scores, val_scores = validation_curve(
            Ridge(random_state=config["model_params"]["random_state"]),
            X_train_scaled,
            y_train,
            param_name="alpha",
            param_range=config["model_params"]["alpha_range"],
            cv=config["model_params"]["cv_folds"],
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )

        # Find best alpha based on minimum validation error
        mean_val_scores = -np.mean(val_scores, axis=1)
        best_alpha_idx = np.argmin(mean_val_scores)
        best_alpha = config["model_params"]["alpha_range"][best_alpha_idx]

        # Train final model with the best alpha
        print(f"🏆 Best alpha selected: {best_alpha:.4f}")
        model = Ridge(alpha=best_alpha, random_state=config["model_params"]["random_state"])
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
            "test_r2": r2_score(y_test, y_test_pred),
            "cv_mse": np.mean(-cross_val_score(
                model, X_train_scaled, y_train,
                cv=config["model_params"]["cv_folds"],
                scoring="neg_mean_squared_error"
            ))
        }

        # Feature importance based on model coefficients
        coefficients = pd.DataFrame({
            'Feature': config["features"],
            'Importance': model.coef_
        }).sort_values('Importance', ascending=False)

        # Generate learning curves
        print("📈 Generating learning curves...")
        train_sizes, train_scores_lc, val_scores_lc = learning_curve(
            model,
            X_train_scaled,
            y_train,
            cv=config["model_params"]["cv_folds"],
            train_sizes=config["model_params"]["learning_curve_sizes"],
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        # Save artifacts (model, scaler, feature importance, performance metrics)
        def save_artifact(obj, filename):
            try:
                path = os.path.join(config["paths"]["model_dir"], filename)
                if filename.endswith('.pkl'):
                    joblib.dump(obj, path)
                else:
                    obj.to_csv(path, index=False)
                print(f"✅ Saved {filename}")
            except Exception as e:
                print(f"⚠️  Failed to save {filename}: {str(e)}")

        # Update the model files so that new training replaces the old models
        save_artifact(model, "ridge_model.pkl")
        save_artifact(scaler, "ridge_scaler.pkl")
        save_artifact(coefficients, "ridge_feature_importance.csv")
        save_artifact(pd.DataFrame({
            'Metric': ['Train MSE', 'Test MSE', 'CV MSE', 'Train R²', 'Test R²'],
            'Value': [
                metrics["train_mse"],
                metrics["test_mse"],
                metrics["cv_mse"],
                metrics["train_r2"],
                metrics["test_r2"]
            ]
        }), "ridge_performance_metrics.csv")

        # Plotting functions with error handling
        def save_plot(figure_func, filename):
            try:
                plt.figure(figsize=(10, 6))
                figure_func()
                plt.tight_layout()
                plt.savefig(os.path.join(config["paths"]["output_dir"], filename))
                plt.close()
                print(f"✅ Saved plot: {filename}")
            except Exception as e:
                print(f"⚠️  Failed to save {filename}: {str(e)}")

        # 1. Actual vs Predicted Plot
        save_plot(lambda: (
            plt.scatter(y_test, y_test_pred, alpha=0.5, color='blue'),
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2),
            plt.title('Ridge Regression: Actual vs Predicted Values'),
            plt.xlabel('Actual Values'),
            plt.ylabel('Predicted Values'),
            plt.grid(True)
        ), "ridge_actual_vs_predicted.png")

        # 2. Learning Curve
        save_plot(lambda: (
            plt.plot(train_sizes, -np.mean(train_scores_lc, axis=1), 'o-', label='Training MSE'),
            plt.plot(train_sizes, -np.mean(val_scores_lc, axis=1), 'o-', label='Validation MSE'),
            plt.title('Ridge Regression Learning Curve'),
            plt.xlabel('Training Set Size'),
            plt.ylabel('Mean Squared Error'),
            plt.legend(),
            plt.grid(True)
        ), "ridge_learning_curve.png")

        # 3. Validation Curve (Loss Curve)
        save_plot(lambda: (
            plt.plot(config["model_params"]["alpha_range"], -np.mean(train_scores, axis=1), label='Training MSE'),
            plt.plot(config["model_params"]["alpha_range"], -np.mean(val_scores, axis=1), label='Validation MSE'),
            plt.xscale('log'),
            plt.axvline(best_alpha, color='r', linestyle='--', label=f'Best Alpha: {best_alpha:.2f}'),
            plt.title('Ridge Regression Validation (Loss) Curve'),
            plt.xlabel('Alpha (Regularization Strength)'),
            plt.ylabel('Mean Squared Error'),
            plt.legend(),
            plt.grid(True)
        ), "ridge_validation_loss_curve.png")

        # 4. Residual Plot
        save_plot(lambda: (
            plt.scatter(y_test_pred, y_test - y_test_pred, alpha=0.5, color='green'),
            plt.axhline(0, color='r', linestyle='--'),
            plt.title('Ridge Regression Residual Analysis'),
            plt.xlabel('Predicted Values'),
            plt.ylabel('Residuals'),
            plt.grid(True)
        ), "ridge_residual_plot.png")

        # Final report
        print("\n📊 Ridge Regression Final Metrics:")
        print(f"Training MSE: {metrics['train_mse']:.4f}")
        print(f"Testing MSE:  {metrics['test_mse']:.4f}")
        print(f"Cross-Validation MSE: {metrics['cv_mse']:.4f}")
        print(f"Training R²:  {metrics['train_r2']:.4f}")
        print(f"Testing R²:   {metrics['test_r2']:.4f}")

    except Exception as e:
        print(f"\n❌ Critical Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
