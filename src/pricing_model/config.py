# Define your configuration settings (paths, model parameters, features, target):
import numpy as np

CONFIG = {
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
