# Centralize your database helper functions (connection, table creation, data insertion, artifact storage, etc.)
import os
import sqlite3
import pickle
import pandas as pd

def connect_db(db_path):
    """Connect to the SQLite database."""
    if not os.path.exists(db_path):
        # If the database file does not exist, create an empty file.
        open(db_path, 'a').close()
    return sqlite3.connect(db_path)

def create_pricing_table(conn):
    """Create the pricing table if it does not exist."""
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
    """Create the artifacts table for storing model objects as BLOBs."""
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
    If a product_id exists, update its record.
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
            row.get('price_adjustment', 0.0)
        ))
    conn.commit()

def fetch_pricing_data(conn):
    """Fetch all pricing data from the database as a DataFrame."""
    df = pd.read_sql_query("SELECT * FROM pricing;", conn)
    return df

def update_price_adjustment(conn, product_id, predicted_adjustment):
    """Update the price_adjustment for a given product_id."""
    conn.execute("UPDATE pricing SET price_adjustment = ? WHERE product_id = ?;",
                 (predicted_adjustment, product_id))
    conn.commit()

def save_artifact_to_db(conn, artifact_name, artifact_obj):
    """Serialize an artifact and store it in the artifacts table."""
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
