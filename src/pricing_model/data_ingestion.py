# A script to ingest new data into your database. In production, you may trigger this periodically or via an API.

import os
import pandas as pd
from db_utils import connect_db, create_pricing_table, create_artifacts_table, insert_or_update_pricing_data
from config import CONFIG

def ingest_data():
    data_path = CONFIG["paths"]["data"]
    db_path = CONFIG["paths"]["db"]

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_json(data_path)
    print("✅ Data loaded from JSON.")

    conn = connect_db(db_path)
    create_pricing_table(conn)
    create_artifacts_table(conn)
    print("✅ Database tables are ready.")

    insert_or_update_pricing_data(conn, df)
    print("✅ Pricing data inserted/updated into the database.")

    conn.close()

if __name__ == "__main__":
    ingest_data()
