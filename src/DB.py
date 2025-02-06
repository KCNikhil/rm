# Centralize your database helper functions (connection, table creation, data insertion, artifact storage, etc.)

import sqlite3
import os

class DatabaseManager:
    def __init__(self, db_path):
        """
        Initialize the DatabaseManager with a path to the SQLite database.
        Checks if the database file exists and then connects to it.
        """
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found at: {self.db_path}")
        self.connection = None
        self.cursor = None
        self.connect()

    def connect(self):
        """Connects to the SQLite database."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.cursor = self.connection.cursor()
            print(f"✅ Connected to database at {self.db_path}")
        except sqlite3.OperationalError as e:
            print("❌ Error connecting to database:", e)

    def list_tables(self):
        """
        Lists all tables in the connected SQLite database.
        Returns a list of table names.
        """
        try:
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = self.cursor.fetchall()
            if tables:
                print("📋 Tables in the database:")
                for table in tables:
                    print(" -", table[0])
            else:
                print("ℹ️  No tables found in the database.")
            return [table[0] for table in tables]
        except Exception as e:
            print("❌ Error listing tables:", e)
            return []

    def execute_query(self, query):
        """
        Executes an arbitrary SQL query.
        If the query returns data, that data is fetched and returned.
        """
        try:
            self.cursor.execute(query)
            self.connection.commit()
            try:
                results = self.cursor.fetchall()
            except sqlite3.ProgrammingError:
                # No results to fetch (e.g., for INSERT or UPDATE)
                results = None
            print("✅ Query executed successfully.")
            return results
        except Exception as e:
            print("❌ Error executing query:", e)
            return None

    def close(self):
        """Closes the database connection."""
        if self.connection:
            self.connection.close()
            print("🔒 Connection closed.")


if __name__ == "__main__":
    # Define the path to your SQLite database file
    db_path = r"C:\Users\Admin\OneDrive - Aayush\Desktop\pricing_model\data\db.sqlite3"
    
    # Create an instance of the DatabaseManager
    db_manager = DatabaseManager(db_path)
    
    # List all tables in the database
    tables = db_manager.list_tables()
    
    # Example: Execute a query on the first table (if any exist)
    if tables:
        first_table = tables[0]
        print(f"\n🔍 Fetching first 5 rows from table '{first_table}':")
        query = f"SELECT * FROM {first_table} LIMIT 5;"
        rows = db_manager.execute_query(query)
        if rows:
            for row in rows:
                print(row)
        else:
            print("No data returned or error occurred.")
    
    # Close the database connection when done
    db_manager.close()
