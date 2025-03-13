import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import os

class Database:
    def __init__(self):
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        self.db_path = 'data/sales.db'
        self._create_tables()
        
    def _create_tables(self):
        """Create the necessary tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sales table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sales_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE,
            month TEXT,
            customer TEXT,
            style TEXT,
            sku TEXT,
            size TEXT,
            quantity REAL,
            rate REAL,
            amount REAL
        )
        ''')
        
        conn.commit()
        conn.close()
        
    def upload_sales_data(self, df):
        """Upload sales data to SQLite database"""
        try:
            # Clean and prepare the data
            # Remove rows with NaN values in critical columns
            required_columns = ['DATE', 'PCS', 'GROSS AMT']
            df = df.dropna(subset=required_columns)
            
            # Convert dates using a custom function
            def parse_date(date_str):
                try:
                    # Format is MM-DD-YY
                    month, day, year = map(int, date_str.split('-'))
                    # Convert 2-digit year to 4-digit year
                    year = 2000 + year if year < 50 else 1900 + year
                    return f"{year}-{month:02d}-{day:02d}"  # Return as ISO format string
                except:
                    return None
            
            # Prepare the data
            data = []
            for _, row in df.iterrows():
                date = parse_date(row['DATE'])
                if date is None:
                    continue
                    
                record = (
                    date,
                    str(row['Months']),
                    str(row['CUSTOMER']),
                    str(row['Style']),
                    str(row['SKU']),
                    str(row['Size']),
                    float(row['PCS']),
                    float(row['RATE']),
                    float(row['GROSS AMT'])
                )
                data.append(record)
            
            # Insert data in batches
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            batch_size = 1000
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                cursor.executemany('''
                INSERT INTO sales_data (
                    date, month, customer, style, sku,
                    size, quantity, rate, amount
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', batch)
                conn.commit()
                print(f"Uploaded batch {i//batch_size + 1}")
            
            conn.close()
                
        except Exception as e:
            print(f"Error uploading data: {str(e)}")
            raise
            
    def get_sales_data(self):
        """Retrieve sales data from SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT * FROM sales_data"
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if not df.empty:
                df['Date'] = pd.to_datetime(df['date'])
                df['Sales'] = df['amount']
            return df
            
        except Exception as e:
            print(f"Error retrieving data: {str(e)}")
            raise
    
    def update_sales_data(self, df):
        """Update existing sales data"""
        try:
            # First, clear existing data
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sales_data")
            conn.commit()
            conn.close()
            
            # Then upload new data
            self.upload_sales_data(df)
        except Exception as e:
            print(f"Error updating data: {str(e)}")
            raise
        
    def get_latest_sales_date(self):
        """Get the most recent date in the sales data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT date FROM sales_data ORDER BY date DESC LIMIT 1")
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return pd.to_datetime(result[0])
            return None
        except Exception as e:
            print(f"Error getting latest date: {str(e)}")
            raise 