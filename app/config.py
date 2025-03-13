import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DB_PATH = 'data/sales.db'

# Application settings
DEBUG = True

# Table name for sales data
SALES_TABLE = 'sales_data' 