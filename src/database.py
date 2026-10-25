import sqlite3
import pandas as pd
import datetime
import os

DB_PATH = "data/econocausal.db"

def get_connection():
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    """Initializes tables for scored customers and campaign decision logs."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Customer Scoring Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customer_scores (
        customer_id INTEGER PRIMARY KEY,
        recency INTEGER,
        history REAL,
        browsing_mins REAL,
        cart_abandon_val REAL,
        email_engagement REAL,
        predicted_ite REAL,
        causal_segment TEXT,
        last_updated TIMESTAMP
    )
    """)
    
    # Campaign Allocation Audit Logs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS campaign_logs (
        campaign_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP,
        budget REAL,
        coupon_value REAL,
        targeted_users INTEGER,
        expected_extra_sales REAL,
        net_roi REAL,
        notes TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_scored_customers(df):
    """Saves or updates customer causal predictions in SQLite."""
    init_db()
    conn = get_connection()
    df_to_save = df.copy()
    if 'customer_id' not in df_to_save.columns:
        df_to_save['customer_id'] = range(1001, 1001 + len(df_to_save))
    df_to_save['last_updated'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cols = ['customer_id', 'recency', 'history', 'browsing_mins', 'cart_abandon_val', 
            'email_engagement', 'predicted_ite', 'causal_segment', 'last_updated']
    df_to_save[cols].to_sql('customer_scores', conn, if_exists='replace', index=False)
    conn.close()

def load_scored_customers():
    """Loads scored customers directly from SQLite."""
    init_db()
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM customer_scores", conn)
    conn.close()
    return df

def log_campaign(budget, coupon_val, targeted_users, expected_sales, net_roi, notes="Standard Run"):
    """Logs an executed optimization campaign to the audit table."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO campaign_logs (timestamp, budget, coupon_value, targeted_users, expected_extra_sales, net_roi, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        budget,
        coupon_val,
        targeted_users,
        expected_sales,
        net_roi,
        notes
    ))
    conn.commit()
    conn.close()

def load_campaign_history():
    """Retrieves all past logged campaigns."""
    init_db()
    conn = get_connection()
    df_logs = pd.read_sql("SELECT * FROM campaign_logs ORDER BY campaign_id DESC", conn)
    conn.close()
    return df_logs