
import sqlite3
import os
from datetime import datetime

from config import token_db_path

DB_FILE = os.path.join(token_db_path, "token_usage.db")

# 데이터베이스 파일이 있는 디렉토리 생성
os.makedirs(token_db_path, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS token_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER,
            month INTEGER,
            usage INTEGER,
            fee REAL DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def record_token_usage(tokens):
    now = datetime.now()
    year = now.year
    month = now.month
    fee = tokens * 0.005
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT usage, fee FROM token_usage WHERE year=? AND month=?', (year, month))
    result = c.fetchone()
    if result:
        new_usage = result[0] + tokens
        new_fee = result[1] + fee
        c.execute('UPDATE token_usage SET usage=?, fee=? WHERE year=? AND month=?', (new_usage, new_fee, year, month))
    else:
        c.execute('INSERT INTO token_usage (year, month, usage, fee) VALUES (?, ?, ?, ?)', (year, month, tokens, fee))
    print(f"{tokens} 저장완료, {fee} 요금 발생")
    conn.commit()
    conn.close()

def get_token_usage(year, month):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT usage, fee FROM token_usage WHERE year=? AND month=?', (year, month))
    result = c.fetchone()
    conn.close()
    return (result[0], result[1]) if result else (0, 0)
