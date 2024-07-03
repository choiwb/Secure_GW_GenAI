
import sqlite3
import os

from config import token_db_path

DB_FILE = os.path.join(token_db_path, "token_debug.db")

# 데이터베이스 파일이 있는 디렉토리 생성
os.makedirs(token_db_path, exist_ok=True)

def token_debug_init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS token_debug (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question VARCHAR,
            context VARCHAR,
            answer VARCHAR
        )
    ''')
    conn.commit()
    conn.close()

def record_token_debug(question, context, answer):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO token_debug (question, context, answer)
        VALUES (?, ?, ?)
    ''', (question, context, answer))
    conn.commit()
    conn.close()


