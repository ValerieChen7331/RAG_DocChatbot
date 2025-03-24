# database_base.py
from pathlib import Path
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)

class BaseDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
    def ensure_db_path_exists(self):
        """確保資料庫文件夾存在。"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def execute_query(self, query: str, params=()):
        """執行資料庫的寫入操作。"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(query, params)
                conn.commit()
        except sqlite3.OperationalError as e:
            logging.error(f"execute_query 資料庫操作錯誤: {e}")
            raise

    def fetch_query(self, query: str, params=()):
        """執行資料庫的查詢操作並返回結果。"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                return cursor.fetchall()
        except sqlite3.OperationalError as e:
            logging.error(f"fetch_query 資料庫操作錯誤: {e}")
            raise