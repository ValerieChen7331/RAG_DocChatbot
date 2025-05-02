import logging
import json
from datetime import datetime
from models.database_base import BaseDB
from apis.file_paths import FilePaths

# 設定 logging 等級為 INFO
logging.basicConfig(level=logging.INFO)

class DeveloperDB:
    def __init__(self):
        """初始化 DeveloperDB 類別：建立資料庫連線與初始化資料表"""
        file_paths = FilePaths()
        self.db_path = file_paths.get_developer_dir().joinpath('DeveloperDB.db')
        self.base_db = BaseDB(self.db_path)
        self.base_db.ensure_db_path_exists()
        self._init_tables()  # 初始化所有資料表

    def _init_tables(self):
        """初始化資料表，若不存在則建立"""
        table_creation_queries = {
            "chat_history": '''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY,
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    username TEXT, agent TEXT, mode TEXT, llm_option TEXT, model TEXT,
                    db_source TEXT, db_name TEXT, conversation_id TEXT,
                    active_window_index INTEGER, num_chat_windows INTEGER,
                    title TEXT, user_query TEXT, ai_response TEXT
                )''',
            "pdf_uploads": '''
                CREATE TABLE IF NOT EXISTS pdf_uploads (
                    id INTEGER PRIMARY KEY,
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    username TEXT, conversation_id TEXT,
                    agent TEXT, embedding TEXT
                )''',
            "file_names": '''
                CREATE TABLE IF NOT EXISTS file_names (
                    id INTEGER PRIMARY KEY,
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    username TEXT, conversation_id TEXT,
                    tmp_name TEXT, org_name TEXT, doc_summary TEXT
                )''',
            "rag_history": '''
                CREATE TABLE IF NOT EXISTS rag_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    conversation_id TEXT,
                    query TEXT,
                    rewritten_query TEXT,
                    response TEXT,
                    timestamp TEXT
                )''',
            "retrieved_docs": '''
                CREATE TABLE IF NOT EXISTS retrieved_docs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rag_history_id INTEGER,
                    doc_index INTEGER,
                    content TEXT,
                    metadata TEXT,
                    FOREIGN KEY (rag_history_id) REFERENCES rag_history(id)
                )''',
            "error_logs": '''
                CREATE TABLE IF NOT EXISTS error_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    context TEXT,
                    error_message TEXT
                )'''
        }

        created_tables = []
        for name, sql in table_creation_queries.items():
            try:
                self.base_db.execute_query(sql)
                result = self.base_db.fetch_query(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
                if result:
                    created_tables.append(name)
            except Exception as e:
                self._log_error(f"init_table:{name}", str(e))

        if created_tables:
            logging.info(f"✅ DeveloperDB 初始化完成，建立表格：{', '.join(created_tables)}")

    def _get_timestamp(self) -> str:
        """取得當前時間字串（統一格式）"""
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _log_error(self, context: str, message: str):
        """將錯誤記錄寫入 error_logs 資料表"""
        try:
            self.base_db.execute_query(
                """
                INSERT INTO error_logs (timestamp, context, error_message)
                VALUES (?, ?, ?)
                """,
                (self._get_timestamp(), context, message)
            )
            logging.error(f"❌ [{context}] 錯誤已寫入 error_logs：{message}")
        except Exception as log_err:
            logging.critical(f"❌❌ 無法寫入 error_logs，原始錯誤：{message}，記錄錯誤：{log_err}")

    def save_chat_history(self, query: str, response: str, chat_session_data: dict):
        """
        將聊天記錄保存到 chat_history 表格。
        """
        upload_time = self._get_timestamp()

        username = chat_session_data.get('username', '')
        agent = chat_session_data.get('agent', '')
        mode = chat_session_data.get('mode', '')
        llm_option = chat_session_data.get('llm_option', '')
        model = chat_session_data.get('model', '')
        db_source = chat_session_data.get('db_source', '')
        db_name = chat_session_data.get('db_name', '')
        conversation_id = chat_session_data.get('conversation_id', '')
        active_window_index = chat_session_data.get('active_window_index', 0)
        num_chat_windows = chat_session_data.get('num_chat_windows', 0)
        title = chat_session_data.get('title', '')

        try:
            self.base_db.execute_query(
                """
                INSERT INTO chat_history 
                (upload_time, username, agent, mode, llm_option, model, db_source, db_name,
                 conversation_id, active_window_index, num_chat_windows, title,
                 user_query, ai_response) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    upload_time, username, agent, mode, llm_option, model, db_source, db_name,
                    conversation_id, active_window_index, num_chat_windows, title,
                    query, response
                )
            )
            logging.info("✅ chat_history 已成功寫入 DeveloperDB。")
        except Exception as e:
            logging.error(f"❌ 儲存 chat_history 時發生錯誤: {e}")

    def save_to_pdf_uploads(self, chat_session_data: dict):
        """將 PDF 上傳紀錄寫入 pdf_uploads 表格"""
        upload_time = self._get_timestamp()
        username = chat_session_data.get("username", "")
        conversation_id = chat_session_data.get("conversation_id", "")
        agent = chat_session_data.get("agent", "")
        embedding = chat_session_data.get("embedding", "")

        try:
            self.base_db.execute_query(
                """
                INSERT INTO pdf_uploads 
                (upload_time, username, conversation_id, agent, embedding) 
                VALUES (?, ?, ?, ?, ?)
                """,
                (upload_time, username, conversation_id, agent, embedding)
            )
            logging.info("✅ pdf_uploads 已成功寫入 DeveloperDB。")
        except Exception as e:
            logging.error(f"❌ 儲存 pdf_uploads 時發生錯誤: {e}")

    def save_to_file_names(self, chat_session_data: dict, doc_summary: str, tmp_name: str, org_name: str):
        """將檔案名稱與摘要寫入 file_names 表格"""
        upload_time = self._get_timestamp()
        username = chat_session_data.get('username', '')
        conversation_id = chat_session_data.get('conversation_id', '')

        try:
            self.base_db.execute_query(
                """
                INSERT INTO file_names 
                (upload_time, username, conversation_id, tmp_name, org_name, doc_summary) 
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (upload_time, username, conversation_id, tmp_name, org_name, doc_summary)
            )
            logging.info(f"✅ file_names 已寫入 DeveloperDB：{org_name}")
        except Exception as e:
            logging.error(f"❌ 儲存 file_names 時發生錯誤: {e}")

    def save_retrieved_data_to_db(self, query: str, rewritten_query: str, retrieved_data: list, response: str, chat_session_data: dict):
        """
        儲存 RAG 查詢與檢索內容至 rag_history（主表）與 retrieved_docs（子表）
        """
        conversation_id = chat_session_data.get("conversation_id", "")
        username = chat_session_data.get("username", "unknown")

        try:
            timestamp = self._get_timestamp()
            rag_history_id = self.base_db.execute_query(
                """
                INSERT INTO rag_history (username, conversation_id, query, rewritten_query, response, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (username, conversation_id, query, rewritten_query, response, timestamp),
                return_lastrowid=True
            )

            if rag_history_id is None:
                logging.error("❌ 無法取得 rag_history_id，主表插入失敗")
                return

            for idx, doc in enumerate(retrieved_data, start=1):
                content = getattr(doc, "page_content", str(doc))
                metadata = json.dumps(getattr(doc, "metadata", {}), ensure_ascii=False)

                self.base_db.execute_query(
                    """
                    INSERT INTO retrieved_docs (rag_history_id, doc_index, content, metadata)
                    VALUES (?, ?, ?, ?)
                    """,
                    (rag_history_id, idx, content, metadata)
                )

            logging.info(f"✅ 已儲存 RAG 查詢與 {len(retrieved_data)} 筆檢索結果 (rag_history_id={rag_history_id})")

        except Exception as e:
            logging.error(f"❌ 儲存 RAG 資料時發生錯誤: {e}")
