# database_userRecords.py
import logging
import json
import pandas as pd
from datetime import datetime
from models.database_base import BaseDB
from apis.file_paths import FilePaths
import traceback

# 設定 logging 等級
logging.basicConfig(level=logging.INFO)


class UserRecordsDB:
    def __init__(self, username: str):
        """初始化 UserRecordsDB 類別，建立與使用者相關的資料庫"""
        self.username = username
        self.file_paths = FilePaths()
        self.db_path = self.file_paths.get_user_records_dir(username).joinpath(f"{username}.db")
        self.base_db = BaseDB(self.db_path)
        self.base_db.ensure_db_path_exists()
        self._init_tables()  # 初始化所有資料表（包含聊天、檔案、RAG）

    def _init_tables(self):
        """初始化所有所需的資料表（第一次使用者登入時建立）"""
        # 即使檔案已存在，仍然可確保表格存在（使用 CREATE TABLE IF NOT EXISTS）
        table_creation_queries = {
            "chat_history": """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY,
                    agent TEXT, mode TEXT, llm_option TEXT, model TEXT,
                    db_source TEXT, db_name TEXT, conversation_id TEXT,
                    active_window_index INTEGER, num_chat_windows INTEGER,
                    title TEXT, user_query TEXT, ai_response TEXT
                )""",
            "pdf_uploads": """
                CREATE TABLE IF NOT EXISTS pdf_uploads (
                    id INTEGER PRIMARY KEY,
                    conversation_id TEXT, agent TEXT, embedding TEXT
                )""",
            "file_names": """
                CREATE TABLE IF NOT EXISTS file_names (
                    id INTEGER PRIMARY KEY,
                    conversation_id TEXT, tmp_name TEXT, org_name TEXT, doc_summary TEXT
                )""",
            "rag_history": """
                CREATE TABLE IF NOT EXISTS rag_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT,
                    query TEXT,
                    rewritten_query TEXT,
                    response TEXT,
                    timestamp TEXT
                )""",
            "retrieved_docs": """
                CREATE TABLE IF NOT EXISTS retrieved_docs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rag_history_id INTEGER,
                    doc_index INTEGER,
                    chunk_id INTEGER,
                    content TEXT,
                    metadata TEXT,
                    FOREIGN KEY (rag_history_id) REFERENCES rag_history(id)
                )"""
        }

        created_tables = []

        for table_name, query in table_creation_queries.items():
            try:
                # 嘗試建立表格（若已存在則不會改變）
                self.base_db.execute_query(query)

                # 使用 sqlite_master 檢查是否存在該表
                check_query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
                result = self.base_db.fetch_query(check_query, (table_name,))
                if result:
                    created_tables.append(table_name)
            except Exception as e:
                logging.error(f"❌ 建立資料表 {table_name} 時發生錯誤: {e}")

        if created_tables:
            logging.info(f" 使用者資料庫初始化完成，已建立以下資料表：{', '.join(created_tables)}")

    # ---------------------------------------
    # 載入指定資料表內容（回傳 DataFrame）
    # ---------------------------------------
    def load_table(self, table_name: str, columns: list = None) -> pd.DataFrame:
        """載入指定資料表內容，並以 pandas DataFrame 格式回傳"""
        default_columns = {
            'chat_history': [
                'id', 'agent', 'mode', 'llm_option', 'model',
                'db_source', 'db_name', 'conversation_id', 'active_window_index',
                'num_chat_windows', 'title', 'user_query', 'ai_response'
            ],
            'rag_history': [
                'id', 'conversation_id', 'query', 'rewritten_query', 'response', 'timestamp'
            ],
            'retrieved_docs': [
                'id', 'rag_history_id', 'doc_index', 'chunk_id', 'content', 'metadata'
            ]
        }

        selected_cols = columns or default_columns.get(table_name, ['*'])
        query = f"SELECT {', '.join(selected_cols)} FROM {table_name}"
        try:
            results = self.base_db.fetch_query(query)
            return pd.DataFrame(results, columns=selected_cols) if results else pd.DataFrame(columns=selected_cols)
        except Exception as e:
            logging.error(f"❌ 載入 {table_name} 發生錯誤: {e}")
            return pd.DataFrame(columns=selected_cols)

    # ---------------------------------------
    # 聊天資料：刪除、更新索引
    # ---------------------------------------
    def delete_chat_by_index(self, index: int):
        """刪除指定聊天視窗索引的紀錄"""
        self.base_db.execute_query("DELETE FROM chat_history WHERE active_window_index = ?", (index,))

    def update_chat_indexes(self, deleted_index: int):
        """更新聊天索引（當刪除一筆聊天視窗後）"""
        rows = self.base_db.fetch_query("SELECT id, active_window_index FROM chat_history ORDER BY active_window_index")
        for row_id, window_index in rows:
            if window_index > deleted_index:
                self.base_db.execute_query(
                    "UPDATE chat_history SET active_window_index = ? WHERE id = ?",
                    (window_index - 1, row_id)
                )

    # ---------------------------------------
    # 聊天設定與歷史紀錄查詢
    # ---------------------------------------
    def get_active_window_setup(self, index: int, session_data: dict) -> dict:
        """載入指定聊天視窗的設定與聊天內容，更新至 session_data"""
        setup_cols = ['conversation_id', 'agent', 'mode', 'llm_option', 'model', 'db_source', 'db_name', 'title']
        history_cols = ['user_query', 'ai_response']
        all_cols = setup_cols + history_cols

        query = f"""
            SELECT {', '.join(all_cols)} FROM chat_history 
            WHERE active_window_index = ? ORDER BY id
        """

        try:
            records = self.base_db.fetch_query(query, (index,))
            if not records:
                return {}

            df = pd.DataFrame(records, columns=all_cols)
            for col in setup_cols:
                session_data[col] = df[col].iloc[-1]
            session_data['chat_history'] = df[history_cols].to_dict(orient='records')
            return session_data
        except Exception as e:
            logging.error(f"❌ 載入聊天設定發生錯誤: {e}")
            return {}

    # ---------------------------------------
    # 檔案名稱、摘要查詢
    # ---------------------------------------
    def get_doc_names(self, session_data: dict) -> str:
        """取得目前對話的所有原始檔名，使用逗號分隔"""
        try:
            conv_id = session_data.get('conversation_id')
            rows = self.base_db.fetch_query("SELECT org_name FROM file_names WHERE conversation_id = ?", (conv_id,))
            return ", ".join(row[0] for row in rows) if rows else ""
        except Exception as e:
            logging.error(f"❌ 取得 org_name 錯誤: {e}")
            return ""

    def get_doc_summaries(self, session_data: dict) -> str:
        """取得每個檔案的摘要文字，格式化為【檔名】摘要：內容"""
        try:
            conv_id = session_data.get('conversation_id')
            rows = self.base_db.fetch_query("SELECT org_name, doc_summary FROM file_names WHERE conversation_id = ?", (conv_id,))
            return "\n\n".join(f"【{r[0]}】摘要：{r[1]}" for r in rows) if rows else ""
        except Exception as e:
            logging.error(f"❌ 取得摘要錯誤: {e}")
            return ""

    def get_org_file_name(self, session_data: dict, tmp_name: str) -> str:
        """根據暫存檔名查詢原始檔名"""
        conv_id = session_data.get("conversation_id")
        if not conv_id or not tmp_name:
            logging.warning("⚠️ conversation_id 或 tmp_name 為空，無法查詢 org_file_name")
            return ""
        try:
            result = self.base_db.fetch_query(
                "SELECT org_name FROM file_names WHERE conversation_id = ? AND tmp_name = ?",
                (conv_id, tmp_name)
            )
            return result[0][0] if result else ""
        except Exception as e:
            logging.error(f"❌ 查詢 org_file_name 發生錯誤: {e}")
            return ""

    # ---------------------------------------
    # 資料儲存：chat、uploads、file_names
    # ---------------------------------------
    def save_to_database(self, query: str, response: str, session_data: dict):
        """儲存一筆聊天資料到 chat_history 表"""
        data = {key: session_data.get(key, default) for key, default in {
            'agent': None, 'mode': None, 'llm_option': None, 'model': None,
            'db_source': None, 'db_name': None, 'conversation_id': None,
            'active_window_index': 0, 'num_chat_windows': 0,
            'title': None, 'user_query': query, 'ai_response': response
        }.items()}

        try:
            self.base_db.execute_query(
                """
                INSERT INTO chat_history 
                (agent, mode, llm_option, model, db_source, db_name, 
                 conversation_id, active_window_index, num_chat_windows, title,
                 user_query, ai_response) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(data.values())
            )
            logging.info("✅ 成功儲存至 chat_history")
        except Exception as e:
            logging.error(f"❌ 儲存 chat_history 時發生錯誤: {e}")

    def save_to_pdf_uploads(self, session_data: dict):
        """儲存 PDF 上傳記錄到 pdf_uploads 表"""
        data = {key: session_data.get(key, default) for key, default in {
            'conversation_id': None, 'agent': None, 'embedding': None
        }.items()}

        try:
            self.base_db.execute_query(
                "INSERT INTO pdf_uploads (conversation_id, agent, embedding) VALUES (?, ?, ?)",
                tuple(data.values())
            )
            logging.info("✅ 成功儲存至 pdf_uploads")
        except Exception as e:
            logging.error(f"❌ 儲存 pdf_uploads 發生錯誤: {e}")

    def save_to_file_names(self, session_data: dict, doc_summary: str, tmp_name: str, org_name: str):
        """儲存檔案名稱與摘要資料至 file_names 表"""
        try:
            self.base_db.execute_query(
                "INSERT INTO file_names (conversation_id, tmp_name, org_name, doc_summary) VALUES (?, ?, ?, ?)",
                (session_data.get('conversation_id'), tmp_name, org_name, doc_summary)
            )
            logging.info(f"✅ 儲存 file_names 成功：{org_name}")
        except Exception as e:
            logging.error(f"❌ 儲存 file_names 發生錯誤: {e}")

    # ---------------------------------------
    # 儲存：RAG 查詢與文件結果
    # ---------------------------------------
    def save_retrieved_data_to_db(self, query: str, rewritten_query: str, retrieved_data: list, response: str,
                                  chat_session_data: dict):
        """
        儲存 RAG 查詢與檢索內容至資料庫中的 rag_history（主表）與 retrieved_docs（子表）
        - query: 使用者原始提問
        - rewritten_query: 重寫後的查詢內容
        - retrieved_data: 檢索結果列表，每個元素為 Document
        - response: AI 回覆內容
        - chat_session_data: 包含 conversation_id 的字典
        """
        try:
            timestamp = datetime.now().isoformat()
            conversation_id = chat_session_data.get("conversation_id", "")

            # 🔹 儲存主查詢至 rag_history 表，取得自動產生的主鍵 ID
            rag_history_id = self.base_db.execute_query(
                """
                INSERT INTO rag_history (conversation_id, query, rewritten_query, response, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, query, rewritten_query, response, timestamp),
                return_lastrowid=True
            )

            # 🔸 若主鍵插入失敗則中止
            if rag_history_id is None:
                logging.error("❌ 無法取得 rag_history_id，主表插入可能失敗")
                return

            # 🔹 儲存每筆檢索文件至 retrieved_docs 子表
            for idx, doc in enumerate(retrieved_data, start=1):
                content = getattr(doc, "page_content", str(doc))
                metadata_dict = getattr(doc, "metadata", {})
                metadata = json.dumps(metadata_dict, ensure_ascii=False)
                chunk_id = metadata_dict.get("chunk_id", 1000)  # ✅ 正確從 metadata 中取得 chunk_id

                self.base_db.execute_query(
                    """
                    INSERT INTO retrieved_docs (rag_history_id, doc_index, chunk_id, content, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (rag_history_id, idx, chunk_id, content, metadata)
                )

            logging.info(
                f"✅ 成功儲存 RAG 查詢與 {len(retrieved_data)} 筆檢索內容至資料庫 (rag_history_id={rag_history_id})"
            )

        except Exception as e:
            logging.error("❌ 儲存 RAG 資料時發生錯誤:\n" + traceback.format_exc())

    # ---------------------------------------
    # 查詢：依 conversation_id 查詢 RAG 記錄
    # ---------------------------------------
    def get_rag_records_by_conversation_id(self, conversation_id: str) -> list:
        """
        根據 conversation_id 查詢 rag_history 主表與 retrieved_docs 子表內容。
        回傳格式為 List[Dict]，每筆包含 query、response、retrieved_docs 等欄位。
        """
        try:
            rag_rows = self.base_db.fetch_query(
                """
                SELECT id, query, rewritten_query, response, timestamp
                FROM rag_history
                WHERE conversation_id = ?
                """,
                (conversation_id,)
            )

            results = []
            for row in rag_rows:
                rag_id, query, rewritten_query, response, timestamp = row

                # 查詢對應的文件內容
                doc_rows = self.base_db.fetch_query(
                    "SELECT doc_index, content, metadata FROM retrieved_docs WHERE rag_history_id = ? ORDER BY doc_index",
                    (rag_id,)
                )

                retrieved_docs = [
                    {'doc_index': d[0], 'content': d[1], 'metadata': json.loads(d[2])}
                    for d in doc_rows
                ]

                results.append({
                    'query': query,
                    'rewritten_query': rewritten_query,
                    'response': response,
                    'timestamp': timestamp,
                    'retrieved_docs': retrieved_docs
                })

            return results

        except Exception as e:
            logging.error(f"❌ 查詢 RAG 記錄失敗: {e}")
            return []
