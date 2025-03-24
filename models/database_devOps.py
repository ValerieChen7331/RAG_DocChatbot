from datetime import datetime
from models.database_base import BaseDB
from apis.file_paths import FilePaths
import logging

logging.basicConfig(level=logging.INFO)

class DevOpsDB:
    def __init__(self):
        """初始化 DevOpsDB 類別。"""
        # 設定資料庫路徑
        file_paths = FilePaths()
        self.db_path = file_paths.get_developer_dir().joinpath('DevOpsDB.db')
        self.base_db = BaseDB(self.db_path)

        # 初始化資料庫表格
        self.base_db.ensure_db_path_exists()
        self._init_db()

    def _init_db(self):
        """初始化資料庫，創建必要的表格。"""
        if not self.db_path.exists():
            # 定義聊天記錄表格的 SQL 語句
            chat_history_query = '''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY,
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    username TEXT,
                    agent TEXT,
                    mode TEXT,
                    llm_option TEXT,
                    model TEXT,
                    db_source TEXT,
                    db_name TEXT,
                    conversation_id TEXT,
                    active_window_index INTEGER,
                    num_chat_windows INTEGER,
                    title TEXT,
                    user_query TEXT,
                    ai_response TEXT
                )
            '''
            # 執行創建聊天記錄表格的 SQL 語句
            self.base_db.execute_query(chat_history_query)

            # 定義 PDF 上傳記錄表格的 SQL 語句
            pdf_uploads_query = '''
                CREATE TABLE IF NOT EXISTS pdf_uploads (
                    id INTEGER PRIMARY KEY,
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    username TEXT,
                    conversation_id TEXT,
                    agent TEXT,
                    embedding TEXT
                )
            '''
            # 執行創建 PDF 上傳記錄表格的 SQL 語句
            self.base_db.execute_query(pdf_uploads_query)

            # 定義文件名稱記錄表格的 SQL 語句
            file_names_query = '''
                CREATE TABLE IF NOT EXISTS file_names (
                    id INTEGER PRIMARY KEY,
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    username TEXT,
                    conversation_id TEXT,
                    tmp_name TEXT,
                    org_name TEXT,
                    doc_summary TEXT
                )
            '''
            # 執行創建文件名稱記錄表格的 SQL 語句
            self.base_db.execute_query(file_names_query)
            logging.info("DevOpsDB 資料庫初始化成功。")

    def save_to_database(self, query: str, response: str, chat_session_data):
        """
        將查詢結果保存到資料庫中。

        Args:
            query (str): 使用者的查詢。
            response (str): AI 回應的結果。
            chat_session_data (dict): 聊天會話的數據，包括歷史記錄和其他相關資訊。
        """
        # 取得當前時間
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 初始化資料字典，從 chat_session_data 中獲取數據
        data = {key: chat_session_data.get(key, default) for key, default in {
            'upload_time': current_time,
            'username': '',
            'agent': '',
            'mode': '',
            'llm_option': '',
            'model': '',
            'db_source': '',
            'db_name': '',
            'conversation_id': '',
            'active_window_index': 0,
            'num_chat_windows': 0,
            'title': '',
            'user_query': query,
            'ai_response': response
        }.items()}

        try:
            # 插入資料到 chat_history 表格
            self.base_db.execute_query(
                """
                INSERT INTO chat_history 
                (upload_time, username, agent, mode, llm_option, model, db_source, db_name,
                 conversation_id, active_window_index, num_chat_windows, title,
                 user_query, ai_response) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(data.values())
            )
            logging.info("查詢結果已成功保存到資料庫 DevOpsDB (chat_history)")
        except Exception as e:
            logging.error(f"保存到 DevOpsDB (chat_history) 資料庫時發生錯誤: {e}")

    def save_to_pdf_uploads(self, chat_session_data):
        """將 PDF 上傳記錄保存到資料庫中。"""
        # 取得當前時間
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 初始化資料字典，從 chat_session_data 中獲取數據
        data = {key: chat_session_data.get(key, default) for key, default in {
            'upload_time': current_time,
            'username': '',
            'conversation_id': '',
            'agent': '',
            'embedding': ''
        }.items()}

        try:
            # 插入資料到 pdf_uploads 表格
            self.base_db.execute_query(
                """
                INSERT INTO pdf_uploads 
                (upload_time, username, conversation_id, agent, embedding) 
                VALUES (?, ?, ?, ?, ?)
                """,
                tuple(data.values())
            )
            logging.info("查詢結果已成功保存到資料庫 DevOpsDB (pdf_uploads)。")
        except Exception as e:
            logging.error(f"保存到 DevOpsDB (pdf_uploads) 資料庫時發生錯誤: {e}")

    # def save_to_file_names(self, chat_session_data, doc_summary):
    #     """將文件名稱記錄保存到資料庫中。"""
    #     # 取得當前時間
    #     upload_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #     # 從 chat_session_data 獲取使用者名稱和對話 ID
    #     username = chat_session_data.get('username', '')
    #     conversation_id = chat_session_data.get('conversation_id', '')
    #     # 獲取文件名稱字典
    #     doc_names = chat_session_data.get('doc_names', {})
    #
    #     for tmp_name, org_name in doc_names.items():
    #         try:
    #             # 插入資料到 file_names 表格
    #             self.base_db.execute_query(
    #                 """
    #                 INSERT INTO file_names
    #                 (upload_time, username, conversation_id, tmp_name, org_name, doc_summary)
    #                 VALUES (?, ?, ?, ?, ?, ?)
    #                 """,
    #                 (upload_time, username, conversation_id, tmp_name, org_name, doc_summary)
    #             )
    #             logging.info(f"file_names 已成功保存到 DevOpsDB: tmp_name={tmp_name}, org_names={org_name}")
    #         except Exception as e:
    #             logging.error(f"file_names 保存到 DevOpsDB 時發生錯誤: {e}")




    def save_to_file_names(self, chat_session_data: dict, doc_summary: list):
        """
        將文件上傳記錄（檔名、使用者、時間、摘要等）
        儲存到資料庫中的 file_names 表格中。

        輸入：
            chat_session_data: dict
                包含以下資訊：
                - username: 使用者名稱
                - conversation_id: 此次對話 ID
                - doc_names: dict，格式 {tmp_name: org_name}
            doc_summary: list
                包含摘要字串的清單，每個摘要與 doc_names 順序相對應

        輸出：
            無（寫入資料庫）
        """

        # 取得當前上傳時間
        upload_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        username = chat_session_data.get('username', '')
        conversation_id = chat_session_data.get('conversation_id', '')
        doc_names = chat_session_data.get('doc_names', {})

        # 使用 zip() 將 doc_names 與對應摘要配對
        for (tmp_name, org_name), summary_for_file in zip(doc_names.items(), doc_summary):
            try:
                # 寫入資料庫
                self.base_db.execute_query(
                    """
                    INSERT INTO file_names 
                    (upload_time, username, conversation_id, tmp_name, org_name, doc_summary) 
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (upload_time, username, conversation_id, tmp_name, org_name, summary_for_file)
                )
                logging.info(f"file_names 已成功保存到 DevOpsDB: tmp_name={tmp_name}, org_name={org_name}")

            except Exception as e:
                logging.error(f"file_names 保存到 DevOpsDB 時發生錯誤: {e}")