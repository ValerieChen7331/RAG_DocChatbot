import pandas as pd
import logging
from models.database_base import BaseDB
from apis.file_paths import FilePaths

logging.basicConfig(level=logging.INFO)


class UserRecordsDB:
    def __init__(self, username):
        """ 初始化 UserRecordsDB 類別。 """
        # 設定資料庫路徑
        self.file_paths = FilePaths()
        self.db_path = self.file_paths.get_user_records_dir(username).joinpath(f"{username}.db")
        # 創建 BaseDB 實例來處理資料庫操作
        self.base_db = BaseDB(self.db_path)
        # 初始化資料庫表格
        self.base_db.ensure_db_path_exists()
        self._init_db()

    def _init_db(self):
        """檢查資料庫文件是否存在，不存在則初始化資料庫。"""
        if not self.db_path.exists():
            # 定義聊天記錄表格的 SQL 語句
            chat_history_query = '''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY,
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
                    conversation_id TEXT,
                    tmp_name TEXT,             
                    org_name TEXT,
                    doc_summary TEXT
                )
            '''
            # 執行創建文件名稱記錄表格的 SQL 語句
            self.base_db.execute_query(file_names_query)
            logging.info("UserRecordsDB 資料庫初始化成功。")

    def load_database(self, database, columns=None) -> pd.DataFrame:
        """
        載入聊天記錄，並以 DataFrame 格式返回。
        """
        # 預設所有欄位
        all_columns = [
            'id', 'agent', 'mode', 'llm_option', 'model',
            'db_source', 'db_name', 'conversation_id', 'active_window_index',
            'num_chat_windows', 'title', 'user_query', 'ai_response'
        ]

        # 如果未提供特定欄位，則使用預設的所有欄位
        selected_columns = columns if columns else all_columns
        # SQL 查詢語句
        query = f"SELECT {', '.join(selected_columns)} FROM {database}"

        # 預設的空 DataFrame，用於返回無結果的情況
        empty_df = pd.DataFrame(columns=selected_columns)

        try:
            # 執行查詢，獲取數據
            data = self.base_db.fetch_query(query)
            if not data:
                return empty_df

            # 返回包含數據的 DataFrame
            return pd.DataFrame(data, columns=selected_columns)
        except Exception as e:
            # 捕捉錯誤，顯示錯誤訊息並返回空的 DataFrame
            print(f"load_database 發生錯誤: {e}")
            return empty_df

    def update_chat_indexes(self, delete_index):
        """更新聊天記錄索引。"""
        chat_histories = self.base_db.fetch_query(
            "SELECT id, active_window_index FROM chat_history ORDER BY active_window_index")

        for id, active_window_index in chat_histories:
            if active_window_index > delete_index:
                new_index = active_window_index - 1
                self.base_db.execute_query(
                    "UPDATE chat_history SET active_window_index = ? WHERE id = ?",
                    (new_index, id))

    # ===== Delete =====
    def delete_chat_by_index(self, delete_index):
        """刪除指定的聊天記錄。"""
        self.base_db.execute_query(
            "DELETE FROM chat_history WHERE active_window_index = ?",
            (delete_index,))

    # ===== Select=====
    def get_active_window_setup(self, index, chat_session_data):
        """
        從資料庫中獲取並加載當前的聊天記錄。
        """
        try:
            # 定義設置和歷史記錄的欄位名稱
            setup_columns = ['conversation_id', 'agent', 'mode', 'llm_option', 'model', 'db_source', 'db_name', 'title']
            history_columns = ['user_query', 'ai_response']

            # 合併所有欄位名稱
            all_columns = setup_columns + history_columns

            # SQL 查詢，用於獲取指定 active_window_index 的聊天記錄
            query = """
                SELECT conversation_id, agent, mode, llm_option, model, db_source, db_name, title,
                       user_query, ai_response
                FROM chat_history 
                WHERE active_window_index = ? 
                ORDER BY id
            """

            # 執行查詢並獲取結果
            active_window_setup = self.base_db.fetch_query(query, (index,))

            if active_window_setup:
                # 將查詢結果轉換為 DataFrame
                df_check = pd.DataFrame(active_window_setup, columns=all_columns)

                # 更新 chat_session_data 的設置列
                for column in setup_columns:
                    chat_session_data[column] = df_check[column].iloc[-1]

                # 更新 chat_history 並轉換為字典格式
                chat_history_df = df_check[history_columns]
                chat_session_data['chat_history'] = chat_history_df.to_dict(orient='records')
            else:
                # 如果無結果，重置 chat_session_data 為預設值
                # chat_session_data = self.reset_session_state_to_defaults()
                chat_session_data = {}

            return chat_session_data

        except Exception as e:
            print(f"get_active_window_setup 發生錯誤: {e}")

    def get_active_window_file_names(self, chat_session_data):
        """
        從資料庫中獲取 file_names
        """
        try:
            conversation_id = chat_session_data.get('conversation_id')
            # SQL 查詢
            query = """
                        SELECT conversation_id, org_name
                        FROM file_names 
                        WHERE conversation_id = ?
                    """

            # 執行查詢
            active_window_file_names = self.base_db.fetch_query(query, (conversation_id,))
            if active_window_file_names:
                # 將查詢結果中 org_name 欄位取出並組成字串
                doc_names = ", ".join(row[1] for row in active_window_file_names)
            else:
                doc_names = ""
            return doc_names

        except Exception as e:
            print(f"取得 get_active_window_file_names 時發生錯誤: {e}")
            return ""

    def get_active_window_doc_summary(self, chat_session_data):
        """
        從資料庫中取得指定 conversation_id 對話的所有檔案名稱及摘要
        並格式化成可讀字串
        """
        try:
            conversation_id = chat_session_data.get('conversation_id')
            query = """
                        SELECT org_name, doc_summary
                        FROM file_names 
                        WHERE conversation_id = ?
                    """
            results = self.base_db.fetch_query(query, (conversation_id,))
            if results:
                doc_summary = "\n\n".join(
                    f"【{row[0]}】摘要：{row[1]}" for row in results
                )
            else:
                doc_summary = ""
            return doc_summary

        except Exception as e:
            print(f"取得 get_active_window_doc_summary 時發生錯誤: {e}")
            return ""

    # ===== Insert =====
    def save_to_database(self, query: str, response: str, chat_session_data):
        """
        將查詢結果保存到資料庫中。
        """
        # 初始化資料字典，從 chat_session_data 中獲取數據
        data = {key: chat_session_data.get(key, default) for key, default in {
            'agent': None,
            'mode': None,
            'llm_option': None,
            'model': None,
            'db_source': None,
            'db_name': None,
            'conversation_id': None,
            'active_window_index': 0,
            'num_chat_windows': 0,
            'title': None,
            'user_query': query,
            'ai_response': response
        }.items()}

        try:
            # 插入資料到 chat_history 表格
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
            logging.info("查詢結果已成功保存到資料庫 UserDB (chat_history)")

        except Exception as e:
            logging.error(f"保存到 UserDB (chat_history) 資料庫時發生錯誤: {e}. Data: {data}")

    def save_to_pdf_uploads(self, chat_session_data):
        """將查詢結果保存到 pdf_uploads 表格中。"""
        # 初始化資料字典，從 chat_session_data 中獲取數據
        data = {key: chat_session_data.get(key, default) for key, default in {
            'conversation_id': None,
            'agent': None,
            'embedding': None
        }.items()}

        try:
            # 插入資料到 pdf_uploads 表格
            self.base_db.execute_query(
                """
                INSERT INTO pdf_uploads 
                (conversation_id, agent, embedding) 
                VALUES (?, ?, ?)
                """,
                tuple(data.values())
            )
            logging.info("查詢結果已成功保存到資料庫 UserDB (pdf_uploads)")
        except Exception as e:
            logging.error(f"保存到 UserDB (pdf_uploads) 資料庫時發生錯誤: {e}")

    def save_to_file_names(self, chat_session_data: dict, doc_summary: list):
        """
        將當前聊天會話中處理的檔案名稱及對應摘要結果
        儲存到資料庫中的 file_names 表格中。
        """

        conversation_id = chat_session_data.get('conversation_id', None)
        doc_names = chat_session_data.get('doc_names', {})

        # 使用 zip() 同步配對 doc_names.items() 與 doc_summary
        for (tmp_name, org_name), summary_for_file in zip(doc_names.items(), doc_summary):
            try:
                # 將 conversation_id、暫存檔名、原始檔名與對應摘要儲存到資料表
                self.base_db.execute_query(
                    """
                    INSERT INTO file_names 
                    (conversation_id, tmp_name, org_name, doc_summary) 
                    VALUES (?, ?, ?, ?)
                    """,
                    (conversation_id, tmp_name, org_name, summary_for_file)
                )
                logging.info(f"file_names 已成功儲存到 UserDB: tmp_name={tmp_name}, org_name={org_name}")

            except Exception as e:
                logging.error(f"file_names 儲存到 UserDB 時發生錯誤: {e}")


