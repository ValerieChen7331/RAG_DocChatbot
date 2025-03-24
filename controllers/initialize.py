import uuid
import pandas as pd
from models.database_userRecords import UserRecordsDB
from apis.file_paths import FilePaths


class SessionInitializer:
    def __init__(self, username, base_dir=None):
        """
        初始化 SessionInitializer，設置使用者名稱和文件路徑。

        參數:
            username (str): 使用者名稱。
            base_dir (str, optional): 基本目錄路徑，默認為 None。
        """
        self.username = username
        self.file_paths = FilePaths(base_dir)

    def initialize_session_state(self):
        """初始化 Session 狀態，並儲存到字典 chat_session_data 中。"""
        # 從 user_records_db 載入資料，只取 active_window_index 欄位
        userRecords_db = UserRecordsDB(self.username)
        database = userRecords_db.load_database(
            'chat_history',
            ['active_window_index'])

        # 設置聊天窗口的數量及活躍窗口索引
        if not database.empty:
            num_chat_windows = len(set(database['active_window_index']))
        else:
            num_chat_windows = 0

        active_window_index = num_chat_windows
        num_chat_windows += 1  # 為新聊天窗口增加計數

        # 初始化 session 狀態參數，並儲存到 chat_session_data 字典中
        chat_session_data = {
            'conversation_id': str(uuid.uuid4()),  # 新的對話 ID
            'num_chat_windows': num_chat_windows,
            'active_window_index': active_window_index,
            'agent': '一般助理',
            'mode': '內部LLM',
            'llm_option': 'Gemma2',
            'model': 'gemma2:latest',
            'api_base': '',
            'api_key': '',
            'embedding': 'bge-m3',
            'doc_names': '',
            'chat_history': [],
            'title': '',

            'upload_time': None,
            'username': self.username,  # 設置使用者名稱

            'empty_window_exists': True  # 新窗口存在
        }

        return chat_session_data
