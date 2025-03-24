from models.database_userRecords import UserRecordsDB
import uuid


class UIController:
    def __init__(self, chat_session_data):
        # 初始化 UserRecordsDB
        self.chat_session_data = chat_session_data
        self.username = chat_session_data.get('username')

    def get_title(self, index):
        """根據窗口索引返回標題"""
        # 從資料庫加載數據
        userRecords_db = UserRecordsDB(self.username)
        df_database = userRecords_db.load_database(
            'chat_history',
            ['active_window_index', 'title'])
        df_window = df_database[df_database['active_window_index'] == index]

        # 設置標題，如果無數據則為新對話
        if not df_window.empty:
            window_title = df_window['title'].iloc[0]
        else:
            window_title = "(新對話)"
        return window_title

    def new_chat(self):
        """創建新聊天窗口，更新 chat_session_data 狀態"""
        if self.chat_session_data.get('empty_window_exists'):
            self.chat_session_data['active_window_index'] = self.chat_session_data.get('num_chat_windows') - 1
        else:
            self.chat_session_data['active_window_index'] = self.chat_session_data.get('num_chat_windows')
            self.chat_session_data['num_chat_windows'] += 1

        print("empty_window_exists", self.chat_session_data.get('empty_window_exists'))
        print("num_chat_windows", self.chat_session_data.get('num_chat_windows'))

        self.chat_session_data['conversation_id'] = str(uuid.uuid4())
        self.chat_session_data = self.reset_session_state_to_defaults()
        self.chat_session_data['empty_window_exists'] = True
        return self.chat_session_data

    def reset_session_state_to_defaults(self):
        """重置 session state 參數至預設值。"""
        reset_session_state = {
            'mode': '內部LLM',
            'llm_option': 'Gemma2',
            'model': None,
            'api_base': None,
            'api_key': None,
            'embedding': 'bge-m3',
            'db_name': None,
            'db_source': None,
            'title': '',
            'chat_history': []
        }
        for key, value in reset_session_state.items():
            self.chat_session_data[key] = value
        return self.chat_session_data

    def delete_chat_history_and_update_indexes(self, delete_index):
        """刪除指定聊天窗口並更新索引"""
        # 刪除指定聊天窗口索引的聊天歷史記錄
        userRecords_db = UserRecordsDB(self.username)
        userRecords_db.delete_chat_by_index(delete_index)
        # 更新剩餘聊天窗口的索引
        userRecords_db.update_chat_indexes(delete_index)
        # 更新聊天窗口的數量
        self.chat_session_data['num_chat_windows'] -= 1
        return self.chat_session_data
