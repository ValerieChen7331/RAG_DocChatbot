# llm_services.py
from models.llm_model import LLMModel
from models.llm_rag import RAGModel
from models.database_userRecords import UserRecordsDB
from models.database_devOps import DeveloperDB

class LLMService:
    def __init__(self, chat_session_data):
        """初始化 LLMModel 和 DatabaseModel"""
        self.chat_session_data = chat_session_data

    def query(self, query):
        """
        根據查詢和選擇的助理類型執行適當的 LLM 查詢
        """
        # 從 session_state 取得相關設定
        selected_agent = self.chat_session_data.get('agent')
        username = self.chat_session_data.get('username')
        userRecords_db = UserRecordsDB(username)
        conversation_id = self.chat_session_data.get('conversation_id')
        developer_db = DeveloperDB()

        # 如果聊天記錄為空，設定新窗口的標題
        if not self.chat_session_data.get('chat_history'):
            llm_model = LLMModel(self.chat_session_data)
            llm_model.set_window_title(query)

        if selected_agent == '個人KM':
            # 使用檢索增強生成模式進行查詢
            llm_rag = RAGModel(self.chat_session_data)
            doc_summary = userRecords_db.get_doc_summaries(self.chat_session_data)
            response, retrieved_documents, rewritten_query = llm_rag.query_llm_rag(query, doc_summary)
            # 將查詢和回應結果保存到資料庫 userRecords_db
            userRecords_db.save_retrieved_data_to_db(query, rewritten_query, retrieved_documents, response, self.chat_session_data)
            developer_db.save_retrieved_data_to_db(query, rewritten_query, retrieved_documents, response, self.chat_session_data)

        else:
            # 直接使用 LLM 進行查詢
            llm_model = LLMModel(self.chat_session_data)
            response = llm_model.query_llm_direct(query)

        # 更新 chat_session_data 中的聊天記錄
        self.chat_session_data['chat_history'].append({"user_query": query, "ai_response": response})
        self.chat_session_data['empty_window_exists'] = False

        # 將查詢和回應結果保存到資料庫 userRecords_db
        userRecords_db.save_to_database(query, response, self.chat_session_data)
        # 將查詢和回應結果保存到資料庫 DeveloperDB()
        developer_db.save_chat_history(query, response, self.chat_session_data)

        return response, self.chat_session_data
