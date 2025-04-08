import streamlit as st
from services.document_services import DocumentService
from models.database_userRecords import UserRecordsDB

class MainContent:
    def __init__(self, chat_session_data):
        """初始化主內容物件"""
        self.chat_session_data = chat_session_data

    def display(self):
        """顯示主內容"""
        # 配置頁面標題
        self.configure_page()
        # 顯示文件上傳欄位
        if self.chat_session_data.get('agent') == '個人KM':
            self.display_input_fields()
        # 顯示聊天記錄
        self.display_active_chat_history()

    def configure_page(self):
        """配置主頁面標題"""
        st.title("📂 文件檢索聊天機器人")
        st.write(f'*Welcome {st.session_state.get("name", "Guest")}*')  # 顯示歡迎訊息

        # 撈取已上傳文件記錄 'doc_names'
        file_names = UserRecordsDB(st.session_state.get("username")).get_doc_names(self.chat_session_data)
        if file_names and self.chat_session_data.get('agent') == '個人KM':
            st.write(f'**📚 {file_names}**')  # 顯示文件記錄

    def display_input_fields(self):
        """顯示文件上傳欄位，僅當選擇 '個人KM' 時顯示"""
        if self.chat_session_data.get('agent') == '個人KM':
            # 顯示文件上傳欄位，允許上傳多個 PDF 和 MD 文件
            uploaded_files = st.file_uploader(
                label="上傳文檔 (PDF 或 MD)",
                type=["pdf", "md"],
                accept_multiple_files=True
            )

            # 準備文件列表，包含檔名、內容及副檔名類型
            source_docs = []
            if uploaded_files:
                for file in uploaded_files:
                    file_type = file.name.split('.')[-1].lower()
                    content = file.read()
                    source_docs.append({
                        'name': file.name,
                        'type': file_type,
                        'content': content
                    })

            # 顯示提交按鈕，點擊時觸發 process_uploaded_documents 方法
            if st.button("提交文件", key="submit", help="提交文件"):
                try:
                    self.chat_session_data = DocumentService(self.chat_session_data).process_uploaded_documents(
                        source_docs)
                    st.success("文件已成功提交並處理")
                except Exception as e:
                    st.error(f"處理文檔時發生錯誤：{e}")

    def display_active_chat_history(self):
        """顯示聊天記錄"""
        chat_records = self.chat_session_data.get('chat_history', [])
        if chat_records:
            # 迭代顯示每一條聊天記錄
            for result in chat_records:
                with st.chat_message("user"):
                    st.markdown(result['user_query'])
                with st.chat_message("ai"):
                    st.markdown(result['ai_response'])

