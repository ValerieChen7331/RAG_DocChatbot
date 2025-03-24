from models.document_model import DocumentModel
from models.database_userRecords import UserRecordsDB
from models.database_devOps import DevOpsDB
from models.llm_model import LLMModel
import logging

logging.basicConfig(level=logging.INFO)
class DocumentService:
    def __init__(self, chat_session_data):
        # 初始化 DocumentModel
        self.chat_session_data = chat_session_data

    def process_uploaded_documents(self, source_docs):
        doc_model = DocumentModel(self.chat_session_data)
        try:
            # 建立臨時文件
            doc_names = doc_model.create_temporary_files(source_docs)
            self.chat_session_data['doc_names'] = doc_names

            # 加載文檔
            documents = doc_model.load_documents()

            # 刪除臨時文件
            # doc_model.delete_temporary_files()

            # 生成摘要
            doc_summarys = LLMModel(self.chat_session_data).doc_summary(documents)

            # 將文檔拆分成塊
            document_chunks = doc_model.split_documents_into_chunks_1(documents)

            # 在本地向量數據庫中嵌入文檔塊
            doc_model.embeddings_on_local_vectordb(document_chunks)

            # 存入 userRecords_db
            username = self.chat_session_data.get('username')
            userRecords_db = UserRecordsDB(username)
            userRecords_db.save_to_file_names(self.chat_session_data, doc_summarys)
            userRecords_db.save_to_pdf_uploads(self.chat_session_data)

            # 存入 devOps_db
            devOps_db = DevOpsDB()
            devOps_db.save_to_file_names(self.chat_session_data, doc_summarys)
            devOps_db.save_to_pdf_uploads(self.chat_session_data)

            return self.chat_session_data

        except Exception as e:
            # 處理文檔時發生錯誤，顯示錯誤訊息
            logging.error(f"處理文檔時發生錯誤 process_uploaded_documents：{e}")