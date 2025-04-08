import logging
from models.document_model import DocumentModel
from models.database_userRecords import UserRecordsDB
from models.database_devOps import DeveloperDB
from models.llm_model import LLMModel

logging.basicConfig(level=logging.INFO)
class DocumentService:
    def __init__(self, chat_session_data):
        self.chat_session_data = chat_session_data

    def process_uploaded_documents(self, source_docs):
        try:
            username = self.chat_session_data.get('username')
            doc_model = DocumentModel(self.chat_session_data)
            user_db = UserRecordsDB(username)
            devops_db = DeveloperDB()
            llm_model = LLMModel(self.chat_session_data)

            for source_doc in source_docs:
                # 1. 建立臨時文件
                tmp_name, org_name = doc_model.create_temporary_files(source_doc)
                self.chat_session_data['doc_names'] += org_name
                print("doc_names: ", self.chat_session_data['doc_names'])

                # 2. 加載文檔
                document = doc_model.load_documents()
                # 刪除臨時文件
                doc_model.delete_temporary_files()

            # for document, doc_name in zip(documents, doc_names):
                # 3. 生成摘要
                summaries = llm_model.doc_summary(document)
                user_db.save_to_file_names(self.chat_session_data, summaries, tmp_name, org_name)
                devops_db.save_to_file_names(self.chat_session_data, summaries, tmp_name, org_name)

                # 4. 文檔切片
                chunks = doc_model.chunk_document_with_md(document, org_name)

                # 5. 向量嵌入
                doc_model.embed_into_vectordb(chunks)

            # 6. 儲存記錄至兩個資料庫
            user_db.save_to_pdf_uploads(self.chat_session_data)
            devops_db.save_to_pdf_uploads(self.chat_session_data)
            return self.chat_session_data

        except Exception as e:
            logging.error(f"❌ 處理文檔時發生錯誤：{e}", exc_info=True)