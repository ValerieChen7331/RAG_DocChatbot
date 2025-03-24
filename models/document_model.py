import tempfile
from langchain_chroma import Chroma
from apis.file_paths import FilePaths
from apis.embedding_api import EmbeddingAPI
import os
from langchain.document_loaders import PyPDFDirectoryLoader, TextLoader
import logging
from pathlib import Path

os.environ["CHROMA_TELEMETRY"] = "False"

# 設定日誌記錄的級別為 INFO
logging.basicConfig(level=logging.INFO)

class DocumentModel:
    def __init__(self, chat_session_data):
        # 初始化 hat_session_data
        self.chat_session_data = chat_session_data
        # 初始化文件路徑
        self.file_paths = FilePaths()
        username = self.chat_session_data.get("username")
        conversation_id = self.chat_session_data.get("conversation_id")
        self.tmp_dir = self.file_paths.get_tmp_dir(username, conversation_id)
        self.vector_store_dir = self.file_paths.get_local_vector_store_dir(username, conversation_id)

    # def create_temporary_files(self, source_docs):
    #     """建立臨時文件並返回檔案名稱對應關係。"""
    #     self.tmp_dir.mkdir(parents=True, exist_ok=True)
    #     doc_names = {}
    #
    #     for source_doc in source_docs:
    #         with tempfile.NamedTemporaryFile(delete=False, dir=self.tmp_dir.as_posix(), suffix='.pdf') as tmp_file:
    #             tmp_file.write(source_doc['content'])  # 寫入文件內容
    #             file_name = Path(tmp_file.name).name
    #             doc_names[file_name] = source_doc['name']
    #             logging.info(f"Created temporary file: {file_name}")
    #
    #     return doc_names
    #
    # def load_documents(self):
    #     # 加載 PDF 文件
    #     # 使用 PyPDFDirectoryLoader 從指定的目錄中加載所有 PDF 文件
    #     loader = PyPDFDirectoryLoader(self.tmp_dir.as_posix(), glob='**/*.pdf')
    #     # 加載 PDF 文件，並將其儲存在 documents 變數中
    #     documents = loader.load()
    #     # 如果沒有加載到任何文件，拋出異常提示
    #     if not documents:
    #         raise ValueError("No documents loaded. Please check the PDF files.")
    #     # 紀錄已加載的文件數量到日誌中
    #     logging.info(f"Loaded {len(documents)} documents")
    #     # 返回已加載的文件
    #     return documents

    def create_temporary_files(self, source_docs):
        """根據上傳檔案類型（pdf / md）建立臨時文件並返回檔案名稱對應關係。"""
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        doc_names = {}

        for source_doc in source_docs:
            file_ext = source_doc.get('type', 'pdf')  # 預設為 pdf，如果有 type 字段就用 type
            suffix = f'.{file_ext}'
            with tempfile.NamedTemporaryFile(delete=False, dir=self.tmp_dir.as_posix(), suffix=suffix) as tmp_file:
                tmp_file.write(source_doc['content'])  # 寫入文件內容
                file_name = Path(tmp_file.name).name
                doc_names[file_name] = source_doc['name']
                logging.info(f"Created temporary file: {file_name}")

        return doc_names

    def load_documents(self):
        """從臨時目錄中加載 PDF 和 MD 文件"""
        documents = []

        # 載入 PDF 文件
        pdf_loader = PyPDFDirectoryLoader(self.tmp_dir.as_posix(), glob='**/*.pdf')
        pdf_documents = pdf_loader.load()
        documents.extend(pdf_documents)

        # 載入 MD 文件
        md_files = list(self.tmp_dir.rglob('*.md'))
        for md_file in md_files:
            loader = TextLoader(md_file.as_posix(), encoding='utf-8')
            md_documents = loader.load()
            documents.extend(md_documents)

        if not documents:
            raise ValueError("No documents loaded. Please check the PDF or MD files.")

        logging.info(f"Loaded {len(documents)} documents (PDF + MD)")

        return documents

    def delete_temporary_files(self):
        # 刪除臨時文件
        for file in self.tmp_dir.iterdir():
            try:
                logging.info(f"Deleting temporary file: {file}")
                file.unlink()
            except Exception as e:
                logging.error(f"Error deleting file {file}: {e}")

    def split_documents_into_chunks_1(self, documents):
        """
        將文件拆分為基於標題的結構和字元級內容小塊。
        """
        # 引入所需模組
        from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
        from langchain.docstore.document import Document
        import logging

        document_chunks = []  # 用於存儲拆分後的小塊
        combined_markdown_content = ""  # 用於合併所有文檔內容

        # 處理輸入文檔，將其合併為一個 Markdown 字串
        for doc in documents:
            if hasattr(doc, "page_content") and doc.page_content:
                combined_markdown_content += str(doc.page_content)  # 合併有效內容
            else:
                logging.warning("跳過一個無內容的文檔。")  # 記錄無內容文檔的警告
                continue

        # 使用 MarkdownHeaderTextSplitter 進行基於標題的拆分
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(combined_markdown_content)  # 按標題拆分文檔

        # 定義字元級拆分器
        chunk_size = 800  # 每個塊的大小
        chunk_overlap = 300  # 塊的重疊大小
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # 處理每個標題分組
        for header_group in md_header_splits:
            # 將每個標題組中的內容拆分為小塊
            splits = text_splitter.split_text(header_group.page_content)
            for split in splits:
                document_chunks.append({
                    "page_content": split,  # 小塊內容
                    "metadata": header_group.metadata  # 對應的標題元數據
                })

        # 將拆分後的塊轉換為 LangChain Document 格式
        transformed_documents = []
        for i, chunk in enumerate(document_chunks):
            transformed_documents.append(
                Document(
                    metadata={**chunk["metadata"], 'page': i},  # 新增頁碼到元數據
                    page_content=chunk["page_content"] + "\n" + str(chunk["metadata"])  # 添加元數據信息
                )
            )

        # 記錄拆分完成的日誌
        logging.info(f"成功將文檔拆分為 {len(transformed_documents)} 個塊。")

        return transformed_documents

    def split_documents_into_chunks_3(self, documents):
        """
        將文件拆分為基於標題的結構小塊，並保留標題元數據。
        """
        from langchain.text_splitter import MarkdownHeaderTextSplitter
        from langchain.docstore.document import Document
        import logging

        # 初始化變數
        document_chunks = []
        markdown_document = ""

        # 合併所有文檔內容為單一 Markdown 字串
        for doc in documents:
            if getattr(doc, "page_content", None):  # 確認文檔有內容
                markdown_document += str(doc.page_content)
            else:
                logging.warning(f"Skipping a document with no content: {doc}")

        # 如果文檔內容為空，直接返回空列表
        if not markdown_document.strip():
            logging.warning("No valid content found in the provided documents.")
            return []

        # 配置標題層級拆分器
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(markdown_document)

        # 將拆分結果轉換為符合格式的文檔物件
        for i, header_group in enumerate(md_header_splits):
            document_chunks.append(
                Document(
                    metadata={**header_group.metadata, 'page': i},  # 包含元數據與頁碼
                    page_content=str(header_group.metadata) + "\n" + header_group.page_content
                )
            )
            print('1. header_group.page_content: ', type(header_group.page_content))
            print('2-1. header_group.metadata: ', type(header_group.metadata))
            print('2-2. header_group.metadata: ', header_group.metadata)

        # 記錄成功處理的日誌
        logging.info(f"Successfully split documents into {len(document_chunks)} chunks.")
        return document_chunks

    def split_documents_into_chunks_4(self, documents):
        """
        將文件拆分為基於標題的結構小塊，並保留標題元數據。
        """
        from langchain.text_splitter import MarkdownHeaderTextSplitter
        from langchain.docstore.document import Document
        import logging

        # 初始化變數
        document_chunks = []
        markdown_document = ""

        # 合併所有文檔內容為單一 Markdown 字串
        for doc in documents:
            if getattr(doc, "page_content", None):  # 確認文檔有內容
                markdown_document += str(doc.page_content)
            else:
                logging.warning(f"Skipping a document with no content: {doc}")

        # 如果文檔內容為空，直接返回空列表
        if not markdown_document.strip():
            logging.warning("No valid content found in the provided documents.")
            return []

        # 配置標題層級拆分器
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(markdown_document)

        # 將拆分結果轉換為符合格式的文檔物件
        for i, header_group in enumerate(md_header_splits):
            document_chunks.append(
                Document(
                    metadata={**header_group.metadata, 'page': i},  # 包含元數據與頁碼
                    page_content=str(header_group.metadata) + "\n" + header_group.page_content
                )
            )
            print('1. header_group.page_content: ', type(header_group.page_content))
            print('2-1. header_group.metadata: ', type(header_group.metadata))
            print('2-2. header_group.metadata: ', header_group.metadata)

        # 記錄成功處理的日誌
        logging.info(f"Successfully split documents into {len(document_chunks)} chunks.")
        return document_chunks

    def embeddings_on_local_vectordb(self, document_chunks):
        # 將文檔塊嵌入本地向量數據庫，並返回檢索器設定
        mode = self.chat_session_data.get("mode")
        embedding = self.chat_session_data.get("embedding")
        embedding_function = EmbeddingAPI.get_embedding_function(mode, embedding)
        if not document_chunks:
            raise ValueError("No document chunks to embed. Please check the text splitting process.")

        Chroma.from_documents(
            documents=document_chunks,
            embedding=embedding_function,
            persist_directory=self.vector_store_dir.as_posix()
        )
        logging.info(f"Persisted vector DB at {self.vector_store_dir}")
