import logging
import tempfile
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from apis.file_paths import FilePaths
from apis.embedding_api import EmbeddingAPI
from pathlib import Path
import os
from langchain.schema import Document

# from langchain_chroma import Chroma
# from langchain.schema.document import Document
# from unstructured.partition.pdf import partition_pdf
# from apis.llm_api import LLMAPI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain


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

    def create_temporary_files(self, source_docs):
        """建立臨時文件並返回檔案名稱對應關係。"""
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        doc_names = {}

        for source_doc in source_docs:
            with tempfile.NamedTemporaryFile(delete=False, dir=self.tmp_dir.as_posix(), suffix='.pdf') as tmp_file:
                tmp_file.write(source_doc['content'])  # 寫入文件內容
                file_name = Path(tmp_file.name).name
                doc_names[file_name] = source_doc['name']
                logging.info(f"Created temporary file: {file_name}")

        return doc_names

    def load_documents(self):
        # 加載 PDF 文件
        # 使用 PyPDFDirectoryLoader 從指定的目錄中加載所有 PDF 文件
        loader = PyPDFDirectoryLoader(self.tmp_dir.as_posix(), glob='**/*.pdf')

        # 加載 PDF 文件，並將其儲存在 documents 變數中
        documents = loader.load()

        # 如果沒有加載到任何文件，拋出異常提示
        if not documents:
            raise ValueError("No documents loaded. Please check the PDF files.")

        # 紀錄已加載的文件數量到日誌中
        logging.info(f"Loaded {len(documents)} documents")

        # 返回已加載的文件
        return documents

    def delete_temporary_files(self):
        # 刪除臨時文件
        for file in self.tmp_dir.iterdir():
            try:
                logging.info(f"Deleting temporary file: {file}")
                file.unlink()
            except Exception as e:
                logging.error(f"Error deleting file {file}: {e}")

    def split_documents_into_chunks(self, documents):
        # print(documents)
        # 將文件拆分成塊
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
        document_chunks = text_splitter.split_documents(documents)
        # print("3. document_chunks: ", document_chunks)
        logging.info(f"Split documents into {len(document_chunks)} chunks")
        return document_chunks

    def split_documents_into_chunks_1(self, documents):
        """
        將文件拆分為基於標題的結構和字元級內容小塊。
        """
        # 引入所需模組
        from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

        document_chunks = []  # 用於存儲符合 embeddings_on_local_vectordb 格式的內容
        markdown_document = ""  # 暫存合併的文檔內容

        # 處理輸入文檔
        for doc in documents:
            # 檢查文檔是否包含有效內容
            if hasattr(doc, "page_content") and doc.page_content:
                markdown_document += str(doc.page_content)  # 將內容轉為字串後合併
            else:
                logging.warning("Skipping a document with no content.")  # 記錄警告訊息
                continue

        # 使用 MarkdownHeaderTextSplitter 進行標題層級拆分
        headers_to_split_on = [
            ("#", "Header 1"),  # 標題層級1
            ("##", "Header 2"),  # 標題層級2
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(markdown_document)  # 拆分文檔
        # print('1. md_header_splits: ', md_header_splits)

        # 定義字元級拆分器
        chunk_size = 1400  # 設定塊大小
        chunk_overlap = 600  # 設定塊重疊大小
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # 處理每個標題分組
        for header_group in md_header_splits:
            # 將內容拆分為更小的塊
            _splits = text_splitter.split_text(header_group.page_content)
            for split in _splits:
                # 將塊包裝成符合 `embeddings_on_local_vectordb` 的格式
                document_chunks.append({
                    "page_content": split,  # 小塊內容
                    "metadata": header_group.metadata  # 原標題的元數據
                })


        # 將 A 格式轉換為 B 格式，但不包含 source_path
        documents = [
            Document(
                metadata={'page': i},  # 僅保留頁碼資訊
                page_content=chunk['page_content']
            )
            for i, chunk in enumerate(document_chunks)
        ]

        # 記錄拆分完成的日誌訊息
        logging.info(f"Split documents into {len(document_chunks)} chunks.")
        return documents

    def split_documents_into_chunks_2(self, documents):
        """
        遞歸地將文檔根據標題層級和字元內容切分成小塊，直到滿足指定大小。
        """
        # 引入所需模組
        from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
        from langchain.schema import Document
        import logging

        # 定義字元級拆分器
        chunk_size = 1000  # 設定塊大小
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=300)

        # 定義標題層級
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
        ]

        def _split_recursively(document_content, current_level):
            """
            遞歸地依據標題層級拆分文件，直到塊大小小於或等於 chunk_size。
            """
            # if current_level >= len(headers_to_split_on):
            #     # 已達到最小標題層級，直接進行字元級拆分
            #     return [{"page_content": chunk, "metadata": {}} for chunk in text_splitter.split_text(document_content)]

            # 依據當前標題層級拆分文檔
            header, header_name = headers_to_split_on[current_level]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[(header, header_name)])
            md_splits = markdown_splitter.split_text(document_content)
            print('1. md_splits', md_splits)

            chunks = []
            for header_group in md_splits:
                # 若拆分後的內容超出塊大小，繼續依據下一層標題層級拆分
                if len(header_group.page_content) > chunk_size:
                    chunks.extend(_split_recursively(header_group.page_content, current_level + 1))
                    print('2-1. chunks', chunks)
                else:
                    # 若內容符合大小，直接加入結果
                    chunks.append({
                        "page_content": header_group.page_content,
                        "metadata": header_group.metadata
                    })
                    print('2-2. chunks', chunks)
            return chunks

        # 處理輸入文檔
        document_chunks = []
        for doc in documents:
            # 檢查文檔是否包含有效內容
            if hasattr(doc, "page_content") and doc.page_content:
                document_chunks.extend(_split_recursively(doc.page_content, 0))  # 開始從最高層級拆分
            else:
                logging.warning("Skipping a document with no content.")  # 記錄警告訊息
                continue


        print('3. document_chunks', document_chunks)

        # 將結果轉換為 Document 格式
        documents = [
            Document(
                metadata={'page': i, **chunk.get('metadata', {})},  # 合併頁碼與元數據
                page_content=chunk['page_content']
            )
            for i, chunk in enumerate(document_chunks)
        ]

        # 記錄拆分完成的日誌訊息
        logging.info(f"Split documents into {len(documents)} chunks.")
        return documents

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
