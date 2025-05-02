import tempfile
from langchain_chroma import Chroma
from apis.file_paths import FilePaths
from apis.embedding_api import EmbeddingAPI
import os
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import logging
from pathlib import Path
from typing import List
# from langchain.document_loaders import PyMuPDFLoader as PyMuPDF4LLMLoader, TextLoader
from langchain_community.document_loaders import PyMuPDFLoader as PyMuPDF4LLMLoader, TextLoader
from langchain.schema import Document

# 關閉 Chroma 遠端遙測資料收集（保護使用者隱私）
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

    from typing import Tuple

    def create_temporary_files(self, source_doc) -> Tuple[str, str]:
        """根據上傳檔案類型（pdf / md）建立臨時文件並返回 (臨時檔名, 原始檔名)。"""
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        file_ext = source_doc.get('type', 'pdf')  # 預設為 pdf，如果有 type 字段就用 type
        suffix = f'.{file_ext}'

        with tempfile.NamedTemporaryFile(delete=False, dir=self.tmp_dir.as_posix(), suffix=suffix) as tmp_file:
            tmp_file.write(source_doc['content'])  # 寫入文件內容
            tmp_name = Path(tmp_file.name).name
            org_name = source_doc['name']
            logging.info(f"Created temporary file: {tmp_name}")
        return tmp_name, org_name

    def load_documents(self) -> Document:
        """從臨時目錄中加載 PDF 和 MD 文件"""
        self._pdf_to_md()
        document = self._load_md_file()
        return document

    def _pdf_to_md(self):
        """
        載入 PDF，轉為 Markdown，並儲存在相同目錄
        """
        for pdf in self.tmp_dir.rglob("*.pdf"):
            try:
                logging.info(f"🔄 載入 PDF：{pdf.name}")
                loader = PyMuPDF4LLMLoader(pdf.as_posix())
                docs = loader.load()

                # 儲存為 Markdown
                md_path = self.tmp_dir / f"{pdf.stem}.md"
                try:
                    with open(md_path, "w", encoding="utf-8") as f:
                        for doc in docs:
                            f.write(doc.page_content + "\n\n---\n\n")
                    logging.info(f"📝 已儲存 PDF 轉 Markdown：{md_path.name}")
                except Exception as e:
                    logging.warning(f"⚠️ 儲存 Markdown 失敗：{md_path.name}，錯誤：{e}")
            except Exception as e:
                logging.warning(f"❌ PDF 載入失敗：{pdf.name}，錯誤：{e}", exc_info=True)

    def _load_md_file(self) -> Document:
        """
        載入單一 Markdown 檔案內容並回傳 Document
        """
        md_files = list(self.tmp_dir.rglob('*.md'))
        if not md_files:
            raise FileNotFoundError("❗ 找不到任何 Markdown (.md) 檔案。")

        md_file = md_files[0]  # 只處理第一個檔案

        try:
            loader = TextLoader(md_file.as_posix(), encoding='utf-8')
            md_documents = loader.load()

            if not md_documents:
                raise ValueError("❗ 找到 Markdown 檔案，但讀取內容為空。")

            logging.info(f"✅ 已載入 Markdown 文件：{md_file.name}")
            return md_documents[0]  # 只取第一份文件

        except Exception as e:
            logging.warning(f"⚠️ 載入 Markdown 文件失敗：{md_file.name}，錯誤：{e}")
            raise e

    def delete_temporary_files(self):
        # 刪除臨時文件
        for file in self.tmp_dir.iterdir():
            try:
                logging.info(f"Deleting temporary file: {file}")
                file.unlink()
            except Exception as e:
                logging.error(f"Error deleting file {file}: {e}")

    def chunk_document_with_md(self, document: Document, org_name: str, header_depth: int = 3):
        """
        將 Markdown 文檔依照標題與字元級拆分成多個小 chunk。
        支援自定義 Markdown 標題深度（例如 Header 1 ~ Header 6）。
        每個 chunk 都會附帶 metadata 與原始檔案名稱。
        """

        document_chunks = []  # 用來暫存每個拆分出來的小段落

        # 1. 動態建立 Markdown 標題層級（例如: [('#', 'Header 1'), ..., ('######', 'Header 6')])
        headers_to_split_on = [(f"{'#' * i}", f"Header {i}") for i in range(1, header_depth + 1)]

        # 2. 使用 MarkdownHeaderTextSplitter 按標題區塊拆分
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        # 傳入純文字內容做拆分（不能是整個 Document 物件）
        md_header_splits = markdown_splitter.split_text(document.page_content)

        # 3. 使用遞迴式字元切割器進行段落內部細拆
        chunk_size = 800  # 每個小塊的最大字元數
        chunk_overlap = 300  # 每塊之間的重疊字數
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # 對每個 Markdown 標題段落做進一步切割
        for header_group in md_header_splits:
            # 拆成更小的 chunk
            splits = text_splitter.split_text(header_group.page_content)

            for split in splits:
                # 將拆出來的小段落與 metadata 一起存起來
                document_chunks.append({
                    "page_content": split,
                    "metadata": header_group.metadata,
                    "org_name": org_name
                })

        # 4. 將所有小段落轉換成 LangChain 的 Document 格式
        transformed_documents = []
        for i, chunk in enumerate(document_chunks):
            # 將原始檔名與標題 metadata 合併進 metadata，並加上 page 編號
            transformed_documents.append(
                Document(
                    metadata={**chunk["metadata"], 'chunk_id': i+1, 'org_name': chunk["org_name"]},
                    page_content=chunk["org_name"] + "\n" + str(chunk["metadata"]) + "\n" + chunk["page_content"]
                )
            )

        # 記錄成功資訊
        logging.info(f"✅ 成功將文檔拆分為 {len(transformed_documents)} 個塊。")

        return transformed_documents

    def embed_into_vectordb(self, document_chunks):
        """
        將已拆分的文件區塊（chunks）嵌入至本地向量資料庫（使用 Chroma），以便後續檢索使用。
        """
        # 從 chat_session_data 中取得用戶指定的嵌入模式
        mode = self.chat_session_data.get("mode")
        embedding = self.chat_session_data.get("embedding")

        # 透過嵌入 API 取得對應的嵌入函式（embedding function）
        embedding_function = EmbeddingAPI.get_embedding_function(mode, embedding)

        # 若文件區塊為空，拋出錯誤提示
        if not document_chunks:
            raise ValueError("❗ 找不到文件區塊可供嵌入。請確認文件是否已成功切分。")

        # 使用 Chroma 向量資料庫，將文件區塊嵌入並儲存至本地磁碟
        Chroma.from_documents(
            documents=document_chunks,  # 要嵌入的 LangChain Document 物件清單
            embedding=embedding_function,  # 使用的嵌入函式
            persist_directory=self.vector_store_dir.as_posix()  # 本地向量資料庫儲存路徑
        )

        # 記錄成功訊息：表示資料已成功嵌入並儲存
        logging.info(f"✅ 向量資料庫已成功儲存於：{self.vector_store_dir}")

