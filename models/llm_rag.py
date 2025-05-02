# -*- coding: utf-8 -*-
"""
RAGModel - 檢索增強生成模型
==========================
【功能】
RAG 結合 retrieval 和 generation，從向量資料庫中檢索相關文件後，由大型語言模型生成回答。

【核心特色】
1. 多層級 Retriever：cosine 相似度 → MMR 多樣性優化 → Reranker 重排序
2. 查詢重寫（Query Rewrite）：使用文件摘要與對話歷史改善查詢
3. GPU 加速與記憶體保護：自動釋放 CUDA 記憶體、防止 OOM、GPU 記憶體不足自動切換 CPU
4. 模型 Singleton 設計：CrossEncoder 預先載入避免重複初始化

【可設定參數】
* cosine: 初始檢索文件數量（預設 20）
* mmr: 多樣性優化後文件數量（預設 15）
* reranker: 最終重新排序後文件數量（預設 5）
* reranker_batch_size: 批次大小（預設 4）
其他設定：
* CrossEncoder: 使用 'BAAI/bge-reranker-v2-m3'

【使用方式】
✅ llm_rag = RAGModel(chat_session_data)
✅ response, docs, rewritten_query = llm_rag.query_llm_rag(query, doc_summary)
"""

import os, torch, gc
from typing import Dict, List, Tuple
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from sentence_transformers import CrossEncoder

from apis.llm_api import LLMAPI
from apis.embedding_api import EmbeddingAPI
from apis.file_paths import FilePaths
from models.database_base import BaseDB

# 關閉 Chroma 遙測功能，避免傳送使用資料
os.environ["CHROMA_TELEMETRY"] = "False"

# ---------- 預設參數 ----------
# 預設檢索文件數量配置
# "cosine": 20, "mmr": 15, "reranker": 5
DEFAULT_K: Dict[str, int] = {"cosine": 15, "reranker": 5}

# 重排序批次大小，影響 GPU 記憶體使用量
DEFAULT_BATCH_SIZE: int = 4

# 是否啟用查詢重寫功能，將使用者輸入轉換為更完整的查詢
DEFAULT_REWRITE_QUERY: bool = True

# 是否在查詢中使用聊天歷史記錄
DEFAULT_USE_CHAT_HISTORY: bool = False

# ---------- CrossEncoder Singleton 實作 ----------
# 使用全域變數儲存 CrossEncoder 實例，避免重複載入模型
_CROSS_ENCODER: CrossEncoder | None = None

def get_cross_encoder(model_name: str = "BAAI/bge-reranker-v2-m3") -> CrossEncoder:
    # bge-reranker-large
    """
    取得 CrossEncoder 單例實例
    採用 Singleton 設計模式，確保全系統只載入一次模型，節省記憶體

    參數:
        model_name: 重排序器模型名稱，預設使用 BAAI/bge-reranker-v2-m3

    返回:
        CrossEncoder 實例
    """
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        try:
            # 優先嘗試使用 GPU 加速
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _CROSS_ENCODER = CrossEncoder(model_name, device=device)
        except RuntimeError:
            # GPU 記憶體不足時自動退回使用 CPU
            _CROSS_ENCODER = CrossEncoder(model_name, device="cpu")
    return _CROSS_ENCODER


class RAGModel:
    """
    檢索增強生成模型主類別
    整合文件檢索與大型語言模型，實現智能問答功能
    """

    def __init__(self, chat_session_data: Dict):
        """
        初始化 RAG 模型

        參數:
            chat_session_data: 包含對話 ID、用戶資訊、模型配置等的字典
        """
        # 儲存會話資料，包含用戶、對話ID、模型選項等
        self.chat_session_data = chat_session_data
        # 獲取模型運行模式 (可能是本地、線上等)
        self.mode = chat_session_data.get("mode")
        # 獲取語言模型選項 (模型名稱、參數等)
        self.llm_option = chat_session_data.get("llm_option")

        # 設定檔案路徑
        file_paths = FilePaths()
        # 輸出目錄路徑
        self.output_dir = file_paths.get_output_dir()
        # 從會話資料中獲取用戶名和對話ID
        username = chat_session_data.get("username")
        conversation_id = chat_session_data.get("conversation_id")
        # 根據用戶和對話ID建立向量儲存目錄
        self.vector_store_dir = file_paths.get_local_vector_store_dir(username, conversation_id)

        # 複製預設檢索配置，避免修改全域變數
        self.k_cfg = DEFAULT_K.copy()
        # 設定批次大小
        self.batch_size = DEFAULT_BATCH_SIZE

        # 初始化重排序器 (使用 Singleton 模式)
        self.reranker = get_cross_encoder()
        # 追蹤最後重寫的查詢，用於調試和記錄
        self._last_rewritten_query: str = ""

    def query_llm_rag(self, query: str, doc_summary: str) -> Tuple[str, List[Document], str]:
        """
        執行 RAG 查詢主流程

        參數:
            query: 使用者問題
            doc_summary: 文件摘要，用於查詢重寫

        返回:
            Tuple[str, List[Document], str]: (答案, 相關文件列表, 重寫後的查詢)
        """
        try:
            # 根據模式和選項獲取適當的語言模型
            llm = LLMAPI.get_llm(self.mode, self.llm_option)
            # 獲取嵌入模型配置
            embedding_name = self.chat_session_data.get("embedding")
            # 獲取嵌入函數
            embed_fn = EmbeddingAPI.get_embedding_function("內部LLM", embedding_name)

            # 初始化向量數據庫連接
            vector_db = Chroma(
                embedding_function=embed_fn,
                persist_directory=self.vector_store_dir.as_posix(),
            )
            # 儲存為實例變數，以便其他方法使用
            self.vector_db = vector_db

            # 創建支援查詢重寫和重排序的檢索器
            history_aware_retriever = self._rewrite_retrieve_rerank(llm, doc_summary)

            # 建立會話式 RAG 鏈
            conversational_rag = self._create_conversational_rag_chain(llm, history_aware_retriever)

            # 執行查詢，傳入用戶問題和聊天歷史
            result = conversational_rag.invoke({
                "input": query,
                "chat_history": self._get_chat_history_from_session()
            })
            # 從結果中提取答案和相關文件
            answer = result.get("answer", "")
            docs = result.get("context", [])

            # 記錄重寫後的查詢
            rewritten_query = self._last_rewritten_query

            # 清理 GPU 記憶體，避免累積佔用
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            return answer, docs, rewritten_query

        except Exception as exc:
            # 發生錯誤時記錄並返回空結果
            print(f"[query_llm_rag] 發生錯誤：{exc}")
            return "", [], ""

    def _rewrite_retrieve_rerank(self, llm, doc_summary: str) -> RunnableLambda:
        """
        創建整合查詢重寫、檢索和重排序的流程

        參數:
            llm: 語言模型實例
            doc_summary: 文件摘要

        返回:
            RunnableLambda: 可執行的檢索流程
        """
        # 建立查詢重寫提示
        prompt = self._build_rewrite_prompt(doc_summary)
        # 創建問題重寫生成器鏈: 提示 -> LLM -> 解析
        question_gen = prompt | llm | StrOutputParser()
        # 獲取重排序器
        reranker = self.reranker

        # 從配置中獲取各階段的文件數量
        mmr_k = self.k_cfg.get("mmr")  # 最大邊際相關性重排序階段
        rerank_k = self.k_cfg.get("reranker")  # 重排序階段
        cosine_k = self.k_cfg.get("cosine", 10)  # 初始餘弦相似度檢索階段

        def _inner(inputs):
            """內部函數，執行三階段檢索過程"""
            # 執行查詢重寫，如果啟用了該功能
            rewritten = question_gen.invoke({
                "input": inputs["input"],
                "chat_history": inputs["chat_history"]
            }) if DEFAULT_REWRITE_QUERY else inputs["input"]

            # 記錄最後重寫的查詢
            self._last_rewritten_query = rewritten

            # 階段1: 基於餘弦相似度的初始檢索
            cosine_docs = self.vector_db.similarity_search(rewritten, k=cosine_k)
            # 提取文件ID用於調試
            cosine_ids = [d.metadata.get("chunk_id", d.metadata.get("id")) for d in cosine_docs]

            # 階段2: 最大邊際相關性重排序 (如果啟用)
            if mmr_k:
                # MMR 算法幫助增加多樣性，減少重複資訊
                mmr_docs = self.vector_db.max_marginal_relevance_search(
                    rewritten, k=mmr_k, fetch_k=cosine_k)
            else:
                # 如果未啟用 MMR，則直接使用餘弦相似度結果
                mmr_docs = cosine_docs
            # 提取文件ID用於調試
            mmr_ids = [d.metadata.get("chunk_id", d.metadata.get("id")) for d in mmr_docs]

            # 階段3: 交叉編碼器重排序 (如果啟用)
            if rerank_k:
                if len(mmr_docs) <= rerank_k:
                    # 如果文件數量已少於需求，直接使用
                    final_docs = mmr_docs
                else:
                    # 建立查詢-文件對，用於重排序
                    pairs = [(rewritten, d.page_content) for d in mmr_docs]
                    # 預測相關性分數
                    scores = reranker.predict(pairs, batch_size=self.batch_size)
                    # 按相關性排序並選取前 rerank_k 個
                    ranked = sorted(zip(scores, mmr_docs), key=lambda x: x[0], reverse=True)[:rerank_k]
                    final_docs = [d for _, d in ranked]
            else:
                # 如果未啟用重排序，直接使用前階段結果
                final_docs = mmr_docs

            # 提取最終文件ID用於調試
            rerank_ids = [d.metadata.get("chunk_id", d.metadata.get("id")) for d in final_docs]

            # 打印調試信息
            print("\n===== Debug info =====")
            print("Rewrite  :", rewritten)
            print("Cosine   :", cosine_ids)
            print("MMR      :", mmr_ids if mmr_k else "(skip)")
            print("Reranker :", rerank_ids if rerank_k else "(skip)")
            print("================================\n")

            return final_docs

        # 將內部函數包裝為可執行的 Lambda
        return RunnableLambda(_inner)

    def _create_conversational_rag_chain(self, llm, history_aware_retriever):
        """
        創建會話式 RAG 鏈

        參數:
            llm: 語言模型實例
            history_aware_retriever: 檢索器

        返回:
            RunnableWithMessageHistory: 可處理聊天歷史的運行鏈
        """
        # 建立 QA 提示模板
        qa_prompt = ChatPromptTemplate.from_messages([
            # 系統提示，引導 LLM 如何回答
            ("system", """
            你是知識問答助手。以下是相關文件內容，請擷取必要段落後作答。
            若內容不足以回答，請誠實回覆「我不知道」。
            {context}
            """),
            # 聊天歷史占位符
            MessagesPlaceholder("chat_history"),
            # 使用者輸入
            ("human", "{input}"),
        ])
        # 創建文件處理鏈，將文件內容與問題結合
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        # 創建檢索鏈，管理檢索和 QA 過程
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        # 將 RAG 鏈與聊天歷史整合
        return RunnableWithMessageHistory(
            rag_chain,
            self._get_chat_history_from_session,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def _build_rewrite_prompt(self, doc_summary: str) -> ChatPromptTemplate:
        """
        創建查詢重寫提示模板

        參數:
            doc_summary: 文件摘要

        返回:
            ChatPromptTemplate: 提示模板
        """
        return ChatPromptTemplate.from_messages([
            # 系統提示，引導 LLM 如何重寫查詢
            ("system", f"""
                參考聊天歷史及文件摘要，將使用者問題重寫成可獨立理解的完整問題。
                只需重寫，不要回答。
                文件摘要：{doc_summary}
            """),
            # 聊天歷史占位符
            MessagesPlaceholder("chat_history"),
            # 使用者輸入
            ("human", "{input}"),
        ])

    def _get_chat_history_from_session(self) -> ChatMessageHistory:
        """
        從會話數據中獲取聊天歷史

        返回:
            ChatMessageHistory: 聊天歷史對象
        """
        # 如果禁用了聊天歷史功能，返回空歷史
        if not DEFAULT_USE_CHAT_HISTORY:
            return ChatMessageHistory()

        # 從會話數據中獲取聊天記錄
        history_data = self.chat_session_data.get("chat_history", [])
        # 創建聊天歷史對象
        history = ChatMessageHistory()
        # 遍歷歷史記錄，添加至聊天歷史對象
        for rec in history_data:
            role = rec.get("role")
            content = rec.get("content", "")
            if role == "user":
                # 添加使用者訊息
                history.add_user_message(content)
            elif role in ("assistant", "ai"):
                # 添加 AI 助手訊息
                history.add_ai_message(content)
        return history