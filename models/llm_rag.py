# -*- coding: utf-8 -*-
"""
RAGModel ‒ 檢索增強生成模型（模組化版本）
================================================
功能
----
* 透過向量資料庫檢索 (cosine / MMR / Cross‑Encoder rerank) 與 LLM 生成，實現 RAG 問答。
* 支援 Query Rewrite、MMR 與 Reranker，可由 `RAGConfig` 彈性開關。
* Singleton 載入 CrossEncoder，並於查詢後自動釋放 CUDA 記憶體。

使用方式
--------
```python
from rag_model import RAGModel, RAGConfig

config = RAGConfig(
    use_rewrite=True,      # 是否啟用查詢重寫
    use_mmr=True,          # 是否啟用 MMR 多樣性
    use_reranker=False,    # 是否啟用 CrossEncoder rerank
    cosine_k=20,
    mmr_k=15,
    rerank_k=5,
)

rag = RAGModel(chat_session_data, config)
answer, docs, rewritten = rag.query_llm_rag(query, doc_summary)
```
"""

import os
import gc
from typing import Dict, List, Tuple, Optional

import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from sentence_transformers import CrossEncoder
from langchain_community.chat_message_histories import ChatMessageHistory


from apis.llm_api import LLMAPI
from apis.embedding_api import EmbeddingAPI
from apis.file_paths import FilePaths

# 關閉 Chroma 遙測
os.environ["CHROMA_TELEMETRY"] = "False"

# ---------------------------------------------------------------------------
# RAGConfig ‒  所有超參集中配置
# ---------------------------------------------------------------------------
class RAGConfig:
    """集中管理 RAG 相關參數，方便外部注入或動態調整。"""

    def __init__(
        self,
        *,
        use_rewrite: bool = True,
        use_mmr: bool = False,
        use_reranker: bool = False,
        cosine_k: int = 5,
        mmr_k: int = 5,
        rerank_k: int = 5,
        batch_size: int = 4,
    ) -> None:
        self.use_rewrite = use_rewrite
        self.use_mmr = use_mmr
        self.use_reranker = use_reranker
        self.cosine_k = cosine_k
        self.mmr_k = mmr_k
        self.rerank_k = rerank_k
        self.batch_size = batch_size


# ---------------------------------------------------------------------------
# CrossEncoder Singleton  (避免重複載入大模型)
# ---------------------------------------------------------------------------
_CROSS_ENCODER: Optional[CrossEncoder] = None

def get_cross_encoder(model_name: str = "BAAI/bge-reranker-v2-m3") -> CrossEncoder:
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _CROSS_ENCODER = CrossEncoder(model_name, device=device)
        except RuntimeError:
            _CROSS_ENCODER = CrossEncoder(model_name, device="cpu")
    return _CROSS_ENCODER


# ---------------------------------------------------------------------------
# RAGModel 主類別
# ---------------------------------------------------------------------------
class RAGModel:
    """檢索增強生成模型 (Retrieval‑Augmented Generation)。"""

    # 是否在查詢中使用聊天歷史 (全域預設，可由 session 覆寫)
    DEFAULT_USE_CHAT_HISTORY: bool = False

    def __init__(self, chat_session_data: Dict, config: Optional[RAGConfig] = None):
        # ----- 會話與路徑 -----
        self.chat_session_data = chat_session_data
        self.mode = chat_session_data.get("mode")
        self.llm_option = chat_session_data.get("llm_option")
        file_paths = FilePaths()
        username = chat_session_data.get("username")
        conversation_id = chat_session_data.get("conversation_id")
        self.vector_store_dir = file_paths.get_local_vector_store_dir(username, conversation_id)

        # ----- 配置 -----
        self.config = config or RAGConfig()
        self.batch_size = self.config.batch_size
        self.reranker = get_cross_encoder() if self.config.use_reranker else None
        self._last_rewritten_query: str = ""

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def query_llm_rag(self, query: str, doc_summary: str) -> Tuple[str, List[Document], str]:
        """主流程：重寫→檢索→生成。"""
        try:
            # 1. 取得 LLM 與 Embedding function
            llm = LLMAPI.get_llm(self.mode, self.llm_option)
            embed_fn = EmbeddingAPI.get_embedding_function(self.mode, self.chat_session_data.get("embedding"))

            # 2. 向量資料庫
            self.vector_db = Chroma(
                embedding_function=embed_fn,
                persist_directory=self.vector_store_dir.as_posix(),
            )

            # 3. 建立 retriever (含重寫 / rerank)
            retriever = self._build_rewrite_retrieve_rerank_chain(llm, doc_summary)

            # 4. 建立 Conversational RAG chain
            rag_chain = self._build_conversational_rag_chain(llm, retriever)

            # 5. 執行
            result = rag_chain.invoke({
                "input": query,
                "chat_history": self._get_chat_history_from_session(),
            })

            answer = result.get("answer", "")
            docs = result.get("context", [])
            return answer, docs, self._last_rewritten_query
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    # ------------------------------------------------------------------
    # Chain Builders
    # ------------------------------------------------------------------
    def _build_rewrite_retrieve_rerank_chain(self, llm, doc_summary: str) -> RunnableLambda:
        """把 Rewrite、Cosine、MMR、Rerank 串為同一個 retriever。"""

        # ---- (A) Query Rewrite Prompt → LLM → parser ----
        rewrite_prompt = self._build_rewrite_prompt(doc_summary)
        rewrite_chain = rewrite_prompt | llm | StrOutputParser()

        # ---- (B) 內部函式：rewrite → retrieve → rerank ----
        def _inner(inputs):
            origin_query = inputs["input"]
            chat_history = inputs["chat_history"]

            # 1) 是否重寫
            if self.config.use_rewrite:
                rewritten = rewrite_chain.invoke({
                    "input": origin_query,
                    "chat_history": chat_history,
                }).strip() or origin_query
            else:
                rewritten = origin_query
            self._last_rewritten_query = rewritten

            # 2) Cosine 檢索
            cosine_docs = self.vector_db.similarity_search(rewritten, k=self.config.cosine_k)

            # 3) MMR（可選）
            if self.config.use_mmr:
                mmr_docs = self.vector_db.max_marginal_relevance_search(
                    rewritten, k=self.config.mmr_k, fetch_k=self.config.cosine_k)
            else:
                mmr_docs = cosine_docs

            # 4) Rerank（可選）
            if self.config.use_reranker and self.reranker:
                if len(mmr_docs) <= self.config.rerank_k:
                    final_docs = mmr_docs
                else:
                    pairs = [(rewritten, d.page_content) for d in mmr_docs]
                    scores = self.reranker.predict(pairs, batch_size=self.batch_size)
                    ranked = sorted(zip(scores, mmr_docs), key=lambda p: p[0], reverse=True)[: self.config.rerank_k]
                    final_docs = [d for _, d in ranked]
            else:
                final_docs = mmr_docs

            return final_docs

        return RunnableLambda(_inner)

    def _build_conversational_rag_chain(self, llm, retriever):
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            你是知識問答助手。以下是相關文件內容，請擷取必要段落後作答。
            若內容不足以回答，請誠實回覆「我不知道」。
            {context}
            """),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)
        return RunnableWithMessageHistory(
            rag_chain,
            self._get_chat_history_from_session,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    # ------------------------------------------------------------------
    # Prompt Builders & Helpers
    # ------------------------------------------------------------------
    def _build_rewrite_prompt(self, doc_summary: str) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            (
                "system",
                f"""
                你是一個查詢重寫助手，目的是根據聊天歷史，將使用者問題改寫成能獨立理解的完整查詢句。
                請**僅重寫，不要回答**，並**保持原始語意**；以聊天歷史為主要依據，文件摘要僅供輔助參考。
                文件摘要：{doc_summary}
                """,
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    from langchain_community.chat_message_histories import ChatMessageHistory

    def _get_chat_history_from_session(self) -> ChatMessageHistory:
        """
        將 chat_session_data["chat_history"]（list[dict]）轉換為 LangChain 所需的 ChatMessageHistory。
        支援以下兩種格式：
        1. {"role": "user" / "assistant", "content": "..."}
        2. {"user_query": "...", "ai_response": "..."}（會自動拆分為 user / assistant）
        如果未啟用記憶，則返回空歷史。
        """
        use_mem = self.chat_session_data.get("use_memory", self.DEFAULT_USE_CHAT_HISTORY)
        history = ChatMessageHistory()

        if not use_mem:
            print("📭 Chat memory is OFF（use_memory=False），不送出歷史紀錄")
            return history

        raw_history = self.chat_session_data.get("chat_history", [])
        # print(f"📚 Chat memory is ON，開始轉換 {len(raw_history)} 則訊息為 ChatMessageHistory")
        # print(raw_history)

        for rec in raw_history:
            # 兼容格式 1: {"role": "user", "content": "..."}
            if "role" in rec and "content" in rec:
                role = rec.get("role", "").lower()
                content = rec.get("content", "")

                if role == "user":
                    history.add_user_message(content)
                elif role in ("assistant", "ai"):
                    history.add_ai_message(content)
                else:
                    print(f"⚠️ 發現未知角色：{role}，內容已略過")

            # 兼容格式 2: {"user_query": "...", "ai_response": "..."}
            elif "user_query" in rec and "ai_response" in rec:
                user_msg = rec.get("user_query", "").strip()
                ai_msg = rec.get("ai_response", "").strip()

                if user_msg:
                    history.add_user_message(user_msg)
                if ai_msg:
                    history.add_ai_message(ai_msg)

            else:
                print(f"⚠️ 發現未知訊息格式：{rec}，內容已略過")

        # print("✅ 已轉換為 ChatMessageHistory，型別為：", type(history))
        print("📝 內容筆數：", len(history.messages))

        # for i, msg in enumerate(history.messages[:2]):
        #     print(f"🧾 第{i + 1}則訊息：{msg.type} / {msg.content}")

        return history

