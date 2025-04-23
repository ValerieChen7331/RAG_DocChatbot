import os
import sqlite3
import pandas as pd
from datetime import datetime
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser

from apis.llm_api import LLMAPI
from apis.embedding_api import EmbeddingAPI
from apis.file_paths import FilePaths
from models.database_base import BaseDB  # ✅ 引入共用的資料庫操作基礎類別

# 關閉 Chroma 遙測功能以保護使用者資料隱私
os.environ["CHROMA_TELEMETRY"] = "False"

class RAGModel:
    def __init__(self, chat_session_data):
        """
        初始化 RAG 模型。
        - 儲存使用者 session 設定
        - 準備向量儲存與資料庫路徑
        """
        self.chat_session_data = chat_session_data
        self.mode = chat_session_data.get("mode")
        self.llm_option = chat_session_data.get("llm_option")

        file_paths = FilePaths()
        self.output_dir = file_paths.get_output_dir()
        username = chat_session_data.get("username")
        conversation_id = chat_session_data.get("conversation_id")
        self.vector_store_dir = file_paths.get_local_vector_store_dir(username, conversation_id)

    def query_llm_rag(self, query, doc_summary):
        """
        主查詢函數：接收使用者問題與文件摘要，執行 RAG 流程，回傳回答與文檔。
        - 包含 query 重寫（使用上下文）
        - 結合檢索文檔與 LLM 回答
        - 將查詢歷史與文檔存入資料庫
        """
        try:
            # 初始化 LLM 與向量嵌入模型
            llm = LLMAPI.get_llm(self.mode, self.llm_option)
            embedding = self.chat_session_data.get("embedding")
            embedding_function = EmbeddingAPI.get_embedding_function('內部LLM', embedding)

            # 建立 Chroma 向量資料庫並初始化檢索器
            vector_db = Chroma(
                embedding_function=embedding_function,
                persist_directory=self.vector_store_dir.as_posix()
            )
            retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10})

            # 創建可重寫 query 的檢索器（加上歷史與摘要）
            history_aware_retriever = self._rewrite_query_and_history_aware_retriever(llm, retriever, doc_summary)

            # 建立對話型 RAG chain
            conversational_rag_chain = self._create_conversational_rag_chain(llm, history_aware_retriever)

            # 執行查詢鏈（包含聊天歷史）
            result_rag = conversational_rag_chain.invoke({
                'input': query,
                'chat_history': self._get_chat_history_from_session()
            })

            # 拆解回傳結果
            response = result_rag.get('answer', '')
            retrieved_documents = result_rag.get('context', [])

            # 再次取得重寫後的查詢文字（方便記錄）
            prompt = self._build_rewrite_prompt(doc_summary)
            question_generator = prompt | llm | StrOutputParser()
            rewritten_query = question_generator.invoke({
                "input": query,
                "chat_history": self._get_chat_history_from_session().messages
            })

            print("1. retrieved_documents: ", retrieved_documents)

            return response, retrieved_documents, rewritten_query

        except Exception as e:
            print(f"查詢 query_llm_rag 時發生錯誤: {e}")
            return "", []

    def _build_rewrite_prompt(self, doc_summary: str):
        """
        建立 Query 重寫提示模板，讓 LLM 能根據聊天歷史與文件摘要產出更明確的查詢。
        """
        return ChatPromptTemplate.from_messages([
            ("system", f"""
                根據聊天記錄和文件摘要，重新表述使用者提問。\
                該提問可能參考聊天記錄中的上下文，請將其重構為一個可以獨立理解的問題。\
                不要回答問題，只需重新表述。\
                文件摘要: {doc_summary}
            """),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    def _rewrite_query_and_history_aware_retriever(self, llm, retriever, doc_summary: str):
        """
        包裝 retriever：在檢索前自動呼叫 LLM 重寫 query。
        回傳符合 Runnable 格式的物件以便整合進 chain 中。
        """
        prompt = self._build_rewrite_prompt(doc_summary)
        question_generator = prompt | llm | StrOutputParser()

        def retriever_with_rewritten_query(inputs):
            rewritten = question_generator.invoke({
                "input": inputs["input"],
                "chat_history": inputs["chat_history"]
            })
            print(f"📝 [查詢重寫] 原始查詢: {inputs['input']}")
            print(f"📝 [查詢重寫] 重寫後查詢: {rewritten}")
            return retriever.invoke({"query": rewritten})

        return RunnableLambda(retriever_with_rewritten_query)

    def _create_conversational_rag_chain(self, llm, history_aware_retriever):
        """
        建立對話型 RAG Chain：結合 retriever 與問答鏈，並保有聊天歷史。
        """
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """
                您是回答問題的助手。
                使用以下檢索到的內容來回答問題，請判斷使用者的問題需要由哪分文檔來回答，不一定每份文檔都使用到。
                請印出原文與頁碼，再進行說明，回答使用者問題。
                若無法從檢索內容得到答案，請誠實回答「我不知道」。不得提供臆測或無依據的答案！
                {context}
            """),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        return RunnableWithMessageHistory(
            rag_chain,
            self._get_chat_history_from_session,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def _get_chat_history_from_session(self) -> ChatMessageHistory:
        """
        從 chat_session_data 中取得使用者對話歷史。
        若尚未儲存任何歷史，則回傳空紀錄。
        """
        chat_history_data = self.chat_session_data.get('chat_history', [])
        chat_history = ChatMessageHistory()
        for record in chat_history_data:
            chat_history.add_user_message("")
            chat_history.add_ai_message("")
        return chat_history