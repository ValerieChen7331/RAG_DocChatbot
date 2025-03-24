import streamlit as st
import pandas as pd
import sqlite3
import logging

from apis.llm_api import LLMAPI
from apis.embedding_api import EmbeddingAPI
from apis.file_paths import FilePaths

# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
# from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory

# from langchain.chains import ConversationChain
# from langchain.chains import ConversationalRetrievalChain

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
# from langchain.memory import ConversationBufferMemory
# from langchain.memory import ChatMessageHistory

class RAGModel:
    def __init__(self, chat_session_data):
        # 初始化 hat_session_data
        self.chat_session_data = chat_session_data
        self.mode = chat_session_data.get("mode")
        self.llm_option = chat_session_data.get('llm_option')

        # 初始化文件路徑
        file_paths = FilePaths()
        self.output_dir = file_paths.get_output_dir()
        username = chat_session_data.get("username")
        conversation_id = chat_session_data.get("conversation_id")
        self.vector_store_dir = file_paths.get_local_vector_store_dir(username, conversation_id)


    def query_llm_rag(self, query):
        """使用 RAG 查詢 LLM，根據給定的問題和檢索的文件內容返回答案。"""
        # 初始化語言模型
        llm = LLMAPI.get_llm(self.mode, self.llm_option)
        # 初始化 embedding 模型
        embedding = self.chat_session_data.get("embedding")
        embedding_function = EmbeddingAPI.get_embedding_function(self.mode, embedding)

        # 建立向量資料庫和檢索器
        vector_db = Chroma(
            embedding_function=embedding_function,
            persist_directory=self.vector_store_dir.as_posix()
        )
        retriever = vector_db.as_retriever(earch_type="mmr", search_kwargs={"k": 5})


        # qa_chain = ConversationalRetrievalChain.from_llm(
        #     llm=llm,  # 獲取 LLM API 物件
        #     retriever=retriever,  # 設置檢索器
        #     return_source_documents=True,  # 返回檢索到的文件
        #     combine_docs_chain_kwargs={"prompt": self._rag_prompt()}  # prompt 模板
        # )

        # 創建文檔組合鏈
        combine_docs_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=self._rag_prompt()
        )

        # 創建檢索鏈
        qa_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=combine_docs_chain
        )


        # !!修改 chat_history!
        chat_history = [ {'user_query': '', 'ai_response': ''} ]

        print("3. result_rag = qa_chain.invoke")
        # 使用 RAG 查詢 LLM，生成答案
        # result_rag = qa_chain.invoke({'question': query, 'chat_history': chat_history})
        print(query)
        print(chat_history)
        result_rag = qa_chain.invoke({'input': query, 'chat_history': chat_history})



        print("3.5. result_rag: ", result_rag)

        response = result_rag.get('answer', '')  # 取得回答
        retrieved_documents = result_rag.get('source_documents', [])  # 取得檢索到的文件
        print("4. return response, retrieved_document")


        # 保存檢索到的數據到 CSV 文件
        self._save_retrieved_data_to_csv(query, retrieved_documents, response)
        return response, retrieved_documents

    def _rag_prompt(self):
        """生成 RAG 查詢 LLM 所需的提示模板。"""
        template = """
        你是會計制度專家，要依照我所建立的國內出差全文去回答:
                    1. 計程車費用查詢
                    - 若需查詢計程車費，請提供起訖點，並優先參考“計程車費報支標準表”及"交通費及計程車費合併"的金額。
                    - 在回答問題時，您可以提供以下資訊，請注意，這個金額僅供參考，實際報支仍需依照“計程車費報支標準表”為主。計程車費不需要憑證，您可以直接報支相關費用。
                    - 每次回答後，請提醒用戶：如需下載"計程車費報支標準表"或"交通費及計程車費合併"，請提問該表的連結下載，以供核對金額是否正確。
                    - 計程車費不需要憑證。
                    2. 住宿費用請優先引用"出差特約旅館"的資料，請將平日合約價、是否提供早餐及地址列出來即可，不要列出合約期間，後面再附加住宿前須提供員工識別證才享有合約價格。
                    3. 經濟路線的處理
                      - 內湖大樓至台北車站的計程車費用應以最近的車站（如南港車站或松山火車站）為依據。
                      - 內湖大樓或台塑大樓前往工三廠區及周遭廠區的出行，請優先考慮搭乘汎航客運。
                    4. 憑證別代號查詢
                    - 若不清楚如何輸入憑證別代號，請提供知識庫及下載點的連結，以便查詢。
                    5. 出差路線概述
                    - 內湖大樓出差至麥寮廠，請提供以下資訊的表格，並說明一律優先搭乘交通車前往，除非客滿才能選擇搭火車或高鐵。
                    - 起點
                    - 終點
                    - 交通方式
                    - 費用
                    - 核決權限
                    6. 不清楚問題的處理
                    - 若問題有不明之處，建議洽詢相關的審核人員，以免引用錯誤的答案。
                    7. 人工智能的角色聲明
                    - 每次回答後，請提醒用戶：本系統為人工智能，僅提供輔助性質的資訊，最終決策仍以會計審核員的意見為主。
                    -----
                    使用以下檢索到的內容來回答問題。\
                    注意! 如果檢索到的內容無法回答，請直接說您不知道。請勿自編造!\
                    檢索到的內容: {context}
                    使用者提問: {input}
        """
        return PromptTemplate(input_variables=["context", "input"], template=template)

    # def _get_chat_history_from_session(self) -> str:
    #     """從 session state 中取得聊天記錄，格式化為字符串形式."""
    #     chat_history_data = self.chat_session_data.get('chat_history', [])
    #     formatted_history = ""
    #     for record in chat_history_data:
    #         user_query = record['user_query']
    #         ai_response = record['ai_response']
    #         formatted_history += f"User: {user_query}\nAI: {ai_response}\n"
    #     return formatted_history

    def _get_chat_history_from_session(self) -> ChatMessageHistory:
        """從 session state 中取得聊天記錄，若無則創建新的聊天記錄。"""
        # 從 session 中獲取聊天記錄，如果不存在，則初始化空的聊天記錄
        chat_history_data = self.chat_session_data.get('chat_history', [])
        chat_history = ChatMessageHistory()
        for record in chat_history_data:
            user_query, ai_response = record['user_query'], record['ai_response']
            chat_history.add_user_message(user_query)
            chat_history.add_ai_message(ai_response)
        return chat_history

    def _save_retrieved_data_to_csv(self, query, retrieved_data, response):
        """將檢索到的數據保存到 CSV 文件中。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)  # 確保輸出目錄存在
        output_file = self.output_dir.joinpath('retrieved_data.csv')  # 定義輸出文件路徑

        # 組合檢索到的文件內容
        # context = "\n\n".join([f"文檔 {i + 1}:\n{chunk}" for i, chunk in enumerate(retrieved_data)])
        context = "\n\n".join([f"文檔 {i + 1}:\n{chunk.page_content}" for i, chunk in enumerate(retrieved_data)])

        new_data = {"Question": [query], "Context": [context], "Response": [response]}  # 新數據
        new_df = pd.DataFrame(new_data)  # 將新數據轉換為 DataFrame

        if output_file.exists():
            existing_df = pd.read_csv(output_file)  # 如果文件已存在，讀取現有數據
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)  # 合併現有數據和新數據
        else:
            combined_df = new_df  # 否則僅使用新數據

        combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')  # 保存數據到 CSV 文件
