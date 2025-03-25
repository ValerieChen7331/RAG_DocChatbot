# llm_model.py
from apis.llm_api import LLMAPI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
import re
from typing import List, Dict

class LLMModel:
    def __init__(self, chat_session_data):
        """
        初始化 LLMModel 物件並儲存使用者 session 設定。
        """
        # 提取 chat_session_data
        self.chat_session_data = chat_session_data
        self.mode = chat_session_data.get('mode')
        self.llm_option = chat_session_data.get('llm_option')

    def query_llm_direct(self, query:str) -> str:
        """
        使用 LLM 直接回答使用者輸入問題（有對話歷史記憶）。
        """
        # 獲取 active_window_index
        active_window_index = self.chat_session_data.get('active_window_index', 0)

        # 確保 self.chat_session_data 中有針對 active_window_index 的 'conversation_memory'
        memory_key = f'conversation_memory_{active_window_index}'
        if memory_key not in self.chat_session_data:
            self.chat_session_data[memory_key] = ConversationBufferMemory(memory_key="history", input_key="input")

        # 使用 ChatMessageHistory 添加對話歷史到 ConversationBufferMemory
        chat_history_data = self.chat_session_data.get('chat_history', [])
        if chat_history_data:
            chat_history = ChatMessageHistory()
            for record in chat_history_data:
                user_query, ai_response = record['user_query'], record['ai_response']
                chat_history.add_user_message(user_query)
                chat_history.add_ai_message(ai_response)

            # 將 ChatMessageHistory 設置為 ConversationBufferMemory 的歷史記錄
            self.chat_session_data[memory_key].chat_memory = chat_history

        llm = LLMAPI.get_llm(self.mode, self.llm_option)
        # 提示模板
        init_prompt = f"""
        你是一位親切、專業且反應快速的問答助理，請使用繁體中文（台灣用語）回覆。 
        以下是對話紀錄：  
        {{history}}
        請根據上述內容，針對以下問題，提供簡單明瞭、正確實用的回答，必要時可適當補充說明，但避免過長或贅述：  
        {{input}}
        """

        # 建立對話鏈
        conversation_chain = ConversationChain(
            llm=llm,
            memory=self.chat_session_data[memory_key],
            prompt=ChatPromptTemplate.from_template(init_prompt)
        )
        # 查詢 LLM 並返回結果
        result = conversation_chain.invoke(input=query)
        response = result.get('response', '')
        return response

    def set_window_title(self, query:str) -> str:
        """
        使用 LLM 根據使用者問題，自動產生簡短視窗標題文字。
        """
        try:
            llm = LLMAPI.get_llm(self.mode, self.llm_option)    # 獲取 LLM API 物件
            prompt_template = self._title_prompt()  # 獲取prompt模板
            formatted_prompt = prompt_template.format(query=query)  # 格式化prompt模板，插入query

            # 調用 LLM 生成 window title
            title = llm.invoke(formatted_prompt)
            if self.chat_session_data.get('mode') != '內部LLM':
                title = title.content

            # 過濾非中文字、英文、數字與空白
            title = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9 ]", "", title)
            # 限制長度
            if len(title) > 12:
                title = title[:12]

            self.chat_session_data['title'] = title
            return title

        except Exception as e:
            print(f"生成視窗標題文字時發生錯誤: {e}")
            return "未產生標題"

    def _title_prompt(self):
        """
        回傳自動產生標題的 Prompt 模板。
        """
        template = """
        根據以下提問(Q:)，列出1個關鍵字。請務必遵守以下規則：
        1.只能輸出關鍵字，不要有其他說明。
        2.輸出字數12字以內。
        ---
        Q: {query}        
        """
        return PromptTemplate(input_variables=["query"], template=template)

    def doc_summary(self, documents: List) -> List[str]:
        """
        使用 LLM 對文件內容進行摘要，回傳多個摘要文字清單。
        """
        try:
            # 取得 LLM API 物件（根據模式與選項）
            llm = LLMAPI.get_llm(self.mode, self.llm_option)
            # 取得摘要所需 prompt 模板
            prompt_template = self._doc_summary_prompt()

            summaries = []
            # 對每個 Document 個別產生摘要
            for doc in documents:
                # 將每份文件內容填入 prompt
                formatted_prompt = prompt_template.format(documents=doc.page_content)
                # 呼叫 LLM 取得摘要
                doc_summary = llm.invoke(formatted_prompt)
                # 如果是外部 LLM，取 .content
                if self.chat_session_data.get('mode') != '內部LLM':
                    doc_summary = doc_summary.content
                # 儲存摘要文字
                summaries.append(doc_summary.strip())
            return summaries
        except Exception as e:
            print(f"doc_summary 時發生錯誤: {e}")
            return []

    def _doc_summary_prompt(self):
        """
        生成摘要提示模板
        """
        template = """
        請根據以下文章進行摘要，文長約300字。
        ---
        文章: {documents}        
        """
        return PromptTemplate(input_variables=["documents"], template=template)
