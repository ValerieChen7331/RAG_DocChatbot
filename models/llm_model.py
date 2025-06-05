# llm_model.py

from apis.llm_api import LLMAPI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
import re
from typing import List, Dict

# 預設是否開啟 query_llm_direct 的聊天記憶功能
DEFAULT_USE_CHAT_HISTORY: bool = False


class LLMModel:
    def __init__(self, chat_session_data):
        """
        初始化 LLMModel 物件，儲存使用者對話 session 設定。
        """
        self.chat_session_data = chat_session_data
        self.mode = chat_session_data.get('mode')
        self.llm_option = chat_session_data.get('llm_option')

    def query_llm_direct(self, query: str) -> str:
        """
        查詢 LLM 回應使用者輸入的問題，
        可依設定決定是否使用歷史對話作為記憶。
        """
        # 根據模式與模型選項取得 LLM 物件
        llm = LLMAPI.get_llm(self.mode, self.llm_option)

        # 🔁 是否使用歷史對話記憶（優先取 session 中設定，其次取預設常數）
        use_memory = self.chat_session_data.get("use_memory", DEFAULT_USE_CHAT_HISTORY)

        if use_memory:
            # ✅ 開啟聊天記憶功能
            active_window_index = self.chat_session_data.get('active_window_index', 0)
            memory_key = f'conversation_memory_{active_window_index}'

            # 如果尚未建立對應記憶體則初始化
            if memory_key not in self.chat_session_data:
                self.chat_session_data[memory_key] = ConversationBufferMemory(
                    memory_key="history", input_key="input"
                )

            # 將過往對話紀錄載入 chat memory
            chat_history_data = self.chat_session_data.get('chat_history', [])
            if chat_history_data:
                chat_history = ChatMessageHistory()
                for record in chat_history_data:
                    chat_history.add_user_message(record['user_query'])
                    chat_history.add_ai_message(record['ai_response'])
                self.chat_session_data[memory_key].chat_memory = chat_history

            # 建立 Prompt 模板（含 history）
            init_prompt = """
            你是一位親切、專業且反應快速的問答助理，請使用繁體中文（台灣用語）回覆。 
            以下是對話紀錄：  
            {history}
            請根據上述內容，針對以下問題，提供簡單明瞭、正確實用的回答，必要時可適當補充說明，但避免過長或贅述：  
            {input}
            """

            # 建立 ConversationChain 並執行查詢
            conversation_chain = ConversationChain(
                llm=llm,
                memory=self.chat_session_data[memory_key],
                prompt=ChatPromptTemplate.from_template(init_prompt)
            )
            result = conversation_chain.invoke(input=query)
            response = result.get('response', '')

        else:
            # 🚫 不使用聊天記憶，單輪問答模式
            prompt_template = PromptTemplate.from_template("""
            你是一位親切、專業且反應快速的問答助理，請使用繁體中文（台灣用語）回覆。  
            以下是使用者的問題：  
            {input}  
            請簡明且正確地回答。
            """)
            formatted_prompt = prompt_template.format(input=query)
            result = llm.invoke(formatted_prompt)

            # 處理不同 LLM 回傳格式
            response = result if isinstance(result, str) else getattr(result, 'content', '')

        return response

    def set_window_title(self, query: str) -> str:
        """
        使用 LLM 根據使用者問題，自動產生簡短的視窗標題。
        """
        try:
            llm = LLMAPI.get_llm(self.mode, self.llm_option)
            prompt_template = self._title_prompt()
            formatted_prompt = prompt_template.format(query=query)
            title = llm.invoke(formatted_prompt)

            # 處理外部 LLM 回傳格式
            if self.chat_session_data.get('mode') != '內部LLM':
                title = title.content

            # 過濾特殊符號，只保留中英數與空白
            title = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9 ]", "", title)

            # 最多取前 12 字
            if len(title) > 12:
                title = title[:12]

            # 存入 session
            self.chat_session_data['title'] = title
            return title

        except Exception as e:
            print(f"生成視窗標題文字時發生錯誤: {e}")
            return "未產生標題"

    def _title_prompt(self):
        """
        回傳視窗標題生成的 Prompt 模板。
        """
        template = """
        根據以下提問(Q:)，列出1個關鍵字作為標題。請遵守以下規則：
        1. 只能輸出標題關鍵字，不可有多餘說明。
        2. 標題請限制在12字以內。
        ---
        Q: {query}        
        """
        return PromptTemplate(input_variables=["query"], template=template)

    def doc_summary(self, doc) -> str:
        """
        對單份文件內容進行摘要，回傳一段約300字的摘要。
        """
        try:
            llm = LLMAPI.get_llm(self.mode, self.llm_option)
            prompt_template = self._doc_summary_prompt()
            formatted_prompt = prompt_template.format(documents=doc.page_content)
            doc_summary = llm.invoke(formatted_prompt)

            # 處理外部 LLM 格式
            if self.chat_session_data.get('mode') != '內部LLM':
                doc_summary = doc_summary.content

            return doc_summary.strip()

        except Exception as e:
            print(f"doc_summary 時發生錯誤: {e}")
            return ""

    def _doc_summary_prompt(self):
        """
        回傳摘要任務的 Prompt 模板。
        """
        template = """
        請根據下方文章內容撰寫摘要，建議長度約為 300 字。
        摘要內容須包含：
        1. 文件的主要功能與核心目的
        2. 出現的專有名詞與簡要解釋
        3. 各章節或段落的結構與重點概覽
        
        請避免添加無根據的推論，僅根據提供的內容進行歸納。
        ---
        文章內容：
        {documents}      
        """
        return PromptTemplate(input_variables=["documents"], template=template)
