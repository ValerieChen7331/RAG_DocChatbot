import time
import streamlit as st
import pandas as pd
from controllers.initialize import SessionInitializer
from views.main_page_sidebar import Sidebar
from views.main_page_content import MainContent
from services.llm_services import LLMService

# 預設 session_state 初始化
if "is_initialized" not in st.session_state:
    st.session_state["is_initialized"] = False
    st.session_state["username"] = "Guest"
    st.session_state["name"] = "Guest"

class MainPage:
    def show(self):
        """
        顯示主頁面並處理使用者互動。
        """

        # 第一次執行時進行 session_state 初始化
        if not st.session_state.get("is_initialized"):
            username = st.session_state.get("username")
            # 根據使用者名稱初始化對話 session 狀態
            st.session_state["chat_session_data"] = SessionInitializer(username).initialize_session_state()
            print(st.session_state["chat_session_data"])
            st.session_state["is_initialized"] = True
            print("SessionInitializer(username)...")

        # 取得當前對話 session 狀態
        chat_session_data = st.session_state.get("chat_session_data")

        # 顯示側邊欄與主內容頁面
        Sidebar(chat_session_data).display()
        MainContent(chat_session_data).display()

        # 處理輸入文字（chat_input）
        if query := st.chat_input():
            st.chat_message("human").write(query)  # 將使用者問題呈現在對話區
            try:
                # 呼叫 LLMService 取得回覆
                response, chat_session_data = LLMService(chat_session_data).query(query)
                # 將 AI 回應寫到 chat_message
                st.chat_message("ai").write(response)
            except Exception as e:
                st.error(f"處理請求時發生錯誤: {e}")

        if chat_session_data.get('agent') in ['個人KM']:
            # ✅ CSV 批量測試按鈕
            if st.button("批次測試: QAData_1819.csv"):
                self._run_csv_queries(chat_session_data)

    def _run_csv_queries(self, chat_session_data, qa_path="mockdata/QAData_1819.csv"):
        """
        從指定的 CSV 中依序讀取問題，並呼叫 LLM 回答顯示在畫面上
        :param chat_session_data: Streamlit session 中的對話狀態
        :param qa_path: CSV 檔案路徑，預設為 QAData_1819.csv
        """
        try:
            df = pd.read_csv(qa_path)
        except Exception as e:
            st.error(f"❌ 無法讀取 CSV 檔案：{e}")
            return

        for idx, row in df.iterrows():
            query = row['Question']
            st.chat_message("human").write(query)
            try:
                response, chat_session_data = LLMService(chat_session_data).query(query)
                st.chat_message("ai").write(response)

            except Exception as e:
                st.chat_message("ai").write(f"❌ 查詢錯誤：{e}")


def main():
    """
    程式進入點，啟動主頁面功能。
    """
    MainPage().show()


if __name__ == "__main__":
    main()
