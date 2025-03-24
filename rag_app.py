import streamlit as st
from controllers.initialize import SessionInitializer
from views.main_page_sidebar import Sidebar
from views.main_page_content import MainContent
from services.llm_services import LLMService

if "is_initialized" not in st.session_state:
    st.session_state["is_initialized"] = False
    st.session_state["username"] = "Guest"
    st.session_state["name"] = "Guest"

class MainPage:
    def show(self):
        """顯示主頁面"""
        # 只在第一次初始化時執行
        if not st.session_state.get("is_initialized"):
            username = st.session_state.get("username")
            st.session_state["chat_session_data"] = SessionInitializer(username).initialize_session_state()
            print(st.session_state["chat_session_data"])
            st.session_state["is_initialized"] = True
            print("SessionInitializer(username)...")

        # 獲取會話數據
        chat_session_data = st.session_state.get("chat_session_data")
        # print('data: ', chat_session_data)

        # 顯示主頁面
        Sidebar(chat_session_data).display()
        MainContent(chat_session_data).display()

        # 處理聊天輸入
        if query := st.chat_input():
            st.chat_message("human").write(query)
            try:
                response, chat_session_data = LLMService(chat_session_data).query(query)
                st.chat_message("ai").write(response)
            except Exception as e:
                st.error(f"An error occurred while processing your request: {e}")


def main():
    """主函數"""
    MainPage().show()

if __name__ == "__main__":
    main()
