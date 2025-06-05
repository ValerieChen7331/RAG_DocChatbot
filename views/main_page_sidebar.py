import streamlit as st
from controllers.ui_controller import UIController
from models.database_userRecords import UserRecordsDB

class Sidebar:
    def __init__(self, chat_session_data):
        """初始化側邊欄物件"""
        self.chat_session_data = chat_session_data
        self._controller = None         # 延遲初始化的 UIController

        # 助理類型選項
        self.agent_options = ['一般助理', '個人KM']

        # 內部或外部 LLM
        self.mode_options = ['內部LLM', '外部LLM']
        # 內部 LLM 選項"
        self.llm_options_internal = ["Gemma3_27b", "Gemma2_27b", "QWQ_32b", "Phi4_14b", "Taiwan_LLaMA3_8b_Instruct", "Mistral_7b_Instruct"]
        # 外部 LLM 選項
        self.llm_options_external = ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-35-turbo"]
        # 內部嵌入模型選項
        self.embedding_options_internal = ["bge-m3"]
        # 外部嵌入模型選項
        self.embedding_options_external = ["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"]

    @property
    def controller(self):
        """延遲初始化 UIController"""
        if self._controller is None:
            self._controller = UIController(self.chat_session_data)
        return self._controller

    def display(self):
        """顯示側邊欄"""
        with st.sidebar:
            self._set_sidebar_button_style()   # 設定側邊欄按鈕樣式
            self.new_chat_button()             # 顯示新聊天按鈕
            self.agent_selection()             # 顯示助理類型選擇
            self.llm_selection()               # 顯示 LLM 模式選項
            if self.chat_session_data.get('agent') in ['個人KM']:
                self.embedding_selection()     # 僅當助理類型為 '個人KM' 時顯示嵌入模型選項
            self.chat_history_buttons()        # 顯示聊天記錄按鈕

    def _set_sidebar_button_style(self):
        """設定側邊欄按鈕的樣式"""
        st.markdown("""
            <style>
                div.stButton > button {
                    background-color: transparent;  /* 設定按鈕背景為透明 */
                    border: 1px solid #ccc;         /* 設定按鈕邊框顏色 */
                    border-radius: 5px;             /* 設定按鈕邊框圓角 */
                    font-weight: bold;              /* 設定按鈕字體加粗 */
                    margin: -5px 0;                 /* 設定按鈕上下邊距 */
                    padding: 10px 20px;             /* 設定按鈕內距 */
                    width: 100%;                    /* 設定按鈕寬度為 100% */
                    display: flex;                  /* 使用 flex 布局 */
                    font-size: 30px !important;     /* 設定按鈕字體大小 */
                    justify-content: center;        /* 設定按鈕內容水平居中 */
                    align-items: center;            /* 設定按鈕內容垂直居中 */
                    text-align: center;             /* 設定按鈕文字居中對齊 */
                    white-space: nowrap;            /* 設定文字不換行 */
                    overflow: hidden;               /* 隱藏超出範圍的內容 */
                    text-overflow: ellipsis;        /* 使用省略號表示超出範圍的文字 */
                    line-height: 1.2;               /* 設定行高，讓文字保持在一行內 */
                    height: 1.2em;                  /* 設定按鈕高度僅能容納一行文字 */
                }
                div.stButton > button:hover {
                    background-color: #e0e0e0;  /* 設定按鈕懸停時的背景顏色 */
                }
            </style>
        """, unsafe_allow_html=True)

    def new_chat_button(self):
        """顯示新聊天按鈕"""
        if st.button("New Chat"):
            self.chat_session_data['agent'] = '一般助理'  # 預設選擇一般助理
            self.controller.new_chat()

    def agent_selection(self):
        """顯示助理類型選擇與資料庫選擇"""
        st.title("Assistants")
        current_agent = self.chat_session_data.get('agent', self.agent_options[0])  # 取得當前助理類型，若無則使用預設值
        selected_agent = st.radio("請選擇助理種類型:", self.agent_options, index=self.agent_options.index(current_agent))

        # 助理類型改變時開啟新聊天
        if selected_agent != self.chat_session_data.get('agent'):
            self.controller.new_chat()
            self.chat_session_data['agent'] = selected_agent  # 更新助理類型

    def llm_selection(self):
        """顯示 LLM 模式選項"""
        st.title("LLM")
        current_mode = self.chat_session_data.get('mode', self.mode_options[0])  # 取得當前 LLM 模式，若無則使用預設值
        selected_mode = st.radio("LLM類型:", self.mode_options, index=self.mode_options.index(current_mode))
        self.chat_session_data['mode'] = selected_mode  # 更新選擇的 LLM 模式

        llm_options = self.llm_options_internal if selected_mode == '內部LLM' else self.llm_options_external
        self.chat_session_data['llm_option'] = llm_options[0]  # 更新選擇的 llm_option 模式
        self._create_selectbox('選擇 LLM：', 'llm_option', llm_options)  # 顯示對應的 LLM 選項

    def embedding_selection(self):
        """顯示嵌入模型選項"""
        st.title("Embedding")
        if self.chat_session_data.get('mode') == '內部LLM':
            embedding_options = self.embedding_options_internal
        else:
            embedding_options = self.embedding_options_external
        self.chat_session_data['embedding'] = embedding_options[0]  # 更新選擇的 embedding 模式
        self._create_selectbox('選擇嵌入模型：', 'embedding', embedding_options)  # 顯示對應的嵌入模型選項

    def chat_history_buttons(self):
        """顯示側邊欄中的聊天記錄"""
        st.title("聊天記錄")
        total_windows = self.chat_session_data.get('num_chat_windows', 0)  # 取得聊天窗口數量，預設為0
        for index in range(total_windows):
            chat_title = self.controller.get_title(index)  # 取得聊天窗口的標題
            chat_window, delete_button = st.columns([4, 1])  # 設置標題和刪除按鈕的佈局

            if chat_window.button(chat_title, key=f'chat_window_select_{index}'):
                self.chat_session_data['active_window_index'] = index  # 更新目前活動的聊天窗口索引
                self._update_window_setup()

            if delete_button.button("X", key=f'chat_delete_{index}'):
                self.controller.delete_chat_history_and_update_indexes(index)
                self._update_active_window_index(index, total_windows)
                self._update_window_setup()

    def _update_active_window_index(self, deleted_index, total_windows):
        """更新目前活動窗口的索引"""
        active_window_index = self.chat_session_data.get('active_window_index', 0)
        if active_window_index > deleted_index:
            self.chat_session_data['active_window_index'] -= 1  # 刪除的窗口在當前活動窗口之前，將索引減1
        elif active_window_index == deleted_index:
            self.chat_session_data['active_window_index'] = max(0, total_windows - 2)  # 刪除的是當前窗口時，設為最後一個窗口

    def _update_window_setup(self):
        """更新窗口設定並重新執行應用程式"""
        index = self.chat_session_data.get('active_window_index', 0)
        username = self.chat_session_data.get('username')
        userRecords_db = UserRecordsDB(username)
        self.chat_session_data = userRecords_db.get_active_window_setup(index, self.chat_session_data)  # 取得當前活動窗口的設定
        if self.chat_session_data == {}:
            self.chat_session_data = self._controller.reset_session_state_to_defaults()
        st.rerun()  # 重新執行應用程式

    def _create_selectbox(self, label, key, options):
        """建立選擇框並更新 session state"""
        selected_value = self.chat_session_data.get(key, options[0])
        selected_index = options.index(selected_value)
        self.chat_session_data[key] = st.selectbox(label, options, index=selected_index)  # 根據選擇框選擇的值更新 session state
