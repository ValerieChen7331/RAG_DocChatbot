# 📚 文件檢索聊天機器人（RAG 知識庫整合）

## 🎯 專案目標
本專案是一個 Retrieval-Augmented Generation (RAG) 文件檢索聊天機器人。
能根據使用者上傳的資料(KEYPO 功能手冊文件)，進行檢索並提供準確回答。
- 當知識庫中找不到答案時，會明確回覆「不知道」，避免產生幻覺回覆。
- 影片連結: https://drive.google.com/file/d/1zonmd5DjB4uLdw6tDbIPVid_UjjnNGO2/view?usp=drive_link

---

## 🚀 執行步驟
### 1️⃣ 安裝依賴
```bash
pip install -r requirements.txt
```

### 2️⃣ 啟動本地 LLM (請先確認 Ollama 已在本地端啟動)
- A建立 `.env` 檔案（請參考格式 `.env`，和 `llm_api.py` 內含 Azure OpenAI API 設定或內部 LLM 設定）

### 3️⃣ 啟動 Streamlit 應用
```bash
streamlit run rag_app.py
```
### ⚠️ 注意事項
- 系統支援內部 LLM (透過 Ollama API) 及外部 Azure OpenAI 模型。
- 當知識庫中無法回答問題時，會清楚回答「不知道」，避免出現無依據回答。

---

## ✅ 主要功能

### 1️⃣ 文件上傳與處理  
支援 PDF / Markdown 檔案上傳，自動解析、摘要並嵌入向量資料庫。

### 2️⃣ 文件檢索聊天回覆（RAG 流程）  
根據使用者問題，檢索相關文件內容並搭配 LLM 回答問題。

### 3️⃣ 雙模式聊天支援  
可切換「個人 KM（RAG 模式）」或「一般 LLM 模式（直接回覆）」。

### 4️⃣ 聊天與上傳歷史記錄  
自動儲存聊天紀錄與文件摘要資訊，可於下次使用時調閱。

### 5️⃣ 安全機制  
如知識庫中無相關內容，回覆「不知道」避免幻覺回答。

---

## 🛠️ 技術架構
| 元件              | 技術                                         |
|-------------------|----------------------------------------------|
| 前端介面          | Streamlit                                   |
| 檔案處理          | langchain + PyPDFLoader + TextLoader        |
| 文件分段與嵌入向量 | langchain text splitter + ChromaDB           |
| 語言模型 (LLM)    | 支援 Azure OpenAI API 及本地 Ollama API（Llama3、Gemma2、Taide） |
| RAG 流程實作      | LangChain 的 Retriever + Stuff Documents Chain |
| 本地向量資料庫    | ChromaDB                                    |
| 後端儲存          | SQLite (User 記錄 + DevOps 記錄)            |

---

## 📂 專案結構

* 避免耦合: 📄 rag_app.py ➡ 📁 views ➡ 📁 controllers ➡ 📁 services ➡ 📁 models ➡ 📁 apis
```
RAG_DocChatbot/
 ├─ 📄 rag_app.py          # 主程式入口（Streamlit 啟動點）
 ├─ 📁 views               # 前端元件（Streamlit 畫面）
 │   ├─ main_page_content.py
 │   └─ main_page_sidebar.py
 ├─ 📁 controllers         # 控制層（初始化與 UI 控制）
 │   ├─ initialize.py
 │   └─ ui_controller.py
 ├─ 📁 services            # 服務層（文件處理、LLM 呼叫）
 │   ├─ document_services.py
 │   └─ llm_services.py
 ├─ 📁 models              # 資料模型、RAG 流程與資料庫互動
 │   ├─ database_base.py
 │   ├─ database_devOps.py
 │   ├─ database_userRecords.py
 │   ├─ document_model.py
 │   ├─ llm_model.py
 │   └─ llm_rag.py
 ├─ 📁 apis                # API 模組 (嵌入服務、LLM 服務、檔案路徑)
 │   ├─ embedding_api.py
 │   ├─ file_paths.py
 │   └─ llm_api.py
 ├─ 📁 data                # 動態產生的用戶資料與結果
 │   ├─ developer/
 │   └─ user/
 └─ 📄 .env                # 環境變數設定檔
```

---

## 📞 聯絡方式
如需協助或有問題，請提出 Issue 或透過電子郵件聯絡專案負責人。
valerie7331@gmail.com


