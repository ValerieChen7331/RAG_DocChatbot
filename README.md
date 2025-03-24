# 📚 文件檢索聊天機器人（RAG 知識庫整合）

## 🎯 專案目標
本專案是一個 Retrieval-Augmented Generation (RAG) 文件檢索聊天機器人，能根據上傳的 KEYPO 功能手冊文件進行檢索並提供準確回答。
當知識庫中找不到答案時，會明確回覆「不知道」，避免產生幻覺回覆。

影片連結: https://drive.google.com/file/d/1zonmd5DjB4uLdw6tDbIPVid_UjjnNGO2/view?usp=drive_link
---

## ✅ 主要功能

1️⃣ 文件上傳與處理  
支援 PDF / Markdown 檔案上傳，自動解析、摘要並嵌入向量資料庫。

2️⃣ 文件檢索聊天回覆（RAG 流程）  
根據使用者問題，檢索相關文件內容並搭配 LLM 回答問題。

3️⃣ 雙模式聊天支援  
可切換「個人 KM（RAG 模式）」或「一般 LLM 模式（直接回覆）」。

4️⃣ 聊天與上傳歷史記錄  
自動儲存聊天紀錄與文件摘要資訊，可於下次使用時調閱。

5️⃣ 安全機制  
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

## 🏗️ 專案結構
```
📦 專案根目錄
 ├─ 📄 rag_app.py          # 主執行檔（Streamlit 入口）
 ├─ 📄 requirements.txt    # 相依套件清單
 ├─ 📁 views               # Streamlit 前端畫面元件
 ├─ 📁 apis                # API 模組 (LLM、嵌入模型、檔案路徑)
 ├─ 📁 controllers         # 初始化及 UI 控制器
 ├─ 📁 services            # 文件及 LLM 服務層
 ├─ 📁 models              # 資料模型、RAG 流程、資料庫模型
 └─ 📁 data                # 動態建立的用戶資料夾
```

---

## 🚀 執行步驟
1. 安裝依賴
```
pip install -r requirements.txt
```

2. 建立 `.env` 檔案（請參考範例格式 .env，內含 Azure OpenAI API 設定或內部 LLM 設定）

3. 執行應用
```
streamlit run rag_app.py
```

---

## ⚠️ 注意事項
- 系統支援內部 LLM (透過 Ollama API) 及外部 Azure OpenAI 模型。
- 當知識庫中無法回答問題時，會清楚回答「不知道」，避免出現無依據回答。

---

## 📞 聯絡方式
如需協助或有問題，請提出 Issue 或透過電子郵件聯絡專案負責人。
valerie7331@gmail.com

---

> 專案用途：KEYPO 功能手冊文件檢索智能問答機器人開發。
> 技術：Python / Streamlit / LangChain / ChromaDB / Azure OpenAI / Ollama

=======
# RAG_DocChatbot
Document Search Chatbot (with RAG integration)
