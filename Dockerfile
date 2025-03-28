# 使用正確的 Python 基底映像
FROM python:3.11.11

# 設定時區與 Python 輸出行為
ENV TZ=Asia/Taipei
ENV PYTHONUNBUFFERED=1

# 設定工作目錄
WORKDIR /rag_app

# 複製需求檔案並安裝依賴
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 複製整個專案進容器
COPY . .

# 開放 Streamlit 預設埠
EXPOSE 8501

# 啟動 Streamlit 應用
CMD ["streamlit", "run", "rag_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

