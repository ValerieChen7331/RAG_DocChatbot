# Dockerfile
FROM python:3.10-slim

# 設定時區及環境參數
ENV TZ=Asia/Taipei
ENV PYTHONUNBUFFERED=1

# 設定工作目錄
WORKDIR /app

# 複製需求檔案並安裝相依套件
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案所有檔案
COPY . .

# 開放 Streamlit 預設埠
EXPOSE 8501

# 啟動指令
CMD ["streamlit", "run", "rag_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
