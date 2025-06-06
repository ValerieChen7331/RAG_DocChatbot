FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# 避免 tzdata 互動畫面
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

# 安裝基礎套件（不含 tzdata，後面單獨處理）
RUN apt update && apt install -y \
    build-essential wget curl git unzip software-properties-common \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libffi-dev liblzma-dev libncursesw5-dev pkg-config

# 安裝 tzdata 並設定時區
RUN apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# 安裝 SQLite 3.45（符合 Chroma 要求 >= 3.35）
WORKDIR /tmp
RUN wget https://www.sqlite.org/2024/sqlite-autoconf-3450000.tar.gz && \
    tar xzf sqlite-autoconf-3450000.tar.gz && \
    cd sqlite-autoconf-3450000 && \
    ./configure --prefix=/usr/local && make -j && make install && ldconfig

ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
ENV PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"

# 安裝 Python 3.11.9
WORKDIR /usr/src
RUN wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz && \
    tar xzf Python-3.11.9.tgz && cd Python-3.11.9 && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make -j && make altinstall

# 指定預設 python/pip 版本為 3.11
RUN ln -sf /usr/local/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3.11 /usr/bin/pip

# 設定工作目錄
WORKDIR /workspace

# 複製 requirements.txt 並安裝套件
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 複製所有專案檔案
COPY . .

# 預設執行指令：啟動 Streamlit App
CMD ["streamlit", "run", "rag_engine.py", "--server.port=8501", "--server.address=0.0.0.0"]
