# app/Dockerfile

FROM python:3.8.10

RUN python -m pip install --upgrade pip

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

ADD . /home/miade

RUN pip install '/home/miade[dashboard]'

WORKDIR /home/miade/streamlit_app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]