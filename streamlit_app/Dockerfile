# app/Dockerfile

FROM python:3.8.10

ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

RUN python -m pip install --upgrade pip

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

ADD . /home/miade

RUN pip install https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/9633b53e7f9488408eae5dfff51b831b3b787657/en_core_med7_lg-any-py3-none-any.whl
RUN pip install /home/miade

WORKDIR /home/miade/streamlit_app

RUN pip install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]