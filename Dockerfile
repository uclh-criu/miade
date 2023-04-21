FROM python:3.8.10

ADD ./.git/ /home/.git
ADD ./pyproject.toml /home/pyproject.toml
ADD ./src/ /home/src
ADD ./configs/ /home/configs
ADD ./notebooks/ /home/notebooks

WORKDIR /home/app

RUN python -m pip install --upgrade pip
RUN pip install /home/

RUN pip install https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl
RUN python -m spacy download en_core_web_md