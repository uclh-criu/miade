FROM python:3.8

ADD ./src/ /home/src
ADD ./configs/ /home/configs
ADD ./notebooks/ /home/notebooks

WORKDIR /home/app

RUN pip install https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl
RUN pip install /home/src

RUN python -m spacy download en_core_web_md