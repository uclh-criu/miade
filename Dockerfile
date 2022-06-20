FROM python:3.8

ADD ./src/ /home/src
ADD ./configs/ /home/configs

WORKDIR /home/app

RUN pip install /home/src