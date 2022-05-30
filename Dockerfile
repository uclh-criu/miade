FROM python:3.8

ADD ./src/ /home/packages/miade/src

WORKDIR /home/app

RUN pip install /home/packages/miade/src
