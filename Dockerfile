FROM python:3.8

ADD ./src/ /home/src

WORKDIR /home/app

RUN pip install ../src
