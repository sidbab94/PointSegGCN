FROM ubuntu:18.04
FROM python:3.7-alpine

RUN pip3 install virtualenv

RUN virtualenv -p python3 pc_gcn
RUN source ./pc_gcn/bin/activate && pip install --upgrade pip

WORKDIR /src

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

RUN python infer.py

