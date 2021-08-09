FROM ubuntu:20.04
FROM python:3.7.11-slim-buster

RUN python3 -m venv /opt/pcgcn

COPY requirements.txt .

RUN . /opt/pcgcn/bin/activate && pip install -r requirements.txt

COPY . .

CMD . /opt/pcgcn/bin/activate && exec python infer.py
