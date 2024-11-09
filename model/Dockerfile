FROM python:3.10-slim AS trainer

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm

COPY . /app/

WORKDIR /app

RUN ["python", "trainer.py", "train", "15"]
