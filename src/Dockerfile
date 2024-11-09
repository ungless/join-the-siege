FROM python:3.11-slim AS api

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt
RUN apt-get update \
  && apt-get -y install tesseract-ocr

EXPOSE 5000

#CMD ["python", "-m", "app"]
CMD ["flask", "--app", "app.py", "--debug", "run", "--host=0.0.0.0", "--port=5000"]

