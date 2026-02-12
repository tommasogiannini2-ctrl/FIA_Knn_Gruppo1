FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev && rm -rf /var/lib/apt/lists/*

COPY codice/requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY ./codice /app

RUN mkdir -p /app/dati /app/risultati && chmod -R 777 /app/risultati

CMD ["python", "main.py"]
