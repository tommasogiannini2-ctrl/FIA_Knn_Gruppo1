FROM python:3.11-slim
WORKDIR /app
# Installiamo le dipendenze di sistema minime se necessarie (opzionale)
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev && rm -rf /var/lib/apt/lists/*

COPY codice/requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY ./codice /app
CMD ["python", "main.py"]
