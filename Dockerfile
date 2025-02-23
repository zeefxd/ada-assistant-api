FROM python:3.10-slim

WORKDIR /app

# Instaluje niezbędne biblioteki do prztwarzania audio
RUN apt-get update && apt-get install -y \
    gcc \
    portaudio19-dev \
    python3-pyaudio \
    && rm -rf /var/lib/apt/lists/*

# Dla lepszego cachowania przed instalacją kopiujemy plik requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopowianie kodu aplikacji
COPY ./api ./api
COPY ./test ./test
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]