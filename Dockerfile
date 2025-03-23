FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    portaudio19-dev \
    python3-pyaudio \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Dla lepszego cachowania przed instalacją trzeba skopiować ten plik
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p ./api/vosk ./model

RUN wget https://alphacephei.com/vosk/models/vosk-model-small-pl-0.22.zip -O vosk-model.zip && \
    unzip vosk-model.zip -d temp_vosk && \
    mv temp_vosk/vosk-model-small-pl-0.22/* ./api/vosk/ && \
    rm -rf vosk-model.zip temp_vosk

ENV TTS_HOME=/app/model

COPY ./api ./api
COPY ./test ./test
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]