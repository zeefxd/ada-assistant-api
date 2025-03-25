from fastapi import APIRouter, UploadFile, File, HTTPException
from vosk import Model, KaldiRecognizer
import tempfile
import json
import wave
import os
import requests
import zipfile
import shutil
from pathlib import Path
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt")

router = APIRouter()

# Ścieżka do modelu https://alphacephei.com/vosk/models/vosk-model-small-pl-0.22.zip
models_dir = Path(__file__).parent.parent / "model" / "vosk"
model_name = "vosk-model-small-pl-0.22"
model_url = "https://alphacephei.com/vosk/models/vosk-model-small-pl-0.22.zip"

def ensure_vosk_model():
    """Pobiera model Vosk jeśli nie istnieje."""
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / model_name
    
    if not model_path.exists() or not any(model_path.iterdir()) if model_path.exists() else True:
        try:
            logger.info(f"Pobieranie modelu Vosk {model_name}...")
            
            model_path.mkdir(parents=True, exist_ok=True)
            
            zip_path = models_dir / f"{model_name}.zip"
            
            logger.info(f"Pobieranie z: {model_url}")
            
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            logger.info("Zapisywanie pliku ZIP...")
            with open(zip_path, 'wb') as file:
                shutil.copyfileobj(response.raw, file)
            
            logger.info("Rozpakowywanie modelu...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(models_dir)
            
            zip_path.unlink()
            
            logger.info(f"Model Vosk {model_name} pobrany pomyślnie!")
        except Exception as e:
            error_details = str(e) + "\n" + traceback.format_exc()
            logger.error(f"Błąd pobierania modelu: {error_details}")
            raise HTTPException(
                status_code=500, 
                detail=f"Nie można pobrać modelu Vosk: {error_details}"
            )
    
    return model_path

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(default=...), language: str = "pl-PL"):
    """Transkrybuje audio na tekst."""
    try:
        model_path = ensure_vosk_model()
        
        model = Model(str(model_path))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio.flush()
            
            wf = wave.open(temp_audio.name, "rb")
            
            recognizer = KaldiRecognizer(model, wf.getframerate())
            
            result = ""
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    part_result = json.loads(recognizer.Result())
                    if "text" in part_result:
                        result += part_result["text"] + " "
            
            final_result = json.loads(recognizer.FinalResult())
            if "text" in final_result:
                result += final_result["text"]
                
        return {"text": result.strip(), "language": language}
    except Exception as e:
        return {"error": str(e)}