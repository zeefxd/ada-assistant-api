from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import tempfile
import os
import subprocess
from pathlib import Path
import requests
import shutil
import traceback
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tts")

router = APIRouter()

class TextToSpeech(BaseModel):
    text: str
    language: str = "pl"
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Prędkość mowy (0.5-2.0)")
    volume: float = Field(default=1.0, ge=0.1, le=5.0, description="Głośność (0.1-5.0)")
    
# https://github.com/rhasspy/piper
models_dir = Path(__file__).parent.parent / "model" / "piper"
model_name = "pl_PL-gosia-medium" 

bin_dir = Path(__file__).parent.parent / "bin"  
piper_exe = bin_dir / "piper.exe"  

def ensure_piper_model():
    """Pobiera model Piper jeśli nie istnieje."""
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / f"{model_name}.onnx"
    config_path = models_dir / f"{model_name}.onnx.json"
    
    if not model_path.exists() or not config_path.exists():
        try:
            print(f"Pobieranie modelu Piper {model_name}...")
            
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
            
            model_url = f"{base_url}/pl/pl_PL/gosia/medium/{model_name}.onnx"
            config_url = f"{base_url}/pl/pl_PL/gosia/medium/{model_name}.onnx.json"
            
            print(f"Pobieranie modelu z: {model_url}")
            print(f"Pobieranie conifgu z: {config_url}")
            
            for url, path in [(model_url, model_path), (config_url, config_path)]:
                print(f"Pobieranie {url}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(path, 'wb') as file:
                    shutil.copyfileobj(response.raw, file)
            
            print(f"Model Piper {model_name} pobrany pomyślnie!")
        except Exception as e:
            error_details = str(e) + "\n" + traceback.format_exc()
            raise HTTPException(
                status_code=500, 
                detail=f"Nie można pobrać modelu Piper: {error_details}"
            )
    
    return model_path, config_path

def normalize_polish_text(text):
    """Optymalizuje tekst dla Piper."""
    replacements = {
        'ą': 'om',
        # Jakbyśmy się w przyszłości spotkali z innymi problemami z polskimi znakami to dodoajemy je tutaj
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    return text

@router.post("/synthesize")
async def synthesize_speech(request: TextToSpeech):
    """Generuje mowę z tekstu przy użyciu Piper."""
    try:
        model_path, config_path = ensure_piper_model()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            output_path = temp_audio.name
        
        normalized_text = normalize_polish_text(request.text)
        
        logger.info(f"Generowanie mowy dla tekstu: '{normalized_text[:50]}...'")
        start_time = time.time()
        
        try:
            # Wywołanie piper.exe z przekazaniem tekstu przez stdin
            cmd = [
                str(piper_exe),
                "--model", str(model_path),
                "--config", str(config_path),
                "--output_file", output_path,
                "--speaker", "0",       
                "--speed", str(request.speed),    
                "--volume", str(request.volume)   
            ]
            
            logger.info(f"Uruchamianie: {' '.join(cmd)}")
            
            process = subprocess.run(
                cmd, 
                input=normalized_text,
                text=True,           
                check=True,          
                timeout=30,          
                capture_output=True  
            )
                
            logger.info(f"Wygenerowano mowę przez Piper w {time.time() - start_time:.2f}s")
            
            if process.stdout:
                logger.info(f"Piper stdout: {process.stdout}")
            
            if os.path.getsize(output_path) < 100:
                raise Exception("Wygenerowany plik audio jest zbyt mały lub pusty")
                
        except subprocess.TimeoutExpired as e:
            logger.error(f"Timeout podczas generowania mowy: {str(e)}")
            raise Exception(f"Przekroczony czas oczekiwania (30s) na wygenerowanie mowy")
            
        except Exception as e:
            logger.error(f"Błąd podczas generowania mowy: {str(e)}")
            raise Exception(f"Błąd podczas generowania mowy: {str(e)}")
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
    except Exception as e:
        error_details = str(e) + "\n" + traceback.format_exc()
        logger.error(f"Błąd: {error_details}")
        raise HTTPException(status_code=500, detail=f"Błąd: {error_details}")