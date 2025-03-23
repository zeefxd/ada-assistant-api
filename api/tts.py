from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tempfile
import os
import subprocess
from pathlib import Path
import requests
import shutil
import traceback

router = APIRouter()

class TextToSpeech(BaseModel):
    text: str
    language: str = "pl"

# https://github.com/rhasspy/piper
models_dir = Path(__file__).parent.parent / "model" / "piper"
model_name = "pl_PL-gosia-medium"  # Specific file name prefix

def ensure_piper_model():
    # Create models directory if it doesn't exist
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Define model paths with specific file names
    model_path = models_dir / f"{model_name}.onnx"
    config_path = models_dir / f"{model_name}.onnx.json"  # Note the .onnx.json extension
    
    # Check if model exists, if not download it
    if not model_path.exists() or not config_path.exists():
        try:
            print(f"Pobieranie modelu Piper {model_name}...")
            
            # Create parent directories if they don't exist
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Base URL for the main branch
            base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
            
            # Correct path to the specific files
            model_url = f"{base_url}/pl/pl_PL/gosia/medium/{model_name}.onnx"
            config_url = f"{base_url}/pl/pl_PL/gosia/medium/{model_name}.onnx.json"
            
            print(f"Downloading model from: {model_url}")
            print(f"Downloading config from: {config_url}")
            
            # Download files
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

@router.post("/synthesize")
async def synthesize_speech(request: TextToSpeech):
    try:
        model_path, config_path = ensure_piper_model()
        
        # Create temporary files for output and input text
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            output_path = temp_audio.name
        
        with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as temp_text:
            temp_text.write(request.text)
            text_path = temp_text.name
        
        print("Generowanie mowy...")
        
        try:
            # Try using the Python API first
            try:
                from piper import PiperVoice
                
                voice = PiperVoice.load(str(model_path), config_path=str(config_path))
                with open(output_path, "wb") as audio_file:
                    voice.synthesize_to_file(request.text, audio_file)
                print("Wygenerowano mowę przez API Pythona.")
                
            except (ImportError, ModuleNotFoundError):
                # Fall back to command-line if Python API is not available
                print("API Pythona niedostępne, użycie CLI piper...")
                subprocess.run([
                    "piper",
                    "--model", str(model_path),
                    "--config", str(config_path),
                    "--output_file", output_path,
                    "--input_file", text_path
                ], check=True)
                print("Wygenerowano mowę przez CLI piper.")
                
        except Exception as e:
            raise Exception(f"Błąd podczas generowania mowy: {str(e)}")
        
        # Clean up the temporary text file
        os.unlink(text_path)
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
    except Exception as e:
        error_details = str(e) + "\n" + traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Błąd: {error_details}")