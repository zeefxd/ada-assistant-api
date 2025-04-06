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
    """
    Data model for text-to-speech request.
    
    Args:
        text (str): Text to synthesize into speech
        language (str): Language code (default "pl")
        speed (float): Speech rate in range 0.5-2.0
        volume (float): Volume level in range 0.1-5.0
    """
    
    text: str
    language: str = "pl"
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech rate (0.5-2.0)")
    volume: float = Field(default=1.0, ge=0.1, le=5.0, description="Volume level (0.1-5.0)")
    
# https://github.com/rhasspy/piper
models_dir = Path(__file__).parent.parent / "model" / "piper"
model_name = "pl_PL-gosia-medium" 

bin_dir = Path(__file__).parent.parent / "bin"  
piper_exe = bin_dir / "piper.exe"  

def ensure_piper_model():
    """
    Downloads the Piper model if it doesn't exist in the file system.
    
    Args:
        None
        
    Returns:
        tuple: Tuple (model_path, config_path) containing paths to model files
        
    Raises:
        HTTPException: When model download fails
    """
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / f"{model_name}.onnx"
    config_path = models_dir / f"{model_name}.onnx.json"
    
    if not model_path.exists() or not config_path.exists():
        try:
            print(f"Downloading Piper model {model_name}...")
            
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
            
            model_url = f"{base_url}/pl/pl_PL/gosia/medium/{model_name}.onnx"
            config_url = f"{base_url}/pl/pl_PL/gosia/medium/{model_name}.onnx.json"
            
            print(f"Downloading model from: {model_url}")
            print(f"Downloading config from: {config_url}")
            
            for url, path in [(model_url, model_path), (config_url, config_path)]:
                print(f"Downloading {url}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(path, 'wb') as file:
                    shutil.copyfileobj(response.raw, file)
            
            print(f"Piper model {model_name} downloaded successfully!")
        except Exception as e:
            error_details = str(e) + "\n" + traceback.format_exc()
            raise HTTPException(
                status_code=500, 
                detail=f"Cannot download Piper model: {error_details}"
            )
    
    return model_path, config_path

@router.post("/synthesize")
async def synthesize_speech(request: TextToSpeech):
    """
    Generates speech from text using Piper TTS.
    
    Args:
        request (TextToSpeech): Object containing text and synthesis parameters
        
    Returns:
        FileResponse: WAV audio file containing synthesized speech
        
    Raises:
        HTTPException: When speech generation fails
    """
    
    try:
        model_path, config_path = ensure_piper_model()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            output_path = temp_audio.name
        
        logger.info(f"Generating speech for text: '{request.text[:50]}...'")
        start_time = time.time()
        
        try:
            # Call piper.exe with input text via stdin
            cmd = [
                str(piper_exe),
                "--model", str(model_path),
                "--config", str(config_path),
                "--output_file", output_path,
                "--speaker", "0",       
                "--speed", str(request.speed),    
                "--volume", str(request.volume)   
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            
            process = subprocess.run(
                cmd, 
                input=request.text,
                text=True,           
                check=True,          
                timeout=30,          
                capture_output=True,
                encoding='utf-8'
            )
                
            logger.info(f"Generated speech via Piper in {time.time() - start_time:.2f}s")
            
            if process.stdout:
                logger.info(f"Piper stdout: {process.stdout}")
            
            if os.path.getsize(output_path) < 100:
                raise Exception("Generated audio file is too small or empty")
                
        except subprocess.TimeoutExpired as e:
            logger.error(f"Timeout during speech generation: {str(e)}")
            raise Exception(f"Execution timeout (30s) while generating speech")
            
        except Exception as e:
            logger.error(f"Error during speech generation: {str(e)}")
            raise Exception(f"Error during speech generation: {str(e)}")
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
    except Exception as e:
        error_details = str(e) + "\n" + traceback.format_exc()
        logger.error(f"Error: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error: {error_details}")
