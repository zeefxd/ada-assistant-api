import tempfile
import os
import traceback
import logging
import time
import torch

import TTS.utils.io as tts_io

from TTS.api import TTS
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

original_load_fsspec = tts_io.load_fsspec

def patched_load_fsspec(filepath, **kwargs):
    """
    Patched version of load_fsspec to ensure weights_only parameter is included.
    
    Args:
        filepath (str): Path to the file
        **kwargs: Additional arguments
    
    Returns:
        Object loaded from the file
    """
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return original_load_fsspec(filepath, **kwargs)

tts_io.load_fsspec = patched_load_fsspec

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    torch.serialization.add_safe_globals([XttsConfig])
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("polish_tts")

router = APIRouter()

class PolishTextToSpeech(BaseModel):
    """
    Data model for text-to-speech request.
    
    Args:
        text (str): Text to synthesize into speech
    """
    text: str

models_dir = Path(__file__).parent.parent / "model" / "xtts"
models_dir.mkdir(parents=True, exist_ok=True)

assets_dir = Path(__file__).parent.parent / "assets"
assets_dir.mkdir(parents=True, exist_ok=True)

POLISH_FEMALE_VOICE = assets_dir / "polish_female_voice.wav"

tts_model = None

def load_xtts_model():
    """
    Loads the XTTS-v2 model if not already loaded.
    
    Args:
        None
    
    Returns:
        TTS: Loaded TTS model instance
    
    Raises:
        HTTPException: If model loading fails
    """
    global tts_model
    
    if tts_model is None:
        try:
            logger.info("Loading XTTS-v2 model...")
            start_time = time.time()
            
            tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
            
            logger.info(f"XTTS-v2 model loaded in {time.time() - start_time:.2f}s")
            
            ensure_voice_sample()
            
        except Exception as e:
            error_details = str(e) + "\n" + traceback.format_exc()
            logger.error(f"Failed to load XTTS model: {error_details}")
            raise HTTPException(
                status_code=500, 
                detail=f"Cannot load XTTS model: {str(e)}"
            )
    
    return tts_model

def ensure_voice_sample():
    """
    Makes sure we have a voice sample.
    
    Args:
        None
    
    Returns:
        None
    
    Raises:
        FileNotFoundError: If voice sample file is missing
    """
    if not POLISH_FEMALE_VOICE.exists():
        logger.info("Voice sample missing. Please add a female voice file.")
        
        with open(POLISH_FEMALE_VOICE.with_suffix('.txt'), 'w', encoding='utf-8') as f:
            f.write("Place a WAV file (44.1kHz, min. 3 seconds) with a Polish female voice sample here.\n")
            f.write("File name should be: polish_female_voice.wav")
            
        raise FileNotFoundError(
            f"Missing voice sample file: {POLISH_FEMALE_VOICE}. "
            "Please add a WAV file with a female Polish voice."
        )


@router.post("/synthesize", response_model=FileResponse)
async def generate_polish_speech(request: PolishTextToSpeech):
    """
    Generates speech.
    
    Args:
        request (PolishTextToSpeech): Request object containing the text to synthesize
    
    Returns:
        FileResponse: Generated audio file
    
    Raises:
        HTTPException: If speech generation fails
    """
    try:
        model = load_xtts_model()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            output_path = temp_audio.name
        
        logger.info(f"Generating speech for text: '{request.text[:50]}...'")
        start_time = time.time()
        
        try:
            model.tts_to_file(
                text=request.text,
                file_path=output_path,
                speaker_wav=str(POLISH_FEMALE_VOICE),
                language="pl"
            )
                
            logger.info(f"Speech generated in {time.time() - start_time:.2f}s")
                
            if os.path.getsize(output_path) < 100:
                raise Exception("Generated audio file is too small or empty")
                
        except Exception as e:
            logger.error(f"Error during speech generation: {str(e)}")
            raise
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
    except Exception as e:
        error_details = str(e) + "\n" + traceback.format_exc()
        logger.error(f"Błąd: {error_details}")
        raise HTTPException(status_code=500, detail=f"Błąd: {str(e)}")