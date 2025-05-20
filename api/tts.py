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
tts_initialized = False
voice_sample_available = False

async def initialize_tts():
    """
    Initializes the TTS service by loading the model and checking for voice samples.
    Called at application startup rather than during endpoint calls.
    
    Returns:
        bool: True if initialization was successful
    """
    global tts_model, tts_initialized, voice_sample_available
    
    try:
        logger.info("Initializing TTS service...")
        
        if not POLISH_FEMALE_VOICE.exists():
            logger.warning(f"Voice sample missing at {POLISH_FEMALE_VOICE}")
            
            with open(POLISH_FEMALE_VOICE.with_suffix('.txt'), 'w', encoding='utf-8') as f:
                f.write("Place a WAV file (44.1kHz, min. 3 seconds) with a Polish female voice sample here.\n")
                f.write("File name should be: polish_female_voice.wav")
            
            voice_sample_available = False
            logger.error("Voice sample not found. Speech synthesis will not be available.")
        else:
            logger.info(f"Voice sample found: {POLISH_FEMALE_VOICE}")
            voice_sample_available = True
        
        logger.info("Loading XTTS-v2 model...")
        start_time = time.time()
        
        try:
            tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
            logger.info(f"XTTS-v2 model loaded in {time.time() - start_time:.2f}s")
            
            if voice_sample_available:
                logger.info("Performing warm-up inference to load model into memory...")
                with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_audio:
                    warmup_output_path = temp_audio.name
                
                warmup_start = time.time()
                tts_model.tts_to_file(
                    text="Dzień dobry",
                    file_path=warmup_output_path,
                    speaker_wav=str(POLISH_FEMALE_VOICE),
                    language="pl"
                )
                
                warmup_time = time.time() - warmup_start
                logger.info(f"TTS warm-up completed in {warmup_time:.2f}s")
                
                if os.path.exists(warmup_output_path) and os.path.getsize(warmup_output_path) > 100:
                    logger.info("Warm-up inference successful")
                else:
                    logger.warning("Warm-up inference did not produce a valid audio file")
            
            tts_initialized = True
            logger.info("TTS service initialized successfully")
            return True
            
        except Exception as model_error:
            logger.error(f"Failed to load XTTS model: {str(model_error)}")
            logger.error(traceback.format_exc())
            tts_initialized = False
            return False
            
    except Exception as e:
        error_details = str(e) + "\n" + traceback.format_exc()
        logger.error(f"TTS initialization error: {error_details}")
        tts_initialized = False
        return False

def get_tts_status():
    """
    Gets the initialized TTS status.
    
    Returns:
        bool: True if the TTS service is available
        
    Raises:
        HTTPException: When TTS is not initialized or voice sample is not available
    """
    if not tts_initialized:
        logger.error("TTS service not initialized. This should have happened at startup.")
        raise HTTPException(
            status_code=500, 
            detail="TTS service not properly initialized. Please restart the server."
        )
    
    if not voice_sample_available:
        raise HTTPException(
            status_code=500,
            detail="Voice sample not available. Please add a Polish female voice sample to assets/polish_female_voice.wav"
        )
    
    return True


@router.post("/synthesize")
async def generate_polish_speech(request: PolishTextToSpeech):
    """
    Generates speech using the pre-initialized TTS model.
    
    Args:
        request (PolishTextToSpeech): Request object containing the text to synthesize
    
    Returns:
        FileResponse: Generated audio file
    
    Raises:
        HTTPException: If speech generation fails
    """
    try:
        get_tts_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            output_path = temp_audio.name
        
        logger.info(f"Generating speech for text: '{request.text[:50]}...'")
        start_time = time.time()
        
        try:
            tts_model.tts_to_file(
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