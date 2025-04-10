import tempfile
import os
import requests
import logging
import traceback
import subprocess
import zipfile
import shutil
import sys
import torch
import time
import multiprocessing

from faster_whisper import WhisperModel

from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt")

router = APIRouter()

# Models to choose from: "tiny", "base", "small", "medium", "large-v2, "large-v3", "large-v3-turbo"
whisper_model_size = "medium"
MODEL_DIR = Path(__file__).parent.parent / "model"
MODEL_DIR.mkdir(exist_ok=True)
FFMPEG_DIR = MODEL_DIR / "ffmpeg"

def download_model(model_name, root):
    """
    Downloads Whisper model from Hugging Face to the specified directory.
    
    Args:
        model_name (str): Name of the model to download
        root (str or Path): Directory to store the model
    
    Returns:
        str: Path to the downloaded model
    """
    # Faster-whisper handles model downloading automatically
    # This function is kept for compatibility
    logger.info(f"Faster-whisper will download model {model_name} automatically if needed")
    return str(root)

def is_ffmpeg_installed():
    """
    Checks if FFmpeg is available in the system.
    
    Args:
        None
    
    Returns:
        bool: True if FFmpeg is installed, False otherwise
    """
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def ensure_ffmpeg():
    """
    Ensures FFmpeg is available, downloading it if necessary.
    
    Args:
        None
    
    Returns:
        None
    """
    if is_ffmpeg_installed():
        logger.info("FFmpeg is already installed in the system.")
        return
    
    ffmpeg_exe = FFMPEG_DIR / "bin" / "ffmpeg.exe"
    if ffmpeg_exe.exists():
        os.environ["PATH"] = f"{str(FFMPEG_DIR / 'bin')};{os.environ['PATH']}"
        logger.info(f"Using local FFmpeg from {ffmpeg_exe}")
        return
    
    logger.info("FFmpeg not found. Downloading automatically...")
    FFMPEG_DIR.mkdir(exist_ok=True)
    
    ffmpeg_url = "https://github.com/GyanD/codexffmpeg/releases/download/6.0/ffmpeg-6.0-essentials_build.zip"
    zip_path = FFMPEG_DIR / "ffmpeg.zip"
    
    try:
        logger.info(f"Downloading FFmpeg from {ffmpeg_url}...")
        response = requests.get(ffmpeg_url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info("Extracting FFmpeg...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(FFMPEG_DIR)
        
        extracted_dir = next(FFMPEG_DIR.glob('ffmpeg-*'))
        
        for item in extracted_dir.iterdir():
            shutil.move(str(item), str(FFMPEG_DIR))
        
        os.remove(zip_path)
        shutil.rmtree(extracted_dir, ignore_errors=True)
        
        os.environ["PATH"] = f"{str(FFMPEG_DIR / 'bin')};{os.environ['PATH']}"
        logger.info(f"FFmpeg installed in {FFMPEG_DIR / 'bin'}")
        
    except Exception as e:
        logger.error(f"Error downloading FFmpeg: {str(e)}")
        raise RuntimeError(f"Cannot download FFmpeg. Install FFmpeg manually: {str(e)}")


def is_gpu_available():
    """
    Checks for GPU availability with CUDA support.
    
    Args:
        None
    
    Returns:
        bool: True if GPU with CUDA is available, False otherwise
    """
    return torch.cuda.is_available()


def get_model(model_name):
    """
    Loads Whisper model using faster-whisper implementation.
    
    Args:
        model_name (str): Name of the model to load
    
    Returns:
        WhisperModel: Loaded Whisper model
    """
    # Determine device and compute type
    device = "cuda" if is_gpu_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    cpu_count = multiprocessing.cpu_count()
    optimal_workers = max(1, cpu_count - 1)
    
    if device == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)} with {compute_type} precision")
    else:
        logger.info(f"GPU is not available, using CPU with {compute_type} quantization")
    
    download_root = str(MODEL_DIR)
    
    if "/" in model_name:
        display_name = model_name.split("/")[-1]
        logger.info(f"Using HuggingFace model: {model_name}")
    else:
        display_name = model_name
    
    logger.info(f"Loading model: {display_name}")
    model = WhisperModel(
        model_name, 
        device=device, 
        compute_type=compute_type, 
        download_root=download_root,
        num_workers=optimal_workers,
        cpu_threads=optimal_workers
    )
    
    return model

@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(default=...)
):
    """
    Transcribes audio to text using the Faster-Whisper model.
    Automatically handles mixed languages in a single recording.
    
    Args:
        file (UploadFile): Audio file to transcribe
    
    Returns:
        dict: Contains transcribed text, detected language, and processing time
    """
    start_total = time.time()
    
    try:
        ensure_ffmpeg()
        
        logger.info(f"Initializing Faster-Whisper model ({whisper_model_size})...")
        model = get_model(whisper_model_size)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio.flush()
            temp_path = temp_audio.name
        
        try:
            logger.info("Transcribing audio file with mixed language support")
            
            assistant_prompt = """
            Transkrypcja zawiera polecenia głosowe dla asystenta o imieniu Ada w języku polskim 
            z możliwymi wtrąceniami nazw własnych i terminów w języku angielskim. 
            
            Komendy mogą dotyczyć:
            - Muzyki: "puść na Spotify piosenkę Shape of You", "włącz utwór ASAP Rocky", "zmniejsz głośność"
            - Kalendarza: "dodaj spotkanie na Google Calendar na jutro", "przypomnij mi o spotkaniu z Tomaszem"
            - Pytań informacyjnych: "jaka będzie dziś pogoda", "jak dojechać do centrum"
            
            Ada, proszę transkrybuj dokładnie polecenia, zachowując oryginalne nazwy własne, 
            aplikacji i angielskie terminy dokładnie tak, jak zostały wypowiedziane.
            Nigdy nie tłumacz fragmentów w innych językach.

            Przykłady:
            Puść Jigsaw falling into place Radiohead
            Dodaj spotkanie na Google Calendar na jutro
            """

            start_transcription = time.time()
            
            segments, info = model.transcribe(
                temp_path,
                task="transcribe",
                language=None, 
                initial_prompt=assistant_prompt,
                temperature=0.0,
                beam_size= 1,
                vad_filter=True,  
                vad_parameters=dict(min_silence_duration_ms=500) 
            )

            transcription_time = time.time() - start_transcription
            logger.info(f"Transcription took {transcription_time:.2f} seconds")
            
            full_text = ""
            for segment in segments:
                full_text += segment.text
            
            detected_language = info.language
            total_time = time.time() - start_total
            
            return {
                "text": full_text.strip(),
                "detected_language": detected_language,
                "transcription_time": round(transcription_time, 2),
                "total_time": round(total_time, 2)
            }
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        error_details = str(e) + "\n" + traceback.format_exc()
        logger.error(f"Transcription error: {error_details}")
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")