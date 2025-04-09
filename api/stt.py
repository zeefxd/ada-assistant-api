import tempfile
import os
import requests
import logging
import traceback
import whisper
import subprocess
import zipfile
import shutil
import sys
import torch

from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt")

router = APIRouter()

# Modele do wyboru: "tiny", "base", "small", "medium", "large"
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
    
    model_path = os.path.join(root, f"{model_name}.pt")
    
    if os.path.exists(model_path):
        logger.info(f"Model already exists: {model_path}")
        return model_path
    
    hf_models = {
        "tiny": "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin",
        "base": "https://huggingface.co/openai/whisper-base/resolve/main/pytorch_model.bin",
        "small": "https://huggingface.co/openai/whisper-small/resolve/main/pytorch_model.bin",
        "medium": "https://huggingface.co/openai/whisper-medium/resolve/main/pytorch_model.bin",
        "large": "https://huggingface.co/openai/whisper-large-v2/resolve/main/pytorch_model.bin"
    }
    
    if model_name not in hf_models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(hf_models.keys())}")
    
    url = hf_models[model_name]
    logger.info(f"Downloading model {model_name} from Hugging Face to {model_path}...")
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    return model_path

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
    Loads Whisper model from a custom folder instead of default cache location.
    
    Args:
        model_name (str): Name of the model to load
    
    Returns:
        whisper.Model: Loaded Whisper model
    """
    model_path = MODEL_DIR / f"{model_name}.pt"
    
    if not model_path.exists():
        download_model(model_name, MODEL_DIR)
    
    device = "cuda" if is_gpu_available() else "cpu"
    if device == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("GPU is not available, using CPU")
    
    return whisper.load_model(model_name, download_root=MODEL_DIR, device=device)

@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(default=...)
):
    """
    Transcribes audio to text using the Whisper model.
    Automatically handles mixed languages in a single recording.
    
    Args:
        file (UploadFile): Audio file to transcribe
    
    Returns:
        dict: Contains transcribed text and detected language
    """
    try:
        ensure_ffmpeg()
        
        logger.info(f"Initializing Whisper model ({whisper_model_size})...")
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
            "Puść Jigsaw falling into place Radiohead",
            "Dodaj spotkanie na Google Calendar na jutro"
            """
            use_fp16 = is_gpu_available()
            
            result = model.transcribe(
                temp_path,
                task="transcribe",
                language=None,
                initial_prompt=assistant_prompt,
                temperature=0.0,
                best_of=5,
                fp16=use_fp16
            )
            
            detected_language = result.get("language", "unknown")
            return {
                "text": result["text"].strip(), 
                "detected_language": detected_language
            }
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        error_details = str(e) + "\n" + traceback.format_exc()
        logger.error(f"Transcription error: {error_details}")
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")