import platform
import tempfile
import os
import requests
import logging
import traceback
import subprocess
import zipfile
import shutil
import time
import multiprocessing
import json
import asyncio
import re
import httpx

from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt_server")

router = APIRouter()

# tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3, etc.
WHISPER_CPP_MODEL_NAME = "large-v3-turbo-q8-v3"

SCRIPT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = SCRIPT_DIR / "model"
WHISPER_CPP_DIR = MODEL_DIR / "whisper.cpp"
WHISPER_CPP_MODEL_DIR = WHISPER_CPP_DIR / "models"
WHISPER_CPP_BIN_DIR = SCRIPT_DIR / "bin" / "whisper.cpp"

WHISPER_SERVER_EXE = WHISPER_CPP_BIN_DIR / "whisper-server.exe"
WHISPER_CLI_EXE = WHISPER_CPP_BIN_DIR / "whisper-cli.exe"

WHISPER_CPP_MODEL_PATH = WHISPER_CPP_MODEL_DIR / f"ggml-{WHISPER_CPP_MODEL_NAME}.bin"
FFMPEG_DIR = MODEL_DIR / "ffmpeg"

WHISPER_SERVER_HOST = "127.0.0.1"
WHISPER_SERVER_PORT = 9090
WHISPER_SERVER_URL = f"http://{WHISPER_SERVER_HOST}:{WHISPER_SERVER_PORT}"
WHISPER_INFERENCE_ENDPOINT = f"{WHISPER_SERVER_URL}/inference"

MODEL_DIR.mkdir(exist_ok=True)
WHISPER_CPP_DIR.mkdir(exist_ok=True)
WHISPER_CPP_MODEL_DIR.mkdir(exist_ok=True)
WHISPER_CPP_BIN_DIR.mkdir(parents=True, exist_ok=True)
FFMPEG_DIR.mkdir(exist_ok=True)

stt_initialized = False
whisper_server_process = None

async def initialize_stt():
    """
    Initializes the STT service by ensuring all required components are available.
    
    Returns:
        bool: True if initialization was successful
    """
    
    global stt_initialized, whisper_server_process
    if stt_initialized:
        logger.info("STT service already initialized.")
        return True

    logger.info("Initializing STT service (Whisper Server Mode)...")
    try:
        ensure_whisper_server_executable()
        ensure_whisper_cpp_model(WHISPER_CPP_MODEL_NAME, WHISPER_CPP_MODEL_PATH)
        ensure_ffmpeg()

        command = [
            str(WHISPER_SERVER_EXE),
            "--model", str(WHISPER_CPP_MODEL_PATH),
            "--host", WHISPER_SERVER_HOST,
            "--port", str(WHISPER_SERVER_PORT),
            "--threads", str(multiprocessing.cpu_count()),
            "--language", "auto",
            # "--convert",
        ]
        logger.info(f"Starting whisper-server.exe with command: {' '.join(command)}")

        whisper_server_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        )

        await asyncio.sleep(3)

        if whisper_server_process.poll() is not None:
            stdout, stderr = whisper_server_process.communicate()
            logger.error(f"whisper-server.exe failed to start. Return code: {whisper_server_process.returncode}")
            logger.error(f"Stdout: {stdout.decode(errors='ignore') if stdout else 'N/A'}")
            logger.error(f"Stderr: {stderr.decode(errors='ignore') if stderr else 'N/A'}")
            stt_initialized = False
            whisper_server_process = None
            return False

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(WHISPER_SERVER_URL, timeout=5)
                if response.status_code == 200 or response.status_code == 405:
                    logger.info(f"whisper-server.exe responded to health check at {WHISPER_SERVER_URL}")
                else:
                    logger.warning(f"whisper-server.exe health check at {WHISPER_SERVER_URL} returned status {response.status_code}. Proceeding, but check server logs.")

        except httpx.RequestError as e:
            logger.error(f"Failed to connect to whisper-server.exe at {WHISPER_SERVER_URL} for health check: {e}")
            logger.error("Make sure the server started correctly and the host/port are accessible.")
            logger.warning("Proceeding despite health check failure. Transcription might not work.")


        stt_initialized = True
        logger.info("STT service (Whisper Server Mode) initialized successfully.")
        return True

    except FileNotFoundError as e:
        logger.error(f"A required executable not found: {e}")
    except RuntimeError as e:
        logger.error(f"A runtime error occurred during initialization: {e}")
    except Exception as e:
        error_details = str(e) + "\n" + traceback.format_exc()
        logger.error(f"STT initialization error: {error_details}")

    stt_initialized = False
    if whisper_server_process and whisper_server_process.poll() is None:
        whisper_server_process.terminate()
        whisper_server_process.wait()
    whisper_server_process = None
    return False

async def shutdown_stt():
    """
    Shuts down the speech-to-text server process if it's running.
        
    Raises:
        Exception: If an error occurs during the shutdown process
    """
    
    global stt_initialized, whisper_server_process
    if whisper_server_process and whisper_server_process.poll() is None:
        logger.info("Shutting down whisper-server.exe...")
        whisper_server_process.terminate()
        try:
            await asyncio.to_thread(whisper_server_process.wait, timeout=10)
            logger.info("whisper-server.exe terminated.")
        except subprocess.TimeoutExpired:
            logger.warning("whisper-server.exe did not terminate gracefully, killing.")
            whisper_server_process.kill()
            await asyncio.to_thread(whisper_server_process.wait)
            logger.info("whisper-server.exe killed.")
        except Exception as e:
            logger.error(f"Error during whisper-server.exe shutdown: {e}")
    else:
        logger.info("whisper-server.exe not running or already shut down.")
    stt_initialized = False
    whisper_server_process = None

def get_stt_status():
    """
    Gets the initialized STT status.
    
    Returns:
        bool: True if the STT service is available
        
    Raises:
        HTTPException: When STT is not initialized
    """
    
    if not stt_initialized:
        logger.error("STT service not initialized. This should have happened at startup.")
        raise HTTPException(
            status_code=500,
            detail="STT service not properly initialized. Please restart the server."
        )
    if whisper_server_process is None or whisper_server_process.poll() is not None:
        logger.error("Whisper server process is not running. STT service unavailable.")
        raise HTTPException(
            status_code=503,
            detail="STT backend server (whisper-server.exe) is not running. Please restart the main server."
        )
    return True

def get_whisper_cpp_model_url(model_name):
    """
    Maps model names to their download URLs from HuggingFace.
    
    Args:
        model_name (str): Name of the whisper.cpp model to download
        
    Returns:
        str: URL to download the model from HuggingFace
        
    Raises:
        ValueError: If the model name is not recognized
    """
    
    # https://huggingface.co/ggerganov/whisper.cpp/tree/main
    model_map = {
        "tiny": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
        "tiny.en": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
        "base": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        "base.en": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
        "small": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
        "small.en": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin",
        "medium": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
        "medium.en": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin",
        "large-v1": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v1.bin",
        "large-v2": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2.bin",
        "large-v3": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
        "large-v3-turbo-q8-v3": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q8_0.bin",
    }
    url = model_map.get(model_name)
    if not url:
        raise ValueError(f"Unknown whisper.cpp model name: {model_name}. Check available models and update `get_whisper_cpp_model_url`.")
    return url

def download_file(url, dest_path):
    """
    Downloads a file from a URL with progress tracking.
    
    Args:
        url (str): URL to download from
        dest_path (Path): Path where the downloaded file will be saved
        
    Raises:
        ConnectionError: If the download fails due to network issues
        Exception: For other download errors
    """
    
    logger.info(f"Downloading {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192*1024):
                f.write(chunk)
        logger.info("Download complete.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
        if dest_path.exists():
            os.remove(dest_path)
        raise ConnectionError(f"Failed to download file from {url}") from e
    except Exception as e:
        logger.error(f"An error occurred during download: {e}")
        if dest_path.exists():
            os.remove(dest_path)
        raise e

def ensure_whisper_cpp_model(model_name, model_path):
    """
    Ensures that the specified whisper.cpp model exists, downloading it if needed.
    
    Args:
        model_name (str): Name of the model to ensure
        model_path (Path): Expected path where the model should be located
        
    Raises:
        RuntimeError: If the model download fails
    """
    
    if not model_path.exists():
        logger.warning(f"whisper.cpp model not found at {model_path}.")
        model_url = get_whisper_cpp_model_url(model_name)
        try:
            download_file(model_url, model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download whisper.cpp model '{model_name}'. Please download it manually from {model_url} and place it at {model_path}. Error: {e}")
    else:
        logger.info(f"Using existing whisper.cpp model: {model_path}")

def ensure_whisper_server_executable():
    """
    Checks if the whisper.cpp server executable exists.
    Raises:
        FileNotFoundError: If the executable is not found
    """
    
    if not WHISPER_SERVER_EXE.exists():
        logger.error(f"whisper-server.exe not found at {WHISPER_SERVER_EXE}")
        cli_exe_to_mention = WHISPER_CLI_EXE if WHISPER_CLI_EXE.exists() else "whisper-server.exe"
        exe_name_to_mention = "whisper-server.exe"

        raise FileNotFoundError(
            f"{exe_name_to_mention} not found in '{WHISPER_CPP_BIN_DIR}'. "
            f"Please compile/download whisper.cpp and place the executable (e.g., '{exe_name_to_mention}') "
            f"in the '{WHISPER_CPP_BIN_DIR}' directory, or update the WHISPER_SERVER_EXE variable."
        )
    logger.info(f"Found whisper-server executable: {WHISPER_SERVER_EXE}")

def is_ffmpeg_installed():
    """
    Checks if FFmpeg is available either locally or in the system PATH.
    
    Returns:
        bool: True if FFmpeg is available, False otherwise
    """
    
    local_ffmpeg_exe = FFMPEG_DIR / "bin" / "ffmpeg.exe"
    if local_ffmpeg_exe.exists():
        return True
    try:
        cmd = ['where', 'ffmpeg'] if platform.system() == "Windows" else ['which', 'ffmpeg']
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode == 0:
            return True
        result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def ensure_ffmpeg():
    """
    Ensures FFmpeg is available, downloading and installing it if needed.
    
    Raises:
        RuntimeError: If FFmpeg setup fails
    """
    
    local_ffmpeg_bin = FFMPEG_DIR / "bin"
    local_ffmpeg_exe = local_ffmpeg_bin / "ffmpeg.exe"

    if local_ffmpeg_exe.exists():
        if str(local_ffmpeg_bin) not in os.environ['PATH']:
             os.environ["PATH"] = f"{str(local_ffmpeg_bin)}{os.pathsep}{os.environ['PATH']}"
        logger.info(f"Using local FFmpeg from {local_ffmpeg_exe}")
        return

    if is_ffmpeg_installed():
         logger.info("FFmpeg found in system PATH.")
         return

    logger.info("FFmpeg not found locally or in system PATH. Downloading automatically...")
    FFMPEG_DIR.mkdir(exist_ok=True)

    ffmpeg_url = "https://github.com/GyanD/codexffmpeg/releases/download/6.1.1/ffmpeg-6.1.1-essentials_build.zip"
    zip_path = FFMPEG_DIR / "ffmpeg.zip"

    try:
        logger.info(f"Downloading FFmpeg from {ffmpeg_url}...")
        download_file(ffmpeg_url, zip_path)

        logger.info("Extracting FFmpeg...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(FFMPEG_DIR)

        extracted_dirs = [d for d in FFMPEG_DIR.iterdir() if d.is_dir() and d.name.startswith('ffmpeg-')]
        if not extracted_dirs:
            raise FileNotFoundError("Could not find extracted FFmpeg directory.")
        extracted_dir = extracted_dirs[0]

        target_bin_dir = FFMPEG_DIR / "bin"
        target_bin_dir.mkdir(exist_ok=True)

        source_bin_dir = extracted_dir / "bin"
        if source_bin_dir.exists():
            for item in source_bin_dir.iterdir():
                shutil.move(str(source_bin_dir / item.name), str(target_bin_dir / item.name))
        else:
             for item in extracted_dir.iterdir():
                shutil.move(str(item), str(FFMPEG_DIR / item.name))


        os.remove(zip_path)
        shutil.rmtree(extracted_dir, ignore_errors=True)

        os.environ["PATH"] = f"{str(local_ffmpeg_bin)}{os.pathsep}{os.environ['PATH']}"
        logger.info(f"FFmpeg installed locally in {local_ffmpeg_bin}")

    except Exception as e:
        logger.error(f"Error downloading or setting up FFmpeg: {str(e)}\n{traceback.format_exc()}")
        raise RuntimeError(f"Failed to automatically install FFmpeg. Please install FFmpeg manually and ensure it's in your system PATH, or place the executable in {local_ffmpeg_exe}. Error: {str(e)}")


async def cleanup_temp_file(temp_path: Path):
    """
    Safely deletes a temporary file.
    
    Args:
        temp_path (Path): Path to the temporary file to delete
    """
    
    if temp_path and temp_path.exists():
        try:
            await asyncio.to_thread(os.unlink, temp_path)
            logger.info(f"Deleted temporary file: {temp_path}")
        except OSError as e:
            logger.error(f"Error deleting temporary file {temp_path}: {e}")

@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(default=...),
    language: str = "auto",
):
    """
    Transcribes the uploaded audio file using whisper-server.cpp and streams
    the transcribed segments back to the client using Server-Sent Events (SSE).
    
    Args:
        file (UploadFile): Audio file to transcribe
        language (str): Language code ("auto" for auto-detect, "PL" for Polish, etc.)
        use_json (bool): If True, attempt to use JSON output mode (may not work with all builds)
    
    Returns:
        StreamingResponse: Server-Sent Events stream with transcription segments
    """
    
    start_time_request = time.time()
    temp_audio_path = None

    try:
        get_stt_status()
    except HTTPException as e:
        logger.error(f"STT service not available: {e.detail}")
        raise

    try:
        suffix = Path(file.filename).suffix if file.filename else ".tmpaudio"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio_file_obj:
            content = await file.read()
            if not content:
                logger.warning("Received empty file upload.")
                raise HTTPException(status_code=400, detail="Received empty file.")
            temp_audio_file_obj.write(content)
            temp_audio_path = Path(temp_audio_file_obj.name)
        logger.info(f"Saved uploaded audio to temporary file: {temp_audio_path}")

        async def sse_transcription_generator():
            nonlocal temp_audio_path
            try:
                form_data = {
                    "language": language,
                    "response_format": "json",
                }
                files_payload = {'file': (file.filename or f"audio{suffix}", content, file.content_type or "application/octet-stream")}

                logger.info(f"Sending audio to whisper-server: {WHISPER_INFERENCE_ENDPOINT} with lang={language}")
                async with httpx.AsyncClient(timeout=300.0) as client:
                    response = await client.post(
                        WHISPER_INFERENCE_ENDPOINT,
                        data=form_data,
                        files=files_payload
                    )
                
                if response.status_code != 200:
                    error_detail = f"Whisper server returned error {response.status_code}."
                    try:
                        server_error = response.json()
                        error_detail += f" Server message: {server_error.get('error', response.text)}"
                    except Exception:
                        error_detail += f" Server response: {response.text}"
                    logger.error(error_detail)
                    err_data = {"type": "error", "message": error_detail}
                    yield f"data: {json.dumps(err_data, ensure_ascii=False)}\n\n"
                    return

                transcription_result = response.json()
                logger.info(f"Received response from whisper-server ({response.status_code})")

                if "transcription" in transcription_result and isinstance(transcription_result["transcription"], dict):
                    main_transcription_block = transcription_result["transcription"]
                elif isinstance(transcription_result, dict) and "text" in transcription_result:
                    main_transcription_block = transcription_result
                else:
                    logger.error(f"Unexpected transcription result structure: {transcription_result}")
                    err_data = {"type": "error", "message": "Unexpected response structure from STT server."}
                    yield f"data: {json.dumps(err_data, ensure_ascii=False)}\n\n"
                    return

                full_text_raw = main_transcription_block.get("text", "")
                full_text = re.sub(r'\s+', ' ', full_text_raw).strip()

                if not full_text and response.status_code == 200:
                    logger.warning("Whisper server returned a successful response but with an empty transcription text.")
                
                elapsed_time_request = time.time() - start_time_request
                final_data = {
                    "type": "done",
                    "message": "Transcription complete.",
                    "full_text": full_text,
                    "elapsed_time": f"{elapsed_time_request:.2f}s",
                }
                logger.info(f"Transcription processed in {elapsed_time_request:.2f}s.")
                yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"

            except httpx.ReadTimeout:
                logger.error("Request to whisper-server timed out.")
                error_data = {"type": "error", "message": "Transcription timed out."}
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            except httpx.RequestError as e:
                logger.error(f"HTTPX Request error communicating with whisper-server: {e}")
                error_data = {"type": "error", "message": f"Error communicating with STT server: {type(e).__name__}"}
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON response from whisper-server: {e}")
                error_data = {"type": "error", "message": "Invalid response from STT server."}
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            except asyncio.CancelledError:
                logger.warning("Transcription stream cancelled by client.")
            except Exception as e:
                logger.error(f"Error during transcription streaming: {e}\n{traceback.format_exc()}")
                error_data = {"type": "error", "message": f"Server error during transcription: {str(e)}"}
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            finally:
                if temp_audio_path:
                    await cleanup_temp_file(temp_audio_path)
                logger.info("Transcription streaming completed and resources cleaned up.")

        return StreamingResponse(
            sse_transcription_generator(),
            media_type="text/event-stream; charset=utf-8"
        )

    except HTTPException:
        await cleanup_temp_file(temp_audio_path)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /transcribe endpoint: {e}\n{traceback.format_exc()}")
        await cleanup_temp_file(temp_audio_path)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")