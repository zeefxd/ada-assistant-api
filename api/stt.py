import platform
import tempfile
import os
import requests
import logging
import traceback
import subprocess
import zipfile
import shutil
import sys
import time
import multiprocessing
import json
import asyncio
import re

from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse 
from asyncio import threads

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt")

router = APIRouter()

# tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3, etc.
WHISPER_CPP_MODEL_NAME = "large-v3-turbo-q8-v3"

SCRIPT_DIR = Path(__file__).parent.parent
MODEL_DIR = SCRIPT_DIR / "model"
WHISPER_CPP_DIR = MODEL_DIR / "whisper.cpp"
WHISPER_CPP_MODEL_DIR = WHISPER_CPP_DIR / "models"
WHISPER_CPP_BIN_DIR = SCRIPT_DIR / "bin" / "whisper.cpp"
WHISPER_CPP_EXE = WHISPER_CPP_BIN_DIR / "whisper-cli.exe"
WHISPER_CPP_MODEL_PATH = WHISPER_CPP_MODEL_DIR / f"ggml-{WHISPER_CPP_MODEL_NAME}.bin"

FFMPEG_DIR = MODEL_DIR / "ffmpeg"

MODEL_DIR.mkdir(exist_ok=True)
WHISPER_CPP_DIR.mkdir(exist_ok=True)
WHISPER_CPP_MODEL_DIR.mkdir(exist_ok=True)
WHISPER_CPP_BIN_DIR.mkdir(parents=True, exist_ok=True)
FFMPEG_DIR.mkdir(exist_ok=True)

stt_initialized = False

async def initialize_stt():
    """
    Initializes the STT service by ensuring all required components are available.
    Called at application startup rather than during endpoint calls.
    
    Returns:
        bool: True if initialization was successful
    """
    global stt_initialized
    
    try:
        logger.info("Initializing STT service...")
        
        try:
            ensure_whisper_cpp_executable()
        except FileNotFoundError as e:
            logger.error(f"Whisper.cpp executable not found: {e}")
            stt_initialized = False
            return False
            
        try:
            ensure_whisper_cpp_model(WHISPER_CPP_MODEL_NAME, WHISPER_CPP_MODEL_PATH)
        except RuntimeError as e:
            logger.error(f"Failed to ensure whisper.cpp model: {e}")
            stt_initialized = False
            return False
            
        try:
            ensure_ffmpeg()
        except RuntimeError as e:
            logger.error(f"Failed to ensure FFmpeg: {e}")
            stt_initialized = False
            return False
        
        stt_initialized = True
        logger.info("STT service initialized successfully")
        return True
    except Exception as e:
        error_details = str(e) + "\n" + traceback.format_exc()
        logger.error(f"STT initialization error: {error_details}")
        stt_initialized = False
        return False

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
        response = requests.get(url, stream=True)
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


def ensure_whisper_cpp_executable():
    """
    Checks if the whisper.cpp executable exists.
    
    Raises:
        FileNotFoundError: If the executable is not found
    """
    if not WHISPER_CPP_EXE.exists():
        logger.error(f"whisper.cpp main executable not found at {WHISPER_CPP_EXE}")
        raise FileNotFoundError(
            f"whisper.cpp main executable ('{WHISPER_CPP_EXE.name}') not found in '{WHISPER_CPP_BIN_DIR}'. "
            f"and place the executable (e.g., 'whisper-cli.exe') in the '{WHISPER_CPP_BIN_DIR}' directory, "
            f"or update the WHISPER_CPP_EXE variable in this script."
        )
    logger.info(f"Found whisper.cpp main executable: {WHISPER_CPP_EXE}")


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
        result = subprocess.run(['ffmpeg', '-version'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True, check=False)
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
             os.environ["PATH"] = f"{str(local_ffmpeg_bin)};{os.environ['PATH']}"
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

        for item in extracted_dir.iterdir():
            shutil.move(str(item), str(FFMPEG_DIR / item.name))

        os.remove(zip_path)
        shutil.rmtree(extracted_dir, ignore_errors=True)

        os.environ["PATH"] = f"{str(local_ffmpeg_bin)};{os.environ['PATH']}"
        logger.info(f"FFmpeg installed locally in {local_ffmpeg_bin}")

    except Exception as e:
        logger.error(f"Error downloading or setting up FFmpeg: {str(e)}")
        raise RuntimeError(f"Failed to automatically install FFmpeg. Please install FFmpeg manually and ensure it's in your system PATH, or place the executable in {local_ffmpeg_exe}. Error: {str(e)}")

async def cleanup_temp_file(temp_path: Path):
    """
    Safely deletes a temporary file.
    
    Args:
        temp_path (Path): Path to the temporary file to delete
    """
    if temp_path and temp_path.exists():
        try:
            os.unlink(temp_path)
            logger.info(f"Deleted temporary file: {temp_path}")
        except OSError as e:
            logger.error(f"Error deleting temporary file {temp_path}: {e}")
    else:
        logger.warning(f"Temporary file {temp_path} not found or already deleted.")


@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(default=...),
    language: str = "auto",
    use_json: bool = False
):
    """
    Transcribes the uploaded audio file using whisper.cpp and streams
    the transcribed segments back to the client using Server-Sent Events (SSE).
    
    Args:
        file (UploadFile): Audio file to transcribe
        language (str): Language code ("auto" for auto-detect, "PL" for Polish, etc.)
        use_json (bool): If True, attempt to use JSON output mode (may not work with all builds)
    
    Returns:
        StreamingResponse: Server-Sent Events stream with transcription segments
    """
    start_time = time.time()
    temp_path = None
    temp_json_path = None
    
    try:
        get_stt_status()
    except HTTPException as e:
        logger.error(f"STT service not properly initialized: {e}")
        raise
    
    try:
        suffix = Path(file.filename).suffix if file.filename else ".tmpaudio"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            content = await file.read()
            if not content:
                logger.warning("Received empty file upload.")
                raise HTTPException(status_code=400, detail="Received empty file.")
            temp_audio.write(content)
            temp_audio.flush()
            temp_path = Path(temp_audio.name)
            if use_json:
                temp_json_path = Path(str(temp_path) + ".json")
            logger.info(f"Saved uploaded audio to temporary file: {temp_path}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to read or save uploaded file: {e}\n{traceback.format_exc()}")
        await cleanup_temp_file(temp_path)
        raise HTTPException(status_code=500, detail=f"Error processing uploaded file: {e}")

    command = [
        str(WHISPER_CPP_EXE),
        "-m", str(WHISPER_CPP_MODEL_PATH),
    ]

    help_result = subprocess.run([str(WHISPER_CPP_EXE), "--help"], 
                               capture_output=True, text=True, check=False)
    if "-f FNAME" in help_result.stdout:
        command.extend(["-f", str(temp_path)])
    else:
        command.append(str(temp_path))
    
    command.extend([
        "-l", language,
        "-t", str(multiprocessing.cpu_count()),
        "-pp",
        "-p", "1",
    ])
    
    if use_json:
        command.append("-oj")
    
    if "-fa" in help_result.stdout:
        command.append("-fa")
    
    logger.info(f"Running command: {' '.join(command)}")

    async def sse_transcription_generator():
        nonlocal temp_path, temp_json_path
        process = None
        json_result = None
        
        try:
            if platform.system() == "Windows":
                logger.info("Using subprocess.Popen for whisper-cli on Windows.")
                string_command = [str(c) for c in command]
                process = subprocess.Popen(
                    string_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    bufsize=1,
                    universal_newlines=True
                )
                logger.info(f"Started whisper-cli process with PID: {process.pid} using subprocess.Popen")

                full_transcription_text = ""
                segment_regex = re.compile(r"\[(\d{2}:\d{2}:\d{2}\.\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(.*)")

                for line in iter(process.stdout.readline, ''):
                    if not line and process.poll() is not None:
                        break
                    line = line.strip()
                    match = segment_regex.match(line)
                    if match:
                        start_time_str = match.group(1)
                        end_time_str = match.group(2)
                        segment_text = match.group(3).strip()

                        if segment_text:
                            logger.info(f"[whisper-cli segment]: [{start_time_str} --> {end_time_str}] {segment_text}")
                            full_transcription_text += segment_text + " "

                            sse_data = {
                                "type": "segment",
                                "start_time": start_time_str,
                                "end_time": end_time_str,
                                "text": segment_text
                            }
                            yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"
                            await asyncio.sleep(0)
                        else:
                           logger.debug(f"[whisper-cli] Matched regex but segment text was empty: {line}")
                    elif "whisper_print_progress_callback" in line:
                        logger.info(f"[whisper-cli progress]: {line}")
                        progress_match = re.search(r"progress\s*=\s*(\d+)%", line)
                        if progress_match:
                            progress_pct = progress_match.group(1)
                            progress_data = {
                                "type": "progress",
                                "percent": progress_pct
                            }
                            yield f"data: {json.dumps(progress_data,ensure_ascii=False)}\n\n"
                            await asyncio.sleep(0)
                    elif "output_json: saving output to" in line:
                        logger.info(f"[whisper-cli json output]: {line}")
                        json_path_match = re.search(r"output to '([^']+)'", line)
                        if json_path_match:
                            json_output_path = json_path_match.group(1)
                            logger.info(f"Detected JSON output path: {json_output_path}")
                            if json_output_path != str(temp_json_path):
                                temp_json_path = Path(json_output_path)
                    elif line:
                        logger.debug(f"[whisper-cli other]: {line}")
                
                stdout_from_communicate, stderr_str_from_communicate = process.communicate() 
                return_code = process.returncode
                stderr_text = stderr_str_from_communicate.strip() if stderr_str_from_communicate else ""

            else:
                logger.info("Using asyncio.create_subprocess_exec for whisper-cli.")
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                logger.info(f"Started whisper-cli process with PID: {process.pid}")
                full_transcription_text = ""
                segment_regex = re.compile(r"\[(\d{2}:\d{2}:\d{2}\.\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(.*)")

                while True:
                    stdout_line_bytes = await process.stdout.readline()
                    if not stdout_line_bytes:
                        logger.info("Whisper stdout stream ended.")
                        break

                    line = stdout_line_bytes.decode('utf-8', errors='replace').strip()
                    
                    match = segment_regex.match(line)
                    if match:
                        start_time_str = match.group(1)
                        end_time_str = match.group(2)
                        segment_text = match.group(3).strip()

                        if segment_text:
                            logger.info(f"[whisper-cli segment]: [{start_time_str} --> {end_time_str}] {segment_text}")
                            full_transcription_text += segment_text + " "

                            sse_data = {
                                "type": "segment",
                                "start_time": start_time_str,
                                "end_time": end_time_str,
                                "text": segment_text
                            }
                            yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"
                            await asyncio.sleep(0)
                        else:
                           logger.debug(f"[whisper-cli] Matched regex but segment text was empty: {line}")
                    elif "whisper_print_progress_callback" in line:
                        logger.info(f"[whisper-cli progress]: {line}")
                        progress_match = re.search(r"progress\s*=\s*(\d+)%", line)
                        if progress_match:
                            progress_pct = progress_match.group(1)
                            progress_data = {
                                "type": "progress",
                                "percent": progress_pct
                            }
                            yield f"data: {json.dumps(progress_data, ensure_ascii=False)}\n\n"
                            await asyncio.sleep(0)
                    elif "output_json: saving output to" in line:
                        logger.info(f"[whisper-cli json output]: {line}")
                        json_path_match = re.search(r"output to '([^']+)'", line)
                        if json_path_match:
                            json_output_path = json_path_match.group(1)
                            logger.info(f"Detected JSON output path: {json_output_path}")
                            if json_output_path != str(temp_json_path):
                                temp_json_path = Path(json_output_path)
                    elif line:
                        logger.debug(f"[whisper-cli other]: {line}")

                await process.wait()
                return_code = process.returncode
                
                stderr_bytes = await process.stderr.read()
                stderr_text = stderr_bytes.decode('utf-8', errors='replace').strip()

            if stderr_text:
                if return_code == 0:
                    logger.warning(f"Whisper stderr output (exit code 0):\n{stderr_text}")
                else:
                    logger.error(f"Whisper stderr output (exit code {return_code}):\n{stderr_text}")

            if use_json and temp_json_path and temp_json_path.exists():
                try:
                    logger.info(f"Reading JSON output from {temp_json_path}")
                    with open(temp_json_path, 'r', encoding='utf-8') as f:
                        json_content = f.read()
                        json_result = json.loads(json_content)
                    
                    logger.debug(f"Parsed JSON result keys: {list(json_result.keys())}")
                    
                    segments_key = "segments" if "segments" in json_result else "transcription"

                    if segments_key in json_result and isinstance(json_result[segments_key], list):
                        json_segments = json_result[segments_key]
                        processed_segments_text = []
                        for segment in json_segments:
                            segment_text_value = segment.get("text", "").strip()
                            if segment_text_value:
                                processed_segments_text.append(segment_text_value)
                                segment_data = {
                                    "type": "json_segment",
                                    "start": segment.get("t0", segment.get("from", 0)),
                                    "end": segment.get("t1", segment.get("to", 0)),
                                    "text": segment_text_value
                                }
                                yield f"data: {json.dumps(segment_data, ensure_ascii=False)}\n\n"
                                await asyncio.sleep(0)
                        
                        if processed_segments_text:
                             full_transcription_text = " ".join(processed_segments_text)
                except Exception as json_err:
                    logger.error(f"Error reading/parsing JSON output: {json_err}")
                    logger.error(traceback.format_exc())
            
            if return_code != 0:
                error_data = {
                    "type": "error",
                    "message": "Transcription process failed.",
                    "exit_code": return_code,
                    "stderr": stderr_text
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            else:
                elapsed_time = time.time() - start_time
                final_data = {
                    "type": "done",
                    "message": "Transcription complete.",
                    "full_text": full_transcription_text.strip(),
                    "elapsed_time": f"{elapsed_time:.2f}s",
                    "has_json": json_result is not None
                }
                logger.info(f"Transcription completed successfully in {elapsed_time:.2f}s")
                yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"

        except asyncio.CancelledError:
            logger.warning("Transcription stream cancelled by client.")
            if process and process.returncode is None:
                if hasattr(process, 'terminate'):
                    try:
                        logger.info(f"Terminating whisper process {process.pid} due to cancellation.")
                        process.terminate()
                        try:
                            await asyncio.wait_for(process.wait(), timeout=5.0)
                            logger.info(f"Process {process.pid} terminated gracefully.")
                        except asyncio.TimeoutError:
                            logger.warning(f"Process {process.pid} did not terminate, killing.")
                            process.kill()
                            await process.wait()
                            logger.info(f"Process {process.pid} killed.")
                    except Exception as term_err:
                        logger.error(f"Error during process termination: {term_err}")
                elif isinstance(process, subprocess.Popen):
                    try:
                        logger.info(f"Terminating whisper process {process.pid} (subprocess.Popen) due to cancellation.")
                        process.terminate()
                        process.wait(timeout=5.0)
                        logger.info(f"Process {process.pid} (subprocess.Popen) terminated gracefully.")
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Process {process.pid} (subprocess.Popen) did not terminate, killing.")
                        process.kill()
                        process.wait()
                        logger.info(f"Process {process.pid} (subprocess.Popen) killed.")
                    except Exception as term_err:
                        logger.error(f"Error during subprocess.Popen termination: {term_err}")

            cancel_data = {
                "type": "cancelled",
                "message": "Transcription cancelled by client."
            }
            yield f"data: {json.dumps(cancel_data, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"Error during transcription streaming: {e}")
            logger.error(traceback.format_exc())
            error_data = {
                "type": "error",
                "message": f"Server error during transcription: {str(e)}"
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            
        finally:
            for path_to_clean in [temp_path, temp_json_path]:
                if path_to_clean and path_to_clean.exists():
                    try:
                        os.unlink(path_to_clean)
                        logger.debug(f"Deleted temporary file: {path_to_clean}")
                    except Exception as del_err:
                        logger.warning(f"Failed to delete temporary file {path_to_clean}: {del_err}")
            
            logger.info("Transcription streaming completed and resources cleaned up.")

    return StreamingResponse(
        sse_transcription_generator(),
        media_type="text/event-stream; charset=utf-8"
    )