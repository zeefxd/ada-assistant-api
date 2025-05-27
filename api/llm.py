import os
import sys
import logging
import time
import gc
import traceback
import ollama
import re
    
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
from api.command_detector import CommandDetector, CommandType
from api.spotify_handler import SpotifyHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

router = APIRouter()

class GenerateRequest(BaseModel):
    prompt: str

model_name = "gemma3:4b"
model_initialized = False
model_available = False

async def initialize_llm():
    """
    Initializes the LLM service by checking for model availability, downloading if needed,
    and performing a warm-up inference to fully load the model into memory.
    Called at application startup rather than during endpoint calls.
    
    Returns:
        bool: True if initialization was successful
    """
    global model_initialized, model_available
    
    try:
        logger.info(f"Initializing LLM service with model: {model_name}")
        
        models = ollama.list()
        logger.debug(f"Response from ollama.list(): {models}")
        
        model_exists = False
        
        if 'models' in models:
            models_list = models['models']
            
            for model in models_list:
                if 'name' in model and model['name'] == model_name:
                    model_exists = True
                    break
                elif 'model' in model and model['model'] == model_name:
                    model_exists = True
                    break
        else:
            models_list = models
            model_exists = any(
                (model.get('name') == model_name or model.get('model') == model_name)
                for model in models_list if isinstance(model, dict)
            )
        
        if not model_exists:
            logger.warning(f"Model {model_name} not found in Ollama. Starting download...")
            
            try:
                logger.info(f"Downloading model {model_name}. This may take a while...")
                response = ollama.pull(model_name)
                logger.info(f"Model {model_name} downloaded successfully!")
                model_available = True
            except Exception as pull_error:
                logger.error(f"Failed to download model {model_name}: {str(pull_error)}")
                model_available = False
                model_initialized = True
                return False
        else:
            logger.info(f"Model {model_name} is already available in Ollama.")
            model_available = True
        
        if model_available:
            try:
                logger.info("Performing warm-up inference to load model into memory...")
                warm_up_start = time.time()
                
                warm_up_response = ollama.generate(
                    model=model_name,
                    prompt="Cześć, jak się masz?",
                    options={"num_predict": 10}
                )
                
                warm_up_time = time.time() - warm_up_start
                logger.info(f"Model warm-up completed in {warm_up_time:.2f}s")
                
                del warm_up_response
                gc.collect()
                
            except Exception as warm_up_error:
                logger.error(f"Model warm-up failed: {str(warm_up_error)}")
        
        model_initialized = True
        return True
    except Exception as e:
        error_details = str(e) + "\n" + traceback.format_exc()
        logger.error(f"Ollama connection error during initialization: {error_details}")
        model_initialized = True
        model_available = False
        return False
            
def get_model():
    """
    Gets the initialized model status.
    
    Returns:
        bool: True if the model is available
        
    Raises:
        HTTPException: When model is not initialized or not available
    """
    if not model_initialized:
        logger.error(f"LLM service not initialized. This should have happened at startup.")
        raise HTTPException(
            status_code=500, 
            detail=f"LLM service not properly initialized. Please restart the server."
        )
        
    if not model_available:
        raise HTTPException(
            status_code=500, 
            detail=f"Model {model_name} is not available. Check server logs for details."
        )
    
    return True


@router.post("/generate")
async def generate_response(
    request: GenerateRequest, 
    spotify_token: str = Header(None, alias="spotify-token"),
    x_spotify_token: str = Header(None, alias="x-spotify-token"),
    authorization: str = Header(None),
    spotify_auth: str = Header(None, alias="spotify-auth")
):
    """
    Generates a response based on the provided prompt, handling both commands and LLM generation.
    
    Args:
        request (GenerateRequest): The request object containing the user prompt
        spotify_token (str): Optional Spotify access token (using kebab-case header)
        x_spotify_token (str): Optional Spotify access token (using X- prefix header)
        authorization (str): Optional Authorization header that might contain the token
        spotify_auth (str): Alternative Spotify auth header
        
    Returns:
        dict: Response data containing either command results or LLM-generated text
        
    Raises:
        HTTPException: When generation fails or errors occur during processing
    """
    try:
        logger.info(f"Received request: '{request.prompt}'")
        
        effective_token = None
        if x_spotify_token:
            logger.info("Using X-Spotify-Token header")
            effective_token = x_spotify_token
        elif spotify_token:
            logger.info("Using spotify-token header")
            effective_token = spotify_token
        elif spotify_auth:
            logger.info("Using spotify-auth header")
            effective_token = spotify_auth
        elif authorization and authorization.startswith("Bearer ") and "spotify" in request.prompt.lower():
            logger.info("Using Authorization Bearer token as Spotify token")
            effective_token = authorization.replace("Bearer ", "")
        
        if effective_token:
            token_preview = effective_token[:10] + "..." if len(effective_token) > 10 else effective_token
            logger.info(f"Received Spotify token: {token_preview}")
        
        command_detector = CommandDetector()
        is_command, cmd_type, params = command_detector.detect_command(request.prompt)
        
        logger.info(f"Command detection results: is_command={is_command}, cmd_type={cmd_type}, params={params}")
        
        if is_command and cmd_type == CommandType.MUSIC:
            command_info = command_detector.execute_command(cmd_type, params)
            logger.info(f"Music command detected with parameters: {params}")
            
            if params.get("targetPlatform", "").lower() == "spotify" and effective_token:
                spotify_handler = SpotifyHandler(access_token=effective_token)
                spotify_result = await spotify_handler.execute_command(params)
                
                logger.info(f"Spotify command result: {spotify_result}")
                
                if "message" in spotify_result:
                    command_info["user_message"] = spotify_result["message"]
                
                command_info["spotify_result"] = spotify_result
            elif params.get("targetPlatform", "").lower() == "spotify" and not effective_token:
                logger.warning("Spotify command detected but no Spotify token provided")
                command_info["user_message"] = "Aby sterować odtwarzaczem muzyki, musisz najpierw połączyć konto Spotify w aplikacji."
                command_info["spotify_result"] = {"success": False, "message": "No Spotify token provided"}
          
            return {
                "is_command": True,
                "command_type": "music",
                "command_data": command_info,
                "response": clean_text_for_tts(command_info["user_message"])
            }
        
        logger.info("No command detected, continuing with standard response generation")
        
        get_model()
        
        system_prompt = """Jesteś pomocną asystentką o imieniu Ada. Odpowiadasz zawsze poprawną polszczyzną.
        Formułujesz odpowiedzi jako krótkie, gramatycznie poprawne, jasne i zwięzłe zdania.
        Na pytania odpowiadasz konkretnie, poprawnie i zgodnie z prawdą.
        Nie zgaduj informacji, których nie znasz - powiedz wtedy, że nie wiesz.
        Twoje odpowiedzi są zawsze pomocne, dokładne i praktyczne.
        Unikaj formatowania Markdown w swoich odpowiedziach."""
        
        start_time = time.time()
        
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': request.prompt}
            ],
            options={
                'temperature': 0.6,
                'top_p': 0.85
            }
        )
        
        generation_time = time.time() - start_time
        logger.info(f"Response generation took: {generation_time:.2f}s")
        
        response_text = response['message']['content']
        
        if not response_text:
            response_text = "Przepraszam, nie wiem, jak odpowiedzieć na to pytanie."
        
        cleaned_text = clean_text_for_tts(response_text)
        
        return {
            "is_command": False,
            "response": cleaned_text
        }
    
    except Exception as e:
        error_details = str(e) + "\n" + traceback.format_exc()
        logger.error(f"Generation error: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error: {error_details}")


def clean_text_for_tts(text):
    """
    Cleans text from Markdown formatting and special characters for text-to-speech compatibility.
    
    Args:
        text (str): The text to be cleaned
        
    Returns:
        str: Cleaned text suitable for TTS processing
    """
    if not text:
        return ""
    
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    def replace_bullet_item(match):
        item = match.group(1).strip()
        if item and item[-1] not in '.!?:;':
            return f". {item}"
        return f" {item}"
        
    text = re.sub(r'\n\s*\*\s+(.*?)(?=\n\s*\*\s+|\n\n|\Z)', replace_bullet_item, text, flags=re.DOTALL)
    
    def replace_numbered_point(match):
        item = match.group(2).strip()
        if item and item[-1] not in '.!?:;':
            return f". {item}"
        return f" {item}"
    
    numbered_list_pattern = r'\n\s*(\d+)\.\s+(.*?)(?=\n\s*\d+\.\s+|\n\n|\Z)'
    text = re.sub(numbered_list_pattern, replace_numbered_point, text, flags=re.DOTALL)
    
    def replace_paragraph_break(match):
        para = match.group(1).strip()
        if para and para[-1] not in '.!?:;':
            return f"{para}. "
        return f"{para} "
        
    text = re.sub(r'([^\n]*?)\n\s*\n', replace_paragraph_break, text)
    
    text = re.sub(r'\n', ' ', text)
    
    text = re.sub(r'\.\.+', '.', text)
    text = re.sub(r'\.\s*\.', '.', text)
    
    text = re.sub(r'\s+\.', '.', text)
    text = re.sub(r'\.\s+', '. ', text)
    
    text = re.sub(r'\s{2,}', ' ', text)
    
    text = re.sub(r'#+\s+', '', text)
    text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)
    
    return text.strip()

@router.get("/info")
async def get_model_info():
    """
    Retrieves and returns basic information about the currently used LLM model.
    
    Args:
        None
        
    Returns:
        dict: Model information including name, size, modification date and engine type
              or error message if model information cannot be retrieved
    """
    try:
        models = ollama.list()
        model_info = None
        
        if 'models' in models:
            for model in models.get('models', []):
                if model.get('name') == model_name or model.get('model') == model_name:
                    model_info = model
                    break
        else:
            for model in models:
                if model.get('name') == model_name or model.get('model') == model_name:
                    model_info = model
                    break
        
        if model_info:
            name_key = 'name' if 'name' in model_info else 'model'
            return {
                "name": model_info.get(name_key, model_name),
                "size": model_info.get('size', 'Unknown'),
                "modified": model_info.get('modified', 'Unknown'),
                "engine": "Ollama"
            }
        else:
            return {"error": f"Model {model_name} not found in Ollama"}
    except Exception as e:
        error_details = str(e) + "\n" + traceback.format_exc()
        return {"error": error_details}

@router.get("/gpu_test")
async def test_gpu():
    """
    Tests GPU functionality with Ollama by running a simple generation task.
    
    Args:
        None
        
    Returns:
        dict: Test results including Ollama availability, GPU information, and performance metrics
              or error details if the test fails
    """
    try:
        results = {
            "ollama_available": True,
            "gpu_info": "Check Ollama logs to see GPU usage information",
            "performance_test": None
        }
        
        try:
            start_time = time.time()
            
            response = ollama.generate(
                model=model_name,
                prompt="Cześć, jak się masz?",
                options={"num_predict": 50}
            )
            
            end_time = time.time()
            
            results["performance_test"] = {
                "time_taken": end_time - start_time,
                "tokens_generated": len(response.get('response', '').split()),
                "model": model_name
            }
                
        except Exception as e:
            results["error"] = f"Test error: {str(e)}\n{traceback.format_exc()}"
        
        return results
    except Exception as e:
        return {"error": f"General error: {str(e)}\n{traceback.format_exc()}"}