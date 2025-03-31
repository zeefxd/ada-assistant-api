import os
import sys
import logging
import time
import gc
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from dotenv import load_dotenv
from api.command_detector import CommandDetector, CommandType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

router = APIRouter()

class GenerateRequest(BaseModel):
    prompt: str

model = None
tokenizer = None
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_dir = Path(__file__).parent.parent / "model" / "tinyllama"
            
HF_TOKEN = os.getenv("HF_TOKEN")

def get_model():
    """Inicjalizuje model LLM z obsługą DirectML."""
    global model, tokenizer
    
    if model is None:
        try:
            model_dir.mkdir(exist_ok=True, parents=True)
            
            # Sprawdzanie DirectML
            dml_available = False
            dml_device = None
            try:
                import torch_directml
                dml_device = torch_directml.device()
                dml_available = True
                logger.info(f"DirectML aktywny na urządzeniu: {dml_device}")
            except Exception as e:
                logger.warning(f"Błąd DirectML: {str(e)}")
                dml_available = False
                device = torch.device("cpu")
            
            # Ładowanie tokenizera
            logger.info(f"Ładowanie tokenizera {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(model_dir),
                token=HF_TOKEN
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Konfiguracja modelu
            logger.info(f"Ładowanie modelu {model_name}...")
            
            if dml_available:
                # Optymalizacje dla DirectML
                logger.info(f"Stosowanie optymalizacji pod DirectML na urządzeniu {dml_device}")
                
                # Czyszczenie pamięci przed ładowaniem
                gc.collect()
                
                # Ładowanie modelu
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  
                    low_cpu_mem_usage=True,
                    cache_dir=str(model_dir),
                    token=HF_TOKEN
                )
                
                # Przenoszenie na DirectML
                logger.info(f"Przenoszenie modelu na urządzenie: {dml_device}")
                model = model.to(dml_device)
                
                # Weryfikacja urządzenia
                weight_device = next(model.parameters()).device
                logger.info(f"Model jest na urządzeniu: {weight_device}")
                
                # Optymalizacje pamięci
                model.eval()
                
                # Włączenie optymalizacji DirectML
                if hasattr(torch_directml, 'enable_aten_execution'):
                    torch_directml.enable_aten_execution(True)
                    logger.info("Włączono optymalizacje DirectML")
                
                # Weryfikacja
                if "privateuseone" in str(weight_device) or "directml" in str(weight_device).lower():
                    logger.info(f"Model {model_name} załadowany na DirectML")
                else:
                    logger.warning(f"Model nie jest na DirectML, a na: {weight_device}")
            else:
                # Ładowanie na CPU
                logger.info(f"Ładowanie modelu {model_name} na CPU...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=str(model_dir),
                    token=HF_TOKEN
                )
                logger.info(f"Model {model_name} załadowany na CPU")
            
        except Exception as e:
            import traceback
            logger.error(f"Błąd inicjalizacji: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return model, tokenizer

@router.post("/generate")
async def generate_response(request: GenerateRequest):
    """Generuje odpowiedź na podstawie prompta."""
    try:
        logger.info(f"Otrzymano zapytanie: '{request.prompt}'")
        
        command_detector = CommandDetector()
        is_command, cmd_type, params = command_detector.detect_command(request.prompt)
        
        logger.info(f"Wynik detekcji poleceń: is_command={is_command}, cmd_type={cmd_type}, params={params}")
        
        if is_command:
            command_info = command_detector.execute_command(cmd_type, params)
            logger.info(f"Wykryto polecenie: {cmd_type.value if cmd_type else 'nieznane'}")
            logger.info(f"Parametry polecenia: {params}")
            
            return {
                "is_command": True,
                "command_data": command_info,
                "response": command_info["user_message"] 
            }
        
        logger.info("Brak polecenia, kontynuacja standardowego generowania odpowiedzi")
        
        # Ładowanie modelu (lub użycie już załadowanego)
        model, tokenizer = get_model()
        
        system_prompt = """Jesteś pomocnym asystentem o imieniu Ada. Odpowiadasz zawsze poprawną polszczyzną.
        Formułujesz odpowiedzi jako krótkie, gramatycznie poprawne, jasne i zwięzłe zdania.
        Na pytania odpowiadasz konkretnie, poprawnie i zgodnie z prawdą.
        Nie zgaduj informacji, których nie znasz - powiedz wtedy, że nie wiesz.
        Twoje odpowiedzi są zawsze pomocne, dokładne i praktyczne."""
        
        # (few-shot prompting)
        examples = """
        <|user|>
        Jak się nazywasz?
        <|assistant|>
        Nazywam się Ada.
        
        <|user|>
        Która jest godzina?
        <|assistant|>
        Nie mam dostępu do aktualnej godziny.
        
        <|user|>
        Co lubisz robić?
        <|assistant|>
        Lubię pomagać użytkownikom odpowiadając na ich pytania w sposób jasny i zwięzły.
        """
        
        # Pełny prompt z przykładami i aktualnym pytaniem
        formatted_prompt = f"<|system|>\n{system_prompt}\n{examples}\n<|user|>\n{request.prompt}\n<|assistant|>\n"
        
        # Tokenizacja
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Czyszczenie pamięci przed generowaniem
        gc.collect()
        
        start_time = time.time()
        
        # Generowanie odpowiedzi ze zoptymalizowanymi parametrami
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                temperature=0.6,  
                top_p=0.85,
                repetition_penalty=1.3,
                do_sample=True,
                num_beams=1,  #
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                no_repeat_ngram_size=3,
                min_length=5,  
                max_new_tokens=100 
            )
        
        
        generation_time = time.time() - start_time
        logger.info(f"Generowanie odpowiedzi zajęło: {generation_time:.2f}s")
        
        # Dekodowanie odpowiedzi
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|assistant|>" in generated_text:
            responses = generated_text.split("<|assistant|>")
            response = responses[-1].strip()
            
            if "<|user|>" in response:
                response = response.split("<|user|>")[0].strip()
        else:
            response = generated_text.replace(request.prompt, "").strip()
        
        if not response:
            response = "Przepraszam, nie wiem, jak odpowiedzieć na to pytanie."
        
        return {
            "is_command": False,
            "response": response
        }
    
    except Exception as e:
        import traceback
        error_details = str(e) + "\n" + traceback.format_exc()
        logger.error(f"Błąd generowania: {error_details}")
        raise HTTPException(status_code=500, detail=f"Błąd: {error_details}")

@router.get("/info")
async def get_model_info():
    """Zwraca podstawowe informacje o modelu LLM."""
    try:
        model, _ = get_model()
        
        # Podstawowe informacje o modelu
        device = next(model.parameters()).device
        info = {
            "name": model.config.name_or_path,
            "parameters": model.num_parameters(),
            "device": str(device),
            "directml_available": "directml" in str(device).lower()
        }
        
        try:
            import torch_directml
            info["directml_version"] = torch_directml.__version__
        except ImportError:
            pass
        
        return info
    except Exception as e:
        return {"error": str(e)}

@router.get("/gpu_test")
async def test_gpu():
    """Testuje działanie DirectML na karcie AMD."""
    try:
        import torch
        
        results = {
            "directml_available": False,
            "device_info": {},
            "performance_test": None,
            "error": None
        }
        
        try:
            import torch_directml
            dml = torch_directml.device()
            results["directml_available"] = True
            results["device_info"]["device_name"] = str(dml)
            try:
                results["device_info"]["directml_version"] = getattr(torch_directml, "__version__", "Nieznana")
            except AttributeError:
                results["device_info"]["directml_version"] = "Nieznana"
        except ImportError:
            results["error"] = "DirectML niedostępne, zainstaluj torch-directml"
            return results
        
        # Test podstawowych operacji
        try:
            size = 2000  # Rozmiar macierzy
            cpu_tensor = torch.randn(size, size)
            dml_tensor = cpu_tensor.to(dml)
            
            # Test na CPU
            start_cpu = time.time()
            for _ in range(3):
                cpu_result = torch.matmul(cpu_tensor, cpu_tensor)
            cpu_time = time.time() - start_cpu
            
            # Test na DirectML
            start_dml = time.time()
            for _ in range(3):
                dml_result = torch.matmul(dml_tensor, dml_tensor)
            
            _ = dml_result.cpu()
            dml_time = time.time() - start_dml
            
            results["performance_test"] = {
                "cpu_time": cpu_time,
                "gpu_time": dml_time,
                "speedup": cpu_time / dml_time if dml_time > 0 else 0,
                "matrix_size": size
            }
                
        except Exception as e:
            import traceback
            results["error"] = f"Błąd testu: {str(e)}\n{traceback.format_exc()}"
        
        return results
    except Exception as e:
        import traceback
        return {"error": f"Ogólny błąd: {str(e)}\n{traceback.format_exc()}"}
    
    
@router.post("/test_command_detection")
async def test_command_detection(request: GenerateRequest):
    """Testuje wykrywanie poleceń bez uruchamiania modelu LLM."""
    try:
        logger.info(f"Testowanie wykrywania poleceń dla: '{request.prompt}'")
        
        command_detector = CommandDetector()
        is_command, cmd_type, params = command_detector.detect_command(request.prompt)
        
        if is_command:
            command_info = command_detector.execute_command(cmd_type, params)
            logger.info(f"Wykryto polecenie: {cmd_type.value if cmd_type else 'nieznane'}")
            logger.info(f"Parametry: {params}")
            return {
                "detected": True,
                "command_type": cmd_type.value if cmd_type else "unknown",
                "parameters": params,
                "esp32_info": command_info,
                "user_message": command_info["user_message"]
            }
        else:
            logger.info("Nie wykryto polecenia")
            return {
                "detected": False,
                "prompt": request.prompt
            }
    except Exception as e:
        import traceback
        error_details = str(e) + "\n" + traceback.format_exc()
        logger.error(f"Błąd testowania poleceń: {error_details}")
        return {"error": error_details}