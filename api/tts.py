from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tempfile
import os
from pathlib import Path

router = APIRouter()

class TextToSpeech(BaseModel):
    text: str
    language: str = "pl"

# https://github.com/coqui-ai/TTS
tts_model = None
models_dir = Path(__file__).parent.parent / "model"

def get_tts_model():
    global tts_model
    if tts_model is None:
        try:
            os.environ["TTS_HOME"] = str(models_dir)
            models_dir.mkdir(exist_ok=True)
            print(f"Ładowanie modelu VITS do {models_dir}...")
            
            from TTS.api import TTS
            tts_model = TTS(
                model_name="tts_models/pl/mai_female/vits", 
                progress_bar=False
            )
            print("Załadowano model VITS!")
            
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Nie można załadować modelu TTS: {str(e)}."
            )
    return tts_model

@router.post("/synthesize")
async def synthesize_speech(request: TextToSpeech):
    try:
        tts = get_tts_model()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            output_path = temp_audio.name
        
        print("Generowanie mowy...")
        tts.tts_to_file(text=request.text, file_path=output_path)
        return FileResponse(
            output_path,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
    except Exception as e:
        import traceback
        error_details = str(e) + "\n" + traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Błąd: {error_details}")