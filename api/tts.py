from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel
from gtts import gTTS
import tempfile

router = APIRouter()

class TextToSpeech(BaseModel):
    text: str
    language: str = "pl"

@router.post("/synthesize")
async def synthesize_speech(request: TextToSpeech):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts = gTTS(text=request.text, lang=request.language)
            tts.save(temp_audio.name)
            return FileResponse(
                temp_audio.name,
                media_type="audio/mpeg",
                headers={"Content-Disposition": "attachment; filename=speech.mp3"}
            )
    except Exception as e:
        return {"error": str(e)}