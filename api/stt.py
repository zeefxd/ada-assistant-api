from fastapi import APIRouter, UploadFile, File
import speech_recognition as sr
import tempfile

router = APIRouter()

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), language: str = "pl-PL"):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio.seek(0)
            
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_audio.name) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio, language=language)
                
        return {"text": text, "language": language}
    except Exception as e:
        return {"error": str(e)}