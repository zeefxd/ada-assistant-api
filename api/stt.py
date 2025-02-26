from fastapi import APIRouter, UploadFile, File
from vosk import Model, KaldiRecognizer
import tempfile
import json
import wave

router = APIRouter()

# Ścieżka do modelu https://alphacephei.com/vosk/models/vosk-model-small-pl-0.22.zip
MODEL_PATH = "./api/vosk"

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(default=...), language: str = "pl-PL"):
    try:
        model = Model(MODEL_PATH)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio.flush()
            
            wf = wave.open(temp_audio.name, "rb")
            
            recognizer = KaldiRecognizer(model, wf.getframerate())
            
            result = ""
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    part_result = json.loads(recognizer.Result())
                    if "text" in part_result:
                        result += part_result["text"] + " "
            
            final_result = json.loads(recognizer.FinalResult())
            if "text" in final_result:
                result += final_result["text"]
                
        return {"text": result.strip(), "language": language}
    except Exception as e:
        return {"error": str(e)}