from fastapi import FastAPI
from api.routes import router
import uvicorn
from pyinstrument import Profiler
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
import logging
from contextlib import asynccontextmanager
import time

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application initialization...")
    
    try:
        from api.llm import initialize_llm
        logger.info("Starting LLM initialization...")
        llm_initialized = await initialize_llm()
        
        if llm_initialized:
            logger.info("LLM service initialized successfully - model is ready for inference")
        else:
            logger.warning("LLM service initialization failed. Some endpoints may not work properly.")
    except Exception as e:
        logger.error(f"LLM initialization error: {str(e)}")
    
    try:
        from api.stt import initialize_stt
        logger.info("Starting STT (Whisper Server) initialization...")
        stt_initialized = await initialize_stt()
        
        if stt_initialized:
            logger.info("STT service initialized successfully - ready for transcription")
        else:
            logger.warning("STT service initialization failed. Transcription endpoints may not work properly.")
    except Exception as e:
        logger.error(f"STT initialization error: {str(e)}")
    
    try:
        from api.tts import initialize_tts
        logger.info("Starting TTS initialization...")
        tts_initialized = await initialize_tts()
        
        if tts_initialized:
            logger.info("TTS service initialized successfully - ready for speech synthesis")
        else:
            logger.warning("TTS service initialization failed. Speech synthesis endpoints may not work properly.")
    except Exception as e:
        logger.error(f"TTS initialization error: {str(e)}")
    
    logger.info("Application initialization completed - ready to handle requests")    

    yield
    
    logger.info("Application shutting down...")
    try:
        from api.stt import shutdown_stt
        logger.info("Shutting down STT (Whisper Server)...")
        await shutdown_stt()
        logger.info("STT service (Whisper Server) shut down.")
    except Exception as e:
        logger.error(f"Error during STT (Whisper Server) shutdown: {str(e)}")
    
    logger.info("Application shutdown complete.")


app = FastAPI(title="Ada API", lifespan=lifespan)

app.include_router(router)

# class PyInstrumentMiddleWare(BaseHTTPMiddleware):
#     async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
#         profiler = Profiler(interval=0.001, async_mode="enabled")
#         profiler.start()
#         response = await call_next(request)
#         profiler.stop()
#         profiler.write_html("profile.html")
#         return response

# app.add_middleware(PyInstrumentMiddleWare)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)