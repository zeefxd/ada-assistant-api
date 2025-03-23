from fastapi import APIRouter
from api import stt, tts, llm

router = APIRouter()

router.include_router(stt.router, prefix="/stt", tags=["Speech-to-Text"])
router.include_router(tts.router, prefix="/tts", tags=["Text-to-Speech"])
router.include_router(llm.router, prefix="/llm", tags=["Language-Model"])