from fastapi import FastAPI
from api.routes import router
import uvicorn
from pyinstrument import Profiler
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

app = FastAPI(title="Ada API")

app.include_router(router)

class PyInstrumentMiddleWare(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        profiler = Profiler(interval=0.001, async_mode="enabled")
        profiler.start()
        response = await call_next(request)
        profiler.stop()
        profiler.write_html("profile.html")
        return response

app.add_middleware(PyInstrumentMiddleWare)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)