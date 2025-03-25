from fastapi import FastAPI
from api.routes import router
import uvicorn

app = FastAPI(title="Ada API")

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)