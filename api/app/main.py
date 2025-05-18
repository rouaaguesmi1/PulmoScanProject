from fastapi import FastAPI
from app.api.routes import router as api_router

app = FastAPI(title="PneumaTect API", version="0.1.0")


app.include_router(api_router, prefix="/api")
