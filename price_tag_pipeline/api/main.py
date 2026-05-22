import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from price_tag_pipeline.api.files import router as files_router


app = FastAPI(title="Lenta Tech Hackathon Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "PRICE_TAG_API_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://91.195.101.74:8021",
        ).split(",")
        if origin.strip()
    ],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(files_router)
