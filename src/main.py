from fastapi import FastAPI
from src.api.router import router as collections_router
from src.api.cache_router import router as cache_router


app = FastAPI(title="FAISS rag service")


app.include_router(collections_router)
app.include_router(cache_router)