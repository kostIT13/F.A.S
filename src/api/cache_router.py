from fastapi import APIRouter
from src.services.collections_service import CollectionService


router = APIRouter(prefix="/cache")
service = CollectionService()


@router.get("/stats")
def get_cache():
    return service.cache_stats()


@router.post("/clear")
def cache_clear():
    return service.clear_cache