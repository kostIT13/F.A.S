from fastapi import APIRouter
from src.services.collections_service import CollectionService
from src.api.schemas import CollectionConfig, UpsertRequest, SearchRequest, DeleteRequest, SearchResult


router = APIRouter(prefix="/collections", tags=["Collections"])
service = CollectionService()


@router.post("")
async def collection_create(config: CollectionConfig):
    return service.create_collection(config)


@router.get("")
async def collections_list():
    return service.get_collections()


@router.get("/{collection_name}")
async def get_collection(name: str):
    return service.get_collection_info(name)


@router.delete("/{collection_name}")
async def collection_delete(name: str):
    return service.delete_collection(name)


@router.put("/{collection_name}/points")
async def upsert_point(name: str, request: UpsertRequest):
    return service.upsert_points(name, request.points)


@router.post("/{collection_name}/points/search")
async def search(name: str, request: SearchRequest):
    results = service.search(name, request.text, request.limit, request.with_payload)
    return [SearchResult(**r) for r in results]


@router.post("/{collection_name}/points/delete")
async def point_delete(name: str, request: DeleteRequest):
    return service.delete_points(name, request.ids)


@router.post("/{collection_name}/points/rebuild")
async def rebuild(name: str):
    return service.rebuild_index(name)