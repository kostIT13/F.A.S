from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum


class DistanceType(str, Enum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLID = "euclid"

class CollectionConfig(BaseModel):
    collection_name: str
    vector_size: int
    distance: DistanceType
    index: Dict[str, str] = {"type": "flat"}

class Point(BaseModel):
    id: int
    text: str
    payload: Optional[Dict[str, Any]] = {}

class UpsertRequest(BaseModel):
    points: List[Point]

class SearchRequest(BaseModel):
    text: str
    limit: int = 10
    with_payload: bool = True

class DeleteRequest(BaseModel):
    ids: List[int]

class SearchResult(BaseModel):
    id: int
    score: float
    payload: Optional[Dict[str, Any]] = None