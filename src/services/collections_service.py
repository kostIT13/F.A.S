from pathlib import Path
from typing import List, Dict, Any
import json
import os
import shutil
import numpy as np
from src.api.schemas import CollectionConfig, Point, DistanceType
from src.core.database import MetadataDB
from src.services.index_manager import IndexManager
from src.services.embedding_sevice import EmbeddingService


class CollectionService:
    def __init__(self, base_path: str = None):
        base_path = base_path or os.getenv("COLLECTIONS_PATH", "./collections")
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.embedding = EmbeddingService()

    @staticmethod
    def _validate_name(name: str):
        if not name or name in ('.', '..'):
            raise ValueError("Invalid collection name")
        if '/' in name or '\\' in name or '\x00' in name:
            raise ValueError("Collection name must not contain path separators")
        if name.startswith('.'):
            raise ValueError("Collection name must not start with a dot")

    def create_collection(self, config: CollectionConfig):
        self._validate_name(config.collection_name)
        if config.vector_size != self.embedding.vector_size:
            raise ValueError(
                f"vector_size {config.vector_size} does not match the model "
                f"dimension ({self.embedding.vector_size}). Use {self.embedding.vector_size}."
            )

        path = self.base_path / config.collection_name
        if path.exists():
            raise ValueError(f"Collection {config.collection_name} already exists")
        path.mkdir(parents=True)

        IndexManager(str(path)).create_index(config.vector_size, config.distance)
        MetadataDB(str(path))

        with open(path / "collection_info.json", 'w') as f:
            json.dump(config.model_dump(), f)

        return {"status": "created", "collection": config.collection_name}

    def get_collections(self) -> List[Dict[str, Any]]:
        result = []
        for p in self.base_path.iterdir():
            if p.is_dir() and (p / "collection_info.json").exists():
                try:
                    result.append(self.get_collection_info(p.name))
                except Exception:
                    pass
        return result

    def get_collection_info(self, name: str) -> Dict[str, Any]:
        self._validate_name(name)
        path = self.base_path / name
        if not path.exists():
            raise ValueError(f"Collection {name} not found")

        with open(path / "collection_info.json") as f:
            config = json.load(f)

        db = MetadataDB(str(path))
        return {
            "name": name,
            "vector_size": config['vector_size'],
            "distance": config['distance'],
            "points_count": db.get_points_count(),
            "index_type": config.get('index', {}).get('type', 'flat')
        }

    def delete_collection(self, name: str):
        self._validate_name(name)
        path = self.base_path / name
        if not path.exists():
            raise ValueError(f"Collection {name} not found")
        IndexManager._cache.invalidate(name)
        shutil.rmtree(path)
        return {"status": "deleted", "collection": name}

    def upsert_points(self, name: str, points: List[Point]):
        self._validate_name(name)
        path = self.base_path / name
        if not path.exists():
            raise ValueError(f"Collection {name} not found")

        texts = [p.text for p in points]
        vectors = self.embedding.embed_texts(texts)

        db = MetadataDB(str(path))
        new_internal_ids = db.upsert_points([
            {'external_id': p.id, 'text': p.text, 'payload': p.payload} for p in points
        ])

        try:
            idx = IndexManager(str(path))
            idx.load_index(use_cache=True)
            idx.add_vectors(new_internal_ids, vectors)
        except Exception as e:
            IndexManager._cache.invalidate(name)
            raise RuntimeError(
                f"Index update failed, but metadata is safe in SQLite. "
                f"Rebuild the index via POST /collections/{name}/points/rebuild"
            ) from e

        return {"status": "upserted", "count": len(points)}

    def search(self, name: str, text: str, limit: int = 10, with_payload: bool = True) -> List[Dict[str, Any]]:
        self._validate_name(name)
        path = self.base_path / name
        if not path.exists():
            raise ValueError(f"Collection {name} not found")

        query = self.embedding.embed_text(text)

        idx = IndexManager(str(path))
        idx.load_index(use_cache=True)

        distances, internal_ids = idx.search(query, limit)

        if len(internal_ids) == 0:
            return []

        db = MetadataDB(str(path))
        db_points = db.get_points([int(i) for i in internal_ids if i != -1])
        db_dict = {p['id']: p for p in db_points}

        results = []
        for i, iid in enumerate(internal_ids):
            if iid == -1:
                continue
            p = db_dict.get(int(iid))
            if p is None or p.get('deleted', False):
                continue  
            item = {'id': p['external_id'], 'score': float(distances[i])}
            if with_payload:
                item['payload'] = p.get('payload', {})
            results.append(item)

        return results[:limit]

    def delete_points(self, name: str, ids: List[int]):
        self._validate_name(name)
        path = self.base_path / name
        if not path.exists():
            raise ValueError(f"Collection {name} not found")

        db = MetadataDB(str(path))
        count = db.delete_points(ids)

        return {"status": "deleted", "count": count}

    def rebuild_index(self, name: str):
        self._validate_name(name)
        path = self.base_path / name
        if not path.exists():
            raise ValueError(f"Collection {name} not found")

        db = MetadataDB(str(path))
        info = self.get_collection_info(name)

        active = db.get_active_points()
        if not active:
            IndexManager(str(path)).create_index(info['vector_size'], DistanceType(info['distance']))
            return {"status": "rebuilt", "count": 0}

        texts = [p['text'] for p in active]
        ids = [p['id'] for p in active]

        vectors = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            vectors.append(self.embedding.embed_texts(texts[i:i + batch_size]))
        vectors = np.vstack(vectors)

        idx = IndexManager(str(path))
        idx.create_index(info['vector_size'], DistanceType(info['distance']))
        idx.add_vectors(ids, vectors)

        return {"status": "rebuilt", "count": len(ids)}

    def cache_stats(self):
        return IndexManager._cache.stats()

    def clear_cache(self):
        IndexManager._cache._cache.clear()
        return {"status": "cache_cleared"}