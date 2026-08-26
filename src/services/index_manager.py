import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from collections import OrderedDict
import threading
import time
from src.api.schemas import DistanceType


class IndexCache:
    def __init__(self, max_size: int = 2):
        self.max_size = max_size
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {'hits': 0, 'misses': 0}
    
    def get(self, name: str, path: str) -> Optional[faiss.Index]:
        with self._lock:
            if name in self._cache:
                self._cache.move_to_end(name)
                entry = self._cache[name]
                index_path = Path(path) / "index.faiss"
                if index_path.exists() and entry['mtime'] == index_path.stat().st_mtime:
                    self._stats['hits'] += 1
                    return entry['index']
                del self._cache[name]
            self._stats['misses'] += 1
            return None
    
    def put(self, name: str, index: faiss.Index, path: str):
        with self._lock:
            index_path = Path(path) / "index.faiss"
            mtime = index_path.stat().st_mtime if index_path.exists() else time.time()
            if name in self._cache:
                self._cache.move_to_end(name)
                self._cache[name] = {'index': index, 'mtime': mtime}
                return
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[name] = {'index': index, 'mtime': mtime}
    
    def invalidate(self, name: str):
        with self._lock:
            if name in self._cache:
                del self._cache[name]
                return True
            return False
    
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._stats['hits'] + self._stats['misses']
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': f"{self._stats['hits']/total*100:.1f}%" if total > 0 else "0%",
                'collections': list(self._cache.keys())
            }

class IndexManager:
    _cache = IndexCache(max_size=2)
    
    def __init__(self, collection_path: str):
        self.path = Path(collection_path)
        self.name = self.path.name
        self.index = None
        self._normalize = False
    
    def create_index(self, vector_size: int, distance: DistanceType):
        metric = faiss.METRIC_INNER_PRODUCT if distance in [DistanceType.COSINE, DistanceType.DOT] else faiss.METRIC_L2
        self._normalize = distance == DistanceType.COSINE
        
        flat_index = faiss.IndexFlatIP(vector_size) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(vector_size)
        self.index = faiss.IndexIDMap2(flat_index)
        
        self._save_config(vector_size, distance)
        self._save_index()
        self._cache.invalidate(self.name)
    
    def load_index(self, use_cache: bool = True) -> faiss.Index:
        if not (self.path / "index.faiss").exists():
            raise FileNotFoundError(f"Index not found: {self.path}")
        
        if use_cache:
            cached = self._cache.get(self.name, str(self.path))
            if cached is not None:
                self.index = cached
                self._load_config()
                return self.index
        
        self.index = faiss.read_index(str(self.path / "index.faiss"))
        self._load_config()
        if use_cache:
            self._cache.put(self.name, self.index, str(self.path))
        return self.index
    
    def add_vectors(self, ids: List[int], vectors: np.ndarray):
        if self.index is None:
            raise RuntimeError("Index not loaded")
        if self._normalize:
            faiss.normalize_L2(vectors)
        self.index.add_with_ids(vectors, np.array(ids).astype('int64'))
        self._save_index()
        self._cache.invalidate(self.name)
    
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("Index not loaded")
        if self._normalize:
            faiss.normalize_L2(query.reshape(1, -1))
            query = query.reshape(1, -1)
        return self.index.search(query.reshape(1, -1), k)
    
    def remove_ids(self, ids: List[int]):
        if self.index is None:
            raise RuntimeError("Index not loaded")
        if hasattr(self.index, 'remove_ids'):
            selector = faiss.IDSelectorArray(np.array(ids).astype('int64'))
            self.index.remove_ids(selector)
            self._save_index()
            self._cache.invalidate(self.name)
            return True
        return False
    
    def _save_index(self):
        faiss.write_index(self.index, str(self.path / "index.faiss"))
    
    def _save_config(self, vector_size: int, distance: DistanceType):
        with open(self.path / "config.json", 'w') as f:
            json.dump({'vector_size': vector_size, 'distance': distance.value, 'normalize': self._normalize}, f)
    
    def _load_config(self):
        with open(self.path / "config.json") as f:
            cfg = json.load(f)
            self._normalize = cfg.get('normalize', False)