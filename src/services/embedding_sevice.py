from fastembed import TextEmbedding
import numpy as np
from typing import List


class EmbeddingService:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.model = TextEmbedding(model_name)
        self.vector_size = self._measure_dim()

    def _measure_dim(self) -> int:
        return len(list(self.model.embed(["dimension probe"]))[0])

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = list(self.model.embed(texts))
        return np.array(embeddings).astype('float32')

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]