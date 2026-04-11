import numpy as np
import time
import faiss
from dataclasses import dataclass
from typing import List


@dataclass
class SearchResult:

    query_text: str = ""
    top_k_indices: List[int] = None
    top_k_scores: List[float] = None
    latency_ms: float = 0.0


class VectorSearch:

    def __init__(self, embeddings):

        self.embeddings = embeddings

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        faiss.normalize_L2(self.embeddings)

        self.index.add(self.embeddings)

    def search(self, query_embedding, top_k=3):

        start = time.time()

        query = np.array([query_embedding])

        faiss.normalize_L2(query)

        scores, indices = self.index.search(
            query,
            top_k
        )

        latency = (time.time() - start) * 1000

        return SearchResult(
            top_k_indices=indices[0].tolist(),
            top_k_scores=scores[0].tolist(),
            latency_ms=latency
        )