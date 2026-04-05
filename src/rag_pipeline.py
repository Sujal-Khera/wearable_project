import time
import numpy as np
from typing import List, Dict, Optional

from new.wearable_project.config import RAGConfig, DEFAULT_CONFIG
from new.wearable_project.src.chunker import load_document
from new.wearable_project.src.embedder import Embedder
from new.wearable_project.src.quantizer import Int8Quantizer
from new.wearable_project.src.encryption import BFVEngine
from new.wearable_project.src.search import BFVSearch, SearchResult


class PrivateRAGPipeline:
    def __init__(self, config: RAGConfig = None):
        if config is None:
            config = DEFAULT_CONFIG

        self.config = config
        self.chunks: List[str] = []
        self.doc_embeddings: np.ndarray = None
        self.int_docs: np.ndarray = None

        self.embedder: Optional[Embedder] = None
        self.quantizer: Optional[Int8Quantizer] = None
        self.engine: Optional[BFVEngine] = None
        self.searcher: Optional[BFVSearch] = None

        self.is_indexed = False
        self.timings: Dict[str, float] = {}

    def _normalize_int_score(self, raw_score: float) -> float:
        if self.quantizer is None or self.doc_embeddings is None:
            return 0.0

        max_possible = self.quantizer.get_max_dot_product(self.doc_embeddings.shape[1])
        if max_possible <= 0:
            return 0.0

        return float(raw_score) / float(max_possible)

    def _build_relevance_info(self, top_score: float) -> Dict[str, float]:
        normalized = self._normalize_int_score(top_score)
        has_match = normalized >= self.config.relevance_threshold
        return {
            "top_score": float(top_score),
            "top_score_normalized": normalized,
            "relevance_threshold": float(self.config.relevance_threshold),
            "has_match": has_match,
        }

    def ingest_document(self, path: str) -> 'PrivateRAGPipeline':
        print("\n" + "=" * 60)
        print("STEP 1: DOCUMENT INGESTION")
        print("=" * 60)

        t0 = time.time()
        self.chunks = load_document(path, self.config.chunk)
        self.timings['ingestion'] = time.time() - t0
        return self

    def generate_embeddings(self) -> 'PrivateRAGPipeline':
        print("\n" + "=" * 60)
        print("STEP 2: EMBEDDING GENERATION")
        print("=" * 60)

        t0 = time.time()
        self.embedder = Embedder(self.config.embedding)
        self.doc_embeddings = self.embedder.embed_documents(self.chunks)
        self.timings['embedding'] = time.time() - t0
        return self

    def quantize(self) -> 'PrivateRAGPipeline':
        print("\n" + "=" * 60)
        print("STEP 3: INT8 QUANTIZATION")
        print("=" * 60)

        t0 = time.time()
        self.quantizer = Int8Quantizer(self.config.quantization)
        self.int_docs = self.quantizer.quantize_documents(self.doc_embeddings)
        self.timings['quantization'] = time.time() - t0

        max_dot = self.quantizer.get_max_dot_product(self.doc_embeddings.shape[1])
        print(f"\nMax possible dot product: {max_dot:,}")
        return self

    def encrypt_and_index(self) -> 'PrivateRAGPipeline':
        print("\n" + "=" * 60)
        print("STEP 4: BFV ENCRYPTION & INDEXING")
        print("=" * 60)

        t0 = time.time()

        self.engine = BFVEngine(self.config.bfv)
        self.engine.setup()

        self.searcher = BFVSearch(self.engine)
        self.searcher.index_documents(
            self.int_docs,
            mode=self.config.search_mode,
        )

        self.timings['encryption'] = time.time() - t0
        self.is_indexed = True
        return self

    def search(self, query: str, top_k: int = None) -> Dict:
        if not self.is_indexed:
            raise RuntimeError("Call encrypt_and_index() first")

        if top_k is None:
            top_k = self.config.top_k

        query_embedding = self.embedder.embed_query(query)
        int_query = self.quantizer.quantize_query(query_embedding)

        result: SearchResult = self.searcher.search(
            int_query,
            top_k=top_k,
            mode=self.config.search_mode,
        )
        result.query_text = query

        retrieved_chunks = []
        for idx in result.top_k_indices:
            retrieved_chunks.append({
                "index": idx,
                "text": self.chunks[idx],
                "score": result.all_scores[idx],
            })

        top_score = result.top_k_scores[0] if result.top_k_scores else 0.0
        relevance = self._build_relevance_info(top_score)

        message = "Relevant chunks found."
        if not relevance["has_match"]:
            message = self.config.no_match_message

        return {
            "query": query,
            "mode": self.config.search_mode,
            "top_k": top_k,
            "results": retrieved_chunks,
            "latency_ms": result.latency_ms,
            "has_match": relevance["has_match"],
            "message": message,
            "top_score": relevance["top_score"],
            "top_score_normalized": relevance["top_score_normalized"],
            "relevance_threshold": relevance["relevance_threshold"],
        }

    def print_summary(self):
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Document chunks: {len(self.chunks)}")
        print(f"Embedding dimension: {self.doc_embeddings.shape[1] if self.doc_embeddings is not None else 'N/A'}")
        print(f"Quantization scale: {self.quantizer.config.scale if self.quantizer else 'N/A'}")
        print(f"Encryption scheme: BFV")
        print(f"Search mode: {self.config.search_mode}")
        print(f"Top-k: {self.config.top_k}")
        print(f"Indexed: {'YES' if self.is_indexed else 'NO'}")

        print("\nTIMINGS:")
        for step, t in self.timings.items():
            print(f"  {step:<20} {t:.2f}s")
        print(f"  {'TOTAL':<20} {sum(self.timings.values()):.2f}s")
