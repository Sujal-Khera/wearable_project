import time
import numpy as np
from typing import List, Dict

from config import DEFAULT_CONFIG
from src.chunker import load_document
from src.embedder import Embedder
from src.search import VectorSearch


class PrivateRAGPipeline:

    def __init__(self, config=None):

        self.config = config or DEFAULT_CONFIG

        self.chunks = []

        self.embedder = None
        self.searcher = None

    def ingest_document(self, path):

        print(f"Ingesting {path}")

        new_chunks = load_document(
            path,
            self.config.chunk
        )

        self.chunks.extend(new_chunks)

        return self

    def generate_embeddings(self):

        print("Generating embeddings...")

        self.embedder = Embedder(
            self.config.embedding
        )

        self.embeddings = self.embedder.embed_documents(
            self.chunks
        )

        return self

    def quantize(self):

        return self

    def encrypt_and_index(self):

        print("Creating FAISS index...")

        self.searcher = VectorSearch(
            self.embeddings
        )

        return self

    def search(self, query, top_k=3):

        query_embedding = self.embedder.embed_query(
            query
        )

        result = self.searcher.search(
            query_embedding,
            top_k
        )

        retrieved = []

        for idx, score in zip(
                result.top_k_indices,
                result.top_k_scores):

            retrieved.append({
                "text": self.chunks[idx],
                "score": score
            })

        top_score = result.top_k_scores[0]

        print(f"[RAG SCORE] {top_score:.4f}")

        return {
            "results": retrieved,
            "has_match":
                top_score >=
                self.config.relevance_threshold,
            "top_score_normalized": top_score
        }

    def print_summary(self):

        print("Chunks:", len(self.chunks))