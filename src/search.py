import time
import numpy as np
from typing import List
from dataclasses import dataclass, field
from new.wearable_project.src.encryption import BFVEngine


@dataclass
class SearchResult:
    method: str = ""
    query_text: str = ""
    top_k_indices: List[int] = field(default_factory=list)
    top_k_scores: List[float] = field(default_factory=list)
    all_scores: List[float] = field(default_factory=list)
    latency_ms: float = 0.0
    crypto_error: float = 0.0


class BFVSearch:
    def __init__(self, engine: BFVEngine):
        self.engine = engine
        self.enc_docs: List[bytes] = []
        self.int_docs: np.ndarray = None
        self.n_docs: int = 0

    def index_documents(self, int_docs: np.ndarray, mode: str = "ct_ct"):
        self.int_docs = int_docs
        self.n_docs = len(int_docs)

        if mode == "ct_ct":
            self.enc_docs = self.engine.encrypt_documents(int_docs)
        else:
            raise ValueError("Only ct_ct mode is supported in this setup")

    def search(self, int_query: np.ndarray, mode: str = "ct_ct", top_k: int = 3) -> SearchResult:
        t0 = time.time()

        enc_q = self.engine.encrypt_vector(int_query)

        enc_score_list = []
        for i in range(self.n_docs):
            score_bytes = self.engine.server_dot_product_ct_ct(
                enc_q, self.enc_docs[i]
            )
            enc_score_list.append(score_bytes)

        scores = []
        for s_bytes in enc_score_list:
            scores.append(self.engine.decrypt_score(s_bytes))

        latency = (time.time() - t0) * 1000

        sorted_idx = list(np.argsort(scores)[::-1])
        top_idx = sorted_idx[:top_k]
        top_scores = [float(scores[i]) for i in top_idx]

        return SearchResult(
            method=f"BFV-int8-{mode}",
            top_k_indices=top_idx,
            top_k_scores=top_scores,
            all_scores=[float(s) for s in scores],
            latency_ms=latency,
            crypto_error=0.0,
        )
