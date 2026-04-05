import numpy as np
from config import QuantizationConfig


class Int8Quantizer:
    def __init__(self, config: QuantizationConfig = None):
        if config is None:
            config = QuantizationConfig()
        self.config = config
        self.scale_factor: float = None
        self.global_max: float = None

    def fit(self, embeddings: np.ndarray) -> 'Int8Quantizer':
        self.global_max = np.max(np.abs(embeddings))
        self.scale_factor = self.config.scale / self.global_max

        print("  Quantizer fitted:")
        print(f"  Global max absolute value: {self.global_max:.6f}")
        print(f"  Scale factor: {self.scale_factor:.2f}")
        print(f"  Integer range: [-{self.config.scale}, +{self.config.scale}]")
        return self

    def quantize(self, embeddings: np.ndarray) -> np.ndarray:
        if self.scale_factor is None:
            raise RuntimeError("Call fit() first with document embeddings")

        quantized = np.round(embeddings * self.scale_factor).astype(np.int64)
        quantized = np.clip(quantized, -self.config.scale, self.config.scale)
        return quantized

    def quantize_documents(self, doc_embeddings: np.ndarray) -> np.ndarray:
        self.fit(doc_embeddings)
        int_docs = self.quantize(doc_embeddings)

        print(f"  Documents quantized: {int_docs.shape}")
        print(f"  Value range: [{int_docs.min()}, {int_docs.max()}]")
        return int_docs

    def quantize_query(self, query_embedding: np.ndarray) -> np.ndarray:
        if self.scale_factor is None:
            raise RuntimeError("Call fit() or quantize_documents() first")

        int_query = np.round(query_embedding * self.scale_factor).astype(np.int64)
        int_query = np.clip(int_query, -self.config.scale, self.config.scale)
        return int_query

    def get_max_dot_product(self, dim: int) -> int:
        return dim * self.config.scale * self.config.scale
