from dataclasses import dataclass, field


@dataclass
class ChunkConfig:
    chunk_size: int = 120
    chunk_overlap: int = 30
    min_chunk_length: int = 40


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize: bool = True


@dataclass
class QuantizationConfig:
    scale: int = 120


@dataclass
class BFVConfig:
    poly_modulus_degree: int = 8192
    plain_mod_bits: int = 20


@dataclass
class RAGConfig:
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    bfv: BFVConfig = field(default_factory=BFVConfig)

    search_mode: str = "ct_ct"
    top_k: int = 3
    relevance_threshold: float = 0.02
    no_match_message: str = "No relevant document context found."


DEFAULT_CONFIG = RAGConfig()
