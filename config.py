from dataclasses import dataclass, field


@dataclass
class ChunkConfig:
    chunk_size: int = 90
    chunk_overlap: int = 20
    min_chunk_length: int = 30


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize: bool = True


@dataclass
class RAGConfig:
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    top_k: int = 3
    relevance_threshold: float = 0.32
    no_match_message: str = "No relevant document context found."


DEFAULT_CONFIG = RAGConfig()