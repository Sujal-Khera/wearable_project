import numpy as np
from sentence_transformers import SentenceTransformer
from config import EmbeddingConfig


class Embedder:

    def __init__(self, config: EmbeddingConfig):

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            config.model_name
        )

        print("Embedding model loaded")

    def embed_documents(self, chunks):

        embeddings = self.model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return np.array(
            embeddings,
            dtype="float32"
        )

    def embed_query(self, query):

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        return embedding[0].astype("float32")