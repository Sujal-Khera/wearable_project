import time
import numpy as np
from typing import List
import tenseal as ts
from new.wearable_project.config import BFVConfig


class BFVEngine:
    def __init__(self, config: BFVConfig = None):
        if config is None:
            config = BFVConfig()
        self.config = config
        self.ctx = None
        self.server_ctx = None
        self.plain_modulus = None
        self.max_safe_value = None
        self.keygen_time = 0.0

    def setup(self) -> 'BFVEngine':
        print("\n  Setting up BFV encryption ...")
        print(f"  poly_modulus_degree = {self.config.poly_modulus_degree}")

        t0 = time.time()

        try:
            self.plain_modulus = ts.plain_modulus_batching(
                self.config.poly_modulus_degree,
                self.config.plain_mod_bits,
            )
        except Exception:
            fallback = {
                4096: 16760833,
                8192: 33538049,
                16384: 67104769,
            }
            self.plain_modulus = fallback.get(
                self.config.poly_modulus_degree, 33538049
            )

        self.max_safe_value = self.plain_modulus // 2

        self.ctx = ts.context(
            ts.SCHEME_TYPE.BFV,
            poly_modulus_degree=self.config.poly_modulus_degree,
            plain_modulus=self.plain_modulus,
        )
        self.ctx.generate_galois_keys()
        self.ctx.generate_relin_keys()

        pub_ctx = self.ctx.copy()
        pub_ctx.make_context_public()
        self.server_ctx = ts.context_from(pub_ctx.serialize())

        self.keygen_time = time.time() - t0

        print(f"  plain_modulus = {self.plain_modulus}")
        print(f"  max safe dot product = +-{self.max_safe_value:,}")
        print(f"  keygen time = {self.keygen_time:.2f}s")
        print("  BFV encryption ready")

        return self

    def encrypt_vector(self, int_vector: np.ndarray) -> bytes:
        int_list = [int(x) for x in int_vector.tolist()]
        enc = ts.bfv_vector(self.ctx, int_list)
        return enc.serialize()

    def encrypt_documents(self, int_embeddings: np.ndarray) -> List[bytes]:
        print(f"\n  Encrypting {len(int_embeddings)} document vectors (BFV) ...")
        t0 = time.time()

        encrypted = []
        for i, vec in enumerate(int_embeddings):
            ct_bytes = self.encrypt_vector(vec)
            encrypted.append(ct_bytes)
            if (i + 1) % 10 == 0 or i == len(int_embeddings) - 1:
                print(f"  [{i+1}/{len(int_embeddings)}]", end='\r')

        encrypt_time = time.time() - t0
        ct_size = len(encrypted[0])

        print(f"\n  BFV documents encrypted:")
        print(f"  Time: {encrypt_time:.1f}s ({encrypt_time/len(int_embeddings)*1000:.1f} ms/vector)")
        print(f"  Ciphertext size: {ct_size:,} bytes/vector")

        return encrypted

    def decrypt_score(self, score_bytes: bytes) -> int:
        sv = ts.bfv_vector_from(self.ctx, score_bytes)
        val = int(sv.decrypt()[0])
        if val > self.max_safe_value:
            val = val - self.plain_modulus
        return val

    def server_dot_product_ct_ct(self, enc_query_bytes: bytes, enc_doc_bytes: bytes) -> bytes:
        enc_q = ts.bfv_vector_from(self.server_ctx, enc_query_bytes)
        enc_d = ts.bfv_vector_from(self.server_ctx, enc_doc_bytes)
        enc_score = enc_q.dot(enc_d)
        return enc_score.serialize()
