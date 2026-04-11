import os
from typing import List
from config import ChunkConfig


def extract_text_from_pdf(pdf_path: str) -> str:

    import fitz

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)

    pages = []

    for page in doc:
        text = page.get_text()

        if text.strip():
            pages.append(text.strip())

    doc.close()

    return "\n\n".join(pages)


def chunk_text(text: str, config: ChunkConfig) -> List[str]:

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + config.chunk_size

        chunk = " ".join(words[start:end])

        if len(chunk) >= config.min_chunk_length:
            chunks.append(chunk)

        start += config.chunk_size - config.chunk_overlap

    print(f"Created {len(chunks)} chunks")

    return chunks


def load_document(path: str, config: ChunkConfig):

    if path.lower().endswith(".pdf"):

        text = extract_text_from_pdf(path)

    else:

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

    return chunk_text(text, config)