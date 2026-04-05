import os
from typing import List
from config import ChunkConfig


def extract_text_from_pdf(pdf_path: str) -> str:
    import fitz  # PyMuPDF

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            pages_text.append(text.strip())
    doc.close()

    full_text = "\n\n".join(pages_text)
    print(f"  Extracted {len(full_text):,} characters from {len(pages_text)} pages")
    return full_text


def chunk_text(text: str, config: ChunkConfig = None) -> List[str]:
    if config is None:
        config = ChunkConfig()

    text = text.replace('\r\n', '\n').replace('\r', '\n')
    words = text.split()

    if len(words) == 0:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + config.chunk_size
        chunk_words = words[start:end]
        current_chunk = " ".join(chunk_words)

        if len(current_chunk) >= config.min_chunk_length:
            chunks.append(current_chunk)

        start += config.chunk_size - config.chunk_overlap

    print(f"  Created {len(chunks)} chunks "
          f"(size={config.chunk_size}, overlap={config.chunk_overlap})")
    return chunks


def load_document(path: str, config: ChunkConfig = None) -> List[str]:
    if config is None:
        config = ChunkConfig()

    if path.lower().endswith('.pdf'):
        text = extract_text_from_pdf(path)
    elif path.lower().endswith(('.txt', '.md')):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"  Loaded {len(text):,} characters from {path}")
    else:
        raise ValueError(f"Unsupported file type: {path}")

    return chunk_text(text, config)
