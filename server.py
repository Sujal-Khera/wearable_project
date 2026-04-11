from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from llama_cpp import Llama

from src.rag_pipeline import PrivateRAGPipeline
from config import DEFAULT_CONFIG

app = FastAPI()


MODEL_PATH = "/home/rpi/sujal/models/qwen/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"

PDF_PATH = "/home/rpi/sujal/wearable_project/data/KLVR Wearable Technical Spec.pdf"
KNOWLEDGE_PATH = "/home/rpi/edge_ai_assistant/data/knowledge.txt"


print("Loading model...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    n_batch=512
)

print("Model loaded")


print("Initializing RAG...")

pipeline = PrivateRAGPipeline(DEFAULT_CONFIG)

pipeline.ingest_document(PDF_PATH)
pipeline.ingest_document(KNOWLEDGE_PATH)

pipeline.generate_embeddings() \
        .quantize() \
        .encrypt_and_index()

pipeline.print_summary()


class Prompt(BaseModel):
    text: str


def append_to_knowledge(query, answer):

    with open(
        KNOWLEDGE_PATH,
        "a"
    ) as f:

        f.write("\n\n")
        f.write(f"Question: {query}\n")
        f.write(f"Answer: {answer}\n")


def generate_stream(prompt):

    stream = llm(
        prompt,
        max_tokens=512,
        temperature=0.7,
        stream=True
    )

    for output in stream:
        yield output["choices"][0]["text"]


@app.post("/generate")
def generate(prompt: Prompt):

    rag = pipeline.search(prompt.text)

    context = "\n\n".join(
        [r["text"] for r in rag["results"]]
    )

    final_prompt = f"""
Context:
{context}

Question:
{prompt.text}

Answer:
"""

    def stream():

        full = ""

        for token in generate_stream(final_prompt):

            full += token
            yield token

        append_to_knowledge(
            prompt.text,
            full
        )

    return StreamingResponse(
        stream(),
        media_type="text/plain"
    )


@app.get("/health")
def health():

    return {
        "status": "running"
    }