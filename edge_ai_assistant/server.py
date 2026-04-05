from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, JSONResponse
from llama_cpp import Llama

from src.rag_pipeline import PrivateRAGPipeline
from config import DEFAULT_CONFIG

app = FastAPI()

MODEL_PATH = "/home/rpi/sujal/models/qwen/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
DOCUMENT_PATH = "/home/rpi/edge_ai_assistant/data/your_document.pdf"

print("Loading model...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    n_batch=512
)
print("Model loaded successfully")

print("Initializing Private RAG pipeline...")
pipeline = None
try:
    pipeline = PrivateRAGPipeline(DEFAULT_CONFIG)
    pipeline.ingest_document(DOCUMENT_PATH) \
            .generate_embeddings() \
            .quantize() \
            .encrypt_and_index()
    pipeline.print_summary()
    print("Private RAG pipeline ready")
except Exception as e:
    print("Failed to initialize RAG pipeline:", e)
    pipeline = None


class Prompt(BaseModel):
    text: str


def build_prompt_with_rag(user_query: str, rag_result: dict) -> str:
    context_chunks = [item["text"] for item in rag_result["results"]]
    context = "\n\n".join(context_chunks)

    return f"""<|system|>
You are a helpful assistant.
Use the provided context if relevant.
If the answer is found in the context, answer based on it clearly.
If the context is insufficient, answer as best as you can.

Context:
{context}

<|user|>
{user_query}
<|assistant|>
"""


def build_prompt_without_rag(user_query: str) -> str:
    return f"<|user|>\n{user_query}\n<|assistant|>\n"


def generate_stream(prompt_text: str):
    stream = llm(
        prompt_text,
        max_tokens=128,
        temperature=0.7,
        stream=True
    )

    for output in stream:
        token = output["choices"][0]["text"]
        yield token


@app.post("/generate")
def generate(prompt: Prompt):
    try:
        rag_used = False

        if pipeline is not None:
            rag_result = pipeline.search(prompt.text, top_k=3)

            if rag_result["has_match"]:
                final_prompt = build_prompt_with_rag(prompt.text, rag_result)
                rag_used = True
                print(f"[RAG HIT] {prompt.text}")
                print(f"Top normalized score: {rag_result['top_score_normalized']:.4f}")
            else:
                final_prompt = build_prompt_without_rag(prompt.text)
                print(f"[RAG MISS] {prompt.text}")
        else:
            final_prompt = build_prompt_without_rag(prompt.text)
            print("[RAG DISABLED] Pipeline not initialized")

        return StreamingResponse(
            generate_stream(final_prompt),
            media_type="text/plain",
            headers={"X-RAG-Used": str(rag_used).lower()}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/rag_search")
def rag_search(prompt: Prompt):
    if pipeline is None:
        return {"error": "RAG pipeline not ready"}
    return pipeline.search(prompt.text, top_k=3)


@app.get("/health")
def health():
    return {
        "status": "running",
        "rag_ready": pipeline is not None
    }
