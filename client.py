import requests
import time
import sys

PI_IP = "192.168.1.15"
PORT = 8000

URL = f"http://{PI_IP}:{PORT}/generate"
HEALTH_URL = f"http://{PI_IP}:{PORT}/health"
RAG_URL = f"http://{PI_IP}:{PORT}/rag_search"


def check_health():
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        print("Server health:", r.json())
    except Exception as e:
        print("Health check failed:", e)


def test_rag_only(prompt):
    try:
        r = requests.post(RAG_URL, json={"text": prompt}, timeout=30)
        print("\nRAG Result:")
        print(r.json())
        print()
    except Exception as e:
        print("RAG test failed:", e)


def ask_llm(prompt):
    try:
        response = requests.post(
            URL,
            json={"text": prompt},
            stream=True,
            timeout=120
        )

        if response.status_code != 200:
            print("Server Error:", response.text)
            return

        rag_used = response.headers.get("X-RAG-Used", "false")
        print(f"\n[RAG used: {rag_used}]")
        print("Assistant: ", end="", flush=True)

        for chunk in response.iter_content(chunk_size=1):
            if chunk:
                char = chunk.decode("utf-8", errors="ignore")
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(0.01)

        print("\n")

    except Exception as e:
        print("\nConnection Error:", e)


def main():
    print("Edge AI Assistant (Private RAG + Streaming)")
    print("Commands:")
    print("  /health           -> check server")
    print("  /rag your query   -> test only RAG retrieval")
    print("  exit              -> quit\n")

    check_health()

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            break
        elif user_input == "/health":
            check_health()
        elif user_input.startswith("/rag "):
            test_rag_only(user_input[5:])
        elif user_input:
            ask_llm(user_input)


if __name__ == "__main__":
    main()
