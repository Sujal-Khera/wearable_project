# Edge AI Assistant (Private RAG + Streaming)

This project runs on a Raspberry Pi and does the following:

- Laptop sends query to Raspberry Pi
- Pi runs Private RAG first
- If RAG finds relevant chunks above threshold, those chunks are fed to local LLM
- If not, LLM answers normally
- Response is streamed back to laptop

## Project structure

```bash
edge_ai_assistant/
├── server.py
├── config.py
├── requirements.txt
├── data/
│   └── your_document.pdf
└── src/
    ├── __init__.py
    ├── chunker.py
    ├── embedder.py
    ├── encryption.py
    ├── quantizer.py
    ├── rag_pipeline.py
    ├── search.py
    └── utils.py
```

Laptop side:

```bash
client.py
voice_client.py
```

## Setup on Raspberry Pi

### 1) Create folders

```bash
cd /home/rpi
mkdir -p edge_ai_assistant/src
mkdir -p edge_ai_assistant/data
cd edge_ai_assistant
```

### 2) Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Put your document in data folder

PDF example:

```bash
cp /path/to/your/document.pdf /home/rpi/edge_ai_assistant/data/your_document.pdf
```

Text example:

```bash
cp /path/to/your/document.txt /home/rpi/edge_ai_assistant/data/your_document.txt
```

### 5) Edit model and document paths in server.py

Set:

```python
MODEL_PATH = "/home/rpi/sujal/models/qwen/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
DOCUMENT_PATH = "/home/rpi/edge_ai_assistant/data/your_document.pdf"
```

If using TXT, update DOCUMENT_PATH accordingly.

### 6) Test imports

```bash
source venv/bin/activate
python -c "from src.rag_pipeline import PrivateRAGPipeline; print('RAG import ok')"
python -c "from llama_cpp import Llama; print('llama import ok')"
python -c "import tenseal; print('tenseal import ok')"
```

### 7) Start server

```bash
cd /home/rpi/edge_ai_assistant
source venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8000
```

Expected startup sequence includes:

- model loading
- document ingestion
- embedding generation
- quantization
- BFV setup
- encryption/indexing

## Test on Raspberry Pi

Health:

```bash
curl http://127.0.0.1:8000/health
```

RAG only:

```bash
curl -X POST http://127.0.0.1:8000/rag_search \
  -H "Content-Type: application/json" \
  -d '{"text":"What is this document about?"}'
```

Generation:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"What is this document about?"}'
```

## Find Raspberry Pi IP

On Raspberry Pi:

```bash
hostname -I
```

Example:

```bash
192.168.1.15
```

Use that IP in laptop client.py:

```python
PI_IP = "192.168.1.15"
```

## Setup on Laptop

Install dependency if needed:

```bash
pip install requests
```

Run client:

```bash
python client.py
```

## Voice Assistant Client on Laptop

The Raspberry Pi server remains unchanged. Microphone and wake/keyboard triggers run on the laptop via [voice_client.py](../voice_client.py).

### What it does

- Keeps microphone listening on the laptop
- Activates when you press `m` or say `Hey KLVR`
- Captures speech and converts speech to text
- Sends query text to Raspberry Pi `/generate`
- Prints streamed response
- Speaks response aloud via TTS

### Configure Pi IP

Edit [voice_client.py](../voice_client.py) and set:

```python
PI_IP = "192.168.1.15"
```

Use your actual Raspberry Pi LAN IP.

### Install laptop dependencies for voice mode

```bash
pip install requests SpeechRecognition pyttsx3 keyboard pyaudio
```

If `pyaudio` fails:

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install portaudio19-dev python3-pyaudio
pip install pyaudio
```

On Windows:

```bash
pip install pipwin
pipwin install pyaudio
```

On macOS:

```bash
brew install portaudio
pip install pyaudio
```

### Run voice client

```bash
python voice_client.py
```

### Voice controls

- Press `m` to trigger a voice query
- Say `Hey KLVR` to trigger wake word mode
- Press `q` to quit the client

### Microphone test (if input device is not detected)

Create `test_mic.py` with:

```python
import speech_recognition as sr
print(sr.Microphone.list_microphone_names())
```

Run:

```bash
python test_mic.py
```

### Practical limitations of current wake-word implementation

- Wake-word detection currently uses continuous normal speech recognition
- It can false-trigger or miss triggers in noisy rooms
- It requires internet for Google recognition
- For production, use dedicated wake-word engines like Porcupine/OpenWakeWord and offline STT such as Vosk

## Client commands

- /health -> check server
- /rag your query -> test only RAG retrieval
- exit -> quit

## Step-by-step testing

### Test 1: health

Input:

```text
/health
```

Expected:

```text
Server health: {'status': 'running', 'rag_ready': True}
```

### Test 2: RAG only

```text
/rag what is the main topic of the document
```

Expected: JSON output and has_match true if relevant.

### Test 3: full answer

```text
what is the main topic of the document
```

Expected: RAG used true if match found and streamed LLM answer.

### Test 4: unrelated query

```text
what is the capital of japan
```

Expected: RAG may be false and normal LLM answer.

## Common issues

### ModuleNotFoundError: src

Run from project root:

```bash
cd /home/rpi/edge_ai_assistant
uvicorn server:app --host 0.0.0.0 --port 8000
```

### PDF not found

Fix DOCUMENT_PATH in server.py.

### llama_cpp install/build issues

Use a compatible wheel or Pi-specific build setup.

### tenseal install issues

TenSEAL can be tricky on ARM/Pi. Check Pi model, OS, and Python version for a Pi-specific install path.

### Threshold too strict

In config.py lower:

```python
relevance_threshold = 0.01
```

### Threshold too loose

In config.py increase:

```python
relevance_threshold = 0.05
```

## Optional quick TXT test

Create file:

```bash
nano /home/rpi/edge_ai_assistant/data/your_document.txt
```

Then in server.py:

```python
DOCUMENT_PATH = "/home/rpi/edge_ai_assistant/data/your_document.txt"
```

## Exact order from scratch

### Raspberry Pi

```bash
cd /home/rpi
mkdir -p edge_ai_assistant/src
mkdir -p edge_ai_assistant/data
cd edge_ai_assistant
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Copy document into data, edit server.py paths, then run:

```bash
source venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Laptop

```bash
pip install requests
python client.py
```

Voice mode:

```bash
pip install requests SpeechRecognition pyttsx3 keyboard pyaudio
python voice_client.py
```

## Current behavior

- RAG search is encrypted using BFV over quantized embeddings
- If relevant chunks found, they are inserted into prompt
- Local LLM generates answer
- Output streams to client

## Not yet included

- saving encrypted index to disk
- loading encrypted index without rebuilding
- direct document answer first, then LLM answer
- chunk-by-chunk encrypted output streaming
- multi-document ingestion
- UI frontend
