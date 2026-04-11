# KLVR — Edge AI Voice Assistant

**Private RAG + Self-Learning + Streaming + Voice + GUI**

---

# Overview

KLVR is an **Edge AI Voice Assistant** running on **Raspberry Pi** that provides:

* Private Retrieval-Augmented Generation (RAG)
* Local LLM inference
* Streaming responses
* Voice interaction
* Self-learning memory
* Multi-document knowledge base
* Beautiful GUI client

All processing happens **locally** — no cloud dependency.

---

# System Architecture

## Raspberry Pi (Server)

* Local LLM (Qwen 1.5B)
* Private RAG Pipeline
* FAISS Vector Search
* Multi-Document Knowledge
* Self-Learning Memory
* FastAPI Server

## Laptop (Client)

* Voice Input
* Wake Word Detection
* Manual Recording
* GUI Interface
* Streaming Response Display
* Text-to-Speech

---

# How It Works (Pipeline)

```
User Voice
    ↓
Wake Word / Manual Start
    ↓
Speech Recognition
    ↓
Send Query to Raspberry Pi
    ↓
RAG Search (PDF + Memory)
    ↓
Threshold Check
    ↓
LLM Generation
    ↓
Streaming Response
    ↓
Save to Memory
    ↓
Better Future Responses
```

---

# Key Features

## Private RAG

* Documents stored locally
* No cloud dependency
* Fast retrieval using FAISS

---

## Multi-Document Knowledge

System loads:

```
KLVR Technical PDF
knowledge.txt
Future Documents
```

All documents merged into one vector database.

---

## Self-Learning Memory

After each response:

* Answer stored in `knowledge.txt`
* Future queries use learned knowledge
* Assistant improves automatically

---

## Streaming Responses

* Token-by-token streaming
* Faster response perception
* Natural conversational experience

---

## Voice Assistant

Supports:

* Wake word detection
* Manual recording
* GUI interface
* Text-to-Speech

---

# Project Structure

```
edge_ai_assistant/
├── server.py
├── config.py
├── requirements.txt
├── data/
│   ├── KLVR.pdf
│   └── knowledge.txt
└── src/
    ├── chunker.py
    ├── embedder.py
    ├── rag_pipeline.py
    └── search.py
```

Client side:

```
voice_client.py
klvr_gui.py
```

---

# RAG Pipeline

```
Document
   ↓
Chunking
   ↓
Embedding
   ↓
FAISS Index
   ↓
Query Embedding
   ↓
Similarity Search
   ↓
Retrieve Context
   ↓
LLM Response
```

---

# Setup — Raspberry Pi

## Create Project

```bash
cd /home/rpi
mkdir -p edge_ai_assistant/src
mkdir -p edge_ai_assistant/data
cd edge_ai_assistant
```

---

## Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

---

## Install Dependencies

```bash
pip install fastapi uvicorn llama-cpp-python \
sentence-transformers faiss-cpu pymupdf numpy
```

---

## Add Documents

Place inside:

```
/home/rpi/edge_ai_assistant/data/
```

Example:

```
KLVR Wearable Technical Spec.pdf
knowledge.txt
```

---

## Start Server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

---

# Server Startup Flow

```
Load LLM
↓
Ingest Documents
↓
Generate Embeddings
↓
Create FAISS Index
↓
Server Ready
```

---

# Test Server

Health Check

```bash
curl http://127.0.0.1:8000/health
```

---

Generate

```bash
curl -X POST http://127.0.0.1:8000/generate \
-d '{"text":"Explain KLVR"}'
```

---

# Voice Client

Install:

```bash
pip install requests SpeechRecognition pyttsx3 pyaudio
```

Run:

```bash
python voice_client.py
```

---

# GUI Client

Modern interface:

```bash
python klvr_gui.py
```

Features:

* Chat Interface
* Start / Stop Speaking
* Streaming Response
* Voice Playback

---

# Self-Learning Example

User asks:

```
Explain KLVR architecture
```

System:

```
RAG search
LLM generation
Store to knowledge.txt
```

Next query becomes **more accurate**.

---

# Applications

## Personal Assistant

Private knowledge assistant

---

## Enterprise AI

Company documentation search

---

## Smart Home Assistant

Local voice control

---

## Healthcare

Privacy-sensitive environments

---

## Industrial Edge AI

Factory assistant

---

# Novelty

KLVR combines:

* Edge AI
* Private RAG
* Self-Learning
* Streaming LLM
* Voice Interface
* GUI Interface

All running **locally on Raspberry Pi**.

---

# Current Capabilities

* Multi-Document RAG
* Self-Learning Memory
* Voice Assistant
* Streaming LLM
* GUI Interface
* Edge Deployment

---

# Future Improvements

* Persistent Vector DB
* Offline Speech Recognition
* Web UI
* Multi-User Support
* Conversation Memory

---

# Final Architecture

```
Laptop Voice Client
        ↓
Raspberry Pi Server
        ↓
Private RAG Pipeline
        ↓
Local LLM
        ↓
Streaming Response
        ↓
Voice Output
```

---

# KLVR

**Intelligent Edge AI Voice Assistant**
**Private • Local • Self-Learning**
