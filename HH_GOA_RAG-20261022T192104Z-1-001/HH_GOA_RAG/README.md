# Voice-Enabled RAG AI

A voice-enabled Retrieval-Augmented Generation system.

## Pipeline

Voice Input → Sarvam STT → Multi-Strategy Chunking → Embeddings → FAISS → Gemini → Sarvam TTS

## Features

- Voice input
- Sarvam Saaras v3 speech-to-text
- Multi-strategy chunking
- Sentence Transformer embeddings
- FAISS vector retrieval
- Gemini grounded answer generation
- Sarvam Bulbul v3 text-to-speech
- Text and voice interfaces
- Retrieved context display

## Project Structure

HH_GOA_RAG/
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
├── data/
│   └── sample_documents.json
└── src/
    ├── chunking.py
    ├── embeddings.py
    ├── retriever.py
    ├── generator.py
    ├── speech.py
    └── rag_pipeline.py

## Run

pip install -r requirements.txt

python app.py

## Environment Variables

GEMINI_API_KEY

SARVAM_API_KEY

Never commit API keys.
