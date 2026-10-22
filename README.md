# Voice-Enabled RAG AI

A voice-first Retrieval-Augmented Generation (RAG) assistant built for the HH Goa 2026 hackathon. Ask a question by typing or speaking, and the app answers using only the documents you give it, with the option to read the answer aloud.

> Built as a learning project with the help of AI tools (Claude).

## Pipeline

Voice input → Speech-to-Text → Chunking → Embeddings → FAISS search → Gemini answer → Text-to-Speech

## How It Works

1. **Speech-to-Text:** The recorded question is transcribed with Sarvam AI (Saaras v3), which supports Indian languages.
2. **Chunking:** Documents are split with two strategies combined: 2-sentence chunks and overlapping word chunks (80 words, 20-word overlap).
3. **Embeddings:** Each chunk is converted into a vector with a Sentence Transformers model (`all-MiniLM-L6-v2`).
4. **Retrieval:** Vectors are stored in a FAISS index. The question is embedded the same way, and the top 3 closest chunks are retrieved.
5. **Generation:** The retrieved chunks go to Google Gemini, which answers using only that context. If the answer isn't in the context, it says so.
6. **Text-to-Speech:** The answer is converted to audio with Sarvam AI (Bulbul v3).

## Features

- Text and voice interfaces (Gradio web app)
- Multi-strategy chunking
- Semantic search with FAISS
- Answers grounded in retrieved context, to reduce hallucination
- Retrieved chunks and similarity scores shown alongside each answer
- Docker support

## Project Structure

```
HH_GOA_RAG/
├── app.py                  # Gradio web app
├── requirements.txt
├── Dockerfile
├── README.md
├── .gitignore
├── data/
│   └── sample_documents.json   # sample knowledge base
├── src/
│   ├── chunking.py         # sentence and word chunking
│   ├── embeddings.py       # sentence embeddings
│   ├── retriever.py        # FAISS search
│   ├── generator.py        # Gemini answer generation
│   ├── speech.py           # Sarvam STT and TTS
│   └── rag_pipeline.py     # ties everything together
└── notebooks/              # development notebooks (Google Colab)
```

## Tech Stack

Python, Sentence Transformers, FAISS, Google Gemini API, Sarvam AI API, Gradio, Docker

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/govindbhaiyya23-pixel/HH_GOA_RAG.git
cd HH_GOA_RAG
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API keys

You need a Gemini API key and a Sarvam API key.

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your_gemini_key"
$env:SARVAM_API_KEY="your_sarvam_key"
```

**macOS / Linux:**
```bash
export GEMINI_API_KEY="your_gemini_key"
export SARVAM_API_KEY="your_sarvam_key"
```

Never commit real API keys to the repo.

### 4. Run the app

```bash
python app.py
```

Then open the local URL shown in the terminal.

### Run with Docker

```bash
docker build -t hh-goa-rag .
docker run -p 8080:8080 -e GEMINI_API_KEY=your_gemini_key -e SARVAM_API_KEY=your_sarvam_key hh-goa-rag
```

## Using Your Own Documents

Edit `data/sample_documents.json`. Each entry needs an `id` and a `text`:

```json
[
  { "id": "doc1", "text": "Your document text here." }
]
```

## Acknowledgements

- [Sarvam AI](https://www.sarvam.ai/) for speech-to-text and text-to-speech
- [Google Gemini](https://ai.google.dev/) for answer generation
- [FAISS](https://github.com/facebookresearch/faiss) and [Sentence Transformers](https://www.sbert.net/) for retrieval
