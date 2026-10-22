# Colab Notebooks

A collection of Google Colab practice notebooks. The main one is a
voice-enabled RAG (Retrieval-Augmented Generation) project built for
the HH Goa 2026 hackathon.

> Note: These notebooks were built with the help of AI tools (Claude)
> as a learning exercise.

## 1. HH_GOA_RAG.ipynb: Voice-Enabled RAG Assistant

The user speaks a question, and the system answers it using only the
information in a document collection, then reads the answer aloud.

### Pipeline

Voice input → Speech-to-Text → Chunking → Embeddings → FAISS search → Gemini answer → Text-to-Speech

### How it works

1. **Speech-to-Text:** The user's recorded question is transcribed
   with Sarvam AI (Saaras v3), which supports Indian languages.
2. **Chunking:** Documents are split into smaller pieces. Three
   strategies are compared: fixed-size, sentence-based, and
   overlapping sentences.
3. **Embeddings:** Each chunk is converted into a vector with the
   `paraphrase-multilingual-MiniLM-L12-v2` Sentence Transformer model.
4. **Retrieval:** The vectors are stored in a FAISS index. The
   question is embedded the same way, and the closest chunks are
   retrieved.
5. **Generation:** The retrieved chunks are sent to Google Gemini as
   context, and it writes an answer based only on that context.
6. **Text-to-Speech:** The answer is converted to audio with Sarvam
   AI (Bulbul v3).

### Project structure

    HH_GOA_RAG/
    ├── app.py              # Gradio web app
    ├── requirements.txt
    ├── Dockerfile
    ├── data/sample_documents.json
    └── src/
        ├── chunking.py     # text splitting strategies
        ├── embeddings.py   # sentence embeddings
        ├── retriever.py    # FAISS search
        ├── generator.py    # Gemini answer generation
        ├── speech.py       # Sarvam STT and TTS
        └── rag_pipeline.py # ties everything together

### Tech stack

Python, Sentence Transformers, FAISS, Google Gemini API, Sarvam AI
API, Gradio, Docker

### Running it

1. Open the notebook in Google Colab.
2. Add your own API keys as environment variables (`GEMINI_API_KEY`,
   `SARVAM_API_KEY`). Never commit real keys.
3. Run the cells in order, or run `python app.py` after installing
   `requirements.txt`.

## 2. Other practice notebooks

- **Untitled2.ipynb:** Bayes' theorem calculation (prior, likelihood,
  evidence, posterior probability).
- **Untitled3.ipynb:** C programs run in Colab: a binary search tree
  and graph traversal (DFS and BFS) using an adjacency matrix.
- **Untitled0.ipynb / Untitled1.ipynb:** Setup, backup, and Git
  push steps for the RAG project.
