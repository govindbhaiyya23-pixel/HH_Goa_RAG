
import re


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sentence_chunking(text, sentences_per_chunk=2):
    text = clean_text(text)

    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(
            sentences[i:i + sentences_per_chunk]
        )

        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks


def word_chunking(text, chunk_size=80, overlap=20):
    text = clean_text(text)

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk.strip())

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def chunk_documents(documents):

    all_chunks = []

    for document in documents:

        doc_id = document["id"]
        text = document["text"]

        sentence_chunks = sentence_chunking(
            text,
            sentences_per_chunk=2
        )

        word_chunks = word_chunking(
            text,
            chunk_size=80,
            overlap=20
        )

        combined_chunks = (
            sentence_chunks +
            word_chunks
        )

        for chunk in combined_chunks:

            all_chunks.append({
                "id": doc_id,
                "text": chunk
            })

    return all_chunks
