
from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def create_embeddings(chunks):

    texts = []

    for chunk in chunks:

        if isinstance(chunk, dict):
            texts.append(chunk["text"])
        else:
            texts.append(chunk)

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings
