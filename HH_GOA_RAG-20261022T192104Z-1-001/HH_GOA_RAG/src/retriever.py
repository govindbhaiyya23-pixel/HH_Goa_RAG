
import faiss
import numpy as np


class FAISSRetriever:

    def __init__(self, embeddings, chunks):

        self.chunks = chunks

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        self.index.add(embeddings)


    def search(
        self,
        query_embedding,
        top_k=3
    ):

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index < len(self.chunks):

                results.append({
                    "text": self.chunks[index],
                    "score": float(score)
                })

        return results
