
from src.chunking import chunk_documents
from src.embeddings import create_embeddings
from src.retriever import FAISSRetriever
from src.generator import GeminiGenerator


class RAGPipeline:

    def __init__(
        self,
        documents,
        gemini_api_key
    ):

        self.documents = documents

        self.generator = GeminiGenerator(
            api_key=gemini_api_key
        )

        self.chunks_data = chunk_documents(
            self.documents
        )

        self.chunk_texts_for_retriever = [
            chunk["text"]
            for chunk in self.chunks_data
        ]

        self.embeddings = create_embeddings(
            self.chunks_data
        )

        self.retriever = FAISSRetriever(
            self.embeddings,
            self.chunk_texts_for_retriever
        )


    def answer(
        self,
        query,
        top_k=3
    ):

        query_embedding = create_embeddings(
            [{"text": query}]
        )

        results = self.retriever.search(
            query_embedding,
            top_k
        )

        context = "\n\n".join(
            result["text"]
            for result in results
        )

        answer = self.generator.generate(
            query=query,
            context=context
        )

        return {
            "answer": answer,
            "context": results
        }
