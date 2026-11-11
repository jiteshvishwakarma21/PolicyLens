from src.embedding_generator import EmbeddingGenerator
from src.vector_store import VectorStore


class PolicyRetriever:
    def __init__(self, collection_name="policylens_final_30docs"):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore(collection_name)

    def search_policies(self, query, top_k=5, where=None):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        query_embedding = self.embedding_generator.generate_embedding(
            query.strip()
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where
        )

        retrieved_documents = []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for index, document in enumerate(documents):
            metadata = metadatas[index] or {}

            retrieved_documents.append({
                "rank": index + 1,
                "score": round(1 - distances[index], 4),
                "document": document,
                "metadata": metadata
            })

        return retrieved_documents


if __name__ == "__main__":
    retriever = PolicyRetriever()

    query = "What are the KYC requirements for bank customers?"
    results = retriever.search_policies(query, top_k=5)

    print("\nPolicy Search Results")
    print("=" * 60)

    if not results:
        print("No matching policy documents found.")

    for result in results:
        metadata = result["metadata"]

        print(f"\nRank: {result['rank']}")
        print(f"Similarity Score: {result['score']}")
        print(f"Document: {metadata.get('title', 'Unknown')}")
        print(f"Document ID: {metadata.get('doc_id', 'Unknown')}")
        print(
            f"Pages: {metadata.get('page_start', 'Unknown')} - "
            f"{metadata.get('page_end', 'Unknown')}"
        )
        print(f"Section: {metadata.get('section_title', 'Unknown')}")
        print(f"Text: {result['document'][:300]}")
        print("-" * 60)