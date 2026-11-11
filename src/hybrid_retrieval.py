
import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class HybridPolicyRetriever:
    def __init__(
        self,
        collection_name="policylens_structure_300_v1",
        model_name="all-MiniLM-L6-v2"
    ):
        print("Loading semantic embedding model...")
        self.model = SentenceTransformer(model_name)

        print("Connecting to ChromaDB...")
        self.client = chromadb.PersistentClient(path="data/chroma_db")
        self.collection = self.client.get_collection(
            name=collection_name
        )

        print("Loading documents for BM25...")
        stored = self.collection.get(
            include=["documents", "metadatas"]
        )

        self.ids = stored["ids"]
        self.documents = stored["documents"]
        self.metadatas = stored["metadatas"]

        self.bm25 = BM25Okapi([
            self.tokenize(document)
            for document in self.documents
        ])

        print(f"Hybrid retriever ready: {len(self.documents)} chunks")

    @staticmethod
    def tokenize(text):
        return re.findall(r"\b\w+\b", text.lower())

    @staticmethod
    def matches_filter(metadata, where):
        if not where:
            return True

        return all(
            metadata.get(key) == value
            for key, value in where.items()
        )

    def search(self, query, top_k=5, where=None, candidate_k=50):
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        # 1. Semantic search
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        semantic_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(candidate_k, len(self.documents)),
            include=["documents", "metadatas", "distances"]
        )

        # 2. BM25 keyword search
        query_tokens = self.tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)

        allowed_indices = [
            index for index, metadata in enumerate(self.metadatas)
            if self.matches_filter(metadata, where)
        ]

        bm25_indices = sorted(
            allowed_indices,
            key=lambda index: bm25_scores[index],
            reverse=True
        )[:candidate_k]

        # 3. Reciprocal Rank Fusion
        rrf_k = 60
        fused_scores = {}
        result_data = {}

        semantic_ids = semantic_results["ids"][0]

        for rank, chunk_id in enumerate(semantic_ids, start=1):
            metadata = semantic_results["metadatas"][0][rank - 1]

            if not self.matches_filter(metadata, where):
                continue

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0)
                + 1 / (rrf_k + rank)
            )

            result_data[chunk_id] = {
                "document": semantic_results["documents"][0][rank - 1],
                "metadata": metadata
            }

        for rank, index in enumerate(bm25_indices, start=1):
            chunk_id = self.ids[index]

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0)
                + 1 / (rrf_k + rank)
            )

            result_data[chunk_id] = {
                "document": self.documents[index],
                "metadata": self.metadatas[index]
            }

        # 4. Sort by combined RRF score
        ranked_ids = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True
        )[:top_k]

        results = []

        for rank, chunk_id in enumerate(ranked_ids, start=1):
            results.append({
                "rank": rank,
                "score": round(fused_scores[chunk_id], 6),
                "document": result_data[chunk_id]["document"],
                "metadata": result_data[chunk_id]["metadata"]
            })

        return results


if __name__ == "__main__":
    retriever = HybridPolicyRetriever()

    query = "When should a bank release the sanctioned loan amount?"

    results = retriever.search(query, top_k=5)

    for result in results:
        metadata = result["metadata"]

        print("\nRank:", result["rank"])
        print("RRF Score:", result["score"])
        print("Title:", metadata.get("title"))
        print(
            "Pages:",
            metadata.get("page_start"),
            "-",
            metadata.get("page_end")
        )
        print("Section:", metadata.get("section_title"))
        print("Text:", result["document"][:200])
