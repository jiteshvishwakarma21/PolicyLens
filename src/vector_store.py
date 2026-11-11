
import chromadb


class VectorStore:

    def __init__(self, collection_name="policylens_v1"):

        self.client = chromadb.PersistentClient(
            path="data/chroma_db"
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"Connected to collection: {collection_name}")


    def add_documents(self, chunks, embeddings):

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match."
            )

        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []

        for index, chunk in enumerate(chunks):

            metadata = chunk.get("metadata", {})
            doc_id = metadata.get("doc_id", "document")

            unique_id = f"{doc_id}_{chunk['chunk_id']}"

            ids.append(unique_id)
            documents.append(chunk["text"])

            # ChromaDB metadata values must be simple scalar values.
            clean_metadata = {
                key: value
                for key, value in metadata.items()
                if isinstance(value, (str, int, float, bool))
            }

            metadatas.append(clean_metadata)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"Successfully stored {len(chunks)} chunks!")


    def search(self, query_embedding, top_k=5, where=None):

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )

        return results


if __name__ == "__main__":

    store = VectorStore()

    print("Vector Store module is ready!")
    print("Stored documents:", store.collection.count())
