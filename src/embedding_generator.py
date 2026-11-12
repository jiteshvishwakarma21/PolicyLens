from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE


class EmbeddingGenerator:

    def __init__(self, model_name=EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        print("Embedding model loaded successfully!")
        print("Maximum sequence length:", self.model.max_seq_length)

    def generate_embedding(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text cannot be empty.")

        # E5 query prefix is used for search queries.
        text = "query: " + text

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding.tolist()

    def generate_embeddings(self, texts, batch_size=EMBEDDING_BATCH_SIZE):
        if not texts:
            return []

        if not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("All texts must be non-empty strings.")

        # E5 passage prefix is used for document chunks.
        passages = ["passage: " + text for text in texts]

        embeddings = self.model.encode(
            passages,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return embeddings.tolist()
