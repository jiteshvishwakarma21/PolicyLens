
import chromadb
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "policylens_structure_300_v1"
MODEL_NAME = "intfloat/e5-base-v2"

def main():
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path="data/chroma_db")
    collection = client.get_collection(name=COLLECTION_NAME)

    results = collection.get(include=["documents"])
    documents = results["documents"]

    lengths = []

    for document in documents:
        text = "passage: " + document

        tokens = model.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False
        )["input_ids"]

        lengths.append(len(tokens))

    maximum = model.max_seq_length
    exceeded = sum(length > maximum for length in lengths)

    print("\n===== E5 TOKEN LENGTH ANALYSIS =====")
    print("Model:", MODEL_NAME)
    print("Total chunks:", len(lengths))
    print("Model maximum:", maximum)
    print("Average tokens:", round(sum(lengths) / len(lengths), 2))
    print("Longest chunk:", max(lengths))
    print("Chunks exceeding limit:", exceeded)
    print("Chunks within limit:", len(lengths) - exceeded)

if __name__ == "__main__":
    main()
