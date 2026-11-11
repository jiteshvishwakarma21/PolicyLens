
import chromadb

COLLECTION_NAME = "policylens_final_v2"
DOC_ID = "RBI_CREDIT_001"

TARGETS = {
    "Q11 - Eligible securities": {
        "page": 10,
        "keywords": ["eligible securities", "equity shares", "government securities"]
    },
    "Q12 - Board-approved credit policy": {
        "page": 16,
        "keywords": ["credit policy", "board approved", "board-approved"]
    },
    "Q16 - Borrower profile": {
        "page": 19,
        "keywords": ["age", "occupation", "income"]
    },
    "Q19 - Project disbursement": {
        "page": 47,
        "keywords": ["disbursement", "project", "equity"]
    }
}


def main():
    client = chromadb.PersistentClient(path="data/chroma_db")

    collection = client.get_collection(name=COLLECTION_NAME)

    print("\nCOLLECTION:", COLLECTION_NAME)
    print("TOTAL CHUNKS:", collection.count())

    results = collection.get(
        where={"doc_id": DOC_ID},
        include=["documents", "metadatas"]
    )

    ids = results["ids"]
    documents = results["documents"] or []
    metadatas = results["metadatas"] or []

    print("DOCUMENT CHUNKS:", len(ids))

    for question, target in TARGETS.items():
        page = target["page"]
        keywords = target["keywords"]

        print("\n" + "=" * 70)
        print(question)
        print("Expected PDF page:", page)
        print("=" * 70)

        matches = []

        for chunk_id, document, metadata in zip(
            ids, documents, metadatas
        ):
            start = metadata.get("page_start")
            end = metadata.get("page_end")

            if start is None or end is None:
                continue

            if start <= page <= end:
                matches.append((chunk_id, document, metadata))

        if not matches:
            print("NO CHUNK COVERS THIS PAGE.")
            continue

        print("Chunks covering expected page:", len(matches))

        for chunk_id, document, metadata in matches:
            print("\nChunk ID:", chunk_id)
            print("Pages:", metadata.get("page_start"),
                  "-", metadata.get("page_end"))
            print("Section:", metadata.get("section_title"))
            print("Token count:", metadata.get("token_count"))

            print("\nText preview:")
            print(document[:1800])

        print("\nKEYWORD CHECK:")

        all_text = " ".join(
            document.lower() for _, document, _ in matches
        )

        for keyword in keywords:
            found = keyword.lower() in all_text
            print(f"{keyword}: {'FOUND' if found else 'NOT FOUND'}")


if __name__ == "__main__":
    main()
