
import os
import csv
import hashlib
import argparse

from src.pdf_extractor import extract_text_from_pdf
from src.ocr_processor import extract_text_using_ocr
from src.text_cleaner import clean_text
from src.document_chunker import TOKENIZER, detect_section_title
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import VectorStore


# Token-based chunking configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def calculate_file_hash(file_path):
    """Generate a SHA-256 hash for a PDF."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(4096), b""):
            sha256.update(block)

    return sha256.hexdigest()


def load_manifest(manifest_path):
    """Read document information from manifest.csv."""

    documents = {}

    with open(manifest_path, "r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "doc_id",
            "title",
            "issuer",
            "issue_date",
            "local_filename"
        }

        if not reader.fieldnames:
            raise ValueError("Manifest CSV is empty.")

        missing = required_columns - set(reader.fieldnames)

        if missing:
            raise ValueError(
                f"Manifest is missing columns: {', '.join(sorted(missing))}"
            )

        for row in reader:
            doc_id = row["doc_id"].strip()
            documents[doc_id] = row

    return documents


def create_document_chunks(pages, pdf_path):
    """
    Combine PDF page tokens and create chunks across pages.
    Preserve page numbers, section titles and token counts.
    """

    all_tokens = []
    token_pages = []
    page_sections = {}

    current_section = "Not detected"

    for page in pages:

        page_number = page["page"]
        page_text = page["text"]

        # Use OCR when a page has little or no extractable text.
        if page["needs_ocr"]:
            print(f"Running OCR on page {page_number}")

            page_text = extract_text_using_ocr(
                pdf_path,
                page_number
            )

        page_text = clean_text(page_text)

        detected_section = detect_section_title(page_text)

        if detected_section != "Not detected":
            current_section = detected_section

        page_sections[page_number] = current_section

        if not page_text:
            continue

        page_tokens = TOKENIZER.encode(page_text)

        if not page_tokens:
            continue

        all_tokens.extend(page_tokens)
        token_pages.extend([page_number] * len(page_tokens))

        # Preserve a separator between pages.
        separator_tokens = TOKENIZER.encode("\n")

        all_tokens.extend(separator_tokens)
        token_pages.extend([page_number] * len(separator_tokens))

    chunks = []

    step_size = CHUNK_SIZE - CHUNK_OVERLAP

    for start in range(0, len(all_tokens), step_size):

        end = min(start + CHUNK_SIZE, len(all_tokens))

        chunk_tokens = all_tokens[start:end]

        if not chunk_tokens:
            continue

        chunk_text = TOKENIZER.decode(chunk_tokens)

        pages_in_chunk = token_pages[start:end]

        page_start = min(pages_in_chunk)
        page_end = max(pages_in_chunk)

        section_title = page_sections.get(
            page_start,
            "Not detected"
        )

        chunks.append({
            "chunk_id": len(chunks) + 1,
            "text": chunk_text,
            "token_count": len(chunk_tokens),

            "metadata": {
                "page_start": page_start,
                "page_end": page_end,
                "section_title": section_title,
                "token_count": len(chunk_tokens)
            }
        })

        if end >= len(all_tokens):
            break

    return chunks


def ingest_documents(input_folder, manifest_path, collection_name):

    if not os.path.isdir(input_folder):
        raise FileNotFoundError(
            f"Input folder not found: {input_folder}"
        )

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Manifest file not found: {manifest_path}"
        )

    manifest = load_manifest(manifest_path)

    embedding_generator = EmbeddingGenerator()
    vector_store = VectorStore(collection_name)

    total_chunks = 0

    for filename in sorted(os.listdir(input_folder)):

        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(input_folder, filename)

        print("\n" + "=" * 60)
        print(f"Processing: {filename}")
        print("=" * 60)

        try:

            file_hash = calculate_file_hash(pdf_path)

            # Find the document in the manifest.
            document_info = None
            doc_id = None

            for key, value in manifest.items():

                if value.get("local_filename", "").strip() == filename:
                    doc_id = key
                    document_info = value
                    break

            if document_info is None:
                print("Skipping: PDF not found in manifest.")
                continue

            # Check whether this document is already indexed.
            existing = vector_store.collection.get(
                where={"doc_id": doc_id},
                include=["metadatas"]
            )

            if existing["ids"]:

                old_hashes = {
                    metadata.get("source_hash")
                    for metadata in (existing.get("metadatas") or [])
                }

                if old_hashes == {file_hash}:
                    print("Document unchanged. Skipping.")
                    continue

            # Extract PDF pages.
            pages = extract_text_from_pdf(pdf_path)

            # Create token-based chunks.
            chunks = create_document_chunks(
                pages,
                pdf_path
            )

            if not chunks:
                print("No text extracted. Skipping document.")
                continue

            # Add document-level metadata.
            for chunk in chunks:

                chunk["metadata"].update({
                    "doc_id": doc_id,
                    "title": document_info["title"],
                    "issuer": document_info["issuer"],
                    "issue_date": document_info["issue_date"],
                    "source_hash": file_hash
                })

            print("Chunks created:", len(chunks))

            token_counts = [
                chunk["token_count"]
                for chunk in chunks
            ]

            print(
                "Average tokens per chunk:",
                round(sum(token_counts) / len(token_counts), 2)
            )

            print("Minimum tokens:", min(token_counts))
            print("Maximum tokens:", max(token_counts))

            # Generate embeddings.
            texts = [
                chunk["text"]
                for chunk in chunks
            ]

            embeddings = embedding_generator.generate_embeddings(
                texts,
                batch_size=32
            )

            # Replace old chunks only after successful processing.
            if existing["ids"]:

                vector_store.collection.delete(
                    where={"doc_id": doc_id}
                )

                print("Previous chunks removed.")

            # Store documents, embeddings and metadata.
            vector_store.add_documents(
                chunks,
                embeddings
            )

            total_chunks += len(chunks)

            print("Successfully indexed:", filename)
            print("Chunks stored:", len(chunks))

        except Exception as error:

            print("\nERROR processing:", filename)
            print("Reason:", error)
            print("Continuing with the next document.")

    print("\n" + "=" * 60)
    print("INGESTION PROCESS COMPLETED")
    print("=" * 60)

    print("New chunks processed:", total_chunks)

    print(
        "Total chunks in collection:",
        vector_store.collection.count()
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="PolicyLens token-based PDF ingestion pipeline"
    )

    parser.add_argument(
        "--input",
        default="data/pdfs",
        help="Folder containing PDF documents"
    )

    parser.add_argument(
        "--manifest",
        default="data/manifest.csv",
        help="Path to manifest CSV"
    )

    parser.add_argument(
        "--collection",
        default="policylens_v1",
        help="ChromaDB collection name"
    )

    args = parser.parse_args()

    ingest_documents(
        args.input,
        args.manifest,
        args.collection
    )
