
import os
import csv
import re
import hashlib
import argparse

from src.pdf_extractor import extract_text_from_pdf
from src.ocr_processor import extract_text_using_ocr
from src.document_chunker import (
    TOKENIZER,
    detect_section_title,
    is_heading
)
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import VectorStore

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEFAULT_INPUT,
    DEFAULT_MANIFEST,
)


# --------------------------------------------------
# FILE HASH
# --------------------------------------------------

def calculate_file_hash(file_path):
    """Calculate SHA-256 hash of a PDF."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            sha256.update(block)

    return sha256.hexdigest()


# --------------------------------------------------
# MANIFEST
# --------------------------------------------------

def load_manifest(manifest_path):
    """Load document information from manifest.csv."""

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    documents = {}

    with open(
        manifest_path,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "doc_id",
            "title",
            "issuer",
            "issue_date",
            "local_filename"
        }

        if not reader.fieldnames or not required_columns.issubset(
            set(reader.fieldnames)
        ):
            raise ValueError(
                "Manifest is missing required columns."
            )

        for row in reader:
            doc_id = (row.get("doc_id") or "").strip()
            filename = (row.get("local_filename") or "").strip()

            if not doc_id or not filename:
                raise ValueError(
                    "Manifest contains an empty doc_id or filename."
                )

            if doc_id in documents:
                raise ValueError(
                    f"Duplicate document ID in manifest: {doc_id}"
                )

            documents[doc_id] = row

    return documents


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text):
    """Clean extracted PDF text."""

    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove unwanted control characters.
    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
        " ",
        text
    )

    # Repair words split across lines.
    text = re.sub(
        r"(?<=\w)-\n(?=\w)",
        "",
        text
    )

    # Normalize spaces and blank lines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# --------------------------------------------------
# PDF EXTRACTION AND OCR
# --------------------------------------------------

def extract_and_clean_pages(pages, pdf_path):
    """Extract text and apply OCR to pages requiring OCR."""

    processed_pages = []

    for page_data in pages:

        page_number = page_data["page"]
        text = page_data.get("text", "")

        if page_data.get("needs_ocr", False):

            print(f"OCR required on page {page_number}")

            try:
                ocr_text = extract_text_using_ocr(
                    pdf_path,
                    page_number
                )

                if ocr_text.strip():
                    text = ocr_text

            except Exception as error:
                print(
                    f"OCR warning on page {page_number}: {error}"
                )

        text = clean_text(text)

        if text:
            processed_pages.append({
                "page": page_number,
                "text": text
            })

    return processed_pages


# --------------------------------------------------
# FIXED-SIZE CHUNKING
# --------------------------------------------------

def create_fixed_size_chunks(pages):
    """Create 300-token chunks with 60-token overlap."""

    all_tokens = []
    token_pages = []

    for page in pages:

        page_number = page["page"]
        page_tokens = TOKENIZER.encode(page["text"])

        all_tokens.extend(page_tokens)
        token_pages.extend(
            [page_number] * len(page_tokens)
        )

        separator_tokens = TOKENIZER.encode("\n")

        all_tokens.extend(separator_tokens)
        token_pages.extend(
            [page_number] * len(separator_tokens)
        )

    chunks = []
    start = 0
    chunk_id = 1

    while start < len(all_tokens):

        end = min(
            start + CHUNK_SIZE,
            len(all_tokens)
        )

        chunk_tokens = all_tokens[start:end]
        chunk_text = TOKENIZER.decode(chunk_tokens)

        page_range = token_pages[start:end]

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "token_count": len(chunk_tokens),
            "page_start": min(page_range),
            "page_end": max(page_range),
            "section_title": detect_section_title(chunk_text),
            "start_token": start,
            "end_token": end
        })

        chunk_id += 1

        if end >= len(all_tokens):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# --------------------------------------------------
# STRUCTURE-AWARE CHUNKING
# --------------------------------------------------

def is_table_of_contents_page(text):
    """Identify likely table-of-contents pages."""

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return False

    if "TABLE OF CONTENTS" in text.upper():
        return True

    toc_lines = sum(
        1 for line in lines
        if re.search(r"\.{3,}\s*\d+\s*$", line)
    )

    return (
        toc_lines >= 3
        and toc_lines / len(lines) >= 0.15
    )


def create_structure_aware_chunks(pages):
    """Create structure-aware chunks and preserve page metadata."""

    sections = []
    current_title = "General"
    current_lines = []

    # Step 1: Identify sections across PDF pages.
    for page in pages:

        page_number = page["page"]
        text = page["text"]

        if is_table_of_contents_page(text):
            print(
                f"Skipping Table of Contents page {page_number}"
            )
            continue

        for line in text.splitlines():

            line = line.strip()

            if not line:
                continue

            if is_heading(line):

                if current_lines:
                    sections.append({
                        "title": current_title,
                        "lines": current_lines
                    })

                current_title = line
                current_lines = []

            current_lines.append(
                (line, page_number)
            )

    if current_lines:
        sections.append({
            "title": current_title,
            "lines": current_lines
        })

    # Step 2: Build chunks while preserving page information.
    chunks = []

    buffer_tokens = []
    buffer_pages = []
    buffer_titles = []

    def save_buffer():

        nonlocal buffer_tokens
        nonlocal buffer_pages
        nonlocal buffer_titles

        if not buffer_tokens:
            return

        chunks.append({
            "chunk_id": len(chunks) + 1,
            "text": TOKENIZER.decode(buffer_tokens),
            "token_count": len(buffer_tokens),
            "page_start": min(buffer_pages),
            "page_end": max(buffer_pages),
            "section_title": " | ".join(
                dict.fromkeys(buffer_titles)
            )
        })

        buffer_tokens = []
        buffer_pages = []
        buffer_titles = []

    # Step 3: Split long sections into 300-token chunks.
    for section in sections:

        for line, page_number in section["lines"]:

            line_tokens = TOKENIZER.encode(line + "\n")
            position = 0

            while position < len(line_tokens):

                available = CHUNK_SIZE - len(buffer_tokens)

                if available == 0:
                    save_buffer()
                    available = CHUNK_SIZE

                take = min(
                    available,
                    len(line_tokens) - position
                )

                selected_tokens = line_tokens[
                    position:position + take
                ]

                buffer_tokens.extend(selected_tokens)
                buffer_pages.extend([page_number] * take)

                if section["title"] not in buffer_titles:
                    buffer_titles.append(section["title"])

                position += take

                if len(buffer_tokens) >= CHUNK_SIZE:
                    save_buffer()

    # Preserve the final short chunk.
    save_buffer()

    return chunks


# --------------------------------------------------
# SELECT CHUNKING STRATEGY
# --------------------------------------------------

def create_document_chunks(pages, strategy):

    if strategy == "fixed":
        return create_fixed_size_chunks(pages)

    if strategy == "structure":
        return create_structure_aware_chunks(pages)

    raise ValueError(
        f"Unknown chunking strategy: {strategy}"
    )


# --------------------------------------------------
# RECURSIVE PDF DISCOVERY
# --------------------------------------------------

def discover_pdf_files(input_folder):
    """Find all PDFs, including those in subfolders."""

    pdf_files = []

    for root, directories, filenames in os.walk(input_folder):

        directories.sort()

        for filename in filenames:

            if filename.lower().endswith(".pdf"):

                pdf_files.append(
                    os.path.join(root, filename)
                )

    return sorted(pdf_files)


# --------------------------------------------------
# MAIN INGESTION PIPELINE
# --------------------------------------------------

def ingest_documents(
    input_folder,
    manifest_path,
    collection_name,
    strategy
):

    print("\nLoading document manifest...")

    manifest = load_manifest(manifest_path)

    print("Documents in manifest:", len(manifest))

    if not os.path.isdir(input_folder):
        raise FileNotFoundError(
            f"PDF folder not found: {input_folder}"
        )

    pdf_files = discover_pdf_files(input_folder)

    print("PDF files discovered:", len(pdf_files))

    if not pdf_files:
        raise ValueError("No PDF files found.")

    # Match filenames to manifest records.
    manifest_by_filename = {}

    for doc_id, info in manifest.items():

        filename = info["local_filename"].strip()

        if filename in manifest_by_filename:
            raise ValueError(
                f"Duplicate filename in manifest: {filename}"
            )

        manifest_by_filename[filename] = (doc_id, info)

    discovered_names = [
        os.path.basename(path)
        for path in pdf_files
    ]

    missing_from_manifest = [
        name for name in discovered_names
        if name not in manifest_by_filename
    ]

    if missing_from_manifest:
        raise ValueError(
            "These PDFs are missing from the manifest: "
            + ", ".join(missing_from_manifest)
        )

    vector_store = VectorStore(
        collection_name=collection_name
    )

    embedding_generator = EmbeddingGenerator()

    print("\nChunking strategy:", strategy)
    print("Collection:", collection_name)

    total_processed = 0
    total_skipped = 0
    successful_documents = 0
    failed_documents = 0

    for pdf_path in pdf_files:

        filename = os.path.basename(pdf_path)

        print("\n" + "=" * 60)
        print("Processing:", filename)
        print("=" * 60)

        try:

            # Find document in manifest.
            doc_id, document_info = manifest_by_filename[filename]

            # Calculate source hash.
            file_hash = calculate_file_hash(pdf_path)

            # Check existing records in this collection.
            existing = vector_store.collection.get(
                where={"doc_id": doc_id},
                include=["metadatas"]
            )

            if existing["ids"]:

                old_hashes = {
                    metadata.get("source_hash")
                    for metadata in (existing.get("metadatas") or [])
                }

                old_strategies = {
                    metadata.get("chunking_strategy")
                    for metadata in (existing.get("metadatas") or [])
                }

                if (
                    old_hashes == {file_hash}
                    and old_strategies == {strategy}
                ):

                    print(
                        "Document unchanged. Skipping."
                    )

                    total_skipped += 1
                    successful_documents += 1
                    continue

            # Extract PDF text.
            print("Extracting PDF text...")

            pages = extract_text_from_pdf(pdf_path)

            # Apply OCR and clean text.
            pages = extract_and_clean_pages(
                pages,
                pdf_path
            )

            if not pages:
                raise ValueError(
                    "No usable text extracted from this PDF."
                )

            print("Pages extracted:", len(pages))

            # Generate chunks.
            chunks = create_document_chunks(
                pages,
                strategy
            )

            if not chunks:
                raise ValueError(
                    "No chunks generated."
                )

            # Attach metadata.
            for chunk in chunks:

                chunk["metadata"] = {
                    "doc_id": doc_id,
                    "title": document_info.get("title", ""),
                    "issuer": document_info.get("issuer", ""),
                    "issue_date": document_info.get("issue_date", ""),
                    "source_hash": file_hash,
                    "chunking_strategy": strategy,
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "section_title": chunk["section_title"],
                    "token_count": chunk["token_count"]
                }

            token_counts = [
                chunk["token_count"]
                for chunk in chunks
            ]

            print("Chunks created:", len(chunks))
            print(
                "Average tokens:",
                round(sum(token_counts) / len(token_counts), 2)
            )
            print("Minimum tokens:", min(token_counts))
            print("Maximum tokens:", max(token_counts))

            # Generate embeddings before replacing old records.
            print("\nGenerating embeddings...")

            texts = [
                chunk["text"]
                for chunk in chunks
            ]

            embeddings = embedding_generator.generate_embeddings(
                texts
            )

            if len(embeddings) != len(chunks):
                raise ValueError(
                    "Embedding count does not match chunk count."
                )

            # Replace previous records only after successful extraction
            # and embedding generation.
            if existing["ids"]:

                print("Replacing previous document records...")

                vector_store.collection.delete(
                    where={"doc_id": doc_id}
                )

            # Store new chunks.
            vector_store.add_documents(
                chunks,
                embeddings
            )

            total_processed += len(chunks)
            successful_documents += 1

            print(
                f"Successfully ingested {filename}"
            )

        except Exception as error:

            failed_documents += 1

            print(
                f"ERROR processing {filename}: {error}"
            )

    # Final summary.
    print("\n" + "=" * 60)
    print("INGESTION PROCESS COMPLETED")
    print("=" * 60)

    print("Documents discovered:", len(pdf_files))
    print("Documents successful:", successful_documents)
    print("Documents failed:", failed_documents)
    print("New chunks processed:", total_processed)
    print("Documents skipped:", total_skipped)

    print(
        "Total chunks in collection:",
        vector_store.collection.count()
    )

    if failed_documents:
        print(
            "\nWARNING: Some documents failed. "
            "Review the errors above before submission."
        )


# --------------------------------------------------
# COMMAND LINE INTERFACE
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="PolicyLens PDF ingestion pipeline"
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Folder containing PDF documents"
    )

    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Path to manifest CSV"
    )

    parser.add_argument(
        "--collection",
        required=True,
        help="ChromaDB collection name"
    )

    parser.add_argument(
        "--strategy",
        choices=["fixed", "structure"],
        required=True,
        help="Chunking strategy"
    )

    args = parser.parse_args()

    ingest_documents(
        input_folder=args.input,
        manifest_path=args.manifest,
        collection_name=args.collection,
        strategy=args.strategy
    )


if __name__ == "__main__":
    main()
