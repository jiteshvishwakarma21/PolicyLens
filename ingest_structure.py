
import os
import csv
import re
import hashlib
import argparse

from src.pdf_extractor import extract_text_from_pdf
from src.ocr_processor import extract_text_using_ocr
from src.text_cleaner import clean_text
from src.document_chunker import TOKENIZER
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import VectorStore


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 300


def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(4096), b""):
            sha256.update(block)

    return sha256.hexdigest()


def load_manifest(manifest_path):
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
                f"Missing manifest columns: {sorted(missing)}"
            )

        for row in reader:
            documents[row["local_filename"].strip()] = row

    return documents


def is_heading(line):
    """
    Identify likely chapter and section headings.
    Avoid treating numbered regulatory sentences as headings.
    """

    line = re.sub(r"\s+", " ", line).strip()

    if not line or len(line) > 150:
        return False

    # Chapter headings, including Chapter IIA.
    chapter_pattern = (
        r"^Chapter\s+[IVXLCDM]+[A-Z]?"
        r"\s*[-:]\s*\S.{1,120}$"
    )

    if re.match(chapter_pattern, line, re.IGNORECASE):
        return True

    # Reject common sentence beginnings.
    sentence_starters = (
        "a ", "an ", "the ", "if ", "where ", "when ",
        "in case ", "in the event ", "provided ",
        "notwithstanding ", "every ", "any ", "each "
    )

    lower_line = line.lower()

    # Numbered section heading, such as:
    # 6. Due diligence requirements with respect to LSPs
    numbered_heading = re.match(
        r"^\d{1,3}[A-Z]?\.\s+(.+)$",
        line
    )

    if numbered_heading:
        heading_text = numbered_heading.group(1).strip()
        lower_heading = heading_text.lower()

        if lower_heading.startswith(sentence_starters):
            return False

        # Regulatory sentences commonly contain these words.
        if re.search(
            r"\b(shall|must|should|may|is|are|was|were|"
            r"will|has|have|had|be|being|been)\b",
            heading_text,
            re.IGNORECASE
        ):
            return False

        # Sentences often contain punctuation.
        if re.search(r"[,;:]", heading_text):
            return False

        if heading_text.endswith("."):
            return False

        # Keep short, title-like numbered headings.
        words = heading_text.split()

        if 2 <= len(words) <= 15:
            return True

    # Uppercase headings, excluding short labels.
    if (
        line.isupper()
        and 5 <= len(line) <= 100
        and len(line.split()) >= 2
    ):
        return True

    return False


def extract_sections(pages, pdf_path):
    """
    Extract page text and divide it into heading-based sections.
    Every token retains its original page number.
    """

    sections = []

    current_title = "General"
    current_tokens = []
    current_pages = []

    def save_section():
        nonlocal current_tokens, current_pages

        if current_tokens:
            sections.append({
                "title": current_title,
                "tokens": current_tokens,
                "pages": current_pages
            })

        current_tokens = []
        current_pages = []

    for page in pages:
        page_number = page["page"]
        page_text = page["text"]

        if page["needs_ocr"]:
            print(f"Running OCR on page {page_number}...")
            page_text = extract_text_using_ocr(
                pdf_path,
                page_number
            )

        page_text = clean_text(page_text)

        if not page_text:
            continue

        for line in page_text.splitlines():
            line = re.sub(r"\s+", " ", line).strip()

            if not line:
                continue

            if is_heading(line):
                save_section()
                current_title = line[:200]

                heading_tokens = TOKENIZER.encode(line)
                current_tokens.extend(heading_tokens)
                current_pages.extend(
                    [page_number] * len(heading_tokens)
                )

            else:
                line_tokens = TOKENIZER.encode(line)

                if line_tokens:
                    # Add a separating space between extracted lines.
                    separator = TOKENIZER.encode(" ")

                    current_tokens.extend(separator)
                    current_pages.extend(
                        [page_number] * len(separator)
                    )

                    current_tokens.extend(line_tokens)
                    current_pages.extend(
                        [page_number] * len(line_tokens)
                    )

        # Do not close the section at each page boundary.
        # A section may continue onto the next page.

    save_section()

    return sections


def merge_small_sections(sections):
    """
    Combine neighboring short sections when they fit within
    the maximum chunk size.
    """

    merged = []

    for section in sections:
        current = {
            "title": section["title"],
            "tokens": list(section["tokens"]),
            "pages": list(section["pages"])
        }

        if not merged:
            merged.append(current)
            continue

        previous = merged[-1]

        combined_size = (
            len(previous["tokens"]) + len(current["tokens"])
        )

        previous_is_small = (
            len(previous["tokens"]) < MIN_CHUNK_SIZE
        )

        current_is_small = (
            len(current["tokens"]) < MIN_CHUNK_SIZE
        )

        if (
            combined_size <= CHUNK_SIZE
            and (previous_is_small or current_is_small)
        ):
            previous["tokens"].extend(current["tokens"])
            previous["pages"].extend(current["pages"])

            # Keep the first meaningful section title.
            if previous["title"] == "General":
                previous["title"] = current["title"]

        else:
            merged.append(current)

    return merged


def create_structure_aware_chunks(pages, pdf_path):
    sections = extract_sections(pages, pdf_path)

    sections = merge_small_sections(sections)

    chunks = []

    for section in sections:
        tokens = section["tokens"]
        page_numbers = section["pages"]

        if not tokens:
            continue

        # Short sections that could not be merged remain intact.
        if len(tokens) <= CHUNK_SIZE:
            chunks.append({
                "chunk_id": len(chunks) + 1,
                "text": TOKENIZER.decode(tokens),
                "metadata": {
                    "page_start": min(page_numbers),
                    "page_end": max(page_numbers),
                    "section_title": section["title"][:200],
                    "token_count": len(tokens)
                }
            })
            continue

        # Split only oversized sections, with overlap.
        step_size = CHUNK_SIZE - CHUNK_OVERLAP

        for start in range(0, len(tokens), step_size):
            end = min(start + CHUNK_SIZE, len(tokens))

            chunk_tokens = tokens[start:end]
            chunk_pages = page_numbers[start:end]

            if not chunk_tokens:
                continue

            chunks.append({
                "chunk_id": len(chunks) + 1,
                "text": TOKENIZER.decode(chunk_tokens),
                "metadata": {
                    "page_start": min(chunk_pages),
                    "page_end": max(chunk_pages),
                    "section_title": section["title"][:200],
                    "token_count": len(chunk_tokens)
                }
            })

            if end >= len(tokens):
                break

    return chunks


def ingest_documents(
    input_folder,
    manifest_path,
    collection_name
):
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(
            f"PDF folder not found: {input_folder}"
        )

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    manifest = load_manifest(manifest_path)

    embedding_generator = EmbeddingGenerator()
    vector_store = VectorStore(
        collection_name=collection_name
    )

    pdf_files = sorted(
        name for name in os.listdir(input_folder)
        if name.lower().endswith(".pdf")
    )

    if not pdf_files:
        print("No PDF files found.")
        return

    total_chunks = 0

    for file_name in pdf_files:
        pdf_path = os.path.join(input_folder, file_name)

        print(f"\nProcessing: {file_name}")

        document_info = manifest.get(file_name)

        if document_info is None:
            print("Skipped: PDF is not listed in the manifest.")
            continue

        doc_id = document_info["doc_id"].strip()
        source_hash = calculate_file_hash(pdf_path)

        existing = vector_store.collection.get(
            where={"doc_id": doc_id},
            include=["metadatas"]
        )

        existing_ids = existing.get("ids", [])
        existing_metadata = existing.get("metadatas", [])

        existing_hashes = {
            metadata.get("source_hash")
            for metadata in existing_metadata
            if metadata
        }

        if existing_ids and existing_hashes == {source_hash}:
            print("Skipped: Document is already up to date.")
            continue

        print("Extracting PDF pages...")
        pages = extract_text_from_pdf(pdf_path)

        print("Creating improved structure-aware chunks...")
        chunks = create_structure_aware_chunks(
            pages,
            pdf_path
        )

        if not chunks:
            print("Skipped: No chunks were generated.")
            continue

        document_metadata = {
            "doc_id": doc_id,
            "title": document_info["title"],
            "issuer": document_info["issuer"],
            "issue_date": document_info["issue_date"],
            "source_hash": source_hash
        }

        for chunk in chunks:
            chunk["metadata"].update(document_metadata)

        texts = [chunk["text"] for chunk in chunks]

        token_counts = [
            chunk["metadata"]["token_count"]
            for chunk in chunks
        ]

        print("Generating embeddings...")

        embeddings = embedding_generator.generate_embeddings(
            texts,
            batch_size=32
        )

        # Replace old records only after embeddings are ready.
        if existing_ids:
            vector_store.collection.delete(ids=existing_ids)

        vector_store.add_documents(chunks, embeddings)

        total_chunks += len(chunks)

        below_300 = sum(
            count < MIN_CHUNK_SIZE for count in token_counts
        )

        print(f"Document ID: {doc_id}")
        print(f"Pages processed: {len(pages)}")
        print(f"Chunks created: {len(chunks)}")
        print(
            f"Average tokens: "
            f"{sum(token_counts) / len(token_counts):.2f}"
        )
        print(f"Minimum tokens: {min(token_counts)}")
        print(f"Maximum tokens: {max(token_counts)}")
        print(f"Chunks below 300 tokens: {below_300}")
        print("Ingestion completed for this document.")

    print("\nIMPROVED STRUCTURE-AWARE INGESTION SUMMARY")
    print(f"Collection: {collection_name}")
    print(f"Total new chunks indexed: {total_chunks}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PolicyLens improved structure-aware ingestion"
    )

    parser.add_argument(
        "--input",
        default="data/pdfs",
        help="Folder containing PDF documents"
    )

    parser.add_argument(
        "--manifest",
        default="data/manifest.csv",
        help="Path to the document manifest"
    )

    parser.add_argument(
        "--collection",
        default="policylens_structure_v2",
        help="ChromaDB collection name"
    )

    args = parser.parse_args()

    ingest_documents(
        args.input,
        args.manifest,
        args.collection
    )
