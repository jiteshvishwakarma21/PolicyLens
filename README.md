# PolicyLens – RBI Policy Document Search and Retrieval System

PolicyLens is a document ingestion and semantic retrieval system designed to help users find relevant information from Reserve Bank of India (RBI) policy documents, reports, circulars and statistical publications.

It combines PDF text extraction, OCR, text cleaning, chunking, transformer-based embeddings and ChromaDB to retrieve relevant passages using natural-language queries.

## 1. Project Objectives

- Build a searchable collection of 30 public RBI PDF documents.
- Extract text from digital and scanned PDFs.
- Apply OCR when PDF pages contain insufficient extractable text.
- Clean and normalize extracted content.
- Implement fixed-size and structure-aware chunking.
- Generate semantic embeddings using a transformer model.
- Store document chunks and metadata in ChromaDB.
- Retrieve relevant passages using natural-language queries.
- Support metadata-based filtering.
- Evaluate retrieval performance using a manually prepared question dataset.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.12 |
| PDF extraction | PyMuPDF |
| OCR | Tesseract OCR, pytesseract |
| Image processing | Pillow |
| Tokenization | tiktoken |
| Embedding model | intfloat/e5-base-v2 |
| Vector database | ChromaDB |
| Machine learning framework | Sentence Transformers |
| Command-line interface | argparse |
| Evaluation | Python, CSV |

## 3. Dataset

The project uses 30 public RBI PDF documents.

| Document category | Count |
|---|---:|
| Text-native | 19 |
| Scanned | 6 |
| Table-heavy | 5 |
| **Total** | **30** |

The documents cover RBI publications such as annual reports, financial stability reports, KYC directions, banking regulations, statistical publications and credit facility guidelines.

Document metadata is maintained in `data/manifest.csv`.

The manifest includes document IDs, titles, issuers, dates, URLs, page counts, document types and local filenames.

## 4. System Architecture

PolicyLens consists of two main stages.

### A. Document Ingestion

1. Read document metadata from the CSV manifest.
2. Discover PDF files in the input directory.
3. Extract page text using PyMuPDF.
4. Apply OCR to pages requiring text recognition.
5. Clean and normalize extracted text.
6. Divide content into chunks.
7. Enrich chunks with document and page metadata.
8. Generate embeddings using the E5 embedding model.
9. Store document chunks, embeddings and metadata in ChromaDB.

### B. Semantic Search and Retrieval

1. Accept a natural-language search query.
2. Generate a query embedding.
3. Search the ChromaDB collection using vector similarity.
4. Retrieve the most relevant document chunks.
5. Display ranked results with similarity scores and metadata.
6. Allow filtering by metadata such as issuer and document ID.

## 5. Project Structure

```text
policylens-final-project/
│
├── data/
│   ├── pdfs/
│   ├── chroma_db/
│   └── manifest.csv
│
├── src/
│   ├── __init__.py
│   ├── document_chunker.py
│   ├── embedding_generator.py
│   ├── hybrid_retrieval.py
│   ├── main.py
│   ├── ocr_processor.py
│   ├── pdf_extractor.py
│   ├── policy_analyzer.py
│   ├── retrieval.py
│   ├── text_cleaner.py
│   └── vector_store.py
│
├── eval/
│   ├── evaluate.py
│   ├── evaluate_hybrid.py
│   ├── questions.csv
│   ├── metrics_30docs.csv
│   ├── metrics_fixed_30docs.csv
│   └── retrieval_log_30docs.csv
│
├── evidence/
├── docs/
├── report/
│
├── ingest.py
├── ingest_backup.py
├── ingest_structure.py
├── search.py
├── requirements.txt
└── README.md
```

## 6. Installation and Setup

### Prerequisites

- Python 3.12
- pip
- Tesseract OCR

Install Tesseract on macOS:

```bash
brew install tesseract
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## 7. Running the Project

### Step 1: Document Ingestion

Run the structure-aware ingestion pipeline:

```bash
python ingest.py \
  --input data/pdfs \
  --manifest data/manifest.csv \
  --collection policylens_final_30docs \
  --strategy structure
```

To use fixed-size chunking:

```bash
python ingest.py \
  --input data/pdfs \
  --manifest data/manifest.csv \
  --collection policylens_fixed_30docs \
  --strategy fixed
```

### Step 2: Semantic Search

```bash
python search.py "What are the KYC requirements for bank customers?" --top_k 5
```

### Step 3: Search with Metadata Filters

Filter by issuer:

```bash
python search.py "What are the KYC requirements?" \
  --top_k 5 \
  --filter "issuer=Reserve Bank of India"
```

Filter by document ID:

```bash
python search.py "What are the KYC requirements?" \
  --top_k 5 \
  --filter doc_id=RBI_KYC_001
```

Search results display:

- Result rank
- Similarity score
- Document title
- Issuer
- Document ID
- Page range
- Section title
- Retrieved text

## 8. Retrieval Evaluation

The retrieval system is evaluated using 25 manually prepared questions:

- 15 standard questions
- 5 paraphrased questions
- 5 table-based questions

The evaluation script measures:

- Recall@3
- Recall@5
- Recall@10
- Mean Reciprocal Rank (MRR)

Run the structure-aware evaluation:

```bash
python eval/evaluate.py \
  --collection policylens_final_30docs \
  --top_k 10
```

Run the fixed-size evaluation:

```bash
python eval/evaluate.py \
  --collection policylens_fixed_30docs \
  --top_k 10
```

### Evaluation Results

| Metric | Structure-aware | Fixed-size |
|---|---:|---:|
| Recall@3 | 88% | 72% |
| Recall@5 | 96% | 84% |
| Recall@10 | 100% | 92% |
| MRR | 0.7743 | 0.7613 |

Structure-aware chunking performed better across all four aggregate metrics on the 25-question evaluation set.

However, fixed-size chunking performed better for selected individual questions, including some table-based queries.

These results measure retrieval performance against manually assigned expected document and page references. They do not measure the factual correctness of generated answers.

## 9. Chunking Strategy Comparison

Two chunking approaches were evaluated using the same 30-document corpus and embedding model.

### Fixed-size chunking

- Maximum chunk size: 300 tokens.
- Overlap: 60 tokens.
- Average chunk size: 299.8 tokens.
- Total chunks: 19,341.

### Structure-aware chunking

- Uses detected document structure and section boundaries.
- Splits long sections into smaller chunks.
- Maximum target size: approximately 300 tokens.
- Total chunks: 15,374.

Structure-aware chunking produced fewer chunks while achieving better aggregate retrieval results in the current evaluation.

## 10. Performance Measurements

The following measurements were collected during project testing.

| Measurement | Result |
|---|---:|
| Controlled PDF ingestion | 12.467 seconds |
| PDF pages processed | 76 |
| Chunks generated in test | 293 |
| Estimated time per 100 pages | 16.40 seconds |
| Observed ChromaDB query latency | 217.66 ms |
| ChromaDB directory size | 429 MB |

**Measurement qualifications:**

- The 12.467-second ingestion test processed one 76-page scanned PDF and included model loading, extraction, chunking, embedding and storage.
- The 16.40-second figure is a linear extrapolation, not a direct 100-page benchmark.
- The 217.66 ms query measurement excludes query-embedding generation and represents one observed query.
- The 429 MB figure represents the previously measured aggregate ChromaDB directory across collections, not an individual final collection.

## 11. OCR and Text Cleaning

PolicyLens uses Tesseract OCR as a fallback for scanned pages.

The OCR process includes image rendering, grayscale conversion, contrast enhancement and orientation detection.

Two pages from the February 1952 RBI Bulletin were manually inspected:

- **Page 17:** Some headings and labels were recovered, but chart-related text contained OCR errors.
- **Page 36:** Major headings and some axis text were recovered, but graph lines and small numerical labels were unreliable.

These are two manual spot checks, not a representative accuracy study. No OCR accuracy percentage or word-error rate was calculated.

The text cleaner normalizes line endings, fixes hyphenated line breaks, removes control characters and normalizes whitespace.

Example:

Before:

```text
Financial inclu-
sion is important.
```

After:

```text
Financial inclusion is important.
```

The current cleaner does not systematically remove repeating headers, footers or page numbers.

## 12. Database and Reliability

PolicyLens uses ChromaDB as its persistent vector database.

Database location:

`data/chroma_db/`

Final collections:

- `policylens_final_30docs`
- `policylens_fixed_30docs`

The final structure-aware collection contains 15,374 chunks, and the fixed-size collection contains 19,341 chunks.

A repeat structure-aware ingestion run skipped all 30 unchanged documents and added zero new chunks, demonstrating idempotent behaviour for the unchanged corpus.

The project uses source hashes to help identify documents that require reprocessing.

## 13. Limitations

- OCR output may contain errors, particularly in historical scanned documents.
- Complex tables and charts may not preserve their original layout.
- Small numerical labels may not be recognized correctly.
- Repeating headers and footers are not systematically removed.
- The evaluation uses only 25 manually prepared questions.
- OCR quality was manually inspected on only two pages.
- Query latency is a single measurement and excludes embedding generation.
- The system retrieves relevant passages but does not independently generate verified answers.
- No paid embedding API cost per 1,000 chunks was measured.

## 14. Future Enhancements

- Expand the evaluation dataset.
- Improve OCR for historical and table-heavy documents.
- Add layout-aware table extraction.
- Improve repeated-header and footer detection.
- Introduce more comprehensive performance benchmarking.
- Add document versioning and withdrawn-circular handling.
- Integrate Retrieval-Augmented Generation (RAG).
- Add source-linked citations in generated answers.
- Develop a web-based user interface.
- Improve hybrid retrieval using semantic and keyword search.

## 15. Conclusion

PolicyLens demonstrates how semantic search and vector databases can organize and retrieve information from RBI documents.

By combining PDF extraction, OCR, text cleaning, structure-aware and fixed-size chunking, transformer-based embeddings and ChromaDB, the system provides a foundation for searching regulatory and financial documents.

On the current 25-question evaluation set, structure-aware chunking achieved Recall@3 of 88%, Recall@5 of 96%, Recall@10 of 100% and MRR of 0.7743.

The project can be extended into a complete Retrieval-Augmented Generation system capable of producing answers grounded in retrieved RBI source documents.

## 16. References and Project Artifacts

- Reserve Bank of India – public RBI documents included in `data/manifest.csv`.
- intfloat/e5-base-v2 – embedding model.
- ChromaDB – vector database.
- Project source code, evaluation datasets, metrics, retrieval logs and evidence are maintained in this repository.
