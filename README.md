
# PolicyLens – RBI Policy Document Search and Retrieval System

PolicyLens is a document search and retrieval system designed to help users find relevant information from Reserve Bank of India (RBI) policy documents, reports, circulars, and statistical publications.

It uses PDF text extraction, OCR, text chunking, sentence embeddings, and a ChromaDB vector database to retrieve relevant passages based on natural-language queries.

## 1. Project Objectives

- Build a searchable collection of RBI policy documents.
- Extract text from both digital and scanned PDF files.
- Apply OCR when PDF pages contain insufficient extractable text.
- Divide extracted content into manageable text chunks.
- Generate semantic embeddings using a transformer model.
- Store document chunks and metadata in ChromaDB.
- Retrieve relevant passages using natural-language queries.
- Support metadata-based filtering.
- Evaluate retrieval performance using a manually prepared question dataset.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| PDF extraction | PyMuPDF |
| OCR | Tesseract OCR, pytesseract |
| Image processing | Pillow |
| Text processing | Python |
| Tokenization | tiktoken |
| Embedding model | intfloat/e5-base-v2 |
| Vector database | ChromaDB |
| Machine learning framework | Sentence Transformers |
| Command-line interface | argparse |
| Evaluation | Python, CSV |

## 3. Dataset

The project uses a collection of 30 RBI PDF documents.

The corpus includes:

- Text-native documents
- Scanned documents
- Table-heavy documents

The documents cover RBI publications such as:

- Annual Reports
- Financial Stability Reports
- KYC Master Directions
- Banking regulation documents
- RBI statistical publications
- Credit facility guidelines
- Banking trends and progress reports

The document metadata is maintained in:

`data/manifest.csv`

The manifest includes document IDs, titles, issuers, dates, URLs, page counts, document types, and local filenames.

## 4. System Architecture

The PolicyLens workflow consists of two main stages.

### A. Document Ingestion

1. Read document metadata from the CSV manifest.
2. Discover PDF files in the input directory.
3. Extract text from PDF pages using PyMuPDF.
4. Apply OCR to pages requiring text recognition.
5. Clean and preprocess extracted text.
6. Divide the content into chunks using the structure-aware strategy.
7. Generate embeddings using the E5 embedding model.
8. Store document chunks, embeddings, and metadata in ChromaDB.

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
│   └── retrieval_log_30docs.csv
│
├── evidence/
├── logs/
├── report/
│
├── ingest.py
├── ingest_backup.py
├── ingest_structure.py
├── search.py
├── check_token_lengths.py
├── diagnose_chunks.py
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

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## 7. Running the Project

### Step 1: Document Ingestion

Run the ingestion pipeline:

```bash
python ingest.py \
  --input data/pdfs \
  --manifest data/manifest.csv \
  --collection policylens_final_30docs \
  --strategy structure
```

This processes the PDF corpus and stores the resulting embeddings and metadata in ChromaDB.

### Step 2: Semantic Search

Run a natural-language search:

```bash
python search.py "What are the KYC requirements for bank customers?" --top_k 5
```

### Step 3: Search with Metadata Filters

Filter results by issuer:

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

The search interface displays:

- Result rank
- Similarity score
- Document title
- Issuer
- Document ID
- Page range
- Section title
- Retrieved text

## 8. Retrieval Evaluation

The retrieval system is evaluated using a manually prepared dataset of 25 questions.

The evaluation script measures:

- Recall@3
- Recall@5
- Recall@10
- Mean Reciprocal Rank (MRR)

Run the evaluation:

```bash
python eval/evaluate.py \
  --collection policylens_final_30docs \
  --top_k 10
```

### Evaluation Results

| Metric | Result |
|---|---:|
| Recall@3 | 88% |
| Recall@5 | 96% |
| Recall@10 | 100% |
| MRR | 0.7743 |

These results are based on the 25-question evaluation dataset and its manually assigned expected document and page references.

The evaluation measures retrieval performance, not the factual correctness of generated answers.

## 9. Database

PolicyLens uses ChromaDB as its persistent vector database.

The database is stored locally at:

`data/chroma_db/`

The active collection is:

`policylens_final_30docs`

It stores document chunks, vector embeddings, and associated metadata.

## 10. Limitations

- OCR output may contain errors, particularly in historical scanned documents.
- Complex tables may not always be extracted in their original layout.
- Similarity scores indicate vector similarity and are not probability or confidence scores.
- Evaluation currently uses a limited dataset of 25 manually prepared questions.
- Retrieval quality depends on document extraction, chunking, and embedding quality.
- The current command-line interface retrieves relevant passages; it does not independently generate verified answers.

## 11. Future Enhancements

- Add a conversational question-answering interface.
- Integrate Retrieval-Augmented Generation (RAG).
- Improve OCR quality for historical and table-heavy documents.
- Add source-linked citations in generated answers.
- Develop a web-based user interface.
- Expand the evaluation dataset.
- Improve hybrid retrieval using semantic and keyword search.

## 12. Conclusion

PolicyLens demonstrates how semantic search and vector databases can be used to organize and retrieve information from a collection of RBI documents.

By combining PDF extraction, OCR, structure-aware chunking, transformer-based embeddings, and ChromaDB, the system provides a foundation for searching large collections of financial and regulatory documents.

The project can be extended into a complete Retrieval-Augmented Generation system capable of producing answers grounded in retrieved RBI source documents.
