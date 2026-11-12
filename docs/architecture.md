# PolicyLens – System Architecture

## Document Ingestion Pipeline

```mermaid
flowchart TD
    A["30 RBI PDF Documents"] --> B["PDF Ingestion"]
    B --> C["PyMuPDF Text Extraction"]
    B --> D["OCR Fallback for Scanned PDFs"]
    C --> E["Extracted Text"]
    D --> E
    E --> F["Text Cleaning and Normalization"]
    F --> G["Chunking"]
    G --> H["Fixed-size Chunking"]
    G --> I["Structure-aware Chunking"]
    H --> J["Metadata Enrichment"]
    I --> J
    J --> K["E5 Embedding Model"]
    K --> L[("ChromaDB Vector Store")]
    N["User Query"] --> M["Semantic Search"]
    L --> M
    O["Metadata Filters"] --> M
    M --> P["Ranked Relevant Passages"]
    Q["25 Evaluation Questions"] --> R["Retrieval Evaluation"]
    L --> R
    R --> S["Recall@3, Recall@5, Recall@10 and MRR"]
```

## Evaluation Results

| Metric | Structure-aware | Fixed-size |
|---|---:|---:|
| Recall@3 | 88% | 72% |
| Recall@5 | 96% | 84% |
| Recall@10 | 100% | 92% |
| MRR | 0.7743 | 0.7613 |

**Preferred strategy:** Structure-aware chunking performed better overall on the 25-question evaluation dataset.
