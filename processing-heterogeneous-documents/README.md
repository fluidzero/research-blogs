# Processing Heterogeneous Documents at 100GB+ Scale

Enterprise document repositories are not curated datasets. They are decades of accumulated PDFs, scanned images, Word documents, spreadsheets, CAD exports, and legacy formats — often with inconsistent naming, duplicate versions, and no centralized metadata. When a customer tells us they have "about 100,000 documents," the reality is 100,000 files in 15+ formats, spanning 20 years, totaling anywhere from 50GB to 500GB.

This post describes the architecture we built to ingest, process, and index these repositories at scale. We cover the design of our processing pipeline, the chunking strategies that preserve document structure, the indexing approach that enables sub-second queries, and the operational patterns we developed from processing hundreds of real-world enterprise document sets.

## Scale Characteristics

Before diving into architecture, it helps to understand what "100GB+ of enterprise documents" actually looks like:

| Metric | Typical Range |
|--------|---------------|
| Total files | 50,000 – 500,000 |
| Unique formats | 8 – 15 |
| Total pages | 500,000 – 5,000,000 |
| Average file size | 200KB – 5MB |
| Largest single file | 50MB – 200MB (spec bundles, scanned manuals) |
| Duplicate rate | 10 – 25% (versions, copies across directories) |
| OCR required | 15 – 40% (scanned documents) |

These are not synthetic benchmarks. They are measured ranges from our production deployments. The variance is important — it means the system must handle both a directory of 10,000 clean PDFs and a mixed bag of legacy formats without configuration changes.

## Pipeline Architecture

Our processing pipeline is designed around three principles:

1. **Format-agnostic ingestion** — every document enters the same pipeline regardless of source format
2. **Structural preservation** — the output preserves document structure (sections, tables, figures, cross-references) rather than flattening to plain text
3. **Incremental processing** — adding new documents does not require reprocessing the entire corpus

```
  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────┐
  │  File Store  │───▶│  Normalizer  │───▶│  Processor   │───▶│  Index  │
  └─────────────┘    └──────────────┘    └──────────────┘    └─────────┘
       │                    │                   │                  │
       │              Format detection     Structure-aware     Embedding
       │              Deduplication         extraction         generation
       │              OCR routing          Chunking            Vector index
       │                                  Metadata             Full-text index
```

### Stage 1: Normalization

Every incoming file passes through the normalization stage, which performs three operations:

**Format detection and conversion.** We do not trust file extensions. A `.pdf` might be a scanned image masquerading as a PDF. A `.doc` might be a renamed `.docx`. We detect actual format by inspecting file headers and magic bytes, then route to the appropriate converter.

```python
FORMAT_HANDLERS = {
    "application/pdf": PdfHandler,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocxHandler,
    "application/vnd.ms-excel": XlsHandler,
    "image/tiff": ImageHandler,
    "image/jpeg": ImageHandler,
    "image/png": ImageHandler,
    "application/octet-stream": FallbackHandler,  # binary detection
}
```

**Content-based deduplication.** Enterprise repositories accumulate duplicates — the same document saved in multiple directories, renamed, or converted between formats. We compute a content hash (SHA-256 of extracted text, normalized for whitespace) and deduplicate at ingestion time. This typically reduces the processing volume by 10–25%.

**OCR routing.** Documents with embedded text are processed directly. Scanned documents and image-based PDFs are routed through our OCR pipeline. The routing decision uses a simple heuristic: if a PDF page yields fewer than 50 characters of extractable text but has image content, it's treated as scanned.

### Stage 2: Structure-Aware Processing

This is where most document AI systems lose information. The common approach is to extract text and discard structure. We preserve it.

**Layout analysis** identifies structural elements on each page:

- Headings (with hierarchy level: H1, H2, H3...)
- Body paragraphs
- Tables (with cell structure)
- Figures and diagrams
- Captions
- Headers/footers (excluded from search content but preserved for metadata)
- Page numbers and cross-references

**Section tree construction** assembles the flat list of elements into a hierarchical tree that mirrors the document's logical structure:

```
Document
├── Section 1: Introduction
│   ├── Paragraph: "This specification defines..."
│   └── Table 1.1: Summary of requirements
├── Section 2: Technical Requirements
│   ├── Section 2.1: Materials
│   │   ├── Paragraph: "All structural members shall..."
│   │   └── Table 2.1: Material properties
│   └── Section 2.2: Dimensions
│       ├── Paragraph: "Refer to Figure 2.1 for..."
│       └── Figure 2.1: Assembly drawing
└── Appendix A: Test Reports
```

This tree is the basis for our chunking strategy.

### Stage 3: Structure-Aware Chunking

Chunking — splitting documents into segments for embedding and retrieval — is the operation that most directly affects search quality. Get it wrong, and no amount of model improvement will compensate.

**The standard approach** (split every N tokens with M overlap) destroys document structure. A chunk boundary might split a table in half, separate a heading from its content, or break a numbered list across two chunks. The retrieved "context" is then a fragment that lacks the structural cues humans use to understand it.

**Our approach** uses the section tree to create chunks that respect document boundaries:

```python
def chunk_section(section: SectionNode, config: ChunkConfig) -> list[Chunk]:
    """
    Recursively chunk a section tree, respecting structural boundaries.
    """
    # Base case: leaf section fits in one chunk
    if section.token_count <= config.max_tokens:
        return [Chunk(
            content=section.render(),
            metadata=section.metadata,
            bbox=section.bbox,
            section_path=section.path,  # e.g., "2.1.3"
        )]

    # Section is too large — split at subsection boundaries
    chunks = []
    current = ChunkBuilder(config)

    for child in section.children:
        if current.would_exceed(child):
            chunks.append(current.build())
            current = ChunkBuilder(config)
        current.add(child)

    if not current.is_empty():
        chunks.append(current.build())

    return chunks
```

Key design decisions in our chunking strategy:

| Decision | Rationale |
|----------|-----------|
| Never split tables | A partial table is useless for retrieval |
| Keep headings with content | Section headers provide critical context for the text that follows |
| Include parent section path | Enables the query "in Section 2.1, what is..." |
| Preserve figure-caption pairs | A caption without its figure (or vice versa) loses meaning |
| Adaptive chunk size | Dense technical sections get smaller chunks; narrative sections get larger ones |

The adaptive sizing deserves explanation. A section containing a single dense table with 50 rows should be one chunk — splitting it would make individual rows unsearchable as a unit. A narrative section of 2,000 tokens should be split into 2–3 chunks to enable more precise retrieval. We use content density (information per token, estimated by entity and numeric value frequency) to adjust the target chunk size.

## Indexing for Sub-Second Queries

With documents chunked and embedded, we need an index that supports three query patterns:

1. **Semantic search** — "find chunks about material strength testing"
2. **Exact value search** — "find the chunk containing the value 450°C"
3. **Filtered search** — "find chunks in Document X, Section 3"

No single index type supports all three efficiently. We use a hybrid approach:

### Vector Index (Semantic Search)

Chunk embeddings are stored in a vector index optimized for approximate nearest neighbor (ANN) search. We use HNSW (Hierarchical Navigable Small World) graphs with the following configuration:

```yaml
vector_index:
  algorithm: hnsw
  dimensions: 768
  metric: cosine
  ef_construction: 200    # build-time quality parameter
  M: 16                   # max connections per node
  ef_search: 100          # query-time quality parameter
```

For a corpus of 1 million chunks, this provides ~5ms query latency with 98.5% recall@10.

### Full-Text Index (Exact Value Search)

Numeric values, part numbers, specification identifiers, and other exact-match content are indexed in a full-text search engine. We extract typed values during chunking:

```python
@dataclass
class ExtractedValue:
    raw_text: str           # "450 °C"
    normalized: float       # 450.0
    unit: str              # "celsius"
    context: str           # "maximum operating temperature"
    chunk_id: str
    position: int          # character offset within chunk
```

This enables queries like "find 450°C" to match "450 °C", "450°C", "450 degrees Celsius", and "723.15 K" (unit conversion).

### Metadata Index (Filtered Search)

Document metadata (title, date, version, section paths) is stored in a structured index that supports filter predicates. This allows scoping any semantic or exact-value search to a subset of the corpus:

```sql
-- Conceptual query (actual implementation uses a structured query DSL)
SELECT chunks FROM index
WHERE document_title LIKE '%Operations Manual%'
  AND section_path STARTS WITH '4.3'
  AND semantic_similarity(query_embedding, chunk_embedding) > 0.7
ORDER BY semantic_similarity DESC
LIMIT 10
```

### Query Fusion

At query time, all three indexes are queried in parallel and results are fused using reciprocal rank fusion (RRF):

```
Score(chunk) = Σ  1 / (k + rank_i(chunk))
               i∈{vector, fulltext, metadata}
```

Where `k` is a constant (we use k=60) that controls how much weight is given to top-ranked results versus lower-ranked ones. This simple formula is surprisingly effective at combining heterogeneous ranking signals.

## Operational Patterns

Processing hundreds of real-world document repositories has taught us patterns that are not obvious from benchmarks.

### The 80/20 Rule of Document Quality

Approximately 80% of documents in any enterprise repository process cleanly with standard pipelines. The remaining 20% require special handling:

- **Corrupted files** (2–5%): Truncated PDFs, password-protected files, zero-byte placeholders
- **Unusual layouts** (5–10%): Multi-column formats, landscape orientation, fold-out pages
- **Legacy formats** (3–8%): WordPerfect, Lotus 1-2-3, proprietary CAD formats
- **Extreme sizes** (1–3%): Single PDFs exceeding 1,000 pages

We handle these with a tiered fallback strategy:

```
Primary pipeline → Format-specific handler → OCR fallback → Manual review queue
```

Each tier is progressively slower and more resource-intensive. The key insight is to process the 80% quickly and queue the 20% for specialized handling rather than slowing down the entire pipeline.

### Incremental Processing

Enterprise document repositories are not static. New documents are added, existing documents are revised, and obsolete documents are archived. Our pipeline supports incremental processing through content-addressed storage:

1. New files are hashed and compared against existing content hashes
2. New documents are processed through the full pipeline
3. Modified documents trigger reprocessing of affected chunks only
4. Deleted documents have their chunks and embeddings removed from all indexes

The incremental approach reduces the cost of updates from O(corpus) to O(delta), which is critical for repositories that receive daily additions.

### Resource Scaling

Processing throughput scales linearly with compute resources up to I/O bottlenecks:

| Stage | Bottleneck | Throughput (per worker) |
|-------|-----------|----------------------|
| Normalization | CPU (format conversion) | ~500 pages/min |
| OCR | GPU (vision model) | ~100 pages/min |
| Layout analysis | GPU (detection model) | ~200 pages/min |
| Chunking | CPU (tree construction) | ~2,000 pages/min |
| Embedding | GPU (transformer inference) | ~1,000 chunks/min |
| Indexing | I/O (index writes) | ~5,000 chunks/min |

A 100GB repository (~1M pages) processes in approximately 8–12 hours on a cluster with 4 GPU workers and 8 CPU workers. The same repository processes in under 2 hours with 16 GPU workers — a near-linear speedup.

## Latency Breakdown

End-to-end query latency for a fully indexed corpus:

| Phase | Time (p50) | Time (p99) |
|-------|-----------|-----------|
| Query encoding | 15ms | 25ms |
| Vector search | 5ms | 12ms |
| Full-text search | 3ms | 8ms |
| Metadata filtering | 2ms | 5ms |
| Result fusion | 1ms | 2ms |
| Chunk retrieval | 8ms | 20ms |
| **Total retrieval** | **34ms** | **72ms** |
| Response generation | 800ms | 1,400ms |
| Citation verification | 400ms | 900ms |
| **Total end-to-end** | **1.2s** | **2.4s** |

The dominant cost is response generation and citation verification, which involve language model inference. Retrieval itself is consistently under 100ms regardless of corpus size (for corpora up to 10M chunks, the largest we have tested).

## Lessons Learned

After two years of building and operating this system, several lessons stand out:

1. **Document structure is signal, not noise.** Every piece of formatting — section numbering, table borders, figure captions, header hierarchy — carries semantic information. Discarding it in pursuit of "clean text" destroys retrievable context.

2. **Chunking quality trumps embedding quality.** We have seen larger improvements from better chunking strategies than from better embedding models. A perfect embedding of a badly chunked document still retrieves fragments.

3. **Deduplication is a feature, not an optimization.** Users do not want 5 identical results from 5 copies of the same document. Content-based dedup at ingestion time dramatically improves result quality.

4. **The last 20% takes 80% of the effort.** Corrupted files, unusual layouts, and edge cases consume disproportionate engineering time. Building robust fallback paths is more valuable than perfecting the happy path.

5. **Latency budgets should be spent on verification.** Users will tolerate 2 seconds if every result is trustworthy. They will not tolerate 500ms if they cannot trust the results.

## Conclusion

Processing heterogeneous documents at scale is fundamentally an engineering problem, not a research problem. The individual components — OCR, layout analysis, chunking, embedding, indexing — are well-understood. The challenge is composing them into a system that handles the full diversity of real-world enterprise documents reliably, efficiently, and incrementally.

Our architecture processes 100GB+ document repositories into a searchable, citation-ready index in hours rather than weeks. More importantly, it does so while preserving the structural information that makes retrieval accurate and citations meaningful.

---

*For details on how we use the processed documents for multimodal retrieval, see [Multimodal Document Understanding at Scale](/research/multimodal-document-understanding). For our citation verification approach, see [Citation-Grounded Retrieval for Enterprise Search](/research/citation-grounded-retrieval).*
