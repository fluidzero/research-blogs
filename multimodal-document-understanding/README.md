# Multimodal Document Understanding at Scale

Modern enterprises operate on documents that communicate through more than just text. Engineering specifications embed critical dimensions in CAD drawings. Financial reports convey trends through charts that resist tabular extraction. Safety manuals pair procedural text with annotated diagrams where the relationship between the two carries the meaning.

Traditional document AI treats these modalities in isolation — OCR for text, object detection for figures, separate pipelines for tables. The result is a fragmented understanding that misses the very connections human readers rely on. At fluidzero, we set out to build a unified multimodal architecture that reads documents the way engineers and analysts do: holistically.

This post details the architecture behind our multimodal document understanding system, the benchmarks we use to measure it, and the design decisions that led to 98.7% precision on complex specification queries.

## The Problem with Unimodal Pipelines

Consider a simple query against a mechanical engineering specification:

> "What is the maximum allowable deflection for the primary support beam under combined loading?"

The answer might appear as a value in a table, referenced in a paragraph, and annotated on a structural diagram — all within the same document. A text-only system finds the paragraph. A table extractor finds the value. Neither captures the diagram annotation that qualifies the value with a specific load case.

Unimodal pipelines create three distinct failure modes:

1. **Cross-reference blindness** — inability to resolve references between text and figures ("see Figure 3.2")
2. **Context loss** — extracting a table cell value without the qualifying conditions stated in surrounding text
3. **Semantic fragmentation** — treating a captioned diagram as two unrelated pieces (image + text) rather than a unified information unit

Our internal analysis of 12,000 enterprise document queries found that **34% required understanding across at least two modalities** to produce a correct, complete answer.

## Architecture Overview

Our system processes documents through three stages: ingestion, multimodal encoding, and unified retrieval.

### Stage 1: Document Ingestion

Documents arrive in heterogeneous formats — PDF, DOCX, scanned images, CAD exports. The ingestion layer normalizes these into a common intermediate representation we call a **Document Graph**.

```
Document → Layout Analysis → Element Extraction → Document Graph
```

Layout analysis identifies structural regions (headers, body text, tables, figures, captions, footnotes) and their spatial relationships. We use a hybrid approach:

- **Rule-based heuristics** for well-structured PDFs with embedded metadata
- **Vision model classification** for scanned documents and complex layouts

The Document Graph preserves spatial proximity and reading order as edge weights, enabling downstream models to reason about which elements relate to each other.

### Stage 2: Multimodal Encoding

Each node in the Document Graph is encoded by a modality-appropriate encoder:

| Modality | Encoder | Embedding Dim | Notes |
|----------|---------|---------------|-------|
| Text | Fine-tuned transformer | 768 | Domain-adapted on technical corpora |
| Tables | Structure-aware encoder | 768 | Preserves row/column semantics |
| Figures | Vision encoder | 768 | Contrastive pre-training on figure-caption pairs |
| Diagrams | Specialized vision model | 768 | Trained on engineering drawing datasets |

The critical innovation is the **cross-modal attention layer** that sits on top of these encoders. Rather than concatenating embeddings, it learns attention patterns between spatially proximate elements of different modalities:

```python
class CrossModalAttention(nn.Module):
    def __init__(self, dim=768, heads=12):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.spatial_bias = SpatialPositionEncoding(dim)

    def forward(self, text_emb, visual_emb, spatial_distances):
        # Spatial bias encourages attention between nearby elements
        bias = self.spatial_bias(spatial_distances)
        combined = torch.cat([text_emb, visual_emb], dim=1)
        attended, weights = self.multihead_attn(
            combined, combined, combined,
            attn_mask=bias
        )
        return attended, weights
```

This layer learns, for example, that a table caption 20px above a table is semantically bound to it, while a paragraph 200px away is contextual but separate.

### Stage 3: Unified Retrieval

At query time, the user's question is encoded into the same 768-dimensional space. Retrieval operates over the unified embeddings, returning **document regions** rather than isolated chunks:

```
Query → Encode → Search unified index → Return ranked regions with citations
```

Each returned region includes:
- The primary content element (text passage, table, figure)
- Contextually linked elements from the Document Graph
- Bounding box coordinates for visual citation

## Benchmark Methodology

We evaluate against three internal benchmarks, each designed to test a specific capability:

### Benchmark 1: Specification Query Accuracy

- **Dataset**: 2,400 queries across 180 engineering specification documents
- **Ground truth**: Expert-annotated answers with source element citations
- **Metric**: Precision@1 — does the top-ranked result contain the correct answer?

**Result: 98.7% Precision@1**

This headline number deserves context. The queries are categorized by complexity:

| Query Type | Count | Precision | Example |
|------------|-------|-----------|---------|
| Single-element | 960 | 99.8% | "What material is specified for Part A?" |
| Cross-reference | 720 | 98.9% | "What tolerance applies to the dimension in Figure 4?" |
| Multi-modal | 480 | 97.1% | "What is the rated capacity shown in both the table and diagram?" |
| Conditional | 240 | 97.5% | "What is the maximum pressure at 200°C per Table 3?" |

The system performs worst on multi-modal queries — precisely because these are the hardest. A 97.1% precision on queries that require synthesizing information across text, tables, and figures represents a significant advance over unimodal baselines, which achieve 61–73% on the same subset.

### Benchmark 2: Table Extraction Fidelity

Tables in engineering documents are notoriously difficult — merged cells, nested headers, unit annotations, and footnote references are the norm rather than the exception.

We measure cell-level extraction accuracy on 850 complex tables:

- **Cell content accuracy**: 99.2%
- **Structure preservation** (row/column mapping): 97.8%
- **Unit extraction**: 96.4%
- **Footnote linkage**: 94.1%

### Benchmark 3: Figure Understanding

For 600 technical figures (engineering drawings, charts, process diagrams):

- **Caption-figure alignment**: 99.5%
- **Annotation extraction**: 93.7%
- **Chart data extraction**: 95.2%

## Design Decisions and Trade-offs

### Why Not End-to-End Vision-Language Models?

Large vision-language models (GPT-4V, Gemini) can process entire document pages. Why build a multi-stage pipeline?

Three reasons:

1. **Precision requirements** — Enterprise users need exact values, not approximate descriptions. End-to-end models occasionally paraphrase or round numerical values. Our pipeline extracts values through dedicated, verifiable extraction paths.

2. **Citation granularity** — Our system cites specific elements (Table 3, Row 7, Column 4) rather than "page 12." This level of citation is essential for compliance and audit workflows.

3. **Latency constraints** — Processing a 200-page specification through a vision-language model per query is prohibitively slow. Our pre-indexed approach returns results in under 2 seconds.

### Spatial Position Encoding

The spatial bias in our cross-modal attention layer was the single most impactful architectural decision. Without it, the model treats all document elements equally regardless of layout position. With it, the model learns the implicit grammar of document layout:

- Captions are above or below their figures
- Table headers are at the top or left
- Footnotes are at the page bottom and reference specific elements
- Section headers scope the content that follows

We experimented with absolute coordinates, relative distances, and learned position encodings. Relative distances normalized by page dimensions performed best, likely because they generalize across document formats and page sizes.

## Limitations and Future Work

We are transparent about current limitations:

- **Handwritten annotations**: The system handles printed text and standard fonts well but struggles with handwritten markup on engineering drawings. We are collecting annotated datasets to address this.
- **3D model references**: Some specifications reference 3D CAD models. Our current system processes 2D renderings but cannot reason about 3D geometry directly.
- **Cross-document queries**: The system excels at within-document understanding but does not yet support queries that span multiple documents (e.g., "Compare the material specs in Rev A vs Rev B").

## Conclusion

Multimodal document understanding is not a feature — it is a prerequisite for trustworthy document AI in enterprise settings. By building a unified architecture that preserves spatial relationships and enables cross-modal reasoning, we achieve precision levels that make AI-assisted document review viable for high-stakes applications.

The 98.7% precision on specification queries is not an abstract benchmark. It represents thousands of engineering queries where the system returned the correct, precisely cited answer — enabling engineers to trust the system for real work.

---

*For technical details on our retrieval architecture, see our companion post on [Citation-Grounded Retrieval for Enterprise Search](/research/citation-grounded-retrieval).*
