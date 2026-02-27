# Citation-Grounded Retrieval for Enterprise Search

The promise of AI-powered document search is simple: ask a question in natural language, get an accurate answer. The reality in enterprise settings is more demanding. An answer without a source is an opinion. In regulated industries — aerospace, pharmaceuticals, energy, finance — an unsourced claim from an AI system is worse than no answer at all.

This post describes our approach to citation-grounded retrieval: an architecture where every generated response is anchored to specific, verifiable source locations within the original documents. We explain why retrieval-augmented generation (RAG) alone is insufficient, how our hierarchical verification pipeline eliminates hallucinations, and the engineering behind sub-second citation resolution.

## Why RAG Is Not Enough

Retrieval-Augmented Generation has become the default architecture for knowledge-grounded AI systems. The pattern is well-established:

1. Embed the user query
2. Retrieve relevant document chunks
3. Pass the chunks as context to a language model
4. Generate a natural language answer

This works well enough for general knowledge queries. It fails for enterprise document search in three specific ways.

### Failure Mode 1: Attribution Ambiguity

Standard RAG retrieves chunks and generates answers, but the mapping between specific claims in the answer and specific chunks in the context is implicit. When the model writes "The maximum operating temperature is 450°C," which of the 5 retrieved chunks contained that value? The user cannot verify without reading all of them.

### Failure Mode 2: Cross-Chunk Confabulation

When multiple chunks contain related but distinct information, language models occasionally synthesize details from different sources into a single claim that exists in neither. For example:

- Chunk A: "Part X is rated for 300°C continuous operation"
- Chunk B: "Part Y is rated for 450°C with thermal shielding"
- Generated answer: "Part X is rated for 450°C" ← **hallucination**

The model has correctly retrieved the right topic but incorrectly attributed a value from one chunk to an entity in another.

### Failure Mode 3: Paraphrase Drift

Language models paraphrase by nature. In enterprise contexts, paraphrasing is dangerous. "Shall" and "should" have different legal weight in specifications. "Maximum" and "typical" are not interchangeable for rated values. Standard RAG generation has no mechanism to preserve the exact language of the source.

## Our Approach: Hierarchical Verification

Our citation-grounded retrieval system adds three layers on top of standard retrieval:

```
Query → Retrieve → Generate Draft → Verify Claims → Anchor Citations → Format Response
```

Each layer serves a specific purpose in ensuring accuracy.

### Layer 1: Claim Decomposition

After initial generation, the draft answer is decomposed into atomic claims. An atomic claim is the smallest independently verifiable statement in the response.

Given a draft answer:

> "The primary coolant pump operates at 2,400 RPM under normal conditions and can sustain up to 3,200 RPM during emergency operation, as specified in Section 4.3 of the Operations Manual."

The decomposition produces:

| Claim | Value | Source Reference |
|-------|-------|-----------------|
| Primary coolant pump normal RPM | 2,400 RPM | Needs verification |
| Primary coolant pump emergency RPM | 3,200 RPM | Needs verification |
| Source section | Section 4.3, Operations Manual | Needs verification |

### Layer 2: Source Anchoring

Each atomic claim is independently verified against the retrieved source material. This is not a simple string match — it is a semantic verification that accounts for:

- **Value equivalence**: "2,400 RPM" matches "2400 rpm" and "2.4 × 10³ rev/min"
- **Contextual scoping**: The value must apply to the correct entity (the primary pump, not the secondary pump)
- **Conditional qualification**: If the source says "up to 3,200 RPM *with operator override enabled*," the condition must be included

The anchoring process produces a structured citation for each claim:

```json
{
  "claim": "Primary coolant pump emergency RPM is 3,200",
  "verified": true,
  "source": {
    "document": "Operations Manual Rev C",
    "section": "4.3.2",
    "page": 47,
    "paragraph": 3,
    "bbox": [72, 340, 540, 365],
    "exact_text": "The primary coolant pump shall not exceed 3,200 RPM during emergency operation with operator override enabled."
  },
  "qualification": "with operator override enabled"
}
```

Note the `qualification` field. The verification process detected that the original source includes a condition not present in the draft answer. This qualification is surfaced to the user.

### Layer 3: Response Assembly

The verified claims are reassembled into a coherent response with inline citations. Unverified claims are either:

- **Dropped** if they cannot be anchored to any source
- **Flagged** with a confidence indicator if partial evidence exists
- **Rephrased** to match the source language exactly for critical values

The final response looks like this:

> The primary coolant pump operates at 2,400 RPM under normal conditions [§4.3.1, p.46] and can sustain up to 3,200 RPM during emergency operation **with operator override enabled** [§4.3.2, p.47].

Every value is cited. The missing qualification has been restored. The user can click any citation to see the exact source text with its bounding box highlighted in the original document.

## The Zero-Hallucination Claim

We report zero uncited responses in our production system. This claim requires careful definition.

**What we mean**: Every factual claim in every generated response is anchored to a specific, verifiable source location. If the system cannot verify a claim, it is not included in the response.

**What we do not mean**: The system never makes mistakes. It can retrieve the wrong document, misparse a table, or fail to find an answer that exists. These are retrieval failures, not hallucinations. The distinction matters:

| Failure Type | Definition | Mitigation |
|---|---|---|
| Retrieval miss | Correct answer exists but wasn't retrieved | Improve indexing, expand search |
| Extraction error | Source found but value incorrectly parsed | Improve extraction pipeline |
| Hallucination | Answer contains claims not in any source | **Eliminated by verification** |

Our verification pipeline targets the third category. When the system lacks confidence, it says so — explicitly, with a message like "I found relevant sections but could not extract a definitive answer. Here are the most relevant passages for manual review."

This is a deliberate design choice. In enterprise contexts, a system that says "I don't know" is more trustworthy than one that always produces an answer.

## Engineering: Sub-Second Citation Resolution

The three-layer verification pipeline adds computational cost. Running claim decomposition and source anchoring naively would add 3–5 seconds to every query — unacceptable for interactive use.

We achieve sub-second total latency through three optimizations:

### Pre-computed Claim Anchors

During document ingestion, we pre-compute **anchor candidates** for every extractable fact in the corpus:

```python
@dataclass
class AnchorCandidate:
    document_id: str
    section: str
    page: int
    bbox: tuple[float, float, float, float]
    text: str
    entities: list[str]       # extracted named entities
    values: list[NumericValue] # parsed numeric values with units
    embedding: np.ndarray      # semantic embedding
```

At query time, claim verification becomes a lookup against pre-computed anchors rather than a full re-analysis of the source documents.

### Parallel Claim Verification

Atomic claims are independent — they can be verified in parallel. We use a pool of verification workers that process claims concurrently:

```
Draft Answer (3 claims) → [Verify Claim 1] → ✓
                         → [Verify Claim 2] → ✓  → Assemble Response
                         → [Verify Claim 3] → ✓
```

The wall-clock time for verification is determined by the slowest claim, not the sum of all claims.

### Speculative Generation

We begin generating the draft answer before retrieval is fully complete. As retrieval results stream in, the generation model incorporates them incrementally. This overlaps the retrieval and generation phases:

```
Timeline:
  Retrieval:   [====>      ]
  Generation:      [====>  ]
  Verification:        [==>]
  Total:       [==========>]  ~1.8s average
```

Without overlap, the same pipeline would take ~3.5 seconds.

## Evaluation

We evaluate citation quality on a held-out set of 1,200 queries across 4 industries:

### Citation Accuracy

| Metric | Score |
|--------|-------|
| Claims with correct citation | 99.4% |
| Claims with correct value | 99.1% |
| Citations with correct bounding box | 98.7% |
| Responses with all claims verified | 97.8% |

The 2.2% of responses with at least one unverified claim correctly flag the uncertainty to the user. In no case does the system present an unverified claim as verified.

### Comparison with Standard RAG

On the same query set, we compared our system against a standard RAG implementation using the same retrieval index and language model:

| Metric | Standard RAG | Citation-Grounded |
|--------|-------------|-------------------|
| Answer accuracy | 94.2% | 99.1% |
| Includes source reference | 67% | 100% |
| Source reference correct | 81% | 99.4% |
| Contains hallucinated claim | 8.3% | 0% |
| Average latency | 1.1s | 1.8s |

The 0.7 second latency increase is the cost of verification. In our user studies, enterprise users unanimously preferred the slower, verified responses — several noted they would accept up to 5 seconds of additional latency for guaranteed citations.

## Limitations

- **Creative synthesis**: The system is designed for factual retrieval, not creative analysis. Queries like "What are the implications of these specifications for future design?" require reasoning beyond source anchoring.
- **Cross-document reasoning**: Citation anchoring currently operates within single documents. Cross-document queries produce citations from each document independently rather than synthesized cross-references.
- **Source quality**: The system faithfully cites its sources. If the source document contains errors, the system will cite those errors accurately. We do not validate the correctness of source material — that remains the user's responsibility.

## Conclusion

Citation-grounded retrieval is not an incremental improvement over RAG — it is a fundamentally different contract with the user. Standard RAG says "here is an answer that is probably based on your documents." Citation-grounded retrieval says "here is an answer, here is exactly where each fact comes from, and here is what I could not verify."

For enterprise applications where accuracy is non-negotiable and auditability is a requirement, this distinction is the difference between a useful tool and a liability.

---

*This system builds on our multimodal document understanding pipeline. For details on how we process complex documents with tables, figures, and diagrams, see [Multimodal Document Understanding at Scale](/research/multimodal-document-understanding).*
