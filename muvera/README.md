# Multi-Vector Embeddings Are Great. Until They're Not.

*A deep dive into multi-vector embeddings, why they're brilliant, and why storing them naively will bankrupt your infrastructure.*

Let's start with a confession.

When we first built document search at [FluidZero](https://www.fluidzero.ai/), we did what most people do: take a document, embed it into a single vector, throw it into a vector database, and call it a day. It works. Kind of.

But "kind of" isn't good enough when you're searching through dense, visual-heavy construction PDFs, financial reports, or research papers where a single page can contain a table, a diagram, a legend, and three paragraphs of text — all of which matter for different queries.

So we went deeper. What we found is a story about a brilliant idea that runs headfirst into a very uncomfortable wall.

## The Problem With Treating a Page Like a Single Thought

Imagine you hand someone a complex architectural floor plan and ask them to describe it in one sentence. They can do it, but they'll lose so much information. 

That's exactly what single-vector document embedding does. You compress an entire page — all its visual patches, text blocks, and spatial relationships — into a single point in high-dimensional space. Then you compare that point to your query.

It's a lossy compression, and the loss hurts exactly when you need precision the most.

## Enter Multi-Vector Embeddings and ColPali

Models like **ColPali** take a fundamentally different approach. Instead of one embedding per page, they produce *one embedding per visual patch*, typically around **1,030 vectors of dimension 128** per page.

Why does this matter? Because now, instead of asking "is this page close to my query?", you can ask "does *any part* of this page nail my query?"

The similarity function that makes this work is called **MaxSim** (or Chamfer Similarity):

```
MaxSim(Q, P) = Σ_q∈Q [ max_p∈P  dot(q, p) ]
```

Read it in plain English: for each token in your query, find the best-matching patch in the document, take their dot product, then sum across all query tokens.

This is beautiful. A query like *"what's the tensile strength spec in section 4?"* can match the exact region of a page that contains that table, not just a vague page-level similarity.

## Late Interaction: Why This Architecture Is Actually Elegant

ColPali belongs to a family of models called **late interaction models**, popularized by ColBERT in the NLP world.

The idea is simple but powerful: don't collapse the full interaction between your query and document into a single dot product early on. Instead, let the query tokens "interact" with individual document tokens *late*, after you've already encoded them independently.

This gives you efficiency at index time (you pre-compute document embeddings offline) and expressiveness at query time (fine-grained token-level matching). Both at once.

In practice, late interaction models consistently outperform single-vector retrievers on complex, multi-modal documents. ColPali was designed specifically for visual document understanding: it encodes pages as image patches, meaning it naturally handles charts, tables, diagrams, and mixed layouts that would completely confuse a text-only embedder.

## Now Let's Talk About Why This Will Explode Your Storage

Here's where things get uncomfortable. Let's do the math. Slowly. Painfully.

**One page:** ~1,030 vectors × 128 dimensions × 4 bytes (float32)

```
1,030 × 128 × 4 = ~527 KB per page
```

Okay, half a megabyte per page. Fine.

**One document, 100 pages:**

```
100 × 527 KB ≈ 52 MB per document
```

Still manageable for a prototype.

**10,000 documents:**

```
10,000 × 52 MB = 520 GB
```

Now we have a problem.

**100,000 documents:**

```
100,000 × 52 MB = 5.2 TB
```

And you need all of this in memory (or at least fast SSD) to do brute-force MaxSim at query time. That's not an infrastructure cost, that's an infrastructure *crisis*.

Even if you somehow store it, the brute-force computation per query is:

```
O(|Q| × max|P_i| × n)
```

At 100,000 documents × 100 pages = 10 million pages, with ~32 query tokens and ~1,030 patches per page:

```
32 × 1,030 × 10,000,000 ≈ 330 billion dot products per query
```

On a modern GPU doing ~100 TFLOPS, that's still **~3 seconds per query**, before any I/O overhead.

This is not a search engine. This is a very expensive space heater.

## What We Actually Tried: ANN + MaxSim Reranking

The naive solution — and the one we started with — is to treat each patch embedding as an independent document, throw everything into an ANN index like HNSW, and then rerank with MaxSim.

Here's the pipeline:

```
Query → ANN search across ALL patch embeddings
      → Retrieve top-K patches
      → Group by source document
      → Rerank documents using MaxSim
      → Return top results
```

It sounds reasonable. But it has a fundamental flaw.

When you run ANN over individual patch embeddings, you're asking: *"which patches look like my query?"* MaxSim asks a different question: *"which document has the best aggregate match across all query tokens?"*

These are not the same question. A document with one very relevant patch and 1,029 mediocre ones might win the ANN lottery for that one patch. But a document where every query token finds a good match — even if no single patch is exceptional — would score higher under MaxSim and *never get retrieved*.

You're optimizing the wrong objective at retrieval time.

## What The Numbers Show

We ran this on 50 documents from the **MMLongBench** dataset, a benchmark designed specifically for multi-modal, long-document understanding. Here's what we saw:

| Config | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|
| colpali (brute force MaxSim) | **0.7829** | **0.7374** | **0.7151** |
| muvera-colpali (ANN only, no rerank) | 0.4257 | 0.3279 | 0.3255 |
| muvera-colpali-rerank | 0.7161 | 0.7065 | 0.6681 |
| muvera-fde4096 | 0.6668 | 0.5717 | 0.5543 |
| muvera-fde4096-rerank | 0.7573 | 0.7266 | 0.6998 |
| muvera-r20-fde2048 | 0.6511 | 0.5744 | 0.5567 |
| muvera-r20-fde2048-rr | 0.7590 | 0.7269 | 0.7027 |

Look at that second row. **ANN over raw patch embeddings with no reranking gets you 0.4257 Recall@5** — you're dropping nearly half your relevant documents before reranking even gets a chance.

Reranking rescues it significantly (0.7161), but you're paying full MaxSim costs on a candidate set that was already filtered by the wrong criterion.

Brute force ColPali is still king. It just doesn't scale.

## So What's The Fix?

This is exactly the problem **MUVERA** (Multi-Vector Retrieval via Fixed Dimensional Encodings) was designed to solve.

The core insight: what if you could compress those 1,030 vectors per page into *a single vector*, but do it in a way that preserves the MaxSim approximation?

```
dot(FDE_query, FDE_document) ≈ MaxSim(Q, P)
```

If you can do that, you can use any standard ANN index — HNSW, DiskANN, IVF — where the distance function actually approximates what you care about. Let's walk through how it works, then run through a concrete example.

### Step 1: SimHash Partitioning

Generate `k` random hyperplanes. For each embedding vector, check which side of each hyperplane it falls on (positive or negative dot product). This gives you a k-bit binary code, a bucket assignment.

```
bucket(x) = ( sign(dot(x, g_1)), sign(dot(x, g_2)), ..., sign(dot(x, g_k)) )
```

The key property: vectors that point in similar directions tend to land in the same bucket. The closer two vectors are in angle, the higher the probability they collide.

### Step 2: Asymmetric Aggregation

Once vectors are bucketed, documents and queries are handled *differently*:

- **Documents:** take the **average** (centroid) of all vectors in each bucket
- **Queries:** take the **sum** of all vectors in each bucket

Why asymmetric? For each query token, you want the dot product with its *best-matching* document patch, not an inflated score from every patch in the bucket. The centroid on the document side prevents double-counting. The sum on the query side is necessary because MaxSim itself sums across query tokens — if two query tokens land in the same bucket, you need both contributions preserved.

### Step 3: Concatenate Into One Long Vector

Lay all bucket vectors end-to-end:

```
FDE = [ bucket_0 | bucket_1 | ... | bucket_{2^k - 1} ]
```

Now you have a single vector, and `dot(FDE_query, FDE_document) ≈ MaxSim(Q, P)`. Plug it into any standard ANN index.

### Step 4: Repeat R Times

Do steps 1–3 with R different random seeds and concatenate the results. Each repetition is an independent random partition of the vector space. Across enough repetitions, the probability that each query token's best-matching document patch ends up in the same bucket in *at least some* repetition gets very high. The MUVERA paper finds R=20 is consistently the most important parameter for quality.

## A Toy Example

Say we have a tiny document with 3 patch vectors and a query with 2 tokens, all in 2D:

```
Document P = { p0 = [1, 0],  p1 = [0, 1],  p2 = [0.9, 0.1] }
Query    Q = { q0 = [0.8, 0.2],  q1 = [-0.1, 1.0] }
```

**True MaxSim:**

```
q0's best match → p0: dot([0.8, 0.2], [1, 0]) = 0.8
q1's best match → p1: dot([-0.1, 1.0], [0, 1]) = 1.0

MaxSim(Q, P) = 0.8 + 1.0 = 1.8
```

**Now with one SimHash hyperplane** `g = [0.5, -0.3]`:

```
p0 → dot = 0.5  > 0 → bucket 1
p1 → dot = -0.3 < 0 → bucket 0
p2 → dot = 0.42 > 0 → bucket 1

q0 → dot = 0.34 > 0 → bucket 1
q1 → dot = -0.35 < 0 → bucket 0
```

q0 landed with its best match p0, and q1 landed with its best match p1. Good sign.

**Aggregate:**
```
Doc  bucket 0: avg([0, 1])             = [0, 1]
Doc  bucket 1: avg([1, 0], [0.9, 0.1]) = [0.95, 0.05]   ← p0 and p2 got merged

Query bucket 0: sum([-0.1, 1.0])       = [-0.1, 1.0]
Query bucket 1: sum([0.8, 0.2])        = [0.8, 0.2]
```

**Concatenate and take the dot product:**
```
FDE_doc   = [0, 1, 0.95, 0.05]
FDE_query = [-0.1, 1.0, 0.8, 0.2]

dot = (−0.1)(0) + (1.0)(1) + (0.8)(0.95) + (0.2)(0.05)
    = 0 + 1.0 + 0.76 + 0.01
    = 1.77
```

| Metric | Value |
|---|---|
| True MaxSim | 1.80 |
| FDE approximation | 1.77 |
| Error | 0.03 (1.7%) |

The small error comes from p0 and p2 sharing bucket 1. Their centroid `[0.95, 0.05]` is slightly off from the true best match `p0 = [1, 0]`.

**What a second repetition does:**

With a different random hyperplane, the partition might look like:

```
Repetition 2:
  bucket 0: { p2 }   ← isolated
  bucket 1: { p0 }   ← isolated! dot(q0, p0) = 0.8 exactly
```

Now p0 and p2 are separated, and the MaxSim contribution for q0 is recovered exactly. Concatenate both repetitions and the approximation tightens. This is why R matters more than k.

Back to the benchmark: `muvera-r20-fde2048-rr` gets to **0.7590 Recall@5 and 0.7027 nDCG@5**, within striking distance of brute-force ColPali, but with an index you can actually query at scale.

## The Honest Tradeoff

MUVERA isn't perfect. You're trading some retrieval quality for scalability. At 50 documents the difference is small. At 500,000 documents it's the difference between a product that ships and one that stays in a Jupyter notebook forever.

The reranking step matters enormously. Without it (raw MUVERA-ColPali ANN), you're at 0.4257. With it, you're at 0.7161+. The FDE encoding helps initial retrieval find better candidates, and reranking does the heavy lifting on a tractable set.

The sweet spot we're converging on: FDE-based ANN for candidate retrieval + MaxSim reranking on top-K, with enough R repetitions that the approximation stays tight and a small enough FDE dimension that the index stays manageable.

## Why We Care About This At FluidZero

At [FluidZero](https://www.fluidzero.ai/), we're building search that *shows its work*, you get an answer and you can see exactly which part of which document it came from.

For that to work, retrieval actually has to find the right region of the right page, not just the right document. A page level embedding that averages over everything loses the specific patch that answers your question. Multi-vector retrieval is what makes the cited answer feel like it came from a real source and not a hallucination.

The benchmark numbers above are from our own runs on 50 MMLongBench documents. Still a lot to figure out, but that's kind of the point.

## References

1. Faysse, M., Sibille, H., Wu, T., Omrani, B., Viaud, G., Hudelot, C., & Colombo, P. (2024). **ColPali: Efficient Document Retrieval with Vision Language Models.** *arXiv:2407.01449.* [https://arxiv.org/abs/2407.01449](https://arxiv.org/abs/2407.01449)

2. Khattab, O., & Zaharia, M. (2020). **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.** *SIGIR 2020. arXiv:2004.12832.* [https://arxiv.org/abs/2004.12832](https://arxiv.org/abs/2004.12832)

3. Singhi, A., Yang, J., Bhatt, N., Bhattacharya, P., Han, S., Rajput, S., & Liang, P. P. (2024). **MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings.** *arXiv:2405.19504.* [https://arxiv.org/abs/2405.19504](https://arxiv.org/abs/2405.19504)

4. Ma, J., Liang, W., Chen, Y., & Chang, K. W. (2024). **MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations.** *arXiv:2407.01523.* [https://arxiv.org/abs/2407.01523](https://arxiv.org/abs/2407.01523)

5. Malkov, Y. A., & Yashunin, D. A. (2018). **Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.** *IEEE TPAMI. arXiv:1603.09320.* [https://arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320)

*This is part of an ongoing series on how we're building FluidZero. If you're wrestling with document retrieval at scale, or just find this stuff interesting, reach out.*

*— Tilak*
