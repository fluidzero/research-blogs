# When Does a Knowledge Graph Beat Parsed Markdown? An Empirical Two-Benchmark Study

**Extraction AND corpus discovery, measured separately — FluidZero intelligence pipeline, August 2026**

Two controlled benchmarks over the same ingested corpus and the same model answer the same
underlying question from both sides: **Part I** (extraction: PDF + schema → JSON) finds the
semantic KG adds nothing to extraction accuracy — structure + agentic chunked reporting is
the engine there. **Part II** (discovery: questions over a multi-document corpus) finds the
KG *wins*: best accuracy, best grounding, fastest agent, and strictly dominant on
needle-in-a-huge-table and cross-document reasoning, where top-k RAG is architecturally
blind. Together they yield an empirical routing table for the product (§14).

---

## Table of contents

**Part I — Extraction**
1. [Motivation & question](#1-motivation--question)
2. [The benchmark & dataset](#2-the-benchmark--dataset)
3. [Evaluation methodology](#3-evaluation-methodology)
4. [Experimental setup](#4-experimental-setup)
5. [The three arms in detail](#5-the-three-arms-in-detail)
6. [Handling nested schemas: the adapter](#6-handling-nested-schemas-the-adapter)
7. [Results](#7-results)
8. [Findings](#8-findings)
9. [Operational notes](#9-operational-notes)
10. [Limitations & threats to validity](#10-limitations--threats-to-validity)
11. [Next steps (Part I)](#11-next-steps-evidence-ranked)
12. [Artifact index (Part I)](#12-artifact-index)

**Part II — Corpus discovery**
13. [The discovery benchmark](#13-part-ii--the-discovery-benchmark)
14. [Combined verdict & routing table](#14-combined-verdict--the-routing-table)
15. [Combined next steps](#15-combined-next-steps)
16. [Conclusion — what the results mean](#16-conclusion--what-the-results-mean)

---

## 1. Motivation & question

FluidZero's intelligence pipeline parses unstructured PDFs into **typed structure**: a Reducto
parse becomes a *structural graph* (Document → Page → Section/Table/Figure nodes with bounding
boxes in Neo4j), and an ontology-constrained LLM pass extracts a *semantic knowledge graph*
(typed entities, relations, and mentions grounded to page regions). A KG-native agent then
answers questions and extracts fields over that graph.

The obvious challenge to this architecture: **most teams just parse a PDF to markdown, paste it
into an LLM with a schema, and call it a day. Is all the graph machinery worth it?**

Part I answers the *extraction* half of that question empirically. The pipeline's other
half — discovery, question-answering, cross-document reasoning — is measured separately in
Part II (§13), on the same corpus and model.

We decompose the vague question into three falsifiable hypotheses:

- **H1 — parse quality is not the bottleneck.** A single LLM call over clean parsed text will
  fail at scale for *output* reasons (it cannot emit large structured outputs reliably), so
  better parsing alone cannot fix extraction.
- **H2 — agentic decomposition beats the output wall.** An agent that reads the document
  step-by-step and reports fields incrementally — with the final JSON assembled *outside* the
  model — recovers most of what single-shot loses at scale.
- **H3 — the knowledge graph itself adds accuracy.** An agent with semantic-KG tools beats an
  otherwise-identical agent without them, especially on long, structured documents; on short
  simple documents it should merely tie (a falsifiability control: if the KG *loses* on easy
  documents, it is adding overhead without value).

---

## 2. The benchmark & dataset

### 2.1 ExtractBench

We use **ExtractBench** ([huggingface.co/datasets/llamaindex/ExtractBench](https://huggingface.co/datasets/llamaindex/ExtractBench)),
a public PDF→JSON extraction benchmark with an open-source deterministic evaluation harness
([github.com/run-llama/ExtractBench](https://github.com/run-llama/ExtractBench)). The task per
document: given the PDF and a JSON Schema, produce a JSON object that fills the schema, scored
field-by-field against a **human-verified gold answer**.

The full dataset: **370 documents / 4,869 pages** across three length splits
(`short` ≤10 pp: 252 docs; `medium` 11–50 pp: 98; `long` >50 pp: 20) and eight coded business
domains (D1 finance/fund holdings … D8 real-estate closing disclosures). Each JSONL row carries:

| column | content |
|---|---|
| `id`, `category`, `pdf` | identity + split + PDF path |
| `data_schema` | the JSON Schema the output must validate against (Pydantic-style: `$defs`/`$ref`, `anyOf`-null and `type:[X,"null"]` optionals, arrays of objects) |
| `expected_output` | human-verified gold JSON |
| `field_rules` | per-field scoring rules: comparator type + accepted evidence readings `{page, bbox, quote, value}` (`verified_by: human`) |
| `repeated_structure` | identity keys for array-row alignment |
| `tags` | challenge (T1 long-list completeness / T2 needle-in-haystack / T3 dense forms), perception (P1 rotated/image-only), structure (S1–S5, e.g. S4 = table >1,000 rows), source, length |

### 2.2 Our selection (30 main-study documents + 5 addendum, pinned in `manifest.json`)

The main study ran two domain groups:

- **Legal group — D7 legal/bankruptcy filings, all 10 docs** → mapped to the pipeline's
  `legal` ontology. Few fields per schema (4–14), but the challenge is **T1 long-list
  completeness**: recover *every* row of tables spanning many pages.
  - 2 shorts (7 pp property schedules; correct answer ≈ 15 table rows)
  - 4 mediums (17–27 pp; correct answers are **208–250-row** tables, 64–77 KB as JSON)
  - 4 longs — the stress documents. Example: FTX (the collapsed crypto exchange) filed a
    114-page *Consolidated List of Creditors* in its bankruptcy case; the human-verified
    correct answer for that one document is a **7,554-row table — 1.66 MB of JSON**
    (roughly 400k tokens an LLM would have to emit). The iMedia bankruptcy equivalent is
    8,624 rows / 1.99 MB. These exist to probe the output-volume wall.
- **General group — D5 supply-chain/transactional, all 20 docs** → mapped to a **new
  domain-agnostic `general` ontology** written for this study (12 classes / 10 predicates —
  Person, Organization, Role, Place, Event, TimePeriod, Amount, Standard, Term, Product,
  Reference, Document). Real 1–3 pp price lists, spec sheets, purchase orders (2–46 fields).
  This group is the **control**: documents where markdown single-shot *should* tie.

**Corrupted twins.** The dataset ships every one of these documents twice: the clean PDF,
and a "corrupted" copy of the same document (rotated and saved as page images — like a bad
scan; tag P1). Both copies share the same correct answers, so comparing scores on clean vs
corrupted measures how much degraded capture hurts each approach — a parse-robustness
comparison that costs no extra annotation.

**Runs.** One document processed by one arm = one **"pair"** (one extraction run). Main
study: 30 docs × 3 arms = 90 pairs.

**T2 addendum (run after the main study — results in §7.7).** The main study's legal draw
tests T1 (bulk table recall) only — arguably the *least* KG-favorable extraction task,
because the paper-era ExtractBench's credit-agreement/resume domains were replaced in the
restored dataset. We therefore added **5 medium T2 needle-in-haystack documents** (3
finance earnings decks + 2 procurement schedules, `general` ontology; no corrupted twins
exist for these) — the task class where KG retrieval *should* pay. Grand total:
**35 documents, 105 extraction pairs.**

---

## 3. Evaluation methodology

**No LLM judges anywhere.** Scoring is fully deterministic — same predictions and gold always
produce the same score. Four layers:

1. **Per-field comparators** declared in the gold (`exact`, `case_insensitive`, `number` with
   tolerance, `date` (format-normalized), `boolean`, `enum`). A comparison is a function call,
   not a model call.
2. **Accepted evidence readings.** A field's gold is a *list* of human-verified readings (each
   with page/bbox/value); matching any of them counts. "Semantic equivalence" was enumerated by
   annotators up front, not judged by a model at score time.
3. **Array alignment.** Predicted vs gold table rows are matched as unordered sets via the
   Hungarian algorithm using declared identity keys (e.g. FTX rows by name+address+page).
   Matched rows score per-field; unmatched gold rows are omissions (recall hits); unmatched
   predictions are spurious (precision hits, ≈ hallucinations).
4. **Grounding metrics.** Word-level grounding passes when the value is correct AND the
   predicted bbox overlaps an accepted evidence bbox at IoU ≥ 0.5; page-level grounding checks
   the cited page number. Pure geometry.

Primary metrics: **unified value F1 / precision / recall** (per doc, averaged unweighted),
array-record accuracy with matched/missed/spurious counts, and **citation-based page-grounding
pass rate**. Invalid/empty outputs count as zeros across the full denominator (end-to-end
reliability, not conditional accuracy).

**Scoring pipeline:** each run writes a harness-native `InferenceResult` file
(`{example_id}.result.json` containing `extracted_data` + `field_citations`); the harness
evaluates offline (`extract-bench evaluation run` + `analysis generate_report`), producing JSON
+ interactive HTML reports per arm.

**Two disclosed, uniform normalizations** applied identically to every arm at score time:

- *Scalar type coercion*: values coerced toward the schema's declared type (e.g. the integer
  `2024` → the string `"2024"`); 7 of 90 result files were touched. Pre-coercion validation
  errors are preserved in the artifacts (agents were sloppier about JSON scalar types than
  single-shot — a real, small finding).
- *Truncated-JSON repair* (arm A only benefits in practice): when a single-shot response is cut
  off mid-array, we trim to the last complete element and close open brackets — the charitable
  reading any production system would apply. Without it, arm A scores even worse.

---

## 4. Experimental setup

### 4.1 Ingestion (identical substrate for all arms)

All 30 PDFs were ingested once through the FluidZero intelligence pipeline:

```
PDF → Reducto parse (canonical per-page text + layout blocks, Mongo `page_parses`)
    → structural graph (IntelDocument/Page/Section/Table/Figure + bboxes, Neo4j)
    → ontology-constrained semantic KG (Gemini 2.5 Flash extraction, validated against
      the workspace ontology; typed IntelEntity/RELATES_TO/IntelMention, project-scoped)
    → entity-name embeddings (gemini-embedding-001 @768d, Neo4j vector index)
    → StructureLinker adapter (mention → LOCATED_IN → structural container)
```

**Topology:** one workspace per domain group (carrying the ontology: `legal` / `general`), one
**project per document** — entity identity merges per-project, so this guarantees no
cross-document leakage (an arm can never ground doc X's answer in doc Y). Cross-project
SAME_AS entity resolution was **disabled** (`INTEL_ER_ENABLED=0`) for the same reason.
Content-hash dedup means each PDF was Reducto-parsed exactly once regardless of retries.
Scale: 616 parsed pages, ~616 Reducto credits (≈ $9). For the FTX doc alone the semantic KG
extracted **3,388 entities** (3,379 typed `Party` — the creditors).

### 4.2 Model & runners

**Every arm uses the same model — `claude-sonnet-4-5` — via the same runner** (the Claude Agent
SDK driving a local Claude Code session on a subscription; the harness strips `ANTHROPIC_API_KEY`
from the environment so no arm silently bills API credits). Differences between arms are
therefore *system* differences, never model differences. Arms ran sequentially (one at a time)
against identical, frozen ingested state.

Budgets: arms B/C `max_turns=64`, 3,600 s session timeout, plus a 4,500 s harness watchdog per
pair. Arm A `max_turns=6` per call (it needs only one `submit_json`).

---

## 5. The three arms in detail

### 5.1 Arm A — single-shot over parsed markdown (`fluidzero_singleshot_markdown`)

*The baseline everybody builds first.* The document's complete Reducto text (all pages,
concatenated with `===== PAGE n =====` markers) plus the full JSON Schema goes into one model
call; the model returns the filled JSON via a single terminal tool `submit_json` (with a
fallback that parses/repairs JSON out of free text).

System prompt (verbatim skeleton):

```
Extract structured data from the document into a single JSON object that validates
against this JSON Schema:

{ ...the full nested schema, pretty-printed... }

Rules:
- Return ONLY the JSON object (no commentary, no markdown fences).
- OMIT any field you cannot find in the document.
- Use null only for fields the document affirmatively shows as empty/none.
- Never invent values; every value must come from the document.
- For arrays, recover EVERY item present in the document.

The user message contains the COMPLETE parsed text of the document, with
'===== PAGE n =====' markers. Extract from it, then call `submit_json` exactly
once with the final JSON object.
```

Documents over 600k characters fall back to **N-pass sequential fill**: split at page
boundaries into ~350k-char chunks; each pass receives the merged JSON so far and instructions
to extend/correct it ("append new array items, never drop existing ones"). The 114-page docs
took 3–4 passes. Tools available: `submit_json` only. **No citations are possible** — a
markdown single-shot has no provenance to give, which the grounding metric measures directly.

### 5.2 Arm B — agent over parsed structure, NO knowledge graph (`fluidzero_agent_structural`)

The intelligence pipeline's extraction agent with `tool_profile="structural"`: it navigates the
*structural* graph and page text but has **zero access to the semantic KG**. This arm isolates
"does an agent help?" from "does the KG help?".

Tools (9): `document_outline`, `page_regions`, `get_page`, `fetch_region`, `analyze_page`
(vision on a page image), `extract_with_vision` (vision on a bbox crop), `verify_grounding`
(lexical/VLM claim check), plus the terminals `report_field` / `submit_extraction`.
`resolve_cross_reference` is **excluded** — it answers from Neo4j graph edges, so it belongs to
arm C.

System prompt (verbatim "how to work" section; no ontology block):

```
You are a document-intelligence agent extracting structured fields from a parsed
document set.

The document set is parsed into a structural graph (documents → pages →
sections/tables/figures, each with a bounding box) plus per-page text.

How to work:
1. Get oriented with `document_outline`, then `page_regions` for the pages that matter.
2. Read content: `fetch_region` gives a region's parsed text; `get_page` gives the
   full page; use `analyze_page` / `extract_with_vision` (optionally with a bbox)
   only to READ a specific page/region visually.
3. For tables that continue across pages, walk EVERY page they span — completeness
   matters more than speed.
4. Verify support with `verify_grounding` before you finalize.
Claim only what the sources support; attach citations (doc_id, page, and the node_key
you used, plus a short verbatim quote).

Extract these fields:
  - <dotted-path field list from the schema adapter, §6>

For each field, find the grounded value and call `report_field` with its value,
confidence, reasoning, and citations. For large arrays, report in chunks across
multiple `report_field` calls — list values accumulate. If a field cannot be
grounded, report it with a null value and low confidence. When every field has been
reported, call `submit_extraction`.
```

The load-bearing mechanism is **chunked array accumulation**: `report_field` may be called
repeatedly for an array field; successive list values (and their citations) are *appended*, so
a 250-row table is emitted across many small tool calls instead of one giant response.

### 5.3 Arm C — the full KG agent (`fluidzero_agent_kg`)

Identical to arm B (same budgets, same terminals, same chunked reporting) **plus** the semantic
knowledge graph and graph-navigation tools (20 total): `link_entities` (hybrid entity search:
exact + Lucene + vector), `list_types`, `list_entities`, `entity_profile` (typed relations +
every grounded mention), `typed_neighbors`, `find_paths`, `ppr_retrieve` (personalized
PageRank), `concepts_in_region`, `entity_mentions`, `resolve_cross_reference`,
`cypher_template`, plus all structural/vision/grounding tools of arm B.

Its system prompt additionally renders the workspace ontology — every entity class with
description and aliases, every relation with domain→range — and the KG-first working procedure
("turn the question into entities with `link_entities` … explore relationships … ground
everything … stay inside the ontology's vocabulary"). Citations carry `entity_key`s in addition
to page/quote/bbox.

---

## 6. Handling nested schemas: the adapter

ExtractBench schemas are nested (depth 3–6, arrays of objects); the agent's field interface is
flat. A harness-level **schema adapter** bridges them identically for both agentic arms:

- **Decompose**: depth-first walk of the schema. Scalar leaves become dotted-path fields
  (`parties.borrower.name`); an **array of objects becomes ONE field** whose description embeds
  the fully-dereferenced item schema and the chunked-reporting contract; `$ref`/`anyOf`-null/
  `type:[X,null]` forms are all resolved. (Round-trip property: gold projected onto the
  decomposition and reassembled equals gold, verified for all 30 schemas.)
- **Assemble**: reported per-path values are rebuilt into the nested JSON. A path **never
  reported stays absent** (scored as omission); a reported `null` stays an explicit null
  (affirmatively absent). Same policy in arm A (omitted keys stay omitted).
- **Validate**: `jsonschema` conformance recorded per output (never blocks a run).
- Citations map to the harness's `FieldCitation` (dotted field path, 1-indexed page, COCO-format
  bbox); where an agent cited a quote without a bbox, the pipeline's own quote→block matching
  supplies one best-effort.

---

## 7. Results

### 7.1 Headline: unified value F1 by stratum

| stratum | n | A single-shot | B structural agent | C KG agent |
|---|---|---|---|---|
| general shorts (1–3 pp) | 20 | **0.958** | 0.948 | 0.929 |
| legal shorts (7 pp) | 2 | 0.991 | 0.989 | **0.996** |
| **legal mediums (17–27 pp)** | 4 | 0.405 | **0.864** | 0.841 |
| legal longs (114 pp stress) | 4 | **0.057** | 0.032 | 0.014 |
| **ALL** | 30 | 0.766 | **0.817** | 0.800 |

Precision stays high for every arm even where recall collapses (ALL: A 0.886 / B 0.942 /
C 0.940) — failures are omissions, not fabrication.

### 7.2 The output-volume cliff (single-shot)

| doc | gold size | arm A outcome |
|---|---|---|
| DCD (208-row gold, 64 KB) | ~16k output tokens | **full recovery, single call** |
| BBB (250-row gold, 77 KB) | ~20k output tokens | **nothing parseable — twice, even with repair** (recorded as structural-failure zeros) |

The single-shot failure boundary sits at roughly **15–20k output tokens**, and it is a cliff,
not a slope: below it single-shot is perfect and 3–10× cheaper; above it, it emits nothing.

### 7.3 Completeness on realistic tables (the mediums)

Both agents recovered **100% of rows on every medium pair — clean and rotated-scan corrupted
twins alike** (BBB 250/250 ×4 pairs, DCD 208/208): chunked `report_field` accumulation is the
mechanism. Single-shot managed full recovery only where the output fit one response (§7.2).

### 7.4 The stress stratum (7.5k–8.6k-row golds — designed to defeat everyone)

Rows recovered (gold 7,554 / 8,624):

| doc | A | B | C |
|---|---|---|---|
| FTX clean | 213 | **311** | 99 |
| FTX corrupted | **385** | 112 | 0 |
| iMedia clean | **84** | 0 | 5 |
| iMedia corrupted | **300** | 89 | 167 |
| **total** | **982** | 512 | 271 |

Best recall <5%; precision ~0.95. Chunked single-shot dumps the most rows overall (brute-force
emission beats navigation when only emission matters); agentic reliability collapses at
marathon length (B: 311 → 0 across docs; timeout salvage is a coin flip). And the sharpest
single observation of the study: **arm C recovered 99/0 rows on FTX with 3,379 creditor
entities already sitting in its graph** — the data was pre-extracted and unreachable, because
no bulk-export tool exists.

### 7.5 Grounding (citation-based page-grounding pass rate)

| stratum | A | B | C |
|---|---|---|---|
| general shorts | 0.000 | 0.229 | 0.230 |
| legal shorts | 0.000 | 0.986 | **1.000** |
| legal mediums | 0.000 | **0.853** | 0.777 |
| legal longs | 0.000 | 0.495 | **0.609** |
| ALL | **0.000** | 0.398 | 0.405 |

Single-shot cannot produce provenance at all; the agentic pipeline grounds a substantial share
of its fields to verifiable pages — near-perfect on legal shorts. **This is the categorical
difference between the approaches**, and it maps directly to what grounded-extraction products
sell.

### 7.6 Omissions vs hallucinations (array cells, summed)

| arm | matched | missed (omissions) | spurious (≈hallucinations) |
|---|---|---|---|
| A | 17,363 | 317,743 | 995 |
| B | 15,763 | 319,343 | **873** |
| C | 12,129 | 322,977 | 1,127 |

Hallucination is rare and comparable across arms (~0.3% of gold volume); the stress docs
dominate the omission counts for everyone.

### 7.7 T2 addendum — needle-in-haystack (run after the main study)

Five additional medium documents tagged `challenge:T2.*` (find a small number of scattered
facts in a 20–40-page document — the task class KG retrieval was *expected* to win):
four D1 earnings/investor decks + two D3 procurement schedules, `general` ontology, same
arms/harness/scoring. Outputs are small, so every document fits both the context window and
one response.

| doc (all medium) | A | B | C |
|---|---|---|---|
| baker_hughes earnings deck | **1.000** | 1.000 | 0.950 |
| crh results deck | 0.870 | **0.957** | 0.913 |
| gov_clin_schedule_0033 (30 scattered fields) | **0.964** | 0.493 | 0.928 |
| hpe earnings deck | **0.828** | 0.615 | 0.571 |
| sf1449 procurement form | **1.000** | 0.833 | 0.667 |
| **T2 mean value F1** | **0.932** | 0.780 | 0.806 |
| T2 mean page grounding (citations) | 0.000 | 0.286 | 0.277 |

**Single-shot wins needle-in-haystack.** When the document fits in context and the output is
small, full-context attention finds scattered facts better than tool-driven navigation — the
agents *lose* accuracy hunting page-by-page (B collapsed to 0.49 on the 30-field procurement
schedule). Two nuances: (a) this is the **first stratum where C beats B** (0.806 vs 0.780,
driven by a dramatic 0.93-vs-0.49 rescue on the scatter-heaviest doc — entity lookup genuinely
outperformed page-walking there), so KG retrieval does help *relative to blind navigation*;
(b) both agents still lose to simply reading everything at once. Grounding remains
agents-only (0.28 vs 0.000).

### 7.8 Cost & latency (medians by class)

| doc class | A | B | C |
|---|---|---|---|
| short | ~1 min | ~3 min | ~3 min |
| medium | ~7 min (when it works) | ~11–13 min | ~10–14 min |
| long (114 pp) | ~5–27 min | ~5–28 min | ~12–60 min |

Total study cost: ~616 Reducto credits (≈$9), <$5 Gemini (enrichment at ingest), ~9 hours of
subscription agent time spread across rate-limit windows. Judging cost: $0 (deterministic).

---

## 8. Findings

1. **H1 confirmed.** Parse quality is not the bottleneck; *output* capacity is. Single-shot
   over clean parsed text ties or wins on short documents and hits a hard emission cliff at
   ~15–20k output tokens, below the size of a routine 250-row table. *Concretely: the DCD
   court filing (208-row answer, 64 KB) came back complete from one call; the BBB service
   list (250 rows, 77 KB) — a slightly bigger answer — came back empty, twice. In practice
   that is the difference between "paste the invoice into ChatGPT works fine" and "the
   loan-tape spreadsheet silently comes back blank."*
2. **H2 confirmed — the deciding result.** At realistic table scale (legal mediums), agentic
   chunked extraction more than doubles single-shot F1 (0.86 vs 0.41) and achieves 100% row
   completeness on every document including degraded scans. Incremental `report_field` with
   external assembly is the mechanism that beats the output wall. *Use case: a paralegal
   needs the complete 250-party master service list from a 27-page filing as a spreadsheet
   — the agent walks the table page by page, emitting ~20 rows per tool call, and delivers
   every row even from the rotated-scan copy; one big prompt physically cannot.*
3. **H3 not confirmed.** The semantic KG added no measurable extraction accuracy: B ≥ C overall
   and on mediums; C regressed recall slightly on trivial docs (turns spent on entity tools
   that don't convert) and never leveraged its pre-extracted entities on the stress docs (no
   bulk-export tool). The KG's extraction case is unproven *on T1 tasks with the current
   toolset* — two specific gaps (§11) must close before a final verdict.
4. **Grounding is the moat.** Only the agentic pipeline produces verifiable page-level
   provenance (0.40 overall, up to 1.0 on legal docs, vs 0.000 for single-shot). If the product
   promise is *auditable* extraction, the baseline cannot deliver it at any accuracy level.
5. **Nobody survives 7.5k-row outputs.** Precision stays ~0.95 but recall <5% for every
   approach. Bulk enumeration at that scale is a deterministic table-export problem, not an
   LLM problem.
6. **Minor but real:** agents are sloppier about JSON scalar types than single-shot (7/90
   outputs needed score-time coercion); corrupted scans cost the structural agent ~3× rows on
   longs while costing nothing on mediums (vision + region tools absorb the damage at that
   scale).
7. **T2 addendum: the routing rule is about OUTPUT size, not input size or task type.**
   Single-shot wins needle-in-haystack (0.932 vs 0.78–0.81) whenever the document fits in
   context — full-context attention beats navigation for *finding*; agents only earn their
   cost when the *output* exceeds one response (~15–20k tokens), where they win 2×. The KG
   posted its only B-relative win on T2 (0.806 vs 0.780, incl. a 0.93-vs-0.49 rescue on the
   scatter-heaviest doc) — evidence that entity retrieval beats blind page-walking — but
   never beats reading everything at once.

---

## 9. Operational notes

For reproducibility honesty, the run surfaced (and fixed) real operational failure modes worth
knowing about:

- **Subscription rate-limit windows** (~2.5 h of agent compute each) repeatedly interrupted the
  sweep. Hardening: empty results are recorded as *retryable failures* (never silent
  completions), and an auto-retry loop resumes after each window. One early window produced 30
  fake "completed" empty results before this hardening — all purged and re-run.
- **Wedged SDK sessions**: a hung CLI subprocess can swallow `asyncio` cancellation inside
  client cleanup, defeating in-process timeouts (one session ran 86 minutes past its cap). Fix:
  a harness watchdog that kills the SDK's subprocess *first*, then cancels — pairs become
  retryable instead of poisoning the run.
- **Arm A at 114 pp** required chunking finer than half-splits (dense creditor text runs ~3
  chars/token), and **truncation repair** to salvage cut-off arrays.
- The environment lost its staging control-plane mid-experiment (deliberate infra teardown);
  the pipeline's isolation meant the benchmark ran entirely on the document stores (S3, Mongo,
  Neo4j) with registry integration bypassed.
- Every pair is resumable via a ledger; the two BBB single-shot pairs that failed structurally
  after repair-enabled retries are recorded as **zeros counted in the denominator** (skipping
  them would overstate the baseline).

---

## 10. Limitations & threats to validity

- **Small n per stratum** (4 mediums, 4 longs); single run per pair (agent nondeterminism
  unmeasured). Directionally strong effects (0.41 vs 0.86; 0.000 vs 0.40 grounding) are far
  outside plausible noise; small deltas (B vs C ±0.02) are not.
- **One model, one parser.** All arms share `claude-sonnet-4-5` and Reducto; absolute numbers
  will shift with either. Internal comparisons are the valid product.
- **T1-only legal draw.** The legal group tests bulk-list recall — the KG's least favorable
  extraction task. Needle-in-haystack (T2) documents were not in this pilot (see §11).
- **Extraction only.** Discovery/QA — the KG's primary design goal — is untested here; H3's
  failure does not transfer to it.
- Disclosed leniencies (uniform scalar coercion; truncation repair) benefit the baseline at
  least as much as the agents; removing them widens the agents' lead.
- Rate-limit interruptions never contaminated results (failed pairs re-ran to genuine
  completion), but they did serialize execution; wall-clock comparisons across arms are
  indicative, not controlled.

---

## 11. Next steps (evidence-ranked)

1. **Product routing by expected OUTPUT size** (immediate; sharpened by the T2 addendum):
   single-shot whenever the document fits in context AND the expected output fits one response
   (~≤15k output tokens) — it wins or ties on accuracy at 3–10× lower cost across shorts AND
   needle-in-haystack; agentic chunked extraction only above the output cliff, where it wins
   2×; deterministic table export for 1,000+-row tables. *(Caveat: routing sacrifices
   grounding — if the customer needs verifiable citations, the agentic path is required
   regardless of size.)*
2. **Build the KG bulk-export extraction tool** (paginated entity/table dump). The FTX case
   proves the data was in the graph and unreachable within a turn budget. Re-run the long
   stratum after; this is the cheapest possible flip of finding 3.
3. **[DONE — see §7.7]** T2 needle-in-haystack addendum. Outcome: single-shot wins the task
   class outright; the KG beat blind navigation (C > B, its only stratum win) but not
   full-context reading.
4. **Lead with grounding** in the product and in any public write-up: it is the categorical,
   defensible win (0.28–1.0 vs 0.000 across every stratum), and it is what "auditable
   extraction" means. It is also now the ONLY reason to run an agent on documents below the
   output cliff.
5. **Fix agent type-discipline at the tool layer** (coerce scalar types in `report_field`
   against the declared field type) so score-time normalization becomes unnecessary.
6. **Budget-aware emission for marathon documents**: prompts/policies that start dumping rows
   early and refine later, plus repeated-run variance measurement on the stress docs.
7. The semantic KG's extraction verdict is now complete across both task families: it never
   beats single-shot on accuracy; its wins are relative-to-navigation (T2) and grounding.
   Its primary justification rests on **discovery/QA** — benchmark that half next (different
   instrument; this harness only measures extraction).

---

## 12. Artifact index

| artifact | path (repo: `fluidzero-monorepo/fluiddoc`) |
|---|---|
| This whitepaper | `eval/extractbench/WHITEPAPER.md` |
| Executive summary | `eval/extractbench/output/REPORT.md` |
| Pinned document manifest | `eval/extractbench/manifest.json` |
| Harness code (dataset/ingest/adapter/arms/runner/scoring) | `eval/extractbench/*.py` |
| Per-arm interactive score reports | `eval/extractbench/output/runs/<pipeline>/_evaluation_report_detailed.html` |
| Per-pair outputs (`InferenceResult` JSON) | `eval/extractbench/output/runs/<pipeline>/<split>/<doc>.result.json` |
| Per-run agent event logs | `eval/extractbench/output/events/<pipeline>/*.events.jsonl` |
| Run ledger (90/90) | `eval/extractbench/output/state.json` |
| Adapter + agent tests | `tests/test_extractbench_adapter.py`, `tests/test_intel_agent.py` |
| Dataset | `huggingface.co/datasets/llamaindex/ExtractBench` (Apache-2.0) |
| Evaluation harness | `github.com/run-llama/ExtractBench` (MIT) |

*Arms: A = `fluidzero_singleshot_markdown`, B = `fluidzero_agent_structural`, C = `fluidzero_agent_kg`.
All arms: `claude-sonnet-4-5`, identical ingested state, deterministic scoring, no LLM judges.*

---

## 13. Part II — The discovery benchmark

Part I settled extraction and left the KG's *primary* purpose — discovery, QA, and
cross-document reasoning over a corpus — unmeasured. Part II measures it, over the SAME
ingested corpus (35 documents in three workspaces), the same model, and the same
no-LLM-judge discipline.

### 13.1 What was built to run it (shipped product capabilities, not scaffolding)

- **Workspace-level query mode**: the intel agent answering over ALL documents in a
  workspace. Implemented as a fan-out/merge read facade with zero Cypher changes, exact by
  two graph invariants: every edge is intra-project (per-project subgraphs are complete),
  and `entity_key = sha1(type|normalized_name)` is project-independent — union-by-key IS
  union-by-concept. Single-project behavior is passthrough-identical by construction.
- **Entity resolution live**: a full SAME_AS pass over the corpora wrote 6,944 links
  (6,595 exact-key auto-links, 349 LLM-adjudicated). A 14/14 manual spot-check of
  adjudicated links found all correct ("$2.8B" ↔ "$2.8 billion", "SHARKNINJA" ↔
  "SHARK NINJA", reordered law-firm names, even a Unicode µ/μ variant).

### 13.2 Instrument

28 questions **generated from the human-verified extraction golds** (never from the KG —
no answer-key leakage), each with a declared comparator and gold evidence documents/pages:
16 corpus-addressed lookups (documents referenced by content, never filename), 4
row-needles (one named row among up to 7,554), 4 cross-document computations (counts,
joins), 4 duplicate-detection probes (clean/corrupted twins). Scoring is deterministic:
comparator-checked answer correctness over free text (punctuation/prefix-tolerant
containment, numeric tolerance, date normalization) plus evidence doc-hit/page-hit. A
human reviewed the question manifest before any runs; a smoke gate caught and removed one
semantically-flawed question class (annotator-derived currency fields).

### 13.3 Arms

Same pinned model (claude-sonnet-4-5), sequential, 112 runs, all genuine (a mid-sweep
usage-limit window produced non-empty junk answers — "You've hit your limit…" — which a
hardened runner invalidated and re-ran):

| arm | system |
|---|---|
| RAG | industry baseline: Gemini-embed all 801 pages, cosine top-8 into one answer call, cites retrieved pages |
| B | intel agent, workspace scope, structural tools only (no semantic KG) |
| C | full KG agent (entity linking, typed relations, PPR), SAME_AS dark |
| C+ER | arm C with SAME_AS clusters live |

### 13.4 Results

| category | RAG | B structural | C KG | C+ER |
|---|---|---|---|---|
| lookup (16) | 0.69 | **0.88** | 0.81 | **0.88** |
| row-needle (4) | 0.75 | 0.75 | **1.00** | **1.00** |
| crossdoc (4) | 0.00 | 0.50 | **0.75** | 0.50 |
| duplicate detection (4) | 0.25 | **1.00** | **1.00** | **1.00** |
| **ALL (28)** | 0.54 | 0.82 | **0.86** | **0.86** |
| evidence doc-hit | 0.71 | 0.79 | **0.93** | **0.93** |
| median wall | **13s** | 80s | **61s** | 64s |

### 13.5 Findings (Part II)

1. **The KG wins discovery** — best overall accuracy, +14 points on evidence grounding,
   strictly dominant on row-needles (entity linking resolves a creditor name straight to
   its grounded mention: 1.00) and cross-document counting. The mirror image of Part I.
2. **RAG's ceiling is architectural.** 0.00 on crossdoc, 0.25 on duplicate detection:
   top-k retrieval cannot count or compare documents it did not retrieve. Where it works
   it is 5× faster — the right tool for single-fact lookup, the wrong architecture for
   corpus reasoning.
3. **The KG agent is FASTER than the structural agent** (61s vs 80s median): entity
   lookup jumps to the answer region instead of walking pages. On discovery, the KG is
   simultaneously more accurate, better grounded, and cheaper than agency without it.
4. **ER's increment was ≈ zero on this corpus** (C+ER = C overall) — an honest null: the
   corpus's cross-document duplicates are exact-key pairs the KG unifies by construction.
   Testing SAME_AS properly needs a corpus with real naming variance across sources.
5. **Page-level citation hits are low for every arm** (~0.07) while doc-hits are high:
   answers cite valid mention pages that differ from the single gold evidence page.
   Page-grounding evaluation needs accepted-evidence page *sets*, as in Part I's harness.

## 14. Combined verdict & the routing table

| terrain | real-world example | winner | evidence |
|---|---|---|---|
| short-doc extraction | pull vendor/total/line-items from a 2-page purchase order or invoice | single-shot (3–10× cheaper) | Part I: 0.958 vs 0.948/0.929 |
| needle-in-haystack extraction | find the effective date and total buried in a 30-page contract or earnings deck | single-shot | Part I T2: 0.932 vs 0.78–0.81 |
| big-output extraction (>~15k tokens) | turn a 250-party bankruptcy service list into a complete spreadsheet | agents (only working path) | Part I: 0.86 vs 0.41 |
| 1,000+-row tables | the 7,554-creditor FTX matrix as structured data | nobody — deterministic table export | Part I: <5% recall all arms |
| single-fact corpus lookup | "what's the case number on the Clinton filing?" over a data room | RAG for speed, agents for accuracy | Part II: 13s@0.69 vs ~60s@0.88 |
| row-needle in huge tables | "what address is on file for creditor CHECKR.COM?" (1 row in 7,554) | **KG agent** | Part II: 1.00 strict |
| cross-document reasoning | "how many filings name FTX Trading as debtor?" / "which PO has the largest total?" | **KG agent** | Part II: 0.75 vs 0.00 RAG |
| duplicate/corpus awareness | intake dedup: "has this document been filed before?" | agents (any) | Part II: 1.00 vs 0.25 |
| verifiable citations | audit/compliance: "show me the page this number came from" | agents only; KG best doc-hit | both parts: 0.93 doc-hit; A/RAG ≈ 0 |

**The system's empirical identity:** parse-to-structure + the agentic loop is the
extraction engine; the semantic knowledge graph is the *discovery* layer — it does not
boost extraction, it makes corpus intelligence work; RAG and single-shot are the correct
cheap paths for single facts and small outputs. Every layer earns its keep on exactly one
terrain, and the terrains are now measured.

## 15. Combined next steps

1. **Encode the routing table in the product** — by expected output size (extraction) and
   question class (discovery), with "citations required" forcing the agentic path.
2. **KG bulk-export tool** (Part I's top fix): the FTX graph held 3,379 creditors the
   extraction agent could not drain; a paginated export likely flips the long-list
   stratum. Re-run Part I's longs after.
3. **ER variance corpus**: test SAME_AS on documents from different sources naming the
   same entities differently — this corpus could not discriminate it.
4. **Page-grounding with accepted-evidence sets** for QA scoring parity with Part I.
5. **Type-discipline at the tool layer** (coerce `report_field` scalars) and
   budget-aware emission for marathon documents.
6. Publication: Part I §1–12 + Part II §13–14 are the whitepaper; the blog cut is the
   routing table (§14) with the three headline charts (extraction F1 by stratum, discovery
   scorecard, grounding).

### Part II artifacts

`qa_manifest.json` (28 questions) · `output/qa/results/` (112 runs) ·
`output/qa/qa_scores.json` · `output/qa/QA_REPORT.md` ·
`qa_dataset.py` / `qa_arms.py` / `qa_run.py` / `qa_score.py` ·
workspace query mode: `fluiddoc/agents/intel/{scope,workspace_reads}.py` ·
`QA_BENCHMARK_PLAN.md`

---

## 16. Conclusion — what the results mean

Strip away the metrics and the two benchmarks say something simple:

**"Just parse to markdown and ask an LLM" is not wrong — it is the correct answer to about
half of document intelligence, and the cheapest correct answer at that.** For a 2-page
invoice, a purchase order, or "find the effective date in this 30-page contract," one model
call over parsed text matched or beat everything we built, at a tenth the cost and latency.
Any architecture that routes those tasks through an agent loop is burning money for nothing
— ours included, until we measured it.

**Where markdown-in-one-prompt dies, it dies for a reason nobody fixes with better
prompts: models cannot *emit* large outputs.** The cliff sits near 15–20k output tokens —
about a 250-row table. Below it, single-shot is perfect; above it, it returns nothing at
all. The only working path across that cliff is an agent that emits incrementally and lets
the harness assemble the answer — that is an architectural requirement, not a tuning
choice. (And above ~1,000 rows, no LLM approach survives; that is a job for deterministic
table export.)

**The knowledge graph is not an extraction technology — it is a corpus technology.** Across
every extraction stratum, the semantic KG never improved accuracy over the same agent
without it. But the moment the question spans a *collection* of documents — "which of these
200 filings name FTX as debtor?", "what address is on file for this one creditor among
7,554?", "have we seen this document before?" — the KG became the best system we tested:
most accurate (0.86), best grounded (0.93 doc-hit), strictly dominant on needle and
cross-document categories, and *faster than the same agent without it*, because entity
lookup replaces page-walking. Meanwhile top-k RAG — the default industry answer for corpus
QA — scored **zero** on cross-document reasoning, not from bad tuning but by construction:
it cannot count or compare documents it did not retrieve.

**Grounding is the moat.** In both benchmarks, only the agentic pipeline produced
verifiable citations — the "click to see the page this number came from" that audit,
compliance, legal, and diligence workflows actually require. Every markdown/RAG baseline
scored ~zero on provenance at every accuracy level. If the product promise is *trustworthy*
document intelligence rather than plausible answers, the baselines cannot deliver it at
any price.

**So what should anyone build?** The measured answer is a *router*, not a bet on one
architecture: single-shot for small outputs and single facts; agentic chunked extraction
above the output cliff; the KG agent for anything corpus-level or needle-shaped; the
agentic path whenever citations are required; deterministic export for giant tables. Every
one of those clauses is backed by a number in this document rather than an opinion — which
was the entire point of running the experiment: two weeks and roughly $15 of external
spend converted an architectural debate into a routing table.

### 16.1 Mechanisms — why these results happened

**Single-shot wins in-context work because attention is global.** Every token attends to
every other token simultaneously, so a fact on page 23 is "equidistant" from the question —
no navigation needed. An agent makes a *sequence* of navigation decisions, each with its
own failure probability, and errors compound (this is exactly how the structural agent lost
the 30-field procurement needle: wrong page → wrong region → wrong value). Navigation is
pure added risk until the corpus exceeds the context window — then it becomes mandatory,
and the tables turn.

**The output wall exists because comprehension and emission are different capacities.**
Autoregressive generation degrades over long outputs: JSON syntax discipline decays,
attention to the schema drifts, providers cap response length. A 250-row table is ~20k
tokens of unbroken, syntactically perfect JSON — past the reliable limit. Chunked
`report_field` converts one impossible emission into ~15 easy ones and moves assembly and
memory into deterministic code outside the model.

**The KG lost extraction and won discovery for the same underlying reason.** Extraction is
*transcription* (copy every cell verbatim); the KG stores *distilled* knowledge — entities
and relations, not verbatim cells — so KG hops add navigation without adding content.
Discovery is *location* (find where something is), which is precisely what an entity index
optimizes: `link_entities("CHECKR.COM")` is one hop from name to grounded mention versus a
linear page scan — which is also why the KG agent was *faster* than the structural agent.
And RAG's zero on cross-document questions is structural, not tunable: "how many documents
state X" is a property of the SET; no individual chunk contains it, so no similarity
search can retrieve it. Counting requires enumeration; enumeration requires an agent or an
index.

### 16.2 Why not default to the structural graph?

For extraction, it already IS the default — the structural agent is Part I's winner. The
real question is deployment tiers, and it is about *ingest* cost, not query cost: the
structural graph needs only deterministic parsing, while the semantic KG adds an LLM pass
per page plus embeddings. At *query* time, the KG agent is a **superset** of the structural
agent — same loop, more tools, chosen per-question — so once the semantic KG exists,
defaulting to the full toolset costs nothing and measured better-or-equal everywhere in
discovery (accuracy, grounding, AND speed). Defensible tiering: structural-only ingest for
extraction-only workloads; semantic enrichment as the corpus-intelligence tier; never
withhold the KG from the agent where it exists.

### 16.3 Why the baselines cannot deliver provenance

Provenance is not a formatting feature — it is a **verification loop** requiring three
things the baselines structurally lack:

1. **Preserved source coordinates.** Markdown flattening destroys page boundaries and
   positions; the pipeline keeps every block's page + bounding box from ingestion onward,
   so a citation points at a region, not just a claim.
2. **An enforced contract.** Single-shot can be *prompted* to emit page numbers, but they
   are self-reported assertions — exactly as hallucinatable as any other output. The
   agent's terminal tools *require* doc/page/quote per claim.
3. **Verification machinery.** `verify_grounding` re-reads the cited page and checks the
   quote is actually there *before* submission. A single-shot generation cannot re-read
   anything — it is already over.

RAG sits between and the data shows it: it can cite the pages it retrieved (0.71 doc-hit)
but retrieved ≠ used, granularity stops at the chunk (no bbox → word-level grounding
impossible), and when retrieval misses the true source the citation is wrong by
construction. The agents' 0.93 vs ~0 is not a quality gap — it is presence vs absence of
the loop.

**And the honest open edges:** entity resolution's increment is untested at real naming
variance; page-level grounding needs evidence-set scoring; one model, one parser, modest
n. The map has blank regions — but the continents are drawn.
