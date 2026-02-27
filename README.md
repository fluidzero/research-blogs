# research-blogs

Research blog content for [fluidzero.ai/research](https://fluidzero.ai/research).

Technical papers, engineering insights, and methodology deep-dives from the fluidzero team.

## Structure

Each blog post lives in its own directory with a `README.md` file:

```
post-slug/
  README.md       # Blog post content (Markdown)
  assets/         # Images, diagrams (optional)
```

Posts are registered in the [fennec-ui](https://github.com/fluidzero/fennec-ui) repository at `lib/blog/posts.ts` with metadata (title, author, date, category). The marketing site fetches README content at runtime via ISR.

## Posts

| Post | Category |
|------|----------|
| [Multimodal Document Understanding at Scale](./multimodal-document-understanding/) | Technical Paper |
| [Citation-Grounded Retrieval for Enterprise Search](./citation-grounded-retrieval/) | Methodology |
| [Processing Heterogeneous Documents at 100GB+ Scale](./processing-heterogeneous-documents/) | Engineering |
