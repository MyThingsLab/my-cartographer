# my-cartographer

[![CI](https://github.com/MyThingsLab/my-cartographer/actions/workflows/ci.yml/badge.svg)](https://github.com/MyThingsLab/my-cartographer/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/MyThingsLab/my-cartographer/branch/main/graph/badge.svg)](https://codecov.io/gh/MyThingsLab/my-cartographer) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Clusters a document corpus — a shelf of PDFs, a folder of notes — into
**named, prerequisite-ordered themes**, and writes a checked-in **topic map**:
`topics/topics.json` (source of truth) + `topics/TOPICS.md` (browsable, with a
Mermaid prerequisite graph).

It answers *"what is actually **in** my collection, and in what order should I
approach it?"* — the latent themes across the material, read straight off it.
Bottom-up (what your shelf contains), the dual of
[MyUni](../my-things-core/docs/tools/my-uni.md)'s top-down field decomposition.
Design plan: [`my-cartographer.md`](../my-things-core/docs/tools/my-cartographer.md).

## Usage

```bash
# Map a corpus into topics/ locally (no git, no PR) — zero tokens with NoopEngine:
mycartographer map --corpus ~/papers --corpus ~/notes --out . --engine noop

# Name and order the themes with a model, and open a PR with the artifacts:
mycartographer map --corpus ~/papers --repo owner/library --engine claude-cli
```

Key flags: `--corpus DIR` (repeatable), `--themes K` (default `clamp(√docs, 3, 12)`),
`--out DIR` (write locally instead of opening a PR), `--no-pr`, `--engine
noop|claude-cli`, `--embed none|hashing|api`.

## How it works

Deterministic pre-work, then exactly one Engine call — the standard fleet shape:

1. **Ingest** every `.md/.txt/.pdf/.epub` under `--corpus` via
   `mythings.corpus` (PDFs through `pdftotext`).
2. **Vectorise + cluster** deterministically. Documents are turned into
   IDF-weighted lexical vectors (stopwords and corpus-saturating terms dropped),
   then grouped by **average-linkage agglomerative clustering** cut at `K`. No
   random seed — the map is byte-stable across runs, so it diffs cleanly.
3. **One Engine call** labels every cluster and proposes the prerequisite edges
   (`label_themes`); invented cluster ids and out-of-set prereqs are dropped and
   the graph is repaired to a DAG. Under `NoopEngine` each theme falls back to
   its top term and the map is still produced.
4. **Render + write** `topics/topics.json` + `topics/TOPICS.md`, either locally
   (`--out`) or as a single PR on a dedicated branch (idempotent: an unchanged
   corpus is a no-op).

### Clustering granularity — document-level (v0)

The design plan proposed clustering *chunks* and rolling up to multi-theme
document membership. Measured on a real 19-document study corpus, chunk-level
lexical clustering **collapsed** — reading-list boilerplate (`arxiv`/`http`/
`org`) chains every document's citation chunks into one blob. Document-level
vectors, where that boilerplate saturates and is dropped by the
max-document-frequency cap, separate the themes cleanly (physics / symmetry /
neural-nets / Bayesian / clustering all fell out correctly). So **v0 clusters at
the document level**, one theme per document. Multi-theme membership returns with
the semantic embedder path — see below.

### Semantic clustering (`--embed`)

By default MyCartographer clusters over its own lexical vectors — good for coarse
thematic separation, but blind to paraphrase, math notation, and cross-language
material (the failure modes in
[core ADR 0003](../my-things-core/docs/adr/0003-embedding-seam.md)). Pass
`--embed api` to cluster over real semantic vectors from a configured
`MYTHINGS_EMBED_URL` (an OpenAI-compatible `/v1/embeddings` endpoint — a
localhost model keeps the corpus private). `--embed hashing` is the offline
deterministic embedder, mainly for exercising that path.

## Requires

`my-things-core` with the `mythings.embed` seam
([core#113](https://github.com/MyThingsLab/my-things-core/pull/113)) and
`mythings.corpus`.
