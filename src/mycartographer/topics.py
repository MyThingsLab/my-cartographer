from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

from mythings.corpus import Chunk, Document
from mythings.embed import Embedder

from mycartographer.cluster import (
    agglomerate,
    central_index,
    similarity_matrix,
    tfidf_vectors,
)


@dataclass(frozen=True)
class ChunkRef:
    doc_id: str
    ordinal: int
    start: int
    end: int

    def marker(self) -> str:
        return f"[{self.doc_id}:{self.ordinal}]"


@dataclass(frozen=True)
class Theme:
    id: int
    top_terms: tuple[str, ...]
    doc_ids: tuple[str, ...]
    citation: ChunkRef
    label: str = ""
    blurb: str = ""
    prereqs: tuple[int, ...] = ()


@dataclass(frozen=True)
class TopicMap:
    themes: tuple[Theme, ...]
    unassigned: tuple[str, ...]
    doc_titles: dict[str, str] = field(default_factory=dict)
    corpus_paths: tuple[str, ...] = ()


def default_k(doc_count: int) -> int:
    # Deterministic, and deliberately not the model's call — clamp sqrt(n) into
    # a browsable band. Three is the floor below which "themes" is meaningless;
    # twelve the ceiling past which a map stops being a map.
    return max(3, min(12, round(doc_count**0.5)))


def build_map(
    docs: list[Document],
    chunks: list[Chunk],
    *,
    themes_k: int,
    top_terms: int = 8,
    embedder: Embedder | None = None,
) -> tuple[TopicMap, dict[int, str]]:
    # The deterministic half of the tool: cluster the documents, then represent
    # each cluster for the one Engine call. Returns the unlabelled map plus the
    # central excerpt text per theme id (what the Engine reads to name the
    # themes). Clustering is at the *document* level: measured on the real study
    # corpus, chunk-level lexical clustering collapsed because reading-list
    # boilerplate (arxiv/http/org) chains every doc's citation chunks together,
    # while document vectors — where that boilerplate saturates and is dropped by
    # the max-document-frequency cap — separate the themes cleanly. (Multi-theme
    # membership, the chunk-level payoff, waits on the embedder path; see README.)
    doc_titles = {d.id: d.title for d in docs}
    doc_texts = [d.text for d in docs]

    # Term weights always come from lexical TF-IDF — theme *labels* are drawn
    # from real terms even when an embedder supplies the clustering vectors.
    lexical = tfidf_vectors(doc_texts)
    vectors = embedder.embed(doc_texts) if embedder is not None else lexical
    sim = similarity_matrix(vectors)
    clusters = agglomerate(sim, themes_k)

    first_chunk: dict[str, Chunk] = {}
    for c in chunks:
        first_chunk.setdefault(c.doc_id, c)

    themes: list[Theme] = []
    excerpts: dict[int, str] = {}
    for theme_id, members in enumerate(clusters):
        top = _top_terms([lexical[i] for i in members], top_terms)
        centre = docs[central_index(sim, members)]
        c = first_chunk.get(centre.id)
        citation = (
            ChunkRef(c.doc_id, c.ordinal, c.start, c.end)
            if c is not None
            else ChunkRef(centre.id, 0, 0, 0)
        )
        themes.append(
            Theme(
                id=theme_id,
                top_terms=tuple(top),
                doc_ids=tuple(sorted(docs[i].id for i in members)),
                citation=citation,
            )
        )
        excerpts[theme_id] = c.text if c is not None else centre.text[:400]

    topic_map = TopicMap(
        themes=tuple(themes),
        unassigned=(),
        doc_titles=doc_titles,
        corpus_paths=tuple(sorted(d.path for d in docs)),
    )
    return topic_map, excerpts


def _top_terms(vectors: list[dict[str, float]], n: int) -> list[str]:
    weights: Counter[str] = Counter()
    for vec in vectors:
        for token, weight in vec.items():
            weights[token] += weight
    # Sort by weight desc, then token asc for a deterministic tie-break.
    ranked = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
    return [token for token, _ in ranked[:n]]


def apply_labels(
    topic_map: TopicMap,
    labels: dict[int, tuple[str, str, list[int]]],
) -> TopicMap:
    # Fold the Engine's (label, blurb, prereqs) per theme id back in, then
    # repair the prerequisite graph to a DAG. A theme the Engine did not name
    # keeps its top term as an honest fallback label.
    themes: list[Theme] = []
    for theme in topic_map.themes:
        label, blurb, prereqs = labels.get(theme.id, ("", "", []))
        fallback = theme.top_terms[0] if theme.top_terms else f"theme-{theme.id}"
        themes.append(
            replace(
                theme,
                label=label or fallback,
                blurb=blurb,
                prereqs=tuple(prereqs),
            )
        )
    themes = repair_dag(themes)
    return replace(topic_map, themes=tuple(themes))


def repair_dag(themes: list[Theme]) -> list[Theme]:
    # Drop any prerequisite edge that would introduce a cycle, adding edges in a
    # deterministic order and skipping one whose target can already reach its
    # source. Same topological-repair the mythings.selection seam owns for
    # ordered output, applied to the theme graph.
    valid_ids = {t.id for t in themes}
    # adjacency of accepted edges: prereq -> {themes that depend on it}
    edges: dict[int, set[int]] = {t.id: set() for t in themes}

    def reaches(src: int, dst: int) -> bool:
        seen = {src}
        stack = [src]
        while stack:
            node = stack.pop()
            if node == dst:
                return True
            for nxt in edges.get(node, ()):  # node -> nxt
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    repaired: dict[int, list[int]] = {t.id: [] for t in themes}
    for theme in sorted(themes, key=lambda t: t.id):
        for prereq in sorted(dict.fromkeys(theme.prereqs)):
            # Edge prereq -> theme ("prereq comes before theme"). Skip a
            # self-loop, an unknown id, or an edge whose addition would let
            # `theme` reach `prereq` (a cycle).
            if prereq == theme.id or prereq not in valid_ids:
                continue
            if reaches(theme.id, prereq):
                continue
            edges[prereq].add(theme.id)
            repaired[theme.id].append(prereq)

    return [replace(t, prereqs=tuple(repaired[t.id])) for t in themes]
