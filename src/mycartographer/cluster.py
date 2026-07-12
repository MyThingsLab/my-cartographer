from __future__ import annotations

import math
from collections.abc import Sequence

# Deterministic clustering primitives. MyCartographer groups chunks against each
# other (corpus.shortlist ranks them against a query); the vectors here are its
# own IDF-weighted lexical vectors by default — the representation the stress
# test in docs/adr/0003 validated for coarse thematic separation — with an
# optional mythings.embed Embedder swapped in for semantic vectors. Everything
# is a pure function of its input: a topic map that reshuffled on re-run would
# be undiffable and untrustworthy, so determinism is the load-bearing property.

# A sparse vector: token -> weight. Kept sparse (not fixed-dim) so IDF weighting
# needs no vocabulary padding; cosine_sparse handles the alignment.
SparseVector = dict[str, float]

# Function words carry no thematic signal but appear in every chunk, so under a
# cosine they drown the discriminative terms — the first run over the real study
# corpus collapsed into a single blob for exactly this reason. Kept deliberately
# small; the max_document_frequency cap below removes the rest (including
# corpus-specific noise like "arxiv" that saturates a citation-heavy corpus)
# without a hand-tuned list.
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in into is it its of on or that the their
    this to was were which with will would can could may might not no nor so than then there
    these those they them we you your our but if while over under about across per via using
    use used one two three first second also more most such each any all both between
    within""".split()
)


def tfidf_vectors(
    texts: Sequence[str], *, max_document_frequency: float = 0.5
) -> list[SparseVector]:
    tfs = [_term_freq(t) for t in texts]
    n = len(tfs)
    doc_freq: dict[str, int] = {}
    for tf in tfs:
        for token in tf:
            doc_freq[token] = doc_freq.get(token, 0) + 1
    # A token in more than max_document_frequency of the chunks is
    # non-discriminative — it can't separate themes because it's everywhere — so
    # it's dropped from the vocabulary entirely alongside stopwords, single
    # characters, and bare numbers.
    cap = max_document_frequency * n
    vocab = {
        token
        for token, df in doc_freq.items()
        if df <= cap and len(token) > 1 and not token.isdigit() and token not in _STOPWORDS
    }
    # Smoothed IDF over the surviving vocabulary: rare tokens discriminate most.
    idf = {token: math.log(1 + n / (1 + doc_freq[token])) for token in vocab}
    vectors: list[SparseVector] = []
    for tf in tfs:
        vec = {t: (1 + math.log(c)) * idf[t] for t, c in tf.items() if t in vocab}
        vectors.append(_normalize_sparse(vec))
    return vectors


def _term_freq(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokenize_ordered(text):
        counts[token] = counts.get(token, 0) + 1
    return counts


def tokenize_ordered(text: str) -> list[str]:
    # corpus.tokenize returns a set (deduped); term frequency needs the count,
    # so re-tokenize preserving repeats via the same lowercasing contract.
    import re

    return [m.lower() for m in re.findall(r"[a-zA-Z0-9_]+", text)]


def _normalize_sparse(vec: SparseVector) -> SparseVector:
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if not norm:
        return vec
    return {t: v / norm for t, v in vec.items()}


def cosine_sparse(a: SparseVector, b: SparseVector) -> float:
    # Iterate the smaller vector's keys — cosine is symmetric and most chunk
    # pairs share few tokens, so this is the cheap direction.
    if len(a) > len(b):
        a, b = b, a
    dot = sum(w * b.get(t, 0.0) for t, w in a.items())
    na = math.sqrt(sum(w * w for w in a.values()))
    nb = math.sqrt(sum(w * w for w in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def similarity_matrix(
    vectors: Sequence[SparseVector | tuple[float, ...]],
) -> list[list[float]]:
    n = len(vectors)
    sim = [[1.0] * n for _ in range(n)]
    dense = bool(vectors) and isinstance(vectors[0], tuple)
    cos = _cos_dense if dense else cosine_sparse
    for i in range(n):
        for j in range(i + 1, n):
            sim[i][j] = sim[j][i] = cos(vectors[i], vectors[j])  # type: ignore[arg-type]
    return sim


def _cos_dense(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def agglomerate(sim: Sequence[Sequence[float]], k: int) -> list[list[int]]:
    # Average-linkage agglomerative clustering cut at k clusters. Chosen over
    # k-means precisely because it needs no random seed: with a fixed tie-break
    # it is a pure function of the similarity matrix, so the partition is
    # byte-stable across runs. Clusters are kept sorted by their smallest member
    # index, and among equal-similarity merges the lexicographically-smallest
    # (min-index-of-A, min-index-of-B) pair wins — the deterministic tie-break.
    n = len(sim)
    if k <= 0:
        raise ValueError("k must be positive")
    clusters = [[i] for i in range(n)]
    if n <= k:
        return clusters

    while len(clusters) > k:
        best = -2.0
        best_pair = (0, 1)
        for p in range(len(clusters)):
            for q in range(p + 1, len(clusters)):
                avg = _avg_link(sim, clusters[p], clusters[q])
                # Strict >: the first (smallest-index) pair at a given score is
                # kept, so ties resolve deterministically to lower indices.
                if avg > best:
                    best = avg
                    best_pair = (p, q)
        p, q = best_pair
        merged = sorted(clusters[p] + clusters[q])
        clusters = [c for idx, c in enumerate(clusters) if idx not in (p, q)]
        clusters.append(merged)
        clusters.sort(key=lambda c: c[0])
    return clusters


def _avg_link(sim: Sequence[Sequence[float]], a: Sequence[int], b: Sequence[int]) -> float:
    total = sum(sim[i][j] for i in a for j in b)
    return total / (len(a) * len(b))


def central_index(sim: Sequence[Sequence[float]], members: Sequence[int]) -> int:
    # The chunk nearest the cluster centroid, approximated by highest mean
    # similarity to its clustermates — the one excerpt that best represents the
    # theme for the Engine's naming call. Tie-break to the smallest index.
    if len(members) == 1:
        return members[0]
    best = members[0]
    best_score = -2.0
    for i in members:
        score = sum(sim[i][j] for j in members if j != i) / (len(members) - 1)
        if score > best_score:
            best_score = score
            best = i
    return best
