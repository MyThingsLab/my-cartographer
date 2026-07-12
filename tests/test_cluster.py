from __future__ import annotations

import pytest

from mycartographer.cluster import (
    agglomerate,
    central_index,
    cosine_sparse,
    similarity_matrix,
    tfidf_vectors,
)


def test_tfidf_vectors_are_deterministic_and_normalized() -> None:
    texts = ["alpha beta beta", "gamma delta"]
    a = tfidf_vectors(texts)
    b = tfidf_vectors(texts)
    assert a == b
    for vec in a:
        assert sum(v * v for v in vec.values()) == pytest.approx(1.0)


def test_cosine_sparse_identical_and_disjoint() -> None:
    assert cosine_sparse({"a": 1.0}, {"a": 1.0}) == pytest.approx(1.0)
    assert cosine_sparse({"a": 1.0}, {"b": 1.0}) == 0.0
    assert cosine_sparse({}, {"a": 1.0}) == 0.0


def test_similarity_matrix_is_symmetric_with_unit_diagonal() -> None:
    # Four docs keep shared within-theme terms under the max-document-frequency
    # cap (a term must be in <=50% of docs to survive).
    vecs = tfidf_vectors(["cat dog pet", "cat dog pet", "moon star sky", "moon star sky"])
    sim = similarity_matrix(vecs)
    assert sim[0][0] == 1.0
    assert sim[0][1] == pytest.approx(sim[1][0])
    assert sim[0][1] > sim[0][2]  # the two animal docs are nearer each other


def test_similarity_matrix_supports_dense_vectors() -> None:
    sim = similarity_matrix([(1.0, 0.0), (0.0, 1.0)])
    assert sim[0][1] == pytest.approx(0.0)


def test_agglomerate_groups_the_obvious_clusters() -> None:
    # Two tight pairs and a loner; cut at 2 must keep the pairs together.
    texts = [
        "linear algebra matrix eigen",
        "matrix eigen linear algebra",
        "thermodynamics entropy heat",
        "entropy heat thermodynamics",
    ]
    sim = similarity_matrix(tfidf_vectors(texts))
    clusters = agglomerate(sim, 2)
    assert sorted(sorted(c) for c in clusters) == [[0, 1], [2, 3]]


def test_agglomerate_is_deterministic() -> None:
    sim = similarity_matrix(tfidf_vectors(["a b", "a b", "c d", "c d", "e f"]))
    assert agglomerate(sim, 3) == agglomerate(sim, 3)


def test_agglomerate_returns_singletons_when_k_exceeds_n() -> None:
    sim = similarity_matrix(tfidf_vectors(["a", "b"]))
    assert agglomerate(sim, 5) == [[0], [1]]


def test_agglomerate_rejects_nonpositive_k() -> None:
    with pytest.raises(ValueError):
        agglomerate([[1.0]], 0)


def test_central_index_picks_the_most_connected_member() -> None:
    # Node 1 is similar to both 0 and 2; 0 and 2 are far apart.
    sim = [
        [1.0, 0.9, 0.1],
        [0.9, 1.0, 0.9],
        [0.1, 0.9, 1.0],
    ]
    assert central_index(sim, [0, 1, 2]) == 1


def test_central_index_of_a_singleton_is_itself() -> None:
    assert central_index([[1.0]], [0]) == 0
