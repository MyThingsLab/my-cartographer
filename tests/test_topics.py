from __future__ import annotations

from pathlib import Path

from mythings.corpus import chunk, ingest

from mycartographer.topics import (
    ChunkRef,
    Theme,
    TopicMap,
    apply_labels,
    build_map,
    default_k,
    repair_dag,
)


def _corpus(texts: dict[str, str]):
    docs = ingest([Path(f"/c/{name}.md") for name in texts], extractor=lambda p: texts[p.stem])
    chunks = []
    for doc in docs:
        chunks.extend(chunk(doc, target_chars=60))
    return list(docs), chunks


def test_default_k_clamps_into_a_browsable_band() -> None:
    assert default_k(1) == 3
    assert default_k(100) == 10
    assert default_k(10_000) == 12


def test_build_map_separates_two_lexical_themes() -> None:
    # Each chunk is dominated by its theme's vocabulary so chunk-level
    # clustering aligns with the document theme (real briefs are this repetitive).
    alg = "matrix vector eigenvalue"
    thr = "entropy heat temperature"
    texts = {
        "algebra": f"{alg} basis\n\n{alg} span\n\n{alg} rank",
        "algebra2": f"{alg} linear\n\n{alg} inverse\n\n{alg} map",
        "thermo": f"{thr} carnot\n\n{thr} work\n\n{thr} gas",
        "thermo2": f"{thr} engine\n\n{thr} reservoir\n\n{thr} cycle",
    }
    docs, chunks = _corpus(texts)
    topic_map, excerpts = build_map(docs, chunks, themes_k=2)

    assert len(topic_map.themes) == 2
    grouped = {tuple(sorted(t.doc_ids)) for t in topic_map.themes}
    assert ("algebra", "algebra2") in grouped
    assert ("thermo", "thermo2") in grouped
    # Every theme carries a citation resolvable back into the source text.
    for theme in topic_map.themes:
        assert theme.top_terms
        assert isinstance(excerpts[theme.id], str)


def test_build_map_is_byte_stable_across_runs() -> None:
    texts = {f"d{i}": f"topic{i % 3} word{i} shared common" for i in range(9)}
    docs, chunks = _corpus(texts)
    from mycartographer.render import to_json

    a = to_json(build_map(docs, chunks, themes_k=3)[0])
    b = to_json(build_map(docs, chunks, themes_k=3)[0])
    assert a == b


def test_build_map_assigns_every_document_to_exactly_one_theme() -> None:
    texts = {
        "algebra": "matrix vector eigenvalue\n\nmatrix vector eigenvalue basis",
        "algebra2": "matrix vector eigenvalue\n\nmatrix vector eigenvalue span",
        "thermo": "entropy heat temperature\n\nentropy heat temperature work",
    }
    docs, chunks = _corpus(texts)
    topic_map, _ = build_map(docs, chunks, themes_k=2)
    assigned = [d for t in topic_map.themes for d in t.doc_ids]
    assert sorted(assigned) == ["algebra", "algebra2", "thermo"]
    assert topic_map.unassigned == ()


def _theme(tid: int, prereqs: tuple[int, ...] = ()) -> Theme:
    return Theme(
        id=tid,
        top_terms=(f"term{tid}",),
        doc_ids=(f"d{tid}",),
        citation=ChunkRef(f"d{tid}", 0, 0, 1),
        prereqs=prereqs,
    )


def test_apply_labels_folds_engine_output_and_falls_back_to_top_term() -> None:
    topic_map = TopicMap(themes=(_theme(0), _theme(1)), unassigned=())
    labelled = apply_labels(topic_map, {0: ("Linear Algebra", "vectors and maps", [])})
    by_id = {t.id: t for t in labelled.themes}
    assert by_id[0].label == "Linear Algebra"
    assert by_id[0].blurb == "vectors and maps"
    assert by_id[1].label == "term1"  # unlabelled → top-term fallback


def test_repair_dag_drops_a_cycle_edge() -> None:
    # 0 -> 1 -> 0 is a cycle; the second edge added must be dropped.
    themes = [_theme(0, prereqs=(1,)), _theme(1, prereqs=(0,))]
    repaired = {t.id: t.prereqs for t in repair_dag(themes)}
    # Exactly one direction survives — the graph is acyclic.
    assert (repaired[0], repaired[1]) in {((1,), ()), ((), (0,))}


def test_repair_dag_drops_self_and_unknown_prereqs() -> None:
    themes = [_theme(0, prereqs=(0, 9)), _theme(1, prereqs=(0,))]
    repaired = {t.id: t.prereqs for t in repair_dag(themes)}
    assert repaired[0] == ()
    assert repaired[1] == (0,)
