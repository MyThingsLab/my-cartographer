from __future__ import annotations

from mycartographer.render import from_json, render_markdown, to_json
from mycartographer.topics import ChunkRef, Theme, TopicMap


def _map() -> TopicMap:
    themes = (
        Theme(
            id=0,
            top_terms=("vector", "matrix"),
            doc_ids=("algebra",),
            citation=ChunkRef("algebra", 1, 10, 20),
            label="Linear Algebra",
            blurb="vectors and maps",
        ),
        Theme(
            id=1,
            top_terms=("entropy",),
            doc_ids=("thermo",),
            citation=ChunkRef("thermo", 0, 0, 5),
            label="Thermodynamics",
            blurb="heat and entropy",
            prereqs=(0,),
        ),
    )
    return TopicMap(
        themes=themes,
        unassigned=("stray",),
        doc_titles={"algebra": "Linear Algebra", "thermo": "Thermodynamics", "stray": "Stray Doc"},
        corpus_paths=("/c/algebra.md", "/c/thermo.md"),
    )


def test_json_round_trips() -> None:
    original = _map()
    restored = from_json(to_json(original))
    assert restored == original


def test_from_json_handles_empty_text() -> None:
    assert from_json("") == TopicMap(themes=(), unassigned=())


def test_markdown_has_mermaid_edges_labels_and_unassigned() -> None:
    md = render_markdown(_map())
    assert "```mermaid" in md
    assert "T0 --> T1" in md  # the prerequisite edge
    assert "## Linear Algebra [algebra:1]" in md
    assert "*Prerequisites: Linear Algebra*" in md
    assert "## Unassigned" in md
    assert "Stray Doc" in md


def test_markdown_lists_member_documents_by_title() -> None:
    md = render_markdown(_map())
    assert "- Thermodynamics" in md
