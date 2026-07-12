from __future__ import annotations

import json
from pathlib import Path

from mythings.ledger import Ledger

from conftest import ScriptedEngine, fake_gh, make_repo, read_committed, write_corpus
from mycartographer.cartographer import Cartographer, discover_paths

_LABELS = json.dumps(
    {
        "themes": [
            {"cluster": 0, "label": "Algebra", "blurb": "vectors", "prereqs": []},
            {"cluster": 1, "label": "Thermo", "blurb": "heat", "prereqs": [0]},
        ]
    }
)

_DOCS = {
    "algebra1.md": "matrix eigenvalue vector\n\nlinear basis rank\n\neigen decomposition span",
    "algebra2.md": "vector basis rank matrix\n\neigenvalue linear span\n\nmatrix inverse basis",
    "thermo1.md": "entropy heat temperature\n\ncarnot cycle work\n\nthermal equilibrium gas",
    "thermo2.md": "temperature entropy heat\n\nwork carnot engine\n\nthermal reservoir gas",
}


def test_discover_paths_finds_supported_files_and_skips_scaffolding(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("x")
    (tmp_path / "b.pdf").write_text("x")
    (tmp_path / "README.md").write_text("x")
    (tmp_path / "notes.rst").write_text("x")  # unsupported extension
    found = [p.name for p in discover_paths([str(tmp_path)])]
    assert found == ["a.md", "b.pdf"]


def test_map_skips_a_corpus_with_fewer_than_two_docs(tmp_path: Path) -> None:
    corpus = write_corpus(tmp_path / "c", {"only.md": "one document"})
    ledger = Ledger(tmp_path / "ledger.jsonl")
    result = Cartographer(source=tmp_path, ledger=ledger, engine=ScriptedEngine("{}")).map(
        corpus=[corpus], out=str(tmp_path / "o")
    )
    assert result.outcome == "skipped"
    assert ledger.read(tool="mycartographer", kind="topic_map")[0].outcome == "skipped"


def test_map_out_mode_writes_artifacts_without_a_pr(tmp_path: Path) -> None:
    corpus = write_corpus(tmp_path / "c", _DOCS)
    out = tmp_path / "o"
    ledger = Ledger(tmp_path / "ledger.jsonl")
    result = Cartographer(
        source=tmp_path, ledger=ledger, engine=ScriptedEngine(_LABELS)
    ).map(corpus=[corpus], themes_k=2, out=str(out))

    assert result.outcome == "success"
    assert result.themes == 2
    assert result.docs == 4
    data = json.loads((out / "topics" / "topics.json").read_text())
    assert len(data["themes"]) == 2
    md = (out / "topics" / "TOPICS.md").read_text()
    assert "```mermaid" in md and "Algebra" in md


def test_map_is_deterministic_in_out_mode(tmp_path: Path) -> None:
    corpus = write_corpus(tmp_path / "c", _DOCS)
    ledger = Ledger(tmp_path / "ledger.jsonl")

    def run(dest: str) -> str:
        Cartographer(source=tmp_path, ledger=ledger, engine=ScriptedEngine(_LABELS)).map(
            corpus=[corpus], themes_k=2, out=dest
        )
        return (Path(dest) / "topics" / "topics.json").read_text()

    assert run(str(tmp_path / "o1")) == run(str(tmp_path / "o2"))


def test_map_writes_artifacts_and_opens_a_pr(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    corpus = write_corpus(tmp_path / "c", _DOCS)
    ledger = Ledger(tmp_path / "ledger.jsonl")
    result = Cartographer(
        source=repo,
        ledger=ledger,
        repo="owner/name",
        runner=fake_gh(),
        engine=ScriptedEngine(_LABELS),
    ).map(corpus=[corpus], themes_k=2)

    assert result.outcome == "success"
    assert result.pr == 7
    committed = read_committed(repo, "my-cartographer/topics", "topics/TOPICS.md")
    assert "Algebra" in committed
    assert ledger.read(tool="mycartographer", kind="topic_map")[0].outcome == "success"


def test_map_no_pr_writes_a_local_branch(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    corpus = write_corpus(tmp_path / "c", _DOCS)
    ledger = Ledger(tmp_path / "ledger.jsonl")
    result = Cartographer(source=repo, ledger=ledger, engine=ScriptedEngine(_LABELS)).map(
        corpus=[corpus], themes_k=2, no_pr=True
    )
    assert result.outcome == "success"
    assert result.pr is None
    # The commit lives on the local branch only — never pushed to origin.
    import subprocess

    local = subprocess.run(
        ["git", "-C", str(repo), "show", "my-cartographer/topics:topics/topics.json"],
        capture_output=True,
        text=True,
    )
    assert local.returncode == 0 and "themes" in local.stdout
    assert read_committed(repo, "my-cartographer/topics", "topics/topics.json") == ""


def test_map_is_idempotent_on_an_unchanged_corpus(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    corpus = write_corpus(tmp_path / "c", _DOCS)
    ledger = Ledger(tmp_path / "ledger.jsonl")

    def run():
        return Cartographer(source=repo, ledger=ledger, engine=ScriptedEngine(_LABELS)).map(
            corpus=[corpus], themes_k=2, no_pr=True
        )

    assert run().outcome == "success"
    assert run().outcome == "skipped"  # nothing changed → no second write
