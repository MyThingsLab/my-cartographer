from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import write_corpus
from mycartographer.cli import build_embedder, build_engine, main

_DOCS = {
    "a1.md": "matrix vector eigenvalue basis\n\nmatrix vector eigenvalue span",
    "a2.md": "matrix vector eigenvalue linear\n\nmatrix vector eigenvalue rank",
    "t1.md": "entropy heat temperature work\n\nentropy heat temperature gas",
    "t2.md": "entropy heat temperature engine\n\nentropy heat temperature cycle",
}


def test_build_engine_and_embedder_selectors() -> None:
    from mythings.embed import ApiEmbedder, HashingEmbedder
    from mythings.engine import ClaudeCLIEngine, NoopEngine

    assert isinstance(build_engine("noop"), NoopEngine)
    assert isinstance(build_engine("claude-cli"), ClaudeCLIEngine)
    assert build_embedder("none") is None
    assert isinstance(build_embedder("hashing"), HashingEmbedder)
    assert isinstance(build_embedder("api"), ApiEmbedder)


def test_main_requires_a_corpus() -> None:
    with pytest.raises(SystemExit):
        main(["map"])


def test_main_maps_a_corpus_in_out_mode(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    corpus = write_corpus(tmp_path / "c", _DOCS)
    out = tmp_path / "o"
    code = main(
        [
            "map",
            "--corpus",
            corpus,
            "--out",
            str(out),
            "--themes",
            "2",
            "--ledger",
            str(tmp_path / "ledger.jsonl"),
            "--json",
        ]
    )
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["outcome"] == "success"
    assert printed["docs"] == 4
    assert (out / "topics" / "topics.json").exists()
