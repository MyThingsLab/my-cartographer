from __future__ import annotations

import json
from pathlib import Path

import pytest

# Shared fakes (FakeGh, ScriptedEngine, make_git_repo, ...) come from the core
# plugin — see my-things-core/docs/CONVENTIONS.md "Shared test fixtures".
# Don't copy fixture code into this file; only domain-specific helpers live here.
from mythings.testing import FakeGh, GitRepo, ScriptedEngine, make_git_repo

pytest_plugins = ("mythings.testing",)

__all__ = ["ScriptedEngine"]


@pytest.fixture(autouse=True)
def _clean_git_env(clean_git_env: None) -> None:
    # Every test in this suite touches real git worktrees; hook-launched
    # pytest (pre-commit) must not leak GIT_* into them.
    pass


def make_repo(tmp_path: Path) -> Path:
    return make_git_repo(tmp_path, files={"README.md": "# study\n"}).path


def read_committed(repo: Path, branch: str, path: str) -> str:
    return GitRepo(path=repo, origin=repo.parent / "origin.git").read_committed(branch, path)


def write_corpus(root: Path, docs: dict[str, str]) -> str:
    # A folder of .md files with controlled vocabulary — the corpus a map runs
    # over. Returns the directory path as a str for --corpus.
    root.mkdir(parents=True, exist_ok=True)
    for name, body in docs.items():
        (root / name).write_text(body, encoding="utf-8")
    return str(root)


def fake_gh() -> FakeGh:
    state: dict[str, object] = {"opened_pr": None}

    def pr_create(argv: list[str]) -> str:
        state["opened_pr"] = {"number": 7, "url": "https://github.com/owner/name/pull/7"}
        return "https://github.com/owner/name/pull/7\n"

    def pr_list(argv: list[str]) -> str:
        return json.dumps([state["opened_pr"]] if state["opened_pr"] else [])

    return FakeGh({("pr", "list"): pr_list, ("pr", "create"): pr_create})
