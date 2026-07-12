from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from mythings.corpus import Chunk, Document, cached_extractor, chunk, extract, ingest
from mythings.embed import Embedder
from mythings.engine import Engine, NoopEngine
from mythings.github import GitHub, PullRequest, Runner, _gh, _pr_number
from mythings.isolation import Workspace, in_github_actions
from mythings.ledger import Ledger
from mythings.policy import ALLOW, Action, Decision, Policy, PolicyResult

from mycartographer.label import label_themes
from mycartographer.render import from_json, render_markdown, to_json
from mycartographer.topics import apply_labels, build_map, default_k

_TOPICS_JSON = "topics/topics.json"
_TOPICS_MD = "topics/TOPICS.md"
_BRANCH = "my-cartographer/topics"
_EXTENSIONS = {".md", ".txt", ".pdf", ".epub"}
_SKIP_NAMES = {"README.md", "CLAUDE.md", "HARNESS.md", "TOPICS.md"}


class _AllowAll:
    def evaluate(self, action: Action) -> PolicyResult:
        return ALLOW


class PolicyDenied(RuntimeError):
    pass


@dataclass(frozen=True)
class Result:
    outcome: str  # success | skipped | failure
    themes: int
    docs: int
    detail: str
    pr: int | None = None


def discover_paths(dirs: list[str]) -> list[Path]:
    # Deterministic corpus resolution: every supported file under the given
    # directories, sorted, minus the repo's own scaffolding files.
    found: list[Path] = []
    for d in dirs:
        root = Path(d)
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _EXTENSIONS:
                continue
            if path.name in _SKIP_NAMES or any(part.startswith(".") for part in path.parts):
                continue
            found.append(path)
    return found


class Cartographer:
    def __init__(
        self,
        *,
        source: str | Path = ".",
        ledger: Ledger,
        repo: str | None = None,
        base: str = "main",
        engine: Engine | None = None,
        policy: Policy | None = None,
        embedder: Embedder | None = None,
        runner: Runner = _gh,
    ) -> None:
        self.source = Path(source)
        self.ledger = ledger
        self.repo = repo
        self.base = base
        self.engine: Engine = engine or NoopEngine()
        self.policy: Policy = policy or _AllowAll()
        self.embedder = embedder
        self.runner = runner
        self.github = GitHub(repo, runner=runner)

    def map(
        self,
        *,
        corpus: list[str],
        themes_k: int | None = None,
        cache: str | None = None,
        out: str | None = None,
        no_pr: bool = False,
    ) -> Result:
        docs, chunks = self._ingest(corpus, cache=cache)
        if len(docs) < 2:
            return self._skip(len(docs), "need at least two documents to map a corpus")

        k = themes_k or default_k(len(docs))
        topic_map, excerpts = build_map(docs, chunks, themes_k=k, embedder=self.embedder)
        labels = label_themes(self.engine, topic_map, excerpts)
        topic_map = apply_labels(topic_map, labels)

        js = to_json(topic_map)
        md = render_markdown(topic_map)
        edges = sum(len(t.prereqs) for t in topic_map.themes)
        n_themes = len(topic_map.themes)
        detail = f"mapped {len(docs)} docs into {n_themes} themes ({edges} prereq edges)"

        # A plain local write for inspection — no git, no PR. This is the mode a
        # human runs over a bare folder of PDFs.
        if out is not None:
            base = Path(out) / "topics"
            base.mkdir(parents=True, exist_ok=True)
            (base / "topics.json").write_text(js, encoding="utf-8")
            (base / "TOPICS.md").write_text(md, encoding="utf-8")
            return self._success(len(topic_map.themes), len(docs), detail, None)

        try:
            pr, wrote = self._write(js, no_pr=no_pr)
        except PolicyDenied as denied:
            self._record("failure", 0, len(docs), str(denied), None)
            return Result("failure", 0, len(docs), str(denied))
        if not wrote:
            return self._skip(len(docs), "topic map already up to date")
        return self._success(len(topic_map.themes), len(docs), detail, pr.number if pr else None)

    def _ingest(
        self, corpus: list[str], *, cache: str | None
    ) -> tuple[list[Document], list[Chunk]]:
        paths = discover_paths(corpus)
        extractor = cached_extractor(Path(cache)) if cache else extract
        docs = ingest(paths, extractor=extractor)
        chunks: list[Chunk] = []
        for doc in docs:
            chunks.extend(chunk(doc))
        return docs, chunks

    def _write(self, js: str, *, no_pr: bool) -> tuple[PullRequest | None, bool]:
        existing_pr = None if no_pr else self._existing_pr()
        base_ref = _BRANCH if (existing_pr is not None or self._local_branch()) else self.base
        with Workspace(self.source, base_ref) as tree:
            target = tree / _TOPICS_JSON
            existing = target.read_text(encoding="utf-8") if target.exists() else ""
            if existing == js:
                return existing_pr, False
            topic_map = from_json(js)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(js, encoding="utf-8")
            (tree / _TOPICS_MD).write_text(render_markdown(topic_map), encoding="utf-8")
            self._git(tree, ["checkout", "-B", _BRANCH])
            self._git(tree, ["add", _TOPICS_JSON, _TOPICS_MD])
            self._git(tree, ["commit", "-m", "topics: refresh topic map"])
            if no_pr:
                return None, True
            self._git(tree, ["push", "--force", "-u", "origin", _BRANCH])
        if existing_pr is not None:
            return existing_pr, True
        self._guard(f"gh pr create --head {_BRANCH} --base {self.base}")
        pr = self.github.open_pr(
            title="topics: refresh topic map",
            body="Refreshed topics/TOPICS.md + topics/topics.json from the latest corpus scan.",
            base=self.base,
            head=_BRANCH,
        )
        return pr, True

    def _existing_pr(self) -> PullRequest | None:
        if self.repo is None:
            return None
        argv = ["pr", "list", "--head", _BRANCH, "--state", "open", "--json", "number,url"]
        argv += ["--repo", self.repo]
        rows = json.loads(self.runner(argv))
        if not rows:
            return None
        row = rows[0]
        return PullRequest(number=row.get("number") or _pr_number(row["url"]), url=row["url"])

    def _local_branch(self) -> bool:
        proc = subprocess.run(
            ["git", "-C", str(self.source), "rev-parse", "--verify", "--quiet", _BRANCH],
            capture_output=True,
            text=True,
        )
        return proc.returncode == 0

    def _git(self, tree: Path, argv: list[str]) -> None:
        self._guard("git " + " ".join(argv))
        proc = subprocess.run(["git", "-C", str(tree), *argv], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"git {' '.join(argv)} failed: {proc.stderr.strip()}")

    def _guard(self, command: str) -> None:
        gate = self.policy.evaluate(Action(kind="bash", payload={"command": command}))
        if gate.under(unattended=in_github_actions()) is not Decision.ALLOW:
            raise PolicyDenied(f"policy blocked: {command} ({gate.reason or gate.decision})")

    def _skip(self, docs: int, detail: str) -> Result:
        self._record("skipped", 0, docs, detail, None)
        return Result("skipped", 0, docs, detail)

    def _success(self, themes: int, docs: int, detail: str, pr: int | None) -> Result:
        self._record("success", themes, docs, detail, pr)
        return Result("success", themes, docs, detail, pr)

    def _record(self, outcome: str, themes: int, docs: int, detail: str, pr: int | None) -> None:
        self.ledger.record(
            tool="mycartographer",
            kind="topic_map",
            outcome=outcome,
            detail=detail,
            theme_count=themes,
            doc_count=docs,
            pr=pr,
        )
