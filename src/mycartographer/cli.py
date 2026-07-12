from __future__ import annotations

import argparse
import json
from pathlib import Path

from mythings.embed import ApiEmbedder, Embedder, HashingEmbedder
from mythings.engine import ClaudeCLIEngine, Engine, NoopEngine
from mythings.ledger import Ledger

from mycartographer.cartographer import Cartographer

_ENGINE_NAMES = ("noop", "claude-cli")
_EMBED_NAMES = ("none", "hashing", "api")


def build_engine(name: str, *, model: str | None = None) -> Engine:
    if name == "claude-cli":
        return ClaudeCLIEngine(model=model)
    return NoopEngine()


def build_embedder(name: str) -> Embedder | None:
    # Default (none) clusters over MyCartographer's own lexical TF-IDF vectors —
    # validated for coarse thematic separation. `api` swaps in real semantic
    # vectors from a configured MYTHINGS_EMBED_URL; `hashing` is the offline
    # deterministic embedder, mostly for exercising the embedder path.
    if name == "hashing":
        return HashingEmbedder()
    if name == "api":
        return ApiEmbedder()
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mycartographer",
        description="Cluster a document corpus into named, prerequisite-ordered themes.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("map", help="cluster a corpus into a topic map and refresh the artifacts")
    m.add_argument("--corpus", action="append", default=[], help="a directory to map (repeatable)")
    m.add_argument("--themes", type=int, help="number of themes (default: clamp(sqrt(docs),3,12))")
    m.add_argument("--cache", help="directory to cache extracted text in")
    m.add_argument("--out", help="write topics/ under this dir instead of opening a PR")
    m.add_argument("--repo", help="GitHub slug owner/name")
    m.add_argument("--source", type=Path, default=Path.cwd(), help="local checkout to write into")
    m.add_argument("--base", default="main")
    m.add_argument("--no-pr", action="store_true")
    m.add_argument("--json", action="store_true")
    m.add_argument("--ledger", type=Path, default=Path(".mythings/ledger.jsonl"))
    m.add_argument("--engine", choices=sorted(_ENGINE_NAMES), default="noop")
    m.add_argument("--engine-model", help="model for --engine claude-cli")
    m.add_argument("--embed", choices=sorted(_EMBED_NAMES), default="none")

    args = parser.parse_args(argv)
    if not args.corpus:
        parser.error("at least one --corpus directory is required")

    cartographer = Cartographer(
        source=args.source,
        ledger=Ledger(args.ledger),
        repo=args.repo,
        base=args.base,
        engine=build_engine(args.engine, model=args.engine_model),
        embedder=build_embedder(args.embed),
    )
    result = cartographer.map(
        corpus=args.corpus,
        themes_k=args.themes,
        cache=args.cache,
        out=args.out,
        no_pr=args.no_pr,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "outcome": result.outcome,
                    "themes": result.themes,
                    "docs": result.docs,
                    "detail": result.detail,
                    "pr": result.pr,
                }
            )
        )
    else:
        print(f"{result.outcome}: {result.detail}")
    return 0 if result.outcome != "failure" else 1


if __name__ == "__main__":
    raise SystemExit(main())
