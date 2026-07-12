from __future__ import annotations

import json

from mythings.engine import Engine, EngineRequest

from mycartographer.topics import TopicMap

_SYSTEM = (
    "You are given numbered clusters of documents, each described by its top "
    "terms and one representative excerpt. For each cluster, give a short "
    "human-readable theme label, a one-line blurb, and its prerequisite "
    "clusters (the ids a learner should study first). Use only the given "
    "numeric cluster ids -- never invent one -- and only list a prereq that is "
    "one of the given ids. Reply with a single JSON object and nothing else."
)


def _parse_json_object(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _prompt(topic_map: TopicMap, excerpts: dict[int, str], *, excerpt_chars: int) -> str:
    lines = ["Clusters:"]
    for theme in topic_map.themes:
        terms = ", ".join(theme.top_terms)
        excerpt = " ".join(excerpts.get(theme.id, "").split())[:excerpt_chars]
        lines.append(f"\n[{theme.id}] top terms: {terms}\n    excerpt: {excerpt}")
    lines.append(
        '\nReturn JSON: {"themes": [{"cluster": <int>, "label": <string>, '
        '"blurb": <string>, "prereqs": [<int>, ...]}, ...]}'
    )
    return "\n".join(lines)


def label_themes(
    engine: Engine,
    topic_map: TopicMap,
    excerpts: dict[int, str],
    *,
    excerpt_chars: int = 400,
) -> dict[int, tuple[str, str, list[int]]]:
    # The tool's single Engine call: one batched request names every cluster and
    # proposes the prerequisite edges, mirroring MyUni's decompose-and-order. An
    # invented cluster id, or a prereq pointing outside the given set, is dropped
    # here; the DAG repair (topics.repair_dag) removes any cycle afterwards.
    # Against NoopEngine (empty reply) every theme falls back to its top term.
    valid = {t.id for t in topic_map.themes}
    reply = engine.run(
        EngineRequest(
            system=_SYSTEM,
            prompt=_prompt(topic_map, excerpts, excerpt_chars=excerpt_chars),
            context={"cluster_count": len(topic_map.themes)},
        )
    )
    obj = _parse_json_object(reply.text)
    out: dict[int, tuple[str, str, list[int]]] = {}
    if obj is None:
        return out
    for item in obj.get("themes") or []:
        if not isinstance(item, dict):
            continue
        try:
            cluster = int(item.get("cluster"))
        except (TypeError, ValueError):
            continue
        if cluster not in valid:
            continue
        label = str(item.get("label", "")).strip()
        blurb = str(item.get("blurb", "")).strip()
        prereqs = _clean_prereqs(item.get("prereqs"), valid, cluster)
        out[cluster] = (label, blurb, prereqs)
    return out


def _clean_prereqs(raw: object, valid: set[int], own: int) -> list[int]:
    if not isinstance(raw, list):
        return []
    seen: dict[int, None] = {}
    for value in raw:
        try:
            pid = int(value)
        except (TypeError, ValueError):
            continue
        # A prereq must be a real, other cluster; self-reference and invented
        # ids are dropped before they ever reach the DAG repair.
        if pid in valid and pid != own:
            seen.setdefault(pid, None)
    return list(seen)
