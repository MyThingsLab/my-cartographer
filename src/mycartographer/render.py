from __future__ import annotations

import json

from mycartographer.topics import ChunkRef, Theme, TopicMap


def to_json(topic_map: TopicMap) -> str:
    payload = {
        "corpus_paths": list(topic_map.corpus_paths),
        "doc_titles": topic_map.doc_titles,
        "unassigned": list(topic_map.unassigned),
        "themes": [
            {
                "id": t.id,
                "label": t.label,
                "blurb": t.blurb,
                "top_terms": list(t.top_terms),
                "doc_ids": list(t.doc_ids),
                "citation": {
                    "doc_id": t.citation.doc_id,
                    "ordinal": t.citation.ordinal,
                    "start": t.citation.start,
                    "end": t.citation.end,
                },
                "prereqs": list(t.prereqs),
            }
            for t in topic_map.themes
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def from_json(text: str) -> TopicMap:
    obj = json.loads(text) if text.strip() else {}
    themes = tuple(
        Theme(
            id=t["id"],
            label=t.get("label", ""),
            blurb=t.get("blurb", ""),
            top_terms=tuple(t.get("top_terms", ())),
            doc_ids=tuple(t.get("doc_ids", ())),
            citation=ChunkRef(**t["citation"]),
            prereqs=tuple(t.get("prereqs", ())),
        )
        for t in obj.get("themes", [])
    )
    return TopicMap(
        themes=themes,
        unassigned=tuple(obj.get("unassigned", ())),
        doc_titles=obj.get("doc_titles", {}),
        corpus_paths=tuple(obj.get("corpus_paths", ())),
    )


def render_markdown(topic_map: TopicMap) -> str:
    out = ["# Topic map", ""]
    out += _mermaid(topic_map)
    for theme in topic_map.themes:
        out.append(f"## {theme.label} {theme.citation.marker()}")
        if theme.blurb:
            out.append("")
            out.append(theme.blurb)
        if theme.prereqs:
            names = ", ".join(_label_of(topic_map, p) for p in theme.prereqs)
            out.append("")
            out.append(f"*Prerequisites: {names}*")
        if theme.top_terms:
            out.append("")
            out.append(f"Top terms: {', '.join(f'`{t}`' for t in theme.top_terms)}")
        out.append("")
        out.append("Documents:")
        for doc_id in theme.doc_ids:
            out.append(f"- {topic_map.doc_titles.get(doc_id, doc_id)}")
        out.append("")
    if topic_map.unassigned:
        out.append("## Unassigned")
        out.append("")
        for doc_id in topic_map.unassigned:
            out.append(f"- {topic_map.doc_titles.get(doc_id, doc_id)}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def _mermaid(topic_map: TopicMap) -> list[str]:
    # A prerequisite graph, rendered as Mermaid so the map is browsable on
    # GitHub without leaving the checked-in markdown.
    lines = ["```mermaid", "graph TD"]
    for theme in topic_map.themes:
        lines.append(f'    T{theme.id}["{theme.label}"]')
    for theme in topic_map.themes:
        for prereq in theme.prereqs:
            lines.append(f"    T{prereq} --> T{theme.id}")
    lines.append("```")
    lines.append("")
    return lines


def _label_of(topic_map: TopicMap, theme_id: int) -> str:
    for theme in topic_map.themes:
        if theme.id == theme_id:
            return theme.label
    return f"theme-{theme_id}"
