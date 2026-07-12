from __future__ import annotations

import json

from mythings.engine import NoopEngine

from conftest import ScriptedEngine
from mycartographer.label import label_themes
from mycartographer.topics import ChunkRef, Theme, TopicMap


def _map() -> TopicMap:
    themes = tuple(
        Theme(id=i, top_terms=(f"t{i}",), doc_ids=(f"d{i}",), citation=ChunkRef(f"d{i}", 0, 0, 1))
        for i in range(2)
    )
    return TopicMap(themes=themes, unassigned=())


def _excerpts() -> dict[int, str]:
    return {0: "excerpt zero", 1: "excerpt one"}


def test_label_themes_parses_labels_blurbs_and_prereqs() -> None:
    reply = json.dumps(
        {
            "themes": [
                {"cluster": 0, "label": "Basics", "blurb": "start here", "prereqs": []},
                {"cluster": 1, "label": "Advanced", "blurb": "then this", "prereqs": [0]},
            ]
        }
    )
    out = label_themes(ScriptedEngine(reply), _map(), _excerpts())
    assert out[0] == ("Basics", "start here", [])
    assert out[1] == ("Advanced", "then this", [0])


def test_label_themes_drops_invented_cluster_ids() -> None:
    reply = json.dumps({"themes": [{"cluster": 99, "label": "Ghost", "prereqs": []}]})
    assert label_themes(ScriptedEngine(reply), _map(), _excerpts()) == {}


def test_label_themes_drops_self_and_out_of_set_prereqs() -> None:
    reply = json.dumps({"themes": [{"cluster": 1, "label": "X", "prereqs": [1, 5, 0]}]})
    out = label_themes(ScriptedEngine(reply), _map(), _excerpts())
    assert out[1] == ("X", "", [0])  # 1 (self) and 5 (unknown) removed


def test_label_themes_strips_a_code_fence() -> None:
    reply = "```json\n" + json.dumps({"themes": [{"cluster": 0, "label": "Y"}]}) + "\n```"
    out = label_themes(ScriptedEngine(reply), _map(), _excerpts())
    assert out[0][0] == "Y"


def test_label_themes_degrades_to_empty_on_noop_engine() -> None:
    assert label_themes(NoopEngine(), _map(), _excerpts()) == {}


def test_label_themes_degrades_on_unparsable_reply() -> None:
    assert label_themes(ScriptedEngine("not json at all"), _map(), _excerpts()) == {}


def test_label_themes_sends_one_call_with_cluster_context() -> None:
    engine = ScriptedEngine("{}")
    label_themes(engine, _map(), _excerpts())
    assert len(engine.calls) == 1
    assert engine.calls[0].context == {"cluster_count": 2}
