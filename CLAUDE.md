# my-cartographer — agent instructions

You are developing **my-cartographer**, a MyThingsLab My[X] tool.

**Inherited rules:** obey [`./HARNESS.md`](./HARNESS.md) in full — the vendored
MyThingsLab build-harness rules. Do not restate or override them. Anything not
covered here defers to `HARNESS.md`, then `my-things-core/docs/CONVENTIONS.md`.

## This tool

- **Purpose:** cluster a document corpus (a shelf of PDFs, a notes folder) into
  named, prerequisite-ordered themes — a checked-in topic map (`topics/topics.json`
  + `topics/TOPICS.md`). Bottom-up (what the material contains), the dual of
  MyUni's top-down field decomposition.
- **The single Engine call:** one batched call labels every deterministically-
  induced cluster and proposes its prerequisite clusters (`label_themes`); the
  clustering itself is deterministic and LLM-free.
- **Invariants / rules:** clustering is a pure function of its input — the map
  must be byte-stable across runs on an unchanged corpus (deterministic
  agglomerative clustering, deterministic tie-breaks). The Engine may only use
  the given cluster ids (invented ids and out-of-set prereqs are dropped); the
  prerequisite graph is always repaired to a DAG before it renders. Never invent
  bibliographic facts or cluster membership.
- **Backlog label:** my-cartographer

## Testing

Fakes come from `mythings.testing` (opt-in via `pytest_plugins` in
`tests/conftest.py`; see `my-things-core/docs/CONVENTIONS.md`, "Shared test
fixtures"). Never copy fixture code into a conftest — only domain-specific
helpers live there.
