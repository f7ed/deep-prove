# DeepProve Experiment Index

This directory is the stable handoff layer on `master`. It contains concise summaries and pointers; full prototype code and raw evidence remain frozen on each experiment branch.

| ID | Experiment | Status | Archive | Main conclusion | Follow-up decision |
|---|---|---|---|---|---|
| 001 | Post-ReQuant KV cache | Archived | `exp/post-requant-kv-cache@b6c32e37` | Bit-exact and reduces repeated historical ReQuant, but does not reliably improve the current end-to-end baseline | Existing pre-ReQuant state can support an initial naive protocol; practical reusable state should target post-ReQuant K/V |

Detailed handoff:

- [Experiment 001 summary](001-post-requant-kv-cache/README.md)
- [Experiment 001 full report](001-post-requant-kv-cache/Aug-26-exp-post-requant-kv-cache.md)

For a new experiment:

1. Copy [TEMPLATE.md](TEMPLATE.md) into `experiments/<NNN>-<short-topic>/README.md`.
2. Create `exp/<short-topic>` from the chosen base commit.
3. Mark the index entry as `In progress`.
4. On completion, freeze the branch and update the entry with its commit and conclusion.

Do not merge experimental source code into `master` merely to make it discoverable. The branch and commit pointer are the source of truth for implementation and raw data.
