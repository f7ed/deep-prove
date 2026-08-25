# Experiment 1 Archive Manifest

Recommended branch: `exp/post-requant-kv-cache`

Create a new archival branch from the current `baseline` worktree instead of renaming `baseline`, because it tracks `origin/baseline`:

```bash
git switch -c exp/post-requant-kv-cache
```

## Source changes to preserve

- `zkml/src/bin/bench/llm.rs`
- `zkml/src/layers/einsum/mod.rs`
- `zkml/src/layers/transformer/mod.rs`
- `zkml/src/model/llm.rs`
- `zkml/src/model/mod.rs`

These files contain the inference-only cache mode, post-ReQuant runner, correctness capture, baseline/decode timing, CLI flags, and runtime cache bypass.

## Reports to preserve

- `takeaway-exp-post-requant-kv-cache.md`
- `exp-results/baseline.md`
- `exp-results/kv-cache.md`
- `exp-results/plan.md`
- `exp-results/archive-manifest.md`

## Final raw and derived data to preserve

Baseline runs:

- `baseline-seq32.csv`
- `exp-results/baseline-seq64.csv`
- `exp-results/baseline-seq128.csv`

Decode A/B:

- `exp-results/kv-cache-pre.csv`
- `exp-results/kv-cache-post-bypass.csv`
- `exp-results/kv-cache-summary.csv`

Correctness and integration:

- `exp-results/kv-cache-bit-exact.csv`
- `exp-results/kv-cache-post-proof-smoke.csv`

Standard baseline-boundary A/B:

- `exp-results/kv-cache-baseline-pre-phases.csv`
- `exp-results/kv-cache-baseline-post-phases.csv`
- `exp-results/kv-cache-baseline-summary.csv`

The repository ignores `*.csv`, so add only these selected results with `git add -f`. Do not use `git add .`, because `model_cache/` is an untracked 524 MB hydrated model directory.

## Do not upload

- `model_cache/`: re-downloadable GPT-2 weights and tokenizer assets.
- `target/`: build artifacts.
- Intermediate `kv-cache-correctness*.csv`, `kv-cache-post-final.csv`, `kv-cache-post.csv`, `kv-smoke-*.csv`, and superseded proof-smoke CSVs.
- Temporary validator dependencies or skill staging directories under `/private/tmp`.

## Suggested archive commands

Review the diff before staging:

```bash
git diff --check
git diff --stat
git status --short
```

Stage source and reports explicitly:

```bash
git add \
  zkml/src/bin/bench/llm.rs \
  zkml/src/layers/einsum/mod.rs \
  zkml/src/layers/transformer/mod.rs \
  zkml/src/model/llm.rs \
  zkml/src/model/mod.rs \
  takeaway-exp-post-requant-kv-cache.md \
  exp-results/baseline.md \
  exp-results/kv-cache.md \
  exp-results/plan.md \
  exp-results/archive-manifest.md
```

Force-add only the selected ignored CSVs:

```bash
git add -f \
  baseline-seq32.csv \
  exp-results/baseline-seq64.csv \
  exp-results/baseline-seq128.csv \
  exp-results/kv-cache-pre.csv \
  exp-results/kv-cache-post-bypass.csv \
  exp-results/kv-cache-summary.csv \
  exp-results/kv-cache-bit-exact.csv \
  exp-results/kv-cache-post-proof-smoke.csv \
  exp-results/kv-cache-baseline-pre-phases.csv \
  exp-results/kv-cache-baseline-post-phases.csv \
  exp-results/kv-cache-baseline-summary.csv
```

Suggested commit subject:

```text
exp: evaluate post-ReQuant KV caching
```

Push only after reviewing the staged diff:

```bash
git push -u origin exp/post-requant-kv-cache
```
