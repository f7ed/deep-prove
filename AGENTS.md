# DeepProve Experiment Handoff

## Start here

Read `experiments/README.md` before continuing local research work. It indexes archived experiments, exact branches and commits, conclusions, and follow-up decisions.

`master` should contain stable handoff metadata only. Keep experimental source changes and full raw results on dedicated `exp/<short-topic>` branches unless the user explicitly decides to merge them.

## Choosing a base

- For independent work, branch from the current `master`.
- To continue an archived experiment, branch from its exact archive commit, not from its written summary alone.
- Record the base commit in the new experiment handoff. Do not assume an experiment branch is based directly on the current `master`.

## Experiment workflow

1. Create `exp/<short-topic>` and register it in `experiments/README.md` as in progress.
2. Define the exact baseline workload and timing boundaries before changing code.
3. Keep setup, model or witness execution, proof generation, and verification separate.
4. For LLM inference, distinguish prompt prefill, incremental decode, and final trace generation when the implementation combines them.
5. Establish correctness before performance and preserve null or negative end-to-end results.
6. Archive one final raw artifact per variant, a derived summary, correctness evidence, configuration, source diff, and a concise takeaway.
7. Freeze the experiment in one commit, push its branch, then update the master experiment index without merging prototype code.

Use `$zkml-experiment-takeaway` when it is available to produce the closing report and archive manifest. Otherwise follow `experiments/TEMPLATE.md`.

## Data hygiene

- Never commit `model_cache/`, `target/`, secrets, or re-downloadable model weights.
- This repository ignores `*.csv`. Force-add only final CSVs named in an experiment manifest; do not use `git add .` for experiment archives.
- Keep exploratory smoke/debug CSVs local unless they are the only evidence of a required correctness or integration check.
- Record commands, random seeds, commit hashes, backend, quantization width, thread count, and machine configuration in the archived report.

## Reporting invariants

- Define the denominator for every latency share.
- Do not turn an operator-level reduction into an end-to-end speedup claim.
- State whether `inference_time` includes token inference, witness construction, or a final trace-generation forward.
- Separate confirmed implementation facts, measured observations, design judgments, and future protocol work.
- For reusable state, state the representation stored today and the representation a future protocol should commit or prove.

## Current handoff

Experiment 001, post-ReQuant KV caching, is archived at branch `exp/post-requant-kv-cache`, commit `b6c32e37`. The master summary is `experiments/001-post-requant-kv-cache/README.md`.
