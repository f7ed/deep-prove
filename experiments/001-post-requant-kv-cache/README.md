# Experiment 001: Post-ReQuant KV Cache

## Archive

| Item | Value |
|---|---|
| Status | Archived |
| Branch | [`exp/post-requant-kv-cache`](https://github.com/f7ed/deep-prove/tree/exp/post-requant-kv-cache) |
| Archive commit | `b6c32e37` |
| Parent experiment setup | `d8b36766` |
| Full report on `master` | [Aug-26-exp-post-requant-kv-cache.md](Aug-26-exp-post-requant-kv-cache.md) |
| Full takeaway | [takeaway-exp-post-requant-kv-cache.md](https://github.com/f7ed/deep-prove/blob/exp/post-requant-kv-cache/takeaway-exp-post-requant-kv-cache.md) |
| Detailed report | [exp-results/kv-cache.md](https://github.com/f7ed/deep-prove/blob/exp/post-requant-kv-cache/exp-results/kv-cache.md) |
| Baseline report | [exp-results/baseline.md](https://github.com/f7ed/deep-prove/blob/exp/post-requant-kv-cache/exp-results/baseline.md) |
| Raw-data manifest | [exp-results/archive-manifest.md](https://github.com/f7ed/deep-prove/blob/exp/post-requant-kv-cache/exp-results/archive-manifest.md) |

The archive branch includes both `d8b36766` and `b6c32e37`; use the branch or archive commit when reproducing the experiment rather than assuming `b6c32e37` is directly based on current `master`.

## Research question

Can DeepProve cache K/V after ReQuant so historical values are reused in their attention-ready representation without changing integer inference semantics?

This inspection also determines which representation a future zkKV protocol could commit as reusable state.

## Confirmed original flow

DeepProve cached wide integer K/V after projection and bias but before downstream ReQuant:

```text
new-token Q/K/V projection
-> append wide K/V to pre-ReQuant history
-> ReQuant the complete historical K/V
-> attention
```

Projection was already cached correctly, but historical K/V were repeatedly ReQuantized during later token forwards.

## Prototype

The inference-only runner changed the execution order to:

```text
new-token Q/K/V projection
-> ReQuant only new K/V
-> append to post-ReQuant history
-> attention consumes the complete attention-ready cache
```

It did not modify PCS commitments, proof statements, lookup proofs, checkpointing, recursion, or proof soundness logic.

## Correctness

| Check | Result |
|---|---:|
| Post-ReQuant K/V tensors | 384/384 bit-exact |
| Attention outputs and final logits | 208/208 bit-exact |
| Generated tokens | 16/16 bit-exact |
| Post-ReQuant inference/prove/verify smoke | Passed |

## Performance

For a 64-token prefix:

| Decode workload | ReQuant elements reduced | ReQuant time reduced | Incremental decode speedup |
|---:|---:|---:|---:|
| +4 | 74.4% | 73.8% | 6.5% |
| +8 | 86.9% | 86.3% | 4.4% |
| +16 | 93.1% | 92.6% | 6.6% |

For the standard sequence-64 baseline:

| Timed region | Original | Post-ReQuant | Interpretation |
|---|---:|---:|---|
| Total `inference_time` | 21.349 s | 21.636 s | No reliable improvement |
| Prompt prefill | 10.225 s | 10.288 s | Unchanged |
| Two incremental forwards | 748 ms | 633 ms | 15.4% faster |
| Final trace-generation forward | 10.417 s | 10.691 s | Unchanged |

Incremental decode represented only about 3.5% of the baseline. Prefill and the final full trace-generation forward dominated the result, so the targeted optimization was real but did not translate into end-to-end baseline acceleration.

## Handoff decisions

1. The existing pre-ReQuant representation can be used for an initial naive KV-state proof with fewer ML-side changes.
2. Post-ReQuant K/V is the better target for a practical reusable-state protocol because it is directly consumable by attention.
3. A post-ReQuant protocol must prove projection plus ReQuant, append-only state transitions, scale/layout/head/position semantics, and binding to the correct model and token history.
4. Formal integration can be deferred because it requires coordinated ML graph, trace, state-management, and proof-statement changes.
5. Trace generation is a larger ML-side bottleneck than repeated K/V ReQuant in the current baseline and should be studied separately.

## Starting the next experiment

- If the next experiment builds on the post-ReQuant runner, branch from `b6c32e37`.
- If it is independent, branch from current `master` and use this document only as evidence.
- Register the new experiment in `experiments/README.md` before implementation.
