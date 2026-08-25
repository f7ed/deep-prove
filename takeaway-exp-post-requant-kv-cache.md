# Experiment 1: Post-ReQuant KV Cache

## Motivation

Prover-side online zkML latency contains two major components:

```text
online prover latency
= inference and trace construction
+ cryptographic proof generation
```

The local DeepProve baseline provides the following end-to-end measurements:

| Sequence | Prompt + generated | Quantization | Context generation | Inference | Proof generation | Verification | Peak RSS | Proof size | Throughput |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 30 + 2 | 1.465 s | 15.731 s | 11.887 s | 52.274 s | 1.291 s | 11,046 MiB | 9,235,306 B (8.81 MiB) | 0.6122 tokens/s |
| 64 | 62 + 2 | 1.736 s | 15.367 s | 20.784 s | 80.785 s | 1.443 s | 8,510 MiB | 10,251,087 B (9.78 MiB) | 0.7922 tokens/s |
| 128 | 126 + 2 | 2.210 s | 14.565 s | 38.459 s | 131.389 s | 1.538 s | 8,096 MiB | 11,292,990 B (10.77 MiB) | 0.9742 tokens/s |

For the prover-side online latency considered here, inference and proof generation contribute:

| Sequence | Inference + proof generation | Inference share | Proof-generation share |
|---:|---:|---:|---:|
| 32 | 64.161 s | 18.5% | 81.5% |
| 64 | 101.569 s | 20.5% | 79.5% |
| 128 | 169.848 s | 22.6% | 77.4% |

Verification is reported separately. Quantization and context generation are also excluded from this composition because they are treated as setup costs rather than part of the online prover comparison. The non-monotonic peak-RSS measurements are single-run process peaks and should not be interpreted as memory improvements.

Inference is therefore not negligible. Moreover, DeepProve's `inference_time` is not ordinary native model inference: it executes the zk-friendly quantized model and produces the computation trace consumed by the prover. It should consequently be included in the zkML system cost, even when the certification algorithm itself focuses primarily on proof generation.

This experiment investigated whether DeepProve's autoregressive KV cache performs avoidable repeated computation and, more importantly, which K/V representation should eventually be committed as reusable zkKV state.

## What DeepProve includes in `inference_time`

DeepProve's inference pipeline contains three stages.

### 1. Prompt prefill

The complete prompt is processed by all GPT-2 layers. This produces the first new token and initializes the K/V history.

### 2. Autoregressive decode

Each subsequent forward receives one new token and reuses historical K/V.

The original implementation follows:

```text
new-token Q/K/V projection
-> wide integer K/V
-> append to pre-ReQuant KV cache
-> ReQuant the complete historical K/V
-> attention
```

Projection is performed only for the new token, but historical K/V values are repeatedly ReQuantized at every decoding step.

### 3. Final trace-generation forward

After token generation, DeepProve resets the relevant caches and runs the complete generated sequence again:

```text
complete generated sequence
-> full quantized forward
-> collect intermediate values
-> produce the trace consumed by the prover
```

This is not cryptographic proof generation, but it is included in DeepProve's `inference_time`.

Repeated sequence-64 measurements provide the following approximate inference breakdown:

| Inference stage | Median time | Approximate share |
|---|---:|---:|
| 62-token prompt prefill | 10.225 s | 48% |
| Two incremental forwards | 0.748 s | 3.5% |
| Final 64-token trace-generation forward | 10.417 s | 49% |

The stage medians were calculated independently, so their percentages are approximate. Nevertheless, they clearly show that incremental decode is only a small part of the current baseline.

## Code modification

We implemented an inference-only post-ReQuant KV-cache prototype:

```text
new-token Q/K/V projection
-> wide integer K/V
-> ReQuant only the new K/V
-> append to post-ReQuant KV cache
-> reuse the attention-ready history directly
```

The prototype temporarily bypasses the original pre-ReQuant K/V cache during autoregressive inference and accumulates the output of the existing K/V ReQuant nodes instead. It preserves:

- tensor layout and head structure;
- quantization scales and integer rounding;
- positional and attention semantics;
- the original proof graph and proof protocol.

No PCS, proof statement, lookup proof, checkpoint, recursion, or soundness logic was modified.

## Correctness

Original and modified inference were compared using identical prompts and token inputs.

| Comparison | Result |
|---|---:|
| Post-ReQuant K/V tensors | 384/384 bit-exact |
| Attention outputs and final logits | 208/208 bit-exact |
| Generated tokens | 16/16 bit-exact |

This confirms that concatenating token-wise post-ReQuant K/V is equivalent to ReQuantizing the concatenated historical tensor under DeepProve's current element-wise integer inference semantics.

## Performance results

### Historical K/V ReQuant and incremental decode

The decode experiment used a 64-token prefix. Each workload was executed three times, and medians are reported.

| Decode workload | ReQuant elements reduced | ReQuant time reduced | Incremental decode speedup |
|---:|---:|---:|---:|
| +4 tokens | 74.4% | 73.8% | 6.5% |
| +8 tokens | 86.9% | 86.3% | 4.4% |
| +16 tokens | 93.1% | 92.6% | 6.6% |

The optimization successfully removes nearly all repeated historical K/V ReQuant work. ReQuant is only one component of a transformer forward, however, so the resulting incremental decode improvement is approximately 4%-7%, rather than 74%-93%.

The physical cache size did not decrease because both representations are currently stored as unpacked Rust `i64` values. Cache packing and compression were outside the scope of this experiment.

### Standard DeepProve sequence-64 baseline

The standard baseline is different from the longer decode workload. It uses a 62-token prompt, retains two generated tokens, and performs a separate 64-token full forward for trace generation.

| Timed region | Original | Post-ReQuant | Change |
|---|---:|---:|---:|
| Total baseline `inference_time` | 21.349 s | 21.636 s | no reliable improvement |
| Prompt prefill | 10.225 s | 10.288 s | unchanged |
| Two incremental forwards | 748 ms | 633 ms | 15.4% faster |
| Final trace-generation forward | 10.417 s | 10.691 s | unchanged |

Incremental decode accounts for only approximately 3.5% of the baseline inference time. The relevant cache work became faster and saved approximately 115 ms, but the unchanged prefill and final trace-generation forward dominate the total and have greater run-to-run variation than the saved time.

The experiment therefore demonstrates a real decode-level optimization but not a measurable end-to-end improvement for the current sequence-64 baseline.

## Analysis

The limited overall improvement has three causes:

1. The baseline performs very few decode steps, so there is little historical KV reuse.
2. ReQuant is only a small part of each token forward; matrix multiplications, attention, FFN layers, and vocabulary projection remain unchanged.
3. DeepProve performs two expensive long-sequence computations: prompt prefill and a final full-sequence forward for trace generation. Post-ReQuant caching does not optimize either one.

The main system-level bottleneck exposed by this experiment is therefore not historical K/V ReQuant itself, but full-sequence execution, particularly the additional trace-generation forward.

## Takeaways

### 1. ML-side optimization

The experiment confirms that DeepProve currently caches pre-ReQuant wide integer K/V. Post-ReQuant K/V is a more natural reusable inference state because it is already in the representation consumed directly by attention.

Post-ReQuant caching is therefore a valid ML-side optimization for long autoregressive decoding, although its end-to-end impact is small in the current baseline.

A larger ML-side opportunity is to eliminate duplicated work between autoregressive inference and the final trace-generation forward. The final forward cannot simply be removed while the prover still requires its trace, but the trace could potentially be streamed, accumulated, or constructed by reusing intermediate results from token generation.

### 2. Implications for the zkKV protocol

The experiment also informs which state a future protocol should commit:

- Committing pre-ReQuant K/V matches the existing implementation and may be sufficient for an initial naive KV-cache proof with fewer ML-side changes.
- Committing post-ReQuant K/V is likely preferable for a practical reusable-state protocol because attention can consume it directly without repeatedly processing historical values.

A real post-ReQuant zkKV protocol would need to prove that:

- cached values were correctly produced by K/V projection and ReQuant;
- each new cache state correctly extends the previous state;
- scale, layout, head structure, token order, and positional semantics are preserved;
- the cache is bound to the correct model, prompt, and generated-token history.

These protocol changes can reasonably be deferred because they require coordinated modifications to the ML graph, trace construction, cache state management, and proof statement. An initial protocol prototype can follow the existing pre-ReQuant representation, while the post-ReQuant representation should remain the target for a practical reusable-state design.

## Overall conclusion

> The experiment validates post-ReQuant K/V as the better conceptual reusable state and removes repeated historical ReQuant computation. Its decode-level benefit is real but modest, and it does not measurably improve the current end-to-end baseline because prompt prefill and final trace generation dominate inference. The primary contribution of the experiment is therefore clarifying the ML/protocol boundary, identifying what a future zkKV protocol should commit, and exposing trace-generation inference as a more important system bottleneck.
