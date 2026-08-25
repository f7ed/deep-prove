# DeepProve GPT-2 Post-ReQuant KV-Cache Experiment

Date: 2026-08-24 (America/New_York)

## Result

The inference-only prototype is bit-exact with the original integer inference path for the tested 64-token prefix and 16 generated tokens.

It removes most repeated historical K/V ReQuant work. For the meaningful incremental-decode cases (+4, +8, and +16), median per-token latency improved by 4.4% to 6.6%. Total prefill-plus-decode time did not improve consistently because the unchanged 64-token prefill dominates these short workloads.

This result answers the narrow research question positively: post-ReQuant K/V caching provides a measurable incremental inference benefit without changing the tested quantized inference outputs. It does not yet justify any proof-protocol or cross-proof cache claim.

## 1. Original Data Flow

GPT-2 builds three integer projections in one `EinSum` layer:

```text
X(se) @ WQ/WK/WV
  -> Q(hsd), K(hsd), V(hsd)
  -> add quantized Q/K/V bias
```

For GPT-2, `h=12` heads and `d=64`. K/V have layout `[heads, sequence, head_dim]`, and cache concatenation uses dimension 1 (the sequence dimension).

The original cache is installed on QKV output ports 1 and 2:

```text
new hidden state
  -> Q/K/V integer matrix multiplication
  -> reshape and add integer bias
  -> append new wide K/V to the existing wide K/V cache
  -> run K/V ReQuant on the entire cached history
  -> QK^T and attention-value multiplication consume ReQuant outputs
```

Concrete implementation points:

- `zkml/src/parser/llm/transformer/attention_layer.rs`: `AttentionMechanism::write_to_model` constructs the QKV, QK, softmax/value, and output-projection graph. The MHA equations define K/V as `[h,s,d]` and both attention multiplications consume those tensors.
- `zkml/src/parser/llm/models/gpt2/decoder/attention.rs`: `GPT2Attention::insert_custom_logic` calls `with_caches(vec![None, Some(1), Some(1)])`; Q is not cached, while K and V are cached along sequence dimension 1.
- `zkml/src/layers/einsum/evaluate.rs`: `EinSum::evaluate_internal` performs matrix multiplication, reshape/permutation, bias addition, and then `cache.concatenate(with_bias)`. Therefore the persistent cached values are the wide integer projection results before ReQuant.
- `zkml/src/layers/einsum/quantise.rs`: quantization computes an `intermediate_bit_size` from input bit width, 12-bit weights, contraction growth, and optional bias. It creates one `Requant` per Q/K/V output when the EinSum requests requantization.
- `zkml/src/quantization/strategy.rs` and `zkml/src/model/mod.rs`: those ReQuant operations are inserted as graph nodes between each QKV output port and its consumers.
- `zkml/src/layers/requant/evaluate.rs`: each ReQuant node applies its fixed-point scaling/clamping lookup to every element of its input tensor.

The graph does not fuse away this boundary. A prefix-64 run followed by one-token forwards therefore projects only the new token, but the original K/V ReQuant nodes receive history lengths 64, 65, 66, and so on.

All runtime tensors use the Rust `Element` type (`i64`). The pre-ReQuant effective range is described by each QKV output's computed `intermediate_bit_size`; after ReQuant, values use the downstream output scaling and target quantization range. The prototype does not change any scaling metadata or ReQuant arithmetic.

## 2. Minimal Prototype

The prototype adds an inference-runner mode rather than rewriting the model graph:

```text
new hidden state
  -> Q/K/V integer matrix multiplication
  -> reshape and add integer bias
  -> temporarily bypass the original wide K/V output caches
  -> ReQuant only the new K/V tensors
  -> append them to post-ReQuant K/V caches
  -> return the full attention-ready K/V history to the existing attention nodes
```

Relevant changes:

- `zkml/src/model/mod.rs`: `KvCacheMode`, `KvCacheRunner`, K/V ReQuant-node discovery, post-ReQuant concatenation, metrics, optional bit-exact shadow verification, and layer capture.
- `zkml/src/layers/transformer/mod.rs`: a runtime-only cache bypass flag on `ConcatenationCache`; `#[serde(skip)]` prevents it from changing the serialized model/protocol representation.
- `zkml/src/layers/einsum/mod.rs`: runtime method to bypass and restore QKV output caches.
- `zkml/src/model/llm.rs`: an inference-only prompt-prefill/autoregressive-decode runner, separate from proof trace generation.
- `zkml/src/bin/bench/llm.rs`: `--decode-only`, `--kv-cache-mode`, `--verify-kv-cache`, and `--compare-kv-cache` experiment flags.

The normal benchmark/proving path remains the default and does not use `KvCacheRunner`. No PCS, proof statement, trace format, lookup proof, checkpoint, IVC, recursion, compression, or soundness logic was modified.

## 3. Correctness

Command:

```bash
RNG_SEED=20260824 RUST_LOG=info ./target/release/bench-llm \
  --model gpt2 \
  --hf openai-community/gpt2 \
  --max-context 80 \
  --min-user-len 64 \
  --decode-only \
  --compare-kv-cache \
  --num-threads 14 \
  --bench exp-results/kv-cache-bit-exact.csv
```

The comparison runs original and prototype inference on the same prompt. Prototype verification also maintains a shadow copy of the original pre-ReQuant history, runs the original full-history ReQuant, and compares it with the accumulated post-ReQuant cache after every K/V node.

| Check | Result |
|---|---:|
| Post-ReQuant K/V tensors | 384 / 384 bit-exact |
| Attention outputs and final logits captures | 208 / 208 bit-exact |
| Generated tokens | 16 / 16 bit-exact |

Generated token IDs:

```text
1340 50 3963 3336 376 3843 11335 3963 3336 376 3843 11335 3963 3336 376 3843
```

There is no new rounding boundary: ReQuant is element-wise and uses the same parameters for each token, so concatenating after per-token ReQuant is bit-equivalent to concatenating first and ReQuantizing the full history.

The correctness mode intentionally evaluates a shadow full-history ReQuant and captures intermediate tensors, so its timing must not be used as a performance measurement.

## 4. Performance

Configuration matches the baseline machine: Apple M4 Pro MacBook Pro, 24 GB RAM, CPU/Burn ndarray backend, release build, 14 configured Rayon threads, default 12-bit quantization, and commit `9d1a53e2ef49ffa2c902b8689cd3c58057a4e662` plus the documented local changes. Prefix length is 64. Each mode/workload has three runs; the table reports the median.

Build:

```bash
cargo build --release -p zkml --bin bench-llm
```

Original runs:

```bash
for max_context in 65 68 72 80; do
  for run in 1 2 3; do
    RNG_SEED=20260824 RUST_LOG=info ./target/release/bench-llm \
      --model gpt2 \
      --hf openai-community/gpt2 \
      --max-context "$max_context" \
      --min-user-len 64 \
      --decode-only \
      --kv-cache-mode pre-requant \
      --num-threads 14 \
      --bench exp-results/kv-cache-pre.csv
  done
done
```

Prototype runs used the same command with:

```text
--kv-cache-mode post-requant
--bench exp-results/kv-cache-post-bypass.csv
```

Median results:

| Decode | Original total | Post total | Total speedup | Original incremental/token | Post incremental/token | Incremental speedup | ReQuant elements reduction | ReQuant time reduction | Peak KV cache |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| +1 | 9.532 s | 9.176 s | 3.7% | n/a | n/a | n/a | 0.0% | -2.2% | 9.00 MiB |
| +4 | 10.065 s | 10.406 s | -3.4% | 313.7 ms | 293.3 ms | 6.5% | 74.4% | 73.8% | 9.42 MiB |
| +8 | 11.427 s | 11.899 s | -4.1% | 314.8 ms | 301.0 ms | 4.4% | 86.9% | 86.3% | 9.98 MiB |
| +16 | 14.106 s | 14.006 s | 0.7% | 319.3 ms | 298.1 ms | 6.6% | 93.1% | 92.6% | 11.11 MiB |

“Incremental/token” excludes the initial prompt prefill, so +4 averages three one-token forwards, +8 averages seven, and +16 averages fifteen. The +1 case has no historical decode step and is a control; it cannot demonstrate reuse benefit.

The number of K/V ReQuant calls remains 24 per forward (K and V for each of 12 transformer blocks). The optimization reduces the tensor size processed by each call. At +16, the original processed 21,086,208 K/V elements while the prototype processed 1,456,128.

The physical KV-cache size is identical in this prototype because both representations are stored unpacked as `i64`, and packing/compression is explicitly out of scope. The whole-process peak RSS was approximately 2.0-2.4 GB for both variants and is dominated by model loading/quantization; three runs do not show a reliable memory change.

### Standard baseline `inference_time`

The decode table above is intentionally more sensitive to KV-cache reuse than the repository's standard sequence-length benchmark. To answer whether the existing `baseline.md` metric changes, the prototype was also connected to the standard `driver.run_elements(...)` generation loop and tested with `--sequence 64`.

That benchmark uses a 62-token prompt, performs only two incremental forwards, and then runs a separate full 64-token inference to create the proof trace. Across three interleaved runs per mode, median total `inference_time` was 21.349 seconds for the original path and 21.636 seconds for post-ReQuant: no reliable total improvement. Within that total, the two incremental forwards improved from 748 ms to 633 ms (15.4%), but the roughly 20.5 seconds of unchanged prefill and trace inference dominated the result.

The full breakdown and commands are recorded in `exp-results/baseline.md`; raw data is in `exp-results/kv-cache-baseline-pre-phases.csv` and `exp-results/kv-cache-baseline-post-phases.csv`.

## 5. Validation and Limitations

- `cargo check -p zkml --bin bench-llm` passed.
- `cargo clippy -p zkml --bin bench-llm -- -D warnings` passed.
- `cargo test -p zkml cache --lib --release` passed both targeted cache tests.
- A sequence-8 inference/prove/verify smoke run passed through the post-ReQuant `run_elements` path; raw output is `exp-results/kv-cache-post-proof-smoke.csv`.
- The full `zkml` library test run had 202 passing, 16 ignored, and 29 failing tests. The 29 failures read unhydrated Git LFS pointer text as SafeTensors/JSON/tokenizer assets (`unknown magic 0x73726576` or JSON parse errors), rather than failing in the cache prototype.

This is a small CPU experiment with three repetitions, not a paper benchmark. Prefill noise is large enough that total latency is not consistently lower for short decode lengths. The robust result is the large reduction in measured K/V ReQuant work and the smaller but repeatable 4.4%-6.6% median improvement in incremental decode latency.

Raw and derived results:

- `exp-results/kv-cache-pre.csv`
- `exp-results/kv-cache-post-bypass.csv`
- `exp-results/kv-cache-bit-exact.csv`
- `exp-results/kv-cache-summary.csv`
- `exp-results/kv-cache-baseline-summary.csv`

## 6. Next Question

The next experiment should decide how a post-ReQuant K/V state can be committed to and certified across proofs. That is intentionally not part of this prototype.
