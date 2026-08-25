# DeepProve GPT-2 Baseline and Post-ReQuant KV-Cache Plan

## Goal

在进行更大的 zkKV 或 proof-system 修改前，先完成两个小实验：

1. 建立可复现的本地 DeepProve GPT-2 性能基线。
2. 检查历史 K/V 是否被重复 requantize；只有确认存在可避免的重复操作后，才实现 post-ReQuant KV cache 的最小原型。

当前阶段不修改证明协议。

## General Rules

- 一次只推进一个步骤，并在每一步结束后记录结果。
- 保持 GPT-2 模型语义、量化尺度语义和证明逻辑不变。
- 将 quantized inference、proof generation 和 verification 分开计时。
- Task 2 的实现必须以代码路径检查结果为前提，不能预设当前 cache 存储的是 pre-ReQuant K/V。
- Task 2 的原始版本和修改版本必须使用与 Task 1 相同的机器、backend、bit width 和线程配置。

## Task 1 — Establish a Local DeepProve Baseline

### Objective

运行仓库提供的原始 GPT-2 end-to-end benchmark，为后续 A/B 对比建立本机基线。本任务不尝试复现整篇论文的 benchmark。

### Execution Steps

- [x] 定位官方入口：`zkml/src/bin/bench/llm.rs`，Cargo binary 为 `bench-llm`。
- [x] 使用 CPU release mode 构建 `bench-llm`。
- [x] 修复阻止 SafeTensors GPT-2 运行的 tied-weight commitment-key collision，并明确记录本地补丁。
- [x] 使用 sequence length 32 完成 smoke/baseline run。
- [x] 使用 sequence length 64 完成主要 baseline run。
- [x] 如果本机时间和内存允许，使用 sequence length 128 完成补充 run。
- [x] 汇总命令、指标、配置和所有复现问题。

### Build and Run Commands

```bash
cargo build --release -p zkml --bin bench-llm

RUST_LOG=info ./target/release/bench-llm \
  --model gpt2 \
  --hf openai-community/gpt2 \
  --sequence <SEQUENCE_LENGTH> \
  --bench exp-results/baseline-seq<SEQUENCE_LENGTH>.csv
```

Planned sequence lengths:

```text
32  smoke run, complete
64  primary baseline
128 optional, if practical
```

`bench-llm --sequence N` 使用 `N-2` 个 prompt tokens，并生成 2 个 tokens；记录时必须同时保存 `max_context=N` 和 `min_user_len=N-2`。

### Metrics to Record Per Run

- Sequence length and prompt length.
- Quantized inference time (`inference_time`).
- Proof generation time (`prove_full`).
- Verification time (`verify_full`).
- Peak RSS (`prove_full_memory_peak`).
- Serialized proof size (`proof_size`).
- Configured Rayon thread count.
- Any internal effective thread count reported by proving components.
- Model quantization time.
- Proving-context generation time.

### Configuration to Record

- Git commit hash and any local patch.
- Model source and model-file hashes.
- PCS and curve configuration.
- CPU/GPU backend and enabled Cargo features.
- Quantization bit width (`ZKML_BIT_LEN`).
- CPU model, physical/logical core count, and RAM.
- Rust/Cargo version and build profile.
- Complete command-line arguments.
- Whether proving parameters were generated, saved, or loaded.

### Reproducibility Checks

- Confirm that Hugging Face files are real model assets rather than Git LFS pointer text.
- Preserve the SafeTensors key-fix diff alongside results; these runs are commit `9d1a53e...` plus that compatibility fix, not a pristine-upstream checkout.
- Record the warning that a configured 14-thread run uses 8 sumcheck workers because sumcheck requires a power-of-two thread count.
- Do not enable `--accuracy`, `--distributed`, CUDA, WGPU, or saved proving parameters unless starting a separately labelled configuration.
- Keep raw CSV output for every run and do not merge timings from different runs.

### Deliverable

Update `exp-results/baseline.md` with:

1. exact commands;
2. one results-table row per sequence length;
3. relevant GPT-2 inference and benchmark code paths;
4. machine and proof configuration;
5. reproducibility issues and local compatibility patches.

## Task 2 — Investigate and Prototype Post-ReQuant KV Caching

Task 2 begins only after the sequence-64 Task 1 baseline has completed.

### Research Question

Determine whether DeepProve can store cached K/V in its attention-ready, post-requantization integer representation, so historical K/V values can be reused without requantization during later autoregressive decoding.

The investigation must distinguish between these possible flows:

```text
Current candidate flow:
Q/K/V projection -> wide integer K/V -> stored state -> ReQuant when consumed

Proposed flow:
Q/K/V projection -> wide integer K/V -> ReQuant once
-> cache post-ReQuant K/V -> reuse directly in later attention
```

### Phase A — Inspect Before Modifying

- [ ] Locate where GPT-2 Q, K, and V are generated.
- [ ] Record tensor types, effective bit widths, layouts, and scales immediately after projection.
- [ ] Locate every requantization boundary on the Q/K/V and attention paths.
- [ ] Identify the exact representation consumed by QKᵀ and attention-value multiplication.
- [ ] Determine whether a persistent autoregressive KV cache already exists.
- [ ] Identify its owning structs, cache lifetime, concatenation dimension, and stored representation.
- [ ] Trace a prefix plus at least one decode step to determine whether historical K/V are requantized again.
- [ ] Determine whether delayed requantization, graph transforms, or operator fusion alter this path.
- [ ] Report concrete files, structs, functions, tensor shapes, and scale transitions.

Likely starting points for inspection include:

- `zkml/src/parser/llm/models/gpt2/decoder/attention.rs`
- `zkml/src/parser/llm/transformer/attention_layer.rs`
- `zkml/src/layers/einsum/`
- `zkml/src/layers/requant/`
- `zkml/src/model/llm.rs`

### Decision Gate

Do not implement a cache change unless Phase A confirms all of the following:

1. historical K/V are retained across autoregressive steps;
2. the retained representation is pre-ReQuant or otherwise causes historical values to be requantized again;
3. attention can consume the post-ReQuant representation without changing layout, scale, positional behavior, or integer rounding semantics.

If repeated historical requantization is not present, stop after the inspection report and do not force a prototype.

### Phase B — Minimal Prototype

Only after passing the decision gate:

- [ ] Requantize only newly generated K/V.
- [ ] Store new K/V in the existing cache after requantization.
- [ ] Reuse historical cached values directly in later attention steps.
- [ ] Preserve tensor layout, head/group structure, concatenation dimension, scale metadata, causal masking, and positional semantics.
- [ ] Avoid unrelated refactoring.
- [ ] Add instrumentation for requantization count/time and KV-cache memory.

### Correctness Tests

For identical prompt and decode inputs, compare original and modified inference at these boundaries:

- [ ] K values after the representation selected for caching.
- [ ] V values after the representation selected for caching.
- [ ] Attention output.
- [ ] Final logits.
- [ ] Generated token.

Prefer bit-exact equality under DeepProve's integer inference semantics. If equality fails, stop performance interpretation and locate the first differing element and exact rounding/requantization boundary responsible.

### Performance Experiment

Use the same configuration recorded for Task 1.

```text
Prefix length: 64
Decode lengths: +1, +4, +8, +16 tokens

Variants:
1. Original DeepProve inference
2. Post-ReQuant KV-cache inference
```

Measure separately for every variant and decode length:

- Total inference latency.
- Per-token decode latency.
- Number of requantization operations.
- Total time spent in requantization.
- Peak RSS.
- KV-cache memory footprint.

Do not include proof generation in the performance claim for this task; this experiment asks only whether post-ReQuant caching benefits inference while preserving quantized inference semantics.

### Task 2 Deliverable

Produce an inspection and experiment report containing:

1. the traced Q/K/V and cache data flow;
2. the implementation decision and supporting evidence;
3. the minimal diff, if the decision gate passes;
4. correctness comparisons;
5. original-versus-modified performance results;
6. limitations and the next research question.

## Out of Scope

Do not modify or design any of the following during these experiments:

- PCS commitments or commitment reuse.
- Proof statements.
- Checkpoint proofs.
- IVC or recursion.
- KV compression or packing.
- Cross-proof KV certification.
- Proof soundness logic.

If post-ReQuant caching is beneficial and bit-exact, cryptographically certifying and reusing cached KV state becomes a separate follow-up project.
