# DeepProve GPT-2 Local Baseline

Date: 2026-08-23 (America/New_York)

## Status

Sequence lengths 32, 64, and 128 completed successfully, including quantized inference, proof generation, and verification.

This is not a pristine-upstream result. It uses commit `9d1a53e2ef49ffa2c902b8689cd3c58057a4e662` plus a minimal local SafeTensors loader fix described below. The fix changes only the commitment key assigned to GPT-2's tied final-projection weight; it does not change its values, quantization algorithm, inference semantics, or proof protocol.

## Commands

```bash
cargo build --release -p zkml --bin bench-llm

RUST_LOG=info ./target/release/bench-llm \
  --model gpt2 \
  --hf openai-community/gpt2 \
  --sequence 32 \
  --bench baseline-seq32.csv

RUST_LOG=info ./target/release/bench-llm \
  --model gpt2 \
  --hf openai-community/gpt2 \
  --sequence 64 \
  --bench exp-results/baseline-seq64.csv

RUST_LOG=info ./target/release/bench-llm \
  --model gpt2 \
  --hf openai-community/gpt2 \
  --sequence 128 \
  --bench exp-results/baseline-seq128.csv
```

`--sequence N` runs an `N-2`-token random prompt and generates 2 additional tokens. The three runs therefore used 30+2, 62+2, and 126+2 tokens, respectively.

## Result

| Sequence | Prompt + generated | Quantization | Context generation | Inference | Proof generation | Verification | Peak RSS | Proof size | Throughput |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 30 + 2 | 1.465 s | 15.731 s | 11.887 s | 52.274 s | 1.291 s | 11,046 MiB | 9,235,306 B (8.81 MiB) | 0.6122 tokens/s |
| 64 | 62 + 2 | 1.736 s | 15.367 s | 20.784 s | 80.785 s | 1.443 s | 8,510 MiB | 10,251,087 B (9.78 MiB) | 0.7922 tokens/s |
| 128 | 126 + 2 | 2.210 s | 14.565 s | 38.459 s | 131.389 s | 1.538 s | 8,096 MiB | 11,292,990 B (10.77 MiB) | 0.9742 tokens/s |

All three processes configured 14 Rayon threads. During proving, sumcheck reported that 14 is not a power of two and reduced its own worker count to 8.

The non-monotonic peak RSS values are single-run observations and should not be interpreted as memory improvements without repeated trials. The metric is the process peak RSS reported after proving, rather than an isolated incremental allocation measurement.

Raw aggregate outputs:

- `baseline-seq32.csv`
- `exp-results/baseline-seq64.csv`
- `exp-results/baseline-seq128.csv`

## Proof-Size Comparison with Published Numbers

The DeepProve paper reports HyperKZG GPT-2 proofs of 8.98 MiB at sequence 64 and 9.89 MiB at sequence 128. This local run produced 9.78 MiB and 10.77 MiB, approximately 0.80 MiB and 0.88 MiB larger. The current repository README contains another reference set, 7.95 MiB and 8.82 MiB, which also does not match either the paper or the current executable.

These differences are not caused by CPU speed, RAM capacity, or proving time. For a fixed protocol, proof structure, and encoder, those factors do not determine serialized proof length. The confirmed reasons that these numbers are not directly byte-for-byte comparable are:

1. The paper, repository README, and this experiment are not pinned to one identical code artifact. This experiment uses public commit `9d1a53e...`; the repository history includes large changes to HyperKZG, chunking, claims, lookup proofs, and proof structs, while the paper's table does not identify an exact public commit hash.
2. The current `bench-llm` defines proof size as `rmp_serde::to_vec(&proof).len()`. The paper reports MiB but does not specify this exact Rust serializer and schema in its evaluation methodology. Production DeepProve paths now use Postcard, while the benchmark still uses MessagePack, so a proof-size claim must always state the encoder.
3. This run uses the default public `bench-llm` quantization path and a local SafeTensors commitment-key compatibility fix. The paper describes 12-bit quantization and accuracy techniques, but its proof-size table does not provide an exact graph/configuration hash or the equivalent current CLI flags.

The local key fix can change proof metadata because the independently quantized final projection receives its own polynomial identity. It does not change inference values, but its isolated byte contribution has not yet been measured. Therefore the current result is a valid local baseline, but it should not be presented as a reproduction of the paper's proof-size number.

## Configuration

| Item | Value |
|---|---|
| Git commit | `9d1a53e2ef49ffa2c902b8689cd3c58057a4e662` |
| Local source state | Commit above plus SafeTensors key fix |
| Model | `openai-community/gpt2` SafeTensors |
| Quantization bit width | 12 bits (default `ZKML_BIT_LEN`) |
| Backend | CPU / Burn ndarray |
| PCS | `HyperKZG<Bn254>` |
| Build profile | Cargo release, optimized with debuginfo |
| Rust | `rustc 1.95.0-nightly (474276961 2026-01-26)` |
| Machine | MacBook Pro, Apple M4 Pro |
| CPU cores | 14 (10 performance + 4 efficiency) |
| RAM | 24 GB |

No `cuda`, `wgpu`, `accuracy`, `distributed`, or saved-parameter option was enabled.

## Reproducibility Issues

### Git LFS model files

Git LFS was not installed locally. The tracked GPT-2 `config.json`, `tokenizer.json`, and `model.safetensors` files were therefore LFS pointer text rather than model data. This initially caused:

```text
Error: parsing config.json
Caused by: expected value at line 1 column 1
```

The three official Hugging Face files were downloaded manually. Their sizes and SHA-256 digests matched the values declared by the repository's LFS pointers. Without Git LFS installed, Git reports the hydrated model files as modified.

### SafeTensors tied-weight commitment-key collision

GPT-2 ties the input embedding and final language-model projection to the same floating-point weight matrix. DeepProve quantizes the embedding matrix with its special wider embedding range, while the final projection uses the regular weight quantization range. They therefore produce different integer tensors and different multilinear extensions.

The SafeTensors loader copied the embedding tensor into the final projection while retaining the same storage key, `wte.weight`. During proving-context construction, both tensors were consequently registered under one `CommitmentId`. DeepProve correctly rejected this when it observed two different MLEs for that ID:

```text
Found different MLE for polynomial wte.weight
```

The GGUF loader already avoids this collision by assigning the projection a separate `<embedding-key>_final_proj` key. The same key separation was added to the GPT-2 SafeTensors loader. This preserves GPT-2's tied floating-point values while correctly representing the two independently quantized tensors as separate committed polynomials.
