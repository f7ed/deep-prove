use std::{fs::File, io::Write};

use anyhow::{bail, ensure};
use ark_bn254::Bn254;
use clap::{ArgGroup, Parser, ValueEnum, builder::ArgPredicate};
#[cfg(not(feature = "cuda"))]
use dp_crypto::arkyper::HyperKZG;
#[cfg(feature = "cuda")]
use dp_crypto::arkyper::hyperkzg_gpu::HyperKZGGpu;
use itertools::Itertools;
use libc::{RUSAGE_SELF, getrusage, rusage};
use tenstore::GenStore;
use timed_core::Output;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;
use zkml::{
    ProverContext,
    measure::{self, Measure},
    model::{
        KvCacheMode,
        exec_graph::InferenceEngine,
        llm::{Driver, LLMVerifierContext, WithMaxContext},
    },
    parser::{
        file_cache,
        gguf::RawGGUF,
        llm::{
            HFTokenizer,
            models::{gemma3::Gemma3, gpt2::GPT2, llama2::Llama2},
            tokenizer::TokenizerLoader,
        },
        safe::RawSafeTensors,
    },
};

type F = ark_bn254::Fr;
// the hasher type is chosen depending on the feature flag inside the mpcs crate
#[cfg(not(feature = "cuda"))]
type Pcs = HyperKZG<Bn254>;
#[cfg(feature = "cuda")]
type Pcs = HyperKZGGpu<Bn254>;

#[derive(Clone, Debug, ValueEnum)]
#[clap(rename_all = "lower")]
enum Model {
    GPT2,
    Gemma3,
    Llama2,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
#[clap(rename_all = "kebab-case")]
enum CacheMode {
    #[default]
    PreRequant,
    PostRequant,
}

impl From<CacheMode> for KvCacheMode {
    fn from(value: CacheMode) -> Self {
        match value {
            CacheMode::PreRequant => KvCacheMode::PreRequant,
            CacheMode::PostRequant => KvCacheMode::PostRequant,
        }
    }
}

impl std::fmt::Display for CacheMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheMode::PreRequant => f.write_str("pre-requant"),
            CacheMode::PostRequant => f.write_str("post-requant"),
        }
    }
}

impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Model::GPT2 => write!(f, "gpt2"),
            Model::Gemma3 => write!(f, "gemma3"),
            Model::Llama2 => write!(f, "llama2"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    group(ArgGroup::new("weights").args(["gguf", "hf"]).required(true)),
    group(ArgGroup::new("length").args(["sequence", "max_context"]).required(true)),
)]
struct LLMArgs {
    /// gguf file to load. It can be a local path or a URL to download.
    #[arg(short, long)]
    gguf: Option<String>,

    /// Hugging Face model ID. It MUST be a safetensors model for now. If the model isn't present
    /// in the cache, it will be downloaded.
    #[arg(long)]
    hf: Option<String>,

    /// max context length (in tokens)
    #[arg(long)]
    max_context: Option<usize>,

    /// When specifying a sequence, the model will try each difference sequence length, just generating 2 tokens each time
    /// So for seqlen = n, it will start with a prompt of n-2 tokens, and generates two tokens.
    #[arg(long, value_delimiter = ',')]
    sequence: Vec<usize>,

    /// min user input length (in tokens)
    #[arg(
        long,
        requires = "max_context",
        conflicts_with = "sequence",
        default_value_t = 1
    )]
    min_user_len: usize,

    /// model to use
    #[arg(short, long, value_enum)]
    model: Model,

    /// DEPRECATED: output file name that records individual methods
    #[arg(short, long, default_value_t = {"bench-llm-deprecated.csv".to_string()})]
    output: String,

    /// Benchmark csv output file name
    #[arg(
        long,
        default_value_if("distributed", ArgPredicate::IsPresent, Some("bench_distributed.csv")),
        default_value_if("measure_layerwise", ArgPredicate::IsPresent, Some("bench_layerwise.csv")),
        default_value = "bench.csv" // Optional: what to use if NOT distributed
    )]
    bench: String,

    /// How many rayon threads to use
    /// If not provided, will use the number of logical cores
    /// If 0, will use the number of physical cores
    #[arg(long)]
    num_threads: Option<usize>,

    /// Profile distributed execution
    #[arg(long, default_value_t = false)]
    distributed: bool,

    /// Use improved accuracy model quantisation and transforms
    #[arg(long, default_value_t = false)]
    accuracy: bool,

    /// basename of the param files to use for the prover and verifier.
    /// e.g. "--params setup" will search for "setup.pk" and "setup.vk" in the current directory.
    #[arg(long)]
    load_params: Option<String>,

    /// Basename of the prover and verifier context files to save.
    /// e.g. "--save_params setup" will save the prover and verifier context to "setup.pk" and "setup.vk" in the current directory.
    #[arg(long)]
    save_params: Option<String>,

    #[arg(long)]
    memory: bool,

    #[arg(long, requires = "distributed")]
    num_chunks: Option<usize>,

    /// Measure layer-wise metrics
    #[arg(long, default_value_t = false)]
    measure_layerwise: bool,

    /// Run only prompt prefill plus autoregressive decode, without trace
    /// generation, proving, or verification.
    #[arg(long, default_value_t = false, requires = "max_context")]
    decode_only: bool,

    /// Select the inference-time K/V cache representation.
    #[arg(long, value_enum, default_value_t = CacheMode::PreRequant)]
    kv_cache_mode: CacheMode,

    /// Run the standard baseline inference path and write its measurements,
    /// but skip proof generation and verification.
    #[arg(long, default_value_t = false, conflicts_with = "decode_only")]
    inference_only: bool,

    /// Bit-exactly compare every post-ReQuant cached K/V tensor against a
    /// shadow execution of the original full-history ReQuant path.
    #[arg(long, default_value_t = false, requires = "decode_only")]
    verify_kv_cache: bool,

    /// Run Original and post-ReQuant modes on the same prompt and compare K/V,
    /// attention outputs, final logits, and generated tokens exactly.
    #[arg(
        long,
        default_value_t = false,
        requires = "decode_only",
        conflicts_with = "verify_kv_cache"
    )]
    compare_kv_cache: bool,
}

const HEADER_MODEL: &str = "model_name";
const HEADER_MODEL_QUANT: &str = "quantization_time";
const HEADER_CONTEXT_GENERATION: &str = "context_generation_time";
const HEADER_MAX_CONTEXT: &str = "max_context";
const HEADER_NUM_THREADS: &str = "num_threads";
const HEADER_MIN_USER_LEN: &str = "min_user_len";
const HEADER_INFERENCE_TIME: &str = "inference_time";
const HEADER_PROOF_SIZE: &str = "proof_size";

const HEADER_CACHE_MODE: &str = "kv_cache_mode";
const HEADER_DECODE_TOTAL_SECONDS: &str = "decode_total_seconds";
const HEADER_PREFILL_TIME: &str = "prefill_seconds";
const HEADER_INCREMENTAL_DECODE_TIME: &str = "incremental_decode_seconds";
const HEADER_MEAN_INCREMENTAL_DECODE_TIME: &str = "mean_incremental_decode_seconds";
const HEADER_DECODE_TOKENS: &str = "decode_tokens";
const HEADER_KV_REQUANT_CALLS: &str = "kv_requant_calls";
const HEADER_KV_REQUANT_ELEMENTS: &str = "kv_requant_elements";
const HEADER_KV_REQUANT_TIME: &str = "kv_requant_seconds";
const HEADER_KV_CACHE_BYTES: &str = "peak_kv_cache_bytes";
const HEADER_GENERATED_TOKENS: &str = "generated_tokens";
const HEADER_VERIFIED_KV_TENSORS: &str = "verified_kv_tensors";

fn main() -> anyhow::Result<()> {
    let subscriber = tracing_subscriber::fmt::Subscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("Failed to set global subscriber");
    timed_core::set_output(Output::CSV("bench-llm.csv".to_string()));

    let args = LLMArgs::parse();

    // either its spceified and if 0 it's the physical cores otherwise what is specified but no more than the logical cores
    let num_threads = if let Some(nt) = args.num_threads {
        if nt == 0 {
            num_cpus::get_physical()
        } else {
            nt.min(num_cpus::get())
        }
    } else {
        num_cpus::get()
    };
    info!("Using {} threads", num_threads);
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    let (max_context, sequence) = if let Some(max_context) = args.max_context {
        (max_context, vec![(args.min_user_len, max_context)])
    } else {
        let mut sequence = args.sequence.clone();
        sequence.sort();
        (
            *sequence.last().unwrap(),
            sequence
                .into_iter()
                .map(|s| (s.saturating_sub(2), s))
                .collect(),
        )
    };

    info!(
        "Running with max context {} and user_prompt->max_length: {:?}",
        max_context,
        sequence
            .iter()
            .map(|(s, m)| format!("({}->{})", s, m))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let (driver, tokeniser) = parse_model(&args, max_context)?;

    let mut premeasure = Measure::new()
        .with(HEADER_MODEL, &args.model.to_string())
        .with(HEADER_NUM_THREADS, &num_threads.to_string());
    if args.measure_layerwise {
        premeasure = premeasure.enable_layerwise_measures();
    }
    info!("Converting model into provable model...");
    let (mut driver, _metadata) = if !args.accuracy {
        premeasure.r(HEADER_MODEL_QUANT, || driver.into_provable_llm(None))?
    } else {
        match args.model {
            Model::GPT2 => premeasure.r(HEADER_MODEL_QUANT, || {
                driver.into_provable_llm_with_transform::<GPT2>(&tokeniser)
            })?,
            Model::Gemma3 => premeasure.r(HEADER_MODEL_QUANT, || {
                driver.into_provable_llm_with_transform::<Gemma3>(&tokeniser)
            })?,
            Model::Llama2 => premeasure.r(HEADER_MODEL_QUANT, || driver.into_provable_llm(None))?,
        }
    };

    if args.decode_only {
        ensure!(
            args.min_user_len < max_context,
            "--min-user-len must be smaller than --max-context for decode-only runs"
        );
        let decode_tokens = max_context - args.min_user_len;
        measure::set_global(premeasure.clone());
        measure::set(HEADER_MAX_CONTEXT, max_context.to_string());
        measure::set(HEADER_MIN_USER_LEN, args.min_user_len.to_string());
        measure::set(HEADER_DECODE_TOKENS, decode_tokens.to_string());
        measure::set(HEADER_CACHE_MODE, args.kv_cache_mode.to_string());

        let user_tokens = driver.random_sequence(args.min_user_len);
        let input_tensor = driver.tokens_to_decode_tensor(&user_tokens)?;
        let mut store = GenStore::default();

        if args.compare_kv_cache {
            let original = driver.run_decode_benchmark(
                input_tensor.clone(),
                decode_tokens,
                &mut store,
                KvCacheMode::PreRequant,
                false,
                true,
            )?;
            let prototype = driver.run_decode_benchmark(
                input_tensor,
                decode_tokens,
                &mut store,
                KvCacheMode::PostRequant,
                true,
                true,
            )?;
            ensure!(
                original.generated_tokens == prototype.generated_tokens,
                "Generated tokens differ between pre- and post-ReQuant K/V cache modes"
            );
            ensure!(
                original.layer_captures == prototype.layer_captures,
                "Attention output or final logits differ between pre- and post-ReQuant K/V cache modes"
            );

            measure::set(
                HEADER_GENERATED_TOKENS,
                prototype.generated_tokens.iter().join(" "),
            );
            measure::set(HEADER_CACHE_MODE, "pre-requant-vs-post-requant");
            measure::set(
                HEADER_VERIFIED_KV_TENSORS,
                prototype.kv_cache_metrics.verified_kv_tensors,
            );
            measure::set(
                "verified_attention_and_logits",
                prototype.layer_captures.len(),
            );
            measure::set("kv_cache_correctness_equal", true);
            info!(
                "K/V cache correctness comparison passed: {} K/V tensors and {} attention/logit captures were bit-exact; generated={:?}",
                prototype.kv_cache_metrics.verified_kv_tensors,
                prototype.layer_captures.len(),
                prototype.generated_tokens,
            );
            measure::to_csv(&args.bench)?;
            return Ok(());
        }

        let result = measure::r(HEADER_INFERENCE_TIME, || {
            driver.run_decode_benchmark(
                input_tensor,
                decode_tokens,
                &mut store,
                args.kv_cache_mode.into(),
                args.verify_kv_cache,
                false,
            )
        })?;

        let incremental_time = result.incremental_decode_time();
        let mean_incremental_time = result
            .mean_incremental_decode_time()
            .map(|duration| duration.as_secs_f64())
            .unwrap_or_default();
        measure::set(HEADER_DECODE_TOTAL_SECONDS, result.total_time.as_secs_f64());
        measure::set(HEADER_PREFILL_TIME, result.prefill_time.as_secs_f64());
        measure::set(
            HEADER_INCREMENTAL_DECODE_TIME,
            incremental_time.as_secs_f64(),
        );
        measure::set(HEADER_MEAN_INCREMENTAL_DECODE_TIME, mean_incremental_time);
        measure::set(
            HEADER_KV_REQUANT_CALLS,
            result.kv_cache_metrics.kv_requant_calls,
        );
        measure::set(
            HEADER_KV_REQUANT_ELEMENTS,
            result.kv_cache_metrics.kv_requant_elements,
        );
        measure::set(
            HEADER_KV_REQUANT_TIME,
            result.kv_cache_metrics.kv_requant_time.as_secs_f64(),
        );
        measure::set(
            HEADER_KV_CACHE_BYTES,
            result.kv_cache_metrics.peak_kv_cache_bytes(),
        );
        measure::set(
            HEADER_GENERATED_TOKENS,
            result.generated_tokens.iter().join(" "),
        );
        measure::set(
            HEADER_VERIFIED_KV_TENSORS,
            result.kv_cache_metrics.verified_kv_tensors,
        );
        measure::set("decode_only_peak_rss", peak_rss_bytes());

        info!(
            "Decode-only result: mode={}, prompt={}, decode={}, total={:.6}s, prefill={:.6}s, incremental={:.6}s, K/V ReQuant elements={}, K/V ReQuant time={:.6}s, generated={:?}",
            args.kv_cache_mode,
            args.min_user_len,
            decode_tokens,
            result.total_time.as_secs_f64(),
            result.prefill_time.as_secs_f64(),
            incremental_time.as_secs_f64(),
            result.kv_cache_metrics.kv_requant_elements,
            result.kv_cache_metrics.kv_requant_time.as_secs_f64(),
            result.generated_tokens,
        );
        measure::to_csv(&args.bench)?;
        return Ok(());
    }

    ensure!(
        !args.distributed || matches!(args.kv_cache_mode, CacheMode::PreRequant),
        "Post-ReQuant K/V cache mode is not implemented for distributed inference"
    );

    let (prover_ctx, mut verifier_ctx) = if let Some(ref params) = args.load_params {
        info!("Loading proving contexts from {params}.pk and {params}.vk...");
        let prover_ctx = bincode::serde::decode_from_slice(
            &std::fs::read(format!("{params}.pk"))?,
            bincode::config::standard(),
        )?
        .0;
        let verifier_ctx = bincode::serde::decode_from_slice(
            &std::fs::read(format!("{params}.vk"))?,
            bincode::config::standard(),
        )?
        .0;
        // set to 0 since we're not spending any time generating it
        premeasure.r(HEADER_CONTEXT_GENERATION, || 0);
        (prover_ctx, verifier_ctx)
    } else {
        info!("Generating proving contexts...");
        let (prover_ctx, verifier_ctx): (ProverContext<F, Pcs>, LLMVerifierContext<F, Pcs>) =
            premeasure.r(HEADER_CONTEXT_GENERATION, || driver.context())?;
        (prover_ctx, verifier_ctx)
    };

    if let Some(ref save_params) = args.save_params {
        if args.load_params.is_some() {
            bail!("Cannot save parameters if loading parameters is also specified");
        }
        info!("Saving proving contexts to {save_params}.pk and {save_params}.vk...");
        let mut io = File::create(format!("{save_params}.pk"))?;
        io.write_all(&bincode::serde::encode_to_vec(
            &prover_ctx,
            bincode::config::standard(),
        )?)?;
        let mut io = File::create(format!("{save_params}.vk"))?;
        io.write_all(&bincode::serde::encode_to_vec(
            &verifier_ctx,
            bincode::config::standard(),
        )?)?;
    }

    for (user_prompt, max_ctx) in sequence {
        // make a new measure for each trial, but always keep the initial measurements for each sample
        measure::set_global(premeasure.clone());

        measure::set(HEADER_MAX_CONTEXT, max_ctx.to_string());
        measure::set(HEADER_MIN_USER_LEN, user_prompt.to_string());
        measure::set(HEADER_CACHE_MODE, args.kv_cache_mode.to_string());

        driver.with_max_context(max_ctx);
        let user_tokens = driver.random_sequence(user_prompt);
        let input_tensor = driver.tokens_to_tensor(&user_tokens)?;
        let (trace, chunks) = if args.distributed {
            let (chunks, split_node_info) = prover_ctx.split_in_chunks(
                args.num_chunks,
                driver.chunking_strategy(&input_tensor, &prover_ctx)?,
            )?;
            let engine = InferenceEngine::LLM(driver);
            let trace = measure::r(HEADER_INFERENCE_TIME, || {
                info!("Running inference...");
                engine.run(
                    vec![input_tensor],
                    &mut GenStore::default(),
                    &split_node_info,
                )
            })?;
            driver = match engine {
                InferenceEngine::LLM(driver) => driver,
                _ => bail!("Expected LLM engine"),
            };
            (trace, Some(chunks))
        } else {
            (
                measure::r(HEADER_INFERENCE_TIME, || {
                    info!("Running inference...");
                    if matches!(args.kv_cache_mode, CacheMode::PostRequant) {
                        driver.run_elements_with_kv_cache(
                            input_tensor,
                            &mut GenStore::default(),
                            KvCacheMode::PostRequant,
                        )
                    } else {
                        driver.run_elements(input_tensor, &mut GenStore::default())
                    }
                })?,
                None,
            )
        };

        if args.inference_only {
            measure::to_csv(&args.bench)?;
            continue;
        }

        if args.memory {
            info!(
                "Running memory profiler - will save into {:?}",
                std::env::var("FLAMEGRAPH")
            );
            utils::track::flame_graph_enable();
        }
        let (proof, io) = if args.distributed {
            info!("Running distributed proving locally...");
            distributed::run_distributed(trace, &driver, &prover_ctx, chunks.unwrap())?
        } else {
            let peak_rss = peak_rss_bytes();
            info!("Running proving locally...");
            let (proof, io) = driver.prove(&prover_ctx, trace)?;
            let new_peak_rss = peak_rss_bytes();
            if new_peak_rss == peak_rss {
                warn!(
                    "Cannot reliably measure peak memory consumption during proving, setting upper bound"
                );
            }
            // new_peak_rss is the peak memory consumption during proving
            measure::set(
                "prove_full_memory_peak",
                (new_peak_rss / 1024 / 1024).to_string(),
            );

            (proof, io)
        };
        if args.memory {
            utils::track::flame_graph();
        }

        let proof_size = rmp_serde::to_vec(&proof)?.len();
        measure::set(HEADER_PROOF_SIZE, proof_size);

        verifier_ctx = verifier_ctx.with_max_context(max_ctx);
        info!("Verifying proof...");
        verifier_ctx
            .verify(proof, user_tokens, io)
            .expect("invalid proof");
        if !args.distributed {
            measure::post_process(|metrics| {
                let Ok(proof_time) = metrics.get("prove_full").unwrap().parse::<usize>() else {
                    return;
                };
                let Ok(ctx_length) = metrics.get(HEADER_MAX_CONTEXT).unwrap().parse::<usize>()
                else {
                    return;
                };
                let token_per_second = ctx_length as f64 / (proof_time as f64 / 1000.0);
                metrics.insert("token/sec".to_string(), token_per_second.to_string());
            })?;
        }
        measure::to_csv(&args.bench)?;
    }

    Ok(())
}

fn peak_rss_bytes() -> u64 {
    unsafe {
        let mut r: rusage = std::mem::zeroed();
        getrusage(RUSAGE_SELF, &mut r);

        #[cfg(target_os = "linux")]
        {
            (r.ru_maxrss as u64) * 1024
        }

        #[cfg(any(target_os = "macos", target_os = "freebsd", target_os = "openbsd"))]
        {
            r.ru_maxrss as u64
        }
    }
}

fn parse_model(args: &LLMArgs, max_context: usize) -> anyhow::Result<(Driver<f32>, HFTokenizer)> {
    if let Some(ref gguf) = args.gguf {
        let model_path = file_cache::from_cache(gguf)?;
        match args.model {
            Model::GPT2 => {
                let model_type = GPT2::new();
                let raw_gguf = RawGGUF::new(model_path);
                Ok((
                    Driver::load_from_model(model_type, &raw_gguf, Some(max_context))?,
                    model_type.load_tokenizer(&raw_gguf)?,
                ))
            }
            _ => bail!("Model {:?} not supported for gguf", args.model),
        }
    } else if let Some(ref hf) = args.hf {
        let safe = RawSafeTensors::from_hugging_face_cached(hf)?;
        match args.model {
            Model::GPT2 => {
                let model_type = GPT2::new();
                Ok((
                    Driver::load_from_model(model_type, &safe, Some(max_context))?,
                    model_type.load_tokenizer(&safe)?,
                ))
            }
            Model::Gemma3 => {
                let model_type = Gemma3::new();
                Ok((
                    Driver::load_from_model(model_type, &safe, Some(max_context))?,
                    model_type.load_tokenizer(&safe)?,
                ))
            }
            Model::Llama2 => {
                let model_type = Llama2::new();
                Ok((
                    Driver::load_from_model(model_type, &safe, Some(max_context))?,
                    model_type.load_tokenizer(&safe)?,
                ))
            }
        }
    } else {
        bail!("Either gguf or hf must be provided");
    }
}

mod distributed {
    use std::collections::HashMap;

    use super::*;
    use anyhow::{anyhow, ensure};
    use dp_crypto::arkyper::transcript::blake3::Blake3Transcript;
    use tracing::debug;

    use zkml::{
        Element, IO, Proof, Prover,
        graph::{
            executor::{Executor, ThreadPoolExecutor},
            partition::PartitionScheduler,
            scheduler::ExecGraph,
        },
        iop::{
            chunking::ModelChunk,
            prover_graph::{LocalProverCtx, ProverGraphIO, ProverGraphNode},
        },
        model::Trace,
    };

    pub type T = Blake3Transcript;

    // Type of nodes of the graph to execute
    pub type Node<'a, 'b> = ProverGraphNode<'a, 'b, F, T, Pcs>;

    // Type of execution graph to be partitioned and executed in the workers
    pub type Graph<'a, 'b> = ExecGraph<Node<'a, 'b>, Color>;

    // Color is used to create the partitions, assign different nodes to different workers.
    // It can be usize or any other type such as IP address etc.
    pub type Color = usize;

    /// What a partition scheduler outputs
    pub type PartitionOutput<'a, 'b> = zkml::graph::partition::PartitionOutput<Node<'a, 'b>, Color>;

    const CHUNK_OUTPUT_SIZE: &str = "chunk_output_size";
    const CHUNK_INPUT_SIZE: &str = "chunk_input_size";

    fn run_next_partition<'a, 'b, E: Executor<Node<'a, 'b>, Color>>(
        schedulers: &mut HashMap<Color, PartitionScheduler<Node<'a, 'b>, Color, E>>,
    ) -> anyhow::Result<Option<PartitionOutput<'a, 'b>>> {
        let mut to_be_sent_outputs = Vec::new();
        let mut final_output = None;
        let mut done_schedulers = Vec::new();
        for (color, scheduler) in schedulers.iter_mut() {
            let outputs = scheduler.try_run_partition()?;
            for out in outputs {
                if let Some(to_node) = out.to {
                    let serialized_output = rmp_serde::to_vec(&out)?;
                    // ToDo: measure serialized_output
                    if to_node == 0 {
                        // this is data sent to the coordinator, so we add it to the set of data sent by workers
                        // to the coordinator
                        measure::accumulate_key(
                            CHUNK_OUTPUT_SIZE,
                            serialized_output.len(),
                            |a, b| a + b,
                        )?
                    } else if *color == 0 {
                        // this is data sent by the coordinator, so we add it to the set of data sent to the workers
                        measure::accumulate_key(
                            CHUNK_INPUT_SIZE,
                            serialized_output.len(),
                            |a, b| a + b,
                        )?
                    } else {
                        unreachable!(
                            "Data is either sent to coordinator or received by coordinator"
                        )
                    };
                    debug!("Node {} sending output to node {}", color, to_node);
                    to_be_sent_outputs.push((to_node, out));
                } else {
                    // we found the final output
                    final_output = Some(out);
                }
                if scheduler.is_done() {
                    done_schedulers.push(*color);
                }
            }
        }

        for (dest_color, out) in to_be_sent_outputs {
            schedulers
                .get_mut(&dest_color)
                .ok_or(anyhow!("Scheduler not found for color {dest_color}"))?
                .set_child_partition_output(out)?
        }

        for color in done_schedulers {
            schedulers.remove(&color);
        }

        Ok(final_output)
    }

    #[allow(clippy::type_complexity)]
    pub(super) fn run_distributed(
        full_trace: Trace<Element>,
        driver: &Driver<Element>,
        prover_ctx: &ProverContext<F, Pcs>,
        chunks: Vec<ModelChunk>,
    ) -> anyhow::Result<(Proof<F, Pcs>, IO<F>)> {
        let io = full_trace.to_verifier_io()?;

        let graph: Graph = Prover::build_execution_graph(chunks)?;

        let inputs = Prover::graph_inputs(full_trace, &graph)?;

        ensure!(
            inputs.len() == 1,
            "Expected exactly one input node (coordinator split)"
        );

        let flat_inputs =
            inputs
                .into_iter()
                .fold(Vec::new(), |mut ios, (node_input, chunk_prover_io)| {
                    ios.push((node_input.node_id(), chunk_prover_io));
                    ios
                });
        let partitions = graph.partition_by_color(flat_inputs)?;

        let mut schedulers = partitions
            .into_iter()
            .map(|(color, partitions)| {
                let ctx = LocalProverCtx::new(prover_ctx, &driver.model);
                Ok((
                    color,
                    PartitionScheduler::<_, _, ThreadPoolExecutor>::new(partitions, ctx, ())?,
                ))
            })
            .collect::<anyhow::Result<HashMap<_, _>>>()?;

        let mut final_outputs = Vec::new();
        let peak_rss = peak_rss_bytes();
        while !schedulers.is_empty() {
            if let Some(final_output) = run_next_partition(&mut schedulers)? {
                final_outputs.push(final_output.output)
            };
        }
        let new_peak_rss = peak_rss_bytes();
        if new_peak_rss == peak_rss {
            warn!(
                "Cannot reliably measure peak memory consumption during proving, setting upper bound"
            );
        }
        measure::set(
            "prove_full_memory_peak",
            (new_peak_rss / 1024 / 1024).to_string(),
        );

        // Creates channels pairs to communicate with all other nodes
        ensure!(
            final_outputs.len() == 1,
            "Expected 1 outputs for the graph, {} outputs received",
            final_outputs.len()
        );
        let proof = match final_outputs.pop().unwrap() {
            ProverGraphIO::FinalProof(proof) => proof,
            _ => bail!("Invalid output type found after execution of ProverGraph"),
        };
        Ok((proof, io))
    }
}
