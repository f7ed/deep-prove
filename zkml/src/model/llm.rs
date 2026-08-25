//! A LLM driver runs the model on a given input and can inspect the output of each layer
//! and the output of the model. It can decide to re-run the model on a different input,
//! to modify the inference trace, to modify the model, etc.
//! The main usage of a driver for now is to run the LLM forward loop until a specific token or
//! the maximum context length is reached. It will also be used to prepend a system model correctly.

use crate::{
    iop::chunking::LLMChunkingStrategy,
    parser::llm::{HFTokenizer, metadata::LLMMetadata},
    quantization::{LLMInferenceObserver, llm_quant::FPTransformModel},
};
use std::{
    fmt::Display,
    ops::Deref,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
};

use crate::{
    Element, IO, Proof, Prover, ProverContext, ScalingStrategy, Shape, Tensor,
    graph::executor::{Executor, SequentialExecutor},
    iop::{
        chunking::{ChunkingStrategy, DefaultChunkingStrategy},
        context::VerifierContext,
        prover_graph::ProverGraphNode,
    },
    layers::{Layer, provable::Evaluate},
    model::{
        BaseRunner, HandleLifetimeRunner, LayerRunner, Model, RunInput, Trace, TrackerRunner,
        tensor_to_handles, trace::SplittedNodesInfo, wrapped_tensor_to_handles,
    },
    number::Number,
    padding::PaddingMode,
    parser::{
        PipelineConfig, default_pipeline_config,
        llm::{Token, models::LLMModelLoader},
        to_quantized,
    },
    quantization::{InferenceObserver, InferenceTracker, ModelMetadata, ToElement},
    tensor::{TensorHandle, TensorTypeParam, WrappedTensor},
    verify,
};
use anyhow::{Context, anyhow, ensure};
use ark_ff::PrimeField;
use ark_std::rand::Rng;
use dp_crypto::arkyper::{CommitmentScheme, transcript::blake3::Blake3Transcript};
use itertools::Itertools;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tenstore::GenStore;
use tracing::{debug, info, info_span};

#[cfg(test)]
use crate::model::SanityCheckRunner;

pub trait WithMaxContext {
    fn with_max_context(self, max_context: usize) -> Self;
}

/// The main struct responsible to verify an LLM proof.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub struct LLMVerifierContext<F, PCS>
where
    F: PrimeField,
    PCS: CommitmentScheme,
{
    pub verifier_ctx: VerifierContext<F, PCS>,
    pub md: LLMMetadata,
    pub max_context: Option<usize>,
}

impl<F, PCS> WithMaxContext for LLMVerifierContext<F, PCS>
where
    F: PrimeField,
    PCS: CommitmentScheme,
{
    /// Sets the `max_context` window for this context.
    fn with_max_context(mut self, max_context: usize) -> Self {
        self.max_context = Some(max_context);
        self
    }
}

impl<'a, F, PCS> WithMaxContext for (ProverContext<'a, F, PCS>, LLMVerifierContext<F, PCS>)
where
    F: PrimeField,
    PCS: CommitmentScheme,
{
    /// Sets the `max_context` window for this context.
    fn with_max_context(self, max_context: usize) -> Self {
        (self.0, self.1.with_max_context(max_context))
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub struct LLMProof<F, PCS>
where
    F: PrimeField,
    PCS: CommitmentScheme,
{
    /// The inference proof.
    pub proof: Proof<F, PCS>,

    /// Note the IO contains the _full_ input, e.g. the user input + the generated tokens
    pub io: IO<F>,
}

/// The main struct responsible for generating the trace and the proof related
/// to LLM proving. This requires a wrapper on top of the model to drive the
/// auto regressive loop correctly.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound(serialize = "N: Serialize", deserialize = "N: DeserializeOwned"))]
pub struct Driver<N>
where
    N: TensorTypeParam,
{
    pub model: Model<N>,
    pub md: LLMMetadata,
    pub(crate) max_context: Option<usize>,
    pub(crate) padding_mode: PaddingMode,
}

impl<N: TensorTypeParam> Driver<N> {
    /// Returns the vocabulary size of the model
    pub fn vocab_size(&self) -> usize {
        self.md.vocab_size()
    }
}

impl Driver<f32> {
    /// Load and returns the raw model in float precision.
    ///
    /// Loads a model from a gguf, safetensors, or json external file.
    ///
    /// NOTE: the max_context is only there to hack around the creation of Rope
    /// to avoid loading the full matrix if we don't need it. That should be
    /// removed if when we remove this hack.
    pub fn load_from_model<DataFormat: Display, M: LLMModelLoader<DataFormat>>(
        mut model_type: M,
        data_format: &DataFormat,
        max_context: Option<usize>,
    ) -> anyhow::Result<Self> {
        info!(
            "LLM Driver: Loading model {} from {}",
            model_type.model_name(),
            data_format
        );
        if let Some(max_context) = max_context {
            model_type = model_type.with_max_context_length(max_context);
        }
        let (model, md) = model_type.parse(data_format)?;
        Ok(Self {
            model,
            md,
            max_context,
            padding_mode: PaddingMode::NoPadding,
        })
    }

    /// Quantizes and pads this model.
    ///
    /// The result can be serialized and deserialized at will to serve
    /// inference+proving for this model.
    pub fn into_provable_llm<'a>(
        self,
        mut pipeline_config: Option<PipelineConfig<'a, InferenceObserver>>,
    ) -> anyhow::Result<(Driver<Element>, ModelMetadata)> {
        let numel = self.max_context.unwrap_or(self.md.context_length());
        let n_inputs = 1;
        let representative_inputs = (0..n_inputs)
            .map(|_| {
                self.random_sequence(numel)
                    .into_iter()
                    .map(|t| t.as_number::<f32>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let quantization_strategy =
            InferenceObserver::new_with_representative_input(vec![representative_inputs]);
        let mut conf = pipeline_config.take().unwrap_or_else(|| {
            // override the shapes to adhere to the expected input shape of the representative inputs
            default_pipeline_config()
        });

        if conf.input_shapes.is_none() {
            conf = conf.with_input_shapes(vec![Shape::from(vec![numel])]);
        }
        if conf.quant_strategy.is_none() {
            conf = conf.with_strategy(quantization_strategy);
        }
        
        let (mut quantized_model, metadata) = to_quantized(self.model, conf)?;
        // just set to one because we run one token after another to derive the full trace.
        quantized_model.input_shapes = vec![Shape::from(vec![1])];
        Ok((
            Driver {
                model: quantized_model,
                md: self.md,
                max_context: self.max_context,
                padding_mode: PaddingMode::Padding,
            },
            metadata,
        ))
    }

    /// Quantizes and pads this model using a specific pipeline configuration.
    ///
    /// The result can be serialized and deserialized at will to serve
    /// inference+proving for this model.
    pub fn into_provable_llm_with_strategy<'a, S: ScalingStrategy>(
        self,
        mut pipeline_config: PipelineConfig<'a, S>,
    ) -> anyhow::Result<(Driver<Element>, ModelMetadata)> {
        let numel = self.max_context.unwrap_or(self.md.context_length());

        if pipeline_config.input_shapes.is_none() {
            pipeline_config = pipeline_config.with_input_shapes(vec![Shape::from(vec![numel])]);
        }

        let (mut quantized_model, metadata) = to_quantized(self.model, pipeline_config)?;
        // just set to one because we run one token after another to derive the full trace.
        quantized_model.input_shapes = vec![Shape::from(vec![1])];
        Ok((
            Driver {
                model: quantized_model,
                md: self.md,
                max_context: self.max_context,
                padding_mode: PaddingMode::Padding,
            },
            metadata,
        ))
    }

    pub fn into_provable_llm_with_transform<M: FPTransformModel>(
        self,
        tokeniser: &HFTokenizer,
    ) -> anyhow::Result<(Driver<Element>, ModelMetadata)> {
        let max_context = self.max_context;
        let md = self.md.clone();

        let llm_inference_observer =
            LLMInferenceObserver::new_with_tokeniser(tokeniser.clone(), self.md.clone(), None);

        let representative_input_length = if let Some(length) = max_context {
            length
        } else {
            llm_inference_observer.get_representative_input_length()
        };

        let pipeline_config = PipelineConfig::<LLMInferenceObserver>::new()
            .with_input_shapes(vec![Shape::from(vec![representative_input_length])])
            .with_strategy(llm_inference_observer);

        let adapted_driver = M::adapt_model(self)?;
        let (mut quantized_model, metadata) = to_quantized(adapted_driver.model, pipeline_config)?;
        // just set to one because we run one token after another to derive the full trace.
        quantized_model.input_shapes = vec![Shape::from(vec![1])];
        Ok((
            Driver {
                model: quantized_model,
                md,
                max_context,
                padding_mode: PaddingMode::Padding,
            },
            metadata,
        ))
    }
}

impl<N> Driver<N>
where
    N: TensorTypeParam,
    Layer<N>: Evaluate<N>,
{
    /// Returns the current `max_context` window, if set.
    pub fn max_context(&self) -> Option<usize> {
        self.max_context
    }

    /// Sets the `max_context` window for this driver.
    pub fn with_max_context(&mut self, max_context: usize) {
        self.max_context = Some(max_context);
    }

    /// Returns a sequence of `len` random tokens.
    pub fn random_sequence(&self, len: usize) -> Vec<Token> {
        let mut rng = crate::rng_from_env_or_random();
        (0..len)
            .map(|_| Token::from(rng.gen_range(0..self.md.vocab_size())))
            .collect()
    }

    /// Converts a slice of [Token]s into a [Tensor] to be used with this driver.
    ///
    /// # Errors
    ///
    /// If the number of tokens exceeds the current configured `context_length`.
    pub fn tokens_to_tensor(&self, input: &[Token]) -> anyhow::Result<Tensor<N>> {
        ensure!(
            input.len() < self.md.config.context_length - 1,
            "Input sequence length must be less than the context length",
        );
        let input_tokens = input
            .iter()
            .map(|t| t.as_tensor_type_param::<N>())
            .collect::<Vec<_>>();
        Tensor::new(Shape::from([input_tokens.len()]), input_tokens)
    }

    /// Returns the strategy employed to chunk the LLM in case of distributed proving.
    pub fn chunking_strategy<F: PrimeField, PCS: CommitmentScheme<Field = F>>(
        &self,
        input_tensor: &Tensor<N>,
        ctx: &ProverContext<F, PCS>,
    ) -> anyhow::Result<LLMChunkingStrategy> {
        // check that the input tensor size is smaller than max context
        let max_window = self.max_context.unwrap_or(self.md.context_length());
        let num_input_tokens = input_tensor.data().len();
        ensure!(
            num_input_tokens <= max_window,
            "Input sequence length greater than the context length",
        );
        Ok(LLMChunkingStrategy::new(
            max_window, /* we always use the max context length to be safe, as before running inference we don't know yet the actual input length for the prover */
            &ctx.model_ctx,
        ))
    }

    /// Performs a single run of the model, returning the produced trace.
    pub fn run(&self, inputs: Vec<Tensor<N>>, store: &mut GenStore) -> anyhow::Result<Trace<N>> {
        let result = self.model.run(inputs, store);
        self.model.reset();
        result
    }

    /// Run a single iteration of the model using the provided `runner`.
    pub fn run_with_runner<R>(
        &self,
        runner: &mut R,
        inputs: Vec<TensorHandle<N>>,
    ) -> anyhow::Result<()>
    where
        R: LayerRunner<N, ()>,
    {
        let result = self.model.run_with_runner(runner, inputs);
        self.model.reset();
        result
    }

    pub(crate) fn run_elements_with_split_info(
        &self,
        input_tensor: Tensor<N>,
        store: &mut GenStore,
        split_node_info: Option<&SplittedNodesInfo>,
    ) -> anyhow::Result<Trace<N>> {
        let runner = BaseRunner::from(store.clone());
        #[cfg(test)]
        let runner = SanityCheckRunner { inner: runner };

        self.run_elements_with_runner(input_tensor, store, runner, split_node_info)
    }

    /// Generate a sentence.
    ///
    /// Notes:
    /// - The sentence is limited by the EOS token or the context length.
    /// - The returned trace contains the whole sequence.
    /// - The layer runner will be used for each token infernece, but not for
    ///   the trace generation.
    pub fn run_elements(
        &self,
        input_tensor: Tensor<N>,
        store: &mut GenStore,
    ) -> anyhow::Result<Trace<N>> {
        self.run_elements_with_split_info(input_tensor, store, None)
    }

    /// Generate a sentence.
    ///
    /// Notes:
    /// - The sentence is limited by the EOS token or the context length.
    /// - The returned trace contains the whole sequence.
    pub fn run_elements_with_tracker(
        &self,
        input_tensor: Tensor<N>,
        store: &mut GenStore,
        tracker: &mut InferenceTracker,
    ) -> anyhow::Result<Trace<N>> {
        let runner = BaseRunner::from(store.clone());
        let runner = TrackerRunner {
            inner: runner,
            tracker,
        };
        #[cfg(test)]
        let runner = SanityCheckRunner { inner: runner };

        self.run_elements_with_runner(input_tensor, store, runner, None)
    }

    /// Generate a sentence.
    ///
    /// Notes:
    /// - The sentence is limited by the EOS token or the context length.
    /// - The returned trace contains the whole sequence.
    /// - The layer runner will be used for each token inferenece, but not for
    ///   the trace generation.
    fn run_elements_with_runner<I>(
        &self,
        input_tensor: Tensor<N>,
        store: &mut GenStore,
        runner: I,
        split_node_info: Option<&SplittedNodesInfo>,
    ) -> anyhow::Result<Trace<N>>
    where
        HandleLifetimeRunner<I, N>: LayerRunner<N, ()>,
        I: LayerRunner<N, RunInput<N>>,
    {
        // Reset the model
        self.model.reset();

        let input_data = input_tensor.data().to_vec();
        let num_input_tokens = input_data.len();

        let mut runner = HandleLifetimeRunner::new(runner, &self.model.graph);

        // -1 because we at least want to generate ONE token
        ensure!(
            num_input_tokens < self.md.context_length() - 1,
            "Input sequence length must be at least one fewer than the context length",
        );

        let max_window = self.max_context.unwrap_or(self.md.context_length());
        ensure!(
            num_input_tokens <= max_window,
            "max window {} is smaller than prompt token count {}",
            max_window,
            num_input_tokens,
        );

        let is_valid_vocab = input_data
            .iter()
            .all(|t| Number::to_usize(t) < self.md.vocab_size());
        ensure!(
            is_valid_vocab,
            "Input tokens must be less than the vocabulary size",
        );

        // Initialise the full tokens with the input data, generated tokens will be appended.
        let mut full_sentence = input_data;

        // Inference always runs on unpadded tensors. Padding is only needed at the proving layer.
        info!(
            "LLM: running model with initial shape: {:?}",
            input_tensor.shape(),
        );
        let mut input_handles = tensor_to_handles(&[input_tensor], &self.model.graph, store)?;

        // This thread waits for the results
        let (tx, rx) = mpsc::sync_channel::<WrappedTensor<N>>(10);
        let stop = Arc::new(AtomicBool::new(false));
        let eos_token: N = self.md.config.eos_token.as_tensor_type_param();
        let handle = {
            let stop = Arc::clone(&stop);
            thread::spawn(move || -> anyhow::Result<Vec<N>> {
                let mut generated_tokens = Vec::with_capacity(max_window);
                while let Ok(tensor) = rx.recv() {
                    let token = tensor.get_data()[0];
                    if token == eos_token {
                        stop.store(true, Ordering::Relaxed);
                        return Ok(generated_tokens);
                    }
                    generated_tokens.push(token);
                }

                // Remove the last token so the final inference will regenerate
                // a full trace with the correct padding
                generated_tokens.pop();

                Ok(generated_tokens)
            })
        };

        // This loop schedules work to be run, when using a GPU backend this allows for multiple
        // kernel calls to be scheduled to increase the hardware occupancy.
        for unpadded_seq_len in num_input_tokens..=max_window {
            if stop.load(Ordering::Relaxed) {
                info!(
                    "Stopping on iteration {}, an EOS was genarted on some of the previous iteraitons",
                    unpadded_seq_len,
                );
                break;
            }

            info!(
                "Running iteration {} with input tensor {:?}",
                unpadded_seq_len,
                input_handles[0].shape(),
            );

            self.model
                .run_with_runner(&mut runner, input_handles)
                .with_context(|| {
                    format!(
                        "running the {} iteration loop",
                        unpadded_seq_len - num_input_tokens
                    )
                })?;

            let model_outputs = runner.model_outputs(&self.model.graph)?;
            ensure!(
                model_outputs.len() == 1,
                "expected 1 output, got {}",
                model_outputs.len(),
            );

            let output = &model_outputs[0]
                .wrapped_tensor()
                .context("Output handler should have associated tensor data")?;

            // Take the last token before the padding
            let index = if output.shape().num_elements() == 1 {
                WrappedTensor::try_from(vec![0])?
            } else {
                WrappedTensor::try_from(vec![(unpadded_seq_len - 1) as Element])?
            };

            let last_token_tensor = output.deref().clone().flatten_1d().select(0, index)?;

            tx.send(last_token_tensor.clone())
                .context("Results thread is gone")?;

            input_handles =
                wrapped_tensor_to_handles(&[last_token_tensor], &self.model.graph, store)?;
        }

        // Drop the sender channel, signaling to the results thread that no more
        // work will be scheduled
        drop(tx);

        let generated_tokens = handle
            .join()
            .map_err(|err| anyhow!("Results thread panicked. err: {err:?}"))?
            .context("Fetch tensor results failed")?;
        full_sentence.extend(generated_tokens);
        // Inference always runs on unpadded tensors. Padding is handled at the proving layer.
        let tensor = Tensor::new(Shape::new(vec![full_sentence.len()]), full_sentence.clone())?;

        // Reset the model and its caches, otherwise a single token is
        // generated.
        self.model
            .reset_selective(&[crate::layers::transformer::softmax::SOFTMAX_LAYER]);

        info!(
            "Running last iteration (heavy) with {} tokens",
            full_sentence.len(),
        );

        // Use the passed store (worker's remote store) so TensorHandle data
        // is accessible during proving in distributed execution
        let trace = self
            .model
            .run_with_split_nodes_info(vec![tensor], store, split_node_info)?;
        for i in num_input_tokens..full_sentence.len() {
            assert_eq!(
                trace.outputs()[0].tensor().unwrap().data()[i - 1],
                trace.inputs()[0].tensor().unwrap().data()[i],
                "Failed for {i}, input: {:?}, output: {:?}",
                trace.inputs()[0].tensor().unwrap(),
                trace.outputs()[0].tensor().unwrap(),
            );
        }
        let argmax_id = self.md.argmax;
        let vocab_size = self.md.vocab_size();
        // If the number of input tokens is less than the max window size then
        // check that the outputs are correctly shifted.
        if num_input_tokens < max_window {
            let output_data = trace.outputs()[0]
                .tensor()
                .context("Output tensor should have data")?
                .data()
                .to_vec();
            let input_data = trace.inputs()[0]
                .tensor()
                .context("Input tensor should have data")?
                .data()
                .to_vec();
            for i in num_input_tokens..full_sentence.len() {
                if output_data[i - 1] != input_data[i] {
                    let logits_input = trace
                        .get_step(&argmax_id)
                        .ok_or(anyhow!("No logits step present in trace"))?
                        .inputs()[0]
                        .tensor()?;
                    let logits_shape = logits_input.shape();
                    let current_row = logits_input.data()[(i - 1) * logits_shape.dim(-1)
                        ..(i - 1) * logits_shape.dim(-1) + vocab_size]
                        .to_vec();

                    let value_chosen = crate::number::Number::to_usize(&output_data[i - 1]);
                    let input_chosen = crate::number::Number::to_usize(&input_data[i]);
                    let mut row_with_indices = current_row
                        .iter()
                        .cloned()
                        .enumerate()
                        .collect::<Vec<(usize, N)>>();
                    row_with_indices.sort_by(|a, b| b.1.cmp(&a.1));

                    assert_eq!(
                        current_row[value_chosen],
                        current_row[input_chosen],
                        "Logits output chosen {} value {}, input chosen {} value {}, logits row: {},first 10 sorted: {:?}, shape:{:?} input: {:?}, output: {:?}",
                        value_chosen,
                        current_row[value_chosen],
                        input_chosen,
                        current_row[input_chosen],
                        i,
                        &row_with_indices[..10],
                        logits_shape,
                        &input_data[num_input_tokens..],
                        &output_data[num_input_tokens - 1..],
                    );
                }
                assert_eq!(
                    output_data[i - 1],
                    input_data[i],
                    "Failed for {i}, input: {:?}, output: {:?}",
                    &input_data[num_input_tokens..],
                    &output_data[num_input_tokens - 1..],
                );
            }
        }

        for i in num_input_tokens..full_sentence.len() {
            debug_assert_eq!(
                trace.outputs()[0].tensor().unwrap().data()[i - 1],
                trace.inputs()[0].tensor().unwrap().data()[i],
                "Failed for {i}, input: {:?}, output: {:?}",
                trace.inputs()[0].tensor().unwrap(),
                trace.outputs()[0].tensor().unwrap(),
            );
        }

        // Reset the model so it can be reused.
        self.model.reset();

        Ok(trace)
    }
}

impl Driver<Element> {
    /// Compute the set of contexts necessary for all the possible input shapes of the LLM.
    /// It returns a `HashMap` which associates a given context to the maximum polynomial size supported
    /// by that context. The proper context to be used for a given input size `input_len` is found in the
    /// entry `HashMap` returned by this method identified by the key `self.get_max_poly_size_for_input(input_len)`
    pub fn compute_all_contexts<F, PCS>(
        &self,
    ) -> anyhow::Result<(ProverContext<'_, F, PCS>, LLMVerifierContext<F, PCS>)>
    where
        F: PrimeField,
        PCS: CommitmentScheme<Field = F>,
    {
        // compute shapes for all possible input sequence lengths
        let max_input_shapes = vec![Shape::new(vec![
            self.md.context_length().next_power_of_two(),
        ])];
        ensure!(
            max_input_shapes.len() == 1,
            "Expected 1 input shape in LLM model, found {}",
            max_input_shapes.len()
        );
        let max_input_length = max_input_shapes[0].numel();
        ensure!(max_input_length.is_power_of_two());

        let (prover_ctx, verifier_ctx) = self
            .model
            .generate_contexts_for_input_shapes(max_input_shapes)?;
        Ok((
            prover_ctx,
            LLMVerifierContext {
                verifier_ctx,
                md: self.md.clone(),
                max_context: self.max_context,
            },
        ))
    }

    /// Create the prover & verifier for a given model
    pub fn context<F, PCS>(
        &self,
    ) -> anyhow::Result<(ProverContext<'static, F, PCS>, LLMVerifierContext<F, PCS>)>
    where
        F: PrimeField,
        PCS: CommitmentScheme<Field = F>,
    {
        debug!(
            "Generating context for model with {} layers",
            self.model.graph.node_count()
        );
        let (prover_ctx, verifier_ctx) = self.model.generate_contexts()?;
        Ok((
            prover_ctx,
            LLMVerifierContext {
                verifier_ctx,
                md: self.md.clone(),
                // The verifier should put itself the max context here
                max_context: None,
            },
        ))
    }

    fn chunked_prove_local<'a, 'b, S, F, PCS, Ex>(
        &'a self,
        ctx: &'b ProverContext<F, PCS>,
        trace: Trace<Element>,
        num_chunks: Option<usize>,
        chunking_strategy: S,
        executor_config: Ex::Config,
    ) -> anyhow::Result<(Proof<F, PCS>, IO<F>)>
    where
        S: ChunkingStrategy,
        F: PrimeField,
        PCS: CommitmentScheme<Field = F> + 'static,
        Ex: Executor<ProverGraphNode<'b, 'a, F, Blake3Transcript, PCS>, usize>,
    {
        info!("Proving the trace");
        let proof = Prover::<F, Blake3Transcript, PCS>::chunked_prove_local::<_, Ex>(
            ctx,
            trace,
            num_chunks,
            chunking_strategy,
            &self.model,
            executor_config,
        )
        .expect("unable to generate proof");
        info!("Proof generated");
        Ok(proof)
    }

    /// Utility method to run `chunked_prove_local` with a predefined executor
    #[cfg(test)]
    pub(crate) fn chunked_prove_default_executor<S, F, PCS>(
        &self,
        ctx: &ProverContext<F, PCS>,
        trace: Trace<Element>,
        num_chunks: Option<usize>,
        chunking_strategy: S,
    ) -> anyhow::Result<(Proof<F, PCS>, IO<F>)>
    where
        S: ChunkingStrategy,
        F: PrimeField,
        PCS: CommitmentScheme<Field = F> + 'static,
    {
        // NOTE: until https://github.com/Plonky3/Plonky3/pull/999 is fixed, we have
        // to use the sequential executor and not the threadpool executor.
        self.chunked_prove_local::<_, _, _, SequentialExecutor>(
            ctx,
            trace,
            num_chunks,
            chunking_strategy,
            (),
        )
    }

    pub fn prove<F, PCS>(
        &self,
        ctx: &ProverContext<F, PCS>,
        trace: Trace<Element>,
    ) -> anyhow::Result<(Proof<F, PCS>, IO<F>)>
    where
        F: PrimeField,
        PCS: CommitmentScheme<Field = F> + 'static,
    {
        let span = info_span!(
            "zkml_llm_prove",
            max_context = self.max_context.unwrap_or_default()
        );
        let _entered = span.enter();
        let chunking_strategy = DefaultChunkingStrategy::from(&trace);
        self.chunked_prove_local::<_, _, _, SequentialExecutor>(
            ctx,
            trace,
            Some(1),
            chunking_strategy,
            (),
        )
    }
}

impl<F, PCS> LLMVerifierContext<F, PCS>
where
    F: PrimeField,
    PCS: CommitmentScheme<Field = F>,
{
    pub fn verify(
        &self,
        proof: Proof<F, PCS>,
        user_input: Vec<Token>,
        inference_output: IO<F>,
    ) -> anyhow::Result<()>
    where
        PCS::Commitment: PartialEq,
    {
        let span = info_span!(
            "zkml_llm_verify",
            max_context = self.max_context.unwrap_or(self.md.config.context_length),
            input_tokens = inference_output.input.len(),
            output_tensors = inference_output.output.len()
        );
        let _entered = span.enter();
        ensure!(
            inference_output.output.len() == 1,
            "Expected 1 output tensor, found {}",
            inference_output.output.len()
        );
        // 0. check the size of the output
        let output = inference_output.output[0].clone();
        let padded_max_len = output.shape().numel();
        let max_window = self.max_context.unwrap_or(self.md.context_length());

        // in any case, the output needs to be less than the max context length
        ensure!(
            padded_max_len <= max_window.next_power_of_two(),
            "output length is greater than the padded maximum context length"
        );
        // get the actual output length: could be either `max_window`, or when an eos token is found
        let eos_token = self.md.config.eos_token;
        let eos_token_found = output
            .data()
            .iter()
            .take(max_window) // consider only the first max_window tokens
            .skip(user_input.len() - 1) // the first user_input.len() - 1 are not meaningful
            .find_position(|token| token.to_element() as usize == usize::from(eos_token));
        let unpadded_output_len = if let Some(pos) = &eos_token_found {
            pos.0
        } else {
            max_window
        };

        // 1. verify the proof it self
        let prover_input = inference_output.input[0].clone();
        let prover_output = inference_output.output[0].clone();
        verify::<_, Blake3Transcript, _>(&self.verifier_ctx, proof, inference_output)?;
        // 2. verify the sequentiality of the output: from the first newly generated token to the last
        // but without including the padding.
        // output is [seq_len] where []
        let seq_len = user_input.len();
        ensure!(
            prover_input.data()[..seq_len]
                .iter()
                .zip(user_input[..seq_len].iter())
                .all(|(a, b)| a.to_element().to_usize() == b.0),
            "user input not the same"
        );

        #[allow(clippy::needless_range_loop)]
        for i in seq_len - 1..unpadded_output_len - 1 {
            // we check the next input token is the one generated by this "row" of the input
            ensure!(
                prover_input.data()[i + 1] == prover_output.data()[i],
                "next input token is not the one generated by this row pos i {}: {:?} != {:?}",
                i,
                prover_input.data(),
                prover_output.data()
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::{
        Number, init_test_logging,
        iop::{ProverContext, chunking::LLMChunkingStrategy},
        model::{
            llm::{Driver, LLMVerifierContext, WithMaxContext},
            test::F,
        },
        parser::{
            file_cache,
            gguf::{RawGGUF, TensorLoader},
            llm::{
                HFTokenizer, LLMTokenizer, Token,
                models::{
                    gemma3::{Gemma3, tests::GEMMA3_Q8},
                    gpt2::{GPT2, GPT2_Q8_0},
                },
                tokenizer::TokenizerLoader,
            },
            safe::RawSafeTensors,
        },
        rng_from_env_or_random,
        testing::Pcs,
    };
    use anyhow::Context;
    use ark_std::rand::Rng;
    use tenstore::GenStore;
    use tracing::info;

    fn test_llm_driver_generic_prove_gpt2<const DISTRIBUTE_PROVE: bool>() -> anyhow::Result<()> {
        const MAX_CONTEXT: usize = 10;
        init_test_logging("debug");

        let model_path = file_cache::from_cache(GPT2_Q8_0)?;
        let (driver, _metadata) = Driver::load_from_model(
            GPT2::new(),
            &RawGGUF::new(model_path.clone()),
            Some(MAX_CONTEXT),
        )?
        .into_provable_llm(None)?;

        let (prover_ctx, verifier_ctx) = driver.context::<F, Pcs>()?.with_max_context(MAX_CONTEXT);

        driver.model.describe();

        let mut store = GenStore::default();
        let sentence = "The sky is";
        let tokenizer = GPT2::new().load_tokenizer(&RawGGUF::new(model_path))?;
        let user_tokens = tokenizer.tokenize(sentence);
        let input_tensor = driver.tokens_to_tensor(&user_tokens)?;
        let trace = driver.run_elements(input_tensor, &mut store)?;

        // Prove the trace
        let num_provers = if DISTRIBUTE_PROVE {
            Some(rng_from_env_or_random().gen_range(1..6))
        } else {
            Some(1) // equivalent to sequential proving
        };
        let chunking_strategy = LLMChunkingStrategy::from((&trace, &prover_ctx.model_ctx));
        let (proof, io) = driver.chunked_prove_default_executor(
            &prover_ctx,
            trace,
            num_provers,
            chunking_strategy,
        )?;

        // Serialize the proof
        let proof_bytes =
            bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap();
        info!(
            "Proof size: {}",
            humansize::format_size(proof_bytes.len(), humansize::BINARY)
        );

        // Verify the proof
        verifier_ctx.verify(proof, user_tokens, io)?;
        Ok(())
    }

    #[test]
    fn test_llm_driver_distributed_prove_gpt2() -> anyhow::Result<()> {
        test_llm_driver_generic_prove_gpt2::<true>()
    }

    #[test]
    #[ignore = "Sequential case covered already by distributed prove test, use this test only when checking performance to ensure sequential proving is used"]
    fn test_llm_driver_prove_gpt2() -> anyhow::Result<()> {
        test_llm_driver_generic_prove_gpt2::<false>()
    }

    #[test]
    #[ignore = "failing test - need to check why"]
    fn test_llm_ctx_serialization() -> anyhow::Result<()> {
        let prover_ctx_path = "prover_ctx.bin";
        let verifier_ctx_path = "verifier_ctx.bin";
        let _ = std::fs::remove_file(prover_ctx_path);
        let _ = std::fs::remove_file(verifier_ctx_path);
        let model_path = file_cache::from_cache("toy_gpt2.gguf")?;
        let (driver, _metadata) =
            Driver::load_from_model(GPT2::new(), &RawGGUF::new(model_path.clone()), Some(10))?
                .into_provable_llm(None)?;
        let (prover_ctx, _) = driver.context::<F, Pcs>()?.with_max_context(10);

        let pctx = bincode::serde::encode_to_vec(&prover_ctx, bincode::config::standard())?;
        let (deser_prover_ctx, _): (ProverContext<F, Pcs>, _) =
            bincode::serde::decode_from_slice(&pctx, bincode::config::standard())?;
        let pctx2 = bincode::serde::encode_to_vec(&deser_prover_ctx, bincode::config::standard())?;
        assert_eq!(pctx, pctx2);
        Ok(())
    }

    #[test]
    fn test_llm_driver_inference_gpt2_gguf() -> anyhow::Result<()> {
        init_test_logging("debug");
        const PRUNED_GPT2: &str = "gpt2.Q2_K.gguf";
        let model_path = file_cache::from_cache(PRUNED_GPT2)?;
        println!("model path: {:?}", model_path);

        let raw = RawGGUF::new(model_path.clone());
        let (driver, _metadata) =
            Driver::load_from_model(GPT2::new(), &raw, Some(10))?.into_provable_llm(None)?;
        test_llm_driver_inference_inner(raw, driver)
    }

    #[test]
    fn test_llm_driver_inference_gpt2_safe() -> anyhow::Result<()> {
        const GPT2_SAFE_MODEL: &str = "openai-community/gpt2";
        init_test_logging("debug");
        let raw = RawSafeTensors::from_hugging_face_cached(GPT2_SAFE_MODEL)?;

        let (driver, _metadata) =
            Driver::load_from_model(GPT2::new(), &raw, Some(10))?.into_provable_llm(None)?;
        test_llm_driver_inference_inner(raw, driver)
    }

    fn test_llm_driver_inference_inner<R>(raw: R, driver: Driver<i64>) -> anyhow::Result<()>
    where
        GPT2: TokenizerLoader<R>,
    {
        let sentence = "The sky is";

        // Best to load the tokenizer from the gguf file if it's available.
        let tokenizer = GPT2::new().load_tokenizer(&raw)?;
        let user_tokens = tokenizer.tokenize(sentence);
        let detokenized = tokenizer.detokenize(&user_tokens);
        let mut store = GenStore::default();
        assert_eq!(detokenized, sentence);
        println!("user input in tokens: {user_tokens:?}");
        let input_tensor = driver.tokens_to_tensor(&user_tokens)?;
        let trace = driver.run_elements(input_tensor, &mut store)?;
        let output = trace
            .outputs()
            .last()
            .unwrap()
            .tensor()
            .unwrap()
            .data()
            .iter()
            .map(|t| Token::from(t.to_usize()))
            .collect::<Vec<_>>();
        let output = tokenizer.detokenize(&output);
        println!("{}", output);
        Ok(())
    }

    #[test]
    fn test_llm_driver_inference_gemma3() -> anyhow::Result<()> {
        const CONTEXT_SIZE: usize = 10;
        init_test_logging("debug");
        let model_path = file_cache::from_cache(GEMMA3_Q8)?;
        let gguf = RawGGUF::new(model_path.clone());
        let driver = Driver::load_from_model(Gemma3::new(), &gguf, Some(CONTEXT_SIZE))?;

        println!("LLM DRIVER: config: {:?}", driver.md.config);

        let mut store = GenStore::default();
        let sentence = "The sky is";
        let tokenizer = Gemma3::new().load_tokenizer(&gguf)?;
        let user_tokens = tokenizer.tokenize(sentence);
        let tensor_inputs = vec![driver.tokens_to_tensor(&user_tokens)?];
        let trace = driver.run(tensor_inputs, &mut store)?;
        let output = trace
            .outputs()
            .last()
            .unwrap()
            .tensor()
            .unwrap()
            .data()
            .iter()
            .map(|t| Token::from(t.to_usize()))
            .collect::<Vec<_>>();
        let _output = tokenizer.detokenize(&output);
        Ok(())
    }

    #[test]
    fn test_llm_driver_int_inference_gemma3() -> anyhow::Result<()> {
        const CONTEXT_SIZE: usize = 10;
        init_test_logging("debug");
        let model_path = file_cache::from_cache(GEMMA3_Q8)?;
        let gguf = RawGGUF::new(model_path.clone());
        let (driver, _metadata) =
            Driver::load_from_model(Gemma3::new(), &gguf, Some(CONTEXT_SIZE))?
                .into_provable_llm(None)?;

        println!("LLM DRIVER: config: {:?}", driver.md.config);

        let mut store = GenStore::default();
        let sentence = "The sky is";
        let tokenizer = Gemma3::new().load_tokenizer(&gguf)?;
        let user_tokens = tokenizer.tokenize(sentence);
        let input_tensor = driver.tokens_to_tensor(&user_tokens)?;
        let trace = driver.run_elements(input_tensor, &mut store)?;
        let output = trace
            .outputs()
            .last()
            .unwrap()
            .tensor()
            .unwrap()
            .data()
            .iter()
            .map(|t| Token::from(t.to_usize()))
            .collect::<Vec<_>>();
        let _output = tokenizer.detokenize(&output);
        Ok(())
    }

    fn test_generic_prove_llm_gemma3<const DISTRIBUTE_PROVE: bool>() -> anyhow::Result<()> {
        init_test_logging("debug");
        const MAX_CONTEXT: usize = 8;
        let model_path = file_cache::from_cache(GEMMA3_Q8)?;
        let cache_filename = {
            let mut hasher = blake3::Hasher::new();
            hasher
                .update_mmap(&model_path)
                .context("hashing model file")?;
            let hash = hasher.finalize();
            format!("cache-{GEMMA3_Q8}-{hash}.bin")
        };

        // Generate or load the prover & verifier contexts
        #[allow(clippy::type_complexity)]
        let (driver, prover_ctx, verifier_ctx): (
            _,
            ProverContext<F, Pcs>,
            LLMVerifierContext<F, Pcs>,
        ) = file_cache::deserialize_or_create_with(&cache_filename, || {
            let gguf = RawGGUF::new(model_path.clone());
            let (driver, _metadata) =
                Driver::load_from_model(Gemma3::new(), &gguf, Some(MAX_CONTEXT))?
                    .into_provable_llm(None)?;

            let (prover_ctx, verifier_ctx) =
                driver.context::<F, Pcs>()?.with_max_context(MAX_CONTEXT);

            Ok((driver, prover_ctx, verifier_ctx))
        })?;

        println!("LLM DRIVER: config: {:?}", driver.md.config);
        // Generate the trace
        let mut store = GenStore::default();
        let sentence = "The sky is";
        let loader = TensorLoader::from_path(model_path)?;
        let tokenizer = HFTokenizer::sentencepiece_from_gguf(&loader)?;
        let user_tokens = tokenizer.tokenize(sentence);
        let input_tensor = driver.tokens_to_tensor(&user_tokens)?;
        let trace = driver.run_elements(input_tensor, &mut store)?;

        // Prove the trace
        let num_provers = if DISTRIBUTE_PROVE {
            Some(rng_from_env_or_random().gen_range(1..6))
        } else {
            Some(1) // equivalent to sequential proving
        };
        let chunking_strategy = LLMChunkingStrategy::from((&trace, &prover_ctx.model_ctx));
        let (proof, io) = driver.chunked_prove_default_executor(
            &prover_ctx,
            trace,
            num_provers,
            chunking_strategy,
        )?;

        // Serialize the proof
        let proof_bytes =
            bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap();
        info!(
            "Proof size: {}",
            humansize::format_size(proof_bytes.len(), humansize::BINARY)
        );

        // Verify the proof
        verifier_ctx.verify(proof, user_tokens, io)?;
        Ok(())
    }

    #[test]
    #[ignore = "Test requires large machine to run"]
    fn test_distribute_prove_llm_gemma3() -> anyhow::Result<()> {
        test_generic_prove_llm_gemma3::<true>()
    }

    #[test]
    #[ignore = "Test requires large machine to run"]
    fn test_prove_llm_gemma3() -> anyhow::Result<()> {
        test_generic_prove_llm_gemma3::<false>()
    }

    // ---- Llama2 (SafeTensors) ----

    fn test_generic_prove_llm_llama2<const DISTRIBUTE_PROVE: bool>() -> anyhow::Result<()> {
        use crate::parser::llm::models::llama2::{Llama2, tests::LLAMA2_SAFE_MODEL};

        init_test_logging("debug");
        const MAX_CONTEXT: usize = 10;

        let raw = RawSafeTensors::from_hugging_face_cached(LLAMA2_SAFE_MODEL)?;
        let (driver, _metadata) = Driver::load_from_model(Llama2::new(), &raw, Some(MAX_CONTEXT))?
            .into_provable_llm(None)?;

        let (prover_ctx, verifier_ctx) = driver.context::<F, Pcs>()?.with_max_context(MAX_CONTEXT);

        driver.model.describe();

        // Generate the trace
        let mut store = GenStore::default();
        let sentence = "The sky is";
        let tokenizer = Llama2::new().load_tokenizer(&raw)?;
        let user_tokens = tokenizer.tokenize(sentence);
        let input_tensor = driver.tokens_to_tensor(&user_tokens)?;
        let trace = driver.run_elements(input_tensor, &mut store)?;

        // Prove the trace
        let num_provers = if DISTRIBUTE_PROVE {
            Some(rng_from_env_or_random().gen_range(1..6))
        } else {
            Some(1)
        };
        let chunking_strategy = LLMChunkingStrategy::from((&trace, &prover_ctx.model_ctx));
        let (proof, io) = driver.chunked_prove_default_executor(
            &prover_ctx,
            trace,
            num_provers,
            chunking_strategy,
        )?;

        // Serialize the proof
        let proof_bytes =
            bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap();
        info!(
            "Proof size: {}",
            humansize::format_size(proof_bytes.len(), humansize::BINARY)
        );

        // Verify the proof
        verifier_ctx.verify(proof, user_tokens, io)?;
        Ok(())
    }

    #[test]
    #[ignore = "Test requires large machine to run"]
    fn test_distribute_prove_llm_llama2() -> anyhow::Result<()> {
        test_generic_prove_llm_llama2::<true>()
    }

    #[test]
    #[ignore = "Test requires large machine to run"]
    fn test_prove_llm_llama2() -> anyhow::Result<()> {
        test_generic_prove_llm_llama2::<false>()
    }
}
