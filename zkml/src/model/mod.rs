use std::{
    mem,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use crate::{
    Element, NextPowerOfTwo, Shape, Tensor,
    graph::{Direction, Edge, Graph, Node, NodeId, NodeInput, NodeOutput, PortLink, Ports},
    iop::prover::{ModelLayers, ModelLayersRef},
    layers::{
        Layer, NodeOut,
        convolution::ConvFFTHandle,
        provable::{Evaluate, OpInfo, ProvingHandle, TrackedDataId},
        requant::Requant,
        transformer::{
            ConcatenationCache,
            logits::ArgmaxHandle,
            normalisation::{
                layernorm::evaluate::LayerNormHandle, rmsnorm::evaluate::RMSNormHandle,
            },
            softmax::SoftmaxHandle,
        },
    },
    model::trace::{SplittedNodesInfo, TraceSplitterInfo},
    padding::PaddingMode,
    quantization::InferenceTracker,
    tensor::{TensorHandle, TensorTypeParam, WrappedTensor},
};
use anyhow::{Context, Result, anyhow, bail, ensure};
use itertools::izip;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    ops::Deref,
};
use tenstore::{GenStore, GenericStore, StorageKey};
use tracing::{info, info_span, warn};

pub(crate) mod context;
pub mod exec_graph;
pub mod llm;
pub(crate) mod trace;
pub mod transform;
pub use context::{ContextGraph, ModelCtx};
pub use trace::{Step, Trace};

pub(crate) type RunResult<N> = anyhow::Result<RunOutput<N>>;

pub struct RunOutput<N>
where
    N: TensorTypeParam,
{
    pub(crate) outputs: Vec<TensorHandle<N>>,
    pub(crate) proving_data: ProvingHandle,
    pub(crate) tracked_data: HashMap<TrackedDataId, WrappedTensor<N>>,
    trace_split_info: TraceSplitterInfo<N>,
}

/// Selects where autoregressive K/V tensors are cached during inference.
///
/// This only affects the inference runner. The model graph and the trace used
/// for proving remain unchanged.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum KvCacheMode {
    /// Preserve the model's original behavior: cache wide QKV projection
    /// outputs before their downstream ReQuant nodes.
    #[default]
    PreRequant,
    /// ReQuantize only the newly projected K/V values and cache the resulting
    /// attention-ready tensors.
    PostRequant,
}

/// Measurements collected for the K/V ReQuant nodes during autoregressive
/// inference.
#[derive(Clone, Debug, Default)]
pub struct KvCacheMetrics {
    pub kv_requant_calls: usize,
    pub kv_requant_elements: usize,
    pub kv_requant_time: Duration,
    pub peak_kv_cache_elements: usize,
    pub verified_kv_tensors: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerCapture {
    pub node_id: NodeId,
    pub outputs: Vec<Vec<Element>>,
}

impl KvCacheMetrics {
    /// Physical cache size for the current unpacked `Element = i64`
    /// representation.
    pub fn peak_kv_cache_bytes(&self) -> usize {
        self.peak_kv_cache_elements * std::mem::size_of::<Element>()
    }
}

/// Inference-only runner that can move GPT-style K/V concatenation from the
/// QKV projection outputs to the corresponding post-ReQuant outputs.
///
/// The underlying graph is not rewritten. In post-ReQuant mode the original
/// wide-integer QKV output caches are temporarily bypassed, while this runner
/// maintains equivalent caches after the K/V ReQuant nodes.
pub struct KvCacheRunner<I> {
    inner: I,
    mode: KvCacheMode,
    qkv_nodes: HashSet<NodeId>,
    kv_requant_caches: HashMap<NodeId, ConcatenationCache<Element>>,
    verification_caches: Option<HashMap<NodeId, ConcatenationCache<Element>>>,
    capture_nodes: Option<HashSet<NodeId>>,
    captures: Vec<LayerCapture>,
    current_cache_elements: HashMap<NodeId, usize>,
    metrics: KvCacheMetrics,
}

impl<I> KvCacheRunner<I> {
    pub fn new(inner: I, graph: &ModelGraph<Element>, mode: KvCacheMode) -> anyhow::Result<Self> {
        let qkv_nodes = graph
            .inner_nodes()
            .filter_map(|(node_id, layer)| match layer {
                Layer::EinSum(einsum)
                    if einsum.check_name(
                        crate::parser::llm::transformer::attention_layer::ATTENTION_QKV_EINSUM_NAME,
                    ) =>
                {
                    Some(node_id)
                }
                _ => None,
            })
            .collect::<HashSet<_>>();

        let mut kv_requant_caches = HashMap::new();
        for qkv_node_id in qkv_nodes.iter().copied() {
            let Layer::EinSum(qkv_einsum) = graph
                .node(qkv_node_id)
                .and_then(Node::as_inner)
                .context("QKV node is not an inner EinSum layer")?
            else {
                bail!("QKV node is not an EinSum layer");
            };

            for feed in graph.outgoing_feeds(qkv_node_id) {
                let source_port = *feed.source().port();
                if source_port != 1 && source_port != 2 {
                    continue;
                }

                let requant_node_id = feed.target().node_id();
                let is_requant = matches!(
                    graph.node(requant_node_id).and_then(Node::as_inner),
                    Some(Layer::Requant(_))
                );
                if !is_requant {
                    continue;
                }

                let source_cache = qkv_einsum.caches[source_port]
                    .as_ref()
                    .context("K/V QKV output is missing its concatenation cache")?;
                let (rank, concatenation_dim) = source_cache.lock().unwrap().cache_info();
                kv_requant_caches.insert(
                    requant_node_id,
                    ConcatenationCache::new(rank, concatenation_dim),
                );
            }
        }

        ensure!(
            !qkv_nodes.is_empty(),
            "No attention QKV EinSum nodes found for K/V cache benchmarking"
        );
        ensure!(
            kv_requant_caches.len() == qkv_nodes.len() * 2,
            "Expected two K/V ReQuant nodes per QKV projection, found {} for {} QKV nodes",
            kv_requant_caches.len(),
            qkv_nodes.len()
        );

        Ok(Self {
            inner,
            mode,
            qkv_nodes,
            kv_requant_caches,
            verification_caches: None,
            capture_nodes: None,
            captures: Vec::new(),
            current_cache_elements: HashMap::new(),
            metrics: KvCacheMetrics::default(),
        })
    }

    /// Enables an inference-only bit-exact check. For every new K/V tensor, a
    /// shadow pre-ReQuant cache evaluates the original full-history ReQuant and
    /// compares it with the concatenated post-ReQuant cache output.
    pub fn with_verification(mut self) -> Self {
        self.verification_caches = Some(self.kv_requant_caches.clone());
        self
    }

    pub fn with_capture_nodes(mut self, capture_nodes: HashSet<NodeId>) -> Self {
        self.capture_nodes = Some(capture_nodes);
        self
    }

    pub fn metrics(&self) -> &KvCacheMetrics {
        &self.metrics
    }

    pub fn into_metrics(self) -> KvCacheMetrics {
        self.metrics
    }

    pub fn into_metrics_and_captures(self) -> (KvCacheMetrics, Vec<LayerCapture>) {
        (self.metrics, self.captures)
    }
}

impl<I> LayerRunner<Element, RunInput<Element>> for KvCacheRunner<I>
where
    I: LayerRunner<Element, RunInput<Element>>,
{
    fn model_inputs(
        &mut self,
        graph: &ModelGraph<Element>,
        inputs: &[TensorHandle<Element>],
    ) -> anyhow::Result<()> {
        self.inner.model_inputs(graph, inputs)
    }

    fn run_layer(
        &mut self,
        node_id: NodeId,
        graph: &ModelGraph<Element>,
        layer: &Layer<Element>,
        inputs: &RunInput<Element>,
    ) -> RunResult<Element>
    where
        Layer<Element>: Evaluate<Element>,
    {
        if self.mode == KvCacheMode::PostRequant && self.qkv_nodes.contains(&node_id) {
            let Layer::EinSum(einsum) = layer else {
                unreachable!("QKV node set must contain only EinSum layers")
            };
            // Bypass the original wide-integer caches for this projection. The
            // newly projected K/V values will be cached after ReQuant below.
            einsum.set_caches_disabled(true);
        }

        let is_kv_requant = self.kv_requant_caches.contains_key(&node_id);
        let requant_elements = if is_kv_requant {
            inputs
                .input_handles
                .first()
                .context("K/V ReQuant node has no input")?
                .unpadded_shape()
                .numel()
        } else {
            0
        };

        let expected_full_requant = if self.mode == KvCacheMode::PostRequant {
            if let Some(verification_caches) = self.verification_caches.as_mut() {
                if is_kv_requant {
                    let new_wide_tensor = inputs
                        .input_handles
                        .first()
                        .context("K/V ReQuant node has no input")?
                        .wrapped_tensor()?
                        .clone();
                    let full_wide_tensor = verification_caches
                        .get_mut(&node_id)
                        .expect("K/V verification cache must exist")
                        .concatenate(new_wide_tensor)?;
                    let expected = layer.evaluate(&[&full_wide_tensor])?;
                    Some(
                        expected
                            .outputs
                            .into_iter()
                            .next()
                            .context("Verified K/V ReQuant has no output")?,
                    )
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let start = Instant::now();
        let run_result = self.inner.run_layer(node_id, graph, layer, inputs);
        let elapsed = start.elapsed();

        if self.mode == KvCacheMode::PostRequant && self.qkv_nodes.contains(&node_id) {
            let Layer::EinSum(einsum) = layer else {
                unreachable!("QKV node set must contain only EinSum layers")
            };
            // Restore the graph's default behavior for callers outside this
            // inference-only runner, including full trace generation.
            einsum.set_caches_disabled(false);
        }
        let mut output = run_result?;

        if is_kv_requant {
            self.metrics.kv_requant_calls += 1;
            self.metrics.kv_requant_elements += requant_elements;
            self.metrics.kv_requant_time += elapsed;

            let cache_elements = if self.mode == KvCacheMode::PostRequant {
                let output_handle = output
                    .outputs
                    .first()
                    .context("K/V ReQuant node has no output")?;
                let storage_key = output_handle.storage_key().clone();
                let store = output_handle.store().clone();
                let new_tensor = output_handle.wrapped_tensor()?.clone();
                let cached_tensor = self
                    .kv_requant_caches
                    .get_mut(&node_id)
                    .expect("K/V ReQuant cache must exist")
                    .concatenate(new_tensor)?;
                let cached_shape = Shape::from(cached_tensor.unpadded_shape());
                let cached_elements = cached_shape.numel();
                if let Some(expected) = expected_full_requant {
                    ensure!(
                        cached_tensor.get_data() == expected.get_data(),
                        "Post-ReQuant K/V cache differs from original full-history ReQuant at node {node_id}"
                    );
                    self.metrics.verified_kv_tensors += 1;
                }
                output.outputs[0] = TensorHandle::from_wrapped_tensor_with_unpadded_shape(
                    storage_key,
                    store,
                    cached_tensor,
                    cached_shape,
                );
                cached_elements
            } else {
                // In the original path the ReQuant input is the full cached K/V
                // tensor produced by the QKV EinSum.
                requant_elements
            };

            self.current_cache_elements.insert(node_id, cache_elements);
            self.metrics.peak_kv_cache_elements = self
                .metrics
                .peak_kv_cache_elements
                .max(self.current_cache_elements.values().sum());
        }

        if self
            .capture_nodes
            .as_ref()
            .is_some_and(|nodes| nodes.contains(&node_id))
        {
            let outputs = output
                .outputs
                .iter()
                .map(|handle| Ok(handle.wrapped_tensor()?.get_data()))
                .collect::<anyhow::Result<Vec<_>>>()?;
            self.captures.push(LayerCapture { node_id, outputs });
        }

        Ok(output)
    }
}

/// Utility to convert model's input tensors to handles.
///
/// This function will match the input tensors in definition order against the
/// model's inputs.
/// The returned values are of the `TensorHandle::Tensor` variant.
pub fn tensor_to_handles<N>(
    inputs: &[Tensor<N>],
    graph: &ModelGraph<N>,
    store: &mut GenStore,
) -> anyhow::Result<Vec<TensorHandle<N>>>
where
    N: TensorTypeParam,
{
    let mut input_handles = Vec::with_capacity(inputs.len());
    for (i, tensor) in inputs.iter().enumerate() {
        let input_node_id = graph.input_node_id(i)?;
        let storage_key = input_node_id.output_at(0).to_storage_key();

        let handle = TensorHandle::from_tensor(storage_key, store.clone(), tensor.clone());
        input_handles.push(handle.clone());
    }
    Ok(input_handles)
}

/// Utility to convert model's inputs as wrapped tensors to handles.
///
/// This function will match the input tensors in definition order against the
/// model's inputs.
/// The returned values are of the `TensorHandle::WrappedTensor` variant.
pub fn wrapped_tensor_to_handles<N>(
    inputs: &[WrappedTensor<N>],
    graph: &ModelGraph<N>,
    store: &mut GenStore,
) -> anyhow::Result<Vec<TensorHandle<N>>>
where
    N: TensorTypeParam,
{
    let mut input_handles = Vec::with_capacity(inputs.len());
    for (i, tensor) in inputs.iter().enumerate() {
        let input_node_id = graph.input_node_id(i)?;
        let storage_key = input_node_id.output_at(0).to_storage_key();

        let handle = TensorHandle::from_wrapped_tensor(storage_key, store.clone(), tensor.clone());
        input_handles.push(handle.clone());
    }
    Ok(input_handles)
}

pub trait ToStorageKey<N> {
    /// Return the key under which the data of the object referred to by the
    /// implementer of this trait is stored.
    fn to_storage_key(&self) -> StorageKey<N>;
}

/// Input data for a layer runner.
#[derive(Debug)]
pub(crate) struct RunInput<N>
where
    N: TensorTypeParam,
{
    input_handles: Vec<TensorHandle<N>>,
}

pub trait LayerRunner<N, I>
where
    N: TensorTypeParam,
{
    /// Called once per model inference run.
    ///
    /// NOTE: For LLMs this will be called once per token.
    fn model_inputs(
        &mut self,
        graph: &ModelGraph<N>,
        inputs: &[TensorHandle<N>],
    ) -> anyhow::Result<()>;

    /// Called once per model's layer.
    fn run_layer(
        &mut self,
        node_id: NodeId,
        graph: &ModelGraph<N>,
        layer: &Layer<N>,
        inputs: &I,
    ) -> RunResult<N>
    where
        Layer<N>: Evaluate<N>;
}

/// Base layer runner.
///
/// This runner bridges the use of [TensorHandle]s and [WrappedTensor]s used to
/// run the layers.
pub struct BaseRunner<'a> {
    pub store: GenStore,
    split_nodes_info: Option<&'a SplittedNodesInfo>,
}

impl<'a> From<GenStore> for BaseRunner<'a> {
    fn from(value: GenStore) -> Self {
        Self {
            store: value,
            split_nodes_info: None,
        }
    }
}

impl<'a, N> LayerRunner<N, RunInput<N>> for BaseRunner<'a>
where
    N: TensorTypeParam,
{
    fn model_inputs(
        &mut self,
        _graph: &ModelGraph<N>,
        _inputs: &[TensorHandle<N>],
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn run_layer(
        &mut self,
        node_id: NodeId,
        graph: &ModelGraph<N>,
        layer: &Layer<N>,
        inputs: &RunInput<N>,
    ) -> RunResult<N>
    where
        Layer<N>: Evaluate<N>,
    {
        let mut wrapped_tensors_guards = Vec::with_capacity(inputs.input_handles.len());

        for handle in inputs.input_handles.iter() {
            wrapped_tensors_guards.push(handle.wrapped_tensor()?);
        }

        let wrapped_tensors = wrapped_tensors_guards
            .iter()
            .map(|guard| guard.deref())
            .collect::<Vec<_>>();

        // We calculate the unpadded output shapes before running the layer
        // in case the layer uses a concatenation cache. This ensures that the
        // shapes are computed correctly without interfering with the layer's
        // internal state during evaluation.
        let prec_unpadded_shapes: Vec<_> = inputs
            .input_handles
            .iter()
            .map(|handle| handle.unpadded_shape().clone())
            .collect();
        let out_shapes = layer.output_shapes(&prec_unpadded_shapes, PaddingMode::NoPadding)?;

        let layer_out = layer.evaluate(&wrapped_tensors)?;

        let mut outputs = Vec::with_capacity(out_shapes.len());
        let out_ports = graph.outgoing_ports(node_id);

        for (port, out_shape, tensor) in izip!(&out_ports, &out_shapes, layer_out.outputs) {
            let storage_key: StorageKey<Vec<N>> = port.to_storage_key();
            let handle = TensorHandle::from_wrapped_tensor_with_unpadded_shape(
                storage_key,
                self.store.clone(),
                tensor,
                out_shape.clone(),
            );
            outputs.push(handle);
        }

        let storage_key = node_id.to_storage_key();
        let proving_data =
            ProvingHandle::new(storage_key, layer_out.proving_data, self.store.clone());

        let trace_split_info = if let Some(trace_split) = self.split_nodes_info {
            trace_split.map_handles(
                node_id,
                graph,
                &inputs.input_handles,
                &outputs,
                &proving_data,
            )?
        } else {
            Default::default()
        };

        Ok(RunOutput {
            outputs,
            proving_data,
            tracked_data: layer_out.tracked_layer_data,
            trace_split_info,
        })
    }
}

/// Message sent by the [DebugRunner] to its background task.
#[derive(Debug)]
enum DebugRunnerMsg<T>
where
    T: TensorTypeParam,
{
    /// Start a new trace
    NewTrace { input_tensors: Vec<TensorHandle<T>> },

    /// Pass a tensor to be saved.
    LayerRun {
        node_id: NodeId,
        node_inputs: Vec<TensorHandle<T>>,
        node_outputs: Vec<TensorHandle<T>>,
        proving_data: Box<ProvingHandle>,
    },
}

pub struct DebugRunner<I, N>
where
    N: TensorTypeParam,
{
    inner: I,
    tx: mpsc::Sender<DebugRunnerMsg<N>>,
    thread_handle: thread::JoinHandle<anyhow::Result<Vec<Trace<N>>>>,
}

impl<I, N> DebugRunner<I, N>
where
    N: TensorTypeParam,
{
    pub fn new(inner: I) -> Self {
        let (tx, rx) = mpsc::channel::<DebugRunnerMsg<N>>();

        // Background thread used to copy the tensor handle data from the
        // wrapped tensor to native tensor.
        //
        // This allow the inference to continue running, while the data is
        // downloaded in the background. The download is necessary to ensure
        // data is freed if the inference is done on the GPU. This is done
        // for CPU runs too, because it is easier to assume the handle always
        // contains a native tensor.
        let thread_handle = thread::spawn(move || -> anyhow::Result<Vec<Trace<N>>> {
            let mut traces = Vec::new();

            let mut trace = match rx.recv() {
                Ok(DebugRunnerMsg::NewTrace { input_tensors }) => {
                    let input_tensors = input_tensors
                        .into_iter()
                        .map(|handle| handle.tensor_variant())
                        .collect::<Result<_, _>>()?;
                    Trace::new(input_tensors)
                }
                Ok(DebugRunnerMsg::LayerRun { .. }) => {
                    bail!("Expects the first message to be DebugRunnerMsg::NewTrace")
                }
                Err(_) => {
                    bail!("Sender disconnected without an initial DebugRunnerMsg::NewTrace");
                }
            };

            while let Ok(msg) = rx.recv() {
                match msg {
                    DebugRunnerMsg::NewTrace { input_tensors } => {
                        traces.push(trace);
                        let input_tensors = input_tensors
                            .into_iter()
                            .map(|handle| handle.tensor_variant())
                            .collect::<Result<_, _>>()?;
                        trace = Trace::new(input_tensors);
                    }
                    DebugRunnerMsg::LayerRun {
                        node_id,
                        node_inputs,
                        node_outputs,
                        proving_data,
                    } => {
                        let node_inputs = node_inputs
                            .into_iter()
                            .map(|handle| handle.tensor_variant())
                            .collect::<Result<_, _>>()?;
                        let node_outputs = node_outputs
                            .into_iter()
                            .map(|handle| handle.tensor_variant())
                            .collect::<Result<_, _>>()?;
                        let proving_data = match *proving_data {
                            ProvingHandle::Convolution(conv_ffthandle) => {
                                ProvingHandle::Convolution(ConvFFTHandle {
                                    handle: conv_ffthandle.handle.tensor_variant()?,
                                })
                            }
                            ProvingHandle::Softmax(softmax_handle) => {
                                ProvingHandle::Softmax(SoftmaxHandle {
                                    shift_handle: softmax_handle.shift_handle.tensor_variant()?,
                                })
                            }
                            ProvingHandle::LayerNorm(layer_norm_handle) => {
                                ProvingHandle::LayerNorm(LayerNormHandle {
                                    mean: layer_norm_handle.mean.tensor_variant()?,
                                    std_dev: layer_norm_handle.std_dev.tensor_variant()?,
                                    lookup_verifier: layer_norm_handle.lookup_verifier,
                                })
                            }
                            ProvingHandle::RMSNorm(rms_norm_handle) => {
                                ProvingHandle::RMSNorm(RMSNormHandle {
                                    normalisation: rms_norm_handle
                                        .normalisation
                                        .tensor_variant()?,
                                    lookup_verifier: rms_norm_handle.lookup_verifier,
                                })
                            }
                            ProvingHandle::ArgMax(argmax_handle) => {
                                ProvingHandle::ArgMax(ArgmaxHandle {
                                    max_values: argmax_handle
                                        .max_values
                                        .into_iter()
                                        .map(|handle| handle.tensor_variant())
                                        .collect::<Result<_, _>>()?,
                                })
                            }
                            ProvingHandle::None => ProvingHandle::None,
                        };
                        let new_step = Step {
                            node_inputs,
                            node_outputs: NodeOut::new(node_outputs, proving_data),
                        };
                        trace.new_step(node_id, new_step);
                    }
                }
            }
            traces.push(trace);

            Ok(traces)
        });

        Self {
            inner,
            tx,
            thread_handle,
        }
    }

    /// Consumes the runner and return the collected traces.
    pub fn into_traces(self) -> anyhow::Result<Vec<Trace<N>>> {
        // signal to the thread that we are done
        drop(self.tx);

        self.thread_handle
            .join()
            .map_err(|_err| anyhow!("Background thread failed"))
            .flatten()
    }
}

impl<I, N> LayerRunner<N, RunInput<N>> for DebugRunner<I, N>
where
    I: LayerRunner<N, RunInput<N>>,
    N: TensorTypeParam,
{
    fn model_inputs(
        &mut self,
        graph: &ModelGraph<N>,
        inputs: &[TensorHandle<N>],
    ) -> anyhow::Result<()> {
        let input_tensors = inputs
            .iter()
            .map(|handle| handle.isolate())
            .collect::<Vec<_>>();
        let _ = self.tx.send(DebugRunnerMsg::NewTrace { input_tensors });

        self.inner.model_inputs(graph, inputs)
    }

    fn run_layer(
        &mut self,
        node_id: NodeId,
        graph: &ModelGraph<N>,
        layer: &Layer<N>,
        inputs: &RunInput<N>,
    ) -> RunResult<N>
    where
        Layer<N>: Evaluate<N>,
    {
        let layer_out = self.inner.run_layer(node_id, graph, layer, inputs)?;

        let node_inputs = inputs
            .input_handles
            .iter()
            .map(|handle| handle.isolate())
            .collect::<Vec<_>>();
        let node_outputs = layer_out
            .outputs
            .iter()
            .map(|handle| handle.isolate())
            .collect::<Vec<_>>();

        let _ = self.tx.send(DebugRunnerMsg::LayerRun {
            node_id,
            node_inputs,
            node_outputs,
            proving_data: Box::new(layer_out.proving_data.isolate()),
        });

        Ok(layer_out)
    }
}

/// Message sent by the [StoreRunner] to its background task.
enum StoreRunnerMsg<T>
where
    T: TensorTypeParam,
{
    /// Used for tensors that vary with `N`.
    ///
    /// Used for the inference data, which may be float or `Element`.
    Generic {
        storage_key: StorageKey<Vec<T>>,
        wrapped_tensor: WrappedTensor<T>,
    },

    /// Used for tensors that are always `Element`
    ///
    /// Used for the proving data, when available.
    Element {
        storage_key: StorageKey<Vec<Element>>,
        wrapped_tensor: WrappedTensor<Element>,
    },
}

/// A runner used to save tensor data to the store in the background.
///
/// This runner will spawn a background thread which waits on the results,
/// and once available saves the data to the store. This allows the inference
/// to continue running without blocking on pending computation and the data
/// transfer.
pub struct StoreRunner<I, N>
where
    N: TensorTypeParam,
{
    inner: I,
    tx: Option<mpsc::Sender<StoreRunnerMsg<N>>>,
    thread_handle: Option<thread::JoinHandle<anyhow::Result<()>>>,
}

impl<I, N> StoreRunner<I, N>
where
    N: TensorTypeParam,
{
    pub fn new(inner: I, store: GenStore) -> Self {
        let (tx, rx) = mpsc::channel::<StoreRunnerMsg<N>>();
        let thread_handle = thread::spawn(move || {
            // The transmitter is closed when the the `StoreRunner` is dropped
            // and all pending messages have been received, at this point there
            // is no longer more work to be done.
            //
            // The code below works even in the presence of drying, this is
            // because drying a [TensorHandle::WrappedTensor] variant only
            // deletes the shared [WrappedTensor] copy, which in turn has
            // its own internal reference count. As long as this copy is not
            // deleted, the data should be available to be used. This does mean
            // it takes longer to free up the accelerator (CPU/GPU) memory.
            while let Ok(msg) = rx.recv() {
                match msg {
                    StoreRunnerMsg::Generic {
                        storage_key,
                        wrapped_tensor,
                    } => {
                        let data = wrapped_tensor.get_data();
                        store.store(&storage_key, &data)?;
                    }
                    StoreRunnerMsg::Element {
                        storage_key,
                        wrapped_tensor,
                    } => {
                        let data = wrapped_tensor.get_data();
                        store.store(&storage_key, &data)?;
                    }
                }
            }

            Ok(())
        });

        Self {
            inner,
            thread_handle: Some(thread_handle),
            tx: Some(tx),
        }
    }

    fn send(&self, handle: &TensorHandle<N>) -> anyhow::Result<()> {
        let wrapped_tensor = handle.wrapped_tensor()?.clone();
        let storage_key = handle.storage_key().clone();
        self.tx
            .as_ref()
            .expect("Sender channel is only be consumed in the Drop")
            .send(StoreRunnerMsg::Generic {
                storage_key,
                wrapped_tensor,
            })?;
        Ok(())
    }

    fn send_element(&self, handle: &TensorHandle<i64>) -> anyhow::Result<()> {
        let wrapped_tensor = handle.wrapped_tensor()?.clone();
        let storage_key = handle.storage_key().clone();
        self.tx
            .as_ref()
            .expect("Sender channel is only be consumed in the Drop")
            .send(StoreRunnerMsg::Element {
                storage_key,
                wrapped_tensor,
            })?;
        Ok(())
    }
}

impl<I, N> Drop for StoreRunner<I, N>
where
    N: TensorTypeParam,
{
    fn drop(&mut self) {
        // Drop the sender channel, signaling to the thread it should stop
        let tx = self.tx.take();
        drop(tx);

        // Wait for the thread to finish processing the buffered work
        let handle = self.thread_handle.take();
        if let Some(handle) = handle {
            handle
                .join()
                .expect("Store thread should not panic")
                .expect("Storing store results should not fail");
        }
    }
}

impl<I, N> LayerRunner<N, RunInput<N>> for StoreRunner<I, N>
where
    I: LayerRunner<N, RunInput<N>>,
    N: TensorTypeParam,
{
    fn model_inputs(
        &mut self,
        graph: &ModelGraph<N>,
        inputs: &[TensorHandle<N>],
    ) -> anyhow::Result<()> {
        for handle in inputs {
            self.send(handle)?;
        }
        self.inner.model_inputs(graph, inputs)?;
        Ok(())
    }

    fn run_layer(
        &mut self,
        node_id: NodeId,
        graph: &ModelGraph<N>,
        layer: &Layer<N>,
        inputs: &RunInput<N>,
    ) -> RunResult<N>
    where
        Layer<N>: Evaluate<N>,
    {
        let layer_out = self.inner.run_layer(node_id, graph, layer, inputs)?;

        for (original_handle, feed) in layer_out.outputs.iter().zip(graph.outgoing_feeds(node_id)) {
            let storage_key = original_handle.storage_key().clone();
            if let Some(new_handles) = layer_out.trace_split_info.output_handles.get(&storage_key) {
                for (handle, _) in new_handles.iter() {
                    self.send(handle)?;
                }
                // send to the store also the original output handle if there is a recombination layer associated to this layer
                if layer_out.trace_split_info.recombination_layer.is_some() {
                    self.send(original_handle)?;
                }
                // check if the current output is also an output of the model, so that we can send it to the store.
                // This is needed since the trace needs to have access to the unchunked outputs as well
                if graph
                    .node(feed.target.node_id)
                    .ok_or(anyhow!(
                        "Target node {} linked to node {node_id} not found",
                        feed.target.node_id
                    ))?
                    .is_output()
                {
                    self.send(original_handle)?;
                }
            } else {
                self.send(original_handle)?;
            }
        }

        if let Some((_, split_layer_handles)) = &layer_out.trace_split_info.split_layer {
            for handle in split_layer_handles {
                self.send(handle)?;
            }
        }

        if layer_out.trace_split_info.new_proving_handles.is_empty() {
            // the current node has not been split, so we store the original proving handle
            for handle in layer_out.proving_data.handles() {
                self.send_element(handle)?
            }
        } else {
            // otherwise, the current node has been split in chunks, so we store the proving handles
            // for each chunk
            for proving_handle in layer_out.trace_split_info.new_proving_handles.values() {
                for handle in proving_handle.handles() {
                    self.send_element(handle)?
                }
            }
        }

        for handle in layer_out.trace_split_info.model_input_handles.values() {
            self.send(handle)?;
        }

        Ok(layer_out)
    }
}

/// A runner used apply an inference tracker.
pub struct TrackerRunner<'a, I> {
    pub inner: I,
    pub tracker: &'a mut InferenceTracker,
}

impl<'a, I, N> LayerRunner<N, RunInput<N>> for TrackerRunner<'a, I>
where
    I: LayerRunner<N, RunInput<N>>,
    N: TensorTypeParam,
{
    fn model_inputs(
        &mut self,
        graph: &ModelGraph<N>,
        inputs: &[TensorHandle<N>],
    ) -> anyhow::Result<()> {
        for (i, handle) in inputs.iter().enumerate() {
            let input_node_id = graph.input_node_id(i)?;
            let port = 0;
            self.tracker.track(input_node_id.output_at(port), handle)?;
        }
        self.inner.model_inputs(graph, inputs)?;
        Ok(())
    }

    fn run_layer(
        &mut self,
        node_id: NodeId,
        graph: &ModelGraph<N>,
        layer: &Layer<N>,
        inputs: &RunInput<N>,
    ) -> RunResult<N>
    where
        Layer<N>: Evaluate<N>,
    {
        let mut layer_out = self.inner.run_layer(node_id, graph, layer, inputs)?;

        let out_ports = graph.outgoing_ports(node_id);
        for (port, handle) in out_ports.iter().zip(&layer_out.outputs) {
            self.tracker.track(node_id.output_at(port.port), handle)?;
        }

        for (data_id, handle) in layer_out.tracked_data.drain() {
            self.tracker
                .track_intermediate_data(node_id, data_id, handle);
        }

        Ok(layer_out)
    }
}

/// A runner which perform a few sanity checks to the layer results.
pub struct SanityCheckRunner<I> {
    pub inner: I,
}

impl<I, N> LayerRunner<N, RunInput<N>> for SanityCheckRunner<I>
where
    I: LayerRunner<N, RunInput<N>>,
    N: TensorTypeParam,
{
    fn model_inputs(
        &mut self,
        graph: &ModelGraph<N>,
        inputs: &[TensorHandle<N>],
    ) -> anyhow::Result<()> {
        assert_eq!(
            inputs.len(),
            graph.input_nodes().count(),
            "The number of input tensors must match the expected number of inputs of the model's graph"
        );
        self.inner.model_inputs(graph, inputs)?;
        Ok(())
    }

    fn run_layer(
        &mut self,
        node_id: NodeId,
        graph: &ModelGraph<N>,
        layer: &Layer<N>,
        inputs: &RunInput<N>,
    ) -> RunResult<N>
    where
        Layer<N>: Evaluate<N>,
    {
        let expected_num_outputs = layer.num_outputs(inputs.input_handles.len())?;
        let prec_unpadded_shapes: Vec<_> = inputs
            .input_handles
            .iter()
            .map(|handle| handle.unpadded_shape().clone())
            .collect();
        let out_shapes = layer.output_shapes(&prec_unpadded_shapes, PaddingMode::NoPadding)?;

        let layer_out = self.inner.run_layer(node_id, graph, layer, inputs)?;
        let out_ports = graph.outgoing_ports(node_id);

        ensure!(
            expected_num_outputs == layer_out.outputs.len(),
            "Unexpected number of output. expected {} got {} layer {}",
            expected_num_outputs,
            layer_out.outputs.len(),
            layer.describe(),
        );
        ensure!(
            out_ports.len() == expected_num_outputs,
            "Unexpected number of output ports. ports {:?} expected {} layer {}",
            out_ports,
            expected_num_outputs,
            layer.describe(),
        );
        ensure!(
            out_ports.len() == out_shapes.len(),
            "Unexpected number of output ports. handles {:?} shapes {:?} layer {}",
            out_ports,
            out_shapes,
            layer.describe(),
        );

        Ok(layer_out)
    }
}

/// A runner used to collect the [Trace].
///
/// # Panics
///
/// This runner may be used only for a single model run, duplicate layer results
/// will cause the runner to panic.
struct TraceRunner<I, N>
where
    N: TensorTypeParam,
{
    inner: I,
    trace: Trace<N>,
    model_output_handles: BTreeMap<usize, TensorHandle<N>>,
}

impl<I, N> TraceRunner<I, N>
where
    N: TensorTypeParam,
{
    pub fn new(inner: I, input_handles: Vec<TensorHandle<N>>) -> Self {
        Self {
            inner,
            trace: Trace::new(input_handles.clone()),
            model_output_handles: BTreeMap::new(),
        }
    }

    /// Consumes this runner and returns its parts.
    fn into_parts(mut self) -> anyhow::Result<(I, Trace<N>)> {
        self.build_trace_outputs()?;
        Ok((self.inner, self.trace))
    }

    fn build_trace_outputs(&mut self) -> anyhow::Result<()> {
        ensure!(
            !self.model_output_handles.is_empty(),
            "Expected at least one output handle for a model"
        );
        ensure!(
            *self.model_output_handles.keys().min().unwrap() == 0,
            "Expected first output index to be 0"
        );
        ensure!(
            *self.model_output_handles.keys().max().unwrap() == self.model_output_handles.len() - 1,
            "Not all output handles have been processed by trace runner"
        );
        self.trace.output = self.model_output_handles.values().cloned().collect();
        Ok(())
    }
}

impl<I, N> LayerRunner<N, RunInput<N>> for TraceRunner<I, N>
where
    I: LayerRunner<N, RunInput<N>>,
    N: TensorTypeParam,
{
    fn model_inputs(
        &mut self,
        graph: &ModelGraph<N>,
        inputs: &[TensorHandle<N>],
    ) -> anyhow::Result<()> {
        self.inner.model_inputs(graph, inputs)?;
        Ok(())
    }

    fn run_layer(
        &mut self,
        node_id: NodeId,
        graph: &ModelGraph<N>,
        layer: &Layer<N>,
        inputs: &RunInput<N>,
    ) -> RunResult<N>
    where
        Layer<N>: Evaluate<N>,
    {
        let mut layer_out = self.inner.run_layer(node_id, graph, layer, inputs)?;

        let trace_steps = self.trace.new_steps_for_splitted_nodes(
            node_id,
            &inputs.input_handles,
            &layer_out.outputs,
            mem::take(&mut layer_out.trace_split_info),
            mem::replace(&mut layer_out.proving_data, ProvingHandle::None),
        )?;
        for (node_id, step) in trace_steps {
            self.trace.new_step(node_id, step);
        }

        // check if there are outputs of this node that are also outputs of the model, so that we can add them
        // to `self.model_output_handles`
        for feed in graph.outgoing_feeds(node_id) {
            let target = graph.node(feed.target.node_id).ok_or(anyhow!(
                "Target node {} linked to node {node_id}not found",
                feed.target.node_id
            ))?;
            if let Some(output_idx) = target.as_output() {
                let output_handle = layer_out.outputs[*feed.source.port]
                    .clone()
                    .into_dry_tensor()?;
                let old_output = self.model_output_handles.insert(*output_idx, output_handle);
                ensure!(
                    old_output.is_none(),
                    "Trying to insert twice an output value for the same index {output_idx}",
                );
            }
        }

        Ok(layer_out)
    }
}

/// A runner to free memory once no longer needed.
pub struct HandleLifetimeRunner<I, N>
where
    N: TensorTypeParam,
{
    inner: I,
    storagekey_to_handle: HashMap<StorageKey<Vec<N>>, TensorHandle<N>>,
    handle_usage_count: HashMap<StorageKey<Vec<N>>, usize>,
}

impl<I, N> HandleLifetimeRunner<I, N>
where
    N: TensorTypeParam,
{
    /// Counts tensor usages from `graph` and initialise a [Handles<T>] from that.
    ///
    /// NOTE: A unique instance of `HandleLifetimeRunner` is required for each
    /// model run.
    pub fn new(inner: I, graph: &ModelGraph<N>) -> Self {
        let mut handle_usage_count = HashMap::new();

        // Ensure the model's outputs are available after the inference run. This
        // is achieved by incrementing the usage counter of the output tensors
        // by one, ensuring it never reaches zero.
        //
        // NOTE: This is done only once in the constructor, LLM models work
        // because the inner counters are refreshed for every new token by the
        // calls on `model_inputs`, _and_ because the handles are overwritten
        // after each layer run.
        for (_edge_id, edge) in graph.edges() {
            let target = edge.target();
            let is_output = graph
                .node(target)
                .map(|node| node.is_output())
                .unwrap_or(false);
            if !is_output {
                continue;
            }

            let source = edge.source();
            for link in edge.ports().iter() {
                let storage_key = source.output_at(link.source_port).to_storage_key();
                handle_usage_count
                    .entry(storage_key)
                    .and_modify(|curr| *curr += 1)
                    .or_insert(1);
            }
        }

        Self {
            inner,
            storagekey_to_handle: HashMap::new(),
            handle_usage_count,
        }
    }

    /// Returns the model's outputs handles.
    pub fn model_outputs(&self, graph: &ModelGraph<N>) -> anyhow::Result<Vec<TensorHandle<N>>> {
        let mut outputs = Vec::new();
        for (_edge_id, edge) in graph.edges() {
            let target = edge.target();
            let is_output = graph
                .node(target)
                .map(|node| node.is_output())
                .unwrap_or(false);
            if !is_output {
                continue;
            }

            let source = edge.source();
            for link in edge.ports().iter() {
                let storage_key = source.output_at(link.source_port).to_storage_key();
                let handle = self.get_by_storage_key(&storage_key)?;
                outputs.push(handle);
            }
        }

        Ok(outputs)
    }

    /// Returns a copy of the [TensorHandle<N>] corresponding to `storage_key`.
    fn get_by_storage_key(&self, storage_key: &StorageKey<Vec<N>>) -> Result<TensorHandle<N>> {
        self.storagekey_to_handle
            .get(storage_key)
            .with_context(|| format!("Missing tensor handle for {}", storage_key))
            .cloned()
    }

    /// Consumes this runner and returns the [I].
    pub fn into_inner(self) -> I {
        self.inner
    }
}

impl<I, N> LayerRunner<N, ()> for HandleLifetimeRunner<I, N>
where
    I: LayerRunner<N, RunInput<N>>,
    N: TensorTypeParam,
{
    fn model_inputs(
        &mut self,
        graph: &ModelGraph<N>,
        inputs: &[TensorHandle<N>],
    ) -> anyhow::Result<()> {
        // Counts how often a given tensor is used as an input to a layer.
        //
        // After each layer evaluation that uses the given tensor, the counter
        // is decrement. The tensor's cached data is freed once the data is no
        // longer needed.
        for (_edge_id, edge) in graph.edges() {
            let source = edge.source();

            for link in edge.ports().iter() {
                let storage_key = source.output_at(link.source_port).to_storage_key();
                self.handle_usage_count
                    .entry(storage_key)
                    .and_modify(|curr| *curr += 1)
                    .or_insert(1);
            }
        }

        for handle in inputs {
            let uses = self
                .handle_usage_count
                .get(handle.storage_key())
                .copied()
                .unwrap_or_default();

            // edge case: the input is not used by any layers
            if uses == 0 {
                warn!("Unused input tensor");
                handle.dry();
            }

            self.storagekey_to_handle.insert(
                handle.storage_key().clone(),
                handle.clone().wrapped_tensor_variant()?,
            );
        }

        self.inner.model_inputs(graph, inputs)?;

        Ok(())
    }

    fn run_layer(
        &mut self,
        node_id: NodeId,
        graph: &ModelGraph<N>,
        layer: &Layer<N>,
        _inputs: &(),
    ) -> RunResult<N>
    where
        Layer<N>: Evaluate<N>,
    {
        let input_storage_keys: Vec<StorageKey<Vec<N>>> = graph
            .incoming_feeds(node_id)
            .iter()
            .map(|feed| feed.source.to_storage_key())
            .collect::<Vec<_>>();

        let input_handles = input_storage_keys
            .iter()
            .map(|storage_key| self.get_by_storage_key(storage_key))
            .collect::<Result<Vec<_>>>()?;

        let inputs = RunInput { input_handles };
        let mut layer_out = self.inner.run_layer(node_id, graph, layer, &inputs)?;

        for storage_key in input_storage_keys.into_iter() {
            let uses = self
                .handle_usage_count
                .entry(storage_key.clone())
                .and_modify(|curr| *curr -= 1)
                .or_default();

            if *uses == 0 {
                self.get_by_storage_key(&storage_key)?.dry();
            }
        }

        for handle in layer_out.outputs.drain(0..) {
            let uses = self
                .handle_usage_count
                .get(handle.storage_key())
                .cloned()
                .unwrap_or_default();

            if uses == 0 {
                warn!("Unused output tensor");
                handle.dry();
            }

            // NOTE: the mapping may already have a tensor handle for the given
            // storage key, this is okay for LLM models which perform multiple
            // runs of the model and each run reuses the existing tensor storage
            // keys.
            self.storagekey_to_handle
                .insert(handle.storage_key().clone(), handle.clone());
        }

        Ok(layer_out)
    }
}

impl<N> ToStorageKey<Vec<N>> for NodeOutput {
    fn to_storage_key(&self) -> StorageKey<Vec<N>> {
        StorageKey::new(format!("{self}"))
    }
}

impl<N> ToStorageKey<Vec<N>> for NodeId {
    fn to_storage_key(&self) -> StorageKey<Vec<N>> {
        StorageKey::new(format!("{self}"))
    }
}

impl<N: TensorTypeParam> Node<Layer<N>> {
    pub fn describe(&self) -> String {
        match self {
            Node::Inner(layer) => layer.describe(),
            Node::Input(i) => format!("Input#{i}"),
            Node::Output(o) => format!("Output#{o}"),
        }
    }
}

/// Graph of layers. We store no weights on the edges.
/// TODO?: maybe make a graph wrapper that deals with empty weights
pub type ModelGraph<N> = Graph<Layer<N>, usize, usize, ()>;

/// Represents a model
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "N: Serialize", deserialize = "N: DeserializeOwned"))]
pub struct Model<N>
where
    N: TensorTypeParam,
{
    /// The graph-representation of the model
    ///
    /// NOTE: two very important conventions:
    ///
    ///   - model global inputs are represented by their own
    ///     `Node::Input(input_id)`, and expose their value on output port 0;
    ///
    ///   - model global outputs are represented by their own
    ///     `Node::Output(output_id)`, and sample their value on input port 0;
    pub(crate) graph: ModelGraph<N>,
    input_shapes: Vec<Shape>,
}

impl<N> Model<N>
where
    N: TensorTypeParam,
{
    /// Return an immutable reference to the underlying graph.
    pub fn graph(&self) -> &ModelGraph<N> {
        &self.graph
    }

    /// Return an immutable reference to the underlying graph.
    pub fn graph_mut(&mut self) -> &mut ModelGraph<N> {
        &mut self.graph
    }

    /// Consumes the [Model] and returns its [ModelGraph].
    pub fn into_graph(self) -> ModelGraph<N> {
        self.graph
    }

    pub fn set_input_shapes(&mut self, input_shapes: Vec<Shape>) {
        self.input_shapes = input_shapes;
    }

    /// Returns an iterator over the nodes in the model, in arbitrary order.
    /// It is more efficient then `ForwardIterator` and `BackwardIterator`, so it
    /// can be used to iterate over the nodes when the order does not matter
    pub fn to_unstable_iterator(&self) -> impl Iterator<Item = (&NodeId, &Node<Layer<N>>)> {
        self.graph.nodes()
    }

    /// Instantiate a model with the given input shapes (always unpadded).
    ///
    /// A corresponding number of input nodes is automatically generated.
    pub fn new_from_input_shapes(input_shapes: Vec<Shape>) -> Self {
        let mut graph = ModelGraph::new();
        for i in 0..input_shapes.len() {
            graph.add_input(i).unwrap();
        }

        Self {
            graph,
            input_shapes,
        }
    }

    pub(crate) fn new(input_shapes: Vec<Shape>, nodes: ModelGraph<N>) -> Self {
        let mut model = Self::new_from_input_shapes(input_shapes);
        model.graph = nodes;

        model
    }

    /// Instantiate a model from the set of nodes and the input shapes (always unpadded).
    pub fn new_from_shapes(input_shapes: Vec<Shape>, nodes: ModelGraph<N>) -> Self {
        Self {
            input_shapes,
            graph: nodes,
        }
    }

    /// Get the input shapes (always unpadded)
    pub fn input_shapes(&self) -> Vec<Shape> {
        self.input_shapes.clone()
    }

    /// Return the number of inputs this model expects.
    pub fn num_inputs(&self) -> usize {
        self.input_shapes.len()
    }

    /// Connect the provided input to the given node port.
    // TODO: will be superseded by the coming model builder
    pub fn connect_model_input(
        &mut self,
        input_idx: usize,
        target: NodeInput,
    ) -> anyhow::Result<()> {
        let input_node_id = self
            .graph
            .input_node_id(input_idx)
            .with_context(|| format!("retrieving node for input {input_idx}"))?;
        self.graph
            .add_edge(input_node_id, target.node_id, (0, *target.port), None)
            .map(|_| ())
    }

    /// Connect the provided inputs to `target` ports, from 0 up to the number
    /// of provided input IDs.
    // TODO: will be superseded by the coming model builder
    pub fn connect_model_inputs<I: IntoIterator<Item = usize>>(
        &mut self,
        input_idxs: I,
        target: NodeId,
    ) -> anyhow::Result<()> {
        for (i, input_id) in input_idxs.into_iter().enumerate() {
            self.connect_model_input(input_id, target.input_at(i))?;
        }
        Ok(())
    }

    /// Prepare the input tensors to be provided to the model according to the
    /// actual input shapes expected by the model
    pub fn prepare_inputs(&self, inputs: Vec<Tensor<N>>) -> Result<Vec<Tensor<N>>> {
        let input_shapes = self.input_shapes.clone();
        ensure!(
            input_shapes.len() == inputs.len(),
            "Unexpected number of inputs tensors: expected {}, found {}",
            input_shapes.len(),
            inputs.len()
        );
        inputs
            .into_iter()
            .zip(input_shapes)
            .map(|(mut input, shape)| {
                if input.shape() == &shape {
                    // no need to pad, simply return the input
                    Ok(input)
                } else {
                    input.pad_to_shape(shape)?;
                    Ok(input)
                }
            })
            .collect::<Result<Vec<_>>>()
    }

    /// Textual description of the model
    pub fn describe(&self) {
        info!("Model description:");
        info!("Input shapes: {:?}", self.input_shapes);
        info!(
            "Padded input shapes: {:?}",
            self.input_shapes.next_power_of_two(),
        );

        for (id, layer) in self.graph.forward_inners() {
            let edges = self
                .graph
                .neighbors(id, Direction::Any)
                .map(|(_, edge)| edge)
                .collect::<Vec<_>>();
            info!("\t- {}: {}", id, layer.describe());
            info!("\t\t- edges: {:?}", edges);
        }
        info!("Input nodes:");
        for (node_id, offset) in self.graph.input_nodes() {
            info!("\t- {}:{:?}", node_id, offset);
        }
        info!("Output nodes:");
        for (node_id, offset) in self.graph.output_nodes() {
            info!("\t- {}:{:?}", node_id, offset);
        }
    }

    /// iterates over all layers and resets their internal state if any
    pub fn reset(&self) {
        for (_, node) in self.graph.inner_nodes() {
            node.reset();
        }
    }

    /// iterates over all layers and resets their internal state if any,
    /// skipping any layers whose short name is in the provided list
    pub fn reset_selective(&self, to_skip: &[&str]) {
        for (_, node) in self.graph.inner_nodes() {
            if to_skip.contains(&node.short_name()) {
                continue;
            }
            node.reset();
        }
    }

    /// Build the inputs tensors, according to the expected input shapes,
    /// from a set of flat data
    pub fn load_input_flat(&self, input: Vec<Vec<N>>) -> Result<Vec<Tensor<N>>> {
        let input_tensor: Result<Vec<_>> = input
            .into_iter()
            .zip(self.input_shapes())
            .map(|(inp, shape)| Tensor::new(shape, inp))
            .collect();
        self.prepare_inputs(input_tensor?)
    }

    /// Add re-quantization nodes to the model after the node with id `input_node_id`
    /// It creates as many requant layers as there are output wires of the input node
    pub(crate) fn add_requant_layer(
        &mut self,
        requants: Vec<Requant>,
        input_node_id: NodeId,
    ) -> anyhow::Result<Vec<NodeId>> {
        ensure!(
            self.graph.node(input_node_id).is_some(),
            "Node {input_node_id} not found in the model"
        );
        // here we collect port links from the source port, since we add one requant _per source port_ only
        let source_edge_per_requant = self
            .graph
            .neighbors(input_node_id, Direction::Outgoing)
            .fold(BTreeMap::new(), |mut acc, (_, edge)| {
                for port in edge.ports().iter() {
                    acc.entry(port.source_port)
                        .or_insert(Vec::new())
                        .push((*edge.target(), port.target_port));
                }
                acc
            });
        // enforce one requant per source port
        ensure!(
            source_edge_per_requant.len() == requants.len(),
            "Unexpected number of requants: expected {}, found {}",
            source_edge_per_requant.len(),
            requants.len()
        );
        // we can already delete the outgoing edges from the input node now that we have collected all info necessary
        // to do the link with requants layers
        let edges_to_remove = self
            .graph
            .neighbors(input_node_id, Direction::Outgoing)
            .map(|(edge_id, _)| *edge_id)
            .collect::<Vec<_>>();
        for edge_id in edges_to_remove {
            self.graph.remove_edge(&edge_id)?;
        }

        let requant_nodes = source_edge_per_requant
            .into_iter()
            .zip(requants.into_iter())
            .map(|((source_port, targets), requant)| {
                // first add the  requant node to be able to  reference it later
                // when modifying the edges of the model
                let requant_node_id = self.graph.add_inner(Layer::Requant(requant))?;
                // we create this new port link as the edge from input node ->
                // requant. Given there is only **one** portlink on **one** edge
                // between input_node_id and this requant, we always set
                // target_port to 0, e.g. first slot.
                self.graph
                    .add_edge(input_node_id, requant_node_id, (*source_port, 0), None)?;
                // we create this new port link as the edge from requant ->
                // output. Here we wanna take exactly the same as the currently
                // existing ones, as if requant took the place of the input
                // node. Since source port can be connected to multiple target
                // ports and we can only insert a node _once_ then we index by
                // edge_id first.
                let portlinks_by_edge_id =
                    targets
                        .into_iter()
                        .fold(HashMap::new(), |mut acc, (target, target_port)| {
                            acc.entry(target).or_insert(Vec::new()).push(*target_port);
                            acc
                        });
                for (target, target_ports) in portlinks_by_edge_id.into_iter() {
                    // add all the port links from requant -> successor
                    let links = target_ports
                        .iter()
                        // Requant should always have one output port since it
                        // comes from a single source port on the node
                        .map(|target_port| (0, *target_port))
                        .collect::<Vec<_>>();
                    let edge = Edge::new(requant_node_id, target, links, None);
                    self.graph.add_edges_raw(vec![edge])?;
                }
                Ok(requant_node_id)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(requant_nodes)
    }

    pub fn num_outputs(&self, node_id: NodeId) -> anyhow::Result<usize> {
        let Some(node) = self.graph.node(node_id) else {
            anyhow::bail!("Node {node_id} not found in model");
        };
        Ok(match node {
            Node::Inner(layer) => {
                // how many targetports are attached to this node, e.g. how many inputs
                // does it receive
                let input_ports = self
                    .graph
                    .neighbors(node_id, Direction::Incoming)
                    .flat_map(|(_, edge)| edge.ports().iter())
                    .fold(HashSet::new(), |mut acc, port| {
                        acc.insert(port.target_port);
                        acc
                    })
                    .len();
                layer.num_outputs(input_ports)?
            }
            Node::Input(_) => 1,
            Node::Output(_) => 0,
        })
    }

    /// Corner-case method to add a node whose inputs correspond to the outputs of a node already inserted in the model
    /// The `NodeId` of the already inserted node is the `previous_node_id` input; if no id is provided, it is assumed
    /// that the inputs of the node correspond to the inputs of the model
    pub fn add_consecutive_layer(
        &mut self,
        layer: Layer<N>,
        previous_node_id: Option<NodeId>,
    ) -> anyhow::Result<NodeId> {
        // We need to correctly connect the outputs of the previous node to the inputs of the new node
        // For this we need to know how many outputs the previous node has
        // To know this, we need to count how many target ports are attached to the previous node, so the number
        // of inputs the previous node receives, and then call the `num_outputs` methods with that number.
        let num_outputs = if let Some(id) = previous_node_id {
            self.num_outputs(id)?
        } else {
            // look at the number of input nodes
            self.graph.input_nodes().count()
        };

        let new_node_id = self.graph.add_inner(layer)?;
        match previous_node_id {
            Some(id) => {
                // map i-th port of previous node to i-th port of new node
                let links = (0..num_outputs)
                    .map(|i| PortLink::new(i, i))
                    .collect::<Vec<_>>();
                self.graph.add_edge(id, new_node_id, links, None)?;
            }
            None => {
                let input_node_ids = self
                    .graph
                    .input_nodes()
                    .map(|(id, _)| id)
                    .collect::<Vec<_>>();
                for (i, input_node_id) in input_node_ids.into_iter().enumerate() {
                    self.graph
                        .add_edge(input_node_id, new_node_id, (0, i), None)?;
                }
            }
        };
        Ok(new_node_id)
    }

    /// Create a new output node for this graph, capturing the provided [`NodeOutput`].
    pub fn add_output(&mut self, output: NodeOutput, output_idx: usize) -> anyhow::Result<NodeId> {
        ensure!(self.graph.nodes().any(|(n_id, _)| *n_id == output.node_id));
        ensure!(
            self.graph.output_nodes().all(|(_, idx)| *idx != output_idx),
            "output {output_idx} already defined"
        );

        let new_node = self.graph.add_output(output_idx)?;
        self.graph
            .add_edge(output.node_id, new_node, (*output.port, 0), None)?;
        Ok(new_node)
    }

    pub fn add_edge<P: Into<Ports>>(
        &mut self,
        source: NodeId,
        target: NodeId,
        ports: P,
    ) -> anyhow::Result<()> {
        self.graph.add_edge(source, target, ports, None).map(|_| ())
    }

    pub fn add_raw_edge<S: Into<NodeId>, T: Into<NodeId>, P: Into<Ports>>(
        &mut self,
        source: S,
        target: T,
        ports: P,
    ) -> anyhow::Result<()> {
        let portlinks = ports.into();
        // a bit of weirdness when you don't have weights, you still need to specify the type of the weight
        let edge = Edge::new(source, target, portlinks, Option::<()>::None);
        self.graph.add_edges_raw(vec![edge]).map(|_| ())
    }

    // This method assumes there is a node without routed output edges, and the outputs of
    // this node will be labelled as the output edges of the model
    pub fn automatic_output_labelling(&mut self) -> Result<Vec<NodeId>> {
        ensure!(
            self.graph.output_nodes().count() == 0,
            "Model already has output nodes defined"
        );
        // find the nodes with no output edges, which will be considered the output nodes
        let out_node_ids = self
            .graph
            .sink_nodes()
            .filter(|node_id| self.graph[*node_id].is_inner())
            .collect::<Vec<_>>();
        ensure!(
            out_node_ids.len() == 1,
            "`automatic_output_labelling` method called on model with more than 1 output node"
        );
        let out_node = out_node_ids[0];
        // collect how many outputs it will produce and
        // set corresponding output edges
        let num_outputs = self.num_outputs(out_node).unwrap();
        (0..num_outputs)
            .map(move |i| self.add_output(out_node.output_at(i), i))
            .collect()
    }

    /// Returns the order the [NodeIds](NodeId) will be visited in a forward pass
    pub fn eval_order(&self) -> impl Iterator<Item = NodeId> + use<'_, N> {
        self.graph.forward_iter().map(|(id, _)| id)
    }

    /// Run a single iteration of the model using the provided `runner`.
    pub fn run_with_runner<R>(
        &self,
        runner: &mut R,
        inputs: Vec<TensorHandle<N>>,
    ) -> anyhow::Result<()>
    where
        Layer<N>: Evaluate<N>,
        R: LayerRunner<N, ()>,
    {
        let inputs = inputs
            .into_iter()
            .map(|handle| handle.wrapped_tensor_variant())
            .collect::<Result<Vec<_>, _>>()?;
        runner.model_inputs(&self.graph, &inputs)?;

        for (node_id, layer) in self.graph.forward_inners() {
            let span = info_span!(
                "zkml_layer_run",
                node_id = %node_id,
                op = layer.as_kind_str()
            );
            let _guard = span.enter();
            runner
                .run_layer(node_id, &self.graph, layer, &())
                .with_context(|| {
                    format!(
                        "Error occurred at node ID: {node_id}, Operation: {}",
                        layer.as_kind_str()
                    )
                })?;
        }

        Ok(())
    }

    pub(crate) fn run_with_split_nodes_info(
        &self,
        inputs: Vec<Tensor<N>>,
        store: &mut GenStore,
        split_nodes_info: Option<&SplittedNodesInfo>,
    ) -> anyhow::Result<Trace<N>>
    where
        Layer<N>: Evaluate<N>,
    {
        let span = info_span!(
            "zkml_model_run",
            inputs = inputs.len(),
            nodes = self.graph.inner_nodes().count()
        );
        let _guard = span.enter();
        let input_handles = tensor_to_handles(&inputs, &self.graph, store)?;

        let runner = BaseRunner {
            store: store.clone(),
            split_nodes_info,
        };
        #[cfg(test)]
        let runner = SanityCheckRunner { inner: runner };
        // store runner must be before trace and handle lifetime, since those
        // can dry the tensors
        let runner = StoreRunner::new(runner, store.clone());
        let runner = TraceRunner::new(runner, input_handles.clone());
        let mut runner = HandleLifetimeRunner::new(runner, &self.graph);

        self.run_with_runner(&mut runner, input_handles)?;

        let trace_runner = runner.into_inner();
        let (_store_runner, mut trace) = trace_runner.into_parts()?;

        if let Some(split_info) = split_nodes_info {
            trace.attach_split_info(split_info);
        }
        Ok(trace)
    }

    /// Performs a single run of the model, returning the produced trace.
    pub fn run(&self, inputs: Vec<Tensor<N>>, store: &mut GenStore) -> anyhow::Result<Trace<N>>
    where
        Layer<N>: Evaluate<N>,
    {
        self.run_with_split_nodes_info(inputs, store, None)
    }
}

impl Model<f32> {
    pub fn run_float(
        &self,
        inputs: Vec<Tensor<f32>>,
        store: &mut GenStore,
    ) -> Result<Vec<TensorHandle<f32>>> {
        let input_handles = tensor_to_handles(&inputs, &self.graph, store)?;

        let runner = BaseRunner {
            store: store.clone(),
            split_nodes_info: None,
        };
        #[cfg(test)]
        let runner = SanityCheckRunner { inner: runner };
        let mut runner = HandleLifetimeRunner::new(runner, &self.graph);
        self.run_with_runner(&mut runner, input_handles)?;

        runner.model_outputs(&self.graph)
    }
}

impl<'a> From<&'a Model<Element>> for ModelLayersRef<'a> {
    fn from(model: &'a Model<Element>) -> Self {
        model.graph.inner_nodes().collect()
    }
}

impl<'a> From<&'a Model<Element>> for ModelLayers {
    fn from(model: &'a Model<Element>) -> Self {
        model
            .graph
            .inner_nodes()
            .map(|(node_id, node)| (node_id, node.clone()))
            .collect()
    }
}

#[cfg(test)]
pub(crate) mod test {
    use std::ops::{Deref, Range};

    use super::Model;
    use crate::{
        Element, Prover, ScalingStrategy, Shape,
        graph::NodeOutput,
        init_test_logging, init_test_logging_default,
        layers::{
            Layer,
            activation::Activation,
            convolution::{ConvCtx, Convolution},
            einsum::EinSum,
            flatten::Flatten,
            pooling::{MAXPOOL2D_KERNEL_SIZE, Maxpool2D, Pooling},
            provable::{OpInfo, evaluate_layer},
        },
        measure::{self, Measure},
        padding::{PaddingMode, pad_model},
        quantization::{InferenceObserver, Quantize},
        rng_from_env_or_random,
        tensor::{KeyedTensor, Tensor, TensorHandle, TensorTypeParam},
        testing::Pcs,
        verify,
    };
    use anyhow::{Ok, Result};
    use ark_std::rand::{Rng, RngCore};

    use dp_crypto::arkyper::transcript::blake3::Blake3Transcript;
    use itertools::Itertools;

    use tenstore::{GenStore, StorageKey};

    pub type F = ark_bn254::Fr;
    const SELECTOR_DENSE: usize = 0;
    const SELECTOR_RELU: usize = 1;
    const SELECTOR_POOLING: usize = 2;
    const MOD_SELECTOR: usize = 2;

    pub type P<'a, 'b> = Prover<'a, 'b, F, T, Pcs>;

    impl Model<Element> {
        pub fn random(num_dense_layers: usize) -> Result<(Self, Vec<Tensor<Element>>)> {
            let mut rng = rng_from_env_or_random();
            Self::random_with_rng(num_dense_layers, &mut rng)
        }
        /// Returns a random model with specified number of dense layers and a matching input.
        /// Note that currently everything is considered padded, e.g. unpadded_shape = padded_shape
        pub fn random_with_rng<R: RngCore>(
            num_dense_layers: usize,
            rng: &mut R,
        ) -> Result<(Self, Vec<Tensor<Element>>)> {
            let mut last_row: usize = rng.gen_range(3..15);
            let mut model = Model::<f32>::new_from_input_shapes(vec![vec![last_row].into()]);

            let mut last_node_id = None;
            for selector in 0..num_dense_layers {
                if selector % MOD_SELECTOR == SELECTOR_DENSE {
                    // last row becomes new column
                    let (nrows, ncols): (usize, usize) = (rng.gen_range(3..15), last_row);
                    last_row = nrows;
                    let dense = EinSum::<f32>::random_dense(
                        vec![nrows, ncols].into(),
                        Some(format!("dense_{selector}").into()),
                    );

                    last_node_id =
                        Some(model.add_consecutive_layer(Layer::EinSum(dense), last_node_id)?);
                } else if selector % MOD_SELECTOR == SELECTOR_RELU {
                    last_node_id = Some(model.add_consecutive_layer(
                        Layer::Activation(Activation::new_relu()),
                        last_node_id,
                    )?);
                    // no need to change the `last_row` since RELU layer keeps the same shape
                    // of outputs
                } else if selector % MOD_SELECTOR == SELECTOR_POOLING {
                    // Currently unreachable until Model is updated to work with higher dimensional tensors
                    // TODO: Implement higher dimensional tensor functionality.
                    last_node_id = Some(model.add_consecutive_layer(
                        Layer::Pooling(Pooling::Maxpool2D(Maxpool2D::default())),
                        last_node_id,
                    )?);
                    last_row -= MAXPOOL2D_KERNEL_SIZE - 1;
                } else {
                    panic!("random selection shouldn't be in that case");
                }
            }
            model.automatic_output_labelling().unwrap();
            let inputs = model.input_shapes().iter().map(Tensor::random).collect();
            let (model, inputs) = quantize_model(model, inputs, None, &mut GenStore::default())?;
            let model = pad_model(model)?;
            let prepped_inputs = model.prepare_inputs(inputs)?;
            Ok((model, prepped_inputs))
        }

        pub fn random_with_matmul(
            num_linear_layers: usize,
            dim_size_range: Range<usize>,
        ) -> Result<(Self, Vec<Tensor<Element>>)> {
            let mut rng = rng_from_env_or_random();
            let (nrows, ncols): (usize, usize) = (
                rng.gen_range(dim_size_range.clone()),
                rng.gen_range(dim_size_range.clone()),
            );
            let mut model = Model::<f32>::new_from_input_shapes(vec![vec![nrows, ncols].into()]);
            let mut last_col = ncols;
            let mut last_node_id = None;
            for selector in 0..num_linear_layers {
                if selector % MOD_SELECTOR == SELECTOR_DENSE {
                    // last row becomes new column
                    let (nrows, ncols): (usize, usize) =
                        (last_col, rng.gen_range(dim_size_range.clone()));
                    last_col = ncols;
                    let weight = KeyedTensor::new(
                        format!("matmul_{selector}_weight"),
                        Tensor::random(&vec![nrows, ncols].into()),
                    );

                    let bias = KeyedTensor::new(
                        format!("matmul_{selector}_bias"),
                        Tensor::random(&vec![ncols].into()),
                    );
                    let linear_layer = EinSum::<f32>::new_matmul(
                        None,
                        Some(weight.into()),
                        false,
                        Some(bias.into()),
                    )?;

                    last_node_id = Some(
                        model.add_consecutive_layer(Layer::EinSum(linear_layer), last_node_id)?,
                    );
                } else if selector % MOD_SELECTOR == SELECTOR_RELU {
                    last_node_id = Some(model.add_consecutive_layer(
                        Layer::Activation(Activation::new_relu()),
                        last_node_id,
                    )?);
                    // no need to change the `last_col` since RELU layer keeps the same shape
                    // of outputs
                } else {
                    panic!("random selection shouldn't be in that case");
                }
            }
            model.automatic_output_labelling().unwrap();
            let inputs = model.input_shapes().iter().map(Tensor::random).collect();
            let (model, inputs) = quantize_model(model, inputs, None, &mut GenStore::default())?;
            let model = pad_model(model)?;
            let prepped_inputs = model.prepare_inputs(inputs)?;
            Ok((model, prepped_inputs))
        }

        /// Returns a model that only contains pooling and relu layers.
        /// The output [`Model`] will contain `num_layers` [`Maxpool2D`] layers and a [`Dense`] layer as well.
        pub fn random_pooling(num_layers: usize) -> Result<(Self, Vec<Tensor<Element>>)> {
            let mut rng = rng_from_env_or_random();
            // Since Maxpool reduces the size of the output based on the kernel size and the stride we need to ensure that
            // Our starting input size is large enough for the number of layers.

            // If maxpool input matrix has dimensions w x h then output has width and height
            // out_w = (w - kernel_size) / stride + 1
            // out_h = (h - kernel_size) / stride + 1
            // Hence to make sure we have a large enough tensor for the last step
            // we need to have that w_first > 2^{num_layers + 1} + 2^{num_layers}
            // and likewise for h_first.

            let minimum_initial_size = (1 << num_layers) * (3usize);

            let mut input_shape = (0..3)
                .map(|i| {
                    if i < 1 {
                        rng.gen_range(1..5usize).next_power_of_two()
                    } else {
                        (minimum_initial_size + rng.gen_range(1..4usize)).next_power_of_two()
                    }
                })
                .collect::<Shape>();

            let mut model = Model::<f32>::new_from_input_shapes(vec![input_shape.clone()]);

            let info = Maxpool2D::default();
            let mut last_node_id = None;
            for _ in 0..num_layers {
                println!("last node id: {:?}", last_node_id);
                input_shape
                    .iter_mut()
                    .skip(1)
                    .for_each(|dim| *dim = (*dim - info.kernel_size) / info.stride + 1);
                last_node_id = Some(model.add_consecutive_layer(
                    Layer::Pooling(Pooling::Maxpool2D(info)),
                    last_node_id,
                )?);
            }

            let (nrows, ncols): (usize, usize) =
                (rng.gen_range(3..15), input_shape.iter().product::<usize>());
            println!("Adding final dense layer of shape {nrows} x {ncols}");
            println!("Input shape before dense: {:?}", input_shape);
            println!("last node id: {:?}", last_node_id);
            last_node_id =
                Some(model.add_consecutive_layer(Layer::Flatten(Flatten(false)), last_node_id)?);
            model.add_consecutive_layer(
                Layer::EinSum(EinSum::random_dense(vec![nrows, ncols].into(), None)),
                last_node_id,
            )?;

            model.automatic_output_labelling()?;
            let inputs = model.input_shapes().iter().map(Tensor::random).collect();
            let (model, inputs) = quantize_model(model, inputs, None, &mut GenStore::default())?;
            let model = pad_model(model)?;
            let prepped_inputs = model.prepare_inputs(inputs)?;
            Ok((model, prepped_inputs))
        }
    }

    #[test]
    fn test_model_long() {
        let (model, inputs) = Model::random(3).unwrap();
        model.run(inputs, &mut Default::default()).unwrap();
    }

    #[test]
    fn test_conv_maxpool() {
        let input_shape: Shape = vec![3usize, 32, 32].into();
        let shape1: Shape = vec![6, 3, 5, 5].into();
        let filter = KeyedTensor::new("conv_filter", Tensor::random(&shape1));
        let bias1 = KeyedTensor::new("conv_bias", Tensor::random(&vec![shape1[0]].into()));

        let mut model = Model::new_from_input_shapes(vec![input_shape.clone()]);
        let conv_layer = model
            .add_consecutive_layer(
                Layer::Convolution(
                    Convolution::new(filter.clone(), bias1.clone())
                        .unwrap()
                        .prepared_for_fft(&input_shape)
                        .unwrap(),
                ),
                None,
            )
            .unwrap();
        let _pool_layer = model
            .add_consecutive_layer(
                Layer::Pooling(Pooling::Maxpool2D(Maxpool2D::default())),
                Some(conv_layer),
            )
            .unwrap();
        model.automatic_output_labelling().unwrap();

        // TODO: have a "builder" for the model that automatically tracks the shape after each layer such that
        // we can just do model.prepare_input(&input).
        // Here is not possible since we didnt run through the onnx loader
        let input = Tensor::random(&input_shape);
        let inputs = model.prepare_inputs(vec![input]).unwrap();

        let _ = model.run(inputs, &mut Default::default()).unwrap();
    }

    #[test]
    fn test_model_manual_run() {
        let dense1 = EinSum::<Element>::random_dense(
            vec![10usize.next_power_of_two(), 11usize.next_power_of_two()].into(),
            Some("dense_1".to_string().into()),
        );
        let dense2 = EinSum::<Element>::random_dense(
            vec![7usize.next_power_of_two(), 10usize.next_power_of_two()].into(),
            Some("dense_2".to_string().into()),
        );
        let input_shape = vec![11usize.next_power_of_two()].into();
        let input = Tensor::<Element>::random(&input_shape).into_wrapped();
        let output1 = evaluate_layer::<_, _>(&dense1, &[&input])
            .unwrap()
            .outputs()[0]
            .clone();
        let final_output = evaluate_layer::<_, _>(&dense2, &[&output1])
            .unwrap()
            .outputs()[0]
            .clone();

        let mut model = Model::<Element>::new_from_input_shapes(vec![input_shape]);
        let first_id = model
            .add_consecutive_layer(Layer::EinSum(dense1.clone()), None)
            .unwrap();
        let second_id = model
            .add_consecutive_layer(Layer::EinSum(dense2.clone()), Some(first_id))
            .unwrap();
        model.automatic_output_labelling().unwrap();

        let mut store = GenStore::default();
        let input = input.to_native();
        let trace = model.run(vec![input], &mut store).unwrap();
        assert_eq!(trace.steps.len(), 2);

        // Verify first step
        assert_eq!(
            trace
                .get_step(&first_id)
                .unwrap()
                .output_tensor_at(0)
                .unwrap()
                .deref(),
            &output1.to_native(),
        );

        // Verify second step
        assert_eq!(
            trace
                .get_step(&second_id)
                .unwrap()
                .output_tensor_at(0)
                .unwrap()
                .deref(),
            &final_output.to_native(),
        );

        assert_eq!(final_output.get_data().len(), 7usize.next_power_of_two());
    }

    #[test]
    fn test_single_cnn_prover() {
        measure::set_global(Measure::new());
        let n_w = 1 << 2;
        let k_w = 1 << 4;
        let n_x = 1 << 5;
        let k_x = 1 << 1;

        let in_dimensions: Vec<Vec<usize>> =
            vec![vec![k_x, n_x, n_x], vec![16, 29, 29], vec![4, 26, 26]];

        let conv1 = KeyedTensor::new(
            "conv_filter",
            Tensor::random(&vec![k_w, k_x, n_w, n_w].into()),
        );
        let input_shape = vec![k_x, n_x, n_x].into();

        let mut model = Model::new_from_input_shapes(vec![input_shape]);
        let input = Tensor::random(&model.input_shapes()[0]);
        let bias = KeyedTensor::new("conv_bias", Tensor::random(&vec![conv1.dim(0)].into()));
        let conv_layer = Convolution::new(conv1.clone(), bias.clone())
            .unwrap()
            .prepared_for_fft(&in_dimensions[0].clone().into())
            .unwrap();
        let _conv_layer_id = model
            .add_consecutive_layer(Layer::Convolution(conv_layer.clone()), None)
            .unwrap();

        assert_eq!(
            conv_layer.conv_context(),
            ConvCtx {
                kw: 16,
                kx: 2,
                real_nw: 4,
                nw: 32,
                filter_size: 1024,
                unpadded_filter_shape: Shape::new(vec![16, 2, 4, 4]),
                padded_filter_shape: Shape::new(vec![16, 2, 4, 4]),
                filter_key: conv1.commitment_id(),
                bias_key: bias.commitment_id(),
            },
        );

        model.automatic_output_labelling().unwrap();
        model.describe();
        let mut store = GenStore::default();
        let trace = model.run(vec![input], &mut store).unwrap();
        let (prover_ctx, verifier_ctx) = model
            .generate_contexts::<F, Pcs>()
            .expect("Unable to generate contexts");

        let (proof, io) = P::prove(&prover_ctx, trace, &model).expect("unable to generate proof");

        verify::<_, T, _>(&verifier_ctx, proof, io).unwrap();
        measure::to_csv("cnn_prover.csv").unwrap();
    }

    type T = Blake3Transcript;

    fn build_test_model<N: TensorTypeParam, const INPUT_SIZE: usize>() -> Model<N> {
        let input_shape: Shape = vec![INPUT_SIZE].into();
        let mut model = Model::<N>::new_from_input_shapes(vec![input_shape.clone()]);
        // add input dense layer
        // generate random dense matrix
        let ncols = input_shape[0];
        let nrows = 42;
        let dense = EinSum::<N>::random_dense(
            vec![nrows, ncols].into(),
            Some("dense_1".to_string().into()),
        );

        let dense_out_shape = &dense
            .output_shapes(&model.input_shapes(), PaddingMode::NoPadding)
            .unwrap()[0];
        let input_node = model
            .add_consecutive_layer(
                Layer::EinSum(dense),
                None, // it's connected to the inputs of the model
            )
            .unwrap();
        // add activation layer
        let relu = Activation::new_relu();
        let relu_node = model
            .add_consecutive_layer(Layer::Activation(relu), Some(input_node))
            .unwrap();
        // add another dense layer as output
        let nrows = 37;
        let ncols = dense_out_shape[0]; // it's a vector, so it has only one dimension
        let dense = EinSum::<N>::random_dense(
            vec![nrows, ncols].into(),
            Some("dense_2".to_string().into()),
        );
        let _ = model
            .add_consecutive_layer(Layer::EinSum(dense), Some(relu_node))
            .unwrap();
        let out_ids = model.automatic_output_labelling().unwrap();

        assert_eq!(model.graph.output_nodes().next().unwrap().0, out_ids[0]);

        model
    }

    #[test]
    fn test_model_inference() {
        const INPUT_SIZE: usize = 45;
        let float_model = build_test_model::<f32, INPUT_SIZE>();
        let input_shape = float_model.input_shapes()[0].clone();
        let float_input = Tensor::<f32>::random(&input_shape);
        let (model, input_tensor) = quantize_model(
            float_model,
            vec![float_input],
            None,
            &mut GenStore::default(),
        )
        .unwrap();

        let trace = model.run(input_tensor, &mut Default::default()).unwrap();
        // 5 steps: 2 dense layers, 1 relu layer, 2 requant layers
        assert_eq!(trace.steps.len(), 5);
    }

    #[test]
    fn test_model_float_inference() {
        const INPUT_SIZE: usize = 45;
        let model = build_test_model::<f32, INPUT_SIZE>();
        let input_shape = model.input_shapes()[0].clone();

        let input_tensor = Tensor::random(&input_shape);
        let trace = model
            .run(vec![input_tensor], &mut Default::default())
            .unwrap();
        assert_eq!(trace.steps.len(), 3);
    }

    // Quantize and run a model over the given input, if any; returns the quantized model and the
    // quantized inputs; if `represantive_inputs` are provided, they are going to be employed to
    // compute scaling factors for quantization, otherwise, random data will be employed
    pub(crate) fn quantize_model(
        model: Model<f32>,
        float_inputs: Vec<Tensor<f32>>,
        representative_inputs: Option<Vec<Tensor<f32>>>,
        store: &mut GenStore,
    ) -> anyhow::Result<(Model<Element>, Vec<Tensor<Element>>)> {
        let observer = if let Some(repr_inputs) = representative_inputs {
            InferenceObserver::new_with_representative_input(vec![
                repr_inputs
                    .iter()
                    .map(|input| input.data().to_vec())
                    .collect(),
            ])
        } else {
            InferenceObserver::new()
        };

        let (quantized_model, md) = observer.quantize(model, store)?;
        let input_tensors = float_inputs
            .into_iter()
            .enumerate()
            .map(|(i, data)| data.quantize(md.input_scaling(i)))
            .collect_vec();
        Ok((quantized_model, input_tensors))
    }

    pub(crate) fn prove_quantized_model(
        model: Model<Element>,
        inputs: Vec<Tensor<Element>>,
        store: &mut GenStore,
    ) -> anyhow::Result<Vec<TensorHandle<Element>>> {
        let model = pad_model(model)?;

        model.describe();

        let input_tensors = model.prepare_inputs(inputs).unwrap();

        let trace = model.run(input_tensors, store)?;
        let (prover_ctx, verifier_ctx) = model
            .generate_contexts::<F, Pcs>()
            .expect("Unable to generate contexts");
        let outputs = trace.outputs().to_vec();
        let (proof, io) = P::prove(&prover_ctx, trace, &model).expect("unable to generate proof");
        verify::<_, T, _>(&verifier_ctx, proof, io)?;
        Ok(outputs)
    }

    pub(crate) fn prove_model_with(
        model: Model<f32>,
        float_inputs: Vec<Tensor<f32>>,
        store: &mut GenStore,
    ) -> anyhow::Result<Vec<TensorHandle<Element>>> {
        let (quantized_model, quantized_inputs) =
            quantize_model(model, float_inputs.clone(), Some(float_inputs), store)?;
        prove_quantized_model(quantized_model, quantized_inputs, store)
    }

    pub(crate) fn prove_model(
        model: Model<f32>,
        store: &mut GenStore,
    ) -> anyhow::Result<Vec<TensorHandle<Element>>> {
        let float_inputs = model
            .input_shapes()
            .into_iter()
            .map(|shape| Tensor::random(&shape))
            .collect_vec();
        prove_model_with(model, float_inputs, store)
    }

    #[test]
    fn test_model_proving() {
        init_test_logging_default();
        const INPUT_SIZE: usize = 57;
        let model = build_test_model::<f32, INPUT_SIZE>();
        prove_model(model, &mut Default::default()).unwrap();
    }

    #[test]
    fn test_model_multiple_outputs() {
        init_test_logging("debug");
        const FIRST_INPUT_SIZE: usize = 27;
        const SECOND_INPUT_SIZE: usize = 49;
        let input_shapes = vec![
            vec![FIRST_INPUT_SIZE].into(),
            vec![SECOND_INPUT_SIZE].into(),
        ];
        let mut model = Model::<f32>::new_from_input_shapes(input_shapes);
        // add first dense layer
        // generate random dense matrix
        let ncols = FIRST_INPUT_SIZE;
        let nrows = 42;
        let dense = EinSum::<f32>::random_dense(
            vec![nrows, ncols].into(),
            Some("dense_1".to_string().into()),
        );
        let first_dense_out_shape = &dense
            .output_shapes(&[model.input_shapes()[0].clone()], PaddingMode::NoPadding)
            .unwrap()[0];
        let first_input_dense = model.graph.add_inner(Layer::EinSum(dense)).unwrap();
        // set that it will consume the first input
        model
            .connect_model_input(0, first_input_dense.input_at(0))
            .unwrap();

        // add second input dense layer
        let ncols = SECOND_INPUT_SIZE;
        let nrows = 47;
        let dense = EinSum::<f32>::random_dense(
            vec![nrows, ncols].into(),
            Some("dense_2".to_string().into()),
        );
        let second_dense_out_shape = &dense
            .output_shapes(&[model.input_shapes()[1].clone()], PaddingMode::NoPadding)
            .unwrap()[0];
        let second_input_dense = model.graph.add_inner(Layer::EinSum(dense)).unwrap();
        model
            .connect_model_input(1, second_input_dense.input_at(0))
            .unwrap();

        // add Relu nodes
        let relu = Activation::new_relu();
        let first_relu_node = model
            .add_consecutive_layer(Layer::Activation(relu.clone()), Some(first_input_dense))
            .unwrap();
        let second_relu_node = model
            .add_consecutive_layer(Layer::Activation(relu), Some(second_input_dense))
            .unwrap();
        // add other dense nodes
        let nrows = 52;
        let ncols = second_dense_out_shape[0]; // it's a vector, so it has only one dimension
        let dense = EinSum::<f32>::random_dense(
            vec![nrows, ncols].into(),
            Some("dense_out_1".to_string().into()),
        );
        let dense1 = model
            .add_consecutive_layer(Layer::EinSum(dense), Some(second_relu_node))
            .unwrap();
        let nrows = 17;
        let ncols = first_dense_out_shape[0];
        let dense = EinSum::<f32>::random_dense(
            vec![nrows, ncols].into(),
            Some("dense_out_2".to_string().into()),
        );
        let dense2 = model
            .add_consecutive_layer(Layer::EinSum(dense), Some(first_relu_node))
            .unwrap();

        let (first_output_node, second_output_node) = (
            model.add_output(dense1.output_at(0), 0).unwrap(),
            model.add_output(dense2.output_at(0), 1).unwrap(),
        );

        let out_node_ids = model
            .graph
            .output_nodes()
            .map(|(node_id, _)| node_id)
            .collect_vec();

        assert_eq!(out_node_ids.len(), 2);
        assert!(out_node_ids.contains(&first_output_node));
        assert!(out_node_ids.contains(&second_output_node));

        model.describe();

        prove_model(model, &mut Default::default()).unwrap();
    }

    #[test]
    fn test_model_with_multiple_inputs() {
        let input_shapes = vec![vec![6, 9].into(), vec![9, 13].into(), vec![11, 9].into()];

        let mut model = Model::new_from_input_shapes(input_shapes);

        // Add an input MatMul layer multiplying second with third input
        let first_input_node = model
            .graph
            .add_inner(Layer::EinSum(
                EinSum::new_matmul(None, None, false, None).unwrap(),
            ))
            .unwrap();
        model
            .connect_model_inputs([2, 1], first_input_node)
            .unwrap();

        // Add another input MatMul layer multiplying second with first input
        let second_input_node = model
            .graph
            .add_inner(Layer::EinSum(
                EinSum::new_matmul(None, None, false, None).unwrap(),
            ))
            .unwrap();
        model
            .connect_model_inputs([0, 1], second_input_node)
            .unwrap();

        // multiply the previous nodes
        let third = model
            .graph
            .add_inner(Layer::EinSum(
                EinSum::new_matmul(None, None, true, None).unwrap(),
            ))
            .unwrap();
        model.add_edge(first_input_node, third, (0, 0)).unwrap();
        // same shorter notation
        model.add_edge(second_input_node, third, (0, 1)).unwrap();
        model.automatic_output_labelling().unwrap();

        prove_model(model, &mut Default::default()).unwrap();
    }

    #[test]
    fn test_model_with_multiple_output_edges() {
        let input_shapes = vec![vec![7, 11].into(), vec![11, 13].into()];

        let mut model = Model::new_from_input_shapes(input_shapes);

        let matmul1 = model
            .graph
            .add_inner(Layer::EinSum(
                EinSum::new_matmul(None, None, false, None).unwrap(),
            ))
            .unwrap();
        let matmul2 = model
            .graph
            .add_inner(Layer::EinSum(
                EinSum::new_matmul(
                    None,
                    Some(TensorHandle::from_tensor(
                        StorageKey::from("first_out_weight"),
                        GenStore::new_empty(),
                        Tensor::random(&Shape::new(vec![13, 9])),
                    )),
                    false,
                    None,
                )
                .unwrap(),
            ))
            .unwrap();
        let matmul3 = model
            .graph
            .add_inner(Layer::EinSum(
                EinSum::new_matmul(
                    None,
                    Some(TensorHandle::from_tensor(
                        StorageKey::from("second_out_weight"),
                        GenStore::new_empty(),
                        Tensor::random(&Shape::new(vec![13, 13])),
                    )),
                    false,
                    None,
                )
                .unwrap(),
            ))
            .unwrap();
        model.connect_model_inputs([0, 1], matmul1).unwrap();

        model.add_edge(matmul1, matmul2, (0, 0)).unwrap();

        model.add_edge(matmul1, matmul3, (0, 0)).unwrap();

        // connect output of `matmul2` to the first output of the model
        model.add_output(NodeOutput::new(matmul2, 0), 0).unwrap();
        model.add_output(NodeOutput::new(matmul3, 0), 1).unwrap();

        prove_model(model, &mut Default::default()).unwrap();
    }

    #[test]
    fn test_model_with_duplicated_static_tensors() {
        // build a model with 2 MatMul layers sharing the same tensor
        let input_shape = vec![17, 14].into();
        let weight_shape = vec![14, 17].into();
        let bias_shape = vec![17].into();
        let matmul_weight = TensorHandle::from_tensor(
            StorageKey::new("matmul_weight"),
            GenStore::new_empty(),
            Tensor::random(&weight_shape),
        );
        let bias = TensorHandle::from_tensor(
            StorageKey::new("matmul_bias"),
            GenStore::new_empty(),
            Tensor::random(&bias_shape),
        );
        let mut model = Model::new_from_input_shapes(vec![input_shape]);
        let first_layer_id = model
            .add_consecutive_layer(
                Layer::EinSum(
                    EinSum::new_matmul(None, Some(matmul_weight.clone()), false, Some(bias))
                        .unwrap(),
                ),
                None,
            )
            .unwrap();
        let _ = model
            .add_consecutive_layer(
                Layer::EinSum(EinSum::new_matmul(None, Some(matmul_weight), true, None).unwrap()),
                Some(first_layer_id),
            )
            .unwrap();
        model.automatic_output_labelling().unwrap();

        prove_model(model, &mut Default::default()).unwrap();
    }
}
