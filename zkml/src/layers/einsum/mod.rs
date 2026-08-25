//! Einstein summation layer for tensor operations.
//!
//! This layer is built via an equation which in full generality looks like:
//!
//! > A(ijk)@B(ikl):C(himk):D(ik)->E(ijl)+BIAS(ij):F(hijm):G(ij)+BIAS(j)
//!
//! The equation is split in two by the arrow `->`, the left hand side of the
//! arrow corresponds to the einsum inputs, and right hand side the result of
//! the operation.
//!
//! In the equantion upper case identifiers represent tensors and lower case
//! their axes, the axes appear after a tensor name inside the parenthesis. The
//! optional keyword `BIAS` is reserved for bias tensors.
//!
//! The left hand side must have a single `@` separator, isolating the first input.
//! The input cannot be a constant tensor, it is followed by an arbitrary number
//! of tensos separated by `:` which are either constant or witness tensors.
//!
//! The right hand size is composed of an expressions with an optional `BIAS`
//! expression, the `BIAS` dimensions must be a subset of its addition pair. E.g
//! `Q(sh)` or with bias `Q(sh)+BIAS(h)`.
//!
//! It is important to note that the LHS tensor "A" cannot be a constant tensor.
//! In addition the contraction axes in the LHS and RHS tensors must appear
//! in the same order (i.e. if the contraction axes in the LHS are "ik" then
//! the contraction axes in the RHS must also be "ik", not "ki"), therefore
//! permutations are not supported. This is to ensure that the einsum operation
//! can be proven via Sumcheck.

use crate::{
    Claim, Element, NextPowerOfTwo, Shape,
    graph::NodeId,
    iop::{context::ContextAux, prover::Prover, verifier::Verifier},
    layers::{
        LayerCtx, LayerProof, ShapeStep,
        provable::{
            Evaluate, LayerOut, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp, QuantizeOutput,
            Splittable, VerifiableCtx,
        },
        transformer::ConcatenationCache,
    },
    model::Step,
    padding::{PaddingMode, pad_einsum},
    quantization::{ScalingFactor, ScalingStrategy},
    tensor::{CommitmentId, TensorHandle, TensorTypeParam, WrappedTensor},
};
use anyhow::{Result, anyhow, ensure};
use ark_ff::PrimeField;
use axis::{AxesMapping, AxisType, Dimension};
use dp_crypto::{
    Expression,
    arkyper::{CommitmentScheme, transcript::Transcript},
    structs::IOPProof,
};
use evaluate::EvaluationInformation3D;
use lazy_static::lazy_static;
use parking_lot::RwLock;
use prove::EinSumProofInfo;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::sync::{Arc, Mutex};
use tract_onnx::tract_hir::internal::num_integer::div_ceil;
use verify::EinSumVerifierInfo;

pub mod axis;
pub(crate) mod constructor;
pub(crate) mod evaluate;
pub(crate) mod op_info;
pub(crate) mod prove;
pub(crate) mod quantise;
pub(crate) mod verify;

/// Identifier for the EinSum layer.
pub(crate) const EINSUM_LAYER: &str = "EINS";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: DeserializeOwned"))]
pub struct EinSum<T>
where
    T: TensorTypeParam,
{
    /// The equation describing the einsum operation.
    pub(crate) equation: String,
    /// An optional name to give the einsum layer (for instance to identify it as an attention projection).
    pub(crate) name: Option<String>,
    /// The parsed mapping of axes from the equation.
    pub mapping: AxesMapping,
    /// The evaluation info for the einsum operation, this is derived from the mapping.
    pub evaluation_info: EvaluationInformation3D,
    /// The constant tensors to be used in the operation, if any.
    /// These correspond to the inputs in the equation that are not provided as inputs to the layer
    pub constant_tensors: Vec<Option<TensorHandle<T>>>,
    /// This vector holds the unpadded constant tensor shapes, if any.
    pub constant_unpadded_shapes: Vec<Option<Shape>>,
    /// The biases to be added after the einsum operation, if any.
    pub biases: Vec<Option<TensorHandle<T>>>,
    /// This vector holds the unpadded bias tensor shapes, if any.
    pub bias_unpadded_shapes: Vec<Option<Shape>>,
    /// used if the outputs of the einsum need to be cached
    pub caches: Vec<Option<Arc<Mutex<ConcatenationCache<T>>>>>,
    /// Flag to indicate if a requantisation step should be inserted after this layer
    pub(crate) requantise: bool,
}

impl<T> EinSum<T>
where
    T: TensorTypeParam,
{
    /// Create a new EinSum layer from the given equation.
    /// The equation should be in the format:
    ///
    /// "identifier_1"(axes1)@"identifier_2"(axes2):...:"identifier_n"(axes_n)->"Output_1"(output_axes_1):...:"Output_n-1"(output_axes_n-1)
    ///
    /// Where each identifier is a unique string of all uppercase letters(e.g. "A", "B", "WQ", etc.),
    /// and axes are strings of lowercase letters (e.g. "abc", "ij", etc.), there should be no spaces in the equation, only one identifier on the left hand side of "@" and
    /// a ":" between each of the tensors the LHS is acting on and each of the outputs.
    ///
    /// For example to specify a batched matrix multiplication between "A" and "B" and "A" and "C" producing outputs "X" and "Y", the equation would be:
    ///
    /// `A(ijm)@B(imk):C(iml)->X(ijk):Y(ijl)`
    ///
    /// Constant tensors and biases can be provided for inputs that are not given at runtime, the LHS of the equation is never a constant tensor.
    /// Currently we limit the number of inputs to be at most 4.
    pub fn new(
        equation: String,
        constant_tensors: Vec<Option<TensorHandle<T>>>,
        biases: Vec<Option<TensorHandle<T>>>,
    ) -> Result<Self> {
        let constant_tensors = constant_tensors
            .into_iter()
            .map(|constant_opt| {
                constant_opt
                    .map(|handle| handle.wrapped_tensor_variant())
                    .transpose()
            })
            .collect::<Result<Vec<Option<TensorHandle<T>>>>>()?;

        let biases = biases
            .into_iter()
            .map(|bias_opt| {
                bias_opt
                    .map(|handle| handle.wrapped_tensor_variant())
                    .transpose()
            })
            .collect::<Result<Vec<Option<TensorHandle<T>>>>>()?;

        let mapping: AxesMapping = AxesMapping::from_string(&equation)?;
        let evaluation_info = EvaluationInformation3D::new(&mapping)?;
        // Ensure the number of constant tensors and biases matches the number of inputs in the equation
        let input_count = mapping.input_count();
        let output_count = mapping.output_count();
        ensure!(
            constant_tensors.len() == input_count - 1,
            "Number of constant tensors ({}) does not match number of inputs in equation {equation} (expected: {} inputs)",
            constant_tensors.len(),
            input_count - 1,
        );
        ensure!(
            biases.len() == output_count,
            "Number of biases ({}) does not match number of outputs in equation {equation} (expected: {output_count} outputs)",
            biases.len(),
        );
        let actual_biases = biases.iter().filter(|b| b.is_some()).count();
        ensure!(
            actual_biases == mapping.bias_count(),
            "Number of biases ({actual_biases}) does not match number of expected biases in equation {equation} (expected: {} biases)",
            mapping.bias_count(),
        );
        ensure!(
            output_count == input_count - 1,
            "EinSum should have exactly one output for each einsum operation (i.e. number of inputs - 1), got {input_count} inputs and {output_count} outputs in equation {equation}"
        );

        // Currently we only support up to 4 inputs
        ensure!(
            input_count <= 4,
            "Currently we only support up to 4 inputs, got {input_count} in equation {equation}"
        );

        // Store the unpadded shapes of the constant tensors and biases
        let constant_unpadded_shapes = constant_tensors
            .iter()
            .map(|handle_opt| handle_opt.as_ref().map(|handle| handle.shape().clone()))
            .collect::<Vec<_>>();

        // Now we have to compute the bias shapes to ensure they are compatible with the output shapes
        let mut bias_id = 0usize;
        let biases = biases
            .into_iter()
            .enumerate()
            .map(|(output_id, bias)| {
                if let Some(mut bias) = bias {
                    let wrapped = bias.take_wrapped_tensor()?;
                    let new_shape = mapping.compute_new_bias_shape(
                        output_id,
                        bias_id,
                        &wrapped.shape().into(),
                    )?;
                    bias_id += 1;
                    let reshaped = wrapped.reshape(new_shape.into())?;
                    bias.set_wrapped_tensor(reshaped)?;
                    Ok(Some(bias))
                } else {
                    Ok(None)
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let bias_unpadded_shapes = biases
            .iter()
            .map(|handle_opt| handle_opt.as_ref().map(|handle| handle.shape().clone()))
            .collect::<Vec<_>>();

        Ok(Self {
            equation,
            name: None,
            mapping,
            evaluation_info,
            constant_tensors,
            constant_unpadded_shapes,
            biases,
            bias_unpadded_shapes,
            caches: vec![None; output_count],
            requantise: true,
        })
    }

    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    pub fn check_name(&self, name: &str) -> bool {
        if let Some(layer_name) = &self.name {
            layer_name == name
        } else {
            false
        }
    }

    pub fn with_caches(&mut self, concatenation_dims: Vec<Option<usize>>) -> Result<()> {
        ensure!(
            concatenation_dims.len() == self.mapping.output_count(),
            "Number of caches to use ({}) does not match number of outputs in equation {} (expected: {} outputs)",
            concatenation_dims.len(),
            self.equation,
            self.mapping.output_count()
        );

        let output_ranks = self.mapping.output_ranks();

        self.caches = concatenation_dims
            .into_iter()
            .zip(output_ranks.into_iter())
            .map(|(concat_dim_opt, output_rank)| {
                if let Some(concat_dim) = concat_dim_opt {
                    ensure!(
                        concat_dim < output_rank,
                        "Concatenation dimension ({}) must be less than output rank ({})",
                        concat_dim,
                        output_rank
                    );
                    Ok(Some(Arc::new(Mutex::new(ConcatenationCache::new(
                        output_rank,
                        concat_dim,
                    )))))
                } else {
                    Ok(None)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(())
    }

    /// Reset the caches used for concatenation.
    pub fn reset_caches(&self) {
        self.caches.iter().for_each(|cache| {
            if let Some(c) = cache {
                let mut c_lock = c.lock().unwrap();
                c_lock.reset();
            }
        });
    }

    /// Temporarily bypass all output caches without changing their static
    /// shape/concatenation configuration.
    pub fn set_caches_disabled(&self, disabled: bool) {
        self.caches.iter().for_each(|cache| {
            if let Some(cache) = cache {
                cache.lock().unwrap().set_disabled(disabled);
            }
        });
    }
    /// Getter for the requantisation flag.
    pub fn requantise(&self) -> bool {
        self.requantise
    }

    /// Set the requantisation flag to [`false`]
    pub fn disable_requantisation(mut self) -> Self {
        self.requantise = false;
        self
    }

    /// Disable requantisation after evaluation when quantised.
    pub fn no_requant(self) -> Self {
        Self {
            requantise: false,
            ..self
        }
    }
}

impl<N> Evaluate<N> for EinSum<N>
where
    N: TensorTypeParam,
{
    fn evaluate(&self, inputs: &[&WrappedTensor<N>]) -> Result<LayerOut<N>> {
        let outputs = self.evaluate_internal(inputs)?;
        Ok(LayerOut::from_vec(outputs))
    }
}

impl PadOp for EinSum<Element> {
    fn pad_node(self, si: &mut crate::padding::ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        pad_einsum(self, si)
    }
}

impl<N> OpInfo for EinSum<N>
where
    N: TensorTypeParam,
{
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        let mut input_shapes_iter = input_shapes.iter();
        // The left hand side of the equation cannot be a constant tensor, so we should always have at least one input shape provided
        let full_input_shapes = match padding_mode {
            PaddingMode::NoPadding => {
                let first_input_shape = input_shapes_iter
                    .next()
                    .cloned()
                    .expect("EinSum layer requires at least one input shape");
                std::iter::once(first_input_shape)
                    .chain(self.constant_unpadded_shapes.iter().map(|opt| {
                        if let Some(shape) = opt.as_ref() {
                            shape.clone()
                        } else {
                            input_shapes_iter
                                .next()
                                .cloned()
                                .expect("Not enough input shapes provided")
                        }
                    }))
                    .collect::<Vec<Shape>>()
            }
            PaddingMode::Padding => {
                let first_input_shape = input_shapes_iter
                    .next()
                    .expect("EinSum layer requires at least one input shape")
                    .next_power_of_two();

                std::iter::once(first_input_shape)
                    .chain(self.constant_unpadded_shapes.iter().map(|opt| {
                        if let Some(shape) = opt.as_ref() {
                            shape.next_power_of_two()
                        } else {
                            input_shapes_iter
                                .next()
                                .expect("Not enough input shapes provided")
                                .next_power_of_two()
                        }
                    }))
                    .collect::<Vec<Shape>>()
            }
        };

        let output_shapes = self.mapping.output_shapes(&full_input_shapes)?;

        output_shapes
            .into_iter()
            .zip(self.caches.iter())
            .map(|(shape, cache_opt)| {
                if let Some(cache) = cache_opt {
                    let c_lock = cache.lock().unwrap();
                    c_lock.next_shape(shape, padding_mode)
                } else {
                    Ok(shape)
                }
            })
            .collect::<Result<Vec<Shape>>>()
    }

    fn num_outputs(&self, _num_inputs: usize) -> Result<usize> {
        Ok(self.mapping.output_count())
    }

    fn describe(&self) -> String {
        format!("EinSum({})", self.equation)
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl ProveInfo for EinSum<Element> {
    fn step_info<F: PrimeField>(&self, aux: ContextAux) -> Result<(LayerCtx<F>, ContextAux)> {
        self.to_context(aux)
            .map(|(ctx, aux)| (LayerCtx::EinSum(ctx), aux))
    }
}

impl QuantizeOp for EinSum<f32> {
    type QuantizedOp = EinSum<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        _data: &S::AuxData,
        _node_id: NodeId,
        input_scaling: &[ScalingFactor],
        unpadded_input_shapes: &[Shape],
        output_scalings: &[ScalingFactor],
        _unpadded_output_shapes: &[Shape],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        ensure!(
            output_scalings.len() == self.mapping.output_count(),
            "Output scaling for EinSum layer different from {}",
            self.mapping.output_count()
        );

        self.quantise(input_scaling, output_scalings, unpadded_input_shapes)
    }
}

impl<F, PCS> ProvableOp<F, PCS> for EinSum<Element>
where
    F: PrimeField,
    PCS: CommitmentScheme<Field = F>,
{
    type Ctx = EinSumContext<F>;

    fn prove<T: Transcript>(
        &self,
        node_id: NodeId,
        ctx: &Self::Ctx,
        last_claims: Vec<&Claim<F>>,
        step_data: &Step<Element>,
        prover: &mut Prover<F, T, PCS>,
    ) -> Result<Vec<Claim<F>>> {
        let inputs = step_data.padded_input_tensors()?;

        let EinSumProofInfo {
            claims,
            proof,
            commitment_map,
        } = self.prove_internal(ctx, last_claims, &inputs, prover.transcript)?;

        // Add the proof to the proof list
        prover.push_proof(node_id, LayerProof::<F, PCS>::EinSum(proof));
        // Add the constant claims to the prover
        prover.add_common_claims(node_id, commitment_map);

        Ok(claims)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
/// Context for an [`EinSum`] layer. The context consists of:
/// - `node_id`: The unique identifier for the node.
/// - `equation`: The equation describing the einsum operation.
/// - `mapping`: The parsed mapping of axes from the equation.
/// - `constant_unpadded_shapes`: The unpadded shapes of the constant tensors used in the operation, if any.
/// - `bias_unpadded_shapes`: The unpadded shapes of the bias tensors used in the operation, if any.
/// - `einsum_sumcheck_expression`: The sumcheck expression for the einsum operation.
/// - `input_aggregation_expression`: The sumcheck expression for the input aggregation operation, this checks that the same tensor was used as the LHS for all einsum operations. It is `None` if there are only two inputs to the einsum operation.
pub struct EinSumContext<F: PrimeField> {
    pub equation: String,
    pub mapping: AxesMapping,
    pub constant_keys: Vec<Option<CommitmentId>>,
    pub constant_unpadded_shapes: Vec<Option<Shape>>,
    pub bias_keys: Vec<Option<CommitmentId>>,
    pub bias_unpadded_shapes: Vec<Option<Shape>>,
    pub input_aggregation_expression: Option<Expression<F>>,
    is_splittable: bool,
}

impl<F: PrimeField> OpInfo for EinSumContext<F> {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        let mut input_shapes_iter = input_shapes.iter();
        // The left hand side of the equation cannot be a constant tensor, so we should always have at least one input shape provided
        let full_input_shapes = match padding_mode {
            PaddingMode::NoPadding => {
                let first_input_shape = input_shapes_iter
                    .next()
                    .cloned()
                    .expect("EinSum layer requires at least one input shape");
                std::iter::once(first_input_shape)
                    .chain(self.constant_unpadded_shapes.iter().map(|opt| {
                        if let Some(shape) = opt.as_ref() {
                            shape.clone()
                        } else {
                            input_shapes_iter
                                .next()
                                .cloned()
                                .expect("Not enough input shapes provided")
                        }
                    }))
                    .collect::<Vec<Shape>>()
            }
            PaddingMode::Padding => {
                let first_input_shape = input_shapes_iter
                    .next()
                    .expect("EinSum layer requires at least one input shape")
                    .next_power_of_two();

                std::iter::once(first_input_shape)
                    .chain(self.constant_unpadded_shapes.iter().map(|opt| {
                        if let Some(shape) = opt.as_ref() {
                            shape.next_power_of_two()
                        } else {
                            input_shapes_iter
                                .next()
                                .expect("Not enough input shapes provided")
                                .next_power_of_two()
                        }
                    }))
                    .collect::<Vec<Shape>>()
            }
        };

        Ok(self
            .mapping
            .output_shapes(&full_input_shapes)
            .expect("Failed to compute output shapes for EinSum"))
    }

    fn num_outputs(&self, _num_inputs: usize) -> Result<usize> {
        Ok(self.mapping.output_count())
    }

    fn describe(&self) -> String {
        format!("EinSum({})", self.equation)
    }

    fn is_provable(&self) -> bool {
        true
    }
}

const DEFAULT_MIN_CHUNK_SIZE: usize = 128;

lazy_static! {
    pub(crate) static ref MIN_CHUNK_SIZE: RwLock<Option<usize>> = RwLock::new(None);
}

impl<F: PrimeField> Splittable for EinSumContext<F> {
    fn ideal_num_chunks(&self, input_shapes: &[Shape]) -> Option<usize> {
        if !self.is_splittable {
            return None;
        }
        assert_eq!(input_shapes.len(), 1); // for now the layer is splittable only if the RHS are all constant tensors, so there must be only 1 input
        let min_chunk_size = MIN_CHUNK_SIZE
            .read_recursive()
            .unwrap_or(DEFAULT_MIN_CHUNK_SIZE);
        let seq_len = input_shapes[0].dim(0);
        Some(div_ceil(seq_len, min_chunk_size))
    }

    fn is_splittable(&self) -> bool {
        self.is_splittable
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
/// Proof for an [`EinSum`] layer. The proof consists of:
/// - `bias_evals`: The evaluations of the bias polynomials at the random challenge points, this vec can be empty if there are no bias tensors.
/// - `einsum_sumcheck`: The sumcheck proof for the einsum operation.
/// - `einsum_evaluations`: The evaluations of the einsum polynomials at the random challenge point produced by the einsum sumcheck.
/// - `input_aggregation_sumcheck`: The sumcheck proof for the input aggregation operation, this checks that the same tensor was used as the LHS for all einsum operations.
pub struct EinSumProof<F: PrimeField> {
    /// Claimed bias evaluations, one for each bias tensor, can be empty if there are no bias tensors.
    #[serde(with = "dp_crypto::serialization")]
    bias_evals: Vec<F>,
    /// Sumcheck proof for the equation specified in the layer.
    einsum_sumcheck: IOPProof<F>,
    /// Evaluations of the polynomials used in the einsum sumcheck, the first `n` of these correspond to the LHS polynomial evaluations, where `n` is the number of einsum operations (i.e. number of inputs - 1 including constant tensors).
    #[serde(with = "dp_crypto::serialization")]
    einsum_evaluations: Vec<F>,
    /// Sumcheck proof for the input aggregation, this checks that the same tensor was used as the LHS for all `n` einsum operations.
    input_aggregation_sumcheck: Option<IOPProof<F>>,
}

impl<F: PrimeField, PCS: CommitmentScheme<Field = F>> VerifiableCtx<F, PCS> for EinSumContext<F> {
    type Proof = EinSumProof<F>;

    fn verify<T: Transcript>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<F>],
        verifier: &mut Verifier<F, T, PCS>,
        shape_step: &ShapeStep,
        node_id: NodeId,
    ) -> Result<Vec<Claim<F>>> {
        // Run the internal method to verify the proof
        let EinSumVerifierInfo {
            claims,
            constants_map,
        } = self.verify_internal(proof, last_claims, shape_step, verifier.transcript)?;
        // Add the constant claims to the verifier
        verifier.add_common_claims(node_id, constants_map);
        Ok(claims)
    }

    fn write_proof_to_transcript<T: Transcript>(
        &self,
        _proof: &Self::Proof,
        _transcript: &mut T,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {

    use dp_crypto::arkyper::transcript::blake3::Blake3Transcript;
    use tenstore::{GenStore, StorageKey};

    use crate::{
        Tensor,
        layers::Layer,
        model::{
            Model,
            test::{F, P, prove_model, quantize_model},
        },
        padding::pad_model,
        testing::Pcs,
    };

    type T = Blake3Transcript;
    use super::*;
    use crate::verify;

    #[test]
    fn test_einsum_larger_context() -> anyhow::Result<()> {
        let [a, b, c] = [3, 16, 32];
        let max_stack = 8;
        let runtime_stack = 4;
        let mut model = Model::new_from_input_shapes(vec![vec![max_stack, a, b].into()]);
        let weight = TensorHandle::from_tensor(
            StorageKey::from("weight"),
            GenStore::new_empty(),
            Tensor::random(&Shape::new(vec![b, c])),
        );
        let einsum = EinSum::new(
            "A(ijk)@B(kl)->C(ijl)".to_string(),
            vec![Some(weight)],
            vec![None],
        )
        .unwrap();
        let _ = model
            .add_consecutive_layer(Layer::EinSum(einsum), None)
            .unwrap();
        model.automatic_output_labelling().unwrap();
        model.describe();
        // that _should_ work since it is a smaller stack than the model
        let float_inputs = vec![Tensor::random(&vec![runtime_stack, a, b].into())];

        let mut store = GenStore::default();
        let (quantized_model, quantized_inputs) =
            quantize_model(model, float_inputs, None, &mut store)?;
        let mut padded_model = pad_model(quantized_model)?;

        padded_model.set_input_shapes(quantized_inputs.iter().map(|t| t.shape().clone()).collect());
        let input_tensors = padded_model
            .prepare_inputs(quantized_inputs.clone())
            .unwrap();
        let trace = padded_model.run(input_tensors, &mut store)?;
        let (prover_ctx, verifier_ctx) = padded_model
            .generate_contexts::<F, Pcs>()
            .expect("Unable to generate contexts");

        let (proof, io) =
            P::prove(&prover_ctx, trace, &padded_model).expect("unable to generate proof");

        verify::<_, T, _>(&verifier_ctx, proof, io)?;
        // prove_model(model, &mut GenStore::default()).unwrap();
        Ok(())
    }

    #[test]
    fn test_einsum_proving_with_bias_and_transpose() {
        let [a, b, d] = [300, 350, 256];
        let first_input_shape = vec![a, b];
        // since we transpose B
        let second_input_shape = vec![d, b];
        let mut model =
            Model::new_from_input_shapes(vec![first_input_shape.into(), second_input_shape.into()]);
        let bias = Tensor::<f32>::random(&vec![d].into());
        let keyed_bias =
            TensorHandle::from_tensor(StorageKey::from("bias1"), GenStore::new_empty(), bias);
        let einsum = EinSum::new(
            "A(ij)@B(kj)->C(ik)+BIAS(k)".to_string(),
            vec![None],
            vec![Some(keyed_bias)],
        )
        .unwrap();
        let _ = model
            .add_consecutive_layer(Layer::EinSum(einsum), None)
            .unwrap();
        model.automatic_output_labelling().unwrap();
        model.describe();
        prove_model(model, &mut Default::default()).unwrap();
    }

    #[test]
    fn test_proven_concat_matmul_einsum() {
        // we test over a model where concat matmul is the first layer, so we need 2 input shapes
        let input_shape_left = vec![5, 14, 27].into();
        let input_shape_right = vec![5, 27, 18].into();

        let mut model = Model::new_from_input_shapes(vec![input_shape_left, input_shape_right]);
        let einsum =
            EinSum::new("A(ijk)@B(ikl)->C(ijl)".to_string(), vec![None], vec![None]).unwrap();

        let _id = model
            .add_consecutive_layer(Layer::EinSum(einsum), None)
            .unwrap();
        model.automatic_output_labelling().unwrap();
        model.describe();
        let outputs = prove_model(model, &mut GenStore::default()).unwrap();

        // check output shape
        assert_eq!(*outputs[0].shape(), Shape::new(vec![5, 14, 18]));
    }

    #[test]
    fn test_proven_broadcasted_bias_einsum() {
        // we test over a model where concat matmul is the first layer, so we need 2 input shapes
        let input_shape_left = vec![5, 14, 27].into();
        let input_shape_right = vec![5, 27, 18].into();

        let mut model = Model::new_from_input_shapes(vec![input_shape_left, input_shape_right]);
        let bias = TensorHandle::from_tensor(
            StorageKey::new("qkv_bias.q"),
            GenStore::new_empty(),
            Tensor::random(&Shape::new(vec![5, 18])),
        );
        let einsum = EinSum::new(
            "A(ijk)@B(ikl)->C(ijl)+BIAS(il)".to_string(),
            vec![None],
            vec![Some(bias)],
        )
        .unwrap();

        let _id = model
            .add_consecutive_layer(Layer::EinSum(einsum), None)
            .unwrap();
        model.automatic_output_labelling().unwrap();
        model.describe();
        let outputs = prove_model(model, &mut GenStore::default()).unwrap();

        // check output shape
        assert_eq!(*outputs[0].shape(), Shape::new(vec![5, 14, 18]));
    }

    #[test]
    fn test_proven_qkv_einsum() {
        let num_inputs = 49;
        let embedding_size = 78;
        let hidden_size = 120;

        let input_shape = vec![num_inputs, embedding_size].into();

        let q = TensorHandle::from_tensor(
            StorageKey::new("qkv_weight.q"),
            GenStore::new_empty(),
            Tensor::random(&vec![embedding_size, hidden_size].into()),
        );
        let q_bias = TensorHandle::from_tensor(
            StorageKey::new("qkv_bias.q"),
            GenStore::new_empty(),
            Tensor::random(&vec![hidden_size].into()),
        );
        let k = TensorHandle::from_tensor(
            StorageKey::new("qkv_weight.k"),
            GenStore::new_empty(),
            Tensor::random(&vec![embedding_size, hidden_size].into()),
        );
        let k_bias = TensorHandle::from_tensor(
            StorageKey::new("qkv_bias.k"),
            GenStore::new_empty(),
            Tensor::random(&vec![hidden_size].into()),
        );
        let v = TensorHandle::from_tensor(
            StorageKey::new("qkv_weight.v"),
            GenStore::new_empty(),
            Tensor::random(&vec![embedding_size, hidden_size].into()),
        );
        let v_bias = TensorHandle::from_tensor(
            StorageKey::new("qkv_bias.v"),
            GenStore::new_empty(),
            Tensor::random(&vec![hidden_size].into()),
        );

        let einsum_layer = EinSum::<f32>::new(
            "X(se)@WQ(eh):WK(eh):WV(eh)->Q(sh)+BIAS(h):K(sh)+BIAS(h):V(sh)+BIAS(h)".to_string(),
            vec![Some(q), Some(k), Some(v)],
            vec![Some(q_bias), Some(k_bias), Some(v_bias)],
        )
        .unwrap();
        let mut model = Model::<f32>::new_from_input_shapes(vec![input_shape]);

        let _einsum_node_id = model
            .add_consecutive_layer(Layer::EinSum(einsum_layer), None)
            .unwrap();

        model.automatic_output_labelling().unwrap();
        model.describe();
        prove_model(model, &mut GenStore::default()).unwrap();
    }
}
