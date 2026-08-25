use crate::{
    parser::{
        Load,
        llm::{
            metadata::LLMMetadata,
            models::{LLMModelLoader, gpt2::decoder::GPT2Decoder},
        },
    },
    tensor::KeyedTensor,
};

use crate::{
    Shape,
    layers::transformer::attention_mask::AttentionSpan,
    model::Model,
    parser::{
        ModelLoader,
        gguf::RawGGUF,
        json,
        json::RawJSON,
        llm::{
            HFTokenizer, LLMConfig, LLMIR,
            config::{AttentionConfig, AttentionHeadType, LLMStructure, PositionalConfig},
            tokenizer::TokenizerLoader,
            transformer::NormType,
        },
        safe,
        safe::RawSafeTensors,
    },
};
use anyhow::Context;
use tenstore::StorageKey;

pub mod decoder;

/// Loader for the GPT2 family of models.
/// For more information about GPT2, see
/// https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
#[derive(Clone, Debug, Default, Copy)]
pub struct GPT2 {
    /// Current hack to avoid committing to huge positional matrix
    max_ctx_length: Option<usize>,
}

impl GPT2 {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<DataFormat> LLMModelLoader<DataFormat> for GPT2
where
    GPT2: ModelLoader<DataFormat, Metadata = LLMMetadata>,
{
    fn with_max_context_length(self, max_ctx_length: usize) -> Self
    where
        Self: Sized,
    {
        Self {
            max_ctx_length: Some(max_ctx_length),
        }
    }
}

pub const GPT2_VARIANTS: &[&str] = &[
    "gpt2",
    "Tmkrzx_X",
    "distilgpt2",
    "toy_gpt2",
    "sshleifer/tiny-gpt2",
];

pub const GPT2_NAME: &str = "gpt2";
pub const GPT2_GGUF_NAME: &str = GPT2_NAME;
pub const GPT2_Q8_0: &str = "gpt2.Q8_0.gguf";
pub const GPT2_SAFE_MODEL: &str = "openai-community/gpt2";

pub fn is_gpt2_model(names: &[String]) -> bool {
    names
        .iter()
        .any(|name| GPT2_VARIANTS.contains(&name.as_str()))
}

impl TokenizerLoader<RawGGUF> for GPT2 {
    fn load_tokenizer(&self, raw: &RawGGUF) -> anyhow::Result<HFTokenizer> {
        let loader = raw.loader()?;
        let tokenizer = HFTokenizer::bpe_from_gguf(&loader)?;
        Ok(tokenizer)
    }
}

impl TokenizerLoader<RawSafeTensors> for GPT2 {
    fn load_tokenizer(&self, raw: &RawSafeTensors) -> anyhow::Result<HFTokenizer> {
        let tokenizer = HFTokenizer::from_tokenizer_json_path(raw.tokenizer_path())?;
        Ok(tokenizer)
    }
}

impl ModelLoader<RawGGUF> for GPT2 {
    type Metadata = LLMMetadata;

    fn model_name(&self) -> String {
        GPT2_NAME.to_string()
    }

    fn parse(&self, raw: &RawGGUF) -> anyhow::Result<(Model<f32>, Self::Metadata)> {
        let loader = raw.loader()?;
        let config = LLMConfig::from_gguf(&loader, "gpt2", self.max_ctx_length)?;
        let structure = gpt2_structure(&config);
        let model = LLMIR::<GPT2Decoder>::from_loader(&loader, &structure)?;
        // even though the llm runtime doesn't care about the model input shape, which is designed for "static" input shapes, we still
        // need to provide one.
        let init_user_shape = Shape::from(vec![1]);
        model.into_model(config, init_user_shape)
    }
}

impl ModelLoader<RawJSON> for GPT2 {
    type Metadata = LLMMetadata;

    fn model_name(&self) -> String {
        GPT2_NAME.to_string()
    }

    fn parse(&self, raw: &RawJSON) -> anyhow::Result<(Model<f32>, Self::Metadata)> {
        let loader = json::FileTensorLoader::new_from_path(&raw.0)?;
        let config = LLMConfig::from_json(&loader, None)?;
        let model = LLMIR::<GPT2Decoder>::from_loader(&loader, &config)?;
        let init_user_shape = Shape::from(vec![1]);
        model.into_model(config, init_user_shape)
    }
}

impl ModelLoader<RawSafeTensors> for GPT2 {
    type Metadata = LLMMetadata;

    fn model_name(&self) -> String {
        GPT2_NAME.to_string()
    }

    /// Load GPT-2 model from HuggingFace SafeTensors format.
    ///
    /// # Reference Implementation
    /// This loader is compatible with models from HuggingFace Hub (e.g., "openai-community/gpt2").
    /// It matches the architecture from:
    /// https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
    ///
    /// # Key Differences from Generic LLM Loader
    /// GPT-2 uses different tensor naming conventions than other models (e.g., Gemma3):
    /// - Transformer blocks: `transformer.h.{i}.*` instead of `model.layers.{i}.*`
    /// - Token embeddings: `transformer.wte.weight` instead of `model.embed_tokens.weight`
    /// - Position embeddings: `transformer.wpe.weight` instead of `model.positional.*`
    /// - Final norm: `transformer.ln_f.*` instead of `model.norm.*`
    ///
    /// # Weight Format: Conv1D
    /// HuggingFace GPT-2 uses `Conv1D` layers (custom linear layer with transposed weight storage).
    /// Conv1D stores weights as (in_features, out_features) and computes `output = input @ weight`.
    /// This differs from PyTorch's `nn.Linear` which stores (out_features, in_features) and computes
    /// `output = input @ weight.T`.
    fn parse(&self, raw: &RawSafeTensors) -> anyhow::Result<(Model<f32>, Self::Metadata)> {
        use crate::{
            layers::{
                einsum::EinSum,
                transformer::{
                    embeddings::Embeddings, normalisation::layernorm::LayerNorm,
                    positional::Positional,
                },
            },
            parser::llm::transformer::Norm,
        };

        let cfg = raw.read_config_json()?;

        let hidden_size = cfg.get("n_embd").context("n_embd not found")?;
        let num_heads = cfg.get("n_head").context("n_head not found")?;
        let context_length = cfg.get("n_ctx").context("n_ctx not found")?;
        let num_block = cfg.get("n_layer").context("n_layer not found")?;
        let norm_epsilon = cfg
            .get("layer_norm_epsilon")
            .context("layer_norm_epsilon not found")?;
        let embedding_size = hidden_size;
        let vocab_size = cfg.get("vocab_size").context("vocab_size not found")?;
        let eos_token: usize = cfg.get("eos_token_id").context("eos_token_id not found")?;

        // The GPT2 SafeTensors ConfigJSON does not contain intermediate size,
        // So we load the first feed-forward layer bias to determine it.
        let loader = safe::FileTensorLoader::from_path(raw.model_path())?;

        let intermediate_size = loader
            .get_tensor("h.0.mlp.c_fc.bias")
            .map(|t| t.shape().numel())
            .context("Could not load first FFN bias in GPT2")?;

        let llm_config = LLMConfig {
            model_name: "gpt2".to_string(),
            hidden_size,
            embedding_size,
            intermediate_size,
            num_heads,
            head_size: hidden_size / num_heads,
            num_block,
            context_length: if let Some(max_context) = self.max_ctx_length {
                max_context
            } else {
                context_length
            },
            norm_epsilon,
            vocab_size,
            eos_token: eos_token.into(),
        };

        let structure = gpt2_structure(&llm_config);

        // GPT-2 SafeTensors use "transformer.h.{i}." prefix instead of "model.layers.{i}."
        // so we manually load all components with the correct prefixes

        // Load embeddings from transformer.wte.weight
        // from_safetensors_loader() already knows to look for "transformer.wte.weight"
        let embeddings = Embeddings::from_safetensors_loader(&loader)?;

        // Load positional encoding from transformer.wpe.weight
        let global_positional = structure
            .global_positional
            .as_ref()
            .map(|p| Positional::from_safetensors_loader(&loader, &structure, p))
            .transpose()?;

        // Load transformer blocks using "transformer.h.{i}." prefix
        let num_layers = structure.generic.num_block;
        let blocks = (0..num_layers)
            .map(|i| {
                GPT2Decoder::from_loader(
                    &loader.pp(&format!("h.{i}.")),
                    &(structure.clone(), cfg.clone()),
                )
            })
            .collect::<anyhow::Result<Vec<GPT2Decoder>>>()?;

        // Load final norm from transformer.ln_f.weight/bias
        let ln_f_weight = loader.get_tensor("ln_f.weight")?;
        let ln_f_bias = loader.get_tensor("ln_f.bias")?;
        let eps = cfg
            .get::<f32, _>("layer_norm_epsilon")
            .context("layer_norm_epsilon not found")?;
        let final_norm =
            Norm::LayerNorm(LayerNorm::new(ln_f_weight.into(), ln_f_bias.into(), eps)?);

        // Final projection (reuses embeddings matrix)
        let input_terms = "X(se)@WE(ve)";
        let proj_bias = loader.get_tensor("output.bias").ok();
        let output_terms = if proj_bias.is_some() {
            "O(sv)+BIAS(v)"
        } else {
            "O(sv)"
        };
        let equation = format!("{input_terms}->{output_terms}");
        let mut proj_weights = KeyedTensor::try_from(&embeddings.mat)?;
        // Embeddings and the final projection start from tied GPT-2 weights, but are
        // quantized at different scales. Give the projection its own commitment key,
        // matching the GGUF loader, so distinct polynomials cannot share an ID.
        proj_weights.key = StorageKey::from(format!("{}_final_proj", proj_weights.storage_key()));
        let final_proj = EinSum::<f32>::new(
            equation,
            vec![Some(proj_weights.into())],
            vec![proj_bias.map(|tensor| tensor.into())],
        )?;

        let llm_model = LLMIR::new(
            embeddings,
            global_positional,
            blocks,
            final_norm,
            final_proj,
        );

        let init_user_shape = Shape::from(vec![1]);
        llm_model.into_model(llm_config, init_user_shape)
    }
}

pub(crate) fn gpt2_structure(config: &LLMConfig) -> LLMStructure {
    LLMStructure {
        generic: config.clone(),
        norm_type: NormType::LayerNorm,
        global_positional: Some(PositionalConfig::FixedPositional),
        attention_config: AttentionConfig {
            span: (1..=config.num_block)
                .map(|_| AttentionSpan::Full)
                .collect(),
            head: AttentionHeadType::MHA,
        },
    }
}

#[cfg(test)]
pub mod tests {

    use tenstore::GenStore;

    use super::*;
    use crate::{
        Tensor,
        parser::{file_cache, llm::LLMTokenizer, safe::RawSafeTensors},
    };

    pub const GPT2_Q8_0: &str = "gpt2.Q8_0.gguf";

    pub const GPT2_SAFE_MODEL: &str = "openai-community/gpt2";

    #[test]
    fn test_gpt2_load_gguf_tokenizer() -> anyhow::Result<()> {
        let model_path = file_cache::from_cache(GPT2_Q8_0)?;
        let mygguf = RawGGUF::new(model_path);
        let tokenizer = GPT2::new().load_tokenizer(&mygguf)?;
        let s = "do or don't. there is no try.";
        let tokens = tokenizer.tokenize(s);
        let s2 = tokenizer.detokenize(&tokens);
        assert_eq!(s, s2);
        Ok(())
    }

    #[test]
    fn test_gpt2_load_gguf_model() -> anyhow::Result<()> {
        let model_path = file_cache::from_cache(GPT2_Q8_0)?;
        let mygguf = RawGGUF::new(model_path);
        let gpt2 = GPT2::new();
        let (model, md) = gpt2.parse(&mygguf)?;
        let config = md.config;
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.num_block, 12);
        assert_eq!(config.embedding_size, 768);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.context_length, 1024);
        assert_eq!(config.norm_epsilon, 1e-5);
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.eos_token, 50256usize.into());
        let input = Tensor::new(vec![1].into(), vec![546.0f32])?;
        model.run_float(vec![input], &mut GenStore::default())?;
        Ok(())
    }

    #[test]
    fn test_safe_gpt2_load_tokeniser() -> anyhow::Result<()> {
        let raw = RawSafeTensors::from_hugging_face_cached(GPT2_SAFE_MODEL)?;
        let tokeniser = GPT2::new().load_tokenizer(&raw)?;
        let tokens = tokeniser.tokenize("Hello, world!");
        let s = tokeniser.detokenize(&tokens);
        assert_eq!(s, "Hello, world!");

        Ok(())
    }

    #[test]
    fn test_safe_gpt2_load_model() -> anyhow::Result<()> {
        let raw = RawSafeTensors::from_hugging_face_cached(GPT2_SAFE_MODEL)?;
        let (model, md) = GPT2::new().parse(&raw)?;

        let config = md.config;
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.num_block, 12);
        assert_eq!(config.embedding_size, 768);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.context_length, 1024);
        assert_eq!(config.norm_epsilon, 1e-5);
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.eos_token, 50256usize.into());

        let input = Tensor::new(vec![1].into(), vec![546.0f32])?;
        model.run_float(vec![input], &mut GenStore::default())?;
        Ok(())
    }
}
