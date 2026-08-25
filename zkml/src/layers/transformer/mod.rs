use crate::{
    NextPowerOfTwo, Shape,
    padding::PaddingMode,
    tensor::{TensorTypeParam, WrappedTensor},
};

use serde::{Deserialize, Serialize, de::DeserializeOwned};

pub mod attention_mask;
pub mod embeddings;
pub mod logits;
pub mod normalisation;
pub mod positional;
pub mod softmax;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "N: Serialize", deserialize = "N: DeserializeOwned"))]
pub struct ConcatenationCache<N: TensorTypeParam> {
    cached_tensor: Option<WrappedTensor<N>>,
    caching_info: CachingInfo,
    /// Inference runners may temporarily bypass this cache while preserving
    /// its static rank and concatenation metadata.
    #[serde(skip)]
    disabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Copy)]
enum CachingInfo {
    /// Used in cases where the cache will know for sure the shape of the tensor it is
    /// caching before run time
    Static {
        rank: usize,
        concatenation_dim: usize,
    },
    /// Used in cases like Softmax caching where we only know that we are caching on the second last dimension, but
    /// aren't sure how many total dimensions there are.
    Dynamic(isize),
}

impl CachingInfo {
    fn new_static(rank: usize, concatenation_dim: usize) -> CachingInfo {
        CachingInfo::Static {
            rank,
            concatenation_dim,
        }
    }

    fn new_dynamic(concatenation_dim: isize) -> CachingInfo {
        CachingInfo::Dynamic(concatenation_dim)
    }

    fn get_concatenation_dim(&self, tensor_rank: usize) -> anyhow::Result<usize> {
        match self {
            CachingInfo::Static {
                concatenation_dim, ..
            } => Ok(*concatenation_dim),
            CachingInfo::Dynamic(concatenation_dim) => {
                if *concatenation_dim < 0 {
                    let rank_isize = tensor_rank as isize;
                    let dim = rank_isize + concatenation_dim;
                    anyhow::ensure!(
                        dim >= 0isize,
                        "Could not acquire concatenation dimension in ConcatenationCache, provided tensor had rank to small, rank: {tensor_rank}, concatenation_dim: {concatenation_dim}"
                    );
                    Ok(dim as usize)
                } else {
                    let dim = *concatenation_dim as usize;
                    anyhow::ensure!(
                        dim < tensor_rank,
                        "Could not acquire concatenation dimension in ConcatenationCache, provided tensor had rank to small, rank: {tensor_rank}, concatenation_dim: {concatenation_dim}"
                    );
                    Ok(dim)
                }
            }
        }
    }
}

impl<N: TensorTypeParam> ConcatenationCache<N> {
    pub fn new(rank: usize, concatenation_dim: usize) -> Self {
        let caching_info = CachingInfo::new_static(rank, concatenation_dim);
        Self {
            cached_tensor: None,
            caching_info,
            disabled: false,
        }
    }

    pub fn new_dynamic(concatenation_dim: isize) -> Self {
        let caching_info = CachingInfo::new_dynamic(concatenation_dim);
        Self {
            cached_tensor: None,
            caching_info,
            disabled: false,
        }
    }

    pub fn reset(&mut self) {
        self.cached_tensor = None;
    }

    pub fn set_disabled(&mut self, disabled: bool) {
        self.disabled = disabled;
        if disabled {
            self.reset();
        }
    }
    pub fn is_initialized(&self) -> bool {
        self.cached_tensor.is_some()
    }

    fn get_concatenation_dim(&self, tensor_rank: usize) -> anyhow::Result<usize> {
        self.caching_info.get_concatenation_dim(tensor_rank)
    }

    pub fn concatenate(
        &mut self,
        new_tensor: WrappedTensor<N>,
    ) -> anyhow::Result<WrappedTensor<N>> {
        if self.disabled {
            return Ok(new_tensor);
        }

        // We retrieve the concatenation dim here to enforce that the tensor to be cached is valid.
        let rank = new_tensor.rank();
        let concatenation_dim = self.get_concatenation_dim(rank)?;

        let output = if self.is_initialized() {
            let mut placeholder = None;
            std::mem::swap(&mut placeholder, &mut self.cached_tensor);

            // Unwrap is safe because we are initialised
            let cached_tensor = placeholder.unwrap();
            let catted = WrappedTensor::cat(vec![cached_tensor, new_tensor], concatenation_dim)?;
            self.cached_tensor = Some(catted.clone());
            catted
        } else {
            self.cached_tensor = Some(new_tensor.clone());
            new_tensor
        };

        Ok(output)
    }

    /// Given a [`Shape`], returns the next shape after concatenation.
    /// Here `padding_mode` determines whether to pad the new shape to the next power of two.
    pub fn next_shape(&self, shape: Shape, padding_mode: PaddingMode) -> anyhow::Result<Shape> {
        if self.disabled {
            return Ok(match padding_mode {
                PaddingMode::NoPadding => shape,
                PaddingMode::Padding => shape.next_power_of_two(),
            });
        }

        let mut new_shape = shape;
        let rank = new_shape.rank();
        let concatenation_dim = self.get_concatenation_dim(rank)?;
        new_shape[concatenation_dim] += self.current_sequence_length();
        if let PaddingMode::Padding = padding_mode {
            new_shape = new_shape.next_power_of_two();
        }
        Ok(new_shape)
    }

    pub fn get_cached(&self) -> anyhow::Result<WrappedTensor<N>> {
        self.cached_tensor
            .clone()
            .ok_or(anyhow::anyhow!("ConcatenationCache is not initialized"))
    }

    pub fn current_sequence_length(&self) -> usize {
        if let Some(tensor) = &self.cached_tensor {
            let rank = tensor.rank();

            // Unwrap is safe because if initialised we have already checked that we can get the concatenation dim
            let concatenation_dim = self.get_concatenation_dim(rank).unwrap();
            tensor.dim(concatenation_dim as isize).unwrap()
        } else {
            0
        }
    }

    pub fn cache_info(&self) -> (usize, usize) {
        match self.caching_info {
            CachingInfo::Static {
                rank,
                concatenation_dim,
            } => (rank, concatenation_dim),
            CachingInfo::Dynamic(..) => {
                panic!("Should not be accessing caching info for Dynamic variant")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Element;

    #[test]
    fn concatenation_cache_can_be_temporarily_bypassed() {
        let mut cache = ConcatenationCache::<Element>::new(1, 0);
        let first = WrappedTensor::try_from(vec![1, 2]).unwrap();

        cache.set_disabled(true);
        let bypassed = cache.concatenate(first.clone()).unwrap();
        assert_eq!(bypassed.get_data(), vec![1, 2]);
        assert!(!cache.is_initialized());
        assert_eq!(
            cache
                .next_shape(Shape::from([2]), PaddingMode::NoPadding)
                .unwrap(),
            Shape::from([2])
        );

        cache.set_disabled(false);
        cache.concatenate(first).unwrap();
        let concatenated = cache
            .concatenate(WrappedTensor::try_from(vec![3]).unwrap())
            .unwrap();
        assert_eq!(concatenated.get_data(), vec![1, 2, 3]);
    }
}
