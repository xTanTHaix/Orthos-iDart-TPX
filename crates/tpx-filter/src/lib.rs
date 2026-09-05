pub mod bit_shuffle;
pub mod byte_shuffle;
pub mod delta_zigzag;

use tpx_core::types::{DTypeId, FilterId};

pub trait Filter: Send + Sync {
    fn id(&self) -> FilterId;
    fn apply(&self, block: &[u8], dtype: DTypeId) -> Vec<u8>;
    fn revert(&self, block: &[u8], dtype: DTypeId) -> Vec<u8>;
}

pub struct NoneFilter;
impl Filter for NoneFilter {
    fn id(&self) -> FilterId {
        FilterId::None
    }
    fn apply(&self, block: &[u8], _dtype: DTypeId) -> Vec<u8> {
        block.to_vec()
    }
    fn revert(&self, block: &[u8], _dtype: DTypeId) -> Vec<u8> {
        block.to_vec()
    }
}

pub struct ByteShuffleFilter;
impl Filter for ByteShuffleFilter {
    fn id(&self) -> FilterId {
        FilterId::ByteShuffle
    }
    fn apply(&self, block: &[u8], dtype: DTypeId) -> Vec<u8> {
        byte_shuffle::shuffle(block, dtype.element_size())
    }
    fn revert(&self, block: &[u8], dtype: DTypeId) -> Vec<u8> {
        byte_shuffle::unshuffle(block, dtype.element_size())
    }
}

pub struct BitShuffleFilter;
impl Filter for BitShuffleFilter {
    fn id(&self) -> FilterId {
        FilterId::BitShuffle
    }
    fn apply(&self, block: &[u8], _dtype: DTypeId) -> Vec<u8> {
        bit_shuffle::bit_shuffle(block)
    }
    fn revert(&self, block: &[u8], _dtype: DTypeId) -> Vec<u8> {
        bit_shuffle::bit_unshuffle(block)
    }
}

pub struct DeltaZigZagFilter;
impl Filter for DeltaZigZagFilter {
    fn id(&self) -> FilterId {
        FilterId::DeltaZigZag
    }
    fn apply(&self, block: &[u8], dtype: DTypeId) -> Vec<u8> {
        delta_zigzag::encode(block, dtype)
    }
    fn revert(&self, block: &[u8], dtype: DTypeId) -> Vec<u8> {
        delta_zigzag::decode(block, dtype)
    }
}

pub fn get_filter(id: FilterId) -> Box<dyn Filter> {
    match id {
        FilterId::None => Box::new(NoneFilter),
        FilterId::ByteShuffle => Box::new(ByteShuffleFilter),
        FilterId::BitShuffle => Box::new(BitShuffleFilter),
        FilterId::DeltaZigZag => Box::new(DeltaZigZagFilter),
    }
}
