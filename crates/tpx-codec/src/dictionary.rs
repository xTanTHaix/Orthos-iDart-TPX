use tpx_core::types::{DTypeId, TpxError};
use xxhash_rust::xxh64::xxh64;

#[derive(Clone, Debug)]
pub struct TrainedDict {
    pub data: Vec<u8>,
    pub signature: u64,
}

impl TrainedDict {
    pub fn new(data: Vec<u8>, signature: u64) -> Self {
        Self { data, signature }
    }

    pub fn compute_signature(shape: &[u32], dtype: DTypeId) -> u64 {
        let mut buf = Vec::with_capacity(1 + shape.len() * 4);
        buf.push(dtype as u8);
        for &dim in shape {
            buf.extend_from_slice(&dim.to_le_bytes());
        }
        xxh64(&buf, 0)
    }
}

pub fn train_dict(samples: &[&[u8]], max_size: usize) -> Result<TrainedDict, TpxError> {
    if samples.is_empty() {
        return Err(TpxError::CodecError(
            "Cannot train dictionary on empty sample set".into(),
        ));
    }

    let dict_data = zstd::dict::from_samples(samples, max_size)
        .map_err(|e| TpxError::CodecError(format!("Dictionary training failed: {e}")))?;

    let signature = xxh64(&dict_data, 0);
    Ok(TrainedDict::new(dict_data, signature))
}
