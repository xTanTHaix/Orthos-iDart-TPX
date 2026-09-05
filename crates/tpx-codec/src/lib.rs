pub mod auto_select;
pub mod chimp128;
pub mod dictionary;
pub mod lz4_codec;
pub mod zstd_codec;

use tpx_core::types::{CodecId, DTypeId, TpxError};

pub use auto_select::{auto_select, CodecSelection};
pub use chimp128::{Chimp128Decoder, Chimp128Encoder};
pub use dictionary::{train_dict, TrainedDict};
pub use lz4_codec::Lz4Codec;
pub use zstd_codec::ZstdCodec;

pub trait Codec: Send + Sync {
    fn id(&self) -> CodecId;
    fn compress(&self, data: &[u8], level: i32, dtype: DTypeId) -> Result<Vec<u8>, TpxError>;
    fn decompress(
        &self,
        data: &[u8],
        uncompressed_len: usize,
        dtype: DTypeId,
    ) -> Result<Vec<u8>, TpxError>;
}

pub struct RawCodec;
impl Codec for RawCodec {
    fn id(&self) -> CodecId {
        CodecId::Raw
    }
    fn compress(&self, data: &[u8], _level: i32, _dtype: DTypeId) -> Result<Vec<u8>, TpxError> {
        Ok(data.to_vec())
    }
    fn decompress(
        &self,
        data: &[u8],
        uncompressed_len: usize,
        _dtype: DTypeId,
    ) -> Result<Vec<u8>, TpxError> {
        if data.len() != uncompressed_len {
            return Err(TpxError::DecompressLengthMismatch {
                expected: uncompressed_len,
                got: data.len(),
            });
        }
        Ok(data.to_vec())
    }
}

pub struct Chimp128Codec;
impl Codec for Chimp128Codec {
    fn id(&self) -> CodecId {
        CodecId::Chimp128
    }
    fn compress(&self, data: &[u8], _level: i32, dtype: DTypeId) -> Result<Vec<u8>, TpxError> {
        let mut encoder = Chimp128Encoder::new();
        if dtype == DTypeId::Float64 {
            let count = data.len() / 8;
            for i in 0..count {
                let start = i * 8;
                let val = f64::from_le_bytes(data[start..start + 8].try_into().unwrap());
                encoder.encode_f64(val);
            }
        } else {
            // Default f32
            let count = data.len() / 4;
            for i in 0..count {
                let start = i * 4;
                let val = f32::from_le_bytes(data[start..start + 4].try_into().unwrap());
                encoder.encode_f32(val);
            }
        }
        Ok(encoder.finish())
    }

    fn decompress(
        &self,
        data: &[u8],
        uncompressed_len: usize,
        dtype: DTypeId,
    ) -> Result<Vec<u8>, TpxError> {
        let mut decoder = Chimp128Decoder::new(data)?;
        let mut out = Vec::with_capacity(uncompressed_len);

        if dtype == DTypeId::Float64 {
            while let Some(val) = decoder.decode_f64() {
                out.extend_from_slice(&val.to_le_bytes());
            }
        } else {
            while let Some(val) = decoder.decode_f32() {
                out.extend_from_slice(&val.to_le_bytes());
            }
        }

        if out.len() != uncompressed_len {
            return Err(TpxError::DecompressLengthMismatch {
                expected: uncompressed_len,
                got: out.len(),
            });
        }
        Ok(out)
    }
}

pub struct ZstdCodecWrapper;
impl Codec for ZstdCodecWrapper {
    fn id(&self) -> CodecId {
        CodecId::Zstd
    }
    fn compress(&self, data: &[u8], level: i32, _dtype: DTypeId) -> Result<Vec<u8>, TpxError> {
        ZstdCodec::compress(data, level)
    }
    fn decompress(
        &self,
        data: &[u8],
        uncompressed_len: usize,
        _dtype: DTypeId,
    ) -> Result<Vec<u8>, TpxError> {
        ZstdCodec::decompress(data, uncompressed_len)
    }
}

pub struct Lz4CodecWrapper;
impl Codec for Lz4CodecWrapper {
    fn id(&self) -> CodecId {
        CodecId::Lz4
    }
    fn compress(&self, data: &[u8], _level: i32, _dtype: DTypeId) -> Result<Vec<u8>, TpxError> {
        Lz4Codec::compress(data)
    }
    fn decompress(
        &self,
        data: &[u8],
        uncompressed_len: usize,
        _dtype: DTypeId,
    ) -> Result<Vec<u8>, TpxError> {
        Lz4Codec::decompress(data, uncompressed_len)
    }
}

pub fn get_codec(id: CodecId) -> Box<dyn Codec> {
    match id {
        CodecId::Raw => Box::new(RawCodec),
        CodecId::Chimp128 => Box::new(Chimp128Codec),
        CodecId::Zstd => Box::new(ZstdCodecWrapper),
        CodecId::Lz4 => Box::new(Lz4CodecWrapper),
    }
}
