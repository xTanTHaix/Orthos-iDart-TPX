use tpx_core::types::TpxError;

pub struct ZstdCodec;

impl ZstdCodec {
    pub fn compress(data: &[u8], level: i32) -> Result<Vec<u8>, TpxError> {
        let level = level.clamp(1, 22);
        zstd::bulk::compress(data, level).map_err(|e| TpxError::CodecError(e.to_string()))
    }

    pub fn compress_with_dict(data: &[u8], level: i32, dict: &[u8]) -> Result<Vec<u8>, TpxError> {
        let level = level.clamp(1, 22);
        let mut encoder = zstd::bulk::Compressor::with_dictionary(level, dict)
            .map_err(|e| TpxError::CodecError(e.to_string()))?;
        encoder
            .compress(data)
            .map_err(|e| TpxError::CodecError(e.to_string()))
    }

    pub fn decompress(data: &[u8], uncompressed_len: usize) -> Result<Vec<u8>, TpxError> {
        zstd::bulk::decompress(data, uncompressed_len)
            .map_err(|e| TpxError::CodecError(e.to_string()))
    }

    pub fn decompress_with_dict(
        data: &[u8],
        uncompressed_len: usize,
        dict: &[u8],
    ) -> Result<Vec<u8>, TpxError> {
        let mut decoder = zstd::bulk::Decompressor::with_dictionary(dict)
            .map_err(|e| TpxError::CodecError(e.to_string()))?;
        decoder
            .decompress(data, uncompressed_len)
            .map_err(|e| TpxError::CodecError(e.to_string()))
    }
}
