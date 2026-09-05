use tpx_core::types::TpxError;

pub struct Lz4Codec;

impl Lz4Codec {
    pub fn compress(data: &[u8]) -> Result<Vec<u8>, TpxError> {
        Ok(lz4_flex::compress_prepend_size(data))
    }

    pub fn decompress(data: &[u8], uncompressed_len: usize) -> Result<Vec<u8>, TpxError> {
        let decompressed = lz4_flex::decompress_size_prepended(data)
            .map_err(|e| TpxError::CodecError(e.to_string()))?;
        if decompressed.len() != uncompressed_len {
            return Err(TpxError::DecompressLengthMismatch {
                expected: uncompressed_len,
                got: decompressed.len(),
            });
        }
        Ok(decompressed)
    }
}
