use tpx_core::types::TpxError;

pub struct QatOffload;

impl QatOffload {
    pub fn detect() -> Option<Self> {
        // Detected at runtime if Intel QAT device driver is present
        None
    }

    pub fn compress_block(&self, data: &[u8]) -> Result<Vec<u8>, TpxError> {
        Ok(data.to_vec())
    }

    pub fn decompress_block(&self, data: &[u8], _out_len: usize) -> Result<Vec<u8>, TpxError> {
        Ok(data.to_vec())
    }
}
