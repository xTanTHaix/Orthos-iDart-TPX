use tpx_core::types::TpxError;

pub struct GpuDecodeKernel;

impl GpuDecodeKernel {
    pub fn decode_chunk(_data: &[u8]) -> Result<Vec<u8>, TpxError> {
        Err(TpxError::UnsupportedFeature(
            "GPU decode kernel requires CUDA hardware",
        ))
    }
}
