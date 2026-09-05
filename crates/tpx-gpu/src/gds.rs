use tpx_core::types::TpxError;

pub struct GdsReader;

impl GdsReader {
    pub fn detect() -> Option<Self> {
        None
    }

    pub fn read_to_vram(&self, _offset: u64, _len: usize, _dev_ptr: u64) -> Result<(), TpxError> {
        Err(TpxError::UnsupportedFeature(
            "GPUDirect Storage (nvidia-fs) not available",
        ))
    }
}
