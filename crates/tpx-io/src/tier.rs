use std::path::Path;
use tpx_core::types::{OpenMode, TpxError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoTier {
    Spdk,
    IoUring { sqpoll: bool, iopoll: bool },
    PosixVectored,
    Mmap,
}

#[derive(Clone, Debug, Default)]
pub struct IoHints {
    pub direct_io: bool,
    pub sqpoll: bool,
    pub iopoll: bool,
    pub spdk: bool,
}

pub trait IoBackend: Send + Sync {
    fn write_at(&self, offset: u64, data: &[u8]) -> Result<(), TpxError>;
    fn read_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, TpxError>;
    fn read_at_into(&self, offset: u64, buf: &mut [u8]) -> Result<(), TpxError>;
    fn append(&self, data: &[u8]) -> Result<u64, TpxError>;
    fn len(&self) -> u64;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn sync(&self) -> Result<(), TpxError>;
    fn truncate(&self, len: u64) -> Result<(), TpxError>;
    fn tier(&self) -> IoTier;
    fn close(&self) -> Result<(), TpxError>;
}

pub fn detect_best_tier(
    path: &Path,
    mode: OpenMode,
    hints: &IoHints,
) -> Result<Box<dyn IoBackend>, TpxError> {
    // On Linux with io_uring enabled, tier 1 would be used.
    // Portable default fallback is Mmap / Posix.
    crate::mmap::MmapBackend::open_or_create(path, mode, hints)
}
