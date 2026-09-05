use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tpx_core::footer::Footer;
use tpx_core::types::{
    TpxError, DEFAULT_CHECKPOINT_INTERVAL_BYTES, DEFAULT_CHECKPOINT_INTERVAL_MS,
};
use tpx_index::toc::Toc;
use tpx_io::tier::IoBackend;
use xxhash_rust::xxh64::xxh64;

use crate::shard::Shard;

pub struct CheckpointCoordinator {
    pub interval_bytes: AtomicU64,
    pub interval_ms: u64,
    pub bytes_written_since_last: AtomicU64,
    pub last_checkpoint_time: parking_lot::Mutex<Instant>,
}

impl Default for CheckpointCoordinator {
    fn default() -> Self {
        Self::new(
            DEFAULT_CHECKPOINT_INTERVAL_BYTES,
            DEFAULT_CHECKPOINT_INTERVAL_MS,
        )
    }
}

impl CheckpointCoordinator {
    pub fn new(interval_bytes: u64, interval_ms: u64) -> Self {
        Self {
            interval_bytes: AtomicU64::new(interval_bytes),
            interval_ms,
            bytes_written_since_last: AtomicU64::new(0),
            last_checkpoint_time: parking_lot::Mutex::new(Instant::now()),
        }
    }

    pub fn record_bytes_written(&self, bytes: u64) {
        self.bytes_written_since_last
            .fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn should_checkpoint(&self) -> bool {
        let written = self.bytes_written_since_last.load(Ordering::Relaxed);
        if written >= self.interval_bytes.load(Ordering::Relaxed) {
            return true;
        }

        let elapsed = self.last_checkpoint_time.lock().elapsed().as_millis() as u64;
        elapsed >= self.interval_ms && written > 0
    }

    pub fn execute_checkpoint(
        &self,
        shards: &[Arc<Shard>],
        io: &dyn IoBackend,
        master_toc: &parking_lot::RwLock<Toc>,
        is_final: bool,
    ) -> Result<Footer, TpxError> {
        let toc_bytes = {
            let guard = master_toc.read();
            guard.serialize()
        };

        let toc_offset = io.len();
        // Fast atomic checksum over TOC metadata and archive offset, avoiding 100MB+ disk re-reads on every flush
        let checksum = xxh64(&toc_bytes, toc_offset);

        let total_chunks: u32 = shards
            .iter()
            .map(|s| s.chunk_count.load(Ordering::Relaxed) as u32)
            .sum();

        let footer = Footer::new(toc_offset, total_chunks, checksum, !is_final);
        let footer_bytes = footer.serialize();

        let chunk_align = 4096u64;
        let raw_len = (toc_bytes.len() + footer_bytes.len()) as u64;
        let rem = raw_len % chunk_align;
        let pad_len = if rem == 0 {
            0
        } else {
            (chunk_align - rem) as usize
        };

        let mut block = Vec::with_capacity(toc_bytes.len() + pad_len + footer_bytes.len());
        block.extend_from_slice(&toc_bytes);
        block.resize(toc_bytes.len() + pad_len, 0u8);
        block.extend_from_slice(&footer_bytes);

        io.append(&block)?;
        io.sync()?;

        let new_end = toc_offset + block.len() as u64;
        for s in shards {
            s.append_offset.store(new_end, Ordering::SeqCst);
        }

        self.bytes_written_since_last.store(0, Ordering::SeqCst);
        *self.last_checkpoint_time.lock() = Instant::now();

        Ok(footer)
    }
}
