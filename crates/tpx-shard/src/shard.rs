use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tpx_core::types::ChunkRef;
use tpx_index::art::{ArtSnapshot, ArtTree};
use tpx_version::chain::VersionChain;

pub struct Shard {
    pub id: u16,
    pub art: ArtTree<ChunkRef>,
    pub append_offset: AtomicU64,
    pub version_chains: RwLock<HashMap<String, VersionChain>>,
    pub chunk_count: AtomicU64,
}

impl Shard {
    pub fn new(id: u16, start_offset: u64) -> Self {
        Self {
            id,
            art: ArtTree::new(),
            append_offset: AtomicU64::new(start_offset),
            version_chains: RwLock::new(HashMap::new()),
            chunk_count: AtomicU64::new(0),
        }
    }

    pub fn allocate_append(&self, bytes_len: u64) -> u64 {
        self.append_offset.fetch_add(bytes_len, Ordering::SeqCst)
    }

    pub fn snapshot_art(&self) -> ArtSnapshot<ChunkRef> {
        self.art.snapshot()
    }
}
