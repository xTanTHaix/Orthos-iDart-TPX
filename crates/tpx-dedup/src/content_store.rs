use dashmap::DashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use tpx_core::types::TpxError;

#[derive(Clone, Debug)]
pub struct ContentEntry {
    pub shard_id: u16,
    pub chunk_offset: u64,
    pub chunk_len: u32,
    pub refcount: Arc<AtomicU32>,
}

impl ContentEntry {
    pub fn new(shard_id: u16, chunk_offset: u64, chunk_len: u32, initial_refcount: u32) -> Self {
        Self {
            shard_id,
            chunk_offset,
            chunk_len,
            refcount: Arc::new(AtomicU32::new(initial_refcount)),
        }
    }
}

pub struct ContentStore {
    shards: DashMap<u128, ContentEntry>,
}

impl Default for ContentStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ContentStore {
    pub fn new() -> Self {
        Self {
            shards: DashMap::new(),
        }
    }

    pub fn lookup(&self, hash: u128) -> Option<ContentEntry> {
        self.shards.get(&hash).map(|entry| entry.clone())
    }

    pub fn insert(&self, hash: u128, entry: ContentEntry) -> bool {
        if self.shards.contains_key(&hash) {
            return false;
        }
        self.shards.insert(hash, entry);
        true
    }

    pub fn add_ref(&self, hash: u128) -> Result<u32, TpxError> {
        if let Some(entry) = self.shards.get(&hash) {
            let prev = entry.refcount.fetch_add(1, Ordering::SeqCst);
            Ok(prev + 1)
        } else {
            Err(TpxError::KeyNotFound(format!("Content hash {:032x}", hash)))
        }
    }

    pub fn release(&self, hash: u128) -> Result<u32, TpxError> {
        if let Some(entry) = self.shards.get(&hash) {
            let prev = entry.refcount.fetch_sub(1, Ordering::SeqCst);
            let new_count = prev.saturating_sub(1);
            Ok(new_count)
        } else {
            Err(TpxError::KeyNotFound(format!("Content hash {:032x}", hash)))
        }
    }

    pub fn gc_candidates(&self) -> Vec<(u128, ContentEntry)> {
        self.shards
            .iter()
            .filter(|entry| entry.value().refcount.load(Ordering::SeqCst) == 0)
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect()
    }

    pub fn remove(&self, hash: u128) -> Option<ContentEntry> {
        self.shards.remove(&hash).map(|(_, v)| v)
    }

    pub fn len(&self) -> usize {
        self.shards.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shards.is_empty()
    }
}
