use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};
use tpx_core::attrs::AttrValue;
use tpx_core::types::{DTypeId, TpxError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetentionPolicy {
    KeepLatest,
    KeepLastN(u32),
    KeepForever,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        RetentionPolicy::KeepLatest
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VersionEntry {
    pub version_id: u32,
    pub timestamp: u64,
    pub dtype_id: DTypeId,
    pub shape: Vec<u32>,
    pub attrs: Vec<(String, AttrValue)>,
    pub chunk_refs: Vec<u128>, // Content hashes
}

impl VersionEntry {
    pub fn new(
        version_id: u32,
        dtype_id: DTypeId,
        shape: Vec<u32>,
        attrs: Vec<(String, AttrValue)>,
        chunk_refs: Vec<u128>,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            version_id,
            timestamp,
            dtype_id,
            shape,
            attrs,
            chunk_refs,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VersionDiff {
    pub added: Vec<u128>,
    pub removed: Vec<u128>,
}

#[derive(Clone, Debug)]
pub struct VersionChain {
    pub key: String,
    pub versions: Vec<VersionEntry>,
    pub retention: RetentionPolicy,
}

impl VersionChain {
    pub fn new(key: impl Into<String>, retention: RetentionPolicy) -> Self {
        Self {
            key: key.into(),
            versions: Vec::new(),
            retention,
        }
    }

    pub fn latest(&self) -> Option<&VersionEntry> {
        self.versions.last()
    }

    pub fn get_version(&self, version: u32) -> Option<&VersionEntry> {
        self.versions.iter().find(|v| v.version_id == version)
    }

    pub fn list_versions(&self) -> Vec<u32> {
        self.versions.iter().map(|v| v.version_id).collect()
    }

    pub fn append(
        &mut self,
        dtype_id: DTypeId,
        shape: Vec<u32>,
        attrs: Vec<(String, AttrValue)>,
        chunk_refs: Vec<u128>,
    ) -> (u32, Vec<u128>) {
        let next_version_id = self.versions.last().map(|v| v.version_id + 1).unwrap_or(0);
        let entry = VersionEntry::new(next_version_id, dtype_id, shape, attrs, chunk_refs);
        self.versions.push(entry);

        // Apply retention policy and return evicted chunk hashes
        let evicted = self.apply_retention();
        (next_version_id, evicted)
    }

    fn apply_retention(&mut self) -> Vec<u128> {
        let mut evicted_hashes = Vec::new();
        match self.retention {
            RetentionPolicy::KeepLatest => {
                if self.versions.len() > 1 {
                    let to_remove = self.versions.len() - 1;
                    for v in self.versions.drain(0..to_remove) {
                        evicted_hashes.extend(v.chunk_refs);
                    }
                }
            }
            RetentionPolicy::KeepLastN(n) => {
                let limit = n.max(1) as usize;
                if self.versions.len() > limit {
                    let to_remove = self.versions.len() - limit;
                    for v in self.versions.drain(0..to_remove) {
                        evicted_hashes.extend(v.chunk_refs);
                    }
                }
            }
            RetentionPolicy::KeepForever => {}
        }
        evicted_hashes
    }

    pub fn diff(&self, v1: u32, v2: u32) -> Result<VersionDiff, TpxError> {
        let e1 = self
            .get_version(v1)
            .ok_or_else(|| TpxError::VersionNotFound {
                key: self.key.clone(),
                version: v1,
            })?;
        let e2 = self
            .get_version(v2)
            .ok_or_else(|| TpxError::VersionNotFound {
                key: self.key.clone(),
                version: v2,
            })?;

        let set1: HashSet<u128> = e1.chunk_refs.iter().copied().collect();
        let set2: HashSet<u128> = e2.chunk_refs.iter().copied().collect();

        let added = set2.difference(&set1).copied().collect();
        let removed = set1.difference(&set2).copied().collect();

        Ok(VersionDiff { added, removed })
    }
}
