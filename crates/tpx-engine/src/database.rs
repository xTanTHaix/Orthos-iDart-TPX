use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tpx_core::attrs::{AttrValue, Attrs};
use tpx_core::types::{DTypeId, OpenMode, OwnedTensor, TpxError};
use tpx_shard::engine::ShardedEngine;
use tpx_version::chain::VersionDiff;

use crate::batch::BatchReader;

#[derive(Clone)]
pub struct TPXDatabase {
    engine: Arc<ShardedEngine>,
}

impl TPXDatabase {
    pub fn create(path: impl AsRef<Path>, shard_count: Option<u16>) -> Result<Self, TpxError> {
        let engine = ShardedEngine::create(path.as_ref(), shard_count)?;
        Ok(Self {
            engine: Arc::new(engine),
        })
    }

    pub fn open(path: impl AsRef<Path>, mode: OpenMode) -> Result<Self, TpxError> {
        let engine = match mode {
            OpenMode::ReadOnly => ShardedEngine::open(path.as_ref(), OpenMode::ReadOnly)?,
            OpenMode::ReadWrite | OpenMode::Append => {
                if path.as_ref().exists() {
                    ShardedEngine::open(path.as_ref(), mode)?
                } else {
                    ShardedEngine::create(path.as_ref(), None)?
                }
            }
        };

        Ok(Self {
            engine: Arc::new(engine),
        })
    }

    pub fn put(
        &self,
        key: &str,
        data: &[u8],
        dtype: DTypeId,
        shape: &[u32],
        attrs: Option<HashMap<String, AttrValue>>,
        retain: Option<&str>,
    ) -> Result<(), TpxError> {
        let mut attrs_obj = Attrs::new();
        if let Some(map) = attrs {
            for (k, v) in map {
                attrs_obj.set(k, v)?;
            }
        }
        self.engine.put_with_retention(
            key,
            data,
            dtype,
            shape,
            attrs_obj,
            retain.unwrap_or("keep_latest"),
        )
    }

    pub fn put_batch(&self, tensors: &[(&str, &[u8], DTypeId, &[u32])]) -> Result<(), TpxError> {
        self.engine.put_batch(tensors)
    }

    pub fn get(&self, key: &str) -> Result<Option<OwnedTensor>, TpxError> {
        self.engine.get(key)
    }

    pub fn get_attrs(&self, key: &str) -> Result<HashMap<String, AttrValue>, TpxError> {
        self.engine.get_attrs(key)
    }

    pub fn get_prefix(&self, prefix: &str) -> Result<HashMap<String, OwnedTensor>, TpxError> {
        self.engine.get_prefix(prefix)
    }

    pub fn get_batch(&self, keys: &[&str]) -> Result<Vec<(String, OwnedTensor)>, TpxError> {
        let reader = BatchReader::new(&self.engine);
        reader.get_batch(keys)
    }

    pub fn keys(&self) -> Result<Vec<String>, TpxError> {
        self.engine.keys()
    }

    pub fn batch_reader(&self) -> BatchReader<'_> {
        BatchReader::new(&self.engine)
    }

    pub fn engine(&self) -> &ShardedEngine {
        &self.engine
    }

    pub fn list_versions(&self, key: &str) -> Result<Vec<u32>, TpxError> {
        self.engine.list_versions(key)
    }

    pub fn get_version(&self, key: &str, version: u32) -> Result<Option<OwnedTensor>, TpxError> {
        self.engine.get_version(key, version)
    }

    pub fn diff(&self, key: &str, v1: u32, v2: u32) -> Result<VersionDiff, TpxError> {
        self.engine.diff(key, v1, v2)
    }

    pub fn delete(&self, key: &str) -> Result<(), TpxError> {
        self.engine.delete(key)
    }

    pub fn flush(&self) -> Result<(), TpxError> {
        self.engine.flush()
    }

    pub fn compact(&self) -> Result<(), TpxError> {
        self.engine.compact()
    }

    pub fn close(&self) -> Result<(), TpxError> {
        self.engine.close()
    }

    /// One-liner: Save a single tensor to a TPX file with auto-flush and close.
    pub fn save_tensor(
        path: impl AsRef<Path>,
        key: &str,
        data: &[u8],
        dtype: DTypeId,
        shape: &[u32],
    ) -> Result<(), TpxError> {
        let db = Self::create(path, None)?;
        db.put(key, data, dtype, shape, None, None)?;
        db.flush()?;
        db.close()?;
        Ok(())
    }

    /// One-liner: Save a batch of tensors to a TPX file with multi-core coalescing.
    pub fn save_batch(
        path: impl AsRef<Path>,
        tensors: &[(&str, &[u8], DTypeId, &[u32])],
    ) -> Result<(), TpxError> {
        let db = Self::create(path, None)?;
        db.put_batch(tensors)?;
        db.flush()?;
        db.close()?;
        Ok(())
    }

    /// One-liner: Load a single tensor by key from a TPX file.
    pub fn load_tensor(path: impl AsRef<Path>, key: &str) -> Result<Option<OwnedTensor>, TpxError> {
        let db = Self::open(path, OpenMode::ReadOnly)?;
        db.get(key)
    }

    /// One-liner: Inspect all keys in a TPX file without loading payloads.
    pub fn list_keys(path: impl AsRef<Path>) -> Result<Vec<String>, TpxError> {
        let db = Self::open(path, OpenMode::ReadOnly)?;
        db.keys()
    }
}
