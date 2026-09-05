use crc32fast::Hasher;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tpx_core::attrs::{AttrValue, Attrs};
use tpx_core::chunk::{pad_to_align, BlockEntry, ChunkHeader};
use tpx_core::footer::{scan_for_footer, Footer};
use tpx_core::header::FileHeader;
use tpx_core::types::{
    ChunkRef, DTypeId, OpenMode, OwnedTensor, TpxError, DEFAULT_CHUNK_ALIGN, HEADER_SIZE,
};
use tpx_dedup::cdc::{cdc_chunk, CdcConfig};
use tpx_dedup::content_store::{ContentEntry, ContentStore};
use tpx_filter::get_filter;
use tpx_index::toc::{Toc, TocEntry};
use tpx_io::tier::{detect_best_tier, IoBackend, IoHints};
use tpx_version::chain::{RetentionPolicy, VersionChain, VersionDiff};

use crate::checkpoint::CheckpointCoordinator;
use crate::router::shard_for_key;
use crate::shard::Shard;

pub struct ShardedEngine {
    pub path: PathBuf,
    pub mode: OpenMode,
    pub shards: Vec<Arc<Shard>>,
    pub io: Box<dyn IoBackend>,
    pub header: FileHeader,
    pub master_toc: RwLock<Toc>,
    pub content_store: Arc<ContentStore>,
    pub checkpoint_coordinator: CheckpointCoordinator,
    pub is_closed: RwLock<bool>,
}

impl ShardedEngine {
    pub fn create(path: &Path, shard_count: Option<u16>) -> Result<Self, TpxError> {
        Self::create_with_retention(path, shard_count, RetentionPolicy::KeepLatest)
    }

    pub fn create_with_retention(
        path: &Path,
        shard_count: Option<u16>,
        _retention: RetentionPolicy,
    ) -> Result<Self, TpxError> {
        let sc = shard_count.unwrap_or(4).max(1);
        let header = FileHeader::new(sc, DEFAULT_CHUNK_ALIGN, 0x02); // Checksummed

        let hints = IoHints::default();
        let io = detect_best_tier(path, OpenMode::ReadWrite, &hints)?;

        // Write initial header padded to chunk alignment boundary
        let mut header_bytes = header.serialize().to_vec();
        pad_to_align(&mut header_bytes, header.chunk_align);
        io.write_at(0, &header_bytes)?;

        let initial_offset = header_bytes.len() as u64;
        let mut shards = Vec::with_capacity(sc as usize);
        for i in 0..sc {
            shards.push(Arc::new(Shard::new(i, initial_offset)));
        }

        let master_toc = RwLock::new(Toc::new());
        let content_store = Arc::new(ContentStore::new());
        let checkpoint_coordinator = CheckpointCoordinator::default();

        Ok(Self {
            path: path.to_path_buf(),
            mode: OpenMode::ReadWrite,
            shards,
            io,
            header,
            master_toc,
            content_store,
            checkpoint_coordinator,
            is_closed: RwLock::new(false),
        })
    }

    pub fn open(path: &Path, mode: OpenMode) -> Result<Self, TpxError> {
        let hints = IoHints::default();
        let io = detect_best_tier(path, mode, &hints)?;

        let file_len = io.len();
        if file_len < HEADER_SIZE as u64 {
            return Err(TpxError::BufferTooSmall {
                component: "FileHeader",
                required: HEADER_SIZE,
                provided: file_len as usize,
            });
        }

        let header_bytes = io.read_at(0, HEADER_SIZE)?;
        let header = FileHeader::deserialize(&header_bytes)?;

        // Scan for footer to read TOC
        let full_file_bytes = io.read_at(0, file_len as usize)?;
        let footer_opt = scan_for_footer(&full_file_bytes, file_len, header.chunk_align)?;
        let (footer, footer_offset) = footer_opt.ok_or_else(|| {
            TpxError::CorruptArchive("No valid footer found in TPX archive".into())
        })?;

        // Read and deserialize TOC using exact footer_offset
        let toc_len = footer_offset - footer.toc_offset;
        let toc_bytes = io.read_at(footer.toc_offset, toc_len as usize)?;
        let toc = Toc::deserialize(&toc_bytes)?;

        // If file had trailing garbage or unaligned bytes beyond the footer, truncate to clean length
        let clean_file_len = footer_offset + 24;
        if file_len > clean_file_len && (mode == OpenMode::ReadWrite || mode == OpenMode::Append) {
            let _ = io.truncate(clean_file_len);
        }

        let sc = header.shard_count.max(1);
        let mut shards = Vec::with_capacity(sc as usize);
        for i in 0..sc {
            let shard = Arc::new(Shard::new(i, file_len));
            shards.push(shard);
        }

        let content_store = Arc::new(ContentStore::new());

        // Populate shard ARTs and version chains from loaded TOC
        for entry in &toc.entries {
            let s_id = shard_for_key(&entry.key, sc);
            if let Some(shard) = shards.get(s_id as usize) {
                if entry.tombstone == 0 {
                    for chunk_ref in &entry.chunk_refs {
                        shard.art.insert(entry.key.as_bytes(), *chunk_ref);
                    }
                    let mut vc_guard = shard.version_chains.write();
                    let vc = vc_guard.entry(entry.key.clone()).or_insert_with(|| {
                        VersionChain::new(&entry.key, RetentionPolicy::KeepForever)
                    });
                    let chunk_hashes: Vec<u128> = entry
                        .chunk_refs
                        .iter()
                        .map(|c| (c.shard_id as u128) << 64 | c.chunk_offset as u128)
                        .collect();
                    vc.append(
                        entry.dtype_id,
                        entry.shape.clone(),
                        entry.attrs.clone(),
                        chunk_hashes,
                    );
                } else {
                    shard.art.remove(entry.key.as_bytes());
                    let mut vc_guard = shard.version_chains.write();
                    vc_guard.remove(&entry.key);
                }
            }
        }

        Ok(Self {
            path: path.to_path_buf(),
            mode,
            shards,
            io,
            header,
            master_toc: RwLock::new(toc),
            content_store,
            checkpoint_coordinator: CheckpointCoordinator::default(),
            is_closed: RwLock::new(false),
        })
    }

    pub fn put(
        &self,
        key: &str,
        data: &[u8],
        dtype: DTypeId,
        shape: &[u32],
        attrs: Attrs,
    ) -> Result<(), TpxError> {
        self.put_with_retention(key, data, dtype, shape, attrs, "keep_latest")
    }

    pub fn put_with_retention(
        &self,
        key: &str,
        data: &[u8],
        dtype: DTypeId,
        shape: &[u32],
        attrs: Attrs,
        retain: &str,
    ) -> Result<(), TpxError> {
        if self.mode == OpenMode::ReadOnly {
            return Err(TpxError::ReadOnlyMode);
        }

        let s_id = shard_for_key(key, self.shards.len() as u16);
        let shard = &self.shards[s_id as usize];

        // Content-defined chunking (FastCDC)
        let cdc_config = CdcConfig::default();
        let segments = cdc_chunk(data, &cdc_config);

        let mut chunk_refs = Vec::with_capacity(segments.len());
        let mut content_hashes = Vec::with_capacity(segments.len());

        let retention_policy = match retain {
            "keep_forever" => RetentionPolicy::KeepForever,
            s if s.starts_with("keep_last_n:") => {
                let n = s
                    .strip_prefix("keep_last_n:")
                    .unwrap_or("1")
                    .parse::<u32>()
                    .unwrap_or(1);
                RetentionPolicy::KeepLastN(n)
            }
            _ => RetentionPolicy::KeepLatest,
        };

        for seg in &segments {
            content_hashes.push(seg.content_hash);

            // Global content store check (dedup hit)
            if let Some(existing) = self.content_store.lookup(seg.content_hash) {
                let _ = self.content_store.add_ref(seg.content_hash);
                chunk_refs.push(ChunkRef {
                    shard_id: existing.shard_id,
                    chunk_offset: existing.chunk_offset,
                    chunk_len: existing.chunk_len,
                });
                continue;
            }

            // Dedup miss: filter + compress + append
            let raw_segment = &data[seg.offset..seg.offset + seg.len];
            let selection = tpx_codec::auto_select::auto_select(raw_segment, dtype, 250.0);

            let filter = get_filter(selection.filter);
            let filtered = filter.apply(raw_segment, dtype);

            let codec = tpx_codec::get_codec(selection.codec);
            let compressed = codec.compress(&filtered, selection.level, dtype)?;

            // Payload CRC32C
            let mut hasher = Hasher::new();
            hasher.update(&compressed);
            let crc = hasher.finalize();

            let chunk_header = ChunkHeader::new(
                selection.codec,
                selection.filter,
                1,
                seg.len as u32,
                seg.len as u32,
                crc,
            );

            let block_table = [BlockEntry {
                compressed_len: compressed.len() as u32,
            }];

            let mut chunk_bytes = chunk_header.serialize(&block_table);
            chunk_bytes.extend_from_slice(&compressed);
            pad_to_align(&mut chunk_bytes, self.header.chunk_align);

            let append_offset = self.io.append(&chunk_bytes)?;
            let chunk_len = chunk_bytes.len() as u32;

            let chunk_ref = ChunkRef {
                shard_id: s_id,
                chunk_offset: append_offset,
                chunk_len,
            };

            self.content_store.insert(
                seg.content_hash,
                ContentEntry::new(s_id, append_offset, chunk_len, 1),
            );

            chunk_refs.push(chunk_ref);
            self.checkpoint_coordinator
                .record_bytes_written(chunk_len as u64);
        }

        // Record in shard ART (for fast in-memory prefix lookup)
        if let Some(first_ref) = chunk_refs.first() {
            shard.art.insert(key.as_bytes(), *first_ref);
            shard.chunk_count.fetch_add(
                chunk_refs.len() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        // Update version chain
        {
            let mut vc_guard = shard.version_chains.write();
            let vc = vc_guard
                .entry(key.to_string())
                .or_insert_with(|| VersionChain::new(key, retention_policy));
            vc.retention = retention_policy;
            let (_v_id, evicted) = vc.append(
                dtype,
                shape.to_vec(),
                attrs.entries.clone().into_iter().collect(),
                content_hashes,
            );

            // Release evicted chunks in content store
            for h in evicted {
                let _ = self.content_store.release(h);
            }
        }

        // Update master TOC
        {
            let mut toc_guard = self.master_toc.write();
            let entry = TocEntry::new(
                key.to_string(),
                dtype,
                shape.to_vec(),
                attrs.entries.into_iter().collect(),
                chunk_refs,
            );
            toc_guard.add_entry(entry);
        }

        if self.checkpoint_coordinator.should_checkpoint() {
            self.checkpoint()?;
        }

        Ok(())
    }

    pub fn put_batch(&self, tensors: &[(&str, &[u8], DTypeId, &[u32])]) -> Result<(), TpxError> {
        if self.mode == OpenMode::ReadOnly {
            return Err(TpxError::ReadOnlyMode);
        }
        if tensors.is_empty() {
            return Ok(());
        }

        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let chunk_size = (tensors.len() + num_threads - 1) / num_threads;

        struct ProcessedChunk {
            content_hash: u128,
            chunk_bytes: Vec<u8>,
        }

        struct ProcessedTensor {
            shard_id: u16,
            chunks: Vec<ProcessedChunk>,
        }

        // Parallel chunking & compression across CPU cores via scoped threads
        let processed_tensors: Vec<ProcessedTensor> = std::thread::scope(|s| {
            let mut handles = Vec::new();
            for tensor_chunk in tensors.chunks(chunk_size) {
                let handle = s.spawn(move || {
                    let cdc_config = CdcConfig::default();
                    let mut chunk_results = Vec::with_capacity(tensor_chunk.len());

                    for &(key, data, dtype, _shape) in tensor_chunk {
                        let s_id = shard_for_key(key, self.shards.len() as u16);
                        let segments = cdc_chunk(data, &cdc_config);
                        let mut processed_chunks = Vec::with_capacity(segments.len());

                        for seg in &segments {
                            let raw_segment = &data[seg.offset..seg.offset + seg.len];
                            let selection =
                                tpx_codec::auto_select::auto_select(raw_segment, dtype, 250.0);

                            let filter = get_filter(selection.filter);
                            let filtered = filter.apply(raw_segment, dtype);

                            let codec = tpx_codec::get_codec(selection.codec);
                            let compressed = match codec.compress(&filtered, selection.level, dtype)
                            {
                                Ok(c) => c,
                                Err(_) => raw_segment.to_vec(),
                            };

                            let mut hasher = Hasher::new();
                            hasher.update(&compressed);
                            let crc = hasher.finalize();

                            let chunk_header = ChunkHeader::new(
                                selection.codec,
                                selection.filter,
                                1,
                                seg.len as u32,
                                seg.len as u32,
                                crc,
                            );

                            let block_table = [BlockEntry {
                                compressed_len: compressed.len() as u32,
                            }];

                            let mut chunk_bytes = chunk_header.serialize(&block_table);
                            chunk_bytes.extend_from_slice(&compressed);
                            pad_to_align(&mut chunk_bytes, self.header.chunk_align);

                            processed_chunks.push(ProcessedChunk {
                                content_hash: seg.content_hash,
                                chunk_bytes,
                            });
                        }

                        chunk_results.push(ProcessedTensor {
                            shard_id: s_id,
                            chunks: processed_chunks,
                        });
                    }

                    chunk_results
                });
                handles.push(handle);
            }

            let mut all = Vec::with_capacity(tensors.len());
            for h in handles {
                if let Ok(res) = h.join() {
                    all.extend(res);
                }
            }
            all
        });

        // Deduplication check, Coalesced Write Buffer, and Master TOC update
        let mut coalesced_buffer = Vec::new();
        struct PendingWrite {
            tensor_idx: usize,
            chunk_idx: usize,
            buffer_offset: usize,
            chunk_len: u32,
            content_hash: u128,
            shard_id: u16,
        }
        let mut pending_writes = Vec::new();
        let mut tensor_chunk_refs: Vec<Vec<Option<ChunkRef>>> = processed_tensors
            .iter()
            .map(|t| vec![None; t.chunks.len()])
            .collect();

        for (t_idx, pt) in processed_tensors.iter().enumerate() {
            for (c_idx, chunk) in pt.chunks.iter().enumerate() {
                if let Some(existing) = self.content_store.lookup(chunk.content_hash) {
                    let _ = self.content_store.add_ref(chunk.content_hash);
                    tensor_chunk_refs[t_idx][c_idx] = Some(ChunkRef {
                        shard_id: existing.shard_id,
                        chunk_offset: existing.chunk_offset,
                        chunk_len: existing.chunk_len,
                    });
                } else {
                    let buf_off = coalesced_buffer.len();
                    coalesced_buffer.extend_from_slice(&chunk.chunk_bytes);
                    pending_writes.push(PendingWrite {
                        tensor_idx: t_idx,
                        chunk_idx: c_idx,
                        buffer_offset: buf_off,
                        chunk_len: chunk.chunk_bytes.len() as u32,
                        content_hash: chunk.content_hash,
                        shard_id: pt.shard_id,
                    });
                }
            }
        }

        // Single sequential append of all new chunks
        let base_append_offset = if !coalesced_buffer.is_empty() {
            self.io.append(&coalesced_buffer)?
        } else {
            0
        };

        for pw in pending_writes {
            let actual_offset = base_append_offset + pw.buffer_offset as u64;
            let chunk_ref = ChunkRef {
                shard_id: pw.shard_id,
                chunk_offset: actual_offset,
                chunk_len: pw.chunk_len,
            };
            self.content_store.insert(
                pw.content_hash,
                ContentEntry::new(pw.shard_id, actual_offset, pw.chunk_len, 1),
            );
            tensor_chunk_refs[pw.tensor_idx][pw.chunk_idx] = Some(chunk_ref);
            self.checkpoint_coordinator
                .record_bytes_written(pw.chunk_len as u64);
        }

        // Update Shard ART, Version Chains, and Master TOC in a single batch
        {
            let mut toc_guard = self.master_toc.write();

            for (i, &(key, _, dtype, shape)) in tensors.iter().enumerate() {
                let pt = &processed_tensors[i];
                let shard = &self.shards[pt.shard_id as usize];
                let refs: Vec<ChunkRef> = tensor_chunk_refs[i].iter().filter_map(|r| *r).collect();

                if let Some(first_ref) = refs.first() {
                    shard.art.insert(key.as_bytes(), *first_ref);
                    shard
                        .chunk_count
                        .fetch_add(refs.len() as u64, std::sync::atomic::Ordering::Relaxed);
                }

                // Update version chain
                {
                    let mut vc_guard = shard.version_chains.write();
                    let vc = vc_guard
                        .entry(key.to_string())
                        .or_insert_with(|| VersionChain::new(key, RetentionPolicy::KeepLatest));
                    let content_hashes = pt.chunks.iter().map(|c| c.content_hash).collect();
                    let (_v_id, evicted) =
                        vc.append(dtype, shape.to_vec(), Vec::new(), content_hashes);
                    for h in evicted {
                        let _ = self.content_store.release(h);
                    }
                }

                // Add to master TOC
                let entry = TocEntry::new(key.to_string(), dtype, shape.to_vec(), Vec::new(), refs);
                toc_guard.add_entry(entry);
            }
        }

        if self.checkpoint_coordinator.should_checkpoint() {
            self.checkpoint()?;
        }

        Ok(())
    }

    pub fn read_tensor_from_entry(&self, entry: &TocEntry) -> Result<OwnedTensor, TpxError> {
        if entry.chunk_refs.is_empty() {
            return Ok(OwnedTensor::new(
                entry.dtype_id,
                entry.shape.clone(),
                Vec::new(),
            ));
        }

        let mut full_data = Vec::new();

        for chunk_ref in &entry.chunk_refs {
            let chunk_bytes = self
                .io
                .read_at(chunk_ref.chunk_offset, chunk_ref.chunk_len as usize)?;
            let (hdr, block_table) = ChunkHeader::deserialize(&chunk_bytes)?;

            let block_table_size = block_table.len() * 4;
            let payload_start = 16 + block_table_size;
            let compressed_len = block_table
                .iter()
                .map(|b| b.compressed_len as usize)
                .sum::<usize>();

            if payload_start + compressed_len > chunk_bytes.len() {
                return Err(TpxError::BufferTooSmall {
                    component: "ChunkPayload",
                    required: payload_start + compressed_len,
                    provided: chunk_bytes.len(),
                });
            }

            let payload = &chunk_bytes[payload_start..payload_start + compressed_len];

            // Verify CRC32C
            let mut hasher = Hasher::new();
            hasher.update(payload);
            let actual_crc = hasher.finalize();
            if actual_crc != hdr.payload_crc32c {
                return Err(TpxError::ChunkCrcMismatch {
                    offset: chunk_ref.chunk_offset,
                    expected: hdr.payload_crc32c,
                    got: actual_crc,
                });
            }

            let codec = tpx_codec::get_codec(hdr.codec_id);
            let decompressed =
                codec.decompress(payload, hdr.uncompressed_len as usize, entry.dtype_id)?;

            let filter = get_filter(hdr.filter_id);
            let reverted = filter.revert(&decompressed, entry.dtype_id);

            full_data.extend_from_slice(&reverted);
        }

        Ok(OwnedTensor::new(
            entry.dtype_id,
            entry.shape.clone(),
            full_data,
        ))
    }

    pub fn get(&self, key: &str) -> Result<Option<OwnedTensor>, TpxError> {
        let entry = {
            let toc_guard = self.master_toc.read();
            match toc_guard.lookup(key) {
                Some(e) => e.clone(),
                None => return Ok(None),
            }
        };

        self.read_tensor_from_entry(&entry).map(Some)
    }

    pub fn get_attrs(&self, key: &str) -> Result<HashMap<String, AttrValue>, TpxError> {
        let toc_guard = self.master_toc.read();
        let entry = toc_guard
            .lookup(key)
            .ok_or_else(|| TpxError::KeyNotFound(key.to_string()))?;
        Ok(entry.attrs.iter().cloned().collect())
    }

    pub fn get_prefix(&self, prefix: &str) -> Result<HashMap<String, OwnedTensor>, TpxError> {
        let matching_entries: Vec<TocEntry> = {
            let toc_guard = self.master_toc.read();
            toc_guard
                .lookup_prefix(prefix)
                .into_iter()
                .cloned()
                .collect()
        };

        let mut results = HashMap::new();
        for entry in matching_entries {
            if let Some(tensor) = self.get(&entry.key)? {
                results.insert(entry.key, tensor);
            }
        }
        Ok(results)
    }

    pub fn get_batch(&self, keys: &[&str]) -> Result<Vec<(String, OwnedTensor)>, TpxError> {
        let mut results = Vec::with_capacity(keys.len());
        for &key in keys {
            if let Some(tensor) = self.get(key)? {
                results.push((key.to_string(), tensor));
            }
        }
        Ok(results)
    }

    pub fn keys(&self) -> Result<Vec<String>, TpxError> {
        let toc_guard = self.master_toc.read();
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for entry in toc_guard.entries.iter().rev() {
            if !entry.key.is_empty() && seen.insert(entry.key.clone()) {
                if entry.tombstone == 0 {
                    result.push(entry.key.clone());
                }
            }
        }
        result.reverse();
        Ok(result)
    }

    pub fn delete(&self, key: &str) -> Result<(), TpxError> {
        if self.mode == OpenMode::ReadOnly {
            return Err(TpxError::ReadOnlyMode);
        }

        let s_id = shard_for_key(key, self.shards.len() as u16);
        let shard = &self.shards[s_id as usize];

        shard.art.remove(key.as_bytes());
        {
            let mut vc_guard = shard.version_chains.write();
            vc_guard.remove(key);
        }
        {
            let mut toc_guard = self.master_toc.write();
            toc_guard.tombstone(key);
        }
        Ok(())
    }

    pub fn list_versions(&self, key: &str) -> Result<Vec<u32>, TpxError> {
        let s_id = shard_for_key(key, self.shards.len() as u16);
        let shard = &self.shards[s_id as usize];
        {
            let vc_guard = shard.version_chains.read();
            if let Some(vc) = vc_guard.get(key) {
                if vc.versions.is_empty() {
                    return Err(TpxError::KeyNotFound(key.to_string()));
                }
                return Ok(vc.list_versions());
            }
        }

        let toc_guard = self.master_toc.read();
        if toc_guard.lookup(key).is_none() {
            return Err(TpxError::KeyNotFound(key.to_string()));
        }
        let count = toc_guard
            .entries
            .iter()
            .filter(|e| e.key == key && e.tombstone == 0)
            .count();
        if count == 0 {
            Err(TpxError::KeyNotFound(key.to_string()))
        } else {
            Ok((0..count as u32).collect())
        }
    }

    pub fn get_version(&self, key: &str, version: u32) -> Result<Option<OwnedTensor>, TpxError> {
        let s_id = shard_for_key(key, self.shards.len() as u16);
        let shard = &self.shards[s_id as usize];
        {
            let vc_guard = shard.version_chains.read();
            if let Some(vc) = vc_guard.get(key) {
                if !vc.list_versions().contains(&version) {
                    return Err(TpxError::VersionNotFound {
                        key: key.to_string(),
                        version,
                    });
                }
            }
        }

        let entry_opt = {
            let toc_guard = self.master_toc.read();
            if toc_guard.lookup(key).is_none() {
                return Err(TpxError::KeyNotFound(key.to_string()));
            }
            let key_entries: Vec<TocEntry> = toc_guard
                .entries
                .iter()
                .filter(|e| e.key == key)
                .cloned()
                .collect();

            if (version as usize) < key_entries.len() {
                Some(key_entries[version as usize].clone())
            } else {
                return Err(TpxError::VersionNotFound {
                    key: key.to_string(),
                    version,
                });
            }
        };

        match entry_opt {
            Some(entry) => self.read_tensor_from_entry(&entry).map(Some),
            None => Ok(None),
        }
    }

    pub fn diff(&self, key: &str, v1: u32, v2: u32) -> Result<VersionDiff, TpxError> {
        let s_id = shard_for_key(key, self.shards.len() as u16);
        let shard = &self.shards[s_id as usize];
        {
            let vc_guard = shard.version_chains.read();
            if let Some(vc) = vc_guard.get(key) {
                let valid = vc.list_versions();
                if !valid.contains(&v1) {
                    return Err(TpxError::VersionNotFound {
                        key: key.to_string(),
                        version: v1,
                    });
                }
                if !valid.contains(&v2) {
                    return Err(TpxError::VersionNotFound {
                        key: key.to_string(),
                        version: v2,
                    });
                }
            }
        }

        let toc_guard = self.master_toc.read();
        let key_entries: Vec<&TocEntry> =
            toc_guard.entries.iter().filter(|e| e.key == key).collect();

        if key_entries.is_empty() {
            return Err(TpxError::KeyNotFound(key.to_string()));
        }

        let e1 = key_entries
            .get(v1 as usize)
            .ok_or_else(|| TpxError::VersionNotFound {
                key: key.to_string(),
                version: v1,
            })?;
        let e2 = key_entries
            .get(v2 as usize)
            .ok_or_else(|| TpxError::VersionNotFound {
                key: key.to_string(),
                version: v2,
            })?;

        let hashes1: std::collections::HashSet<u128> = e1
            .chunk_refs
            .iter()
            .map(|c| c.chunk_offset as u128)
            .collect();
        let hashes2: std::collections::HashSet<u128> = e2
            .chunk_refs
            .iter()
            .map(|c| c.chunk_offset as u128)
            .collect();

        let added = hashes2.difference(&hashes1).copied().collect();
        let removed = hashes1.difference(&hashes2).copied().collect();

        Ok(VersionDiff { added, removed })
    }

    pub fn checkpoint(&self) -> Result<Footer, TpxError> {
        self.checkpoint_coordinator.execute_checkpoint(
            &self.shards,
            &*self.io,
            &self.master_toc,
            false,
        )
    }

    pub fn flush(&self) -> Result<(), TpxError> {
        let _ = self.checkpoint()?;
        self.io.sync()
    }

    pub fn compact(&self) -> Result<(), TpxError> {
        if self.mode == OpenMode::ReadOnly {
            return Err(TpxError::ReadOnlyMode);
        }

        let live_tensors: Vec<(String, OwnedTensor, HashMap<String, AttrValue>)> = {
            let toc_guard = self.master_toc.read();
            let mut items = Vec::new();
            for entry in &toc_guard.entries {
                if entry.tombstone == 0 {
                    if let Some(tensor) = self.get(&entry.key)? {
                        items.push((
                            entry.key.clone(),
                            tensor,
                            entry.attrs.iter().cloned().collect(),
                        ));
                    }
                }
            }
            items
        };

        let mut header_bytes = self.header.serialize().to_vec();
        pad_to_align(&mut header_bytes, self.header.chunk_align);
        self.io.write_at(0, &header_bytes)?;
        let initial_offset = header_bytes.len() as u64;
        self.io.truncate(initial_offset)?;

        for shard in &self.shards {
            shard
                .append_offset
                .store(initial_offset, std::sync::atomic::Ordering::SeqCst);
        }

        {
            let mut toc_guard = self.master_toc.write();
            *toc_guard = Toc::new();
        }

        for (key, tensor, attrs) in live_tensors {
            let mut attrs_obj = Attrs::new();
            for (k, v) in attrs {
                let _ = attrs_obj.set(k, v);
            }
            self.put(&key, &tensor.data, tensor.dtype, &tensor.shape, attrs_obj)?;
        }

        self.flush()?;
        Ok(())
    }

    pub fn close(&self) -> Result<(), TpxError> {
        let mut closed = self.is_closed.write();
        if *closed {
            return Ok(());
        }

        if self.mode != OpenMode::ReadOnly {
            let _ = self.checkpoint_coordinator.execute_checkpoint(
                &self.shards,
                &*self.io,
                &self.master_toc,
                true,
            )?;
        }

        self.io.close()?;
        *closed = true;
        Ok(())
    }
}

impl Drop for ShardedEngine {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
