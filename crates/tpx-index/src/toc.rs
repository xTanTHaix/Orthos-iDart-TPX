use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashMap;
use std::io::{Cursor, Read, Write};
use tpx_core::attrs::{AttrValue, Attrs};
use tpx_core::types::{ChunkRef, DTypeId, TpxError};
use xxhash_rust::xxh64::xxh64;

use crate::radix_spline::RadixSpline;
use crate::surf::Surf;

#[derive(Clone, Debug, PartialEq)]
pub struct TocEntry {
    pub key_hash: u64,
    pub key: String,
    pub dtype_id: DTypeId,
    pub shape: Vec<u32>,
    pub attrs: Vec<(String, AttrValue)>,
    pub chunk_refs: Vec<ChunkRef>,
    pub tombstone: u8,
}

impl TocEntry {
    pub fn new(
        key: String,
        dtype_id: DTypeId,
        shape: Vec<u32>,
        attrs: Vec<(String, AttrValue)>,
        chunk_refs: Vec<ChunkRef>,
    ) -> Self {
        let key_hash = xxh64(key.as_bytes(), 0);
        Self {
            key_hash,
            key,
            dtype_id,
            shape,
            attrs,
            chunk_refs,
            tombstone: 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Toc {
    pub entries: Vec<TocEntry>,
    pub surf: Surf,
    pub radix_spline: RadixSpline,
    pub hash_index: HashMap<u64, u32>,
}

impl Toc {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build(entries: Vec<TocEntry>) -> Self {
        let mut hash_index = HashMap::with_capacity(entries.len());
        let mut keys_ref: Vec<&[u8]> = Vec::with_capacity(entries.len());
        let mut sorted_pairs: Vec<(u64, u32)> = Vec::with_capacity(entries.len());

        for (idx, entry) in entries.iter().enumerate() {
            hash_index.insert(entry.key_hash, idx as u32);
            keys_ref.push(entry.key.as_bytes());
            sorted_pairs.push((entry.key_hash, idx as u32));
        }

        sorted_pairs.sort_by_key(|&(k, _)| k);
        let sorted_keys: Vec<u64> = sorted_pairs.iter().map(|&(k, _)| k).collect();
        let positions: Vec<u32> = sorted_pairs.iter().map(|&(_, p)| p).collect();

        let surf = Surf::build(&keys_ref);
        let radix_spline = RadixSpline::build(&sorted_keys, &positions);

        Self {
            entries,
            surf,
            radix_spline,
            hash_index,
        }
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn lookup(&self, key: &str) -> Option<&TocEntry> {
        let hash = xxh64(key.as_bytes(), 0);
        let &idx = self.hash_index.get(&hash)?;
        let entry = self.entries.get(idx as usize)?;
        if entry.key == key && entry.tombstone == 0 {
            Some(entry)
        } else {
            None
        }
    }

    pub fn lookup_prefix(&self, prefix: &str) -> Vec<&TocEntry> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();

        // Iterate backwards so the latest entry for each key is inspected first
        for entry in self.entries.iter().rev() {
            if entry.key.starts_with(prefix) && seen.insert(&entry.key) {
                if entry.tombstone == 0 {
                    result.push(entry);
                }
            }
        }
        result
    }

    pub fn add_entry(&mut self, entry: TocEntry) {
        let hash = entry.key_hash;
        let idx = self.entries.len() as u32;
        self.hash_index.insert(hash, idx);
        self.entries.push(entry);
    }

    pub fn tombstone(&mut self, key: &str) {
        for entry in &mut self.entries {
            if entry.key == key {
                entry.tombstone = 1;
            }
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(self.entries.len() as u32)
            .unwrap();

        for entry in &self.entries {
            buf.write_u64::<LittleEndian>(entry.key_hash).unwrap();
            let key_bytes = entry.key.as_bytes();
            buf.write_u16::<LittleEndian>(key_bytes.len() as u16)
                .unwrap();
            buf.write_all(key_bytes).unwrap();

            buf.write_u8(entry.dtype_id as u8).unwrap();
            buf.write_u8(entry.shape.len() as u8).unwrap();
            for &dim in &entry.shape {
                buf.write_u32::<LittleEndian>(dim).unwrap();
            }

            // Attrs
            let mut attrs_obj = Attrs::new();
            for (k, v) in &entry.attrs {
                let _ = attrs_obj.set(k.clone(), v.clone());
            }
            let attrs_bytes = attrs_obj.serialize();
            buf.write_u32::<LittleEndian>(attrs_bytes.len() as u32)
                .unwrap();
            buf.write_all(&attrs_bytes).unwrap();

            // Chunk refs
            buf.write_u16::<LittleEndian>(entry.chunk_refs.len() as u16)
                .unwrap();
            for chunk_ref in &entry.chunk_refs {
                buf.write_u16::<LittleEndian>(chunk_ref.shard_id).unwrap();
                buf.write_u64::<LittleEndian>(chunk_ref.chunk_offset)
                    .unwrap();
                buf.write_u32::<LittleEndian>(chunk_ref.chunk_len).unwrap();
            }

            buf.write_u8(entry.tombstone).unwrap();
        }

        buf
    }

    pub fn deserialize(buf: &[u8]) -> Result<Self, TpxError> {
        let mut cursor = Cursor::new(buf);
        let num_entries = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| TpxError::Io(e))? as usize;

        let mut entries = Vec::with_capacity(num_entries);

        for _ in 0..num_entries {
            let key_hash = cursor.read_u64::<LittleEndian>()?;
            let key_len = cursor.read_u16::<LittleEndian>()? as usize;
            let mut key_bytes = vec![0u8; key_len];
            cursor.read_exact(&mut key_bytes)?;
            let key = String::from_utf8(key_bytes)
                .map_err(|e| TpxError::CorruptArchive(e.to_string()))?;

            let dtype_id_u8 = cursor.read_u8()?;
            let dtype_id = DTypeId::from_u8(dtype_id_u8)?;

            let ndim = cursor.read_u8()? as usize;
            let mut shape = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                shape.push(cursor.read_u32::<LittleEndian>()?);
            }

            let attrs_len = cursor.read_u32::<LittleEndian>()? as usize;
            let mut attrs_bytes = vec![0u8; attrs_len];
            cursor.read_exact(&mut attrs_bytes)?;
            let attrs_obj = Attrs::deserialize(&attrs_bytes)?;
            let attrs: Vec<(String, AttrValue)> = attrs_obj.entries.into_iter().collect();

            let chunk_count = cursor.read_u16::<LittleEndian>()? as usize;
            let mut chunk_refs = Vec::with_capacity(chunk_count);
            for _ in 0..chunk_count {
                let shard_id = cursor.read_u16::<LittleEndian>()?;
                let chunk_offset = cursor.read_u64::<LittleEndian>()?;
                let chunk_len = cursor.read_u32::<LittleEndian>()?;
                chunk_refs.push(ChunkRef {
                    shard_id,
                    chunk_offset,
                    chunk_len,
                });
            }

            let tombstone = cursor.read_u8()?;

            entries.push(TocEntry {
                key_hash,
                key,
                dtype_id,
                shape,
                attrs,
                chunk_refs,
                tombstone,
            });
        }

        Ok(Self::build(entries))
    }
}
