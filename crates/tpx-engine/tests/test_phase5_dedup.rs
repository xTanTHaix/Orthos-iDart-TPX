use tempfile::NamedTempFile;
use tpx_core::types::*;
use tpx_dedup::cdc::{cdc_chunk, CdcConfig};
use tpx_dedup::content_store::{ContentEntry, ContentStore};
use tpx_engine::database::TPXDatabase;

#[test]
fn test_cdc_chunking_bounds() {
    let config = CdcConfig::default();
    let data = vec![0xABu8; 1024 * 1024]; // 1 MB
    let chunks = cdc_chunk(&data, &config);
    assert!(!chunks.is_empty());
    for c in &chunks {
        assert!(c.len <= config.max_size);
    }
}

#[test]
fn test_content_store_refcounting() {
    let store = ContentStore::new();
    let hash = 0x12345678_9ABCDEF0_u128;
    let entry = ContentEntry::new(0, 1024, 2048, 1);

    assert!(store.insert(hash, entry));
    assert_eq!(store.add_ref(hash).unwrap(), 2);
    assert_eq!(store.release(hash).unwrap(), 1);
    assert_eq!(store.release(hash).unwrap(), 0);

    let gc = store.gc_candidates();
    assert_eq!(gc.len(), 1);
    assert_eq!(gc[0].0, hash);
}

#[test]
fn test_dedup_identical_writes_save_space() {
    let tmp = NamedTempFile::new().unwrap();
    let db = TPXDatabase::create(tmp.path(), Some(2)).unwrap();

    let data = vec![1.234f32; 50_000]; // ~200 KB
    let bytes = bytemuck::cast_slice(&data);

    db.put("tensor_v1", bytes, DTypeId::Float32, &[50_000], None, None)
        .unwrap();
    db.flush().unwrap();
    let size_v1 = std::fs::metadata(tmp.path()).unwrap().len();

    // Write exact same data under a new key
    db.put("tensor_v2", bytes, DTypeId::Float32, &[50_000], None, None)
        .unwrap();
    db.flush().unwrap();
    let size_v2 = std::fs::metadata(tmp.path()).unwrap().len();

    let growth = size_v2 - size_v1;
    // Because of dedup hit, second 200KB tensor payload is not re-written; only 1 aligned TOC block is added.
    assert!(
        growth <= 4096,
        "growth was {} bytes, expected <= 4096 for 200KB payload",
        growth
    );
}
