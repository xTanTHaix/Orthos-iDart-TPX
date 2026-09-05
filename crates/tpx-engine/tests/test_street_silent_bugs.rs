use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use tempfile::tempdir;
use tpx_core::types::*;
use tpx_engine::database::TPXDatabase;

#[test]
fn test_silent_crc_corruption_detection() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("crc_detect.tpx");

    // 1. Create and write a valid tensor
    {
        let db = TPXDatabase::create(&db_path, Some(2)).unwrap();
        let payload = vec![0x33u8; 8192];
        db.put(
            "important_layer",
            &payload,
            DTypeId::UInt8,
            &[8192],
            None,
            None,
        )
        .unwrap();
        db.close().unwrap();
    }

    // 2. Corrupt a single bit inside the payload block on disk
    {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&db_path)
            .unwrap();

        // Header is at 0..4096, first chunk starts at 4096.
        // ChunkHeader (16B) + BlockTable (4B) = payload starts at 4096 + 20 = 4116.
        file.seek(SeekFrom::Start(4096 + 20 + 5)).unwrap();
        let mut b = [0u8; 1];
        file.read_exact(&mut b).unwrap();
        b[0] ^= 0x01; // flip 1 bit inside the compressed payload
        file.seek(SeekFrom::Start(4096 + 20 + 5)).unwrap();
        file.write_all(&b).unwrap();
        file.sync_all().unwrap();
    }

    // 3. Verify get() and get_batch() detect corruption with ChunkCrcMismatch
    {
        let db = TPXDatabase::open(&db_path, OpenMode::ReadOnly).unwrap();
        let get_err = db.get("important_layer");
        match get_err {
            Err(TpxError::ChunkCrcMismatch { .. }) => {
                // Expected: bit flip was caught by hardware CRC32C!
            }
            Err(e) => panic!("Expected ChunkCrcMismatch, got other error: {:?}", e),
            Ok(_) => panic!("Silent data corruption! CRC check failed to catch bit flip!"),
        }

        // get_batch must also catch and propagate the CRC error
        let batch_err = db.get_batch(&["important_layer"]);
        assert!(
            batch_err.is_err(),
            "get_batch must propagate CRC corruption!"
        );
    }
}

#[test]
fn test_silent_tombstone_overwrites_in_prefix_scan() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("tombstone_prefix.tpx");
    let db = TPXDatabase::create(&db_path, Some(2)).unwrap();

    // 1. Overwrite a key 4 times with different values
    for v in 1..=4 {
        let val = vec![v as u8; 32];
        db.put(
            "model/layer_0/weight",
            &val,
            DTypeId::UInt8,
            &[32],
            None,
            None,
        )
        .unwrap();
    }

    // 2. Delete the key (tombstone)
    db.delete("model/layer_0/weight").unwrap();

    // 3. Prefix scan must NOT resurrect old overwritten versions
    let prefix_items = db.get_prefix("model/layer_0/").unwrap();
    assert!(
        prefix_items.is_empty(),
        "Deleted key must not resurrect in get_prefix! Got: {:?}",
        prefix_items.keys()
    );

    // 4. Add adjacent sibling keys
    db.put(
        "model/layer_0/bias",
        &[10u8; 16],
        DTypeId::UInt8,
        &[16],
        None,
        None,
    )
    .unwrap();
    db.put(
        "model/layer_0/norm",
        &[20u8; 16],
        DTypeId::UInt8,
        &[16],
        None,
        None,
    )
    .unwrap();

    let prefix_items2 = db.get_prefix("model/layer_0/").unwrap();
    assert_eq!(prefix_items2.len(), 2);
    assert!(prefix_items2.contains_key("model/layer_0/bias"));
    assert!(prefix_items2.contains_key("model/layer_0/norm"));
    assert!(!prefix_items2.contains_key("model/layer_0/weight"));
}

#[test]
fn test_silent_dedup_refcount_under_version_eviction() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("dedup_evict.tpx");
    let db = TPXDatabase::create(&db_path, Some(2)).unwrap();

    // Identical data across 10 updates with KeepLastN(3)
    let weights = vec![1.234f32; 10_000]; // ~40KB
    let bytes = bytemuck::cast_slice(&weights);

    for _epoch in 1..=10 {
        db.put(
            "training/encoder/weight",
            bytes,
            DTypeId::Float32,
            &[10_000],
            None,
            Some("keep_last_n:3"),
        )
        .unwrap();
    }
    db.flush().unwrap();

    // Old versions (1..7) were evicted, but the shared dedup chunks are still in use by versions 8, 9, 10.
    // Verify that the latest version reads with 100% data integrity!
    let latest = db
        .get("training/encoder/weight")
        .unwrap()
        .expect("Must exist");
    assert_eq!(latest.data, bytes);

    // Verify version history length
    let versions = db.list_versions("training/encoder/weight").unwrap();
    assert_eq!(versions.len(), 3);
    assert_eq!(versions, vec![7, 8, 9]);

    // Read specific non-evicted version 8
    let v8 = db
        .get_version("training/encoder/weight", 8)
        .unwrap()
        .expect("Version 8 must exist");
    assert_eq!(v8.data, bytes);

    // Evicted version 2 must return VersionNotFound
    let v2_res = db.get_version("training/encoder/weight", 2);
    assert!(v2_res.is_err(), "Evicted version must return error");

    db.close().unwrap();
}

#[test]
fn test_silent_deep_hierarchy_and_multilingual_unicode_keys() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("deep_unicode.tpx");
    let db = TPXDatabase::create(&db_path, Some(4)).unwrap();

    // 1. 50-level deep hierarchy key
    let mut deep_key = String::new();
    for i in 0..50 {
        deep_key.push_str(&format!("level_{}/", i));
    }
    deep_key.push_str("leaf_weight");

    let val = vec![77u8; 100];
    db.put(&deep_key, &val, DTypeId::UInt8, &[100], None, None)
        .unwrap();

    // 2. Multilingual & Emoji keys
    let keys = vec![
        "โมเดล/โครงข่ายประสาทเทียม/เลเยอร์_1",   // Thai
        "モデル/ニューラルネットワーク/重み", // Japanese
        "模型/神经网络/权重_alpha",           // Chinese
        "neural/⚡/🚀/🧠/diamond",            // Emojis
    ];

    for (i, &k) in keys.iter().enumerate() {
        let d = vec![i as u8; 64];
        db.put(k, &d, DTypeId::UInt8, &[64], None, None).unwrap();
    }

    db.flush().unwrap();
    db.close().unwrap();

    // 3. Re-open and verify all deep and unicode keys
    let db_read = TPXDatabase::open(&db_path, OpenMode::ReadOnly).unwrap();
    let deep_res = db_read
        .get(&deep_key)
        .unwrap()
        .expect("Deep key must exist");
    assert_eq!(deep_res.data, val);

    for (i, &k) in keys.iter().enumerate() {
        let res = db_read.get(k).unwrap().expect("Unicode key must exist");
        assert_eq!(res.data, vec![i as u8; 64]);
    }
}

#[test]
fn test_silent_massive_scale_10000_keys() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("scale_10k.tpx");
    let db = TPXDatabase::create(&db_path, Some(8)).unwrap();

    let count = 10_000;
    // Rapidly insert 10,000 distinct tensor keys with varying prefixes
    for i in 0..count {
        let key = format!("cluster_{:02}/node_{:03}/tensor_{:04}", i % 10, i % 100, i);
        let data = (i as u32).to_le_bytes();
        db.put(&key, &data, DTypeId::UInt32, &[1], None, None)
            .unwrap();
    }

    db.flush().unwrap();
    db.close().unwrap();

    // Reopen and sample 500 keys
    let db_read = TPXDatabase::open(&db_path, OpenMode::ReadOnly).unwrap();
    for i in (0..count).step_by(20) {
        let key = format!("cluster_{:02}/node_{:03}/tensor_{:04}", i % 10, i % 100, i);
        let tensor = db_read.get(&key).unwrap().expect("Sampled key must exist");
        let expected = (i as u32).to_le_bytes();
        assert_eq!(tensor.data, expected);
    }

    // Prefix scan cluster_05/
    let cluster_5 = db_read.get_prefix("cluster_05/").unwrap();
    assert_eq!(cluster_5.len(), 1000);
}

#[test]
fn test_silent_pattern_predictor_and_batch_reader() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("pattern_batch.tpx");
    let db = TPXDatabase::create(&db_path, Some(4)).unwrap();

    // Write 20 sequential layers
    for i in 0..20 {
        let key = format!("model/layer_{:02}", i);
        let data = vec![i as u8; 32];
        db.put(&key, &data, DTypeId::UInt8, &[32], None, None)
            .unwrap();
    }
    db.flush().unwrap();

    let batch_reader = db.batch_reader();

    // Sequential training loop accesses
    let batch1 = ["model/layer_00", "model/layer_01", "model/layer_02"];
    let res1 = batch_reader.get_batch(&batch1).unwrap();
    assert_eq!(res1.len(), 3);

    let batch2 = ["model/layer_01", "model/layer_02", "model/layer_03"];
    let res2 = batch_reader.get_batch(&batch2).unwrap();
    assert_eq!(res2.len(), 3);

    // Predict next after layer_01 and layer_02
    let h1 = xxhash_rust::xxh64::xxh64(b"model/layer_01", 0);
    let h2 = xxhash_rust::xxh64::xxh64(b"model/layer_02", 0);
    let expected_h3 = xxhash_rust::xxh64::xxh64(b"model/layer_03", 0);

    let hist = batch_reader.history.lock();
    let predicted = hist.predictor.predict_next(h1, h2);
    assert_eq!(predicted, Some(expected_h3));

    db.close().unwrap();
}
