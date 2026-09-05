use std::sync::Arc;
use std::thread;
use tempfile::tempdir;
use tpx_codec::chimp128::{Chimp128Decoder, Chimp128Encoder};
use tpx_core::types::*;
use tpx_dedup::cdc::{cdc_chunk, CdcConfig};
use tpx_engine::database::TPXDatabase;
use tpx_filter::bit_shuffle::{bit_shuffle, bit_unshuffle};
use tpx_filter::byte_shuffle::{shuffle, unshuffle};
use tpx_filter::delta_zigzag::{decode as dz_decode, encode as dz_encode};
use tpx_index::art::ArtTree;

#[test]
fn test_street_chimp128_edge_cases() {
    // 1. Repeating values (verifying history buffer match and xor == 0)
    let repeating = vec![
        1.0f32, 2.0f32, 1.0f32, 2.0f32, 3.0f32, 1.0f32, 4.0f32, 2.0f32, 1.0f32, 0.0f32, 0.0f32,
    ];
    let mut encoder = Chimp128Encoder::new();
    for &val in &repeating {
        encoder.encode_f32(val);
    }
    let compressed = encoder.finish();
    let mut decoder = Chimp128Decoder::new(&compressed).unwrap();
    let mut decoded = Vec::new();
    while let Some(val) = decoder.decode_f32() {
        decoded.push(val);
    }
    assert_eq!(decoded, repeating);

    // 2. Float special values and boundaries: NaN, +Inf, -Inf, 0.0, -0.0, subnormals, min/max
    let subnormal = f32::from_bits(0x00000001);
    let special_f32 = vec![
        0.0f32,
        -0.0f32,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN,
        f32::MAX,
        f32::MIN_POSITIVE,
        subnormal,
        1e-30f32,
        1e30f32,
    ];
    let mut enc = Chimp128Encoder::new();
    for &val in &special_f32 {
        enc.encode_f32(val);
    }
    let comp = enc.finish();
    let mut dec = Chimp128Decoder::new(&comp).unwrap();
    let mut out = Vec::new();
    while let Some(v) = dec.decode_f32() {
        out.push(v);
    }
    assert_eq!(out.len(), special_f32.len());
    for (a, b) in out.iter().zip(special_f32.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // 3. 5,000 identical floats
    let uniform = vec![42.5f64; 5000];
    let mut enc64 = Chimp128Encoder::new();
    for &val in &uniform {
        enc64.encode_f64(val);
    }
    let comp64 = enc64.finish();
    let mut dec64 = Chimp128Decoder::new(&comp64).unwrap();
    let mut out64 = Vec::new();
    while let Some(v) = dec64.decode_f64() {
        out64.push(v);
    }
    assert_eq!(out64, uniform);
}

#[test]
fn test_street_filter_edge_cases() {
    // 1. ByteShuffle with unaligned remainder bytes
    for len in [
        1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 103,
    ] {
        let raw: Vec<u8> = (0..len).map(|i| (i * 31 % 256) as u8).collect();
        for elem_size in [1, 2, 4, 8] {
            let shuffled = shuffle(&raw, elem_size);
            let unshuffled = unshuffle(&shuffled, elem_size);
            assert_eq!(
                unshuffled, raw,
                "Failed at len {} elem_size {}",
                len, elem_size
            );
        }
    }

    // 2. BitShuffle with odd lengths
    for len in [1usize, 2, 3, 5, 7, 8, 9, 13, 16, 31, 64, 127] {
        let raw: Vec<u8> = (0..len).map(|i| ((i * 17 + 5) % 256) as u8).collect();
        let shuffled = bit_shuffle(&raw);
        let unshuffled = bit_unshuffle(&shuffled);
        assert_eq!(unshuffled, raw, "BitShuffle failed at len {}", len);
    }

    // 3. DeltaZigZag with extreme numbers and signed boundaries
    let extreme_i64 = vec![
        0i64,
        -1,
        1,
        i64::MIN,
        i64::MAX,
        i64::MIN + 1,
        i64::MAX - 1,
        -1_000_000_000_000i64,
        1_000_000_000_000i64,
    ];
    let bytes64 = bytemuck::cast_slice(&extreme_i64);
    let enc64 = dz_encode(bytes64, DTypeId::Int64);
    let dec64 = dz_decode(&enc64, DTypeId::Int64);
    assert_eq!(dec64, bytes64);

    let extreme_i32 = vec![
        0i32,
        -1,
        1,
        i32::MIN,
        i32::MAX,
        -500_000,
        500_000,
        i32::MIN + 1,
    ];
    let bytes32 = bytemuck::cast_slice(&extreme_i32);
    let enc32 = dz_encode(bytes32, DTypeId::Int32);
    let dec32 = dz_decode(&enc32, DTypeId::Int32);
    assert_eq!(dec32, bytes32);
}

#[test]
fn test_street_art_prefix_and_edge_keys() {
    let art = ArtTree::<u64>::new();

    // 1. Keys that are prefixes of other keys
    art.insert(b"model", 100);
    art.insert(b"model/encoder", 200);
    art.insert(b"model/encoder/layer_0", 300);
    art.insert(b"model/encoder/layer_0/weight", 400);
    art.insert(b"model/encoder/layer_0/weight_decay", 500);

    assert_eq!(art.get(b"model"), Some(100));
    assert_eq!(art.get(b"model/encoder"), Some(200));
    assert_eq!(art.get(b"model/encoder/layer_0"), Some(300));
    assert_eq!(art.get(b"model/encoder/layer_0/weight"), Some(400));
    assert_eq!(art.get(b"model/encoder/layer_0/weight_decay"), Some(500));
    assert_eq!(art.get(b"model/enco"), None);

    // 2. Empty string key and binary keys with embedded nulls
    art.insert(b"", 999);
    assert_eq!(art.get(b""), Some(999));

    let null_key = b"tensor\0with\0nulls";
    art.insert(null_key, 777);
    assert_eq!(art.get(null_key), Some(777));

    // 3. Remove prefix key without disturbing children
    let removed = art.remove(b"model/encoder");
    assert_eq!(removed, Some(200));
    assert_eq!(art.get(b"model/encoder"), None);
    assert_eq!(art.get(b"model"), Some(100));
    assert_eq!(art.get(b"model/encoder/layer_0"), Some(300));
    assert_eq!(art.get(b"model/encoder/layer_0/weight"), Some(400));

    // 4. Prefix scanning returns parent and children
    let prefix_items = art.get_prefix(b"model");
    let keys: Vec<Vec<u8>> = prefix_items.into_iter().map(|(k, _)| k).collect();
    assert!(keys.contains(&b"model".to_vec()));
    assert!(keys.contains(&b"model/encoder/layer_0".to_vec()));
    assert!(keys.contains(&b"model/encoder/layer_0/weight".to_vec()));
    assert!(!keys.contains(&b"model/encoder".to_vec())); // was removed
}

#[test]
fn test_street_cdc_dedup_stress() {
    let cfg = CdcConfig::default();

    // 1. Empty and tiny inputs
    assert!(cdc_chunk(&[], &cfg).is_empty());
    let single = cdc_chunk(&[42], &cfg);
    assert_eq!(single.len(), 1);
    assert_eq!(single[0].len, 1);

    // 2. Repetitive buffer (512 KB of identical bytes)
    let rep_data = vec![0xABu8; 512 * 1024];
    let chunks = cdc_chunk(&rep_data, &cfg);
    assert!(!chunks.is_empty());
    let total_len: usize = chunks.iter().map(|c| c.len).sum();
    assert_eq!(total_len, rep_data.len());
    // All chunks except possibly remainder should be within bounds
    for c in &chunks {
        assert!(c.len <= cfg.max_size);
    }
}

#[test]
fn test_street_database_concurrency_stress() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("concurrency_street.tpx");

    let db = Arc::new(TPXDatabase::create(&db_path, Some(8)).unwrap());

    // Spawn 16 threads doing concurrent puts, gets, and flushes
    let num_threads = 16;
    let ops_per_thread = 50;
    let mut handles = Vec::new();

    for t in 0..num_threads {
        let db_clone = Arc::clone(&db);
        let h = thread::spawn(move || {
            for i in 0..ops_per_thread {
                let key = format!("worker_{}/tensor_{}", t, i);
                let data = vec![(t as u8).wrapping_add(i as u8); 128];
                db_clone
                    .put(
                        &key,
                        &data,
                        DTypeId::UInt8,
                        &[128],
                        None,
                        Some("keep_latest"),
                    )
                    .unwrap();

                // Periodic read and flush
                if i % 10 == 0 {
                    let _ = db_clone.flush();
                    let read_back = db_clone.get(&key).unwrap();
                    assert!(read_back.is_some());
                }
            }
        });
        handles.push(h);
    }

    for h in handles {
        h.join().unwrap();
    }

    db.flush().unwrap();
    db.close().unwrap();

    // Verify all 800 keys can be read cleanly from a newly opened read-only database
    let db_read = TPXDatabase::open(&db_path, OpenMode::ReadOnly).unwrap();
    for t in 0..num_threads {
        for i in 0..ops_per_thread {
            let key = format!("worker_{}/tensor_{}", t, i);
            let tensor = db_read.get(&key).unwrap().expect("Key must exist");
            assert_eq!(tensor.data.len(), 128);
            assert_eq!(tensor.data[0], (t as u8).wrapping_add(i as u8));
        }
    }
}

#[test]
fn test_street_crash_recovery_and_compaction() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("crash_compact.tpx");

    // 1. Create and write 100 tensors, then flush a valid checkpoint
    {
        let db = TPXDatabase::create(&db_path, Some(4)).unwrap();
        for i in 0..100 {
            let key = format!("ckpt_key_{}", i);
            let val = vec![i as u8; 64];
            db.put(&key, &val, DTypeId::UInt8, &[64], None, Some("keep_latest"))
                .unwrap();
        }
        db.flush().unwrap();
        // Simulate crash: intentionally do NOT call db.close()
    }

    // 2. Simulate dirty crash by appending corrupted trailing bytes to the file
    {
        use std::fs::OpenOptions;
        use std::io::Write;
        let mut file = OpenOptions::new().append(true).open(&db_path).unwrap();
        let garbage = vec![0xDEu8, 0xAD, 0xBE, 0xEF].repeat(333); // 1332 bytes unaligned garbage
        file.write_all(&garbage).unwrap();
        file.sync_all().unwrap();
    }

    // 3. Open archive: verify backward footer recovery recovers to last checkpoint
    {
        let db = TPXDatabase::open(&db_path, OpenMode::ReadWrite).unwrap();
        for i in 0..100 {
            let key = format!("ckpt_key_{}", i);
            let tensor = db.get(&key).unwrap().expect("Must recover checkpoint key");
            assert_eq!(tensor.data[0], i as u8);
        }

        // Delete half the keys (tombstone 50 keys)
        for i in 0..50 {
            let key = format!("ckpt_key_{}", i);
            db.delete(&key).unwrap();
        }

        // Run compact to purge deleted chunks and reset file
        db.compact().unwrap();
        db.close().unwrap();
    }

    // 4. Re-open after compaction: deleted keys must be gone, remaining 50 keys intact
    {
        let db = TPXDatabase::open(&db_path, OpenMode::ReadOnly).unwrap();
        for i in 0..50 {
            let key = format!("ckpt_key_{}", i);
            assert_eq!(db.get(&key).unwrap(), None);
        }
        for i in 50..100 {
            let key = format!("ckpt_key_{}", i);
            let tensor = db.get(&key).unwrap().expect("Remaining key must exist");
            assert_eq!(tensor.data[0], i as u8);
        }
    }
}

#[test]
fn test_street_extreme_tensor_shapes_and_large_blobs() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("extreme_tensors.tpx");
    let db = TPXDatabase::create(&db_path, Some(4)).unwrap();

    // 1. Zero-byte tensor (empty data, shape [0])
    db.put("empty_tensor", &[], DTypeId::Float32, &[0], None, None)
        .unwrap();
    let empty_res = db.get("empty_tensor").unwrap().unwrap();
    assert_eq!(empty_res.data.len(), 0);
    assert_eq!(empty_res.shape, vec![0]);

    // 2. High-dimensional 8D tensor: shape [2, 1, 2, 1, 2, 1, 2, 1] (16 elements = 64 bytes)
    let shape_8d = vec![2, 1, 2, 1, 2, 1, 2, 1];
    let data_8d: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let bytes_8d = bytemuck::cast_slice(&data_8d);
    db.put(
        "tensor_8d",
        bytes_8d,
        DTypeId::Float32,
        &shape_8d,
        None,
        None,
    )
    .unwrap();

    let res_8d = db.get("tensor_8d").unwrap().unwrap();
    assert_eq!(res_8d.shape, shape_8d);
    assert_eq!(res_8d.data, bytes_8d);

    // 3. Multi-MB tensor (5 MB float32 tensor = 1,310,720 floats)
    let float_count = 1_310_720;
    let big_data: Vec<f32> = (0..float_count).map(|i| (i % 100) as f32 * 0.1).collect();
    let big_bytes = bytemuck::cast_slice(&big_data);
    db.put(
        "big_weights_5mb",
        big_bytes,
        DTypeId::Float32,
        &[float_count as u32],
        None,
        None,
    )
    .unwrap();

    db.flush().unwrap();

    let res_big = db.get("big_weights_5mb").unwrap().unwrap();
    assert_eq!(res_big.data.len(), big_bytes.len());
    assert_eq!(res_big.data, big_bytes);

    db.close().unwrap();
}
