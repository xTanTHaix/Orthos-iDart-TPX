use std::sync::Arc;
use std::thread;
use tempfile::NamedTempFile;
use tpx_core::types::*;
use tpx_engine::database::TPXDatabase;

#[test]
fn test_concurrent_writes_across_shards() {
    let tmp = NamedTempFile::new().unwrap();
    let db = Arc::new(TPXDatabase::create(tmp.path(), Some(4)).unwrap());

    let mut handles = Vec::new();
    for t in 0..4 {
        let db_ref = db.clone();
        handles.push(thread::spawn(move || {
            for i in 0..25 {
                let key = format!("thread_{}/key_{}", t, i);
                let data = vec![(t * 25 + i) as u8; 256];
                db_ref
                    .put(&key, &data, DTypeId::UInt8, &[256], None, None)
                    .unwrap();
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // Verify all 100 keys were stored
    for t in 0..4 {
        for i in 0..25 {
            let key = format!("thread_{}/key_{}", t, i);
            let tensor = db.get(&key).unwrap().expect("tensor found");
            assert_eq!(tensor.data[0], (t * 25 + i) as u8);
        }
    }
}

#[test]
fn test_get_prefix_multi_shard() {
    let tmp = NamedTempFile::new().unwrap();
    let db = TPXDatabase::create(tmp.path(), Some(4)).unwrap();

    for i in 0..20 {
        let key = format!("models/sub/item_{}", i);
        db.put(&key, &[i as u8; 32], DTypeId::UInt8, &[32], None, None)
            .unwrap();
    }

    for i in 0..10 {
        let key = format!("other/item_{}", i);
        db.put(&key, &[i as u8; 32], DTypeId::UInt8, &[32], None, None)
            .unwrap();
    }

    let prefix_items = db.get_prefix("models/sub/").unwrap();
    assert_eq!(prefix_items.len(), 20);
}
