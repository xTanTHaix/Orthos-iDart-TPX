use tempfile::NamedTempFile;
use tpx_core::types::*;
use tpx_engine::batch::PatternPredictor;
use tpx_engine::database::TPXDatabase;

#[test]
fn test_pattern_predictor() {
    let mut predictor = PatternPredictor::new();
    let seq = vec![100u64, 200, 300, 100, 200, 300];
    predictor.record_sequence(&seq);

    let next = predictor.predict_next(100, 200);
    assert_eq!(next, Some(300));
}

#[test]
fn test_get_batch_all_keys() {
    let tmp = NamedTempFile::new().unwrap();
    let db = TPXDatabase::create(tmp.path(), Some(4)).unwrap();

    for i in 0..50 {
        let key = format!("batch_key_{}", i);
        db.put(&key, &[i as u8; 64], DTypeId::UInt8, &[64], None, None)
            .unwrap();
    }

    let keys: Vec<String> = (0..50).map(|i| format!("batch_key_{}", i)).collect();
    let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();

    let batch = db.get_batch(&key_refs).unwrap();
    assert_eq!(batch.len(), 50);

    for (i, (key, tensor)) in batch.iter().enumerate() {
        assert_eq!(key, &format!("batch_key_{}", i));
        assert_eq!(tensor.data[0], i as u8);
    }
}
