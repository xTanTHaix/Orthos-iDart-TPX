use tempfile::NamedTempFile;
use tpx_core::types::*;
use tpx_engine::database::TPXDatabase;
use tpx_version::chain::{RetentionPolicy, VersionChain};

#[test]
fn test_version_chain_append_and_list() {
    let mut chain = VersionChain::new("weights", RetentionPolicy::KeepLastN(3));
    let (v0, _) = chain.append(DTypeId::Float32, vec![10], vec![], vec![1, 2]);
    let (v1, _) = chain.append(DTypeId::Float32, vec![10], vec![], vec![2, 3]);
    let (v2, _) = chain.append(DTypeId::Float32, vec![10], vec![], vec![3, 4]);
    let (v3, evicted) = chain.append(DTypeId::Float32, vec![10], vec![], vec![4, 5]);

    assert_eq!(v0, 0);
    assert_eq!(v1, 1);
    assert_eq!(v2, 2);
    assert_eq!(v3, 3);
    assert_eq!(chain.list_versions(), vec![1, 2, 3]); // v0 was evicted by KeepLastN(3)
    assert_eq!(evicted, vec![1, 2]);
}

#[test]
fn test_version_diff() {
    let mut chain = VersionChain::new("bias", RetentionPolicy::KeepForever);
    chain.append(DTypeId::Float32, vec![5], vec![], vec![10, 20]);
    chain.append(DTypeId::Float32, vec![5], vec![], vec![20, 30]);

    let diff = chain.diff(0, 1).unwrap();
    assert_eq!(diff.added, vec![30]);
    assert_eq!(diff.removed, vec![10]);
}

#[test]
fn test_versioning_integration() {
    let tmp = NamedTempFile::new().unwrap();
    let db = TPXDatabase::create(tmp.path(), Some(2)).unwrap();

    let d1 = [1u8; 100];
    let d2 = [2u8; 100];

    db.put(
        "ckpt",
        &d1,
        DTypeId::UInt8,
        &[100],
        None,
        Some("keep_forever"),
    )
    .unwrap();
    db.put(
        "ckpt",
        &d2,
        DTypeId::UInt8,
        &[100],
        None,
        Some("keep_forever"),
    )
    .unwrap();

    let versions = db.list_versions("ckpt").unwrap();
    assert_eq!(versions.len(), 2);
}
