use tpx_core::types::*;
use tpx_index::art::ArtTree;
use tpx_index::radix_spline::RadixSpline;
use tpx_index::surf::Surf;
use tpx_index::toc::{Toc, TocEntry};

#[test]
fn test_art_insert_get() {
    let tree = ArtTree::new();
    tree.insert(
        b"models/transformer/q",
        ChunkRef {
            shard_id: 0,
            chunk_offset: 100,
            chunk_len: 200,
        },
    );
    tree.insert(
        b"models/transformer/k",
        ChunkRef {
            shard_id: 0,
            chunk_offset: 300,
            chunk_len: 200,
        },
    );
    tree.insert(
        b"data/images",
        ChunkRef {
            shard_id: 1,
            chunk_offset: 500,
            chunk_len: 1000,
        },
    );

    let q = tree.get(b"models/transformer/q").unwrap();
    assert_eq!(q.chunk_offset, 100);

    let prefix = tree.get_prefix(b"models/");
    assert_eq!(prefix.len(), 2);
}

#[test]
fn test_art_remove() {
    let tree = ArtTree::new();
    tree.insert(b"k1", 100);
    assert_eq!(tree.get(b"k1"), Some(100));
    let removed = tree.remove(b"k1");
    assert_eq!(removed, Some(100));
    assert_eq!(tree.get(b"k1"), None);
}

#[test]
fn test_surf_no_false_negatives() {
    let keys = vec![
        b"apple".as_slice(),
        b"banana".as_slice(),
        b"cherry".as_slice(),
    ];
    let surf = Surf::build(&keys);

    assert!(surf.range_lookup(b"apple", b"apple"));
    assert!(surf.range_lookup(b"banana", b"banana"));
    assert!(surf.range_lookup(b"cherry", b"cherry"));
    assert!(!surf.range_lookup(b"zebra", b"zebra"));
}

#[test]
fn test_radix_spline() {
    let keys = vec![10u64, 20, 30, 40, 50];
    let positions = vec![0u32, 100, 200, 300, 400];
    let rs = RadixSpline::build(&keys, &positions);

    let (lo, hi) = rs.lookup(30);
    assert!(lo <= 200 && 200 <= hi);
}

#[test]
fn test_toc_serialize_deserialize() {
    let mut entries = Vec::new();
    for i in 0..10 {
        entries.push(TocEntry::new(
            format!("layer_{}", i),
            DTypeId::Float32,
            vec![128, 128],
            vec![("id".into(), tpx_core::attrs::AttrValue::Int64(i as i64))],
            vec![ChunkRef {
                shard_id: 0,
                chunk_offset: i as u64 * 4096,
                chunk_len: 4096,
            }],
        ));
    }

    let toc = Toc::build(entries);
    let bytes = toc.serialize();
    let toc2 = Toc::deserialize(&bytes).unwrap();

    assert_eq!(toc.entry_count(), toc2.entry_count());
    let e = toc2.lookup("layer_3").unwrap();
    assert_eq!(e.shape, vec![128, 128]);
}
