use tempfile::NamedTempFile;
use tpx_core::attrs::{AttrValue, GlobalAttrs};
use tpx_core::chunk::{pad_to_align, BlockEntry, ChunkHeader};
use tpx_core::footer::Footer;
use tpx_core::header::FileHeader;
use tpx_core::types::*;
use tpx_engine::database::TPXDatabase;

#[test]
fn test_header_magic_round_trip() {
    let h = FileHeader::new(4, 4096, FLAG_CHECKSUMMED);
    let bytes = h.serialize();
    let h2 = FileHeader::deserialize(&bytes).expect("deserialize header");
    assert_eq!(h.magic, h2.magic);
    assert_eq!(h.format_version, h2.format_version);
    assert_eq!(h.flags, h2.flags);
    assert_eq!(h.shard_count, h2.shard_count);
}

#[test]
fn test_header_rejects_bad_magic() {
    let mut bytes = FileHeader::default().serialize();
    bytes[0] = b'X';
    assert!(matches!(
        FileHeader::deserialize(&bytes),
        Err(TpxError::InvalidMagic { .. })
    ));
}

#[test]
fn test_footer_round_trip() {
    let f = Footer::new(12345, 42, 0xDEADBEEF, false);
    let bytes = f.serialize();
    let f2 = Footer::deserialize(&bytes).expect("deserialize footer");
    assert_eq!(f.toc_offset, f2.toc_offset);
    assert_eq!(f.total_chunks, f2.total_chunks);
    assert_eq!(f.checksum, f2.checksum);
    assert!(f2.is_final());
}

#[test]
fn test_footer_checkpoint_flag() {
    let f = Footer::new(500, 10, 0x1234, true);
    assert!(f.is_checkpoint());
    assert!(!f.is_final());
    let bytes = f.serialize();
    let f2 = Footer::deserialize(&bytes).unwrap();
    assert!(f2.is_checkpoint());
}

#[test]
fn test_global_attrs_tlv_round_trip() {
    let mut attrs = GlobalAttrs::new();
    attrs
        .set("producer", AttrValue::String("tpx-test-1.0".into()))
        .unwrap();
    attrs
        .set("created_at", AttrValue::Int64(1700000000))
        .unwrap();
    attrs
        .set("ratio", AttrValue::Float64(std::f64::consts::PI))
        .unwrap();
    attrs.set("flag", AttrValue::Bool(true)).unwrap();
    attrs
        .set("blob", AttrValue::Bytes(vec![0xCA, 0xFE, 0xBA, 0xBE]))
        .unwrap();

    let bytes = attrs.serialize();
    let attrs2 = GlobalAttrs::deserialize(&bytes).unwrap();
    assert_eq!(attrs.len(), attrs2.len());
    assert_eq!(
        attrs2.get("producer"),
        Some(&AttrValue::String("tpx-test-1.0".into()))
    );
    assert_eq!(attrs2.get("flag"), Some(&AttrValue::Bool(true)));
}

#[test]
fn test_chunk_header_round_trip() {
    let header = ChunkHeader::new(CodecId::Raw, FilterId::None, 3, 4096, 12288, 0xCAFEBABE);
    let table = vec![
        BlockEntry {
            compressed_len: 4096,
        },
        BlockEntry {
            compressed_len: 4096,
        },
        BlockEntry {
            compressed_len: 4096,
        },
    ];
    let bytes = header.serialize(&table);
    let (h2, t2) = ChunkHeader::deserialize(&bytes).unwrap();
    assert_eq!(header, h2);
    assert_eq!(table, t2);
}

#[test]
fn test_pad_to_align_boundary() {
    let mut buf = vec![0u8; 5000];
    pad_to_align(&mut buf, 4096);
    assert_eq!(buf.len(), 8192);
}

#[test]
fn test_pad_already_aligned_is_noop() {
    let mut buf = vec![0u8; 8192];
    pad_to_align(&mut buf, 4096);
    assert_eq!(buf.len(), 8192);
}

#[test]
fn test_create_put_get_close_reopen_get() {
    let tmp = NamedTempFile::new().unwrap();
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let bytes = bytemuck::cast_slice(&data);

    {
        let db = TPXDatabase::create(tmp.path(), Some(2)).unwrap();
        db.put("test/tensor", bytes, DTypeId::Float32, &[4], None, None)
            .unwrap();
        db.close().unwrap();
    }
    {
        let db = TPXDatabase::open(tmp.path(), OpenMode::ReadOnly).unwrap();
        let tensor = db.get("test/tensor").unwrap().expect("tensor found");
        assert_eq!(tensor.dtype, DTypeId::Float32);
        assert_eq!(tensor.shape, vec![4]);
        let result: &[f32] = bytemuck::cast_slice(&tensor.data);
        assert_eq!(result, &data[..]);
    }
}
