use tpx_codec::chimp128::{Chimp128Decoder, Chimp128Encoder};
use tpx_codec::lz4_codec::Lz4Codec;
use tpx_codec::zstd_codec::ZstdCodec;
use tpx_core::types::*;
use tpx_filter::byte_shuffle::{shuffle, unshuffle};
use tpx_filter::delta_zigzag::{decode as dz_decode, encode as dz_encode};

#[test]
fn test_byte_shuffle_round_trip() {
    let raw = (0..1024u32).map(|i| i * 17).collect::<Vec<u32>>();
    let bytes = bytemuck::cast_slice(&raw);
    let shuffled = shuffle(bytes, 4);
    let unshuffled = unshuffle(&shuffled, 4);
    assert_eq!(unshuffled, bytes);
}

#[test]
fn test_delta_zigzag_round_trip_i32() {
    let raw = (0..500i32).map(|i| i * 3 - 250).collect::<Vec<i32>>();
    let bytes = bytemuck::cast_slice(&raw);
    let encoded = dz_encode(bytes, DTypeId::Int32);
    let decoded = dz_decode(&encoded, DTypeId::Int32);
    assert_eq!(decoded, bytes);
}

#[test]
fn test_delta_zigzag_round_trip_i64() {
    let raw = (0..500i64).map(|i| i * 99).collect::<Vec<i64>>();
    let bytes = bytemuck::cast_slice(&raw);
    let encoded = dz_encode(bytes, DTypeId::Int64);
    let decoded = dz_decode(&encoded, DTypeId::Int64);
    assert_eq!(decoded, bytes);
}

#[test]
fn test_chimp128_f32_round_trip() {
    let floats = (0..2000).map(|i| (i as f32) * 0.05).collect::<Vec<f32>>();
    let mut encoder = Chimp128Encoder::new();
    for &f in &floats {
        encoder.encode_f32(f);
    }
    let compressed = encoder.finish();

    let mut decoder = Chimp128Decoder::new(&compressed).unwrap();
    let mut decoded = Vec::new();
    while let Some(val) = decoder.decode_f32() {
        decoded.push(val);
    }

    assert_eq!(decoded.len(), floats.len());
    assert_eq!(decoded, floats);
}

#[test]
fn test_chimp128_f64_round_trip() {
    let doubles = (0..1000)
        .map(|i| (i as f64).sin() * 100.0)
        .collect::<Vec<f64>>();
    let mut encoder = Chimp128Encoder::new();
    for &d in &doubles {
        encoder.encode_f64(d);
    }
    let compressed = encoder.finish();

    let mut decoder = Chimp128Decoder::new(&compressed).unwrap();
    let mut decoded = Vec::new();
    while let Some(val) = decoder.decode_f64() {
        decoded.push(val);
    }

    assert_eq!(decoded.len(), doubles.len());
    assert_eq!(decoded, doubles);
}

#[test]
fn test_zstd_round_trip() {
    let data = vec![42u8; 65536];
    let compressed = ZstdCodec::compress(&data, 3).unwrap();
    assert!(compressed.len() < data.len());
    let decompressed = ZstdCodec::decompress(&compressed, data.len()).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn test_lz4_round_trip() {
    let data = vec![99u8; 65536];
    let compressed = Lz4Codec::compress(&data).unwrap();
    assert!(compressed.len() < data.len());
    let decompressed = Lz4Codec::decompress(&compressed, data.len()).unwrap();
    assert_eq!(decompressed, data);
}
