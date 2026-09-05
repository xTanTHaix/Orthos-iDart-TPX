use xxhash_rust::xxh64::xxh64;

pub fn shard_for_key(key: &str, shard_count: u16) -> u16 {
    if shard_count <= 1 {
        return 0;
    }
    (xxh64(key.as_bytes(), 0) % (shard_count as u64)) as u16
}
