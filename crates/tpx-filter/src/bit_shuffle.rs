/// Bit-level shuffle filter.
/// Transposes bits across byte streams. Useful for boolean and low-entropy bitmasks.

pub fn bit_shuffle(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    let n = data.len();
    let mut out = vec![0u8; n];

    // For every byte out, bit b comes from data[(out_idx * 8 + b) / 8]
    // Bit transpose across 8 bit-planes
    for bit_plane in 0..8 {
        for (i, &byte_val) in data.iter().enumerate() {
            let bit = (byte_val >> bit_plane) & 1;
            let dest_bit_pos = bit_plane * n + i;
            let dest_byte = dest_bit_pos / 8;
            let dest_bit_in_byte = dest_bit_pos % 8;
            if dest_byte < n {
                out[dest_byte] |= bit << dest_bit_in_byte;
            }
        }
    }

    out
}

pub fn bit_unshuffle(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    let n = data.len();
    let mut out = vec![0u8; n];

    for bit_plane in 0..8 {
        for i in 0..n {
            let src_bit_pos = bit_plane * n + i;
            let src_byte = src_bit_pos / 8;
            let src_bit_in_byte = src_bit_pos % 8;
            if src_byte < n {
                let bit = (data[src_byte] >> src_bit_in_byte) & 1;
                out[i] |= bit << bit_plane;
            }
        }
    }

    out
}
