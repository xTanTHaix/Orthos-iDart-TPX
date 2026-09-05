/// Byte-level shuffle filter (transposing bytes of elements of size `elem_size`).
/// Turns repeating patterns in same-significance bytes into long contiguous runs,
/// greatly increasing compression ratio for floating-point and integer tensors.

pub fn shuffle_scalar(data: &[u8], elem_size: usize) -> Vec<u8> {
    if data.is_empty() || elem_size <= 1 {
        return data.to_vec();
    }

    let num_elements = data.len() / elem_size;
    let remainder = data.len() % elem_size;
    let mut out = vec![0u8; data.len()];

    for b in 0..elem_size {
        let dest_offset = b * num_elements;
        for i in 0..num_elements {
            out[dest_offset + i] = data[i * elem_size + b];
        }
    }

    // Handle any trailing unaligned remainder bytes
    if remainder > 0 {
        let tail_start = num_elements * elem_size;
        out[tail_start..].copy_from_slice(&data[tail_start..]);
    }

    out
}

pub fn unshuffle_scalar(data: &[u8], elem_size: usize) -> Vec<u8> {
    if data.is_empty() || elem_size <= 1 {
        return data.to_vec();
    }

    let num_elements = data.len() / elem_size;
    let remainder = data.len() % elem_size;
    let mut out = vec![0u8; data.len()];

    for b in 0..elem_size {
        let src_offset = b * num_elements;
        for i in 0..num_elements {
            out[i * elem_size + b] = data[src_offset + i];
        }
    }

    if remainder > 0 {
        let tail_start = num_elements * elem_size;
        out[tail_start..].copy_from_slice(&data[tail_start..]);
    }

    out
}

#[inline]
pub fn shuffle(data: &[u8], elem_size: usize) -> Vec<u8> {
    // For x86_64, AVX2/AVX-512 can be hooked here; scalar fallback is 100% correct
    shuffle_scalar(data, elem_size)
}

#[inline]
pub fn unshuffle(data: &[u8], elem_size: usize) -> Vec<u8> {
    unshuffle_scalar(data, elem_size)
}
