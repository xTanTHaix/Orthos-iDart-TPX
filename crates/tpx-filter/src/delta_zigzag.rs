use tpx_core::types::DTypeId;

/// Delta-ZigZag encoding for integer data.
/// Computes differences between consecutive elements and encodes with ZigZag,
/// turning monotonic or near-monotonic values into small positive integers.

pub fn encode(data: &[u8], dtype: DTypeId) -> Vec<u8> {
    match dtype {
        DTypeId::Int64 | DTypeId::UInt64 => {
            let elem_size = 8;
            let count = data.len() / elem_size;
            if count == 0 {
                return data.to_vec();
            }

            let mut out = vec![0u8; data.len()];
            let mut prev: i64 = 0;

            for i in 0..count {
                let start = i * 8;
                let val = i64::from_le_bytes(data[start..start + 8].try_into().unwrap());
                let delta = val.wrapping_sub(prev);
                let zigzag = (delta as u64).wrapping_shl(1) ^ ((delta >> 63) as u64);
                out[start..start + 8].copy_from_slice(&zigzag.to_le_bytes());
                prev = val;
            }

            let remainder = data.len() % 8;
            if remainder > 0 {
                let tail = count * 8;
                out[tail..].copy_from_slice(&data[tail..]);
            }
            out
        }
        DTypeId::Int32 | DTypeId::UInt32 => {
            let elem_size = 4;
            let count = data.len() / elem_size;
            if count == 0 {
                return data.to_vec();
            }

            let mut out = vec![0u8; data.len()];
            let mut prev: i32 = 0;

            for i in 0..count {
                let start = i * 4;
                let val = i32::from_le_bytes(data[start..start + 4].try_into().unwrap());
                let delta = val.wrapping_sub(prev);
                let zigzag = (delta as u32).wrapping_shl(1) ^ ((delta >> 31) as u32);
                out[start..start + 4].copy_from_slice(&zigzag.to_le_bytes());
                prev = val;
            }

            let remainder = data.len() % 4;
            if remainder > 0 {
                let tail = count * 4;
                out[tail..].copy_from_slice(&data[tail..]);
            }
            out
        }
        DTypeId::Int16 | DTypeId::UInt16 => {
            let elem_size = 2;
            let count = data.len() / elem_size;
            if count == 0 {
                return data.to_vec();
            }

            let mut out = vec![0u8; data.len()];
            let mut prev: i16 = 0;

            for i in 0..count {
                let start = i * 2;
                let val = i16::from_le_bytes(data[start..start + 2].try_into().unwrap());
                let delta = val.wrapping_sub(prev);
                let zigzag = (delta as u16).wrapping_shl(1) ^ ((delta >> 15) as u16);
                out[start..start + 2].copy_from_slice(&zigzag.to_le_bytes());
                prev = val;
            }

            let remainder = data.len() % 2;
            if remainder > 0 {
                let tail = count * 2;
                out[tail..].copy_from_slice(&data[tail..]);
            }
            out
        }
        DTypeId::Int8 | DTypeId::UInt8 | DTypeId::Bool => {
            let count = data.len();
            if count == 0 {
                return data.to_vec();
            }

            let mut out = vec![0u8; count];
            let mut prev: i8 = 0;

            for i in 0..count {
                let val = data[i] as i8;
                let delta = val.wrapping_sub(prev);
                let zigzag = (delta as u8).wrapping_shl(1) ^ ((delta >> 7) as u8);
                out[i] = zigzag;
                prev = val;
            }
            out
        }
        _ => data.to_vec(),
    }
}

pub fn decode(data: &[u8], dtype: DTypeId) -> Vec<u8> {
    match dtype {
        DTypeId::Int64 | DTypeId::UInt64 => {
            let count = data.len() / 8;
            if count == 0 {
                return data.to_vec();
            }

            let mut out = vec![0u8; data.len()];
            let mut prev: i64 = 0;

            for i in 0..count {
                let start = i * 8;
                let zigzag = u64::from_le_bytes(data[start..start + 8].try_into().unwrap());
                let delta = ((zigzag >> 1) as i64) ^ (-((zigzag & 1) as i64));
                let val = prev.wrapping_add(delta);
                out[start..start + 8].copy_from_slice(&val.to_le_bytes());
                prev = val;
            }

            let remainder = data.len() % 8;
            if remainder > 0 {
                let tail = count * 8;
                out[tail..].copy_from_slice(&data[tail..]);
            }
            out
        }
        DTypeId::Int32 | DTypeId::UInt32 => {
            let count = data.len() / 4;
            if count == 0 {
                return data.to_vec();
            }

            let mut out = vec![0u8; data.len()];
            let mut prev: i32 = 0;

            for i in 0..count {
                let start = i * 4;
                let zigzag = u32::from_le_bytes(data[start..start + 4].try_into().unwrap());
                let delta = ((zigzag >> 1) as i32) ^ (-((zigzag & 1) as i32));
                let val = prev.wrapping_add(delta);
                out[start..start + 4].copy_from_slice(&val.to_le_bytes());
                prev = val;
            }

            let remainder = data.len() % 4;
            if remainder > 0 {
                let tail = count * 4;
                out[tail..].copy_from_slice(&data[tail..]);
            }
            out
        }
        DTypeId::Int16 | DTypeId::UInt16 => {
            let count = data.len() / 2;
            if count == 0 {
                return data.to_vec();
            }

            let mut out = vec![0u8; data.len()];
            let mut prev: i16 = 0;

            for i in 0..count {
                let start = i * 2;
                let zigzag = u16::from_le_bytes(data[start..start + 2].try_into().unwrap());
                let delta = ((zigzag >> 1) as i16) ^ (-((zigzag & 1) as i16));
                let val = prev.wrapping_add(delta);
                out[start..start + 2].copy_from_slice(&val.to_le_bytes());
                prev = val;
            }

            let remainder = data.len() % 2;
            if remainder > 0 {
                let tail = count * 2;
                out[tail..].copy_from_slice(&data[tail..]);
            }
            out
        }
        DTypeId::Int8 | DTypeId::UInt8 | DTypeId::Bool => {
            let count = data.len();
            if count == 0 {
                return data.to_vec();
            }

            let mut out = vec![0u8; count];
            let mut prev: i8 = 0;

            for i in 0..count {
                let zigzag = data[i];
                let delta = ((zigzag >> 1) as i8) ^ (-((zigzag & 1) as i8));
                let val = prev.wrapping_add(delta);
                out[i] = val as u8;
                prev = val;
            }
            out
        }
        _ => data.to_vec(),
    }
}
