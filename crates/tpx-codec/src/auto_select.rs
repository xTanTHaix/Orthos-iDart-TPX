use tpx_core::types::{CodecId, DTypeId, FilterId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodecSelection {
    pub codec: CodecId,
    pub filter: FilterId,
    pub level: i32,
}

impl Default for CodecSelection {
    fn default() -> Self {
        Self {
            codec: CodecId::Zstd,
            filter: FilterId::ByteShuffle,
            level: 3,
        }
    }
}

pub fn auto_select(sample: &[u8], dtype: DTypeId, target_mb_s_floor: f64) -> CodecSelection {
    if sample.is_empty() {
        return CodecSelection {
            codec: CodecId::Raw,
            filter: FilterId::None,
            level: 0,
        };
    }

    // High throughput requirement: LZ4 dominates (3000+ MB/s)
    if target_mb_s_floor >= 200.0 {
        return CodecSelection {
            codec: CodecId::Lz4,
            filter: FilterId::None,
            level: 1,
        };
    }

    // Floating-point types: evaluate Chimp128 vs LZ4 vs ByteShuffle+Zstd
    if dtype == DTypeId::Float32 || dtype == DTypeId::Float64 {
        // Quick correlation check between consecutive floats for XOR-based compression
        if is_correlated_float(sample, dtype) {
            return CodecSelection {
                codec: CodecId::Chimp128,
                filter: FilterId::None,
                level: 0,
            };
        }
        if estimate_entropy(sample) > 7.6 {
            return CodecSelection {
                codec: CodecId::Lz4,
                filter: FilterId::None,
                level: 1,
            };
        }
        return CodecSelection {
            codec: CodecId::Zstd,
            filter: FilterId::ByteShuffle,
            level: 3,
        };
    }

    // Integer types: check for monotonic or near-monotonic trend
    if is_monotonic_trend(sample, dtype) {
        return CodecSelection {
            codec: CodecId::Zstd,
            filter: FilterId::DeltaZigZag,
            level: 3,
        };
    }

    // Check entropy: if random/incompressible, choose Raw to avoid wasting CPU
    let entropy = estimate_entropy(sample);
    if entropy > 7.6 {
        return CodecSelection {
            codec: CodecId::Raw,
            filter: FilterId::None,
            level: 0,
        };
    }

    // General default
    CodecSelection {
        codec: CodecId::Zstd,
        filter: FilterId::ByteShuffle,
        level: 3,
    }
}

fn estimate_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut counts = [0usize; 256];
    let len = data.len().min(4096);
    for &b in &data[..len] {
        counts[b as usize] += 1;
    }

    let mut entropy = 0.0;
    let total = len as f64;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / total;
            entropy -= p * p.log2();
        }
    }
    entropy
}

fn is_correlated_float(data: &[u8], dtype: DTypeId) -> bool {
    let elem_size = dtype.element_size();
    let count = (data.len() / elem_size).min(64);
    if count < 4 {
        return false;
    }

    let mut high_similarity_count = 0;
    if dtype == DTypeId::Float32 {
        let mut prev = u32::from_le_bytes(data[0..4].try_into().unwrap_or([0; 4]));
        for i in 1..count {
            let curr = u32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap_or([0; 4]));
            if (prev ^ curr).leading_zeros() >= 12 {
                high_similarity_count += 1;
            }
            prev = curr;
        }
    } else {
        let mut prev = u64::from_le_bytes(data[0..8].try_into().unwrap_or([0; 8]));
        for i in 1..count {
            let curr = u64::from_le_bytes(data[i * 8..i * 8 + 8].try_into().unwrap_or([0; 8]));
            if (prev ^ curr).leading_zeros() >= 16 {
                high_similarity_count += 1;
            }
            prev = curr;
        }
    }

    // Require at least 75% of consecutive elements to share high leading bits
    high_similarity_count >= (count * 3 / 4)
}

fn is_monotonic_trend(data: &[u8], dtype: DTypeId) -> bool {
    if dtype.element_size() < 2 {
        return false;
    }
    let count = (data.len() / dtype.element_size()).min(64);
    if count < 8 {
        return false;
    }

    let mut monotonic_steps = 0;
    let mut prev = 0i64;

    for i in 0..count {
        let val = match dtype.element_size() {
            4 => {
                let bytes: [u8; 4] = data[i * 4..i * 4 + 4].try_into().unwrap_or([0; 4]);
                i32::from_le_bytes(bytes) as i64
            }
            8 => {
                let bytes: [u8; 8] = data[i * 8..i * 8 + 8].try_into().unwrap_or([0; 8]);
                i64::from_le_bytes(bytes)
            }
            _ => 0,
        };

        if i > 0 && val >= prev && (val - prev) < 1000 {
            monotonic_steps += 1;
        }
        prev = val;
    }

    monotonic_steps > (count * 3 / 4)
}
