use xxhash_rust::xxh3::xxh3_128;

const fn generate_gear_table() -> [u64; 256] {
    let mut table = [0u64; 256];
    let mut val = 0x9E3779B97F4A7C15u64;
    let mut i = 0;
    while i < 256 {
        val = val
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        table[i] = val;
        i += 1;
    }
    table
}

pub static GEAR_TABLE: [u64; 256] = generate_gear_table();

pub struct GearHash {
    table: &'static [u64; 256],
}

impl Default for GearHash {
    fn default() -> Self {
        Self::new()
    }
}

impl GearHash {
    pub const fn new() -> Self {
        Self { table: &GEAR_TABLE }
    }

    pub fn table(&self) -> &'static [u64; 256] {
        self.table
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CdcSegment {
    pub offset: usize,
    pub len: usize,
    pub content_hash: u128,
}

#[derive(Clone, Copy, Debug)]
pub struct CdcConfig {
    pub min_size: usize,
    pub max_size: usize,
    pub target_avg: usize,
    pub boundary_mask: u64,
}

impl Default for CdcConfig {
    fn default() -> Self {
        Self {
            min_size: 32 * 1024,        // 32 KB
            max_size: 512 * 1024,       // 512 KB
            target_avg: 128 * 1024,     // 128 KB
            boundary_mask: 0x0001_FFFF, // Average ~128 KB
        }
    }
}

pub fn cdc_chunk(data: &[u8], config: &CdcConfig) -> Vec<CdcSegment> {
    if data.is_empty() {
        return Vec::new();
    }

    if data.len() <= config.min_size {
        return vec![CdcSegment {
            offset: 0,
            len: data.len(),
            content_hash: xxh3_128(data),
        }];
    }

    let gear = GearHash::new();
    let mut segments = Vec::new();
    let mut chunk_start = 0;
    let n = data.len();

    while chunk_start < n {
        let remaining = n - chunk_start;
        if remaining <= config.min_size {
            segments.push(CdcSegment {
                offset: chunk_start,
                len: remaining,
                content_hash: xxh3_128(&data[chunk_start..n]),
            });
            break;
        }

        let max_chunk_len = remaining.min(config.max_size);
        let scan_start = chunk_start + config.min_size;
        let scan_end = chunk_start + max_chunk_len;

        let mut hash = 0u64;
        let mut cut_point = scan_end;

        for i in scan_start..scan_end {
            let b = data[i];
            hash = (hash << 1).wrapping_add(gear.table[b as usize]);
            if (hash & config.boundary_mask) == 0 {
                cut_point = i + 1;
                break;
            }
        }

        let chunk_len = cut_point - chunk_start;
        segments.push(CdcSegment {
            offset: chunk_start,
            len: chunk_len,
            content_hash: xxh3_128(&data[chunk_start..cut_point]),
        });

        chunk_start = cut_point;
    }

    segments
}
