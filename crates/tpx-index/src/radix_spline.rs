/// RadixSpline index.
/// Fast learned-index mapping 64-bit keys to bounded position ranges.

#[derive(Clone, Debug, Default)]
pub struct RadixSpline {
    keys: Vec<u64>,
    positions: Vec<u32>,
}

impl RadixSpline {
    pub fn build(sorted_keys: &[u64], positions: &[u32]) -> Self {
        assert_eq!(sorted_keys.len(), positions.len());
        Self {
            keys: sorted_keys.to_vec(),
            positions: positions.to_vec(),
        }
    }

    pub fn lookup(&self, key: u64) -> (u32, u32) {
        if self.keys.is_empty() {
            return (0, 0);
        }

        match self.keys.binary_search(&key) {
            Ok(idx) => {
                // Find lower and upper bound for duplicate keys
                let mut lo_idx = idx;
                while lo_idx > 0 && self.keys[lo_idx - 1] == key {
                    lo_idx -= 1;
                }
                let mut hi_idx = idx;
                while hi_idx + 1 < self.keys.len() && self.keys[hi_idx + 1] == key {
                    hi_idx += 1;
                }
                (self.positions[lo_idx], self.positions[hi_idx])
            }
            Err(idx) => {
                let lo = if idx > 0 { self.positions[idx - 1] } else { 0 };
                let hi = if idx < self.positions.len() {
                    self.positions[idx]
                } else {
                    *self.positions.last().unwrap_or(&0)
                };
                (lo, hi)
            }
        }
    }
}
