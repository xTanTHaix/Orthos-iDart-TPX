/// Succinct Range Filter (SuRF).
/// Provides fast range membership tests with zero false negatives (hard guarantee).

#[derive(Clone, Debug, Default)]
pub struct Surf {
    keys: Vec<Vec<u8>>,
}

impl Surf {
    pub fn build(keys: &[&[u8]]) -> Self {
        let mut sorted_keys: Vec<Vec<u8>> = keys.iter().map(|k| k.to_vec()).collect();
        sorted_keys.sort();
        sorted_keys.dedup();
        Self { keys: sorted_keys }
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Range lookup query: returns false if key is definitely not in the index,
    /// or true if it might be (with low false positive rate).
    pub fn range_lookup(&self, left: &[u8], right: &[u8]) -> bool {
        if self.keys.is_empty() {
            return false;
        }

        match self
            .keys
            .binary_search_by(|probe| probe.as_slice().cmp(left))
        {
            Ok(_) => true,
            Err(idx) => {
                if idx < self.keys.len() {
                    self.keys[idx].as_slice() <= right
                } else {
                    false
                }
            }
        }
    }

    pub fn approx_mem_usage(&self) -> usize {
        self.keys.iter().map(|k| k.len()).sum::<usize>() + self.keys.len() * 24
    }
}
