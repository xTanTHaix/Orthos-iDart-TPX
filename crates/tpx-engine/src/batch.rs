use std::collections::{HashMap, VecDeque};
use tpx_core::types::{OwnedTensor, TpxError};
use tpx_shard::engine::ShardedEngine;

pub struct PatternPredictor {
    ngram: HashMap<(u64, u64), Vec<u64>>,
}

impl Default for PatternPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternPredictor {
    pub fn new() -> Self {
        Self {
            ngram: HashMap::new(),
        }
    }

    pub fn record_sequence(&mut self, seq: &[u64]) {
        for window in seq.windows(3) {
            let key = (window[0], window[1]);
            self.ngram.entry(key).or_default().push(window[2]);
        }
    }

    pub fn predict_next(&self, k1: u64, k2: u64) -> Option<u64> {
        self.ngram
            .get(&(k1, k2))
            .and_then(|candidates| candidates.last().copied())
    }
}

pub struct PrefetchHistory {
    pub recent_sequences: VecDeque<Vec<u64>>,
    pub predictor: PatternPredictor,
}

impl Default for PrefetchHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl PrefetchHistory {
    pub fn new() -> Self {
        Self {
            recent_sequences: VecDeque::with_capacity(32),
            predictor: PatternPredictor::new(),
        }
    }

    pub fn record(&mut self, hashes: Vec<u64>) {
        self.predictor.record_sequence(&hashes);
        if self.recent_sequences.len() >= 32 {
            self.recent_sequences.pop_front();
        }
        self.recent_sequences.push_back(hashes);
    }
}

pub struct BatchReader<'a> {
    engine: &'a ShardedEngine,
    pub coalesce_threshold: usize,
    pub history: parking_lot::Mutex<PrefetchHistory>,
}

impl<'a> BatchReader<'a> {
    pub fn new(engine: &'a ShardedEngine) -> Self {
        Self {
            engine,
            coalesce_threshold: 128 * 1024,
            history: parking_lot::Mutex::new(PrefetchHistory::new()),
        }
    }

    pub fn get_batch(&self, keys: &[&str]) -> Result<Vec<(String, OwnedTensor)>, TpxError> {
        let results = self.engine.get_batch(keys)?;
        let key_hashes: Vec<u64> = keys
            .iter()
            .map(|k| xxhash_rust::xxh64::xxh64(k.as_bytes(), 0))
            .collect();
        self.history.lock().record(key_hashes);
        Ok(results)
    }
}
