use std::collections::VecDeque;
use std::time::{Duration, Instant};

// =========================================================================
// WorkingSet — concepts accessed in the last N seconds
// =========================================================================

pub struct WorkingSet {
    recent: VecDeque<(u32, Instant)>,
    ttl: Duration,
    max_size: usize,
}

impl WorkingSet {
    pub fn new() -> Self {
        Self {
            recent: VecDeque::new(),
            ttl: Duration::from_secs(30),
            max_size: 1000,
        }
    }

    pub fn touch(&mut self, id: u32) {
        self.evict_stale();
        if self.recent.len() >= self.max_size {
            self.recent.pop_front();
        }
        self.recent.push_back((id, Instant::now()));
    }

    pub fn contains(&self, id: u32) -> bool {
        self.recent.iter().any(|(i, _)| *i == id)
    }

    pub fn len(&self) -> usize {
        self.evict_stale_len();
        self.recent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return all active concept IDs in the working set
    pub fn ids(&self) -> Vec<u32> {
        let now = Instant::now();
        self.recent.iter()
            .filter(|(_, t)| now.duration_since(*t) < self.ttl)
            .map(|(id, _)| *id)
            .collect()
    }

    fn evict_stale(&mut self) {
        let now = Instant::now();
        while let Some((_, t)) = self.recent.front() {
            if now.duration_since(*t) < self.ttl { break; }
            self.recent.pop_front();
        }
    }

    fn evict_stale_len(&self) -> usize {
        let now = Instant::now();
        self.recent.iter()
            .filter(|(_, t)| now.duration_since(*t) < self.ttl)
            .count()
    }
}

// =========================================================================
// SimilarityCache — LRU cache with TTL
// =========================================================================

pub struct SimilarityCache {
    entries: Vec<((u32, u32), (f64, Instant))>,
    capacity: usize,
    ttl: Duration,
    hits: u64,
    misses: u64,
}

impl SimilarityCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
            ttl: Duration::from_millis(500),
            hits: 0,
            misses: 0,
        }
    }

    pub fn get(&mut self, a: u32, b: u32) -> Option<f64> {
        let key = if a < b { (a, b) } else { (b, a) };
        let now = Instant::now();
        if let Some(pos) = self.entries.iter().position(|(k, (_, t))| *k == key && now.duration_since(*t) < self.ttl) {
            let val = self.entries[pos].1 .0;
            self.entries[pos].1 .1 = now;
            self.hits += 1;
            Some(val)
        } else {
            self.misses += 1;
            None
        }
    }

    pub fn insert(&mut self, a: u32, b: u32, score: f64) {
        let key = if a < b { (a, b) } else { (b, a) };
        if let Some(pos) = self.entries.iter().position(|(k, _)| *k == key) {
            self.entries[pos] = (key, (score, Instant::now()));
            return;
        }
        if self.entries.len() >= self.capacity {
            self.entries.remove(0);
        }
        self.entries.push((key, (score, Instant::now())));
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 1.0 } else { self.hits as f64 / total as f64 }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.hits = 0;
        self.misses = 0;
    }
}

// =========================================================================
// ForgettingCurve — tracks access patterns for compression decisions
// =========================================================================

pub struct ForgettingCurve {
    entries: Vec<(u32, Instant, u64)>,
}

impl ForgettingCurve {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    pub fn touch(&mut self, id: u32) {
        if let Some((_, last, count)) = self.entries.iter_mut().find(|(i, _, _)| *i == id) {
            *last = Instant::now();
            *count += 1;
        } else {
            self.entries.push((id, Instant::now(), 1));
        }
    }

    /// Importance score: access_count / (1 + age_secs)
    pub fn score(&self, id: u32) -> f64 {
        let now = Instant::now();
        self.entries.iter()
            .find(|(i, _, _)| *i == id)
            .map(|(_, last, count)| {
                let age = now.duration_since(*last).as_secs_f64();
                *count as f64 / (1.0 + age)
            })
            .unwrap_or(0.0)
    }

    /// Return IDs with score below threshold
    pub fn stale(&self, threshold: f64) -> Vec<u32> {
        self.entries.iter()
            .filter(|(_, _, _)| true)
            .map(|(id, _, _)| *id)
            .filter(|id| self.score(*id) < threshold)
            .collect()
    }

    pub fn remove(&mut self, id: u32) {
        self.entries.retain(|(i, _, _)| *i != id);
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn test_working_set() {
        let mut ws = WorkingSet::new();
        ws.touch(1);
        ws.touch(2);
        assert!(ws.contains(1));
        assert!(ws.contains(2));
        assert_eq!(ws.len(), 2);
    }

    #[test]
    fn test_similarity_cache() {
        let mut cache = SimilarityCache::new(100);
        assert!(cache.get(1, 2).is_none());
        cache.insert(1, 2, 0.95);
        assert!((cache.get(1, 2).unwrap() - 0.95).abs() < 1e-6);
        assert!(cache.hit_rate() > 0.0);
    }

    #[test]
    fn test_forgetting_curve() {
        let mut fc = ForgettingCurve::new();
        fc.touch(1);
        fc.touch(2);
        assert!(fc.score(1) > 0.0);
        assert!(fc.score(99) == 0.0);
        sleep(Duration::from_millis(10));
        let stale = fc.stale(100.0);
        assert!(!stale.is_empty());
    }
}
