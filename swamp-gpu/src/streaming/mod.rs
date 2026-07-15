pub mod transfer;
pub mod triple_buffer;
pub mod shard;

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub struct StreamTelemetry {
    pub bytes_transferred: AtomicU64,
    pub transfers_completed: AtomicU64,
    pub transfers_missed: AtomicU64,
    pub prefetch_hits: AtomicU64,
    pub prefetch_misses: AtomicU64,
    pub k_current: AtomicU64,
    pub bandwidth_gbps: AtomicU64,
    start: Instant,
}

impl StreamTelemetry {
    pub fn new() -> Self {
        Self {
            bytes_transferred: AtomicU64::new(0),
            transfers_completed: AtomicU64::new(0),
            transfers_missed: AtomicU64::new(0),
            prefetch_hits: AtomicU64::new(0),
            prefetch_misses: AtomicU64::new(0),
            k_current: AtomicU64::new(0),
            bandwidth_gbps: AtomicU64::new(0),
            start: Instant::now(),
        }
    }

    pub fn record_transfer(&self, bytes: u64) {
        self.bytes_transferred.fetch_add(bytes, Ordering::Relaxed);
        self.transfers_completed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_miss(&self) {
        self.transfers_missed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_prefetch_hit(&self) {
        self.prefetch_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_prefetch_miss(&self) {
        self.prefetch_misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn hit_rate(&self) -> f64 {
        let hits = self.prefetch_hits.load(Ordering::Relaxed);
        let misses = self.prefetch_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 { 1.0 } else { hits as f64 / total as f64 }
    }

    pub fn snapshot_csv(&self) -> String {
        let elapsed = self.start.elapsed().as_secs_f64();
        let bytes = self.bytes_transferred.load(Ordering::Relaxed);
        let bw = if elapsed > 0.0 { bytes as f64 / elapsed / 1e9 } else { 0.0 };
        format!(
            "{:.2},{},{:.3},{:.3},{}",
            elapsed,
            bytes,
            bw,
            self.hit_rate(),
            self.k_current.load(Ordering::Relaxed),
        )
    }

    pub fn snapshot_header() -> &'static str {
        "elapsed_s,bytes,bw_gbps,hit_rate,k"
    }
}
