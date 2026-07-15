// swamp-engine/src/aimd.rs
// AIMD Resource Ramp — Additive Increase Multiplicative Decrease.
//
// A cada 0.5s sem sinal de estresse (temp, RAPL throttle, latência alta):
//   DOBRA o orçamento de recursos (threads/batch/concorrência).
// No PRIMEIRO sinal ruim:
//   CORTA PELA METADE instantaneamente.
// Sem MSR, sem tocar em frequência — só controle de concorrência.

use std::time::{Duration, Instant};

pub struct AimdRamp {
    budget: f64,
    min_budget: f64,
    max_budget: f64,
    interval: Duration,
    last_assessment: Instant,
    stressed: bool,
    cycles: u64,
    halvings: u64,
    consecutive_good_steps: u64,
}

impl AimdRamp {
    pub fn new() -> Self {
        Self {
            budget: 1.0,
            min_budget: 0.125,
            max_budget: 16.0,
            interval: Duration::from_millis(500),
            last_assessment: Instant::now(),
            stressed: false,
            cycles: 0,
            halvings: 0,
            consecutive_good_steps: 0,
        }
    }

    /// Core assess logic (no interval guard). Used by both `assess` and `force_assess`.
    fn assess_inner(&mut self, stressed: bool, hit_rate: f64) -> f64 {
        self.cycles += 1;

        if stressed {
            self.budget = (self.budget * 0.5).max(self.min_budget);
            self.halvings += 1;
            self.stressed = true;
            self.consecutive_good_steps = 0;
        } else {
            self.budget = (self.budget * 2.0).min(self.max_budget);
            self.stressed = false;
            if hit_rate > 0.9 {
                self.consecutive_good_steps += 1;
            } else {
                self.consecutive_good_steps = 0;
            }
        }
        self.budget
    }

    /// Assess system state. Call once per decode step (~every 500ms).
    /// `stressed` = any of: temp > warning, RAPL power > 90% PL1, decode latency spiked.
    /// `hit_rate` = prefetch hit ratio (0.0–1.0), used for consecutive-good-step boost.
    /// Returns the current resource budget multiplier.
    pub fn assess(&mut self, stressed: bool, hit_rate: f64) -> f64 {
        let elapsed = self.last_assessment.elapsed();
        if elapsed < self.interval {
            return self.budget;
        }
        self.last_assessment = Instant::now();
        self.assess_inner(stressed, hit_rate)
    }

    /// Assess without the minimum-interval guard — used by tests and immediate re-evaluations.
    pub fn force_assess(&mut self, stressed: bool, hit_rate: f64) -> f64 {
        self.last_assessment = Instant::now();
        self.assess_inner(stressed, hit_rate)
    }

    /// When hit_rate > 0.9 for 5+ consecutive assessments, caller should do an extra AI
    /// increment (e.g. K+1 on the prefetch window) beyond the standard budget doubling.
    pub fn should_boost_k(&self) -> bool {
        self.consecutive_good_steps >= 5
    }

    pub fn budget(&self) -> f64 {
        self.budget
    }

    /// Number of active instances = budget rounded up (min 1).
    pub fn concurrency(&self) -> usize {
        (self.budget.ceil() as usize).max(1)
    }

    /// Recommended thread count (base_n_threads * budget, clamped).
    pub fn scaled_threads(&self, base_threads: usize) -> usize {
        let scaled = (base_threads as f64 * self.budget).round() as usize;
        scaled.clamp(1, 12)
    }

    pub fn cycles(&self) -> u64 { self.cycles }
    pub fn halvings(&self) -> u64 { self.halvings }
    pub fn just_stressed(&self) -> bool { self.stressed }
    pub fn consecutive_good(&self) -> u64 { self.consecutive_good_steps }

    /// Scale streaming prefetch window K based on AIMD budget.
    /// High budget → more aggressive prefetch, low budget → conservative.
    pub fn scaled_k(&self, base_k: usize) -> usize {
        let k = (base_k as f64 * self.budget.sqrt()).round() as usize;
        k.max(1).min(base_k.saturating_mul(2))
    }

    /// Calculate the physical max prefetch window K given bandwidth and layer size.
    /// bandwidth_gbps: measured transfer bandwidth in GB/s
    /// layer_bytes: total bytes for one layer's weights
    /// Returns K_max (minimum 1, maximum 32)
    pub fn k_max_from_bandwidth(&self, bandwidth_gbps: f64, layer_bytes: usize) -> usize {
        let step_time_s = 0.050;
        let bytes_per_sec = bandwidth_gbps * 1e9;
        let layers_per_step = (bytes_per_sec * step_time_s) / layer_bytes.max(1) as f64;
        (layers_per_step as usize).max(1).min(32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k_max_from_bandwidth() {
        let ramp = AimdRamp::new();
        // 50 GB/s, 100 MB per layer → 25 layers/step
        let k = ramp.k_max_from_bandwidth(50.0, 100_000_000);
        assert_eq!(k, 25);
        // 10 GB/s, 100 MB per layer → 5 layers/step
        let k = ramp.k_max_from_bandwidth(10.0, 100_000_000);
        assert_eq!(k, 5);
        // Very slow: 1 GB/s, 200 MB per layer → 0.25 → clamped to 1
        let k = ramp.k_max_from_bandwidth(1.0, 200_000_000);
        assert_eq!(k, 1);
    }

    #[test]
    fn test_scaled_k_never_zero() {
        let mut ramp = AimdRamp::new();
        // Stress repeatedly to drive budget to min
        for _ in 0..20 {
            ramp.force_assess(true, 0.0);
        }
        // Even at min budget, scaled_k should be at least 1
        for base_k in [1, 2, 3, 6, 10] {
            let k = ramp.scaled_k(base_k);
            assert!(k >= 1, "scaled_k({}) = {} should be >= 1", base_k, k);
        }
    }

    #[test]
    fn test_consecutive_good_increases_k() {
        let mut ramp = AimdRamp::new();
        // Hit rate above 0.9 for 4 assess calls → should_boost_k still false
        for _ in 0..4 {
            ramp.force_assess(false, 0.95);
        }
        assert!(!ramp.should_boost_k());
        assert_eq!(ramp.consecutive_good(), 4);
        // 5th assessment → should_boost_k true
        ramp.force_assess(false, 0.95);
        assert!(ramp.should_boost_k());
        assert_eq!(ramp.consecutive_good(), 5);
        // One bad hit rate resets counter
        ramp.force_assess(false, 0.5);
        assert!(!ramp.should_boost_k());
        assert_eq!(ramp.consecutive_good(), 0);
    }
}
