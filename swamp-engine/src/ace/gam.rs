// =========================================================================
// GAM v2 — GeoVex Adaptive Morph
// Adaptive k-cutoff management for the Morph operation
// =========================================================================
//
// PRINCÍPIO: Nunca ultrapassar 33ms. Preferencialmente sub-10ms.

#[derive(Debug, Clone, Copy)]
pub enum DepthMode {
    Surface,
    Deep,
    Exhaustive,
}

#[derive(Debug, Clone, Copy)]
pub struct GamParams {
    pub k: usize,
    pub sinkhorn_iters: usize,
    pub early_term: bool,
}

/// Adaptive k selection per GAM v2 spec.
/// Returns (k, sinkhorn_iters, early_term).
pub fn adaptive_k(
    n_active: usize,
    urgency: f64,
    depth: DepthMode,
    thermal_headroom: f64,
    n_recent: usize,
) -> GamParams {
    // Passo 1: k maximo viravel para N
    let (k_max, k_default) = if n_active <= 1000 {
        (128, 64)
    } else if n_active <= 2500 {
        (64, 32)
    } else if n_active <= 5000 {
        (32, 32)
    } else if n_active <= 10000 {
        (16, 16)
    } else {
        (8, 8)
    };

    // Passo 2: ajuste por modo
    let mut k: usize = k_default;
    let mut sinkhorn_iters: usize = 25;
    let mut early_term: bool = false;
    match depth {
        DepthMode::Surface => { k = k_default.max(8) / 2; sinkhorn_iters = 15; early_term = true; }
        DepthMode::Deep => { k = k_default; sinkhorn_iters = 25; early_term = false; }
        DepthMode::Exhaustive => { k = k_max; sinkhorn_iters = 40; early_term = false; }
    }

    // Passo 3: ajuste por urgencia
    if urgency > 0.8 {
        k = k.max(8) / 2;
        sinkhorn_iters = sinkhorn_iters.saturating_sub(10).max(10);
    } else if urgency < 0.2 {
        let k2 = (k * 2).min(k_max);
        if predict_latency(k2, n_active) < 20.0 {
            k = k2;
        }
    }

    // Passo 4: ajuste termico
    if thermal_headroom < 0.25 {
        k = k.max(8) / 2;
        sinkhorn_iters = sinkhorn_iters.saturating_sub(15).max(10);
    } else if thermal_headroom > 0.75 {
        let k16 = (k + 16).min(k_max);
        if predict_latency(k16, n_active) < 10.0 {
            k = k16;
        }
    }

    // Passo 5: working set optimization
    if n_recent > 0 && n_recent < n_active / 4 {
        let effective_n = n_recent;
        if effective_n <= 1000 {
            k = (k * 2).min(128);
        } else if effective_n <= 2500 {
            k = (k * 2).min(64);
        }
    }

    // Passo 6: garantia final de latencia
    let mut estimated = predict_latency(k, n_active);
    if estimated > 33.0 {
        while k > 8 && predict_latency(k, n_active) > 33.0 {
            k = k.saturating_sub(8).max(8);
        }
        if predict_latency(k, n_active) > 33.0 {
            early_term = true;
            sinkhorn_iters = 10;
        }
    }

    GamParams { k, sinkhorn_iters, early_term }
}

/// Empirical latency model: predicts ms for given (k, N).
/// Coarse model: ~10 cycles per (N * k) pair, 3.0 GHz, cache-aware.
/// Fine-tuned at boot via HybridLatencyModel, updated online via EWMA.
pub fn predict_latency(k: usize, n: usize) -> f64 {
    let n = n as f64;
    let k = k as f64;
    let base_cycles = n * k * 10.0;
    let sinkhorn_cycles = n * k * 20.0;
    let deform_cycles = n * k * 5.0;
    let total = base_cycles + sinkhorn_cycles + deform_cycles;

    let mem_bytes = n * 64.0;
    let factor = if mem_bytes <= 48_000.0 {
        0.5
    } else if mem_bytes <= 512_000.0 {
        0.7
    } else if mem_bytes <= 8_000_000.0 {
        1.0
    } else {
        2.5
    };

    total * factor / 3.0e9 * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_surface() {
        let p = adaptive_k(500, 0.5, DepthMode::Surface, 0.8, 150);
        assert!(p.k >= 8);
        assert!(p.sinkhorn_iters >= 10);
    }

    #[test]
    fn test_large_exhaustive() {
        let p = adaptive_k(15000, 0.1, DepthMode::Exhaustive, 0.9, 2000);
        assert!(p.k >= 8);
        assert!(p.k <= 128);
    }

    #[test]
    fn test_urgent() {
        let p = adaptive_k(3000, 0.95, DepthMode::Deep, 0.5, 300);
        assert!(p.k <= 32);
        assert!(p.sinkhorn_iters <= 15);
    }

    #[test]
    fn test_hot_chip() {
        let p = adaptive_k(4000, 0.9, DepthMode::Surface, 0.1, 400);
        assert!(p.k <= 16);
        assert!(p.sinkhorn_iters >= 10);
    }

    #[test]
    fn test_predict_latency_bounds() {
        let ms = predict_latency(32, 5000);
        assert!(ms > 0.0);
        assert!(ms < 100.0);
    }
}
