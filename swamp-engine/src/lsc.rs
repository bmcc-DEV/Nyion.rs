// swamp-engine/src/lsc.rs
// LscPrefetcher: calcula a curva de ganho de prefetch baseada em contexto

pub struct LscPrefetcher {
    pub mu: f32,
    pub beta: f32,
}

impl LscPrefetcher {
    pub fn new(mu: f32, beta: f32) -> Self {
        Self { mu, beta }
    }

    /// Calcula o ganho G(Ce) = 1 / [(1 - Ce) + mu * e^(beta * Ce)]
    /// onde Ce e a fracao de contexto consumida
    pub fn gain(&self, context_fraction: f32) -> f32 {
        let ce = context_fraction.clamp(0.0, 1.0);
        let denominator = (1.0 - ce) + self.mu * (self.beta * ce).exp();
        if denominator.abs() < 1e-6 {
            1e6 // evita divisao por zero
        } else {
            1.0 / denominator
        }
    }
}

impl Default for LscPrefetcher {
    fn default() -> Self {
        Self::new(0.1, 2.0)
    }
}
