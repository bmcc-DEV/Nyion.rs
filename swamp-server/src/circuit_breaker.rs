// swamp-server/src/circuit_breaker.rs
// CircuitBreaker: resiliencia sob carga e controle de falhas

use std::sync::atomic::{AtomicU32, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone)]
pub struct CircuitBreaker {
    failures: Arc<AtomicU32>,
    open: Arc<AtomicBool>,
    threshold: u32,
    reset_timeout: Duration,
}

impl CircuitBreaker {
    pub fn new(threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            failures: Arc::new(AtomicU32::new(0)),
            open: Arc::new(AtomicBool::new(false)),
            threshold,
            reset_timeout,
        }
    }

    pub fn is_open(&self) -> bool {
        self.open.load(Ordering::Acquire)
    }

    pub fn record_success(&self) {
        self.failures.store(0, Ordering::Relaxed);
    }

    pub fn record_failure(&self) {
        let failures = self.failures.fetch_add(1, Ordering::Relaxed) + 1;
        if failures >= self.threshold {
            self.open.store(true, Ordering::Release);
            tracing::warn!("CircuitBreaker aberto! Throttling requisicoes...");

            // Inicia timer para reset automatico (cooldown)
            let open = self.open.clone();
            let failures_clone = self.failures.clone();
            let timeout = self.reset_timeout;
            tokio::spawn(async move {
                tokio::time::sleep(timeout).await;
                open.store(false, Ordering::Release);
                failures_clone.store(0, Ordering::Relaxed);
                tracing::info!("CircuitBreaker fechado. Retomando processamento normal.");
            });
        }
    }
}
