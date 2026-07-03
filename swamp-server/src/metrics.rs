// swamp-server/src/metrics.rs
// Metricas de observabilidade Prometheus para o LLamañón.rs

use prometheus::{Registry, Gauge, Counter, Histogram, HistogramOpts};

#[derive(Clone)]
pub struct ConcurrencyMetrics {
    pub registry: Registry,
    pub active_requests: Gauge,
    pub queued_requests: Gauge,
    pub rejected_requests: Counter,
    pub inference_latency: Histogram,
    pub tokens_generated: Counter,
}

impl ConcurrencyMetrics {
    pub fn new() -> Self {
        let registry = Registry::new();

        let active_requests = Gauge::new(
            "swamp_active_requests",
            "Numero de requisicoes sendo processadas ativamente"
        ).unwrap();

        let queued_requests = Gauge::new(
            "swamp_queued_requests",
            "Numero de requisicoes aguardando na fila de prioridades"
        ).unwrap();

        let rejected_requests = Counter::new(
            "swamp_rejected_requests_total",
            "Total de requisicoes rejeitadas devido a sobrecarga (backpressure)"
        ).unwrap();

        let inference_latency = Histogram::with_opts(
            HistogramOpts::new(
                "swamp_inference_latency_seconds",
                "Latencia total para geracao de respostas completas"
            ).buckets(vec![0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0])
        ).unwrap();

        let tokens_generated = Counter::new(
            "swamp_tokens_generated_total",
            "Total de tokens gerados pelo motor de inferencia"
        ).unwrap();

        registry.register(Box::new(active_requests.clone())).unwrap();
        registry.register(Box::new(queued_requests.clone())).unwrap();
        registry.register(Box::new(rejected_requests.clone())).unwrap();
        registry.register(Box::new(inference_latency.clone())).unwrap();
        registry.register(Box::new(tokens_generated.clone())).unwrap();

        Self {
            registry,
            active_requests,
            queued_requests,
            rejected_requests,
            inference_latency,
            tokens_generated,
        }
    }
}

impl Default for ConcurrencyMetrics {
    fn default() -> Self {
        Self::new()
    }
}
