# Sprint 5 — ResourceGovernor AIMD e Ajuste Dinâmico

**Status**: ✅ COMPLETO
- ✅ AIMD controller (`AdditiveIncreaseMultiplicativeDecrease`) — `swamp-engine/src/aimd.rs`
- ✅ StreamTelemetry (contadores atômicos) — `swamp-gpu/src/streaming/mod.rs`
- ✅ **AIMD → K adaptativo** — `aimd.rs::scaled_k()` + `executor.rs` feedback loop
- ✅ **Medição contínua no hot path** — `executor.rs` lê `drain_prefetch_stats()` a cada 20 steps
- 🔴 **Stress tests** — **POR FAZER**

## O Que Já Existe

### AIMD (`aimd.rs`)

```rust
pub struct AIMDRamp {
    budget: f64,        // 0.0 a 1.0
    inc: f64,           // additive increase step
    dec: f64,           // multiplicative decrease factor (ex: 0.5)
    min_budget: f64,
    max_budget: f64,
}
```

Usado atualmente para:
- `assess(stressed: bool)` → ajusta budget
- `scaled_threads(base: usize)` → ajusta thread count
- `budget()` → consulta
- `concurrency()` → budget mapeado para 1..6

### StreamTelemetry (`streaming/mod.rs`)

```rust
pub struct StreamTelemetry {
    bytes_transferred: AtomicU64,
    transfers_completed: AtomicU64,
    transfers_missed: AtomicU64,
    prefetch_hits: AtomicU64,
    prefetch_misses: AtomicU64,
    k_current: AtomicU64,
    bandwidth_gbps: AtomicU64,
}
```

Métricas derivadas:
- `hit_rate() = hits / (hits + misses)`
- `bandwidth = bytes / elapsed`
- `snapshot_csv()` → `elapsed_s,bytes,bw_gbps,hit_rate,k`

## Implementado

### 1. Medição Contínua no Hot Path — `executor.rs`

No decode loop, a cada 20 steps:
- `PerLayerGpuState::drain_prefetch_stats()` retorna (hits, misses)
- `hit_rate = hits / (hits + misses)`
- `aimd.assess(stressed)` onde stress inclui `hit_rate < 0.6`
- `prefetch_k = aimd.scaled_k(6)` — atualiza a janela de prefetch

```rust
let (pf_hits, pf_misses) = per_layer_gpu.as_ref()
    .map(|g| g.drain_prefetch_stats()).unwrap_or((0, 0));
let hit_rate = if pf_total > 0 { pf_hits as f64 / pf_total as f64 } else { 1.0 };
let stressed = temp_celsius > 80.0 || c_epsilon < 0.85 || hit_rate < 0.6;
aimd.assess(stressed);
prefetch_k = aimd.scaled_k(6);
```

### 2. AIMD → K Adaptativo — `aimd.rs:scaled_k()`

```rust
pub fn scaled_k(&self, base_k: usize) -> usize {
    let k = (base_k as f64 * self.budget.sqrt()).round() as usize;
    k.max(1).min(base_k.saturating_mul(2))
}
```

Usa a raiz quadrada do budget AIMD para escalar K suavemente:
- budget=0.125 → K ≈ base_k * 0.35 (conservador)
- budget=1.0 → K = base_k (neutro)
- budget=16.0 → K = base_k * 2 (agressivo, clamped)

### 3. Prefetch Adaptativo no Decode Loop — `executor.rs`

Após cada layer, o prefetch usa `prefetch_k` para carregar multiplas layers ahead:

```rust
for ahead in 1..=prefetch_k {
    let next = l + ahead;
    if next < num_layers {
        gpu.ensure_layer_loaded(next, &model);
    }
}
```

### 4. Stress Tests

| Cenário | Configuração | Comportamento Esperado |
|---------|-------------|----------------------|
| Banda infinita | NVMe RAID0, 10 GB/s | K alto, hit_rate ~100% |
| Banda limitada | cgroup throttle 100 MB/s | K reduz até hit_rate estabilizar |
| Preditor perfeito | DSPark acerta 100% | K máximo, sem fallbacks |
| Preditor aleatório | DSPark retorna tokens aleatórios | K mínimo, fallbacks frequentes |
| Burst de tokens | 5 tokens em 1μs | AIMD reduz K, fallback CPU nos primeiros |
| Degradação gradual | NVMe → SATA → SD | K reduz suavemente |

### 4. Métricas em Tempo Real

Log a cada 20 steps:
```
[Stream] elapsed=14.32s bw=3.21 GB/s hit_rate=0.94 K=5 transfers=847 misses=12 fallbacks=3
```

CSV final ao término:
```
elapsed_s,bytes,bw_gbps,hit_rate,k
14.32,48000000000,3.21,0.94,5
```

## Acceptance Checklist

- [x] `drain_prefetch_stats()` conectado no hot path do executor
- [x] AIMD ajusta K baseado em hit_rate via scaled_k()
- [x] Logs mostram K evoluindo (tracing::info!)
- [ ] Stress tests passam sem stalls longos (>1s)
- [ ] Relatório: degradation graceful com banda limitada

## Dependências

- [[sprint3-dspark-scheduling.md]] — Prefetch queue alimenta telemetry
- [[sprint4-fallback-cpu.md]] — Fallback CPU é consequência de miss

## Próximos Passos

1. ✅ Conectar StreamTelemetry ao hot path
2. ✅ AIMD → scaled_k() + feedback loop
3. 🔴 Criar stress test suite
4. 🔴 [[sprint6-qat-e2e.md]] — End-to-end com 7B
