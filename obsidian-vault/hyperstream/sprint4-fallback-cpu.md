# Sprint 4 — Fallback CPU + Kernels

**Status**: ✅ COMPLETO
- ✅ CPU SIMD kernels (Q4_K GEMV, Q6_K GEMV, softmax) — `swamp-kernels/src/`
- ✅ GPU/CPU fallpath via `#[cfg(feature = "gpu")]` — `executor.rs`
- ✅ **Fallback por timeout de streaming** — `streamer.rs::wait_for_layer()` com timeout
- ✅ **LayerResult com FallbackDecision (GPU/CPU)** — `streamer.rs`

## O Que Já Existe

### CPU Kernels (`swamp-kernels/src/`)

| Kernel | Arquivo | Paths SIMD |
|--------|---------|------------|
| GEMV Q4_K (fused dequant + dot) | `fused_gemv_q4k.rs` | VNNI, AVX2, scalar |
| GEMV Q6_K | `fused_gemv_q6k.rs` | VNNI, AVX2, scalar |
| Softmax Q16.16 | `lib.rs` | AVX-512, AVX2 |
| Matmul W3 | `lib.rs` | VNNI, AVX2, scalar |
| Dot product primitives | `simd.rs` | Raw SIMD ops |

### GPU/CPU Fallback Atual (`executor.rs`)

Cada operação GPU (gemv_qkv_async, execute_attention_async, etc.) retorna `bool`. Se `false`, o executor cai para `forward_linear()` no CPU:

```rust
if !gemv_qkv_ok {
    forward_linear_multi(&model.gguf, &[q_t, k_t, v_t], &x_norm, ...)?;
}
```

Esse padrão já é exatamente o que precisamos para fallback de streaming — a diferença é que o trigger será **timeout de timeline** em vez de `feature gate`.

## Implementado

### 1. Fallback por Timeout — `streamer.rs::wait_for_layer()`

`HyperStreamEngine::wait_for_layer(layer_id, tensor_name, timeout_ns)` procura na fila de prefetch, espera timeline semaphore com timeout. Se ready → `FallbackDecision::Gpu(slot)`. Se timeout → `FallbackDecision::Cpu` com `tensor_data` opcional (dados do mmap para fallback CPU).

### 2. FallbackDecision — `streamer.rs`

```rust
pub enum FallbackDecision {
    Gpu(usize),   // slot index pronto
    Cpu,          // fallback para CPU
}

pub struct LayerResult {
    pub decision: FallbackDecision,
    pub tensor_data: Option<Vec<u8>>,
}
```

## Acceptance Checklist

- [ ] `wait_for_layer()` retorna GPU ou CPU baseado em timeout
- [ ] CPU fallback produz saída numericamente plausível (comparar com GPU)
- [ ] Log de fallback events: layer_id, motivo (timeout / miss), latência
- [ ] Política configurável (FORCE_GPU, FORCE_CPU, ADAPTIVE)
- [ ] Sem crashes em cenários de miss total

## Dependências

- [[sprint3-dspark-scheduling.md]] — Prefetch queue alimenta os buffers
- [[sprint5-aimd-governor.md]] — AIMD ajusta timeout/policy

## Próximos Passos

1. Implementar `wait_for_layer()` com timeout
2. NyionVM dispatcher no decode loop
3. Testar com cenários de miss forçada (K=0)
4. [[sprint5-aimd-governor.md]] — Integrated AIMD
