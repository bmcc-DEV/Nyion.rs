# Integration Plan — Bridging Existing Code to HyperStream

## Mapping: TikTok Task → Rust Module

| Sprint | Task | Código Existente | O que Precisamos Criar/Modificar |
|--------|------|-------------------|-----------------------------------|
| S1 T1 | Vulkan bootstrap | `vulkan.rs:1` (VkBackend) | ✅ Completo |
| S1 T2 | Timeline semaphore | `transfer.rs:58` | ✅ Completo |
| S1 T3 | Triple-buffer | `triple_buffer.rs:34` | ✅ Completo |
| S1 T4 | Dummy consumer | `benchmark_streaming.rs:1` | ✅ Completo (benchmark) |
| S2 T1 | Shard converter | **N/A** | **NOVO**: `swamp-tools/src/bin/convert_shard.rs` |
| S2 T2 | mmap reader | `shard.rs:42` (ShardReader) | ✅ Completo |
| S2 T3 | Integrar reader + transfer | `shard.rs` + `transfer.rs` | **NOVO**: `swamp-engine/src/streamer.rs` (HyperStreamEngine) |
| S3 T1 | DSPark predictor | `dspark.rs` | ✅ Completo |
| S3 T2 | Layer schedule resolver | **N/A** | **NOVO**: resolver em `swamp-engine/src/streamer.rs` |
| S3 T3 | Integrar DSPark + streamer | `dspark.rs` + `streamer.rs` | **NOVO**: HyperStreamEngine usa DSPark para K tokens ahead |
| S3 T4 | Telemetria | `streaming/mod.rs:8` (StreamTelemetry) | ✅ Completo (falta conectar ao hot path) |
| S4 T1 | Fallback CPU (streaming) | `executor.rs` CPU gates | **MODIFICAR**: executor detecta timeout e cai pra CPU |
| S4 T2 | CPU kernels | `swamp-kernels/src/` | ✅ Completo |
| S4 T3 | NyionVM dispatch | `executor.rs` + `scheduler.rs` | **MODIFICAR**: dispatcher GPU/CPU por layer com threshold |
| S5 T1 | Medição contínua | StreamTelemetry | **MODIFICAR**: conectar ao hot path do executor |
| S5 T2 | AIMD controller | `aimd.rs` | **MODIFICAR**: AIMD ajusta K (prefetch window) |
| S5 T3 | Stress tests | `benchmark_streaming.rs` | **NOVO** suite de testes |
| S6 T1 | QAT per-block | `qat.rs` | **MODIFICAR**: pipeline de calibration durante conversão |
| S6 T2 | E2E 7B/13B | `executor.rs` + `benchmark_prefill.rs` | **NOVO**: teste integrado |
| S7 | DirectStorage | **N/A** | Futuro (R1) |
| S10 | CUDA opcional | `fused_attention.cu` | Existe mas stubs em Rust retornam erro |

## Critical Integration Path (✅ COMPLETA)

```
                           ┌──────────────────┐
                           │  GGUF ───► model  │
                           │  rings (mmap)     │── fallback
                           └──────────────────┘
                                        │
  ┌──────────────────┐      ┌─────────────────────────┐
  │  Shard (shard.bin│─────►│ HyperStreamEngine       │
  │  + index.json)   │      │ ─ read_tensor(layer)    │
  └──────────────────┘      │ ─ prefetch_layer()      │
                            │ ─ wait_for_layer()      │
                            └───────────┬─────────────┘
                                        │ data via shard mmap
                                        ▼
                            ┌─────────────────────────┐
                            │ PerLayerGpuState         │
                            │ ─ ensure_layer_loaded()  │
                            │   ├─ streamer path (shard)│
                            │   └─ fallback (model ring)│
                            └───────────┬─────────────┘
                                        │ ctx.upload_layer_weights()
                                        ▼
                            ┌─────────────────────────┐
                            │ GpuComputeContext        │
                            │ ─ weight buffers on GPU  │
                            │ ─ ComputeGraph (kernels) │
                            └─────────────────────────┘
                                        ▲
                            ┌───────────┴─────────────┐
                            │ DSPark (K tokens ahead)  │
                            │ predict → prefetch       │
                            └─────────────────────────┘
```

### Status de implementação

1. ✅ **`convert_shard.rs`** — `swamp-tools/src/bin/convert_shard.rs` — lê GGUF, escreve shard.bin + index.json
2. ✅ **`streamer.rs` (HyperStreamEngine)** — `swamp-engine/src/streamer.rs` — ShardReader + TransferEngine + LayerSchedule + prefetch_with_dspark + wait_for_layer + read_tensor
3. ✅ **`ensure_layer_loaded()` com streamer** — `scheduler.rs` — carrega pesos via shard (HyperStreamEngine) ou model rings (fallback), com `layer_loaded[]` tracker + contadores hit/miss
4. ✅ **Decode loop modificado** — `executor.rs` — `upload_all_weights()` removido; `ensure_layer_loaded()` chamado antes de cada GEMV; prefetch adaptativo via `prefetch_k`
5. ✅ **AIMD → K** — `aimd.rs::scaled_k()` + `executor.rs` feedback loop: `drain_prefetch_stats()` → hit_rate → AIMD stress → `scaled_k()` → prefetch window adaptativo
6. ✅ **Init streamer via env var** — `executor.rs` lê `SWAMP_SHARD_INDEX`/`SWAMP_SHARD_DATA` e chama `try_init_streamer()`

## Arquivos Criados

| Arquivo | Linhas | Descrição |
|---------|--------|-----------|
| `swamp-tools/src/bin/convert_shard.rs` | ~150 | Conversor GGUF → shard.bin + index.json |
| `swamp-engine/src/streamer.rs` | ~220 | HyperStreamEngine (prefetch + schedule + telemetry) |

## Arquivos Modificados

| Arquivo | Mudança |
|---------|---------|
| `swamp-engine/src/scheduler.rs` | `layer_loaded[]` tracker + `ensure_layer_loaded()` com streamer path + `try_init_streamer()` + `load_layer_from_streamer()` |
| `swamp-engine/src/executor.rs` | On-demand loading + prefetch adaptativo + init streamer via env var + hit_rate→AIMD feedback loop |
| `swamp-engine/src/streamer.rs` | `read_tensor()` público para acesso raw via mmap |
| `swamp-engine/src/aimd.rs` | `scaled_k()` para prefetch window adaptativo |
| `swamp-tools/Cargo.toml` | Adicionado bin `convert-shard` + serde deps |
| `swamp-engine/src/lib.rs` | Adicionado `pub mod streamer` |
| `swamp-tools/src/bin/benchmark_streaming.rs` | Reescrito para usar HyperStreamEngine |

## Observações

- O **fallback CPU** já existe via `#[cfg(not(feature = "gpu"))]` e `forward_linear()` nas funções do executor. O que muda é o *gatilho*: hoje cai se GPU desabilitado; amanhã cai também se timeline expirar.
- O `StreamTelemetry` já tem todos os contadores atômicos necessários. Só precisa ser exposto ao executor.
- O `AIMD` em `aimd.rs` já implementa additive-increase/multiplicative-decrease. Precisa de um novo parâmetro `K` e da leitura de `hit_rate` como feedback.
