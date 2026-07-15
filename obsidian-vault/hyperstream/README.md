# Nyion HyperStream — Map of Content

> Speculative Weight Streaming — inference com pesos sob demanda via Vulkan timeline semaphores + triple buffering, fallback CPU e predição DSPark.

## Status (Sprint 0)

| Sprint | Status | Notas |
|--------|--------|-------|
| 0 — Preparação | ✅ `sprint0-research.md` | Decisões, hardware, dependências |
| 1 — Vulkan Transfer | ✅ `sprint1-vulkan-transfer.md` | Triple buffer + timeline + TransferEngine **completo** |
| 2 — Shard IO | ✅ `sprint2-shard-io.md` | **Converter tool + ShardReader + HyperStreamEngine completos** |
| 3 — DSPark + Scheduling | ✅ `sprint3-dspark-scheduling.md` | **LayerSchedule + resolve_k + prefetch_with_dspark implementados** |
| 4 — Fallback CPU | ✅ `sprint4-fallback-cpu.md` | **wait_for_layer com timeout + LayerResult implementados** |
| 5 — AIMD Governor | ✅ `sprint5-aimd-governor.md` | **scaled_k() + feedback loop hit_rate→AIMD→K** |
| 6 — QAT + E2E | 🔴 `sprint6-qat-e2e.md` | **Por fazer** — pipeline + teste 7B |
| R1 — Robustez | 🔴 `tasks-backlog.md` | DirectStorage, performance tune, 1T sharding |

## Links Rápidos

- [[architecture.md]] — Diagrama completo do sistema
- [[integration-plan.md]] — O que já existe e o que precisa ser ligado
- [[tasks-backlog.md]] — Backlog formato GitHub Issues
- [[runbook.md]] — Build, teste, troubleshooting
- [[shard-format-spec.md]] — Especificação do formato de shards

## Repositório

- `swamp-gpu/src/streaming/` — TransferEngine, TripleBuffer, ShardReader, StreamTelemetry
- `swamp-gpu/src/vulkan.rs` — VkBackend (instance/device/queue/buffer allocation)
- `swamp-gpu/src/compute_graph.rs` — DAG de compute nodes + timeline semaphore
- `swamp-engine/src/executor.rs` — Laço de inferência principal
- `swamp-engine/src/scheduler.rs` — PerLayerGpuState (ponte CPU ↔ GPU)
- `swamp-engine/src/streamer.rs` — HyperStreamEngine (orquestrador de streaming)
- `swamp-engine/src/dspark.rs` — DSPark speculative decoding engine
- `swamp-engine/src/aimd.rs` — AIMD resource controller + scaled_k()
- `swamp-tools/src/bin/benchmark_streaming.rs` — Benchmark de streaming integrado
- `swamp-tools/src/bin/convert_shard.rs` — Conversor GGUF → shard.bin
