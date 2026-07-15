# Nyion HyperStream — Architecture

## Data Flow Diagram

```
┌──────────────────────────────────────────────────┐
│  Duas fontes de dados (selecionadas via env var) │
│                                                   │
│  ┌──────────────────┐  ┌──────────────────────┐  │
│  │ Shard (shard.bin │  │ GGUF (model rings)   │  │
│  │  + index.json)   │  │  mmap via swamp-gguf │  │
│  │ SWAMP_SHARD_*    │  │  (fallback padrao)   │  │
│  └────────┬─────────┘  └──────────┬───────────┘  │
│           │ &[u8] slices          │ &[u8] slices  │
│           ▼                       ▼               │
│  ┌──────────────────────────────────────────┐     │
│  │  PerLayerGpuState::ensure_layer_loaded() │     │
│  │  ├─ streamer path: read_tensor() → shard │     │
│  │  └─ fallback path: model.layer_rings[l]  │     │
│  └────────────────┬─────────────────────────┘     │
│                   │ ctx.upload_layer_weights()     │
│                   ▼                                │
│  ┌──────────────────────────────────────┐         │
│  │  GpuComputeContext                   │         │
│  │  - LayerWeights (device-local)       │         │
│  │  - Scratch buffers + GpuKVCache     │         │
│  │  `swamp-gpu/src/lib.rs`             │         │
│  └────────┬─────────────────────────────┘         │
│           │ ComputeGraph (DAG de kernels)          │
│           ▼                                        │
│  ┌──────────────────────────────────────┐         │
│  │  Vulkan Compute Queue                │         │
│  │  - GEMVQ4K / Attention / RoPE ...    │         │
│  │  `swamp-gpu/shaders/*.comp`          │         │
│  └────────┬─────────────────────────────┘         │
│           │ Se GEMV retorna false                  │
│           ▼                                        │
│  ┌──────────────────────┐                         │
│  │  CPU Fallback Path   │  forward_linear()       │
│  │  (SIMD kernels)      │  via swamp-kernels      │
│  └──────────────────────┘                         │
└──────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────┐
│  Prefetch + AIMD feedback loop                    │
│                                                   │
│  ensure_layer_loaded(l+1..l+K) apos cada layer    │
│  drain_prefetch_stats() → hit_rate → AIMD → K     │
│  adaptive prefetch window                         │
└──────────────────────────────────────────────────┘
```

## Timeline Semaphore Signal Chain

```
Transfer Queue                Compute Queue
─────────────────             ─────────────────
submit(copy) ──► signal(t+1)  wait(t+1) ◄── ready
                                       │
         ╔══════════════════════════════╝
         ║
         ▼  acquire_compute_slot() → executa kernel
              signal(t+2) → CPU wait no sync()
```

Cada upload incrementa o timeline counter. O consumidor (compute) espera o valor correspondente antes de tocar no slot. Isso elimina **fences entre transfer e compute** — eles rodam em paralelo em queues diferentes.

## TripleBuffer State Machine

```
┌─────────┐    ┌─────────┐    ┌─────────┐
│ Slot 0  │    │ Slot 1  │    │ Slot 2  │
│ Free ───┤   │ Free ───┤   │ Free ───┤
│ Staging │   │DeviceRdy│   │Computing│
│DeviceRdy│   │Computing│   │ Free    │
│Computing│   │ Free    │   │ Staging │
│ Free    │   │ Staging │   │DeviceRdy│
└─────────┘    └─────────┘    └─────────┘
      ↑ write_index rotaciona circularmente
```

3 buffers permitem pipeline perfeito: transfer → device-ready → compute → free → transfer.

## Fallback Decision

```
                           ┌──────────────────┐
              ┌────────────┤ Timeline OK?     │
              │ NO         │ (wait_for_...    │
              │            │  timeout_ns)     │
              ▼            └────────┬─────────┘
   ┌──────────────────┐          YES │
   │ CPU Fallback      │            ▼
   │ forward_linear() │   ┌──────────────────┐
   │ (SIMD / VNNI)    │   │ GPU Compute      │
   │ dados do mmap    │   │ (Vulkan kernels) │
   └──────────────────┘   └──────────────────┘
```

Timeout configurável (default 100μs). Se o buffer não ficou pronto, cai pra CPU. Política futura: AIMD ajusta K (quantos tokens de prefetch) para minimizar fallbacks.

## Dependências de Componentes

| Componente | Crate | Feature gate |
|---|---|---|
| VkBackend | `swamp-gpu` | — |
| TransferEngine | `swamp-gpu::streaming` | — |
| ComputeGraph | `swamp-gpu` | — |
| ShaderCache | `swamp-gpu` | — |
| ShardReader | `swamp-gpu::streaming` | — |
| GpuComputeContext | `swamp-gpu` | — |
| PerLayerGpuState | `swamp-engine` | `gpu` |
| ModelExecutor | `swamp-engine` | — |
| DSPark | `swamp-engine` | — |
| AIMD | `swamp-engine` | — |
| CPU Kernels | `swamp-kernels` | — |

## Referências

- [[sprint1-vulkan-transfer.md]] — Detalhes da implementação Vulkan
- [[sprint2-shard-io.md]] — ShardReader + mmap
- [[sprint3-dspark-scheduling.md]] — DSPark + schedule resolver
- [[sprint4-fallback-cpu.md]] — Fallback CPU path
- [[sprint5-aimd-governor.md]] — AIMD + ResourceGovernor
- [[integration-plan.md]] — Pontes entre componentes
