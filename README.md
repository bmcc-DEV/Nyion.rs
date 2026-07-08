# LLamanyon.rs

Inferência de LLM em Rust + Mojo + CUDA, targeting CPU AVX-512 e GPU NVIDIA.

**Branch:** `experimental-1m` — otimizado para 1M contexto + GPU GEMV dispatch.

---

## Stack

```
┌─────────────────────────────────────────────────┐
│  swamp-tools (benchmarks, CLI)                  │
├─────────────────────────────────────────────────┤
│  swamp-engine (executor, ops, linear, cache)     │
│    ├── DSpark (speculative decoding n-gram)      │
│    ├── Fugu (strategy orchestrator)              │
│    ├── Execution MoE (expert router VNNI/AVX2)   │
│    ├── VNpu (EWMA schedulers)                    │
│    ├── QAT (calibration 4-bit)                   │
│    ├── LSC (prefetcher)                          │
│    ├── AIMD Resource Ramp                        │
│    ├── PowerArbiter CPU+iGPU                     │
│    ├── StagingBuffer decoupled RAM               │
│    ├── ModelRegistry + ModelSwapper              │
│    └── Pipeline (concurrent stages)              │
├─────────────────────────────────────────────────┤
│  swamp-kernels (fused_gemv_q4k, fused_gemv_q6k) │
│    ├── CPU: VNNI (AVX-512) + AVX2 + scalar      │
│    └── GPU: kernel_gemv_q4k (CUDA, sm_75)        │
├─────────────────────────────────────────────────┤
│  swamp-gpu (CUDA + Mojo FFI bridge)              │
│    ├── fused_attention.cu (858 linhas)           │
│    ├── gpu_gemv_q4k (device-only kernel)         │
│    └── libswamp_mojo.so (Mojo 1.0.0b2)          │
├─────────────────────────────────────────────────┤
│  swamp-gguf (leitor GGUF, dequant)               │
├─────────────────────────────────────────────────┤
│  swamp-tensors (operações tensoriais)            │
└─────────────────────────────────────────────────┘
```

---

## Performance Atual

### Decode (TinyLlama 1.1B Q4_K_M, 128 ctx, 6 threads)

| Configuração | ms/tok | tok/s | vs baseline |
|---|---|---|---|
| Baseline (CPU-only) | 75.4 | 13 | 1× |
| + Pre-quant + Ring | 38.0 | 26 | 2.0× |
| + GPU GEMV dispatch (7/layer) | **~19** | **~50** | **~4×** |

### Prefill (128 tokens)

| Configuração | TTFT | tok/s |
|---|---|---|
| CPU (pre-quant + ring) | ~4100ms | 31 |
| VNNI weight-sharing prefill | ~2500ms | 50 |
| GPU GEMV + attention (prefill) | ~1200ms (estimado) | ~106 |

### 1M Contexto (projetado)

| Componente | Consumo |
|---|---|
| KV cache f32 (full attention) | 44 GB ❌ |
| KV cache 4-bit (full attention) | 7 GB ⚠️ |
| KV cache 4-bit + atenção esparsa (window=4096) | 7 GB ✅ |
| Modelo (Q4_K) | 549 MB ✅ |
| **Total RAM** | **~8 GB** ✅ |

---

## Componentes

### ✅ CPU (estável)

| Componente | Arquivo | Descrição |
|---|---|---|
| **Pre-quant kernel** | `swamp-kernels/src/fused_gemv_q4k.rs` | Phase 1: quantiza x uma vez. Phase 2: linhas sequenciais. |
| **Ring dispatch** | `swamp-engine/src/linear.rs` | `forward_gemvs_ring()` — 4 dispatches Rayon/layer. |
| **Atenção esparsa** | `swamp-engine/src/ops.rs` | `attention_sparse()` — window=4096 + global tokens. O(n). |
| **KV cache 4-bit** | `swamp-engine/src/cache.rs` | `save_q4()`, compressão 6.4:1. 7 GB p/ 1M ctx. |
| **mlockall + HUGEPAGE** | `model.rs` | Zero page faults. |
| **MSR 0x1FC unlock** | `wrmsr -a 0x1FC 0x4004005f` | +29% AVX-512. |

### ✅ GPU (ativado via `--features gpu`)

| Componente | Arquivo | Status |
|---|---|---|
| **7 GEMVs/layer dispatch** | `scheduler.rs` + `executor.rs:617-791` | `gemv_qkv_async`, `gemv_gate_up_async`, `execute_gemv_async` com batch x copy. Fallback CPU automático. |
| **CUDA attention graph** | `scheduler.rs` | Graph cache por seq_len, replay em 1 launch. |
| **FP16 KV cache GPU** | `scheduler.rs` | Upload async via copy stream. |
| **Upload pesos VRAM** | `executor.rs:419-452` | 7 tensores (Q/K/V/O/Gate/Up/Down) enviados na init. |
| **swamp_continuum SMs** | `fused_attention.cu:1144` | `<<<sm_count,256>>>` (usa 14/14 SMs, antes 1/14). |
| **Prefetch GEMV tiling** | `fused_gemv_q4k.rs` | `_mm_prefetch(next_row, T0)` no VNNI. |

### ✅ Inovação (experimental-1m)

| Componente | Arquivo | Descrição |
|---|---|---|
| **AIMD Resource Ramp** | `aimd.rs` | Dobra budget a cada 500ms sem stress, corta no 1º sinal. |
| **StagingBuffer** | `staging.rs` | `num_slices` × `slice_size` + `combine_into()`. |
| **PowerArbiter** | `power_arbiter.rs` | RAPL+temp define split CPU/iGPU. |
| **ModelRegistry** | `model_registry.rs` | Múltiplos GGUFs por tier VRAM/RAM/CPU. |
| **ModelSwapper** | `model_swapper.rs` | LRU + pipeline prefetch. |
| **Pipeline** | `pipeline.rs` | 7 estágios, `execute_concurrent()` com `sync_channel`. |
| **Speculative decoding** | `dspark.rs` + `executor.rs:748` | Draft + acceptance check (confiança > 0.6). |
| **Adaptive precision** | `linear.rs:115` | SENSITIVITY_MAP: rows sensíveis em FP16, resto Q4_K. |

---

## Uso

### Pré-requisitos

```bash
# AVX-512 unlock (antes de cada sessão)
sudo wrmsr -a 0x1FC 0x4004005f

# Verificar
sudo rdmsr -a 0x1FC
# Esperado: 0x4004005f
```

### Benchmark CPU

```bash
cargo run --release -p swamp-tools --bin swamp-benchmark-prefill -- \
  /caminho/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  /caminho/tokenizer.json \
  --prompt-tokens 128 --repeats 5
```

### Benchmark GPU (7 GEMVs/layer dispatch)

```bash
cargo run --release --features gpu -p swamp-tools --bin swamp-benchmark-prefill -- \
  /caminho/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  /caminho/tokenizer.json \
  --prompt-tokens 128 --repeats 5 --n-threads 6
```

### Opções do benchmark

| Flag | Default | Descrição |
|---|---|---|
| `--prompt-tokens` | 128 | Tokens de prefill |
| `--repeats` | 5 | Repetições |
| `--n-threads` | 6 | Threads CPU |
| `--qat` | off | Calibração QAT antes do decode |

---

## GPU GEMV Dispatch Flow

```
Layer l:
  ┌─ x_norm (CPU RMSNorm) ────────────────────────┐
  │                                                │
  │  [GPU] gemv_qkv_async(x_norm → q,k,v)         │
  │    ⋮── 1× copy x to VRAM                      │
  │    ⋮── 3× kernel_gemv_q4k (Q,K,V)             │
  │    └── 3× copy out to host                       │
  │                                                │
  │  RoPE (CPU)                                    │
  │  KV cache save (CPU)                           │
  │                                                │
  │  [GPU] execute_attention_async(q → attn_out)   │
  │    ⋮── 1× copy q to VRAM                       │
  │    ⋮── 1× CUDA Graph (attention)               │
  │    └── 1× copy out to host                       │
  │                                                │
  │  [GPU] execute_gemv_async(attn_out → wo_out)   │
  │    ⋮── 1× copy + kernel + copy                 │
  │                                                │
  │  add_in_place (CPU)                            │
  │  RMSNorm FFN (CPU)                             │
  │                                                │
  │  [GPU] gemv_gate_up_async(x_norm → gate,up)   │
  │    ⋮── 1× copy x to VRAM                       │
  │    ⋮── 2× kernel_gemv_q4k (Gate,Up)           │
  │    └── 2× copy out to host                       │
  │                                                │
  │  silu + mul (CPU)                              │
  │                                                │
  │  [GPU] execute_gemv_async(ffn_gate → ffn_down)│
  │    ⋮── 1× copy + kernel + copy                 │
  │                                                │
  │  add_in_place (CPU)                            │
  └───────────────────────────────────────────────┘
  Total: 4 GPU syncs/layer (1 por batch GEMV)
```

---

## Roadmap pra 100+ tok/s

```
1. ✅ GPU GEMV dispatch (7/layer) ─── batch QKV + GateUp
2. ⬜ CUDA Graph fused (GEMVs + attention em 1 launch/layer)
3. ⬜ Atenção GPU esparsa (window=4096 em VRAM)
4. ⬜ Pipeline concurrente = prefill + decode sobrepostos
```

---

## Comandos Úteis

```bash
# Monitorar GPU
nvidia-smi -l 1

# MSR (precisa reaplicar após reboot)
sudo wrmsr -a 0x1FC 0x4004005f

# Build GPU
cargo build --release --features gpu

# Build CPU
cargo build --release

# Verificar erros
cargo build --release 2>&1 | grep "^error"
```
