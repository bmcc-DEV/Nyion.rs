# LLamanyon.rs

Inferência de LLM em Rust + Mojo + CUDA, targeting CPU AVX-512 e GPU NVIDIA.

**Branch:** `experimental-1m` — otimizado para 1M contexto + 25 tok/s decode.

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
│    └── LSC (prefetcher)                          │
├─────────────────────────────────────────────────┤
│  swamp-kernels (fused_gemv_q4k, fused_gemv_q6k) │
│    ├── CPU: VNNI (AVX-512) + AVX2 + scalar      │
│    └── GPU: kernel_gemv_q4k (CUDA, sm_75)        │
├─────────────────────────────────────────────────┤
│  swamp-gpu (CUDA + Mojo FFI bridge)              │
│    ├── fused_attention.cu (858 linhas)           │
│    ├── attention.mojo (esqueleto CPU/GPU)        │
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
| Baseline (commitado) | 75.4 | 13 | 1× |
| + Pre-quant kernel (Phase 1+2 sequencial) | 52.6 | 19 | 1.46× |
| + Ring dispatch (4 grupos/layer) | 45.5 | 22 | 1.69× |
| + `sudo wrmsr -a 0x1FC 0x4004005f` | **38.0** | **26** | **2.0×** |
| + GPU GEMV (pesos em VRAM, 192 GB/s) | 79.0 | 13 | 1.0× |

**GPU lento** devido a 154 lançamentos de kernel por token (cada tensor individual). **Solução: kernel fundido** (7 GEMVs/layer em 1 lançamento).

### Prefill (128 tokens)

| Configuração | TTFT | tok/s |
|---|---|---|
| CPU (pre-quant + ring) | ~4100ms | 31 |
| VNNI weight-sharing prefill | ~2500ms | 50 |

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

### ✅ Funcionando (CPU)

| Componente | Arquivo | Descrição |
|---|---|---|
| **Pre-quant kernel** | `swamp-kernels/src/fused_gemv_q4k.rs` | Fase 1: quantiza x uma vez. Fase 2: linhas sequenciais em DDR4. Sem strides. |
| **Ring dispatch** | `swamp-engine/src/linear.rs` | `forward_gemvs_ring()` — 4 dispatches Rayon/layer (QKV, O, GateUp, Down). |
| **Atenção esparsa** | `swamp-engine/src/ops.rs` | `attention_sparse()` — window=4096 + 128 tokens globais. O(n) em vez de O(n²). |
| **KV cache 4-bit** | `swamp-engine/src/cache.rs` | `save_q4()` + `k_q4_page_ptr()`. Compressão 6.4:1. 7 GB pra 1M ctx. |
| **Attention sparse 4-bit AVX-512** | `swamp-engine/src/ops.rs` | `attention_sparse_q4()` — 32 valores/SIMD. Desquantização on-the-fly. |
| **mlockall + MADV_HUGEPAGE** | `model.rs` + `benchmark` | Zero page faults durante inferência. |
| **Ring buffer pre-allocation** | `model.rs` | Pré-aloca 512MB pra forçar páginas físicas na região dual-channel. |
| **MSR 0x1FC unlock** | `wrmsr -a 0x1FC 0x4004005f` | +29% frequência AVX-512. |

### ✅ Existentes (precisa conectar)

| Componente | Arquivo | Status |
|---|---|---|
| **DSpark speculative** | `dspark.rs` + `executor.rs:616` | Batched verify integrado. Sampler NaN fixado. |
| **Fugu orchestrator** | `fugu.rs` + `executor.rs:456` | Auto-sparse + auto-DSpark. EWMA acceptance rate. |
| **Execution MoE** | `execution_moe.rs` | Router VNNI/AVX2/Scalar adaptativo (EWMA por profile). |
| **VNpu scheduler** | `vnpu.rs` | Agendamento com budget. |
| **QAT calibration** | `qat.rs` | 3.718 super-blocks otimizados (layers 0-2). `--qat` no benchmark. |
| **GPU attention pipeline** | `scheduler.rs` | Async CUDA streams, graph cache, FP16 KV. |

### 🟡 GPU (parcial)

| Componente | Arquivo | Status |
|---|---|---|
| CUDA attention kernel | `fused_attention.cu` | Compilado (sm_75, 858 linhas) |
| CUDA GEMV kernel | `fused_attention.cu` | `kernel_gemv_q4k` + `gpu_gemv_q4k_prealloc` |
| Upload pesos p/ VRAM | `benchmark_prefill.rs` | 549 MB enviados ✅ |
| Pre-alloc buffers GPU | `benchmark_prefill.rs` | `d_x`, `d_out` persistentes |
| **Fused GEMV (7 em 1)** | ❌ | **Necessário pra 100+ tok/s** |

### ❌ Não implementado

| Componente | Motivo |
|---|---|
| Fused GPU GEMV (7 tensores/layer) | Overhead de 154 kernel launches domina |
| Mojo kernel estável | Mojo 1.0.0b2 API muito volátil |
| Assembly Forth hot path | Mojo precisa estabilizar primeiro |

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

### Benchmark GPU

```bash
cargo run --release --features gpu -p swamp-tools --bin swamp-benchmark-prefill -- \
  /caminho/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  /caminho/tokenizer.json \
  --prompt-tokens 128 --repeats 5
```

### Opções do benchmark

| Flag | Default | Descrição |
|---|---|---|
| `--prompt-tokens` | 128 | Tokens de prefill |
| `--repeats` | 5 | Repetições |

---

## Arquitetura dos Kernels

### Pre-quant VNNI (CPU, Phase 1 + 2)

```
Phase 1: [encontrar x_max global] → quantizar TODO x_i8 → pre-computar va_fulls
Phase 2: [para cada linha] → [para cada superbloco] → VNNI dpbusd → collapse único
```

**Por que é rápido:** Acesso sequencial à DDR4 (sem strides entre linhas). 
**Antes:** column-first processava blk×coluna para TODAS as linhas → stride de 1152 bytes.
**Depois:** row-first processa UMA linha inteira (todos os blocos) → acesso 100% sequencial.

### Atenção esparsa + KV 4-bit

```
[Q head] × [K posições esparsas (window + global)] → scores → softmax online → weighted sum V
                                                                              
K/V armazenados como: [d(f16), dmin(f16), 32 nibbles(4-bit)] = 20 bytes / 32 valores
Compressão: 128 bytes f32 → 20 bytes = 6.4:1
```

### Ring dispatch

```
Layer l: [Q_data][K_data][V_data][O_data][Gate_data][Up_data][Down_data]
          └─── 4 dispatches ───┘
          QKV  |  O  | GateUp | Down
```

Cada thread Rayon processa linhas `t*rpt..(t+1)*rpt` para todos os tensores do grupo.

---

## Roadmap pra 100+ tok/s

```
1. Kernel GPU fundido (7 tensores/layer em 1 kernel CUDA) ─── 100+ tok/s
   ├── Um único cudaMemcpyAsync pra x (entrada)
   ├── Um kernel processa Q,K,V,O,Gate,Up,Down
   └── Um cudaMemcpyAsync pra resultados

2. Atenção GPU esparsa ─── win=4096 vai pra VRAM, processa a 192 GB/s

3. DSpark + Fugu + MoE + VNpu + QAT ─── orquestração total no executor
```

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
```
