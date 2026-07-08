# LLamanyon.rs

Inferência de LLM em Rust + CUDA, targeting GPU NVIDIA (GTX 1650) + CPU AVX-512.

**Branch:** `experimental-1m` — GPU GEMV dispatch, CUDA Graphs, atenção esparsa (window=4096).

---

## Stack

```
┌─────────────────────────────────────────────────┐
│  swamp-tools (benchmarks, CLI)                  │
├─────────────────────────────────────────────────┤
│  swamp-engine (executor, ops, linear, cache)     │
│    ├── ModelExecutor (generate + generate_batch)│
│    ├── GPU GEMV dispatch (7 GEMVs/layer)        │
│    ├── CUDA Graph cache (QKV, GateUp, O, Down)  │
│    ├── Atenção esparsa (window=4096 em VRAM)    │
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
│  swamp-gpu (CUDA FFI bridge)                     │
│    ├── fused_attention.cu (1348 linhas)          │
│    │   ├── kernel_gemv_q4k (device-only)         │
│    │   ├── gpu_graph_create_gemv_qkv             │
│    │   ├── gpu_graph_create_gemv_gate_up         │
│    │   └── gpu_graph_create_gemv_single          │
│    └── libswamp_gpu.so (rebuilt via make)        │
├─────────────────────────────────────────────────┤
│  swamp-gguf (leitor GGUF, dequant)               │
└─────────────────────────────────────────────────┘
```

---

## Performance

### Decode (TinyLlama 1.1B Q4_K_M, GTX 1650, 128 ctx)

| Configuração | ms/tok | tok/s | vs baseline |
|---|---|---|---|
| CPU-only (AVX-512) | 75.4 | 13 | 1× |
| + Ring dispatch | 38.0 | 26 | 2.0× |
| + GPU GEMV dispatch (7/layer) | ~19 | ~50 | ~4× |
| + CUDA Graph fused (4 replays/layer) | ~15 | ~65 | ~5× |
| + Pipeline concurrente (--pipeline 4) | ~4 | ~250 | ~19× |

### 1M Contexto

| Componente | Consumo |
|---|---|
| KV cache f32 (full attention) | 44 GB ❌ |
| KV cache 4-bit + window=4096 GPU | 22 MB VRAM + 7 GB RAM ✅ |
| Modelo (Q4_K) | 549 MB ✅ |
| **Total VRAM** | **~1.1 GB** ✅ (cabe na GTX 1650 4GB) |

---

## GPU: como funciona

### GEMV dispatch (7 operações por camada)

```
Layer l:
  ┌─ x_norm (CPU RMSNorm) ────────────────────────┐
  │                                                │
  │  [CUDA Graph] gemv_qkv_async                   │
  │    ├── cudaMemcpyAsync x → d_gemv_x            │
  │    ├── kernel_gemv_q4k (Q, K, V)              │
  │    └── 3× cudaMemcpyAsync → host (q,k,v)       │
  │                                                │
  │  RoPE + KV save (CPU)                          │
  │                                                │
  │  [CUDA Graph] execute_attention_async          │
  │    ├── cudaMemcpyAsync q → d_q                 │
  │    ├── CUDA Graph replay (attention)           │
  │    └── cudaMemcpyAsync → host (attn_out)       │
  │                                                │
  │  [CUDA Graph] execute_gemv_async (O)           │
  │  add_in_place + RMSNorm (CPU)                  │
  │                                                │
  │  [CUDA Graph] gemv_gate_up_async               │
  │  silu + mul (CPU)                              │
  │                                                │
  │  [CUDA Graph] execute_gemv_async (Down)        │
  │  add_in_place (CPU)                            │
  └───────────────────────────────────────────────┘
  Total: 4 CUDA Graph replays + 4 syncs/layer
```

### Atenção esparsa (window=4096)

KV cache em VRAM usa **ring buffer de 4096 posições** (22 MB/layer × 22 = 484 MB). Tokens anteriores a window são servidos da RAM do host via PCIe.

### CUDA Graphs (fused GEMV)

Cada graph captura: `cudaMemcpyAsync` + `kernel_gemv_q4k` × N + `cudaMemcpyAsync` × N numa única chamada `cudaGraphLaunch`. Criado lazy no 1º uso, replay nos seguintes.

3 graph types:
- `gpu_graph_create_gemv_qkv` — 1 copy + 3 kernels + 3 copy backs
- `gpu_graph_create_gemv_gate_up` — 1 copy + 2 kernels + 2 copy backs
- `gpu_graph_create_gemv_single` — 1 copy + 1 kernel + 1 copy back

---

## Uso

### Build

```bash
# GPU habilitado (--features gpu)
cargo build --release --features gpu

# CPU-only
cargo build --release
```

### Benchmark

```bash
# Single request (usa GPU GEMV + CUDA Graphs + atenção sparsa)
cargo run --release --features gpu -p swamp-tools --bin swamp-benchmark-prefill -- \
  /caminho/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  /caminho/tokenizer.json \
  --prompt-tokens 128 --repeats 3

# Pipeline concurrente (N requests em paralelo)
cargo run --release --features gpu -p swamp-tools --bin swamp-benchmark-prefill -- \
  /caminho/modelo.gguf /caminho/tokenizer.json \
  --prompt-tokens 128 --pipeline 4
```

### Opções

| Flag | Default | Descrição |
|---|---|---|
| `--prompt-tokens` | 128 | Tokens de prefill |
| `--repeats` | 1 | Repetições do benchmark |
| `--n-threads` | 6 | Threads CPU |
| `--qat` | off | Calibração QAT |
| `--pipeline` | 1 | Pipeline parallelism (N requests concorrentes) |

### MSR unlock (AVX-512)

```bash
sudo wrmsr -a 0x1FC 0x4004005f  # +29% frequência AVX-512
```

---

## Pipeline Concurrente

```bash
# 4 requests em paralelo (prefill sobrepõe decode)
cargo run --release --features gpu -p swamp-tools --bin swamp-benchmark-prefill -- \
  /caminho/modelo.gguf /caminho/tokenizer.json --pipeline 4
```

Cada worker roda `ModelExecutor::generate()` independente com KV cache própria. Job queue via `Arc<Mutex<Vec>>>`.

---

## Componentes

### ✅ GPU (via `--features gpu`)

| Componente | Arquivo | Descrição |
|---|---|---|
| **7 GEMVs/layer dispatch** | `scheduler.rs:350-471` | CUDA Graph replay com fallback async |
| **CUDA Graph fused** | `fused_attention.cu:1218-1348` | QKV, GateUp, Single — 4 replays/layer |
| **Atenção sparsa window** | `scheduler.rs:267-340` | Ring buffer KV em VRAM (window=4096) |
| **CUDA attention graph** | `scheduler.rs:70-164` | Graph cache por seq_len |
| **FP16 KV cache GPU** | `scheduler.rs:262-265` | Upload async via copy stream |
| **Upload pesos VRAM** | `executor.rs:399-424` | 7 tensores na init via stream temporário |
| **swamp_continuum SMs** | `fused_attention.cu:1144` | `<<<sm_count,256>>>` (14/14 SMs) |

### ✅ Inovação

| Componente | Arquivo | Descrição |
|---|---|---|
| **AIMD Resource Ramp** | `aimd.rs` | Dobra budget a cada 500ms, corta no 1º sinal |
| **PowerArbiter** | `power_arbiter.rs` | RAPL+temp define split CPU/iGPU |
| **ModelRegistry** | `model_registry.rs` | Múltiplos GGUFs por tier VRAM/RAM/CPU |
| **ModelSwapper** | `model_swapper.rs` | LRU + pipeline prefetch |
| **Pipeline** | `pipeline.rs` | 7 estágios, `execute_concurrent()` |
| **Speculative decoding** | `dspark.rs` | Draft + acceptance check |
| **Adaptive precision** | `linear.rs:115` | SENSITIVITY_MAP: rows sensíveis em FP16 |

---

## Comandos Úteis

```bash
# Rebuild .so (após modificar .cu)
make -C swamp-gpu

# Build GPU
cargo build --release --features gpu

# Verificar erros
cargo build --release 2>&1 | grep "^error"

# Testes
cargo test --release -p swamp-engine -p swamp-kernels

# MSR unlock
sudo modprobe msr && sudo wrmsr -a 0x1FC 0x4004005f
```
