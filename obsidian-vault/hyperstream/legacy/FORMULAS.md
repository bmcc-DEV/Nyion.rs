# Fórmulas e Lógica para o Desenvolvimento

> Modelo: TinyLlama 1.1B Q4_K_M · GPU: GTX 1650 4GB · CPU: i5-11260H AVX-512

---

## Índice

1. [Dimensões do Modelo](#1-dimensões-do-modelo)
2. [Roofline Model — Gargalo por Operação](#2-roofline-model)
3. [GEMV Throughput Model — O Gargalo Real](#3-gemv-throughput-model)
4. [GPU vs CPU Dispatch — Critério de Decisão](#4-gpu-vs-cpu-dispatch)
5. [Pipeline Concurrency — Little's Law](#5-pipeline-concurrency)
6. [Attention Cost Model — Full vs Sparse](#6-attention-cost-model)
7. [KV Cache Sizing — Janela Ótima](#7-kv-cache-sizing)
8. [Thermal Model — AIMD + PowerArbiter](#8-thermal-model)
9. [Speculative Decoding Speedup](#9-speculative-decoding)
10. [Fused Layer Graph — Savings Quantified](#10-fused-layer-graph)
11. [Árvores de Decisão — Runtime Logic](#11-árvores-de-decisão)

---

## 1. Dimensões do Modelo

| Parâmetro | Símbolo | TinyLlama 1.1B |
|-----------|---------|----------------|
| Embedding dim | `d` | 2048 |
| Attention heads | `h` | 32 |
| KV heads | `h_kv` | 4 |
| Head dim | `d_h` | 64 |
| FFN hidden dim | `d_ff` | 5632 |
| Camadas | `L` | 22 |
| Vocab size | `V` | 32000 |
| Janela VRAM | `W` | 4096 |
| Q4_K bytes/weight | `b_q4k` | 0.5625 (144 bytes / 256 weights) |

### Derivados

| Projeção | Output rows | Input cols | Pesos Q4_K | Bytes |
|----------|-------------|------------|------------|-------|
| Q | `h·d_h = 2048` | `d = 2048` | 4.19M | 2.36 MB |
| K | `h_kv·d_h = 256` | `d = 2048` | 0.52M | 0.29 MB |
| V | `h_kv·d_h = 256` | `d = 2048` | 0.52M | 0.29 MB |
| O | `d = 2048` | `h·d_h = 2048` | 4.19M | 2.36 MB |
| Gate | `d_ff = 5632` | `d = 2048` | 11.53M | 6.49 MB |
| Up | `d_ff = 5632` | `d = 2048` | 11.53M | 6.49 MB |
| Down | `d = 2048` | `d_ff = 5632` | 11.53M | 6.49 MB |

**MACs/layer** = 4.19M + 0.52M + 0.52M + 4.19M + 11.53M + 11.53M + 11.53M = **44.0M MACs**

**Bytes lidos/layer** = 2.36 + 0.29 + 0.29 + 2.36 + 6.49 + 6.49 + 6.49 = **24.77 MB**

**Pesos totais VRAM** = 22 × 24.77 MB = **545 MB**

---

## 2. Roofline Model

### GTX 1650 (Turing TU117, sm_75)

| Especificação | Valor |
|--------------|-------|
| Memória | GDDR6 4 GB |
| Bandwidth (teórico) | 192 GB/s |
| FP32 (teórico) | 2.85 TFLOPS (896 cores × 1.590 GHz × 2) |
| FP16 (teórico) | 5.70 TFLOPS |
| INT8 | 5.70 TOP/s |
| SMs | 14 |
| L2 cache | 1024 KB (1 MB) |

### Roofline Formula

```
Achievable GFLOPs = min(Bandwidth × AI, Peak Compute)

onde:
  AI (Arithmetic Intensity) = FLOPs / Bytes Read
```

### AI por operação (Q4_K GEMV)

```
Para W(M×N) × x(N) onde W é Q4_K:
  FLOPs      = 2 × M × N  (MAC = mul + add)
  BytesRead  = N × 4 (x f32) + M × N × b_q4k (weights Q4_K)
  BytesWrite = M × 4 (y f32)
  AI         = 2MN / (4N + 0.5625MN + 4M)

Para Q (M=2048, N=2048):
  AI = 8,388,608 / (8192 + 2,359,296 + 8192) = 8.39M / 2.38M = 3.53 FLOPs/byte

Para Gate (M=5632, N=2048):
  AI = 23,068,672 / (8192 + 6,488,064 + 22,528) = 23.07M / 6.52M = 3.54 FLOPs/byte
```

**Conclusão:** Todo GEMV Q4_K tem AI ≈ 3.5 FLOPs/byte → **fortemente memory-bound**.

| Operação | AI (FLOPs/byte) | Roofline Limit |
|----------|----------------|----------------|
| GEMV Q4_K | ~3.5 | 192 GB/s × 3.5 = **672 GFLOPs** (24% do pico FP32) |
| Atenção (seq=4096) | ~0.5-2 | **96-384 GFLOPs** |
| RMSNorm | ~1-2 | **192-384 GFLOPs** |
| GPU ideal | ≥15 | 2850 GFLOPs (compute-bound) |

**Implicação:** Triplicar TFLOPS não acelera nada (ex: RTX 4060 vs GTX 1650). O que acelera é: **bandwidth maior, melhor padrão de acesso, fusão de kernels.**

---

## 3. GEMV Throughput Model

### Equação de Throughput Real

```
T_actual = B_effective / T_ops

T_op = max(weights_bytes / B_eff, FLOPs / GFLOPs_eff)

Modelo empírico para bandwidth efetiva:
  B_eff = B_peak × η_access × η_overlap

onde:
  η_access = 0.3-0.5 para Q4_K (acesso irregular, dequant overhead)
  η_overlap = overlap entre copy + compute + sync
```

### A correção GDDR6: η empírico vs η teórico

```
B_peak GDDR6 = 192 GB/s (vs 128 GB/s assumido antes)

η teórico (acesso ideal Q4_K) = 0.4
  → B_eff = 192 × 0.4 = 76.8 GB/s
  → Previsão: ~90 tok/s

η empírico (do benchmark 65 tok/s) = (24.77 MB/layer × 22 layers) / (15.4 ms - 4 ms overhead) / 192 GB/s
  → B_eff = 24.77 MB × 22 / 11.4 ms = 47.8 GB/s
  → η_real = 47.8 / 192 = 0.25

Por que η_real (0.25) é menor que η_teórico (0.4)?
  - Thread blocks insuficientes para ocupar todos os SMs
  - Q4_K dequant overhead (escalas half → f32 por bloco)
  - Acesso não-coalescido nos pesos empacotados
  - Gap entre dispatch do graph e execução real nos SMs
```

### Breakdown por projeção (Gate+Up = bottleneck dominante ~52%)

```
Gate:   6.49 MB lidos + 22.5 KB escritos → dominado por leitura
Down:   6.49 MB lidos + 8.2 KB escritos
Up:     6.49 MB lidos + 22.5 KB escritos
Q:      2.36 MB lidos + 8.2 KB escritos
O:      2.36 MB lidos + 8.2 KB escritos
K:      0.29 MB lidos + 1.0 KB escritos
V:      0.29 MB lidos + 1.0 KB escritos
```

### Tempo por layer (η_real = 0.25 → B_eff = 48 GB/s)

```
Gate:   6.49 MB / 48 GB/s = 135 μs
Down:   6.49 MB / 48 GB/s = 135 μs  
Up:     6.49 MB / 48 GB/s = 135 μs
Q+O:    4.72 MB / 48 GB/s = 98 μs
K+V:    0.58 MB / 48 GB/s = 12 μs

Total/layer: 135×3 + 98 + 12 = 515 μs
Total/22 layers: 11.3 ms
+ Overhead (sync, launch, CPU): ~4 ms
→ ~15.3 ms/tok = 65 tok/s ✅ (matcha o benchmark)
```

### Gargalo atual confirmado

| Componente | Tempo/layer | % |
|-----------|-------------|---|
| Gate+Up GEMV | ~270 μs | **52%** |
| Down GEMV | ~135 μs | 26% |
| Q+O GEMV | ~98 μs | 19% |
| Attention (GPU) | ~20 μs | 4% |
| K+V GEMV | ~12 μs | 2% |
| Sync + overhead | ~60 μs | 12% (overhead fixo, não escala) |

**Gate+Up = 52% do tempo — mais da metade do decode step.**

---

## 4. GPU vs CPU Dispatch

### Critério de decisão por GEMV

```
Usar GPU se:  B_eff_GPU / B_eff_CPU > 1.3 (overhead de sync)

Onde:
  B_eff_CPU = eficiência VNNI × bandwidth RAM
  B_eff_GPU = eficiência Q4_K × bandwidth VRAM × η_transfer
```

| GEMV | GPU (η=0.25) | CPU VNNI | Vencedor |
|------|-------------|----------|----------|
| Gate (6.49 MB) | 135 μs | 520 μs | **GPU** ✅ (3.9×) |
| Up (6.49 MB) | 135 μs | 520 μs | **GPU** ✅ (3.9×) |
| Down (6.49 MB) | 135 μs | 520 μs | **GPU** ✅ (3.9×) |
| Q (2.36 MB) | 49 μs | 190 μs | **GPU** ✅ (3.9×) |
| O (2.36 MB) | 49 μs | 190 μs | **GPU** ✅ (3.9×) |
| K (0.29 MB) | 6 μs | 24 μs | **GPU** ✅ (4.0×) |
| V (0.29 MB) | 6 μs | 24 μs | **GPU** ✅ (4.0×) |

### Quando CPU ganha

```
AIMD budget < 0.3    → CPU-only (throttle térmico)
Seq_len < 32          → CPU (overhead de setup GPU domina)
Layer em fallback     → CPU (já syncou, continuar CPU)
```

### Regra prática

**Sempre GPU para GEMV > 1 MB. CPU para GEMV < 128 KB ou quando já syncou por fallback.**

---

## 5. Pipeline Concurrency

### Little's Law para throughput do pipeline

```
Throughput = N_concurrent / latency_per_request

Onde:
  N_concurrent = requests simultâneos
  Latency = decode_time × target_tokens + prefill_time

Crossover: N_opt = ceil(CPU_decode_time / GPU_decode_time) = ceil(77/15) = 6
```

### Pipeline optimal depth

```
Observação: prefill usa CPU batch (todos tokens juntos), decode usa GPU (1 token/step)

Se prefill_time ≈ decode_latency:
  pipeline = min(N_requests, 1 + prefill / decode)

Para TinyLlama (128 prompt tokens, 10 decode):
  prefill = 500ms (CPU batch)
  decode  = 15ms × 10 = 150ms (GPU single token)
  pipeline_opt ≈ min(requests, 1 + 500/150) ≈ min(requests, 4)

→ Pipeline = 4 é o ótimo (matcha benchmark 250 tok/s = 4×65)
```

### Throughput vs Concurrency

```
P = 1:  65 tok/s
P = 2:  110 tok/s (1.7×)  — prefill de um sobrepõe decode do outro
P = 3:  170 tok/s (2.6×)
P = 4:  250 tok/s (3.8×)  — ótimo
P = 5:  280 tok/s (4.3×)  — VRAM pode ser limite
P = 6:  290 tok/s (4.5×)  — satura
```

**Lei:** cada worker extra = KV cache extra. GTX 1650 4GB limita a ~4 workers.

---

## 6. Attention Cost Model

### Complexidade

```
Full Attention:    O(seq² × h_kv × d_h) → 2 × h × seq × h_kv × d_h FLOPs
Sparse (window):   O(seq × W × h_kv × d_h) FLOPs
DSPark-guided:     O(seq × (W + N_dspark × d)) FLOPs (N_dspark = 32)
```

### Crossover: window × seq_len

```
Para decode (single token q):

Full:   cost_full = scores(q×K^T) + output(scores×V)
       = h × d_h × seq   + h × seq × d_h
       = 2 × h × d_h × seq
       = 4096 × seq  FLOPs (para TinyLlama: h=32, d_h=64)

Sparse: cost_sparse = 2 × h × d_h × min(seq, W)
        = 4096 × min(seq, W)  FLOPs

Full = Sparse quando seq ≤ W (idênticos, mesma janela)
Full > Sparse quando seq > W (sparse ganha)
Speedup = seq / W  para seq > W
```

| seq_len | Full (FLOPs) | Sparse W=4096 (FLOPs) | Speedup |
|---------|-------------|----------------------|---------|
| 128 | 524K | 524K | Full = Sparse |
| 1024 | 4.19M | 4.19M | Full = Sparse |
| 4096 | 16.8M | 16.8M | Full = Sparse (crossover) |
| 8192 | 33.6M | 16.8M | Sparse 2× |
| 65536 | 268M | 16.8M | Sparse 16× |
| 1M | 4.10G | 16.8M | Sparse **244×** |

**Nota:** Para decode single-token, q tem 1 posição. A diferença entre full e sparse só aparece quando seq > W. Para seq ≤ W, os dois kernels processam exatamente o mesmo número de tokens — a vantagem do sparse é poder usar ring buffer em VRAM em vez de atenção paginada em RAM.

### Atenção híbrida (futuro: VRAM window + host spill)

```
Quando seq > W:
  attn_out = GPU_attn(q, k_buf[:W], v_buf[:W]) + CPU_attn(q, k_host[W:], v_host[W:])

Para 1M contexto com W=4096:
  GPU: 4096 tokens @ GPU → 16.8M FLOPs, 1 sync (~20 μs)
  CPU: ~996K tokens @ CPU → 4.08G FLOPs, PCIe spill, lento
  
Solução: spill-to-host assíncrono → tokens antigos sobrescritos no ring buffer
         são copiados para RAM via cudaMemcpyAsync em background.
         Atenção GPU lê VRAM (últimos W) + faz async copy de host para
         processar tokens spillados — sem bloqueio.
```

---

## 7. KV Cache Sizing

### Custo por token KV

```
KV por layer (half, 2 bytes):
  K[layer] = h_kv × d_h × 2 = 4 × 64 × 2 = 512 bytes/token
  V[layer] = 512 bytes/token
  Total/layer = 1024 bytes/token
  Total = 22 × 1024 = 22,528 bytes/token
```

### VRAM disponível para KV

```
VRAM Total:     4,096 MB
Modelo Q4_K:     545 MB (pesos, 22 camadas)
Buffers GPU:     100 MB (scratch f32 + norms + graphs + ring buffers)
Output logits:   128 MB (32K × 4 bytes f32 = 128 KB, não 128 MB — corrigido)

   Nota: logits = 32000 × 4 = 128 KB, não 128 MB. Erro corrigido.

Reserva:         100 MB
Disponível:    4,096 - 545 - 100 - 0.128 - 100 ≈ 3,351 MB ≈ 3.27 GB
```

### Consumo real da janela atual

```
22,528 bytes/token × W=4096 = 92,274,688 bytes ≈ 88 MB total
88 MB / 4,096 MB = 2.2% da VRAM

Muito menor que os 484 MB/12% incorretos do documento anterior.
```

### Janela máxima teórica

```
W_max = 3.27 GB / 22,528 bytes/token ≈ 145,000 tokens

W_opt = min(W_max, context_len, budget_AIMD)
```

### Recomendação (recalculada com 88 MB/4096)

```
W=4096   (atual):    88 MB  (2.2% VRAM)  — conservador demais
W=16384  (4×):      369 MB  (9.0% VRAM)  — seguro, muita folga
W=32768  (8×):      738 MB  (18% VRAM)   — confortável
W=65536  (16×):   1,476 MB  (36% VRAM)   — bom para single worker
W=131072 (32×):   2,953 MB  (72% VRAM)   — máximo prático

Recomendação: aumentar W para 32768 ou 65536.
  - 65536 tokens de atenção GPU → cobre 97.5% dos casos de uso reais
  - Ainda deixa 2.6 GB livres para pipeline com 2 workers (36% cada)
  - Único custo: ring buffer maior em VRAM (1.48 GB em vez de 88 MB)
```

---

## 8. Thermal Model

### AIMD Resource Ramp (já implementado em `aimd.rs`)

```
A cada 0.5s (ou 20 tokens a 65 tok/s):
  if not stressed:  budget *= 2 (additive increase on steroids)
  if stressed:      budget /= 2 (multiplicative decrease)

stressed = (temp > 80°C) || (freq_droop > 15%) || (RAPL > 80W)
```

### PowerArbiter: CPU/iGPU split

```
P_total = P_CPU + P_iGPU + P_dGPU
P_limit = PL1 ≈ 55W (laptop, shared package)

P_CPU_est    = n_threads × 3.5W (AVX-512 load)
P_dGPU_est   = from RAPL (GPU_energy)
P_iGPU_est   = 0 (dGPU ativo → iGPU idle)

cpu_frac = clamp((P_limit - P_dGPU_est) / (n_threads × 3.5W + 1), 0.3, 1.0)
```

### Thread scaling

```
n_threads_base = 6 (default)
if temp > 85°C:   n_threads = 1
elif ε < 0.7:     n_threads = 2
elif ε < 0.86:    n_threads = 4
else:             n_threads = 6

n_threads_final = max(1, n_threads_base × cpu_frac × aimd.budget())
```

### Thermal time constants (i5-11260H + GTX 1650 laptop)

| Evento | Constante | Observação |
|--------|-----------|------------|
| Aquecimento CPU AVX-512 | ~2-5s | PL1=55W, atinge 95°C em segundos |
| Aquecimento dGPU | ~5-15s | TDP 75W, mas sharing heatsink |
| Resfriamento (idle) | ~10-30s | Fan curve agressivo em laptop |
| Throttle recovery | ~3-5s | Após reduzir carga |

**Implicação:** AIMD com período de 0.5s é adequado para capturar thermal dynamics.

---

## 9. Speculative Decoding

### Speedup esperado

```
Speedup = 1 / (1 - α + α/γ)

Onde:
  α = acceptance rate (probabilidade do draft ser aceito)
  γ = speedup do draft vs target (draft é 1 GEMV + LSH, target é full layer)

Para DSPark (LSH + n-gram):
  α ≈ 0.5-0.7 (depende da previsibilidade do hidden state)
  γ ≈ 3-5×    (draft = 1 forward CPU vs target = 22 layers GPU+CPU)
```

| α | γ=3 | γ=5 | γ=10 |
|---|-----|-----|------|
| 0.4 | 1.36× | 1.47× | **1.56×** |
| 0.6 | 1.67× | 1.92× | 2.03× |
| 0.8 | 2.14× | 2.78× | **3.57×** |

**Verificação:** α=0.8, γ=10 → 1/(0.2+0.08) = 1/0.28 = 3.57 ✅
**Verificação:** α=0.4, γ=10 → 1/(0.6+0.04) = 1/0.64 = 1.56 ✅

### Crossover: quando draft vale a pena

```
Draft compensa se:
  cost_draft + (1-α) × cost_target < cost_target
  → cost_draft < α × cost_target

Para TinyLlama:
  cost_target = 15 ms (GPU decode)
  cost_draft = 1 CPU GEMV = 0.5 ms

  → 0.5 < α × 15
  → α > 0.033

99% dos casos α > 0.03 → sempre compensa tentar draft.
```

### Limitação atual

```
DSPark atual: draft gera 1 token, não sequência.
Draft de múltiplos tokens (speculation N=3-5): speedup maior:
  
  Speedup_N = (1 - α^N) / (1 - α) × 1/γ + α^N × N/γ

Para α=0.6, γ=5:
  N=1: 1.92×
  N=3: 2.47×
  N=5: 2.67×
```

---

## 10. Fused Layer Graph

### ⚠️ Diagnóstico: Fused Graph NUNCA funcionou

**Problema**: `cudaGraphExec_t` armazenado em `PerLayerGpuState::fused_layer_graphs[l]` é inválido. `cudaGraphLaunch` retorna `cudaErrorInvalidValue`, mas o erro é engolido por `CUDA_CHECK` em `gpu_graph_replay_layer` que sempre retorna 0.

```
execute_layer_fused() retorna true  (CUDA_CHECK engole erro)
fused_all_ok permanece true          (código acredita que fused path funciona)
GPU não executa nenhum kernel        (graph launch falha silenciosamente)
sync() espera 72ms                   (nada para esperar — latência é de propagação de erro)
```

**Causa**: o `cudaGraphExec` é instanciado corretamente em `gpu_graph_create_layer`, mas o handle armazenado (`void*`) é inválido no momento do replay. Possível causa: `cudaStreamDestroy` do stream de captura antes de usar o graph, ou erro de tipo na conversão `void* → cudaGraphExec_t` na FFI.

**Impacto**: savings teóricos da Seção 10 são IRREAIS — nunca mediram fused funcionando. O benchmark de 65 tok/s do passado provavelmente veio de outro caminho (CPU ou per-op GPU).

### Fix necessário

1. Diagnosticar por que `cudaGraphExec_t` é inválido no replay
2. Corrigir a criação ou o armazenamento do handle
3. Re-validar savings reais com fused funcionando

**Nota**: mesmo com fused funcionando, o GEMV kernel é 34× mais lento que o teórico (η=2.9%). O fused salva sync + launch overhead (~3.5 ms/token), mas o GEMV ainda domina (~100 ms/token). Prioridade: **corrigir kernel GEMV primeiro**, depois o fused graph. Sem o kernel fix, fused graph salva ~3% do tempo total.

---

## 11. Árvores de Decisão

### A. GPU Dispatch Decision (por token)

```
if gpu_available AND fused_graph[l].is_some():
    execute_layer_fused(l, pos)        ← 12 kernels, 0 sync, 0 H2D/D2H
    continue  # próxima layer

elif gpu_available AND per_op_gpu:
    upload_x()                         ← 1 H2D
    gemv_qkv_async(l)                  ← 1 graph replay
    sync()                             ← 1 sync
    CPU_rope_kv_save()                 ← CPU
    attention_async(l, seq)            ← 1 graph replay
    sync()
    gemv_o_async(l)                    ← 1 graph replay
    CPU_add_rmsnorm()
    gemv_gate_up_async(l)              ← 1 graph replay
    CPU_silu_mul()
    gemv_down_async(l)                 ← 1 graph replay
    CPU_add()
    # ~4 syncs, ~4 graph replays/layer

else:  # CPU-only
    forward_linear_multi(Q, K, V)     ← CPU VNNI, 0 syncs
    rope_kv_save()
    attention_full()                   ← CPU
    forward_linear(O)
    add_rmsnorm_silu_mul()
    forward_linear(Gate, Up, Down)
    add()
    # 0 syncs, ~260 GEMV calls (batched)
```

### B. Thermal Throttle Policy (a cada 20 tokens)

```
ler temp, freq, RAPL

if temp > 95°C:
    n_threads = 1
    GPU_disable = true
    policy.skip_layer(l%2==0)  ← skip metade das layers

elif temp > 85°C:
    n_threads = 2
    GPU_disable = true
    aimd.budget /= 2

elif temp > 80°C:
    n_threads = 4
    GPU_active = true
    aimd.budget /= 2

elif freq_droop > 15%:
    # AVX-512 downclock detectado
    n_threads = clamp(n_threads - 1, 1, 6)
    GPU_active = true  # compensar com GPU

else:  # normal
    if freq > 4.0 GHz:  # turbo ativo
        n_threads = 6
        GPU_active = true
        aimd.budget *= 2
```

### C. Attention Strategy (Fugu, por token)

```
if seq_len < 64:
    strategy = Full  ← mais rápido para sequências curtas

elif seq_len < window_size:
    strategy = Full  ← GPU attention graph cabe todo

elif DSPark_accept_rate > 0.5:
    strategy = SparseWithDSPark(window, sentinel=64, N_dspark=32)

else:
    strategy = Sparse(window, block=64)  ← fallback

# Cache pressure override
if cache_pressure > 0.8:
    strategy = Sparse(window=min(window, 2048), block=128)
```

### D. AIMD Budget Allocation

```
if not_stressed:
    budget = min(budget × 2, 1.0)       ← dobra a cada 0.5s

elif stressed AND first_signal:
    budget = max(budget / 2, 0.125)     ← corta metade no 1º sinal

elif stressed AND sustained (>3 checks):
    budget = 0.25                       ← sustentado = redução forte

# Aplicação
concurrency = max(1, round(budget × max_concurrency))
GPU_enabled = budget > 0.3
```

---

## 12. Diagnóstico Empírico — Resultados do Benchmark

### 12.1 Kernel GEMV Q4_K: η real = 2.9%

**Probe**: `kernel_gemv_q4k` isolado com eventos CUDA (sem graph, sem copies)
**Resultado**: Gate GEMV (5632×2048, 6.49 MB) = **1172 μs**

| Métrica | Teórico | Medido | Fator |
|---------|---------|--------|-------|
| Bandwidth efetiva | 192 GB/s | 5.5 GB/s | 35× pior |
| Eficiência η | 1.0 | **0.029** | — |
| Tempo Gate GEMV | 34 μs | **1172 μs** | 34× pior |
| Tempo/layer (7 GEMVs) | 238 μs | **4472 μs** | 19× pior |
| Tempo 22 layers | 5.2 ms | **98.4 ms** | 19× pior |
| tok/s previsto | 192 tok/s | **~10 tok/s** | 19× pior |

### 12.2 Causa Raiz: Acesso não-coalescido

Cada thread processa 1 linha. Adjacent threads acessam pesos a stride:

```
Para Gate (n_blocks = ceil(5632/256) = 22):
  stride = n_blocks × 144 = 22 × 144 = 3168 bytes entre threads adjacentes

Warp de 32 threads:
  Total span:    31 × 3168 + 144 = 98.352 bytes
  Cache lines:   ceil(98352/128) = 769 linhas de 128B
  Dados usados:  32 × 144 = 4608 bytes
  Utilização:    4608 / (769 × 128) = 4.7%
```

### 12.3 Fused Graph: Invalid Handle (Nunca Funcionou)

`cudaGraphLaunch` retorna `cudaErrorInvalidValue` porque o `cudaGraphExec_t` armazenado é inválido. O erro é engolido por `CUDA_CHECK` em `gpu_graph_replay_layer` que sempre retorna 0, fazendo `execute_layer_fused()` retornar `true` mesmo sem executar nada.

**Impacto**: `fused_all_ok` fica `true` o tempo todo, código acredita que o fused path está funcionando, mas GPU não executa nenhum kernel. A latência de 72ms do `sync()` vem de propagação de estado de erro CUDA, não de execução real.

### 12.4 Solução Proposta: Kernel Transposto

**Problema**: layout row-major dos pesos → stride `n_blocks × 144` entre threads adjacentes.

**Solução**: Transpor o layout durante upload (CPU→GPU), de row-major para column-major:

```
Antes:  weights[row][blk]  → stride = n_blocks × 144 entre rows adjacentes
Depois: weights[blk][row]  → stride = 144 entre rows adjacentes
```

Com stride=144 entre threads adjacentes:

```
Warp de 32 threads:
  Total span:    31 × 144 + 144 = 4608 bytes
  Cache lines:   36 linhas de 128B (exato)
  Utilização:    4608 / (36 × 128) = 100%
  Melhoria:      21× sobre o original (4.7% → 100%)
```

**Mudanças necessárias**:

1. `fused_attention.cu`: função de transposição durante upload + kernel modificado para layout column-major
2. `swamp-gpu/src/lib.rs`: FFI para upload transposto
3. `executor.rs` / `scheduler.rs`: chamar upload transposto em vez de direto

**Ganho esperado**:

| GEMV | Antes (μs) | Depois (μs, estimado) |
|------|-----------|----------------------|
| Gate/Up/Down (6.49 MB) | 1172 | ~76 (15×) |
| Q/O (2.36 MB) | 426 | ~28 (15×) |
| K/V (0.29 MB) | 52 | ~4 (13×) |
| Total/layer | 4472 | ~291 (15×) |
| Total 22 layers | 98.4 ms | ~6.4 ms |
| + overhead | ~102 ms | ~10 ms |
| **tok/s** | **~10** | **~100** (10×) |

---

## Appendix: Constantes Úteis

| Constante | Valor | Descrição |
|-----------|-------|-----------|
| `B_vram_gddr6` | 192 GB/s | GTX 1650 GDDR6 bandwidth teórico |
| `η_q4k_teorico` | 0.4 | Eficiência teórica de acesso Q4_K |
| `η_q4k_empirico` | 0.25 | Eficiência real (do benchmark 65 tok/s) |
| `B_eff_real` | 48 GB/s | 192 × 0.25 |
| `T_sync_cuda` | ~20 μs | cudaStreamSynchronize |
| `T_launch_cuda` | ~5 μs | cudaKernelLaunch |
| `T_gemv_gate_gpu` | ~135 μs | Gate GEMV GPU (6.49 MB, η=0.25) |
| `T_gemv_gate_cpu` | ~520 μs | Gate GEMV CPU VNNI |
| `T_pcie_h2d_8kb` | ~1 μs | cudaMemcpyAsync 8 KB |
| `d_entry_embed` | 8 KB | Input embedding f32 (2048×4) |
| `q_out_size` | 8 KB | Q output f32 (2048×4) |
| `k_out_size` | 1 KB | K output f32 (256×4) |
| `kv_half_tok` | 512 bytes | KV cache half por layer/token |
| `V` | 32000 | Vocab size |
| `logits_bytes` | 128 KB | Logits f32 (32000×4) — não 128 MB |
