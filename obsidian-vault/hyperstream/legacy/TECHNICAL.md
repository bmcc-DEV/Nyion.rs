# Nyion Engine — Technical Reference

## 1. Sync Analysis (Gargalo #1)

Cada layer de decode executava **5 syncs CUDA** antes da Fused Layer Graph:

```
QKV GEMV ─sync①→ RoPE+KV save ─sync②→ Attention ─sync③→ O GEMV ─sync④→
add+RMSNorm → GateUp GEMVs ─sync⑤→ SiLU+Mul → Down GEMV
```

22 layers × 5 syncs = **110 syncs/passo**. Syncs existiam porque ops CPU (RoPE, RMSNorm, SiLU, add) precisavam dos outputs GPU antes de prosseguir.

### Status atual: todos os kernels movidos para CUDA

| Op atual | Local | CUDA kernel | Status |
|----------|-------|-------------|--------|
| RoPE (q,k) | Rust `ops.rs` | `kernel_rope` | ✅ |
| KV save ring buffer (half) | Rust | `kernel_kv_save` | ✅ |
| add_in_place (x += wo, x += down) | Rust `add_in_place` | `kernel_add` | ✅ |
| RMSNorm (attn + ffn) | Rust `rmsnorm` | `kernel_rmsnorm` | ✅ |
| SiLU + Mul | Rust | `kernel_silu_mul` | ✅ |

### Resultado: 5 syncs → 1 sync total (todas as layers)

```
upload_x ─→ [Layer 0 fused graph] ─→ [Layer 1 fused graph] ─→ ... ─→ [Layer N fused graph] ─→ sync + download_x
             └── 12 CUDA kernels capturados em 1 cudaGraphExec ──┘
```

No caso all-fused: **1 sync total** (antes 110). Fallback CPU por layer preserva a correção.

---

## 2. Fused Layer Graph

### Fluxo capturado (12 kernels, 1 CUDA Graph)

```cuda
// ① RMSNorm: x_norm = rmsnorm(x, attn_norm)
// ② Q GEMV: x_norm → q_out
// ③ K GEMV: x_norm → k_out
// ④ V GEMV: x_norm → v_out
// ⑤ RoPE: q_out, k_out in-place (pos = *d_seq_len - 1)
// ⑥ KV save (half): k_out/v_out → ring buffer (wpos = *d_wpos)
// ⑦ Attention: q_out, k_buf, v_buf → attn_out (seq_len = *d_seq_len)
// ⑧ O GEMV: attn_out → o_out
// ⑨ x += o_out (residual add)
// ⑩ RMSNorm: x_norm = rmsnorm(x, ffn_norm)
// ⑪ Gate GEMV: x_norm → gate_out
// ⑫ Up GEMV: x_norm → up_out
// ⑬ SiLU+Mul: gate_out *= silu(gate_out) * up_out (in-place)
// ⑭ Down GEMV: gate_out → down_out
// ⑮ x += down_out (residual add)
```

seq_len e wpos são lidos de device pointers (`d_wpos`, `d_seq_len`) — atualizados via `cudaMemcpyAsync` antes de cada replay.

### GPU-side state

| Buffer | Tamanho | Descrição |
|--------|---------|-----------|
| `d_attn_norms[l]` | embed_dim f32 | Pesos RMSNorm attn por layer (device) |
| `d_ffn_norms[l]` | embed_dim f32 | Pesos RMSNorm ffn por layer (device) |
| `d_k_buf` | window×n_kv_heads×head_dim half | KV cache K ring buffer |
| `d_v_buf` | window×n_kv_heads×head_dim half | KV cache V ring buffer |
| `d_x` | embed_dim f32 | Hidden state (persistente entre layers) |
| `d_x_norm` | embed_dim f32 | RMSNorm output (reused pre-attn e pre-ffn) |
| `d_q_out` | n_heads×head_dim f32 | Output Q GEMV |
| `d_k_out` | n_kv_heads×head_dim f32 | Output K GEMV |
| `d_v_out` | n_kv_heads×head_dim f32 | Output V GEMV |
| `d_attn_out` | embed_dim f32 | Output attention |
| `d_o_out` | embed_dim f32 | Output O GEMV |
| `d_gate_out` | ffn_dim f32 | Output Gate GEMV (reused por SiLU+Mul+Down) |
| `d_up_out` | ffn_dim f32 | Output Up GEMV |
| `d_down_out` | embed_dim f32 | Output Down GEMV |
| `d_scores` | n_heads×window f32 | Scores atenção (softmax) |
| `d_wpos` | 1 i32 | wpos = pos % window_size (device ptr) |
| `d_seq_len` | 1 i32 | seq_len = pos + 1 (device ptr) |

### Mecanismo de replay

1. `create_layer_graph(l)` — lazy: cria o `cudaGraphExec` no primeiro uso da layer `l`
2. `execute_layer_fused(l, pos)` — atualiza `*d_wpos`, `*d_seq_len` via `cudaMemcpyAsync` no compute stream, depois `cudaGraphLaunch`
3. Decode loop tenta fused primeiro; se falha, cai no per-op GPU → CPU

### Implicações

- **All-fused**: 1 sync + 1 D2H no fim (vs 110 syncs antes)
- **Fallback misto**: primeira layer CPU causa sync + D2H; seguintes rodam síncrono
- **Zero H2D/D2H por layer** no all-fused (exceto upload_x inicial + download_x final)
- **Cache de graph**: índice por `layer_idx`, criado lazy

---

## 3. KV Cache Híbrido

### Estrutura atual

- `d_k_buf[window_size][n_kv_heads][head_dim]` (half) — ring buffer em VRAM
- `d_v_buf[window_size][n_kv_heads][head_dim]` (half) — ring buffer em VRAM
- `PagedKVCache` em RAM para tokens fora da window (fallback `ensure_pages_hot`)

### Como o fallback funciona hoje

1. Token `pos` está na window? (`pos % window_size`): GPU attention usa só VRAM.
2. Token `pos` fora da window? Atenção CPU (`attention_sparse` ou `attention_full`) lê da `PagedKVCache` em RAM.

### Problema

A atenção GPU (CUDA attention kernel) só enxerga a window em VRAM. Tokens antigos (>window_size atrás) não participam da atenção GPU.

### Solução futura: spill-to-host com prefetch

1. KV save em kernel CUDA: além de escrever no ring buffer VRAM, faz `cudaMemcpyAsync` do token mais antigo sendo sobrescrito para um buffer host (spill).
2. Atenção híbrida GPU: kernel de atenção lê da VRAM para tokens na window, e faz `cudaMemcpyAsync` de buffers host (prefetch) para tokens spillados.
3. Não implementado ainda — a atenção CPU (fallback `attention_full`) já cobre o caso completo.

---

## 4. Teste de Perplexity (Planejado)

```bash
cargo test --release --features gpu --test tinyllama_perplexity
```

- Usa subset WikiText-2 (~100 sentences)
- Compara perplexity GPU vs CPU (devem ser iguais, diferença <0.01)
- Falha se perplexity divergir OU se throughput < threshold

---

## 5. Roadmap

### Fase 1 (completa): Fused Layer Graph — 5 syncs → 1 sync total
- [x] Análise de syncs
- [x] `kernel_rope` em fused_attention.cu
- [x] `kernel_kv_save` (half) em fused_attention.cu
- [x] `kernel_add` em fused_attention.cu
- [x] `kernel_rmsnorm` (shared mem) em fused_attention.cu
- [x] `kernel_silu_mul` em fused_attention.cu
- [x] `kernel_scores_dptr` + `kernel_softmax_v_dptr` (seq_len device ptr)
- [x] Upload normas GPU (d_attn_norms, d_ffn_norms)
- [x] `gpu_graph_create_layer` — captura layer inteiro
- [x] `gpu_graph_replay_layer` / `gpu_graph_destroy_layer`
- [x] Rust FFI: LayerGraph struct + 3 funções
- [x] PerLayerGpuState: fused_layer_graphs, buffers, upload_x/download_x
- [x] Decode loop dispatch: fused → per-op GPU → CPU

### Fase 2: Atenção esparsa + Pipeline concurrente
- [ ] Atenção híbrida GPU (VRAM window + host spill)
- [ ] Pipeline concurrente com NyionVM dispatcher real

### Fase 3: Governor + NyionVM + Speculative
- [ ] ResourceGovernor cycle rodando
- [ ] NyionVM dispatcher thread
- [ ] execute_cpu/gpu conectados ao layer loop
- [ ] DSPark speculative decode

---

## 6. CUDA Graph Layer — API FFI (Rust side)

```rust
// swamp-gpu/src/lib.rs

pub struct LayerGraph(*mut std::ffi::c_void);

pub fn gpu_graph_create_layer(
    // Weight pointers (device, per-layer)
    d_w_q: *const u8, d_w_k: *const u8, d_w_v: *const u8,
    d_w_o: *const u8, d_w_gate: *const u8, d_w_up: *const u8, d_w_down: *const u8,
    // Norm weights (device, per-layer)
    d_attn_norm: *const f32, d_ffn_norm: *const f32,
    // KV ring buffer (device, half)
    d_k_buf: *mut c_void, d_v_buf: *mut c_void,
    // State buffers (device)
    d_x: *mut f32, d_x_norm: *mut f32,
    d_q_out: *mut f32, d_k_out: *mut f32, d_v_out: *mut f32,
    d_attn_out: *mut f32, d_o_out: *mut f32,
    d_gate_out: *mut f32, d_up_out: *mut f32, d_down_out: *mut f32,
    d_scores: *mut f32,
    // Variable param pointers (device, updated per replay)
    d_wpos: *mut i32, d_seq_len: *mut i32,
    // Config (fixed at creation)
    n_heads: usize, n_kv_heads: usize, head_dim: usize, window_size: usize,
    embed_dim: usize, ffn_dim: usize,
    n_rows_q, n_rows_k, n_rows_v, n_rows_o, n_rows_gate_up, n_rows_down: usize,
    n_blocks_attn: usize, n_blocks_down: usize,
    rms_eps: f32,
) -> Result<LayerGraph>;

pub fn gpu_graph_replay_layer(
    graph: &LayerGraph,
    stream: CudaStream,
    pos: usize,
    window_size: usize,
) -> Result<()>;

pub fn gpu_graph_destroy_layer(graph: LayerGraph) -> Result<()>;
```

### Motivação da API

- **Criação (1x por layer)**: todos os ponteiros de peso, norma e buffers scratch são fixados no `cudaGraphExec`. Isso inclui os 7 pesos Q4_K, 2 normas f32, 13 buffers scratch, e 2 device pointers para parâmetros variáveis.
- **Replay (N× por step)**: só precisa do `pos` (calcula wpos/seq_len internamente). `cudaMemcpyAsync` atualiza `*d_wpos` e `*d_seq_len` no device antes do `cudaGraphLaunch`.
- **sem H2D/D2H no replay**: todos os buffers intermediários estão no device. O host só vê `upload_x` antes do layer loop e `download_x` depois.
- **seq_len variável**: kernels `kernel_scores_dptr` e `kernel_softmax_v_dptr` leem `*d_seq_len` para saber quantos tokens atender. O grid lança `window_size` blocos e cada bloco faz bounds-check.

---

## 7. Constantes para GTX 1650

| Parâmetro | Valor |
|-----------|-------|
| VRAM | 4 GB |
| SM count | 14 |
| Warp size | 32 |
| Max threads/block | 1024 |
| Shared mem/block | 48 KB |
| Compute capability | 7.5 (Turing) |
| PCIe gen | 3.0 ×16 (~16 GB/s) |
| PL1 (CPU package) | ~55W (térmico compartilhado) |
