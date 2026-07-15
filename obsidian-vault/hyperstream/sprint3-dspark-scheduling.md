# Sprint 3 — DSPark + Scheduling

**Status**: ✅ COMPLETO
- ✅ DSPark predictor (top-1 greedy, LSH-based) — `swamp-engine/src/dspark.rs`
- ✅ Camada de policy LuaJIT — `swamp-engine/src/policy.rs`
- ✅ **LayerSchedule + resolve_k** — `swamp-engine/src/streamer.rs`
- ✅ **prefetch_with_dspark + prefetch_k_ahead** — `swamp-engine/src/streamer.rs`

## O Que Já Existe

### DSPark Engine (`dspark.rs`)

```rust
pub struct DSparkEngine {
    pub draft_model: DraftModel,
    pub lsh: LSHIndex<u64>,
    pub cold_page_predictor: ColdPagePredictor,
}
```

Principais capacidades:
- `draft_model.draft(&hidden_state)` → `(Vec<usize>, Vec<f32>)` — tokens preditos + confianças
- `find_attention_candidates(&x, n)` → `Vec<(usize, f32)>` — posições relevantes para atenção esparsa
- `predict_cold_pages(&x, pos, n, ...)` → `Vec<usize>` — páginas KV que serão acessadas
- `observe_at(&x, token, pos)` — alimenta feedback loop
- `hash_hidden(&x)` → LSH hash para MoE expert prefetch

Já usado no executor:
- Decodificação especulativa (linha ~854 em `executor.rs`)
- DSPark-guided cold page prefetch (linha ~900)
- Expert row prefetch via AIMD budget (linha ~878)

## Implementado

### 1. LayerSchedule + resolve_k — `swamp-engine/src/streamer.rs`

```rust
pub struct LayerSchedule {
    pub token_id: usize,
    pub transfers: Vec<LayerTransfer>,
}

pub struct LayerTransfer {
    pub layer_id: usize,
    pub tensor_name: String,
}

impl HyperStreamEngine {
    pub fn resolve_k(&self, tokens: &[usize], k: usize) -> Vec<LayerSchedule>;
}
```

### 2. prefetch_with_dspark — `swamp-engine/src/streamer.rs`

Usa DSPark draft predictions para guiar prefetch. Tokens com alta confiança (>0.7) prefetch todos os pesos; confiança média (>0.3) prefetch só Q/K/V.

```rust
impl HyperStreamEngine {
    pub fn prefetch_with_dspark(
        &mut self,
        hidden_state: &[f32],
        pos: usize,
        dspark: &DSparkEngine,
        current_layer: usize,
    );
}
```

### 3. Hipóteses de Prefetch

| Cenário | Ação |
|---------|------|
| Hit: buffer já DeviceReady | `wait_for_layer` retorna GPU → executa shader |
| Timeout: buffer não ficou pronto | retorna CPU → leitura direta do mmap |
| Erro de predição | cai pra CPU via fallback automático |

### 4. K Adaptativo

`AimdRamp::scaled_k(base_k)` controla K via AIMD. Ver [[sprint5-aimd-governor.md]].

## Acceptance Checklist

- [ ] LayerSchedule produz offsets válidos do ShardIndex
- [ ] DSPark predictions alimentam prefetch queue
- [ ] Logs mostram prefetch hits/misses por token
- [ ] Telemetria registra hit_rate, bandwidth, transfers_completed
- [ ] K adaptativo responde a mudanças de bandwidth (Sprint 5)

## Dependências

- [[sprint2-shard-io.md]] — ShardReader para resolver offsets
- [[architecture.md]] — Fluxo de dados completo

## Próximos Passos

1. Implementar `LayerSchedule` + resolver
2. Integrar DSPark predictions → prefetch queue
3. Conectar telemetria
4. [[sprint4-fallback-cpu.md]] — Tratar misses com fallback CPU
5. [[sprint5-aimd-governor.md]] — AIMD + K adaptativo
