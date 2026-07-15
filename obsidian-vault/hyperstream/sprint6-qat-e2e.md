# Sprint 6 — QAT + End-to-End 7B

**Status**: 🔴 POR FAZER

## O Que Existe

### QAT (`swamp-engine/src/qat.rs`)

- Estrutura básica de calibration tool
- Não integrada ao converter ou ao streaming pipeline

## O Que Falta

### 1. QAT Per-Block Pipeline

Integrar quantização QAT durante a conversão de shards:

```
1. Load modelo GGUF (já quantizado Q4_K ou FP32)
2. Para cada layer:
   a. Rodar calibration com dados de calibração (amostras do dataset)
   b. Aplicar QAT per-block: ajustar scales/mins para minimizar erro
   c. Escrever blocos Q4_K ajustados no shard.bin
3. index.json inclui metadados de calibração (opcional)
```

```rust
pub fn convert_with_qat(
    gguf_path: &Path,
    output_dir: &Path,
    calibration_data: &[&[u32]],  // tokens de calibração
    qat_config: &QatConfig,
) -> Result<()> {
    // ...
}
```

### 2. End-to-End com TinyLlama 1.1B Q4_K

Pipeline completo de teste:

```bash
# 1. Converter modelo para shard
swamp-convert-shard --gguf tinyllama-1.1b-q4_k.gguf --output-dir ./shards/

# 2. Rodar benchmark de streaming
swamp-benchmark-streaming \
  ./shards/index.json ./shards/shard.bin \
  --layer-id 0 --tensor-slot 0 --repeats 100

# 3. Rodar inferência completa com streaming
SWAMP_STREAMING=1 \
cargo run --release --features gpu -p swamp-tools --bin swamp-benchmark-prefill -- \
  tinyllama-1.1b-q4_k.gguf tokenizer.json \
  --prompt-tokens 128 --repeats 3 --n-threads 6
```

### 3. Métricas-Alvo

| Métrica | Alvo (MVP) | Notas |
|---------|-----------|-------|
| Tokens/s | >= 10 tok/s | TinyLlama 1.1B, GPU + streaming |
| Hit rate avg | > 80% | Prefetch accuracy |
| Fallback ratio | < 20% | % de layers executadas no CPU |
| Bandwidth eficaz | > 2 GB/s | NVMe → VRAM sustained |
| Latência p95 | < 50ms | Por layer, incluindo fallback |
| Degradação graceful | ≤ 2x slowdown | Comparar com carregamento full GPU |

### 4. Cenários de Teste

| Cenário | Configuração |
|---------|-------------|
| Ideal | K=6, NVMe NVMe 3GB/s, preditor perfeito |
| Misto | K=3, NVMe 1GB/s, preditor 80% |
| Fallback-heavy | K=1, SATA SSD 200MB/s, preditor 50% |
| CPU-only | Sem GPU (fallback total) como baseline |

### 5. Documentação Dev

- README atualizado com instruções de streaming
- Runbook ([[runbook.md]]) com comandos de teste
- Guia de troubleshooting para fallback excessivo

## Acceptance Checklist

- [ ] Converter tool com flag `--qat` produz shards calibrados
- [ ] Inferência completa com streaming roda sem crash
- [ ] Logs: tokens/s, hit_rate, fallback_count, bandwidth
- [ ] Resultados numéricos batem com baseline (CPU-only ou full-GPU)
- [ ] Documentação dev completa
- [ ] Testes nos 3 cenários (ideal, misto, fallback)

## Dependências

- [[sprint2-shard-io.md]] — Converter tool
- [[sprint3-dspark-scheduling.md]] — Prefetch scheduling
- [[sprint4-fallback-cpu.md]] — Fallback path
- [[sprint5-aimd-governor.md]] — AIMD governor

## Próximos Passos (R1)

Após Sprint 6, o MVP está completo. Para R1, ver [[tasks-backlog.md]] sprints 7-11:
- DirectStorage (Windows)
- CUDA optimization
- Large-model tooling (1T shards)
- Performance tuning
