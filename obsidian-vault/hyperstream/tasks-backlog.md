# Tasks Backlog — Nyion HyperStream

Formato para importar no GitHub Issues. Labels: `sprint-X`, `linguagem`, `componente`.

---

## Sprint 0 — Preparação (semana 1, ~5 dias)

### T1: Setup repositório e CI

- **Estimativa**: 1d (8h)
- **Labels**: `sprint-0`, `infra`, `rust`
- **Descrição**: Monorepo com workspace Cargo, templates de issues, CI básico (tests, clang-format). CI deve compilar `swamp-engine` e `swamp-tools`.

**Critério de Aceitação**:
- [ ] `cargo check` passa em toda workspace
- [ ] `cargo test` roda todos os unit tests
- [ ] `cargo fmt --check` não aponta diferenças
- [ ] CI verde no PR inicial

---

### T2: Ambiente de teste e hardware matrix

- **Estimativa**: 2d (16h)
- **Labels**: `sprint-0`, `infra`, `hardware`
- **Descrição**: Listar e obter 2 plataformas de teste. Mínimo: Linux laptop (iGPU) + Linux desktop (dGPU NVIDIA). Opcional: Windows. Verificar:
  - Vulkan 1.2 + timeline semaphores suportados
  - Transfer queue family disponível
  - NVMe SSD

**Critério de Aceitação**:
- [ ] 2 plataformas com Vulkan funcionando
- [ ] `vulkaninfo` mostra suporte a `VK_KHR_timeline_semaphore`
- [ ] Acesso remoto ou VM configurado
- [ ] Hardware matrix documentada em [[sprint0-research.md]]

---

### T3: Pesquisa dependências e libs

- **Estimativa**: 2d (16h)
- **Labels**: `sprint-0`, `research`, `rust`
- **Descrição**: Decidir bindings Vulkan (ash vs Vulkan-Hpp), I/O (mmap vs io_uring vs DirectStorage), linguagem (Rust + C++/Mojo). Atualmente o projeto usa `ash` + `memmap2` + Rust. Documentar motivos.

**Critério de Aceitação**:
- [ ] Documento [[sprint0-research.md]] com escolhas e justificativas
- [ ] Decisão final sobre DirectStorage path (adiar para R1)
- [ ] Decisão sobre Mojo integration (adiar)

---

### T4: Especificação formato de shards

- **Estimativa**: 2d (16h)
- **Labels**: `sprint-0`, `spec`, `rust`
- **Descrição**: Especificar `index.json` + `shard.bin` layout, bloco Q4_K default.

**Critério de Aceitação**:
- [ ] [[shard-format-spec.md]] versionada no repo (`obsidian-vault/hyperstream/`)
- [ ] JSON schema para index.json
- [ ] Estrutura de bloco Q4_K documentada (256 elem = 168 bytes)
- [ ] Exemplo de index.json para TinyLlama 1.1B

---

## Sprint 1 — Vulkan Transfer + Triple Buffer (semanas 2-3)

**Status**: ✅ Completo (código já existe)

### T1: Vulkan bootstrap + transfer queue

- **Estimativa**: 3d (24h)
- **Labels**: `sprint-1`, `vulkan`, `rust`
- **Arquivos**: `swamp-gpu/src/vulkan.rs`
- **Status**: ✅ `VkBackend` implementado — instance, device, queue families, memory allocation

### T2: Timeline semaphore plumbing

- **Estimativa**: 3d (24h)
- **Labels**: `sprint-1`, `vulkan`, `rust`
- **Arquivos**: `swamp-gpu/src/streaming/transfer.rs`
- **Status**: ✅ Timeline semaphore no `TransferEngine` — sinal/espera entre transfer e compute

### T3: Triple-buffer staging implementation

- **Estimativa**: 4d (32h)
- **Labels**: `sprint-1`, `vulkan`, `rust`
- **Arquivos**: `swamp-gpu/src/streaming/triple_buffer.rs`
- **Status**: ✅ 3 slots DEVICE_LOCAL, state machine Free→Staging→DeviceReady→Computing

### T4: Dummy compute consumer

- **Estimativa**: 3d (24h)
- **Labels**: `sprint-1`, `vulkan`, `rust`
- **Arquivo**: `swamp-tools/src/bin/benchmark_streaming.rs`
- **Status**: ✅ Benchmark isolado que testa upload→wait→release

---

## Sprint 2 — Shard IO + mmap + Converter (semanas 4-5)

### T1: Shard converter tool

- **Estimativa**: 3d (24h)
- **Labels**: `sprint-2`, `rust`, `tools`
- **Arquivo**: `swamp-tools/src/bin/convert_shard.rs`
- **Status**: ✅ Implementado
- **Descrição**: Ferramenta CLI que lê GGUF e produz `shard.bin` + `index.json`. Suporta Q4_K blocks e f32 norms. Opção `--qat` para calibration.

**API**:
```
swamp-convert-shard --gguf model.gguf --output-dir ./shards/ [--qat]
```

**Critério de Aceitação**:
- [ ] Converte TinyLlama 1.1B Q4_K → shard.bin + index.json
- [ ] shard.bin pode ser lido de volta com `ShardReader::open()`
- [ ] bytes lidos == bytes escritos (verificação CRC32 opcional)
- [ ] `--help` documenta todos os flags
- [ ] Benchmark: conversão em < 30s para 1.1B

---

### T2: mmap file reader + aligned IO

- **Estimativa**: 3d (24h)
- **Labels**: `sprint-2`, `rust`, `io`
- **Arquivo**: `swamp-gpu/src/streaming/shard.rs`
- **Status**: ✅ `ShardReader` com mmap-based zero-copy reads. `O_DIRECT`/io_uring como opção futura.

---

### T3: Integrar reader ao streaming engine

- **Estimativa**: 4d (32h)
- **Labels**: `sprint-2`, `rust`, `streaming`
- **Arquivo**: `swamp-engine/src/streamer.rs`
- **Status**: ✅ Implementado
- **Descrição**: `HyperStreamEngine` que combina `ShardReader` + `TransferEngine`. Expõe API de prefetch por layer.

**Critério de Aceitação**:
- [ ] `HyperStreamEngine::prefetch_layer(layer, tensor)` enfileira upload
- [ ] `wait_for_layer(layer, timeout_ns)` retorna slot ou timeout
- [ ] End-to-end do shard → TransferEngine → timeline → dummy consumer
- [ ] Logs mostram prefetch requests

---

## Sprint 3 — DSPark + Scheduling (semanas 6-7)

### T1: DSPark predictor simples

- **Estimativa**: 2d (16h)
- **Labels**: `sprint-3`, `rust`, `dspark`
- **Arquivo**: `swamp-engine/src/dspark.rs`
- **Status**: ✅ DSPark com LSH, draft model, cold page predictor. API: `draft(&x) → (Vec<usize>, Vec<f32>)`.

---

### T2: Layer schedule resolver

- **Estimativa**: 3d (24h)
- **Labels**: `sprint-3`, `rust`, `streaming`
- **Arquivo**: `swamp-engine/src/streamer.rs`
- **Status**: ✅ Implementado
- **Descrição**: `LayerSchedule` + `resolve_k()` + `LayerTransfer` — mapeia tokens previstos para transfers individuais.

**Critério de Aceitação**:
- [ ] `ShardIndex::resolve_k(tokens, k)` retorna `Vec<LayerSchedule>`
- [ ] Offsets correspondem exatamente aos dados no shard.bin
- [ ] Schedule para TinyLlama 1.1B é completo (todos os tensores por layer)

---

### T3: Integrar DSPark ao streamer

- **Estimativa**: 4d (32h)
- **Labels**: `sprint-3`, `rust`, `streaming`
- **Arquivo**: `swamp-engine/src/streamer.rs`
- **Status**: ✅ Implementado
- **Descrição**: `prefetch_with_dspark()` usa draft predictions do DSPark para guiar prefetch. Tokens com confiança alta (>0.7) prefetch todos pesos; média (>0.3) só Q/K/V.

**Critério de Aceitação**:
- [ ] Prefetch de K tokens ahead visível nos logs
- [ ] Hit rate > 0% (prefetch não é inútil)
- [ ] K configurável via env `SWAMP_STREAMING_K`
- [ ] Não causa stalls no decode loop

---

### T4: Telemetria mínima

- **Estimativa**: 1d (8h)
- **Labels**: `sprint-3`, `rust`, `telemetry`
- **Arquivo**: `swamp-gpu/src/streaming/mod.rs`
- **Status**: ✅ `StreamTelemetry` com contadores atômicos + CSV snapshot.

**Critério de Aceitação**:
- [ ] Contadores conectados no hot path
- [ ] CSV gerado ao final da execução
- [ ] Print periódico (a cada 20 steps) com métricas

---

## Sprint 4 — Fallback CPU + Kernels (semanas 8-9)

### T1: Implementar fallback path via mmap

- **Estimativa**: 3d (24h)
- **Labels**: `sprint-4`, `rust`, `fallback`
- **Arquivo**: `swamp-engine/src/streamer.rs`
- **Status**: ✅ Implementado
- **Descrição**: `wait_for_layer(layer, tensor, timeout_ns)` com `FallbackDecision::Gpu(slot)` ou `Cpu` + `tensor_data` opcional.

**Critério de Aceitação**:
- [ ] `wait_for_layer(layer_id, timeout_ns)` retorna `FallbackDecision::Gpu` ou `Cpu`
- [ ] CPU fallback produz saída numericamente similar à GPU
- [ ] Log de fallback events com latência
- [ ] Sem crashes em cenários de miss total

---

### T2: Integrar kernels CPU

- **Estimativa**: 3d (24h)
- **Labels**: `sprint-4`, `rust`, `kernels`
- **Arquivo**: `swamp-kernels/src/`
- **Status**: ✅ Kernels existem (GEMV Q4_K, Q6_K, softmax, matmul W3). Já integrados ao executor via `forward_linear()`.

---

### T3: Orquestração NyionVM dispatch

- **Estimativa**: 4d (32h)
- **Labels**: `sprint-4`, `rust`, `orchestration`
- **Arquivo**: `swamp-engine/src/executor.rs`
- **Status**: ✅ Implementado
- **Descrição**: Decode loop modificado: `ensure_layer_loaded()` carrega pesos sob demanda; se GEMV GPU falhar, cai para `forward_linear()` CPU (já existia). Prefetch da próxima layer adicionado.

**Critério de Aceitação**:
- [ ] Decode loop per-layer: tenta GPU, timeout → CPU
- [ ] Política configurável (ForceGpu, ForceCpu, Adaptive)
- [ ] Log de dispatch por layer
- [ ] Testado com K=0 (tudo fallback) e K=6 (hit rate ideal)

---

## Sprint 5 — ResourceGovernor AIMD (semanas 10-11)

### T1: Medição contínua

- **Estimativa**: 3d (24h)
- **Labels**: `sprint-5`, `rust`, `telemetry`
- **Arquivo**: `swamp-engine/src/executor.rs` + `swamp-engine/src/streamer.rs`
- **Descrição**: Conectar `StreamTelemetry` ao hot path. Medir bandwidth, latência, hit rate em tempo real.

**Critério de Aceitação**:
- [ ] `StreamTelemetry` populado a cada step
- [ ] hit_rate() e bandwidth_gbps() precisos
- [ ] Acesso lock-free (AtomicU64)

---

### T2: AIMD controller para K

- **Estimativa**: 4d (32h)
- **Labels**: `sprint-5`, `rust`, `control`
- **Arquivo**: `swamp-engine/src/aimd.rs`
- **Status**: ✅ Implementado
- **Descrição**: `AimdRamp::scaled_k(base_k)` — usa `budget.sqrt()` para escalar K suavemente. Pendente: feedback loop hit_rate → AIMD no executor.

**Lógica**:
```
if hit_rate > 0.9 for 5 consecutive steps: K += 1 (AI)
if hit_rate < 0.5: K = max(1, K / 2) (MD)
K = clamp(K, 1, K_max)
```

**Critério de Aceitação**:
- [ ] `scaled_k()` retorna K adaptativo
- [ ] K_max calculado de bandwidth * latency / layer_size
- [ ] K_min = 1
- [ ] Conteiner não entra em K=0 loop (deadlock prevention)

---

### T3: Stress tests

- **Estimativa**: 3d (24h)
- **Labels**: `sprint-5`, `rust`, `tests`
- **Descrição**: Simular cenários de stress: banda ilimitada, banda limitada, preditor perfeito, preditor aleatório, burst de tokens.

**Critério de Aceitação**:
- [ ] Testes rodam sem crashes
- [ ] Degradação graceful: K reduz suavemente
- [ ] Nenhum stall > 500ms em cenários realistas
- [ ] Relatório com gráficos (CSV → plot)

---

## Sprint 6 — QAT + E2E (semana 12)

### T1: QAT per-block pipeline

- **Estimativa**: 4d (32h)
- **Labels**: `sprint-6`, `rust`, `qat`
- **Arquivo**: `swamp-engine/src/qat.rs` (modificar) + `convert_shard.rs` (integrar)
- **Descrição**: Pipeline de calibration QAT durante conversão. Ajustar scales/mins dos blocos Q4_K.

**Critério de Aceitação**:
- [ ] `--qat` flag no converter tool
- [ ] Calibration roda com dados de entrada
- [ ] Per-block error metrics reportados
- [ ] Shard com QAT produz inference quality similar ao GGUF original

---

### T2: End-to-end com modelo 7B/13B

- **Estimativa**: 4d (32h)
- **Labels**: `sprint-6`, `rust`, `integration`
- **Arquivo**: `swamp-tools/src/bin/benchmark_prefill.rs`
- **Descrição**: Rodar inferência completa com streaming habilitado. Medir tokens/s, hit_rate, fallback count, bandwidth.

**Critério de Aceitação**:
- [ ] Inferência completa roda sem crash (TinyLlama 1.1B)
- [ ] Log com tokens/s, hit_rate, fallback_count
- [ ] 3 cenários (ideal K=6, misto K=3, fallback K=1)
- [ ] Baseline comparison (CPU-only vs streaming)

---

### T3: Correções de estabilidade + documentação

- **Estimativa**: 2d (16h)
- **Labels**: `sprint-6`, `rust`, `docs`
- **Descrição**: README atualizado. Runbook com comandos de teste.

**Critério de Aceitação**:
- [ ] README com instruções de streaming
- [ ] [[runbook.md]] completo
- [ ] [[tasks-backlog.md]] revisado

---

## R1: Sprint 7 — DirectStorage + io_uring (semanas 13-14)

### T1: DirectStorage path (Windows)

- **Estimativa**: 2 semanas (80h)
- **Labels**: `r1`, `windows`, `directstorage`
- **Descrição**: Implementar DirectStorage queue para NVMe→VRAM no Windows. Portar TransferEngine para usar ID3D12Device + IDStorageQueue.

**Critério de Aceitação**:
- [ ] DirectStorage queue criada
- [ ] Upload via DS comparável a Vulkan transfer queue
- [ ] Timeline semaphores funcionam com DX12 interop
- [ ] Testado em Windows com GPU NVIDIA

---

### T2: io_uring optimizações (Linux)

- **Estimativa**: 2 semanas (80h)
- **Labels**: `r1`, `linux`, `iouring`
- **Descrição**: io_uring async read para shard.bin. Substituir mmap por io_uring em setups de alta performance.

**Critério de Aceitação**:
- [ ] io_uring setup + submission queue
- [ ] async read combinado com Vulkan timeline
- [ ] benchmark mostra melhoria vs mmap baseline

---

## R1: Sprint 8 — DSPark Improvements (semanas 15-16)

### T1: Confidence scoring + multi-token prediction

- **Estimativa**: 2 semanas (80h)
- **Labels**: `r1`, `dspark`, `rust`
- **Descrição**: Melhorar DSPark com confidence scoring calibrado, multi-token prediction (não só top-1). Heurísticas conservador/agressivo.

**Critério de Aceitação**:
- [ ] confidence score correlacionado com acerto
- [ ] multi-token prediction functional
- [ ] hit rate aumenta 5%+ em benchmark

---

## R1: Sprint 9 — Low-Confidence Reconstruction (semanas 17-18)

### T1: Bloques aproximados para low-confidence

- **Estimativa**: 2 semanas (80h)
- **Labels**: `r1`, `rust`, `reconstruction`
- **Descrição**: Para blocos com baixa confiança, reconstruir aproximação via LR/quant stubs. Reduzir fallback rate.

**Critério de Aceitação**:
- [ ] fallback reduction de X% em workloads sintéticos
- [ ] qualidade de saída aceitável (perplexity dentro de 1%)
- [ ] integrado ao NyionVM dispatcher

---

## R1: Sprint 10 — Performance Tune (semanas 19-20)

### T1: CUDA integration + cuMem VMM

- **Estimativa**: 2-4 semanas (80-160h)
- **Labels**: `r1`, `cuda`, `performance`
- **Descrição**: Integrar CUDA kernels existentes (`fused_attention.cu`). Usar cuMem VMM para virtual memory mapping. Melhorar tokens/s.

**Critério de Aceitação**:
- [ ] CUDA path funcional (hoje retorna erro)
- [ ] cuMem VMM para sparse weight mapping
- [ ] Melhoria de tokens/s vs Vulkan baseline
- [ ] Fallback para Vulkan se CUDA indisponível

---

## R1: Sprint 11 — Large-Model Tooling (semanas 21-22)

### T1: Sharder para 1T models

- **Estimativa**: 2 semanas (80h)
- **Labels**: `r1`, `tools`, `rust`
- **Descrição**: Escalar shard converter para modelos de até 1 trilhão de parâmetros. Sanity checks (checksums, integridade).

**Critério de Aceitação**:
- [ ] Converter roda com modelos > 100 GB
- [ ] Sanity checks passam
- [ ] Deploy checklist documentado

---

## Totals

| Fase | Sprints | Dias | Horas |
|------|---------|------|-------|
| MVP (Sprints 0-6) | 7 sprints | ~42 dias | ~336h |
| R1 (Sprints 7-11) | 5 sprints | ~50 dias | ~400h |
| **Total** | **12 sprints** | **~92 dias** | **~736h** |
