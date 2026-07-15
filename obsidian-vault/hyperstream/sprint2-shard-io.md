# Sprint 2 — Shard IO + mmap + Converter

**Status**: ✅ COMPLETO
- ✅ ShardReader (mmap-based, zero-copy) — `swamp-gpu/src/streaming/shard.rs`
- ✅ mmap file reader — `memmap2` via `ShardReader::open()`
- ✅ **Shard converter tool** — `swamp-tools/src/bin/convert_shard.rs`
- ✅ **HyperStreamEngine** (ShardReader + TransferEngine integrados) — `swamp-engine/src/streamer.rs`

## O Que Já Existe

### ShardReader (`shard.rs`)

```rust
pub struct ShardReader {
    pub index: ShardIndex,
    data: memmap2::Mmap,
}
```

- `ShardReader::open(index_path, data_path)` → mmap do shard.bin
- `get_tensor(layer_id, tensor_name)` → `Option<&[u8]>` (zero-copy slice)
- `total_size()` → bytes mapeados

### ShardIndex (`shard.rs`)

```rust
pub struct ShardIndex {
    pub format: String,
    pub model: String,
    pub quant: String,
    pub num_layers: u32,
    pub layers: Vec<LayerInfo>,
}
```

- `from_path()` → JSON parse
- `layer_offsets(layer_id, tensor_name)` → `Option<&TensorInfo>`

## Implementado

### 1. Converter Tool — `swamp-tools/src/bin/convert_shard.rs`

Lê GGUF via `swamp-gguf` e produz `shard.bin` + `index.json`.

```bash
swamp-convert-shard --gguf model.gguf --output-dir ./shards/ [--qat]
```

### 2. HyperStreamEngine — `swamp-engine/src/streamer.rs`

Integra `ShardReader` + `TransferEngine` em um orquestrador único:
- `prefetch_layer(layer, tensor)` → enfileira upload via triple-buffer
- `prefetch_k_ahead(current_layer)` → prefetch K layers ahead
- `wait_for_layer(layer, tensor, timeout_ns)` → retorna GPU slot ou CPU fallback

### 3. Benchmark Atualizado — `swamp-tools/src/bin/benchmark_streaming.rs`

Agora usa `HyperStreamEngine` em vez de `TransferEngine` diretamente.

## Acceptance Checklist

- [ ] `convert_shard` lê GGUF e produz shard.bin + index.json válidos
- [ ] Shard.bin pode ser lido de volta com ShardReader
- [ ] Integridade: bytes lidos == bytes escritos (checksum CRC32 opcional)
- [ ] Converter roda para modelo 7B toy (TinyLlama)
- [ ] ShardReader + TransferEngine integrados no executor (próximo sprint)

## Formato do Shard

Ver [[shard-format-spec.md]] para a especificação completa.

## Dependências

- [[sprint1-vulkan-transfer.md]] — TransferEngine (necessário para upload)
- [[architecture.md]] — Diagrama de fluxo

## Próximos Passos

1. Implementar `convert_shard.rs`
2. Validar com TinyLlama 1.1B Q4_K
3. Integrar no executor ([sprint3-dspark-scheduling.md])
