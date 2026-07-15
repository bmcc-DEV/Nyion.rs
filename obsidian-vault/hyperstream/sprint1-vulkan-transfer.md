# Sprint 1 — Vulkan Transfer + Triple Buffer

**Status**: ✅ COMPLETO

## Arquivos Envolvidos

| Arquivo | Linhas | O que faz |
|---------|--------|-----------|
| `swamp-gpu/src/vulkan.rs` | 286 | `VkBackend`: instance, device, queue, memory allocation, shader modules, compute pipelines. Vulkan 1.2. |
| `swamp-gpu/src/compute_graph.rs` | 247 | `ComputeGraph`: DAG builder de compute nodes, timeline semaphore, pipeline barriers, fence sync |
| `swamp-gpu/src/streaming/transfer.rs` | 217 | `TransferEngine`: staging buffer (HOST_VISIBLE), timeline semaphore, copy commands |
| `swamp-gpu/src/streaming/triple_buffer.rs` | 112 | `TripleBuffer`: 3 device-local buffers com state machine Free→Staging→DeviceReady→Computing |
| `swamp-gpu/src/lib.rs` | 677 | `GpuDevice`, `GpuBuf`, `GpuComputeContext` — integração dos componentes |

## Implementação Detalhada

### VkBackend (`vulkan.rs`)

```rust
pub struct VkBackend {
    pub instance: vk::Instance,
    pub device: vk::Device,
    pub physical: vk::PhysicalDevice,
    pub _queue: vk::Queue,
    pub _queue_family: u32,
    pub enabled: bool,
}
```

- Cria instância Vulkan com debug messenger opcional
- Seleciona physical device com queue transfer (ou compute com TRANSFER_BIT)
- Expõe `allocate_buffer(size, usage, mem_flags) → (Buffer, DeviceMemory)`
- Carrega shaders SPIR-V → `vk::ShaderModule`
- Cria `vk::ComputePipeline` para cada shader type

### TransferEngine (`transfer.rs`)

```rust
pub struct TransferEngine {
    backend: Arc<VkBackend>,
    pub triple_buffer: TripleBuffer,
    timeline_sem: vk::Semaphore,
    timeline_val: AtomicU64,
    staging_buf: vk::Buffer,
    staging_mapped: *mut u8,
    staging_size: u64,
    pub telemetry: StreamTelemetry,
}
```

- Timeline semaphore criado com `VK_SEMAPHORE_TYPE_TIMELINE`, initial value 0
- `upload_slot(data)` → copia pra staging → `cmd_copy_buffer` → sinaliza timeline → marca DeviceReady
- `wait_for_timeline(value, timeout_ns)` → CPU espera GPU via `vkWaitSemaphores`
- `sync_all()` → espera último timeline value

### TripleBuffer (`triple_buffer.rs`)

```rust
pub struct TripleBuffer {
    pub slots: [BufferSlot; 3],
    write_index: usize,
    pub max_slot_size: u64,
}
```

Cada slot é DEVICE_LOCAL, usado como TRANSFER_DST | STORAGE_BUFFER.

State machine:
- `acquire_write_slot()` → Free → Staging
- `mark_device_ready(idx, tv)` → Staging → DeviceReady
- `acquire_compute_slot()` → DeviceReady → Computing
- `release_slot(idx)` → Computing → Free

### Benchmark (`benchmark_streaming.rs`)

Já existe e funciona:
- Abre shard via `ShardReader`
- Cria `TransferEngine`
- Loop: `upload_slot()` → `wait_for_timeline()` → `release_slot()`
- Reporta throughput GB/s

## Acceptance Checklist

- [x] VkBackend enumera queue families e cria transfer queue
- [x] Timeline semaphore criado e sinalizado em cada submit
- [x] 3 buffers DEVICE_LOCAL alocados
- [x] upload_slot transfere dados sem blocking CPU (exceto no sync final)
- [x] benchmark reproduzível com números de throughput

## Timeline Semaphore Signal Chain

```
CPU: upload_slot(data)
  → staging[0] = data (memcpy)
  → cmd_copy_buffer(staging → slot[N])
  → queue_submit(..., signal=timeline_val+1)
  → mark_device_ready(N, timeline_val)
  ──┐
GPU: │ cmd_copy_buffer executa
  ←─┘ sinaliza timeline
CPU: wait_for_timeline(val) → GPU completou

Compute: acquire_compute_slot() → timeline_val match → kernel roda
```

## Próximos Passos

- [[sprint2-shard-io.md]] — Converter tool + integrar ShardReader ao TransferEngine
- [[sprint3-dspark-scheduling.md]] — DSPark + layer schedule resolver
