# Runbook — Dev Setup, Build, Test, Troubleshooting

## Pré-requisitos

- Rust toolchain (stable, 1.75+)
- Vulkan SDK 1.2+ (para compilar GLSL → SPIR-V)
- CUDA Toolkit (opcional, para `fused_attention.cu`)
- NVMe SSD (recomendado para testes de streaming)

### Verificar Vulkan

```bash
vulkaninfo | grep "VK_KHR_timeline_semaphore"
vulkaninfo | grep "apiVersion"  # deve ser >= 1.2
```

## Build

```bash
# Release build (engine + tools)
cargo build --release -p swamp-engine -p swamp-tools

# Com GPU (Vulkan backend ativo)
cargo build --release --features gpu -p swamp-engine -p swamp-tools

# CUDA kernels (opcional, para GPU Nvidia com CUDA)
cd swamp-gpu/kernels && nvcc -O2 --std=c++17 -arch=sm_75 -shared -Xcompiler "-fPIC" \
  fused_attention.cu -o ../libswamp_gpu.so && cd ../..

# Apenas verificar erros
cargo build --release 2>&1 | grep "^error"
```

## Testes

```bash
# Testes unitários
cargo test -p swamp-kernels
cargo test -p swamp-engine
cargo test -p swamp-gpu

# Benchmark de streaming (NVMe → VRAM)
cargo run --release -p swamp-tools --bin swamp-benchmark-streaming -- \
  /path/to/index.json /path/to/shard.bin \
  --layer-id 0 --tensor-slot 0 --repeats 100

# Benchmark de inferência completo
cargo run --release -p swamp-tools --bin swamp-benchmark-prefill -- \
  /path/to/model.gguf /path/to/tokenizer.json \
  --prompt-tokens 128 --repeats 3 --n-threads 6
```

## Ferramentas de Diagnóstico

### Timeline Semaphore Status
```rust
// Código para debug
let val = engine.get_timeline_value_host()?;
println!("Timeline: signaled={}, completed={}, pending={}",
    engine.current_timeline(), val, engine.current_timeline() - val);
```

### StreamTelemetry CSV
```rust
println!("{}", StreamTelemetry::snapshot_header());
println!("{}", telemetry.snapshot_csv());
// elapsed_s,bytes,bw_gbps,hit_rate,k
```

### Vulkan Validation Layers
```bash
export VK_LAYER_PATH=/usr/share/vulkan/explicit_layer.d
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
cargo run ...
```

## Power Tuning (AVX-512 downclock)

```bash
sudo modprobe msr && sudo wrmsr -a 0x1FC 0x4004005f
```

## Troubleshooting

| Problema | Causa Provável | Solução |
|----------|---------------|---------|
| `VK_ERROR_FEATURE_NOT_PRESENT` | Driver não suporta timeline semaphores | Verificar driver Vulkan >= 1.2 |
| timeline semaphore trava | GPU hang / driver bug | `sync_all()` com timeout; verificar dmesg |
| staging buffer overflow | data.len() > slot_size | Aumentar slot_size no construtor |
| `TransferEngine::acquire_write_slot()` retorna None | 3 slots ocupados | Aumentar throughput ou K menor |
| fallback CPU frequente | hit_rate baixo | AIMD vai reduzir K; verificar bandwidth NVMe |
| CUDA FFI retorna erro | `libswamp_gpu.so` não carregado | Compilar com nvcc (ver build) |
| `feature = "gpu"` não ativo | Build sem `--features gpu` | Adicionar flag |

## CI Pipeline (Planejado)

1. `cargo check` — análise estática
2. `cargo fmt --check` — formatação
3. `cargo clippy` — lints
4. `cargo test` — unit tests
5. `cargo build --release --features gpu` — build completo
6. `swamp-benchmark-streaming` — smoke test (se GPU disponível)

## Referências

- [[architecture.md]] — Diagrama do sistema
- [[integration-plan.md]] — Pontes entre componentes
- [[tasks-backlog.md]] — Issues organizadas por sprint
