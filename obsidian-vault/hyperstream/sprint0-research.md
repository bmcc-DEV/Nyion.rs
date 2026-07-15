# Sprint 0 — Preparação

## Decisões de Dependências

| Escolha | Opção | Motivo |
|---------|-------|--------|
| Vulkan bindings | **ash** (Rust) | Já usado no projeto. Type-safe, zero-overhead, mantido ativamente. Alternativa Vulkan-Hpp (C++) exigiria FFI extra. |
| I/O assíncrono | **memmap2** (MVP) | `ShardReader` usa mmap. Zero-copy, simples. Futuramente io_uring (Linux) e DirectStorage (Windows) como otimizações. |
| Linguagem | **Rust** (engine) + **GLSL** (shaders) + **C++/CUDA** (kernels GPU) | Engine em Rust. Shaders em GLSL compilados para SPIR-V. CUDA kernels em C++ compilados para .so (já existem em `swamp-gpu/kernels/fused_attention.cu`). |
| Model format | **GGUF** (input) → **shard.bin** (streaming) | GGUF parser já existe em `swamp-gguf`. Shard.bin é formato custom otimizado para streaming (blocos Q4_K contíguos). |
| Serialização | **serde_json** | index.json é JSON. Já usado no projeto. |

## Hardware Matrix

| Plataforma | GPU | VRAM | RAM | NVMe | Status |
|------------|-----|------|-----|------|--------|
| Linux Laptop | iGPU Intel (UHD 770) | Shared (até 512MB) | 16 GB | Sim | ✅ Para testes de fallback |
| Linux Desktop | dGPU (RTX 3060/4060) | 12 GB | 32 GB | Sim | ✅ Pipeline completo |
| Windows Desktop | dGPU + DirectStorage | 12 GB+ | 32 GB | Sim | 🔶 Sprint 7+ |

### Requisitos Mínimos
- Vulkan 1.2 com suporte a timeline semaphores
- `VK_KHR_timeline_semaphore` (parte do Vulkan 1.2 core)
- Transfer queue family distinta (ideal) ou compute queue com `TRANSFER_BIT`
- 2 GB VRAM (modelos 7B Q4_K cabem em ~4 GB)
- 8 GB RAM (sistema + mmap + CPU fallback buffers)

## Formato de Shards

Decisão documentada em [[shard-format-spec.md]].

- Layout: `index.json` (metadados) + `shard.bin` (pesos)
- Block quantization: Q4_K (256 elementos por bloco, 4-bit)
- Tensores organizados por layer: `blk.{N}.{name}`
- index.json contém offset + size + n_blocks por tensor

## Template de Issue (GitHub)

Toda issue deve conter:
- **Título**: `[Sprint X] Tarefa: descrição`
- **Labels**: `sprint-X`, `linguagem`, `componente`
- **Estimativa**: horas (1d = 8h)
- **Critério de aceitação**: bullet points verificáveis
- **Code references**: links para arquivos relevantes
- **Dependencies**: issues que bloqueiam ou são bloqueadas

## Próximos Passos

1. Verificar suporte Vulkan 1.2 timeline semaphores no hardware alvo
2. Configurar CI com testes de smoke para Vulkan
3. Criar primeira issue no GitHub: `[Sprint 1] Vulkan bootstrap + transfer queue`
4. Começar [[sprint1-vulkan-transfer.md]]
