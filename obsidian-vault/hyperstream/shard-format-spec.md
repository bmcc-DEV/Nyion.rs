# Shard Format Specification v1.0

## Visão Geral

O formato de shard consiste em dois arquivos:

- **`index.json`** — Metadados: modelo, quantização, offsets e sizes por layer/tensor
- **`shard.bin`** — Dados binários: blocos Q4_K contíguos, alinhados a 256 bytes

## index.json Schema

```json
{
  "format": "swamp-hyperstream-v1",
  "model": "tinyllama-1.1b",
  "quant": "Q4_K",
  "num_layers": 22,
  "layers": [
    {
      "id": 0,
      "name": "blk.0",
      "tensors": {
        "q":         { "offset": 0,         "size": 2621440,  "n_blocks": 2560 },
        "k":         { "offset": 2621440,   "size": 1310720,  "n_blocks": 1280 },
        "v":         { "offset": 3932160,   "size": 1310720,  "n_blocks": 1280 },
        "o":         { "offset": 5242880,   "size": 2621440,  "n_blocks": 2560 },
        "gate":      { "offset": 7864320,   "size": 6881280,  "n_blocks": 6720 },
        "up":        { "offset": 14745600,  "size": 6881280,  "n_blocks": 6720 },
        "down":      { "offset": 21626880,  "size": 6881280,  "n_blocks": 6720 },
        "attn_norm": { "offset": 28508160,  "size": 16384,    "n_blocks": 0 },
        "ffn_norm":  { "offset": 28524544,  "size": 16384,    "n_blocks": 0 }
      }
    }
  ]
}
```

### Campos

- `format`: Identificador do formato (obrigatório: `"swamp-hyperstream-v1"`)
- `model`: Nome do modelo (ex: `"tinyllama-1.1b"`)
- `quant`: Quantização (ex: `"Q4_K"`)
- `num_layers`: Número de layers
- `layers[]`: Array de layers
  - `id`: Índice da layer (0-based)
  - `name`: Nome (ex: `"blk.0"`)
  - `tensors`: Mapa de nome → info
    - `offset`: Offset em bytes no shard.bin
    - `size`: Tamanho em bytes
    - `n_blocks`: Número de blocos Q4_K (0 se não quantizado, ex: norms)

### Nomes de Tensores

| Nome | Tipo | Quantizado? | Descrição |
|------|------|-------------|-----------|
| `q` | weight | Q4_K | Q projection (`blk.{N}.attn_q.weight` no GGUF) |
| `k` | weight | Q4_K | K projection (`blk.{N}.attn_k.weight`) |
| `v` | weight | Q4_K | V projection (`blk.{N}.attn_v.weight`) |
| `o` | weight | Q4_K | Output projection (`blk.{N}.attn_output.weight`) |
| `gate` | weight | Q4_K | FFN gate (`blk.{N}.ffn_gate.weight`) |
| `up` | weight | Q4_K | FFN up (`blk.{N}.ffn_up.weight`) |
| `down` | weight | Q4_K | FFN down (`blk.{N}.ffn_down.weight`) |
| `attn_norm` | norm | f32 | Attention RMS norm (`blk.{N}.attn_norm.weight`) |
| `ffn_norm` | norm | f32 | FFN RMS norm (`blk.{N}.ffn_norm.weight`) |

## shard.bin Layout

```
┌──────────────────────┐
│ Layer 0              │
│  ├─ attn_q           │  blocos Q4_K (256 elementos cada)
│  ├─ attn_k           │
│  ├─ attn_v           │
│  ├─ attn_o           │
│  ├─ ffn_gate         │
│  ├─ ffn_up           │
│  ├─ ffn_down         │
│  ├─ attn_norm (f32)  │  não quantizado
│  └─ ffn_norm (f32)   │  não quantizado
├──────────────────────┤
│ Layer 1              │
│  ...                 │
├──────────────────────┤
│ ...                  │
└──────────────────────┘
```

### Formato Q4_K Block (256 elementos = 168 bytes)

Cada bloco Q4_K (16 sub-blocos de 16):

```
Offset  Size    Campo
0       2       d (super-block scale, half)
2       2       dmin (super-block min, half)
4       48      scales (16 × 3 bytes = 6 bits cada)
52      108     qs (256 × 3.375 bits = 108 bytes)
160     8       padding (alinhamento para 168)
```

### Alinhamento

- Cada tensor começa em offset múltiplo de 256 bytes
- O shard.bin inteiro pode ser mapeado com `mmap` e acessado sem cópias

## Converter Tool

A ferramenta `convert_shard` (a ser implementada em `swamp-tools/src/bin/convert_shard.rs`) deve:

1. Ler GGUF via `swamp-gguf`
2. Para cada layer, extrair cada tensor
3. Se quantizado (Q4_K), copiar blocos raw
4. Se f32 (norms), copiar raw
5. Escrever `shard.bin` sequencialmente
6. Gerar `index.json` com offsets calculados

## Consumidor

`swamp-gpu/src/streaming/shard.rs` — `ShardReader`:
- Abre index.json + shard.bin via mmap
- `get_tensor(layer_id, tensor_name)` → `Option<&[u8]>` (zero-copy)

## Versionamento

- `format` field no index.json permite evolução futura (v2, v3)
- Breaking changes incrementam major version: `swamp-hyperstream-v2`
- Compatibilidade retroativa via conversão offline
