use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::io::{Seek, Write};
use std::path::PathBuf;
use swamp_gguf::GgufFile;

#[derive(Parser)]
#[command(name = "nyion-convert-shard")]
#[command(about = "Converte GGUF para shard.bin + index.json (Nyion HyperStream)")]
struct Cli {
    #[arg(long = "gguf")]
    gguf_path: PathBuf,

    #[arg(long = "output-dir")]
    output_dir: PathBuf,

    #[arg(long, default_value = "")]
    model_name: String,

    #[arg(long, default_value = "Q4_K")]
    quant: String,

    #[arg(long)]
    qat: bool,
}

#[derive(Debug, Serialize)]
struct TensorInfo {
    offset: u64,
    size: u64,
    n_blocks: u32,
}

#[derive(Debug, Serialize)]
struct LayerInfo {
    id: u32,
    name: String,
    tensors: HashMap<String, TensorInfo>,
}

#[derive(Debug, Serialize)]
struct ShardIndex {
    format: String,
    model: String,
    quant: String,
    num_layers: u32,
    layers: Vec<LayerInfo>,
}

const TENSOR_NAMES: &[( &str,  &str)] = &[
    ("q", "blk.{}.attn_q.weight"),
    ("k", "blk.{}.attn_k.weight"),
    ("v", "blk.{}.attn_v.weight"),
    ("o", "blk.{}.attn_output.weight"),
    ("gate", "blk.{}.ffn_gate.weight"),
    ("up", "blk.{}.ffn_up.weight"),
    ("down", "blk.{}.ffn_down.weight"),
    ("attn_norm", "blk.{}.attn_norm.weight"),
    ("ffn_norm", "blk.{}.ffn_norm.weight"),
];

fn gguf_tensor_name(layer: usize, pattern: &str) -> String {
    pattern.replace("{}", &layer.to_string())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    fs::create_dir_all(&cli.output_dir)
        .with_context(|| format!("criando {}", cli.output_dir.display()))?;

    let gguf = GgufFile::open(&cli.gguf_path)
        .with_context(|| format!("abrindo {}", cli.gguf_path.display()))?;

    let num_layers = gguf.metadata.n_layer()
        .context("modelo sem n_layer no metadata")? as u32;
    let architecture = gguf.architecture();
    let model_name = if cli.model_name.is_empty() {
        cli.gguf_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string()
    } else {
        cli.model_name.clone()
    };

    println!("=== Nyion HyperStream - Shard Converter ===");
    println!("  Modelo:       {}", model_name);
    println!("  Arquitetura:  {}", architecture);
    println!("  GGUF:         {}", cli.gguf_path.display());
    println!("  Output:       {}", cli.output_dir.display());
    println!("  Layers:       {}", num_layers);
    if cli.qat {
        println!("  QAT:          habilitado");
    }

    let shard_path = cli.output_dir.join("shard.bin");
    let index_path = cli.output_dir.join("index.json");
    let mut shard = fs::File::create(&shard_path)
        .with_context(|| format!("criando {}", shard_path.display()))?;

    let mut layers: Vec<LayerInfo> = Vec::with_capacity(num_layers as usize);

    for layer in 0..num_layers {
        let mut tensors = HashMap::new();

        for &(short_name, gguf_pattern) in TENSOR_NAMES {
            let gguf_name = gguf_tensor_name(layer as usize, gguf_pattern);

            let tensor = match gguf.tensor(&gguf_name) {
                Some(t) => t,
                None => {
                    eprintln!("  Aviso: tensor {} nao encontrado na layer {}", gguf_name, layer);
                    continue;
                }
            };

            let raw = gguf.tensor_raw_bytes(tensor)
                .with_context(|| format!("lendo {} bytes de {}", tensor.nbytes(), gguf_name))?;

            let offset = shard.stream_position()?;
            shard.write_all(raw)
                .with_context(|| format!("escrevendo {}", gguf_name))?;

            let n_blocks = if tensor.dtype.block_size() > 1 {
                (tensor.n_elems() / tensor.dtype.block_size()) as u32
            } else {
                0
            };

            tensors.insert(short_name.to_string(), TensorInfo {
                offset,
                size: raw.len() as u64,
                n_blocks,
            });
        }

        layers.push(LayerInfo {
            id: layer,
            name: format!("blk.{}", layer),
            tensors,
        });

        if (layer + 1) % 8 == 0 || layer == num_layers - 1 {
            println!("  Progresso: layer {}/{} convertida", layer + 1, num_layers);
        }
    }

    let shard_size = shard.stream_position()?;
    drop(shard);

    let index = ShardIndex {
        format: "swamp-hyperstream-v1".to_string(),
        model: model_name,
        quant: cli.quant,
        num_layers,
        layers,
    };

    let index_json = serde_json::to_string_pretty(&index)
        .context("serializando index.json")?;
    fs::write(&index_path, &index_json)
        .with_context(|| format!("escrevendo {}", index_path.display()))?;

    println!();
    println!("=== Conversao completa ===");
    println!("  shard.bin:  {:.2} MB", shard_size as f64 / 1_048_576.0);
    println!("  index.json: {} bytes", index_json.len());
    println!();
    println!("  Para testar o streaming:");
    println!("    cargo run --release -p swamp-tools --bin swamp-benchmark-streaming -- \\");
    println!("      {} {} \\", index_path.display(), shard_path.display());
    println!("      --layer-id 0 --tensor-slot 0 --repeats 100");

    Ok(())
}
