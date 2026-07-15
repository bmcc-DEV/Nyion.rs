use anyhow::{Context, Result};
use clap::Parser;
use rand::Rng;
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
        model: model_name.clone(),
        quant: cli.quant.clone(),
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
    // QAT calibration: reload model, run calibration with dummy activations, rewrite shard
    if cli.qat {
        println!();
        println!("=== QAT Calibration ===");
        let mut model = swamp_engine::model::Model::load(&cli.gguf_path)
            .context("carregando modelo para QAT")?;
        let calibrator = swamp_engine::qat::QatCalibrator::new(&model);

        let num_layers = model.config.num_layers;
        let mut rng = rand::thread_rng();
        for l in 0..num_layers {
            let ring = &model.layer_rings[l];
            let tensor_info: [(usize, usize, usize); 7] = [
                (0, ring.q_nr, ring.q_nc / 256),
                (1, ring.k_nr, ring.k_nc / 256),
                (2, ring.v_nr, ring.v_nc / 256),
                (3, ring.o_nr, ring.o_nc / 256),
                (4, ring.gate_nr, ring.gate_nc / 256),
                (5, ring.up_nr,   ring.up_nc / 256),
                (6, ring.down_nr, ring.down_nc / 256),
            ];
            for &(ti, n_rows, n_blocks_per_row) in &tensor_info {
                for row in 0..n_rows {
                    for bc in 0..n_blocks_per_row {
                        let x_blk: Vec<f32> = (0..256).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
                        let bi = swamp_engine::qat::QatCalibrator::block_idx(row, bc, n_blocks_per_row);
                        calibrator.record_activation(l, ti, bi, &x_blk);
                    }
                }
            }
            if (l + 1) % 8 == 0 || l == num_layers - 1 {
                println!("  Calibrando: layer {}/{}", l + 1, num_layers);
            }
        }

        calibrator.apply_ring(&mut model)?;
        println!("  QAT calibration aplicada");

        // Rewrite shard with calibrated weights
        let mut shard = fs::File::create(&shard_path)
            .with_context(|| format!("recriando {} para QAT", shard_path.display()))?;

        fn ring_slice_for<'a>(ring: &'a swamp_engine::model::RingView, short_name: &str) -> &'a [u8] {
            match short_name {
                "q"    => ring.q_slice(),
                "k"    => ring.k_slice(),
                "v"    => ring.v_slice(),
                "o"    => ring.o_slice(),
                "gate" => ring.gate_slice(),
                "up"   => ring.up_slice(),
                "down" => ring.down_slice(),
                _      => panic!("unknown tensor short name: {}", short_name),
            }
        }

        let mut layers_qat: Vec<LayerInfo> = Vec::with_capacity(num_layers as usize);
        for l in 0..num_layers as usize {
            let mut tensors = HashMap::new();
            for &(short_name, gguf_pattern) in TENSOR_NAMES {
                let gguf_name = gguf_tensor_name(l, gguf_pattern);
                let raw = if let Ok(tensor) = gguf.tensor_or_err(&gguf_name) {
                    if short_name == "attn_norm" || short_name == "ffn_norm" {
                        gguf.tensor_raw_bytes(tensor)
                            .with_context(|| format!("lendo {} bytes de {}", tensor.nbytes(), gguf_name))?
                            .to_vec()
                    } else {
                        ring_slice_for(&model.layer_rings[l], short_name).to_vec()
                    }
                } else {
                    eprintln!("  Aviso: tensor {} nao encontrado na layer {}", gguf_name, l);
                    continue;
                };
                let offset = shard.stream_position()?;
                shard.write_all(&raw)
                    .with_context(|| format!("escrevendo {} (QAT)", gguf_name))?;
                let n_blocks = 0u32;
                tensors.insert(short_name.to_string(), TensorInfo {
                    offset,
                    size: raw.len() as u64,
                    n_blocks,
                });
            }
            layers_qat.push(LayerInfo {
                id: l as u32,
                name: format!("blk.{}", l),
                tensors,
            });
            if (l + 1) % 8 == 0 || l == num_layers as usize - 1 {
                println!("  Re-escrevendo (QAT): layer {}/{}", l + 1, num_layers);
            }
        }

        let shard_size_qat = shard.stream_position()?;
        drop(shard);

        let index_qat = ShardIndex {
            format: "swamp-hyperstream-v1".to_string(),
            model: model_name.clone(),
            quant: cli.quant.clone(),
            num_layers: num_layers as u32,
            layers: layers_qat,
        };
        let index_json_qat = serde_json::to_string_pretty(&index_qat)
            .context("serializando index.json (QAT)")?;
        fs::write(&index_path, &index_json_qat)
            .with_context(|| format!("escrevendo {} (QAT)", index_path.display()))?;
        println!("  shard.bin (QAT): {:.2} MB", shard_size_qat as f64 / 1_048_576.0);
    }

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
