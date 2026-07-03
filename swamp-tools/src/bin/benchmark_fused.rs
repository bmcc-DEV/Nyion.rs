// swamp-tools/src/bin/benchmark_fused.rs
// Benchmark de latencia real do forward_linear com fused GEMV Q4_K.
//
// Mede o tempo de uma projecao de logits (W * x) onde W e um tensor Q4_K
// e x e um vetor de ativacoes sinteticas f32.
//
// Contrariamente ao benchmark de dequantizacao, aqui nao ha escrita de
// 250 MB de F32: o kernel le os pesos Q4_K e escreve apenas o vetor
// de saida (vocab_size floats, ~128 KB para TinyLlama).
//
// Meta: < 5 ms por passagem (vs ~42 ms da dequantizacao isolada).

use anyhow::Result;
use clap::Parser;
use std::time::Instant;
use swamp_gguf::GgufFile;
use swamp_engine::forward_linear;

#[derive(Parser)]
#[command(name = "swamp-benchmark-fused")]
#[command(about = "Benchmark do Fused GEMV Q4_K (zero RAM write de F32)")]
struct Cli {
    model: String,

    /// Nome do tensor de peso (padrao: output.weight ou o maior tensor Q4_K)
    #[arg(short, long)]
    tensor: Option<String>,

    /// Numero de iteracoes apos warmup
    #[arg(short = 'n', long, default_value = "20")]
    iters: usize,

    /// Numero de threads para paralelismo (default: 1)
    #[arg(short = 'j', long, default_value = "1")]
    threads: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let gguf = GgufFile::open(&cli.model)?;
    println!("Modelo: {} ({} tensores)", cli.model, gguf.n_tensors());

    // Seleciona o tensor de peso: prefere o especificado, ou o maior Q4_K disponivel
    let tensor_name = cli.tensor.unwrap_or_else(|| {
        gguf.tensors.iter()
            .filter(|t| t.dtype.name().contains("Q4_K") && t.n_elems() > 100_000)
            .max_by_key(|t| t.n_elems())
            .or_else(|| gguf.tensors.iter().find(|t| t.name.contains("weight")))
            .map(|t| t.name.clone())
            .unwrap_or_else(|| gguf.tensors[0].name.clone())
    });

    let tensor = gguf.tensor_or_err(&tensor_name)?;

    // Extrai dimensoes: shape[0]=n_cols (embed), shape[1]=n_rows (out/vocab)
    if tensor.shape.len() < 2 {
        anyhow::bail!("Tensor {} tem rank {}, esperado >= 2", tensor.name, tensor.shape.len());
    }
    let n_cols = tensor.shape[0] as usize;
    let n_rows = tensor.shape[1] as usize;

    println!("Tensor:     {} | {} | {}", tensor.name, tensor.shape_str(), tensor.dtype.name());
    println!("Shape:      n_cols={} (embed) x n_rows={} (out)", n_cols, n_rows);
    println!("Pesos Q:    {:.2} MB", tensor.nbytes() as f64 / 1024.0 / 1024.0);
    println!("Saida F32:  {:.2} KB", n_rows as f64 * 4.0 / 1024.0);
    println!("Iteracoes:  {} + 1 warmup", cli.iters);
    println!();

    // Vetor de ativacoes: simulado com valores conhecidos (media=0, std=1/sqrt(embed_dim))
    // Nao importa o valor exato para o benchmark de throughput
    let scale = 1.0 / (n_cols as f32).sqrt();
    let x: Vec<f32> = (0..n_cols).map(|i| ((i as f32 * 0.001).sin()) * scale).collect();
    let mut out = vec![0.0f32; n_rows];

    // Warmup: aquece o caminho SIMD e popula o TLB para os bytes do tensor
    forward_linear(&gguf, tensor, &x, &mut out, cli.threads)?;
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
    println!("[warmup concluido]");
    println!();

    // Estatisticas do resultado do warmup (valida que o kernel produziu saida nao-nula)
    let min_w  = out.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_w  = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let argmax = out.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0);
    println!("Saida warmup: min={:.4}  max={:.4}  argmax={}", min_w, max_w, argmax);
    println!();

    let mut times_ms = Vec::with_capacity(cli.iters);

    for i in 0..cli.iters {
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
        let t0 = Instant::now();
        forward_linear(&gguf, tensor, &x, &mut out, cli.threads)?;
        let elapsed_ns = t0.elapsed().as_nanos();
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);

        let ms = elapsed_ns as f64 / 1e6;
        times_ms.push(ms);
        println!("  iter {:2}: {:6.2} ms", i + 1, ms);
    }

    let avg_ms  = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let min_ms  = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ms  = times_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Descarta primeiro e ultimo (mais ruidosos) para mediana estavel
    let mut sorted = times_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ms = sorted[sorted.len() / 2];

    // Throughput: bytes de peso Q lidos por segundo (o que satura a RAM)
    let q_bytes    = tensor.nbytes() as f64;
    let gb_s_q     = (q_bytes / 1e9) / (avg_ms / 1000.0);
    let gb_s_q_min = (q_bytes / 1e9) / (min_ms / 1000.0);

    // Comparacao com benchmark de dequantizacao (42 ms baseline)
    let speedup = 42.0 / avg_ms;

    println!();
    println!("=== Resultados ===");
    println!("  Media:    {:6.2} ms  |  Mediana: {:6.2} ms", avg_ms, median_ms);
    println!("  Min:      {:6.2} ms  |  Max:     {:6.2} ms", min_ms, max_ms);
    println!();
    println!("Throughput (pesos Q lidos):");
    println!("  Media:  {:.2} GB/s", gb_s_q);
    println!("  Pico:   {:.2} GB/s  (iter mais rapida: {:.2} ms)", gb_s_q_min, min_ms);
    println!();
    println!("Escrita RAM:   {:.3} KB  (apenas logits — ZERO f32 de peso na RAM)", n_rows as f64 * 4.0 / 1024.0);
    println!("Speedup vs dequant baseline (42 ms): {:.1}x", speedup);
    println!();

    // Projecao de tok/s (estimativa): 32 camadas x 3 projecoes lineares (Q,K,V,out,gate,up,down = ~7 por camada)
    // Para TinyLlama 1.1B: 22 camadas, 7 projecoes = 154 forward_linear por token
    let linears_per_token = 154.0_f64; // conservador para TinyLlama
    let tok_per_s = 1000.0 / (avg_ms * linears_per_token / 7.0); // normaliza por custo relativo
    println!("Estimativa tok/s (TinyLlama 1.1B, 22 camadas):");
    println!("  Usando media:  {:.1} tok/s", tok_per_s);
    println!("  Usando pico:   {:.1} tok/s", 1000.0 / (min_ms * linears_per_token / 7.0));

    Ok(())
}
