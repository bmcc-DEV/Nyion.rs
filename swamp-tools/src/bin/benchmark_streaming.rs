use anyhow::{Context, Result};
use clap::Parser;
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "nyion-benchmark-streaming")]
#[command(about = "Nyion HyperStream - benchmark de transferencia NVMe->VRAM via Vulkan")]
struct Cli {
    index_path: String,
    data_path: String,

    #[arg(long, default_value = "1")]
    layer_id: u32,

    #[arg(long, default_value = "0")]
    tensor_slot: usize,

    #[arg(long, default_value = "10")]
    repeats: usize,

    #[arg(long, default_value = "16777216")]
    slot_size: u64,

    #[arg(long, default_value = "3")]
    prefetch_k: usize,
}

const TENSOR_NAMES: &[&str] = &["q", "k", "v", "o", "gate", "up", "down", "attn_norm", "ffn_norm"];

fn main() -> Result<()> {
    let cli = Cli::parse();

    let backend = Arc::new(swamp_gpu::VkBackend::new());
    if !backend.enabled {
        anyhow::bail!("Vulkan backend nao disponivel");
    }
    println!("Vulkan backend ready");

    let mut engine = swamp_engine::streamer::HyperStreamEngine::open(
        &backend,
        &cli.index_path,
        &cli.data_path,
        cli.slot_size,
    ).context("abrindo HyperStreamEngine")?;

    engine.set_k(cli.prefetch_k);

    let tensor_name = if cli.tensor_slot < TENSOR_NAMES.len() {
        TENSOR_NAMES[cli.tensor_slot]
    } else {
        "q"
    };

    println!("Benchmark: layer={}, tensor={}, repeats={}, K={}",
        cli.layer_id, tensor_name, cli.repeats, cli.prefetch_k);

    engine.sync_all();

    let start = Instant::now();
    for i in 0..cli.repeats {
        engine.prefetch_layer(cli.layer_id as usize, tensor_name);

        let result = engine.wait_for_layer(
            cli.layer_id as usize,
            tensor_name,
            1_000_000_000,
        );

        match result.decision {
            swamp_engine::streamer::FallbackDecision::Gpu(_) => {
                // GPU path: slot was ready in time
            }
            swamp_engine::streamer::FallbackDecision::Cpu => {
                // CPU fallback (not an error in benchmark, but notable)
                print!("F");
            }
        }

        if (i + 1) % 10 == 0 {
            print!(".");
        }
    }
    let elapsed = start.elapsed();

    engine.sync_all();
    let telemetry = engine.telemetry();
    let total_bytes = telemetry.bytes_transferred.load(std::sync::atomic::Ordering::Relaxed);
    let transfers = telemetry.transfers_completed.load(std::sync::atomic::Ordering::Relaxed);
    let hits = telemetry.prefetch_hits.load(std::sync::atomic::Ordering::Relaxed);
    let misses = telemetry.prefetch_misses.load(std::sync::atomic::Ordering::Relaxed);
    let bw = total_bytes as f64 / elapsed.as_secs_f64() / 1_000_000_000.0;

    println!();
    println!("=== Streaming Benchmark Results ===");
    println!("  Repeats:      {}", cli.repeats);
    println!("  Elapsed:      {:.3} s", elapsed.as_secs_f64());
    println!("  Throughput:   {:.2} GB/s", bw);
    println!("  Transfers:    {}", transfers);
    println!("  Total bytes:  {} MB", total_bytes / 1_000_000);
    println!("  Prefetch hits:{}/{} ({:.1}%)", hits, hits + misses,
        if hits + misses > 0 { hits as f64 / (hits + misses) as f64 * 100.0 } else { 0.0 });
    println!("  K:            {}", engine.k());

    Ok(())
}
