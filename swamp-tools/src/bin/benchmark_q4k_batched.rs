// Microbenchmark: batched Q4_K GEMV vs single GEMV
// Uses synthetic Q4_K weight data matching realistic layer shapes

use std::time::Instant;

fn main() {
    // TinyLlama-1.1B feed-forward dimensions
    let n_rows = 4096;      // ffn output dim
    let n_cols = 2048;      // ffn hidden dim (huggingface: 2048 -> 5632 gate/up, 4096 down)
    let batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128];
    let n_warmup = 5;
    let n_iters  = 20;

    // Build synthetic Q4_K weight data
    let n_blocks = n_cols / 256;
    let weight_bytes = n_rows * n_blocks * 144;
    let mut w = vec![0u8; weight_bytes];

    // Fill with semi-random Q4_K blocks
    for block_start in (0..weight_bytes).step_by(144) {
        let d: f32 = 1.0 + (block_start % 7) as f32 * 0.1;
        let dm: f32 = 0.05 + (block_start % 3) as f32 * 0.02;
        w[block_start..block_start + 2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
        w[block_start + 2..block_start + 4].copy_from_slice(&half::f16::from_f32(dm).to_le_bytes());
        // scales 0-3 (6-bit, bytes 4-7)
        for j in 0..4 {
            w[block_start + 4 + j] = ((block_start / 144 + j * 7) & 0x3F) as u8;
            w[block_start + 8 + j] = ((block_start / 144 + j * 3 + 1) & 0x3F) as u8;
        }
        // high bytes 12-15: low/high nibbles for scales[4..7], mins[4..7]
        w[block_start + 12] = 0x12;
        w[block_start + 13] = 0x34;
        w[block_start + 14] = 0x56;
        w[block_start + 15] = 0x78;
        // nibbles: alternating 0x08, 0x19, etc.
        for i in 0..128 {
            let lo = (i * 7 + block_start) % 16;
            let hi = (i * 3 + block_start / 144) % 16;
            w[block_start + 16 + i] = lo as u8 | ((hi as u8) << 4);
        }
    }

    // Pre-allocate input/output buffers
    let mut xs = vec![0.0f32; batch_sizes[batch_sizes.len() - 1] * n_cols];
    let mut outs_single = vec![0.0f32; n_rows];
    let mut outs_batched = vec![0.0f32; batch_sizes[batch_sizes.len() - 1] * n_rows];

    for i in 0..xs.len() {
        xs[i] = ((i % 997) as f32 / 500.0 - 1.0) * 0.5
            + (((i * 3 + 7) % 100) as f32 / 50.0 - 1.0) * 0.3;
    }

    println!("=== Q4_K Batched GEMV Benchmark ===");
    println!("Rows: {}, Cols: {}, Blocks: {}", n_rows, n_cols, n_blocks);
    println!("Weight data: {:.1} MB", weight_bytes as f64 / 1_048_576.0);
    println!("");

    for &bs in &batch_sizes {
        let x_ptrs: Vec<*const f32> = (0..bs)
            .map(|t| xs[t * n_cols..].as_ptr())
            .collect();
        let out_ptrs_batched: Vec<*mut f32> = (0..bs)
            .map(|t| outs_batched[t * n_rows..].as_mut_ptr())
            .collect();

        // warmup
        for _ in 0..n_warmup {
            for t in 0..bs {
                outs_single.fill(0.0);
                swamp_kernels::fused_gemv_q4k(&w, &xs[t * n_cols..(t + 1) * n_cols], &mut outs_single, n_rows, n_cols);
            }
            outs_batched.fill(0.0);
            swamp_kernels::fused_gemv_q4k_batched(&w, &x_ptrs, &out_ptrs_batched, n_rows, n_cols, bs);
        }

        // Benchmark single (sequential calls to fused_gemv_q4k)
        let mut single_elapsed = std::time::Duration::ZERO;
        for _ in 0..n_iters {
            for t in 0..bs {
                outs_single.fill(0.0);
                let t0 = Instant::now();
                swamp_kernels::fused_gemv_q4k(
                    &w,
                    &xs[t * n_cols..(t + 1) * n_cols],
                    &mut outs_single,
                    n_rows,
                    n_cols,
                );
                single_elapsed += t0.elapsed();
            }
        }
        let single_gop = (n_rows as f64 * n_cols as f64 * 2.0 * bs as f64 * n_iters as f64)
            / single_elapsed.as_secs_f64()
            / 1e9;

        // Benchmark batched (single call to fused_gemv_q4k_batched)
        let mut batched_elapsed = std::time::Duration::ZERO;
        for _ in 0..n_iters {
            outs_batched.fill(0.0);
            let t0 = Instant::now();
            swamp_kernels::fused_gemv_q4k_batched(
                &w,
                &x_ptrs,
                &out_ptrs_batched,
                n_rows,
                n_cols,
                bs,
            );
            batched_elapsed += t0.elapsed();
        }
        let batched_gop = (n_rows as f64 * n_cols as f64 * 2.0 * bs as f64 * n_iters as f64)
            / batched_elapsed.as_secs_f64()
            / 1e9;

        let speedup = batched_gop / single_gop;

        println!(
            "batch={:>3}:  single={:7.2} GFLOPS  batched={:7.2} GFLOPS  speedup={:.2}x",
            bs, single_gop, batched_gop, speedup,
        );
    }
}
