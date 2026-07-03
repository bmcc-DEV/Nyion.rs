// swamp-gpu/examples/bench_persistent.rs
// Benchmark: GPU attention with persistent K/V cache vs non-persistent vs CPU

use swamp_gpu::*;

use std::time::Instant;

fn main() {
    let n_heads = 32;
    let n_kv_heads = 4;
    let head_dim = 64;
    let seq_lens = [64, 128, 256, 512, 1024, 2048];

    println!("Benchmark: GPU attention (persistent K/V vs non-persistent)");
    println!("n_heads={}, n_kv_heads={}, head_dim={}", n_heads, n_kv_heads, head_dim);
    println!("{:<8} {:>12} {:>12} {:>12} {:>12}", "seq_len", "CPU(ms)", "GPU_nonpersist(ms)", "GPU_persist(ms)", "speedup");

    let max_seq = 2048;

    // Initialize GPU
    if gpu_init().is_err() {
        eprintln!("GPU init failed!");
        return;
    }

    // Allocate persistent K/V buffers
    let d_k_buf = gpu_alloc_kv_buffer(n_kv_heads, max_seq, head_dim).unwrap();
    let d_v_buf = gpu_alloc_kv_buffer(n_kv_heads, max_seq, head_dim).unwrap();

    let mut all_k: Vec<Vec<f32>> = Vec::new();
    let mut all_v: Vec<Vec<f32>> = Vec::new();

    // Pre-populate K/V
    for pos in 0..max_seq {
        let mut k_pos = vec![0.0f32; n_kv_heads * head_dim];
        let mut v_pos = vec![0.0f32; n_kv_heads * head_dim];
        for h in 0..n_kv_heads {
            for d in 0..head_dim {
                let v = ((pos * n_kv_heads + h) * head_dim + d) as f32 / 100.0;
                k_pos[h * head_dim + d] = v.sin();
                v_pos[h * head_dim + d] = v.cos();
            }
        }
        all_k.push(k_pos);
        all_v.push(v_pos);
    }

    // Q vector (fixed, arbitrary)
    let q: Vec<f32> = (0..n_heads * head_dim).map(|i| ((i as f32) * 0.7).sin()).collect();

    for &seq_len in &seq_lens {
        // --- Non-persistent GPU attention (old approach) ---
        let start = Instant::now();
        let trials = 10;
        let mut out_nonpersist = vec![0.0f32; n_heads * head_dim];
        for _ in 0..trials {
            // Build contiguous K/V (gather from "cache")
            let mut k_contig = vec![0.0f32; n_kv_heads * seq_len * head_dim];
            let mut v_contig = vec![0.0f32; n_kv_heads * seq_len * head_dim];
            for pos in 0..seq_len {
                for h in 0..n_kv_heads {
                    let src_off = h * head_dim;
                    let dst_off = (h * seq_len + pos) * head_dim;
                    k_contig[dst_off..dst_off + head_dim].copy_from_slice(&all_k[pos][src_off..src_off + head_dim]);
                    v_contig[dst_off..dst_off + head_dim].copy_from_slice(&all_v[pos][src_off..src_off + head_dim]);
                }
            }
            let _ = gpu_attention_forward(
                &q, &k_contig, &v_contig, &mut out_nonpersist,
                n_heads, n_kv_heads, seq_len, head_dim,
            );
        }
        let elapsed_nonpersist = start.elapsed().as_secs_f64() / trials as f64;

        // --- Persistent GPU attention (new approach) ---
        // Copy K/V to GPU buffer incrementally (as would happen during prompt processing)
        for pos in 0..seq_len {
            let _ = gpu_copy_kv_layer(d_k_buf, &all_k[pos], pos, n_kv_heads, max_seq, head_dim);
            let _ = gpu_copy_kv_layer(d_v_buf, &all_v[pos], pos, n_kv_heads, max_seq, head_dim);
        }

        let start = Instant::now();
        let mut out_persist = vec![0.0f32; n_heads * head_dim];
        for _ in 0..trials {
            let _ = gpu_sync(); // ensure copies done
            let d_q = gpu_alloc_and_copy_q(&q, n_heads, head_dim).unwrap();
            let d_out = gpu_alloc_kv_buffer(1, n_heads, head_dim).unwrap();
            let _ = gpu_attention_device(
                d_q, d_k_buf, d_v_buf, d_out,
                n_heads, n_kv_heads, seq_len, head_dim, max_seq,
            );
            let _ = gpu_copy_output_to_host(&mut out_persist, d_out, n_heads, head_dim);
            gpu_free(d_q as *mut _).ok();
            gpu_free(d_out as *mut _).ok();
        }
        let elapsed_persist = start.elapsed().as_secs_f64() / trials as f64;

        // --- CPU attention (reference) ---
        let start = Instant::now();
        let mut out_cpu = vec![0.0f32; n_heads * head_dim];
        for _ in 0..trials {
            for h in 0..n_heads {
                let kv_h = h * n_kv_heads / n_heads;
                let mut scores = vec![0.0f32; seq_len];
                for t in 0..seq_len {
                    let mut dot = 0.0;
                    let q_ptr = &q[h * head_dim..];
                    let k_ptr = &all_k[t][kv_h * head_dim..];
                    for d in 0..head_dim { dot += q_ptr[d] * k_ptr[d]; }
                    scores[t] = dot * (1.0 / (head_dim as f32).sqrt());
                }
                // softmax
                let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0;
                for t in 0..seq_len { sum_exp += (scores[t] - max_val).exp(); }
                let inv_sum = 1.0 / sum_exp;
                let mut out_v = vec![0.0f32; head_dim];
                for t in 0..seq_len {
                    let prob = (scores[t] - max_val).exp() * inv_sum;
                    let v_ptr = &all_v[t][kv_h * head_dim..];
                    for d in 0..head_dim { out_v[d] += prob * v_ptr[d]; }
                }
                let out_slice = &mut out_cpu[h * head_dim..(h + 1) * head_dim];
                out_slice.copy_from_slice(&out_v);
            }
        }
        let elapsed_cpu = start.elapsed().as_secs_f64() / trials as f64;

        // Verify correctness (persistent vs CPU)
        let mut max_diff = 0.0;
        for i in 0..n_heads * head_dim {
            let diff = (out_persist[i] - out_cpu[i]).abs();
            if diff > max_diff { max_diff = diff; }
        }
        let speedup = elapsed_nonpersist / elapsed_persist;
        println!("{:<8} {:>8.4}ms {:>12.4}ms {:>12.4}ms {:>8.2}x (err={:.2e})",
                 seq_len, elapsed_cpu * 1000.0, elapsed_nonpersist * 1000.0, elapsed_persist * 1000.0, speedup, max_diff);
    }

    // Cleanup
    gpu_free(d_k_buf as *mut _).ok();
    gpu_free(d_v_buf as *mut _).ok();
}
