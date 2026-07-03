use std::time::Instant;
use clap::Parser;
use anyhow::Result;
use swamp_engine::Model;

#[derive(Parser)]
#[command(name = "swamp-benchmark-prefill")]
struct Args {
    model: String,
    tokenizer: String,
    #[arg(long, default_value = "128")]
    prompt_tokens: usize,
    #[arg(long, default_value = "1")]
    repeats: usize,
}

fn dummy_tokens(n: usize) -> Vec<usize> {
    (0..n).map(|i| 1usize + (i % 100)).collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    swamp_engine::tokenizer::init_tokenizer(&args.tokenizer);

    println!("Loading model...");
    let model = Model::load(&args.model)?;
    model.print_info();

    let head_dim = model.config.embed_dim / model.config.num_heads;
    swamp_engine::ops::init_rope_lut(head_dim, model.config.context_len);
    let ffn_gate_t = model.gguf.tensor_or_err("blk.0.ffn_gate.weight")?;
    let ffn_dim = ffn_gate_t.shape[1] as usize;
    let num_layers = model.config.num_layers;

    let token_embd = model.gguf.dequantize_tensor_alloc(model.gguf.tensor_or_err("token_embd.weight").unwrap()).unwrap();
    let mut attn_norms = Vec::with_capacity(num_layers);
    let mut ffn_norms = Vec::with_capacity(num_layers);
    for l in 0..num_layers {
        attn_norms.push(model.gguf.dequantize_tensor_alloc(
            model.gguf.tensor_or_err(&format!("blk.{}.attn_norm.weight", l)).unwrap()).unwrap());
        ffn_norms.push(model.gguf.dequantize_tensor_alloc(
            model.gguf.tensor_or_err(&format!("blk.{}.ffn_norm.weight", l)).unwrap()).unwrap());
    }
    let output_norm = model.gguf.dequantize_tensor_alloc(model.gguf.tensor_or_err("output_norm.weight").unwrap()).unwrap();

    let n_threads = 6;

    let tokens = dummy_tokens(args.prompt_tokens);
    println!("Benchmark: {} prompt tokens, {} repeats, {} threads", args.prompt_tokens, args.repeats, n_threads);

    for rep in 0..args.repeats {
        let mut kv_cache = swamp_engine::cache::PagedKVCache::new(
            num_layers, model.config.num_kv_heads, model.config.context_len.max(2048), head_dim);

        let t0 = Instant::now();
        #[cfg(feature = "gpu")]
        let mut gpu_state = init_gpu_state(&model, head_dim);
        let last_x = swamp_engine::executor::prefill_batch(
            num_layers, model.config.embed_dim, model.config.num_heads,
            model.config.num_kv_heads, head_dim, ffn_dim, 1e-5,
            &tokens, &token_embd, &attn_norms, &ffn_norms, &model, &mut kv_cache,
            #[cfg(feature = "gpu")] &mut gpu_state,
            n_threads);
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        {
            let mut x = last_x;
            let mut xn = vec![0.0f32; model.config.embed_dim];
            let mut q = vec![0.0f32; model.config.num_heads * head_dim];
            let mut k = vec![0.0f32; model.config.num_kv_heads * head_dim];
            let mut v = vec![0.0f32; model.config.num_kv_heads * head_dim];
            let mut ao = vec![0.0f32; model.config.embed_dim];
            let mut wo = vec![0.0f32; model.config.embed_dim];
            let mut fg = vec![0.0f32; ffn_dim];
            let mut fu = vec![0.0f32; ffn_dim];
            let mut fd = vec![0.0f32; model.config.embed_dim];
            let mut logits = vec![0.0f32; model.config.vocab_size];

            for l in 0..num_layers {
                let q_t = model.gguf.tensor_or_err(&format!("blk.{}.attn_q.weight", l))?;
                let k_t = model.gguf.tensor_or_err(&format!("blk.{}.attn_k.weight", l))?;
                let v_t = model.gguf.tensor_or_err(&format!("blk.{}.attn_v.weight", l))?;
                let o_t = model.gguf.tensor_or_err(&format!("blk.{}.attn_output.weight", l))?;
                let gt = model.gguf.tensor_or_err(&format!("blk.{}.ffn_gate.weight", l))?;
                let ut = model.gguf.tensor_or_err(&format!("blk.{}.ffn_up.weight", l))?;
                let dt = model.gguf.tensor_or_err(&format!("blk.{}.ffn_down.weight", l))?;

                swamp_engine::ops::rmsnorm(&mut xn, &x, &attn_norms[l], 1e-5);
                swamp_engine::linear::forward_linear_multi(
                    &model.gguf, &[q_t, k_t, v_t], &xn, &mut [&mut q, &mut k, &mut v], n_threads)?;
                swamp_engine::ops::apply_rope_ufc(&mut q, &mut k, args.prompt_tokens,
                    model.config.num_heads, model.config.num_kv_heads, head_dim, model.config.context_len);
                kv_cache.save(l, &k, &v);
                let seq_len = kv_cache.current_pos() + 1;
                swamp_engine::ops::attention(&mut ao, &q, &kv_cache, l, seq_len, args.prompt_tokens,
                    model.config.num_heads, model.config.num_kv_heads, head_dim);
                swamp_engine::linear::forward_linear(&model.gguf, o_t, &ao, &mut wo, n_threads)?;
                swamp_engine::ops::add_in_place(&mut x, &wo);
                swamp_engine::ops::rmsnorm(&mut xn, &x, &ffn_norms[l], 1e-5);
                swamp_engine::linear::forward_linear_multi(
                    &model.gguf, &[gt, ut], &xn, &mut [&mut fg, &mut fu], n_threads)?;
                swamp_engine::ops::silu(&mut fg);
                swamp_engine::ops::mul_in_place(&mut fg, &fu);
                swamp_engine::linear::forward_linear(&model.gguf, dt, &fg, &mut fd, n_threads)?;
                swamp_engine::ops::add_in_place(&mut x, &fd);
                kv_cache.advance();
            }
            swamp_engine::ops::rmsnorm(&mut xn, &x, &output_norm, 1e-5);
            swamp_engine::linear::forward_linear(
                &model.gguf, model.gguf.tensor_or_err("output.weight")?,
                &xn, &mut logits, n_threads)?;
        }
        let decode_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let auto_reg_est = decode_ms * args.prompt_tokens as f64;
        println!("--- Repeat {} ---", rep + 1);
        println!("  Prefill TTFT: {:.1}ms ({:.0} tok/s)", prefill_ms,
            args.prompt_tokens as f64 / (prefill_ms / 1000.0));
        println!("  Single decode: {:.1}ms", decode_ms);
        println!("  Est. auto-regressive {}toks: {:.1}s", args.prompt_tokens, auto_reg_est / 1000.0);
        println!("  Prefill speedup vs auto-regressive: {:.0}x", auto_reg_est / prefill_ms);
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn init_gpu_state(model: &swamp_engine::Model, head_dim: usize) -> Option<swamp_engine::scheduler::PerLayerGpuState> {
    use swamp_engine::scheduler::{PerLayerGpuState, GpuStreamManager};
    let n_heads = model.config.num_heads;
    let n_kv_heads = model.config.num_kv_heads;
    let max_seq = model.config.context_len.max(2048);
    let stream_mgr = GpuStreamManager::new(n_heads, n_kv_heads, max_seq, head_dim);
    let (gpu, d_k, d_v) = match stream_mgr {
        Some(gpu) => {
            if swamp_gpu::gpu_init().is_err() {
                eprintln!("  GPU init failed - falling back to CPU attention");
                (None, None, None)
            } else {
                let dk = swamp_gpu::gpu_alloc_kv_buffer_half(n_kv_heads, max_seq, head_dim).ok();
                let dv = swamp_gpu::gpu_alloc_kv_buffer_half(n_kv_heads, max_seq, head_dim).ok();
                match (dk, dv) {
                    (Some(dk_ptr), Some(dv_ptr)) => (Some(gpu), Some(dk_ptr), Some(dv_ptr)),
                    _ => {
                        eprintln!("  GPU buffer alloc failed - falling back to CPU attention");
                        (None, None, None)
                    }
                }
            }
        }
        None => (None, None, None),
    };
    gpu.map(|gpu| PerLayerGpuState {
        gpu: Box::new(gpu),
        d_k_buf: d_k.unwrap_or(std::ptr::null_mut()),
        d_v_buf: d_v.unwrap_or(std::ptr::null_mut()),
        max_seq_len: max_seq,
        n_kv_heads,
    })
}
