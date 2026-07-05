use std::time::Instant;
use clap::Parser;
use anyhow::Result;
use swamp_engine::Model;
use swamp_engine::linear::forward_gemvs_ring;
use swamp_gpu::{gpu_init, gpu_stream_create, gpu_upload_weights, gpu_swamp_init, gpu_swamp_launch, gpu_swamp_enqueue, gpu_swamp_shutdown, gpu_copy_to_device, gpu_copy_to_host};

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

    let mut _use_gpu = false;
    let mut d_w: *mut u8 = std::ptr::null_mut();
    let mut d_ring: *mut std::ffi::c_void = std::ptr::null_mut();
    let mut d_state: *mut f32 = std::ptr::null_mut();
    let mut d_shutdown: *mut i32 = std::ptr::null_mut();
    let mut stream = unsafe { std::mem::zeroed() };
    if let Ok(_) = gpu_init() {
        if let Ok(s) = gpu_stream_create() {
            stream = s;
            let total_w: usize = model.layer_rings.iter().map(|r| r.ring.len()).sum();
            let mut all_w: Vec<u8> = Vec::with_capacity(total_w);
            for r in &model.layer_rings { all_w.extend_from_slice(&r.ring); }
            if gpu_upload_weights(all_w.as_ptr(), &mut d_w, total_w, stream).is_ok() {
                unsafe { let _ = gpu_swamp_init(&mut d_ring, &mut d_state, &mut d_shutdown, stream); }
                unsafe { let _ = gpu_swamp_launch(d_ring, d_w, d_state, d_shutdown, stream); }
                println!("  Swamp Continuum: pesos ({} MB), ring, state buffer", total_w / (1024*1024));
                _use_gpu = true;
            }
        }
    }
    if !_use_gpu { eprintln!("  GPU not available — using CPU"); }

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

        // 10 decode steps com ring dispatch (pre-quant kernel + MSR unlock)
        let decode_steps = 10usize;
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
        let mut pos = args.prompt_tokens;

        // Pre-compute layer sizes for GPU offset calculation
        let mut layer_sizes: Vec<usize> = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            let r = &model.layer_rings[l];
            layer_sizes.push(r.ring.len());
        }

        let t1 = Instant::now();
        for _step in 0..decode_steps {
            for l in 0..num_layers {
                let ring = &model.layer_rings[l];
                swamp_engine::ops::rmsnorm(&mut xn, &x, &attn_norms[l], 1e-5);
                if _use_gpu {
                    let layer_off: usize = layer_sizes[..l].iter().sum();
                    let w_base = (d_w as usize + layer_off) as i32;
                    let embed_bytes = (model.config.embed_dim * 4) as usize;
                    let q_bytes = (model.config.num_heads * head_dim * 4) as usize;
                    let kv_bytes = (model.config.num_kv_heads * head_dim * 4) as usize;
                    let _ = gpu_copy_to_device(d_state as *mut _, xn.as_ptr() as *const _, embed_bytes);
                    let _ = gpu_swamp_enqueue(d_ring, 0, l as i32, 0, w_base + ring.q_off as i32, 0x10000, ring.q_nr as i32, (ring.q_nc / 256) as i32, stream);
                    let _ = gpu_swamp_enqueue(d_ring, 0, l as i32, 0, w_base + ring.k_off as i32, 0x20000, ring.k_nr as i32, (ring.k_nc / 256) as i32, stream);
                    let _ = gpu_swamp_enqueue(d_ring, 0, l as i32, 0, w_base + ring.v_off as i32, 0x30000, ring.v_nr as i32, (ring.v_nc / 256) as i32, stream);
                    let _ = gpu_copy_to_host(q.as_mut_ptr() as *mut _, (d_state as usize + 0x10000) as *const _, q_bytes);
                    let _ = gpu_copy_to_host(k.as_mut_ptr() as *mut _, (d_state as usize + 0x20000) as *const _, kv_bytes);
                    let _ = gpu_copy_to_host(v.as_mut_ptr() as *mut _, (d_state as usize + 0x30000) as *const _, kv_bytes);
                } else {
                    swamp_engine::linear::forward_gemvs_ring(&mut [
                        (ring.q_slice(), &xn, &mut q, ring.q_nr, ring.q_nc, ring.q_bs),
                        (ring.k_slice(), &xn, &mut k, ring.k_nr, ring.k_nc, ring.k_bs),
                        (ring.v_slice(), &xn, &mut v, ring.v_nr, ring.v_nc, ring.v_bs),
                    ], n_threads);
                }
                swamp_engine::ops::apply_rope_ufc(&mut q, &mut k, pos,
                    model.config.num_heads, model.config.num_kv_heads, head_dim, model.config.context_len);
                kv_cache.save(l, &k, &v);
                let seq_len = pos + 1;
                swamp_engine::ops::attention(&mut ao, &q, &mut kv_cache, l, seq_len, pos,
                    model.config.num_heads, model.config.num_kv_heads, head_dim);
                if _use_gpu {
                    let layer_off: usize = layer_sizes[..l].iter().sum();
                    let w_base = (d_w as usize + layer_off) as i32;
                    let embed_bytes = (model.config.embed_dim * 4) as usize;
                    let _ = gpu_copy_to_device((d_state as usize + 0x40000) as *mut _, ao.as_ptr() as *const _, embed_bytes);
                    let _ = gpu_swamp_enqueue(d_ring, 0, l as i32, 0x40000, w_base + ring.o_off as i32, 0x50000, ring.o_nr as i32, (ring.o_nc / 256) as i32, stream);
                    let _ = gpu_copy_to_host(wo.as_mut_ptr() as *mut _, (d_state as usize + 0x50000) as *const _, embed_bytes);
                } else {
                    swamp_engine::linear::forward_gemvs_ring(&mut [
                        (ring.o_slice(), &ao, &mut wo, ring.o_nr, ring.o_nc, ring.o_bs),
                    ], n_threads);
                }
                swamp_engine::ops::add_in_place(&mut x, &wo);
                swamp_engine::ops::rmsnorm(&mut xn, &x, &ffn_norms[l], 1e-5);
                if _use_gpu {
                    let layer_off: usize = layer_sizes[..l].iter().sum();
                    let w_base = (d_w as usize + layer_off) as i32;
                    let embed_bytes = (model.config.embed_dim * 4) as usize;
                    let ffn_bytes = (ffn_dim * 4) as usize;
                    let _ = gpu_copy_to_device((d_state as usize + 0x60000) as *mut _, xn.as_ptr() as *const _, embed_bytes);
                    let _ = gpu_swamp_enqueue(d_ring, 0, l as i32, 0x60000, w_base + ring.gate_off as i32, 0x70000, ring.gate_nr as i32, (ring.gate_nc / 256) as i32, stream);
                    let _ = gpu_swamp_enqueue(d_ring, 0, l as i32, 0x60000, w_base + ring.up_off as i32, 0x80000, ring.up_nr as i32, (ring.up_nc / 256) as i32, stream);
                    let _ = gpu_copy_to_host(fg.as_mut_ptr() as *mut _, (d_state as usize + 0x70000) as *const _, ffn_bytes);
                    let _ = gpu_copy_to_host(fu.as_mut_ptr() as *mut _, (d_state as usize + 0x80000) as *const _, ffn_bytes);
                } else {
                    swamp_engine::linear::forward_gemvs_ring(&mut [
                        (ring.gate_slice(), &xn, &mut fg, ring.gate_nr, ring.gate_nc, ring.gate_bs),
                        (ring.up_slice(), &xn, &mut fu, ring.up_nr, ring.up_nc, ring.up_bs),
                    ], n_threads);
                }
                swamp_engine::ops::silu(&mut fg);
                swamp_engine::ops::mul_in_place(&mut fg, &fu);
                if _use_gpu {
                    let layer_off: usize = layer_sizes[..l].iter().sum();
                    let w_base = (d_w as usize + layer_off) as i32;
                    let ffn_bytes = (ffn_dim * 4) as usize;
                    let embed_bytes = (model.config.embed_dim * 4) as usize;
                    let _ = gpu_copy_to_device((d_state as usize + 0x90000) as *mut _, fg.as_ptr() as *const _, ffn_bytes);
                    let _ = gpu_swamp_enqueue(d_ring, 0, l as i32, 0x90000, w_base + ring.down_off as i32, 0xA0000, ring.down_nr as i32, (ring.down_nc / 256) as i32, stream);
                    let _ = gpu_copy_to_host(fd.as_mut_ptr() as *mut _, (d_state as usize + 0xA0000) as *const _, embed_bytes);
                } else {
                    swamp_engine::linear::forward_gemvs_ring(&mut [
                        (ring.down_slice(), &fg, &mut fd, ring.down_nr, ring.down_nc, ring.down_bs),
                    ], n_threads);
                }
                swamp_engine::ops::add_in_place(&mut x, &fd);
                kv_cache.advance();
            }
            pos += 1;
            x.copy_from_slice(&token_embd[1 * model.config.embed_dim..(1 + 1) * model.config.embed_dim]);
        }
        let decode_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let per_token = decode_ms / decode_steps as f64;
        println!("--- Repeat {} (ring dispatch) ---", rep + 1);
        println!("  Prefill TTFT: {:.1}ms ({:.0} tok/s)", prefill_ms,
            args.prompt_tokens as f64 / (prefill_ms / 1000.0));
        println!("  Decode: {:.1}ms for {} tokens ({:.1} ms/tok, {:.0} tok/s)",
            decode_ms, decode_steps, per_token, decode_steps as f64 / (decode_ms / 1000.0));
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
