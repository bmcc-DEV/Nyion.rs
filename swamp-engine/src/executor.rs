// swamp-engine/src/executor.rs
// Executor: laço de inferência de tokens que envia resultados via mpsc channel

use crate::model::Model;

use crate::sampler::Sampler;
use crate::thermal::ThermalCoordinator;
use crate::lsc::LscPrefetcher;
use tokio::sync::mpsc::Sender;
use std::sync::Arc;
use crate::cache::PagedKVCache;

pub struct InferenceRequest {
    pub prompt: Option<String>,
    pub messages: Option<Vec<crate::chat_template::ChatMessage>>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

use rayon::ThreadPool;
use std::sync::OnceLock;

pub static RAYON_POOL: OnceLock<ThreadPool> = OnceLock::new();

fn get_rayon_pool() -> &'static ThreadPool {
    RAYON_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(6)
            .thread_name(|i| format!("swamp-vnni-{}", i))
            .build()
            .expect("failed to build Rayon thread pool")
    })
}

/// Batched prefill: process all prompt tokens simultaneously per layer.
///
/// For each layer:
///   1. Batched RMSNorm + QKV linear (all positions)
///   2. RoPE + KV cache store (per position)
///   3. GPU attention loop (sequential per position)
///   4. Batched output projection + FFN (all positions)
///
/// Returns the last token's hidden state (x) for decode to continue from.
#[allow(unused_variables)]
pub fn prefill_batch(
    num_layers: usize,
    embed_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    ffn_dim: usize,
    rms_eps: f32,
    prompt_tokens: &[usize],
    token_embd: &[f32],
    attn_norms: &[Vec<f32>],
    ffn_norms: &[Vec<f32>],
    model: &Model,
    kv_cache: &mut PagedKVCache,
    #[cfg(feature = "gpu")] per_layer_gpu: &mut Option<crate::scheduler::PerLayerGpuState>,
    n_threads: usize,
) -> Vec<f32> {
    use crate::linear::{forward_linear_batch};
    use crate::ops::{rmsnorm, silu, add_in_place, mul_in_place, apply_rope_ufc};
    use rayon::prelude::*;

    let batch = prompt_tokens.len();

    // Pre-allocate batched buffers
    let mut xs: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.0f32; embed_dim]).collect();
    let mut x_norms: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.0f32; embed_dim]).collect();
    let mut qs: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.0f32; num_heads * head_dim]).collect();
    let mut ks: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.0f32; num_kv_heads * head_dim]).collect();
    let mut vs: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.0f32; num_kv_heads * head_dim]).collect();
    let mut attn_outs: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.0f32; embed_dim]).collect();
    let mut wo_outs: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.0f32; embed_dim]).collect();
    let mut ffn_gates: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.0f32; ffn_dim]).collect();
    let mut ffn_ups: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.0f32; ffn_dim]).collect();
    let mut ffn_downs: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.0f32; embed_dim]).collect();

    // Embed all prompt tokens
    xs.par_iter_mut().enumerate().for_each(|(i, x)| {
        let tok = prompt_tokens[i] as usize;
        let embd_start = tok * embed_dim;
        x.copy_from_slice(&token_embd[embd_start..embd_start + embed_dim]);
    });

    for l in 0..num_layers {
        let attn_w = &attn_norms[l];
        let ffn_w = &ffn_norms[l];

        // 1. Batched RMSNorm + QKV linear
        {
            x_norms.par_iter_mut().zip(xs.par_iter()).for_each(|(xn, x)| {
                rmsnorm(xn, x, attn_w, rms_eps);
            });

            let q_t = model.gguf.tensor_or_err(&format!("blk.{}.attn_q.weight", l)).unwrap();
            let k_t = model.gguf.tensor_or_err(&format!("blk.{}.attn_k.weight", l)).unwrap();
            let v_t = model.gguf.tensor_or_err(&format!("blk.{}.attn_v.weight", l)).unwrap();

            let xn_refs: Vec<&[f32]> = x_norms.iter().map(|v| v.as_slice()).collect();
            let mut q_refs: Vec<&mut [f32]> = qs.iter_mut().map(|v| v.as_mut_slice()).collect();
            forward_linear_batch(&model.gguf, q_t, &xn_refs, &mut q_refs, n_threads).ok();
            let mut k_refs: Vec<&mut [f32]> = ks.iter_mut().map(|v| v.as_mut_slice()).collect();
            forward_linear_batch(&model.gguf, k_t, &xn_refs, &mut k_refs, n_threads).ok();
            let mut v_refs: Vec<&mut [f32]> = vs.iter_mut().map(|v| v.as_mut_slice()).collect();
            forward_linear_batch(&model.gguf, v_t, &xn_refs, &mut v_refs, n_threads).ok();
        }

        // 2. RoPE + KV cache store per position
        for t in 0..batch {
            apply_rope_ufc(&mut qs[t], &mut ks[t], t, num_heads, num_kv_heads, head_dim, model.config.context_len);
            kv_cache.save_at(l, t, &ks[t], &vs[t]);
        }

        // 3. Attention loop (sequential per position)
        let use_gpu = {
            #[cfg(feature = "gpu")]
            { per_layer_gpu.is_some() }
            #[cfg(not(feature = "gpu"))]
            { false }
        };
        if use_gpu {
            #[cfg(feature = "gpu")]
            if let Some(gpu) = per_layer_gpu {
                for t in 0..batch {
                    let seq_len = t + 1;
                    gpu.execute_attention_async(&qs[t], &ks[t], &vs[t], &mut attn_outs[t], t, seq_len);
                    gpu.sync();
                }
            }
        } else {
            for t in 0..batch {
                let seq_len = t + 1;
                crate::ops::attention(&mut attn_outs[t], &qs[t], kv_cache, l, seq_len, t, num_heads, num_kv_heads, head_dim);
            }
        }

        // 4. Batched output projection + residual
        {
            let o_t = model.gguf.tensor_or_err(&format!("blk.{}.attn_output.weight", l)).unwrap();
            let attn_refs: Vec<&[f32]> = attn_outs.iter().map(|v| v.as_slice()).collect();
            let mut wo_refs: Vec<&mut [f32]> = wo_outs.iter_mut().map(|v| v.as_mut_slice()).collect();
            forward_linear_batch(&model.gguf, o_t, &attn_refs, &mut wo_refs, n_threads).ok();
        }

        xs.par_iter_mut().zip(wo_outs.par_iter()).for_each(|(x, wo)| {
            add_in_place(x, wo);
        });

        // 5. Batched FFN
        {
            x_norms.par_iter_mut().zip(xs.par_iter()).for_each(|(xn, x)| {
                rmsnorm(xn, x, ffn_w, rms_eps);
            });

            let xn_refs: Vec<&[f32]> = x_norms.iter().map(|v| v.as_slice()).collect();

            let gate_t = model.gguf.tensor_or_err(&format!("blk.{}.ffn_gate.weight", l)).unwrap();
            let up_t = model.gguf.tensor_or_err(&format!("blk.{}.ffn_up.weight", l)).unwrap();

            let mut gate_refs: Vec<&mut [f32]> = ffn_gates.iter_mut().map(|v| v.as_mut_slice()).collect();
            forward_linear_batch(&model.gguf, gate_t, &xn_refs, &mut gate_refs, n_threads).ok();
            let mut up_refs: Vec<&mut [f32]> = ffn_ups.iter_mut().map(|v| v.as_mut_slice()).collect();
            forward_linear_batch(&model.gguf, up_t, &xn_refs, &mut up_refs, n_threads).ok();
        }

        // Silu + mul per position
        for t in 0..batch {
            silu(&mut ffn_gates[t]);
            mul_in_place(&mut ffn_gates[t], &ffn_ups[t]);
        }

        let down_t = model.gguf.tensor_or_err(&format!("blk.{}.ffn_down.weight", l)).unwrap();
        let ffn_gate_refs: Vec<&[f32]> = ffn_gates.iter().map(|v| v.as_slice()).collect();
        let mut down_refs: Vec<&mut [f32]> = ffn_downs.iter_mut().map(|v| v.as_mut_slice()).collect();
        forward_linear_batch(&model.gguf, down_t, &ffn_gate_refs, &mut down_refs, n_threads).ok();

        // Residual add
        xs.par_iter_mut().zip(ffn_downs.par_iter()).for_each(|(x, fd)| {
            add_in_place(x, fd);
        });
    }

    xs.into_iter().last().unwrap_or_default()
}

pub struct ModelExecutor {
    pub model: Arc<Model>,
    pub thermal_coordinator: ThermalCoordinator,
    pub prefetcher: LscPrefetcher,
    // Tensor cache for non-linear operations (RMSNorm weights and embeddings)
    pub token_embd: Vec<f32>,
    pub attn_norms: Vec<Vec<f32>>,
    pub ffn_norms: Vec<Vec<f32>>,
    pub output_norm: Vec<f32>,
}

impl ModelExecutor {
    pub fn new(model: Arc<Model>) -> Self {
        let head_dim = model.config.embed_dim / model.config.num_heads;
        crate::ops::init_rope_lut(head_dim, model.config.context_len);
        let thermal_coordinator = ThermalCoordinator::default();
        let prefetcher = LscPrefetcher::default();

        // Carga de pesos 1D para F32 (normas e embeddings)
        let token_embd = model.gguf.dequantize_tensor_alloc(model.gguf.tensor_or_err("token_embd.weight").unwrap()).unwrap();
        
        let num_layers = model.config.num_layers;
        let mut attn_norms = Vec::with_capacity(num_layers);
        let mut ffn_norms = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            let attn_norm = model.gguf.dequantize_tensor_alloc(model.gguf.tensor_or_err(&format!("blk.{}.attn_norm.weight", l)).unwrap()).unwrap();
            let ffn_norm = model.gguf.dequantize_tensor_alloc(model.gguf.tensor_or_err(&format!("blk.{}.ffn_norm.weight", l)).unwrap()).unwrap();
            attn_norms.push(attn_norm);
            ffn_norms.push(ffn_norm);
        }
        let output_norm = model.gguf.dequantize_tensor_alloc(model.gguf.tensor_or_err("output_norm.weight").unwrap()).unwrap();

        Self {
            model,
            thermal_coordinator,
            prefetcher,
            token_embd,
            attn_norms,
            ffn_norms,
            output_norm,
        }
    }

    pub async fn generate(&mut self, req: InferenceRequest, tx: Sender<String>) -> anyhow::Result<()> {
        use std::time::Instant;

        let sampler = Sampler::new(req.temperature, req.top_k, req.top_p);
        let prompt_str = if let Some(msgs) = &req.messages {
            crate::chat_template::format_chat_prompt(msgs)
        } else if let Some(p) = &req.prompt {
            p.clone()
        } else {
            String::new()
        };

        let prompt_tokens = if prompt_str.is_empty() {
            vec![1usize]
        } else {
            crate::tokenizer::encode(&prompt_str)
        };

        println!("DEBUG: Formatted prompt:\n{}\nTokens: {:?}", prompt_str, prompt_tokens);

        let num_layers = self.model.config.num_layers;
        let embed_dim = self.model.config.embed_dim;
        let num_heads = self.model.config.num_heads;
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = embed_dim / num_heads;
        let rms_eps = 1e-5;

        // Determine FFN dim from the first layer's gate weight
        let ffn_gate_t = self.model.gguf.tensor_or_err("blk.0.ffn_gate.weight")?;
        let ffn_dim = ffn_gate_t.shape[1] as usize;

        // Clone references to read-only buffers so we can move them into spawn_blocking
        let token_embd = self.token_embd.clone();
        let attn_norms = self.attn_norms.clone();
        let ffn_norms = self.ffn_norms.clone();
        let output_norm = self.output_norm.clone();
        let model = self.model.clone();

        // Determina numero de threads UMA vez antes de entrar no blocking, baseado na coerencia (LSC)
        let c_epsilon = self.thermal_coordinator.coherence();
        let n_threads = if self.thermal_coordinator.is_throttling() || c_epsilon < 0.6 {
            1
        } else if c_epsilon < 0.86 {
            2
        } else {
            6
        };

        let result = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            use crate::linear::{forward_linear, forward_linear_multi};
            use crate::ops::{rmsnorm, silu, add_in_place, mul_in_place, apply_rope_ufc};

            // Per-layer profiler (HLC timestamps)
            let mut profiler = crate::hlc::ProfileSink::new();

            // Buffers de estado (Forward Pass) locais para a thread bloqueante
            let mut x = vec![0.0f32; embed_dim];
            let mut x_norm = vec![0.0f32; embed_dim];
            let mut q = vec![0.0f32; num_heads * head_dim];
            let mut k = vec![0.0f32; num_kv_heads * head_dim];
            let mut v = vec![0.0f32; num_kv_heads * head_dim];
            let mut attn_out = vec![0.0f32; embed_dim];
            let mut ffn_gate = vec![0.0f32; ffn_dim];
            let mut ffn_up = vec![0.0f32; ffn_dim];
            let mut ffn_down = vec![0.0f32; embed_dim];
            let mut logits = vec![0.0f32; model.config.vocab_size];
            let mut wo_out = vec![0.0f32; embed_dim];

            // KV Cache Paginado
            let mut kv_cache = crate::cache::PagedKVCache::new(
                num_layers,
                num_kv_heads,
                model.config.context_len.max(2048),
                head_dim,
            );

            // M:N Scheduler — GPU stream manager + persistent K/V buffers
            #[cfg(feature = "gpu")]
            let mut per_layer_gpu: Option<crate::scheduler::PerLayerGpuState> = {
                let max_seq = model.config.context_len.max(2048);
                let stream_mgr = crate::scheduler::GpuStreamManager::new(
                    num_heads, num_kv_heads, max_seq, head_dim,
                );
                let (gpu, d_k, d_v) = match stream_mgr {
                    Some(gpu) => {
                        if swamp_gpu::gpu_init().is_err() {
                            tracing::warn!("GPU init failed (falling back to CPU attention)");
                            (None, None, None)
                        } else {
                            let dk = swamp_gpu::gpu_alloc_kv_buffer_half(num_kv_heads, max_seq, head_dim).ok();
                            let dv = swamp_gpu::gpu_alloc_kv_buffer_half(num_kv_heads, max_seq, head_dim).ok();
                            match (dk, dv) {
                                (Some(dk_ptr), Some(dv_ptr)) => (Some(gpu), Some(dk_ptr), Some(dv_ptr)),
                                _ => {
                                    tracing::warn!("GPU KV buffer alloc failed (falling back to CPU attention)");
                                    (None, None, None)
                                }
                            }
                        }
                    }
                    None => (None, None, None),
                };
                gpu.map(|gpu| crate::scheduler::PerLayerGpuState {
                    gpu: Box::new(gpu),
                    d_k_buf: d_k.unwrap_or(std::ptr::null_mut()),
                    d_v_buf: d_v.unwrap_or(std::ptr::null_mut()),
                    max_seq_len: max_seq,
                    n_kv_heads: num_kv_heads,
                })
            };

            // PrefetchEngine for madvise-based page prefetch
            let (mmap_ptr, mmap_len) = model.gguf.mmap_ptr_and_len();
            let prefetch_engine = crate::prefetch::PrefetchEngine::new(mmap_ptr, mmap_len);

            let mut tokens = prompt_tokens.clone();

            let mut tokens_generated = 0;
            let t_start = Instant::now();

            // Run in a dedicated Rayon pool
            get_rayon_pool().install(|| -> anyhow::Result<()> {
                // Batched prefill for prompt tokens (skip auto-regressive loop)
                #[allow(unused_mut)]
                let mut prefill_pos = 0;
                if prompt_tokens.len() > 1 {
                    let n_prompt = prompt_tokens.len();
                    let last_x = prefill_batch(
                        num_layers, embed_dim, num_heads, num_kv_heads, head_dim, ffn_dim, rms_eps as f32,
                        &prompt_tokens, &token_embd, &attn_norms, &ffn_norms, &model, &mut kv_cache,
                        #[cfg(feature = "gpu")] &mut per_layer_gpu,
                        n_threads,
                    );
                    // Advance KV cache position past prefill
                    for _ in 0..n_prompt - 1 {
                        kv_cache.advance();
                    }
                    // Final RMSNorm + logits + sample from last prompt token
                    rmsnorm(&mut x_norm, &last_x, &output_norm, rms_eps);
                    forward_linear(
                        &model.gguf, model.gguf.tensor_or_err("output.weight").unwrap(),
                        &x_norm, &mut logits, n_threads,
                    )?;
                    let next_token = sampler.sample(&logits);
                    tokens.push(next_token);
                    let next_word = crate::tokenizer::decode(&[next_token]);
                    let _ = tx.blocking_send(next_word);
                    tokens_generated += 1;
                    kv_cache.advance(); // advance past the last prompt token

                    prefill_pos = n_prompt;
                    tracing::info!("Prefill done: {} tokens in {:.2}s", n_prompt, t_start.elapsed().as_secs_f64());
                }

                let total_steps = req.max_tokens + prompt_tokens.len() - 1;
                for step in prefill_pos..total_steps {
                    let token_id = if step < prompt_tokens.len() {
                        prompt_tokens[step]
                    } else {
                        *tokens.last().unwrap()
                    };
                    let pos = step;

                    // Embedding lookup
                    let embd_start = token_id * embed_dim;
                    x.copy_from_slice(&token_embd[embd_start..embd_start + embed_dim]);

                    // Forward Pass: Camadas
                    for l in 0..num_layers {
                        #[cfg(feature = "gpu")]
                        let mut layer_profile = profiler.begin_layer(l, per_layer_gpu.is_some());
                        #[cfg(not(feature = "gpu"))]
                        let mut layer_profile = profiler.begin_layer(l, false);

                        // RMSNorm Attn
                        rmsnorm(&mut x_norm, &x, &attn_norms[l], rms_eps);

                        // QKV Projections FUNDIDAS (x_norm lido 1×)
                        let q_t = model.gguf.tensor_or_err(&format!("blk.{}.attn_q.weight", l))?;
                        let k_t = model.gguf.tensor_or_err(&format!("blk.{}.attn_k.weight", l))?;
                        let v_t = model.gguf.tensor_or_err(&format!("blk.{}.attn_v.weight", l))?;
                        forward_linear_multi(
                            &model.gguf,
                            &[q_t, k_t, v_t],
                            &x_norm,
                            &mut [&mut q, &mut k, &mut v],
                            n_threads,
                        )?;

                        // RoPE
                        apply_rope_ufc(&mut q, &mut k, pos, num_heads, num_kv_heads, head_dim, model.config.context_len);

                        // KV Cache (CPU)
                        kv_cache.save(l, &k, &v);
                        let seq_len = kv_cache.current_pos() + 1;

                        // Attention: async stream-based GPU path via M:N scheduler
                        #[cfg(feature = "gpu")]
                        {
                            let gpu_ok = per_layer_gpu.as_mut().map_or(false, |gpu| {
                                let gpu_tic = profiler.clock().now();
                                let launched = gpu.execute_attention_async(
                                    &q, &k, &v, &mut attn_out, pos, seq_len,
                                );
                                if !launched {
                                    return false;
                                }
                                let synced = gpu.sync();
                                let gpu_toc = profiler.clock().now();
                                if synced {
                                    let gpu_ms = (gpu_toc.wall.saturating_sub(gpu_tic.wall)) as f64 / 1_000_000.0;
                                    profiler.record_gpu_attention(&mut layer_profile, gpu_ms);
                                }
                                synced
                            });
                            if !gpu_ok {
                                crate::ops::attention(&mut attn_out, &q, &kv_cache, l, seq_len, pos, num_heads, num_kv_heads, head_dim);
                            }
                        }
                        #[cfg(not(feature = "gpu"))]
                        crate::ops::attention(&mut attn_out, &q, &kv_cache, l, seq_len, pos, num_heads, num_kv_heads, head_dim);

                        // Output Projection
                        forward_linear(&model.gguf, model.gguf.tensor_or_err(&format!("blk.{}.attn_output.weight", l))?, &attn_out, &mut wo_out, n_threads)?;
                        add_in_place(&mut x, &wo_out);

                        // RMSNorm FFN
                        rmsnorm(&mut x_norm, &x, &ffn_norms[l], rms_eps);

                        // FFN Gate & Up FUNDIDOS (x_norm lido 1×)
                        let gate_t = model.gguf.tensor_or_err(&format!("blk.{}.ffn_gate.weight", l))?;
                        let up_t = model.gguf.tensor_or_err(&format!("blk.{}.ffn_up.weight", l))?;
                        forward_linear_multi(
                            &model.gguf,
                            &[gate_t, up_t],
                            &x_norm,
                            &mut [&mut ffn_gate, &mut ffn_up],
                            n_threads,
                        )?;

                        silu(&mut ffn_gate);
                        mul_in_place(&mut ffn_gate, &ffn_up);

                        // FFN Down
                        forward_linear(&model.gguf, model.gguf.tensor_or_err(&format!("blk.{}.ffn_down.weight", l))?, &ffn_gate, &mut ffn_down, n_threads)?;
                        add_in_place(&mut x, &ffn_down);

                        // Prefetch next layer's tensors (madvise WILLNEED)
                        let next = l + 1;
                        if next < num_layers {
                            for tensor_name in &[
                                format!("blk.{}.attn_q.weight", next),
                                format!("blk.{}.attn_k.weight", next),
                                format!("blk.{}.attn_v.weight", next),
                                format!("blk.{}.attn_output.weight", next),
                                format!("blk.{}.ffn_gate.weight", next),
                                format!("blk.{}.ffn_up.weight", next),
                                format!("blk.{}.ffn_down.weight", next),
                            ] {
                                if let Some((off, len)) = model.gguf.tensor_raw_offset_len(tensor_name) {
                                    prefetch_engine.prefetch_range(off, len);
                                }
                            }
                        }

                        profiler.end_layer(layer_profile);
                    }

                    // RMSNorm Final
                    rmsnorm(&mut x_norm, &x, &output_norm, rms_eps);

                    // Logits
                    forward_linear(&model.gguf, model.gguf.tensor_or_err("output.weight")?, &x_norm, &mut logits, n_threads)?;

                    // Amostragem (só depois de preencher o cache com o prompt)
                    if step >= prompt_tokens.len() - 1 {
                        let next_token = sampler.sample(&logits);
                        tokens.push(next_token);

                        let next_word = crate::tokenizer::decode(&[next_token]);
                        
                        if let Err(_) = tx.blocking_send(next_word) {
                            break; // Canal fechado
                        }

                        tokens_generated += 1;
                    }
                    
                    kv_cache.advance();
                }
                
                let elapsed = t_start.elapsed().as_secs_f64();

                // Profiler summary
                if !profiler.is_empty() {
                    let _ = tx.blocking_send(format!("\n—— HLC Profile ——"));
                    let report = profiler.report();
                    let _ = tx.blocking_send(report);
                }

                if elapsed > 0.0 {
                    let _ = tx.blocking_send(format!("\n\n[Tokens: {} | Throughput: {:.2} tok/s]", tokens_generated, tokens_generated as f64 / elapsed));
                }
                Ok(())
            })
        }).await;

        match result {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(e)) => Err(e),
            Err(e) => Err(anyhow::anyhow!("Join Error: {:?}", e)),
        }
    }
}
