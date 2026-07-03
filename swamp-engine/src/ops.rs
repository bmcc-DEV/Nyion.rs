use std::sync::{OnceLock, Mutex};
use std::cell::RefCell;
use std::collections::HashMap;
use rayon::prelude::*;
use swamp_kernels::simd::{dot_product_raw, weighted_sum_raw};

thread_local! {
    static SCORES_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

const ROPE_THETA: f32 = 10000.0;

/// LUT de RoPE: mapeia head_dim -> Vec<[cos, sin]> (flat: [pos * half_dim + k][cos, sin])
static ROPE_LUTS: OnceLock<Mutex<HashMap<usize, Vec<[f32; 2]>>>> = OnceLock::new();

fn rope_luts() -> &'static Mutex<HashMap<usize, Vec<[f32; 2]>>> {
    ROPE_LUTS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn init_rope_lut(head_dim: usize, max_seq_len: usize) {
    let half_dim = head_dim / 2;
    let mut luts = rope_luts().lock().unwrap();
    if luts.contains_key(&head_dim) {
        return;
    }

    let mut lut = vec![[0.0f32; 2]; max_seq_len * half_dim];
    for m in 0..max_seq_len {
        for k in 0..half_dim {
            let freq = 1.0 / (ROPE_THETA.powf((2 * k) as f32 / head_dim as f32));
            let angle = m as f32 * freq;
            let idx = m * half_dim + k;
            lut[idx][0] = angle.cos();
            lut[idx][1] = angle.sin();
        }
    }
    luts.insert(head_dim, lut);
}

/// Aplica RoPE in-place para um token
#[inline(always)]
pub fn apply_rope_ufc(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_context: usize,
) {
    let half_dim = head_dim / 2;
    let luts = rope_luts().lock().unwrap();
    let lut = luts.get(&head_dim).expect("RoPE LUT not initialized for this head_dim");
    let pos = pos.min(max_context - 1);
    let lut_base = pos * half_dim;

    for h in 0..num_q_heads {
        let base = h * head_dim;
        apply_rotation_pair(&mut q[base..base + head_dim], &lut[lut_base..lut_base + half_dim], half_dim);
    }

    for h in 0..num_kv_heads {
        let base = h * head_dim;
        apply_rotation_pair(&mut k[base..base + head_dim], &lut[lut_base..lut_base + half_dim], half_dim);
    }
}

#[inline(always)]
fn apply_rotation_pair(vec: &mut [f32], cos_sin: &[[f32; 2]], half_dim: usize) {
    for k in 0..half_dim {
        let i = k * 2;
        let x = vec[i];
        let y = vec[i + 1];
        let cos = cos_sin[k][0];
        let sin = cos_sin[k][1];

        vec[i] = x * cos - y * sin;
        vec[i + 1] = x * sin + y * cos;
    }
}

pub fn attention(
    out: &mut [f32],
    q: &[f32],
    kv_cache: &mut crate::cache::PagedKVCache,
    layer_idx: usize,
    seq_len: usize,
    q_pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_rep = n_heads / n_kv_heads;
    let valid_end = q_pos.min(seq_len - 1);
    let block_size = kv_cache.block_size;

    // Pre-heat all pages that attention will touch
    let end_page = valid_end / block_size;
    kv_cache.ensure_pages_hot(0, end_page);

    out.par_chunks_mut(head_dim)
       .enumerate()
       .for_each(|(h, out_head)| {
            let kv_h = h / n_rep;
            let q_ptr = unsafe { q.as_ptr().add(h * head_dim) };
            let start_page = 0;
            let end_page = valid_end / block_size;

            SCORES_BUF.with(|buf| {
                let mut scores = buf.borrow_mut();
                if scores.len() < seq_len {
                    scores.resize(seq_len, 0.0);
                }
                let scores = &mut scores[..seq_len];

                // === SCORE LOOP (page by page) ===
                let mut max_score = f32::NEG_INFINITY;
                for pid in start_page..=end_page {
                    let page_start = pid * block_size;
                    let page_end = (page_start + block_size - 1).min(valid_end);

                    let k_page_base = kv_cache.k_page_ptr(layer_idx, kv_h, pid);

                    unsafe {
                        for t in page_start..=page_end {
                            #[cfg(target_arch = "x86_64")]
                            if t + 4 <= page_end {
                                std::arch::x86_64::_mm_prefetch(
                                    k_page_base.add((t + 4 - page_start) * head_dim) as *const i8,
                                    std::arch::x86_64::_MM_HINT_T0,
                                );
                            }
                            let k_ptr = k_page_base.add((t - page_start) * head_dim);
                            let score = dot_product_raw(q_ptr, k_ptr, head_dim) * scale;
                            scores[t] = score;
                            if score > max_score {
                                max_score = score;
                            }
                        }
                    }
                }

                // Mascara posicoes futuras (causal)
                for t in (valid_end + 1)..seq_len {
                    scores[t] = f32::NEG_INFINITY;
                }

                // Softmax
                let mut sum_exp = 0.0f32;
                for t in 0..seq_len {
                    let exp_score = (scores[t] - max_score).exp();
                    scores[t] = exp_score;
                    sum_exp += exp_score;
                }
                let inv_sum = 1.0 / sum_exp;
                for t in 0..seq_len {
                    scores[t] *= inv_sum;
                }

                // === WEIGHTED SUM LOOP (page by page) ===
                unsafe {
                    let out_ptr = out_head.as_mut_ptr();
                    std::ptr::write_bytes(out_ptr, 0, head_dim);
                    for pid in start_page..=end_page {
                        let page_start = pid * block_size;
                        let page_end = (page_start + block_size - 1).min(valid_end);

                        let v_page_base = kv_cache.v_page_ptr(layer_idx, kv_h, pid);

                        for t in page_start..=page_end {
                            #[cfg(target_arch = "x86_64")]
                            if t + 4 <= page_end {
                                std::arch::x86_64::_mm_prefetch(
                                    v_page_base.add((t + 4 - page_start) * head_dim) as *const i8,
                                    std::arch::x86_64::_MM_HINT_T0,
                                );
                            }
                            let v_ptr = v_page_base.add((t - page_start) * head_dim);
                            weighted_sum_raw(out_ptr, v_ptr, scores[t], head_dim);
                        }
                    }
                }
            });
        });
}

pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let n = x.len();
    let mut ss = 0.0f32;
    for i in 0..n {
        ss += x[i] * x[i];
    }
    ss /= n as f32;
    ss += eps;
    let inv_rms = 1.0 / ss.sqrt();

    for i in 0..n {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

pub fn silu(x: &mut [f32]) {
    for i in 0..x.len() {
        let val = x[i];
        x[i] = val / (1.0 + (-val).exp());
    }
}

pub fn mul_in_place(a: &mut [f32], b: &[f32]) {
    for i in 0..a.len() {
        a[i] *= b[i];
    }
}

pub fn add_in_place(a: &mut [f32], b: &[f32]) {
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

/// GPU-accelerated attention (falls back to CPU if GPU unavailable)
#[cfg(feature = "gpu")]
pub fn gpu_attention_forward(
    out: &mut [f32],
    q: &[f32],
    kv_cache: &mut crate::cache::PagedKVCache,
    layer_idx: usize,
    seq_len: usize,
    _q_pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    // Allocate contiguous K and V buffers
    let kv_size = n_kv_heads * seq_len * head_dim;
    let mut k_buf = vec![0.0f32; kv_size];
    let mut v_buf = vec![0.0f32; kv_size];

    tracing::debug!("GPU attention: seq_len={} heads={}/{} head_dim={}",
        seq_len, n_heads, n_kv_heads, head_dim);

    // Copy K and V from page cache into contiguous buffers
    for kv_h in 0..n_kv_heads {
        for t in 0..seq_len {
            let src_k = kv_cache.k_page_ptr(layer_idx, kv_h, t / kv_cache.block_size);
            let src_v = kv_cache.v_page_ptr(layer_idx, kv_h, t / kv_cache.block_size);
            let slot = t % kv_cache.block_size;
            unsafe {
                let dst_off = (kv_h * seq_len + t) * head_dim;
                std::ptr::copy_nonoverlapping(
                    src_k.add(slot * head_dim),
                    k_buf.as_mut_ptr().add(dst_off),
                    head_dim,
                );
                std::ptr::copy_nonoverlapping(
                    src_v.add(slot * head_dim),
                    v_buf.as_mut_ptr().add(dst_off),
                    head_dim,
                );
            }
        }
    }

    // Call GPU attention
    if let Err(e) = swamp_gpu::gpu_attention_forward(q, &k_buf, &v_buf, out, n_heads, n_kv_heads, seq_len, head_dim) {
        tracing::warn!("GPU attention failed, falling back to CPU: {:?}", e);
        attention(out, q, kv_cache, layer_idx, seq_len, _q_pos, n_heads, n_kv_heads, head_dim);
    }
}
