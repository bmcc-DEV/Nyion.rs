// swamp-tensors/src/ops.rs
// Operacoes CPU para CpuTensor (matmul, softmax, RoPE, RMSNorm) otimizadas com SIMD AVX-512 / VNNI / FMA

use crate::tensor::CpuTensor;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Structs de blocos GGUF para FFI e descompactacao direta
#[repr(C)]
pub struct BlockQ8_0 {
    pub d: [u8; 2],
    pub qs: [i8; 32],
}

#[repr(C)]
pub struct BlockQ4K {
    pub d: [u8; 2],
    pub dmin: [u8; 2],
    pub scales: [u8; 12],
    pub qs: [u8; 128],
}

// =========================================================================
// HELPER REDUCTIONS E AUXILIARES AVX-512
// =========================================================================

#[cfg(target_arch = "x86_64")]
unsafe fn reduce_max_epi32(v: __m512i) -> i32 {
    let low: __m256i = _mm512_castsi512_si256(v);
    let high: __m256i = std::mem::transmute(_mm512_extracti64x4_epi64(v, 1));
    let v256 = _mm256_max_epi32(low, high);
    
    let v128_low = _mm256_castsi256_si128(v256);
    let v128_high = _mm256_extractf128_si256(v256, 1);
    let v128 = _mm_max_epi32(v128_low, v128_high);
    
    let v64 = _mm_max_epi32(v128, _mm_srli_si128(v128, 8));
    let v32 = _mm_max_epi32(v64, _mm_srli_si128(v64, 4));
    _mm_cvtsi128_si32(v32)
}

#[cfg(target_arch = "x86_64")]
unsafe fn reduce_add_epi32(v: __m512i) -> i32 {
    let low: __m256i = _mm512_castsi512_si256(v);
    let high: __m256i = std::mem::transmute(_mm512_extracti64x4_epi64(v, 1));
    let v256 = _mm256_add_epi32(low, high);
    
    let v128_low = _mm256_castsi256_si128(v256);
    let v128_high = _mm256_extractf128_si256(v256, 1);
    let v128 = _mm_add_epi32(v128_low, v128_high);
    
    let v64 = _mm_add_epi32(v128, _mm_srli_si128(v128, 8));
    let v32 = _mm_add_epi32(v64, _mm_srli_si128(v64, 4));
    _mm_cvtsi128_si32(v32)
}

// =========================================================================
// KERNELS AVX-512 + FMA / VNNI
// =========================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vnni,avx512vl")]
pub unsafe fn gemv_q8_0_vnni(
    a_i8: &[i8],
    a_scale: f32,
    blocks: &[BlockQ8_0],
    output: &mut [f32],
) {
    let n_blocks = blocks.len();
    let mut sum_vec = _mm256_setzero_ps();

    for b in 0..n_blocks {
        let block = &blocks[b];
        let d = half::f16::from_le_bytes(block.d).to_f32();
        let scale = d * a_scale;
        let v_scale = _mm256_set1_ps(scale);

        let w_raw = _mm256_loadu_si256(block.qs.as_ptr() as *const __m256i);
        let a_raw = _mm256_loadu_si256(a_i8.as_ptr().add(b * 32) as *const __m256i);

        let mut acc = _mm256_setzero_si256();
        acc = _mm256_dpbssd_epi32(acc, w_raw, a_raw);

        let acc_f32 = _mm256_cvtepi32_ps(acc);
        sum_vec = _mm256_fmadd_ps(acc_f32, v_scale, sum_vec);
    }

    let mut buffer = [0.0f32; 8];
    _mm256_storeu_ps(buffer.as_mut_ptr(), sum_vec);
    output[0] = buffer.iter().sum::<f32>();
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn dequantize_q4_k_block_avx512(block: &BlockQ4K, dst: &mut [f32]) {
    let d = half::f16::from_le_bytes(block.d).to_f32();
    let dmin = half::f16::from_le_bytes(block.dmin).to_f32();
    let sc = &block.scales;
    let qs = &block.qs;

    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    for i in 0..4 {
        let s0 = sc[i];
        let s1 = sc[i + 4];
        let s2 = sc[i + 8];
        scales[2*i]     = (s0 & 0x3F) | ((s2 & 0x0F) << 6);
        scales[2*i + 1] = (s1 & 0x3F) | ((s2 >> 4) << 6);
        mins[2*i]       = (s0 >> 6) | ((s2 & 0x0F) << 2);
        mins[2*i + 1]   = (s1 >> 6) | ((s2 >> 4) << 2);
    }

    for sb in 0..8 {
        let scale_val = d * (scales[sb] & 0x3F) as f32;
        let min_val = dmin * (mins[sb] & 0x3F) as f32;
        let v_scale = _mm256_set1_ps(scale_val);
        let v_min = _mm256_set1_ps(min_val);

        let qs_ptr = qs.as_ptr().add(sb * 16);
        let v_qs_raw = _mm_loadu_si128(qs_ptr as *const __m128i);
        let v_qs = _mm256_cvtepu8_epi16(v_qs_raw);

        let mask_low = _mm256_set1_epi16(0x0F);
        let v_low = _mm256_and_si256(v_qs, mask_low);
        let v_low_f32_part1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_low)));
        let v_low_f32_part2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extractf128_si256(v_low, 1)));

        let v_high = _mm256_and_si256(_mm256_srli_epi16(v_qs, 4), mask_low);
        let v_high_f32_part1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_high)));
        let v_high_f32_part2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extractf128_si256(v_high, 1)));

        let res_low_1 = _mm256_fmsub_ps(v_low_f32_part1, v_scale, v_min);
        let res_low_2 = _mm256_fmsub_ps(v_low_f32_part2, v_scale, v_min);
        let res_high_1 = _mm256_fmsub_ps(v_high_f32_part1, v_scale, v_min);
        let res_high_2 = _mm256_fmsub_ps(v_high_f32_part2, v_scale, v_min);

        _mm256_storeu_ps(dst.as_mut_ptr().add(sb * 32), res_low_1);
        _mm256_storeu_ps(dst.as_mut_ptr().add(sb * 32 + 8), res_low_2);
        _mm256_storeu_ps(dst.as_mut_ptr().add(sb * 32 + 16), res_high_1);
        _mm256_storeu_ps(dst.as_mut_ptr().add(sb * 32 + 24), res_high_2);
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
#[target_feature(enable = "avx512f,avx512bw,avx512dq")]
unsafe fn quantize_activations_avx512(a: &[f32], a_i8: &mut [i8]) -> f32 {
    let n = a.len();
    let mut max_vec = _mm512_setzero_ps();
    let mut i = 0;
    while i + 16 <= n {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
        let va_abs = _mm512_and_ps(va, abs_mask);
        max_vec = _mm512_max_ps(max_vec, va_abs);
        i += 16;
    }
    let mut buffer = [0.0f32; 16];
    _mm512_storeu_ps(buffer.as_mut_ptr(), max_vec);
    let mut max_val = buffer.iter().cloned().fold(0.0f32, f32::max);
    while i < n {
        max_val = max_val.max(a[i].abs());
        i += 1;
    }

    let scale = max_val / 127.0;
    let inv_scale = if max_val > 0.0 { 127.0 / max_val } else { 0.0 };
    let v_inv_scale = _mm512_set1_ps(inv_scale);

    i = 0;
    while i + 16 <= n {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let scaled = _mm512_mul_ps(va, v_inv_scale);
        let vi32 = _mm512_cvtps_epi32(scaled);
        let vi8_128 = _mm512_cvtepi32_epi8(vi32);
        _mm_storeu_si128(a_i8.as_mut_ptr().add(i) as *mut __m128i, vi8_128);
        i += 16;
    }
    while i < n {
        a_i8[i] = (a[i] * inv_scale).round() as i8;
        i += 1;
    }
    scale
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq")]
pub unsafe fn softmax_padic_avx512(
    logits: *const i32,
    probs: *mut i32,
    n: usize,
) {
    let mut max_vec = _mm512_loadu_si512(logits as *const __m512i);
    let mut i = 16;
    while i + 16 <= n {
        let v = _mm512_loadu_si512(logits.add(i) as *const __m512i);
        max_vec = _mm512_max_epi32(max_vec, v);
        i += 16;
    }
    let mut max_val = reduce_max_epi32(max_vec);
    while i < n {
        max_val = max_val.max(*logits.add(i));
        i += 1;
    }
    let max_broadcast = _mm512_set1_epi32(max_val);

    let mut quire = _mm512_setzero_si512();
    let one = _mm512_set1_epi32(1 << 16);

    i = 0;
    while i + 16 <= n {
        let x = _mm512_sub_epi32(
            _mm512_loadu_si512(logits.add(i) as *const __m512i),
            max_broadcast
        );

        let x_scaled_down = _mm512_srai_epi32(x, 8);
        let x2_scaled = _mm512_mullo_epi32(x_scaled_down, x_scaled_down);
        let x2_half = _mm512_srai_epi32(x2_scaled, 1);

        let exp_x = _mm512_add_epi32(
            _mm512_add_epi32(one, x),
            x2_half
        );

        _mm512_storeu_si512(probs.add(i) as *mut __m512i, exp_x);
        quire = _mm512_add_epi32(quire, exp_x);
        i += 16;
    }

    let mut sum_exp = reduce_add_epi32(quire) as u64;
    while i < n {
        let diff = *logits.add(i) - max_val;
        let exp_val = exp_q16_16(diff) as i32;
        *probs.add(i) = exp_val;
        sum_exp += exp_val as u64;
        i += 1;
    }

    if sum_exp > 0 {
        let inv_sum = ((1u64 << 32) / sum_exp) as u32;
        let inv_vec = _mm512_set1_epi32(inv_sum as i32);

        i = 0;
        while i + 16 <= n {
            let p = _mm512_loadu_si512(probs.add(i) as *const __m512i);
            let mul = _mm512_mullo_epi32(p, inv_vec);
            let p_out = _mm512_srli_epi32(mul, 16);
            _mm512_storeu_si512(probs.add(i) as *mut __m512i, p_out);
            i += 16;
        }
        while i < n {
            let p = *probs.add(i) as u64;
            *probs.add(i) = ((p * inv_sum as u64) >> 16) as i32;
            i += 1;
        }
    }
}

// =========================================================================
// SOFTMAX P-ÁDICO Q16.16 (INTEIROS EM PONTO FIXO)
// =========================================================================

fn exp_q16_16(x_fixed: i32) -> u32 {
    if x_fixed < -655360 {
        return 0;
    }
    if x_fixed >= 0 {
        return 65536;
    }

    let y = ((x_fixed as i64 * 94548i64) >> 16) as i32;
    let y_int = y >> 16;
    let y_frac = (y & 0xFFFF) as u32;

    let f = y_frac as f64 / 65536.0;
    let two_to_f = 1.0 + f * (0.693147 + f * (0.240226 + f * 0.055504));
    let res = (two_to_f * 65536.0) as u32;

    let shift = -y_int;
    if shift >= 32 {
        0
    } else {
        res >> shift
    }
}

// =========================================================================
// FUNCOES PUBLICAS COM RUNTIME DISPATCH
// =========================================================================

/// RMSNorm (Root Mean Square Normalization)
pub fn rms_norm(x: &CpuTensor, w: &CpuTensor, eps: f32) -> CpuTensor {
    let shape = x.shape();
    assert!(shape.len() >= 1, "Tensor deve ter pelo menos 1 dimensao");
    let n = shape[shape.len() - 1];
    let w_data = w.data();
    assert_eq!(w_data.len(), n);

    let x_data = x.data();
    let mut y_data = vec![0.0f32; x.size()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vl") {
            unsafe {
                rms_norm_avx2(x_data, w_data, &mut y_data, eps, n);
                return CpuTensor::new(shape.to_vec(), y_data);
            }
        }
    }

    // Fallback CPU Escalar
    let num_rows = x.size() / n;
    for r in 0..num_rows {
        let base = r * n;
        let mut sum_sq = 0.0f32;
        for i in 0..n {
            let val = x_data[base + i];
            sum_sq += val * val;
        }
        let rms = (sum_sq / n as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        for i in 0..n {
            y_data[base + i] = (x_data[base + i] * inv_rms) * w_data[i];
        }
    }

    CpuTensor::new(shape.to_vec(), y_data)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rms_norm_avx2(x: &[f32], w: &[f32], y: &mut [f32], eps: f32, n: usize) {
    let num_rows = x.len() / n;
    for r in 0..num_rows {
        let base = r * n;
        let mut sum_sq_vec = _mm256_setzero_ps();
        let mut i = 0;

        while i + 8 <= n {
            let vx = _mm256_loadu_ps(x.as_ptr().add(base + i));
            sum_sq_vec = _mm256_fmadd_ps(vx, vx, sum_sq_vec);
            i += 8;
        }

        let mut buffer = [0.0f32; 8];
        _mm256_storeu_ps(buffer.as_mut_ptr(), sum_sq_vec);
        let mut sum_sq = buffer.iter().sum::<f32>();

        while i < n {
            let val = x[base + i];
            sum_sq += val * val;
            i += 1;
        }

        let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
        let v_inv_rms = _mm256_set1_ps(inv_rms);

        i = 0;
        while i + 8 <= n {
            let vx = _mm256_loadu_ps(x.as_ptr().add(base + i));
            let vw = _mm256_loadu_ps(w.as_ptr().add(i));
            let vy = _mm256_mul_ps(_mm256_mul_ps(vx, v_inv_rms), vw);
            _mm256_storeu_ps(y.as_mut_ptr().add(base + i), vy);
            i += 8;
        }

        while i < n {
            y[base + i] = (x[base + i] * inv_rms) * w[i];
            i += 1;
        }
    }
}

/// Softmax P-ádico Q16.16 sobre a última dimensão acelerado por AVX-512
pub fn softmax(x: &CpuTensor, temperature: f32) -> CpuTensor {
    let shape = x.shape();
    assert!(shape.len() >= 1, "Tensor deve ter pelo menos 1 dimensao");
    let n = shape[shape.len() - 1];

    let x_data = x.data();
    let mut y_data = vec![0.0f32; x.size()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq") {
            let num_rows = x.size() / n;
            let inv_temp = 1.0 / temperature.max(1e-6);

            for r in 0..num_rows {
                let base = r * n;
                // Prepara buffers de ponto fixo
                let mut logits_fixed = vec![0i32; n];
                let mut probs_fixed = vec![0i32; n];

                // 1. Converte float -> Q16.16 fixed point
                for i in 0..n {
                    logits_fixed[i] = (x_data[base + i] * inv_temp * 65536.0) as i32;
                }

                // 2. Invoca o kernel de softmax P-adico AVX-512
                unsafe {
                    softmax_padic_avx512(logits_fixed.as_ptr(), probs_fixed.as_mut_ptr(), n);
                }

                // 3. Converte Q16.16 back to float
                for i in 0..n {
                    y_data[base + i] = probs_fixed[i] as f32 / 65536.0;
                }
            }
            return CpuTensor::new(shape.to_vec(), y_data);
        }
    }

    // Fallback CPU Escalar P-adico
    let inv_temp = 1.0 / temperature.max(1e-6);
    let num_rows = x.size() / n;

    for r in 0..num_rows {
        let base = r * n;
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..n {
            max_val = max_val.max(x_data[base + i] * inv_temp);
        }

        let mut sum_exp = 0u64;
        let mut exp_vals = vec![0u32; n];
        for i in 0..n {
            let diff = (x_data[base + i] * inv_temp) - max_val;
            let diff_fixed = (diff * 65536.0) as i32;
            let exp_val = exp_q16_16(diff_fixed);
            exp_vals[i] = exp_val;
            sum_exp += exp_val as u64;
        }

        if sum_exp > 0 {
            let inv_sum = 1.0 / sum_exp as f32;
            for i in 0..n {
                y_data[base + i] = exp_vals[i] as f32 * inv_sum;
            }
        } else {
            let val = 1.0 / n as f32;
            for i in 0..n {
                y_data[base + i] = val;
            }
        }
    }

    CpuTensor::new(shape.to_vec(), y_data)
}

/// RoPE (Rotary Position Embedding)
pub fn rope(
    x: &mut CpuTensor,
    pos: &[usize],
    rope_dim: usize,
    freq_base: f32,
) {
    let shape = x.shape().to_vec();
    assert!(shape.len() >= 3, "RoPE requer tensor de pelo menos 3 dimensoes");
    let head_dim = shape[shape.len() - 1];
    let n_heads = shape[shape.len() - 2];
    let seq_len = shape[shape.len() - 3];
    assert!(rope_dim <= head_dim);

    let x_data = x.data_mut();

    for s in 0..seq_len {
        let p = pos[s] as f32;

        for h in 0..n_heads {
            let base_offset = s * n_heads * head_dim + h * head_dim;

            for i in (0..rope_dim).step_by(2) {
                let theta = freq_base.powf(-(i as f32) / rope_dim as f32);
                let angle = p * theta;
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                let x0 = x_data[base_offset + i];
                let x1 = x_data[base_offset + i + 1];

                x_data[base_offset + i]     = x0 * cos_val - x1 * sin_val;
                x_data[base_offset + i + 1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}

/// Multiplicacao de matrizes C = A * B
pub fn matmul(a: &CpuTensor, b: &CpuTensor, transpose_b: bool) -> CpuTensor {
    let shape_a = a.shape();
    let shape_b = b.shape();
    assert_eq!(shape_a.len(), 2, "A deve ser 2D");
    assert_eq!(shape_b.len(), 2, "B deve ser 2D");

    let m = shape_a[0];
    let k = shape_a[1];
    let n = if transpose_b { shape_b[0] } else { shape_b[1] };
    let k_b = if transpose_b { shape_b[1] } else { shape_b[0] };
    assert_eq!(k, k_b);

    let a_data = a.data();
    let b_data = b.data();
    let mut c_data = vec![0.0f32; m * n];

    if transpose_b {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx512vl") {
                unsafe {
                    for i in 0..m {
                        let a_base = i * k;
                        let a_slice = &a_data[a_base..a_base + k];
                        
                        // Prefetch explícito de dados 4 iterações à frente
                        if a_base + k * 4 < a_data.len() {
                            _mm_prefetch(a_data.as_ptr().add(a_base + k * 4) as *const i8, _MM_HINT_T0);
                        }

                        for j in 0..n {
                            let b_base = j * k;
                            let b_slice = &b_data[b_base..b_base + k];
                            
                            if b_base + k * 4 < b_data.len() {
                                _mm_prefetch(b_data.as_ptr().add(b_base + k * 4) as *const i8, _MM_HINT_T0);
                            }

                            // Dot product exato de 256 bits (AVX-512VL + FMA)
                            let mut sum_vec = _mm256_setzero_ps();
                            let mut kk = 0;
                            while kk + 8 <= k {
                                let va = _mm256_loadu_ps(a_slice.as_ptr().add(kk));
                                let vb = _mm256_loadu_ps(b_slice.as_ptr().add(kk));
                                sum_vec = _mm256_fmadd_ps(va, vb, sum_vec);
                                kk += 8;
                            }
                            let mut buffer = [0.0f32; 8];
                            _mm256_storeu_ps(buffer.as_mut_ptr(), sum_vec);
                            let mut sum = buffer.iter().sum::<f32>();
                            while kk < k {
                                sum += a_slice[kk] * b_slice[kk];
                                kk += 1;
                            }
                            c_data[i * n + j] = sum;
                        }
                    }
                }
                return CpuTensor::new(vec![m, n], c_data);
            }
        }

        // Fallback CPU Escalar
        for i in 0..m {
            let a_base = i * k;
            for j in 0..n {
                let b_base = j * k;
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a_data[a_base + kk] * b_data[b_base + kk];
                }
                c_data[i * n + j] = sum;
            }
        }
    } else {
        // Multiplicacao padrao A * B
        for i in 0..m {
            let a_base = i * k;
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a_data[a_base + kk] * b_data[kk * n + j];
                }
                c_data[i * n + j] = sum;
            }
        }
    }

    CpuTensor::new(vec![m, n], c_data)
}
