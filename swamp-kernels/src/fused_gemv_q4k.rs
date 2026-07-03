// swamp-kernels/src/fused_gemv_q4k.rs
// Fused GEMV Q4_K + VNNI: lê bytes Q4_K diretamente da RAM, desquantiza
// dentro dos registradores ZMM e acumula o produto escalar em Quire INT32
// sem nunca escrever f32 na RAM.
//
// Fluxo de dados:
//   RAM (Q4_K, 4.5 bits/peso) -> ZMM (INT8) -> VNNI (INT32 Quire) -> f32 escalar
//
// Reducao de trafego de memoria:
//   Convencional: leitura Q (36 MB) + escrita F32 (262 MB) + releitura F32 (262 MB)
//   Fused:        leitura Q (36 MB) apenas
//
// O resultado de cada neuronio (dot product) e escalar e escrito direto.
//
// API C (para ser chamada do swamp-server e swamp-engine via FFI):
//   swamp_fused_gemv_q4k(w_raw, x, out, n_rows, n_cols, out_rows, out_cols)

#![allow(dead_code)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;

const Q4K_BLOCK_SIZE: usize  = 256;
const Q4K_BLOCK_BYTES: usize = 144;

// Desempacota os 8 sub-blocos de escala/min do formato Q4_K (12 bytes -> 8+8)
#[inline(always)]
pub fn unpack_scales_q4k(sc: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];

    for j in 0..4 {
        scales[j] = sc[j] & 63;
        mins[j]   = sc[j + 4] & 63;
    }
    for j in 4..8 {
        scales[j] = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
        mins[j]   = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }
    (scales, mins)
}

/// Fused GEMV Q4_K x F32 -> F32
///
/// Para cada linha `i` da matriz de pesos W (shape: n_rows x n_cols, em Q4_K):
///   out[i] = Σ_j  W[i,j] * x[j]
///
/// W e armazenado como bytes brutos Q4_K (blocos de 144 bytes cada 256 elementos).
/// x e um vetor denso de f32.
/// n_cols deve ser multiplo de 256 (tamanho do bloco Q4_K).
///
/// Estrategia:
///   - Loop externo: rows (output neurons)
///   - Loop interno: blocos de 256 pesos Q4_K
///   - Dentro de cada bloco: quantiza x[j] para INT8 localmente, usa VNNI
///     para acumular dot product em INT32. Nunca escreve f32 de pesos na RAM.
pub fn fused_gemv_q4k(
    w_raw:  &[u8],   // bytes brutos Q4_K (n_rows * n_blocks_per_row * 144)
    x:      &[f32],  // vetor de entrada, len = n_cols
    out:    &mut [f32], // vetor de saida,  len = n_rows
    n_rows: usize,
    n_cols: usize,
) {
    assert_eq!(n_cols % Q4K_BLOCK_SIZE, 0, "n_cols deve ser multiplo de 256 (Q4_K block size)");
    let n_blocks_per_row = n_cols / Q4K_BLOCK_SIZE;
    assert_eq!(w_raw.len(), n_rows * n_blocks_per_row * Q4K_BLOCK_BYTES);
    assert_eq!(x.len(), n_cols);
    assert!(out.len() >= n_rows);

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vnni") {
            return unsafe { fused_gemv_q4k_vnni(w_raw, x, out, n_rows, n_cols, n_blocks_per_row) };
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe { fused_gemv_q4k_avx2(w_raw, x, out, n_rows, n_cols, n_blocks_per_row) };
        }
    }
    fused_gemv_q4k_scalar(w_raw, x, out, n_rows, n_cols, n_blocks_per_row);
}

// =========================================================================
// AVX-512 VNNI path
// =========================================================================
//
// Por bloco de 256 pesos Q4_K:
//   1. Le 128 bytes de nibbles (qs) e os 12 bytes de escalas da RAM.
//   2. Quantiza os 256 valores de x[j] correspondentes para INT8
//      (scale dinamico por bloco, preservando a magnitude maxima).
//   3. Expande os nibbles em INT8 dentro dos registradores ZMM.
//   4. Aplica a correcao de min (subtracao do dmin * sum_x).
//   5. Acumula dot product com _mm512_dpbusd_epi32 (VNNI).
//   6. Ao final da linha, converte o Quire INT32 para f32 e salva.
//
// Zero escrita de f32 intermediarios na RAM.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vl")]
unsafe fn fused_gemv_q4k_vnni(
    w_raw: &[u8], x: &[f32], out: &mut [f32],
    n_rows: usize, _n_cols: usize, n_blocks: usize,
) {
    let xor_mask = _mm512_set1_epi8(-128i8); // i8 -> u8: flip bit de sinal
    let mask_nibble = _mm_set1_epi8(0x0F);
    let eight = _mm_set1_epi8(8i8);

    // Arrays temporarios para o bloco atual de x
    let mut x_i8_buf = [0i8; 256];
    let mut sum_x_subblocks = [0i32; 8];
    let mut va_fulls = [_mm512_setzero_si512(); 8];

    // Loop externo: blocos de pesos (colunas / 256)
    // Inversao de loop (column-first): quantiza x_blk UMA VEZ para todas as linhas
    for blk in 0..n_blocks {
        let x_blk = &x[blk * Q4K_BLOCK_SIZE..(blk + 1) * Q4K_BLOCK_SIZE];

        // 1. Quantiza o bloco de x para INT8
        let mut max_abs = _mm512_setzero_ps();
        let abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFF_FFFF));
        let mut j = 0;
        while j + 16 <= 256 {
            let v = _mm512_loadu_ps(x_blk.as_ptr().add(j));
            max_abs = _mm512_max_ps(max_abs, _mm512_and_ps(v, abs_mask));
            j += 16;
        }
        let mut buf16 = [0.0f32; 16];
        _mm512_storeu_ps(buf16.as_mut_ptr(), max_abs);
        let x_max = buf16.iter().cloned().fold(0.0f32, f32::max);
        let inv_scale_x = if x_max > 1e-6 { 127.0 / x_max } else { 0.0 };
        let scale_x     = if x_max > 1e-6 { x_max / 127.0 } else { 1.0 };

        j = 0;
        while j + 16 <= 256 {
            let vx  = _mm512_loadu_ps(x_blk.as_ptr().add(j));
            let vi  = _mm512_cvtps_epi32(_mm512_mul_ps(vx, _mm512_set1_ps(inv_scale_x)));
            let vi8 = _mm512_cvtepi32_epi8(vi);
            _mm_storeu_si128(x_i8_buf.as_mut_ptr().add(j) as *mut __m128i, vi8);
            j += 16;
        }

        // 2. Pre-calcula as somas de sub-blocos de x_i8 e carrega para ZMM (usados pelo VNNI e Bias)
        for sb in 0..8 {
            let sb_off = sb * 32;
            let mut sum = 0i32;
            for k in 0..32 {
                sum += x_i8_buf[sb_off + k] as i32;
            }
            sum_x_subblocks[sb] = sum;

            let va_lo = _mm256_loadu_si256(x_i8_buf.as_ptr().add(sb_off) as *const __m256i);
            let va_hi = _mm256_loadu_si256(x_i8_buf.as_ptr().add(sb_off + 16) as *const __m256i);
            va_fulls[sb] = _mm512_inserti64x4(_mm512_castsi256_si512(va_lo), va_hi, 1);
        }

        // 3. Loop interno: processa 4 linhas simultaneamente (maximiza IPC / reuso de registradores)
        let mut row = 0;
        while row + 4 <= n_rows {
            let mut quire0 = _mm512_setzero_si512();
            let mut quire1 = _mm512_setzero_si512();
            let mut quire2 = _mm512_setzero_si512();
            let mut quire3 = _mm512_setzero_si512();

            let mut dot_corr0 = 0.0f32;
            let mut dot_corr1 = 0.0f32;
            let mut dot_corr2 = 0.0f32;
            let mut dot_corr3 = 0.0f32;

            let base0 = (row * n_blocks + blk) * Q4K_BLOCK_BYTES;
            let base1 = ((row + 1) * n_blocks + blk) * Q4K_BLOCK_BYTES;
            let base2 = ((row + 2) * n_blocks + blk) * Q4K_BLOCK_BYTES;
            let base3 = ((row + 3) * n_blocks + blk) * Q4K_BLOCK_BYTES;

            // Prefetch agressivo para a proxima iteracao (strided prefetch)
            if blk + 1 < n_blocks {
                _mm_prefetch(w_raw.as_ptr().add(base0 + Q4K_BLOCK_BYTES) as *const i8, _MM_HINT_T0);
                _mm_prefetch(w_raw.as_ptr().add(base1 + Q4K_BLOCK_BYTES) as *const i8, _MM_HINT_T0);
                _mm_prefetch(w_raw.as_ptr().add(base2 + Q4K_BLOCK_BYTES) as *const i8, _MM_HINT_T0);
                _mm_prefetch(w_raw.as_ptr().add(base3 + Q4K_BLOCK_BYTES) as *const i8, _MM_HINT_T0);
            }

            let d0 = f16::from_le_bytes([w_raw[base0], w_raw[base0 + 1]]).to_f32();
            let dmin0 = f16::from_le_bytes([w_raw[base0 + 2], w_raw[base0 + 3]]).to_f32();
            let sc0 = &w_raw[base0 + 4..base0 + 16];
            let qs0 = w_raw.as_ptr().add(base0 + 16);

            let d1 = f16::from_le_bytes([w_raw[base1], w_raw[base1 + 1]]).to_f32();
            let dmin1 = f16::from_le_bytes([w_raw[base1 + 2], w_raw[base1 + 3]]).to_f32();
            let sc1 = &w_raw[base1 + 4..base1 + 16];
            let qs1 = w_raw.as_ptr().add(base1 + 16);

            let d2 = f16::from_le_bytes([w_raw[base2], w_raw[base2 + 1]]).to_f32();
            let dmin2 = f16::from_le_bytes([w_raw[base2 + 2], w_raw[base2 + 3]]).to_f32();
            let sc2 = &w_raw[base2 + 4..base2 + 16];
            let qs2 = w_raw.as_ptr().add(base2 + 16);

            let d3 = f16::from_le_bytes([w_raw[base3], w_raw[base3 + 1]]).to_f32();
            let dmin3 = f16::from_le_bytes([w_raw[base3 + 2], w_raw[base3 + 3]]).to_f32();
            let sc3 = &w_raw[base3 + 4..base3 + 16];
            let qs3 = w_raw.as_ptr().add(base3 + 16);

            let (scales0, mins0) = unpack_scales_q4k(sc0);
            let (scales1, mins1) = unpack_scales_q4k(sc1);
            let (scales2, mins2) = unpack_scales_q4k(sc2);
            let (scales3, mins3) = unpack_scales_q4k(sc3);

            for sb in 0..8 {
                let sum_x = sum_x_subblocks[sb] as f32 * scale_x;
                
                let sv_f0 = d0 * (scales0[sb] & 0x3F) as f32;
                let mv_f0 = dmin0 * (mins0[sb] & 0x3F) as f32;
                dot_corr0 += mv_f0 * sum_x + (128.0 + 8.0) * sum_x * sv_f0;

                let sv_f1 = d1 * (scales1[sb] & 0x3F) as f32;
                let mv_f1 = dmin1 * (mins1[sb] & 0x3F) as f32;
                dot_corr1 += mv_f1 * sum_x + (128.0 + 8.0) * sum_x * sv_f1;

                let sv_f2 = d2 * (scales2[sb] & 0x3F) as f32;
                let mv_f2 = dmin2 * (mins2[sb] & 0x3F) as f32;
                dot_corr2 += mv_f2 * sum_x + (128.0 + 8.0) * sum_x * sv_f2;

                let sv_f3 = d3 * (scales3[sb] & 0x3F) as f32;
                let mv_f3 = dmin3 * (mins3[sb] & 0x3F) as f32;
                dot_corr3 += mv_f3 * sum_x + (128.0 + 8.0) * sum_x * sv_f3;

                let va_full = va_fulls[sb];

                macro_rules! process_row {
                    ($qs:expr, $quire:expr) => {
                        let packed = _mm_loadu_si128($qs.add(sb * 16) as *const __m128i);
                        let lo_nib = _mm_and_si128(packed, mask_nibble);
                        let hi_nib = _mm_and_si128(_mm_srli_epi16(packed, 4), mask_nibble);
                        let lo_i8 = _mm_sub_epi8(lo_nib, eight);
                        let hi_i8 = _mm_sub_epi8(hi_nib, eight);
                        let w_256 = _mm256_set_m128i(hi_i8, lo_i8);
                        let w_512 = _mm512_inserti64x4(_mm512_castsi256_si512(w_256), w_256, 1);
                        let vw_u8 = _mm512_xor_si512(w_512, xor_mask);
                        $quire = _mm512_dpbusd_epi32($quire, vw_u8, va_full);
                    }
                }

                process_row!(qs0, quire0);
                process_row!(qs1, quire1);
                process_row!(qs2, quire2);
                process_row!(qs3, quire3);
            }

            // Colapsa os 4 quires via macro auxiliar
            macro_rules! collapse_quire {
                ($quire:expr) => {{
                    let lo = _mm512_castsi512_si256($quire);
                    let hi = std::mem::transmute(_mm512_extracti64x4_epi64($quire, 1));
                    let s256 = _mm256_add_epi32(lo, hi);
                    let lo128 = _mm256_castsi256_si128(s256);
                    let hi128 = _mm256_extracti128_si256(s256, 1);
                    let s128 = _mm_add_epi32(lo128, hi128);
                    let s64 = _mm_add_epi32(s128, _mm_srli_si128(s128, 8));
                    let s32 = _mm_add_epi32(s64, _mm_srli_si128(s64, 4));
                    _mm_cvtsi128_si32(s32)
                }}
            }

            out[row]     += collapse_quire!(quire0) as f32 * scale_x - dot_corr0;
            out[row + 1] += collapse_quire!(quire1) as f32 * scale_x - dot_corr1;
            out[row + 2] += collapse_quire!(quire2) as f32 * scale_x - dot_corr2;
            out[row + 3] += collapse_quire!(quire3) as f32 * scale_x - dot_corr3;

            row += 4;
        }

        // Remainder loop para as linhas restantes (se houver)
        while row < n_rows {
            let mut quire = _mm512_setzero_si512();
            let mut dot_corr = 0.0f32;
            let base = (row * n_blocks + blk) * Q4K_BLOCK_BYTES;
            let d    = f16::from_le_bytes([w_raw[base],     w_raw[base + 1]]).to_f32();
            let dmin = f16::from_le_bytes([w_raw[base + 2], w_raw[base + 3]]).to_f32();
            let sc   = &w_raw[base + 4..base + 16];
            let qs   = w_raw.as_ptr().add(base + 16);

            let (scales, mins) = unpack_scales_q4k(sc);

            for sb in 0..8 {
                let sv_f = d * (scales[sb] & 0x3F) as f32;
                let mv_f = dmin * (mins[sb] & 0x3F) as f32;
                let sum_x = sum_x_subblocks[sb] as f32 * scale_x;
                dot_corr += mv_f * sum_x + (128.0 + 8.0) * sum_x * sv_f;

                let va_full = va_fulls[sb];

                let packed = _mm_loadu_si128(qs.add(sb * 16) as *const __m128i);
                let lo_nib = _mm_and_si128(packed, mask_nibble);
                let hi_nib = _mm_and_si128(_mm_srli_epi16(packed, 4), mask_nibble);
                let lo_i8 = _mm_sub_epi8(lo_nib, eight);
                let hi_i8 = _mm_sub_epi8(hi_nib, eight);
                let w_256 = _mm256_set_m128i(hi_i8, lo_i8);
                let w_512 = _mm512_inserti64x4(_mm512_castsi256_si512(w_256), w_256, 1);
                let vw_u8 = _mm512_xor_si512(w_512, xor_mask);
                quire = _mm512_dpbusd_epi32(quire, vw_u8, va_full);
            }

            let lo = _mm512_castsi512_si256(quire);
            let hi = std::mem::transmute(_mm512_extracti64x4_epi64(quire, 1));
            let s256 = _mm256_add_epi32(lo, hi);
            let lo128 = _mm256_castsi256_si128(s256);
            let hi128 = _mm256_extracti128_si256(s256, 1);
            let s128 = _mm_add_epi32(lo128, hi128);
            let s64 = _mm_add_epi32(s128, _mm_srli_si128(s128, 8));
            let s32 = _mm_add_epi32(s64, _mm_srli_si128(s64, 4));
            let quire_scalar = _mm_cvtsi128_si32(s32);

            out[row] += quire_scalar as f32 * scale_x - dot_corr;
            row += 1;
        }
    }
}

// =========================================================================
// AVX2 FMA fallback
// =========================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_gemv_q4k_avx2(
    w_raw: &[u8], x: &[f32], out: &mut [f32],
    n_rows: usize, _n_cols: usize, n_blocks: usize,
) {
    for row in 0..n_rows {
        let mut acc = 0.0f32;

        for blk in 0..n_blocks {
            let base = (row * n_blocks + blk) * Q4K_BLOCK_BYTES;
            let d    = f16::from_le_bytes([w_raw[base],     w_raw[base + 1]]).to_f32();
            let dmin = f16::from_le_bytes([w_raw[base + 2], w_raw[base + 3]]).to_f32();
            let sc   = &w_raw[base + 4..base + 16];
            let qs   = &w_raw[base + 16..base + 144];

            let (scales, mins) = unpack_scales_q4k(sc);
            let x_blk = &x[blk * Q4K_BLOCK_SIZE..(blk + 1) * Q4K_BLOCK_SIZE];

            for sb in 0..8usize {
                let sv = d    * (scales[sb] & 0x3F) as f32;
                let mv = dmin * (mins[sb]   & 0x3F) as f32;

                let mut v_acc = _mm256_setzero_ps();
                let v_mv  = _mm256_set1_ps(-mv);

                // Processa 16 pesos low nibble + 16 pesos high nibble do sub-bloco
                for half in 0..2usize {
                    let elem_start = sb * 32 + half * 16;
                    let x_half = &x_blk[elem_start..elem_start + 16];
                    let qs_off = sb * 16;

                    // Carrega 16 bytes de qs, extrai os nibbles do half correto
                    let src = _mm_loadu_si128(qs.as_ptr().add(qs_off) as *const __m128i);
                    let mask = _mm_set1_epi8(0x0F);
                    let nibbles = if half == 0 {
                        _mm_and_si128(src, mask)
                    } else {
                        _mm_and_si128(_mm_srli_epi16(src, 4), mask)
                    };

                    // Dois chunks de 8 floats
                    for chunk in 0..2usize {
                        let chunk_start = chunk * 8;
                        if elem_start + chunk_start >= x_blk.len() { break; }

                        let src64 = _mm_loadl_epi64(
                            (&nibbles as *const __m128i as *const u8).add(chunk_start) as *const __m128i
                        );
                        let qi_i32 = _mm256_cvtepu8_epi32(src64);
                        let qi_f32 = _mm256_cvtepi32_ps(qi_i32);
                        let vx = _mm256_loadu_ps(x_half.as_ptr().add(chunk_start));
                        let vw = _mm256_set1_ps(sv);
                        // out += vw * qi_f32 * vx - mv * vx = (sv * qi - mv) * vx
                        let fused = _mm256_fmadd_ps(
                            _mm256_fmadd_ps(vw, qi_f32, v_mv),
                            vx, _mm256_setzero_ps()
                        );
                        v_acc = _mm256_add_ps(v_acc, fused);
                    }
                }

                let mut buf = [0.0f32; 8];
                _mm256_storeu_ps(buf.as_mut_ptr(), v_acc);
                acc += buf.iter().sum::<f32>();
            }
        }
        out[row] = acc;
    }
}

// =========================================================================
// FUSED MULTI: processa N matrizes com leitura única de x por bloco
// =========================================================================

/// Fused GEMV Q4_K para multiplas matrizes.
/// Le x UMA vez por bloco e computa dot products para todas as linhas
/// de todas as matrizes simultaneamente.
pub fn fused_gemv_q4k_multi(
    raws: &[&[u8]],
    x: &[f32],
    outputs: &mut [&mut [f32]],
    row_counts: &[usize],
    n_cols: usize,
) {
    debug_assert_eq!(n_cols % Q4K_BLOCK_SIZE, 0);
    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let n_matrices = raws.len();
    debug_assert_eq!(n_matrices, outputs.len());
    debug_assert_eq!(n_matrices, row_counts.len());

    for out in outputs.iter_mut() {
        out.fill(0.0);
    }

    // Column-first: bloco externo, x lido UMA vez, reusado por todas as linhas
    for blk in 0..n_blocks {
        let x_blk = &x[blk * Q4K_BLOCK_SIZE..(blk + 1) * Q4K_BLOCK_SIZE];

        for m in 0..n_matrices {
            let raw = raws[m];
            let out = &mut outputs[m];
            let n_rows = row_counts[m];

            for row in 0..n_rows {
                let base = (row * n_blocks + blk) * Q4K_BLOCK_BYTES;
                let d    = f16::from_le_bytes([raw[base], raw[base + 1]]).to_f32();
                let dmin = f16::from_le_bytes([raw[base + 2], raw[base + 3]]).to_f32();
                let sc   = &raw[base + 4..base + 16];
                let qs   = &raw[base + 16..base + 144];

                let (scales, mins) = unpack_scales_q4k(sc);
                let mut acc = out[row];

                for pair in 0..4 {
                    let q_base = &qs[pair * 32..(pair + 1) * 32];

                    let sb_even = pair * 2;
                    let sv_even = d * (scales[sb_even] & 0x3F) as f32;
                    let mv_even = dmin * (mins[sb_even] & 0x3F) as f32;
                    for i in 0..32 {
                        let qi = q_base[i] & 0x0F;
                        acc += (sv_even * qi as f32 - mv_even) * x_blk[sb_even * 32 + i];
                    }

                    let sb_odd = pair * 2 + 1;
                    let sv_odd = d * (scales[sb_odd] & 0x3F) as f32;
                    let mv_odd = dmin * (mins[sb_odd] & 0x3F) as f32;
                    for i in 0..32 {
                        let qi = q_base[i] >> 4;
                        acc += (sv_odd * qi as f32 - mv_odd) * x_blk[sb_odd * 32 + i];
                    }
                }
                out[row] = acc;
            }
        }
    }
}

// =========================================================================
// Escalar fallback
// =========================================================================

fn fused_gemv_q4k_scalar(
    w_raw: &[u8], x: &[f32], out: &mut [f32],
    n_rows: usize, _n_cols: usize, n_blocks: usize,
) {
    for row in 0..n_rows {
        let mut acc = 0.0f32;

        for blk in 0..n_blocks {
            let base = (row * n_blocks + blk) * Q4K_BLOCK_BYTES;
            let d    = f16::from_le_bytes([w_raw[base],     w_raw[base + 1]]).to_f32();
            let dmin = f16::from_le_bytes([w_raw[base + 2], w_raw[base + 3]]).to_f32();
            let sc   = &w_raw[base + 4..base + 16];
            let qs   = &w_raw[base + 16..base + 144];

            let (scales, mins) = unpack_scales_q4k(sc);
            let x_blk = &x[blk * Q4K_BLOCK_SIZE..(blk + 1) * Q4K_BLOCK_SIZE];

            // 4 pares de subblocks (0,1), (2,3), (4,5), (6,7)
            for pair in 0..4 {
                let q_base = &qs[pair * 32..(pair + 1) * 32];
                
                let sb_even = pair * 2;
                let sv_even = d * (scales[sb_even] & 0x3F) as f32;
                let mv_even = dmin * (mins[sb_even] & 0x3F) as f32;
                for i in 0..32 {
                    let qi = q_base[i] & 0x0F;
                    let w_f = sv_even * qi as f32 - mv_even;
                    acc += w_f * x_blk[sb_even * 32 + i];
                }

                let sb_odd = pair * 2 + 1;
                let sv_odd = d * (scales[sb_odd] & 0x3F) as f32;
                let mv_odd = dmin * (mins[sb_odd] & 0x3F) as f32;
                for i in 0..32 {
                    let qi = q_base[i] >> 4;
                    let w_f = sv_odd * qi as f32 - mv_odd;
                    acc += w_f * x_blk[sb_odd * 32 + i];
                }
            }
        }
        out[row] = acc;
    }
}

// =========================================================================
// BATCHED GEMV Q4_K: block-level weight reuse across batch dimension
// =========================================================================
//
// Column-first (block-outer) kernel that processes a batch of input vectors
// against the same weight matrix with shared weight loading.
//
// For each block of 256 columns:
//   1. Pre-quantize all batch tokens' x[blk] to i8 (once per token)
//   2. Process rows (4 at a time):
//      a. Load weight data (scales, mins, nibbles) ONCE for all tokens
//      b. For each token: VNNI dot product using shared weights
//      c. Bias correction and accumulation
//
// This reduces weight memory traffic by batch_size × vs calling single GEMV
// repeatedly. For batch_size=128, approx 100x reduction in weight reads.

pub fn fused_gemv_q4k_batched(
    w_raw:  &[u8],        // Q4_K bytes: n_rows * n_blocks * 144
    x_ptrs: &[*const f32], // batch_size pointers to input vectors, each n_cols
    out_ptrs: &[*mut f32], // batch_size pointers to output vectors, each n_rows
    n_rows: usize,
    n_cols: usize,
    batch_size: usize,
) {
    debug_assert_eq!(n_cols % Q4K_BLOCK_SIZE, 0);
    let n_blocks = n_cols / Q4K_BLOCK_SIZE;

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512vnni")
        {
            return unsafe {
                batched_vnni(w_raw, x_ptrs, out_ptrs, n_rows, n_cols, n_blocks, batch_size)
            };
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe {
                batched_avx2(w_raw, x_ptrs, out_ptrs, n_rows, n_cols, n_blocks, batch_size)
            };
        }
    }
    unsafe { batched_scalar(w_raw, x_ptrs, out_ptrs, n_rows, n_cols, n_blocks, batch_size); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vl")]
unsafe fn batched_vnni(
    w_raw: &[u8],
    x_ptrs: &[*const f32],
    out_ptrs: &[*mut f32],
    n_rows: usize,
    n_cols: usize,
    n_blocks: usize,
    batch_size: usize,
) {
    // Per-token loop calling fused_gemv_q4k_vnni for each token.
    // Weight-sharing optimization disabled due to compiler sensitivity;
    // AVX2 and scalar batched paths do share weights correctly.
    for t in 0..batch_size {
        let x_slice  = std::slice::from_raw_parts(x_ptrs[t], n_cols);
        let out_slice = std::slice::from_raw_parts_mut(out_ptrs[t], n_rows);
        fused_gemv_q4k_vnni(w_raw, x_slice, out_slice, n_rows, n_cols, n_blocks);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn batched_avx2(
    w_raw: &[u8],
    x_ptrs: &[*const f32],
    out_ptrs: &[*mut f32],
    n_rows: usize,
    _n_cols: usize,
    n_blocks: usize,
    batch_size: usize,
) {
    for blk in 0..n_blocks {
        let blk_offset = blk * Q4K_BLOCK_SIZE;

        for row in 0..n_rows {
            let base = (row * n_blocks + blk) * Q4K_BLOCK_BYTES;
            let d = f16::from_le_bytes([w_raw[base], w_raw[base + 1]]).to_f32();
            let dmin = f16::from_le_bytes([w_raw[base + 2], w_raw[base + 3]]).to_f32();
            let sc = &w_raw[base + 4..base + 16];
            let qs = &w_raw[base + 16..base + 144];
            let (scales, mins) = unpack_scales_q4k(sc);

            for t in 0..batch_size {
                let x_blk = std::slice::from_raw_parts(x_ptrs[t].add(blk_offset), Q4K_BLOCK_SIZE);
                let mut acc = 0.0f32;

                for sb in 0..8 {
                    let sv = d * (scales[sb] & 0x3F) as f32;
                    let mv = dmin * (mins[sb] & 0x3F) as f32;

                    let mut v_acc = _mm256_setzero_ps();
                    let v_mv = _mm256_set1_ps(-mv);

                    for half in 0..2 {
                        let elem_start = sb * 32 + half * 16;
                        let qs_off = sb * 16;

                        let src = _mm_loadu_si128(qs.as_ptr().add(qs_off) as *const __m128i);
                        let mask = _mm_set1_epi8(0x0F);
                        let nibbles = if half == 0 {
                            _mm_and_si128(src, mask)
                        } else {
                            _mm_and_si128(_mm_srli_epi16(src, 4), mask)
                        };

                        for chunk in 0..2 {
                            let chunk_start = chunk * 8;
                            let src64 = _mm_loadl_epi64(
                                (&nibbles as *const __m128i as *const u8).add(chunk_start) as *const __m128i
                            );
                            let qi_i32 = _mm256_cvtepu8_epi32(src64);
                            let qi_f32 = _mm256_cvtepi32_ps(qi_i32);
                            let vx = _mm256_loadu_ps(x_blk.as_ptr().add(elem_start + chunk_start));
                            let vw = _mm256_set1_ps(sv);
                            let fused = _mm256_fmadd_ps(
                                _mm256_fmadd_ps(vw, qi_f32, v_mv),
                                vx, _mm256_setzero_ps()
                            );
                            v_acc = _mm256_add_ps(v_acc, fused);
                        }
                    }

                    let mut buf = [0.0f32; 8];
                    _mm256_storeu_ps(buf.as_mut_ptr(), v_acc);
                    acc += buf.iter().sum::<f32>();
                }

                *out_ptrs[t].add(row) += acc;
            }
        }
    }
}

unsafe fn batched_scalar(
    w_raw: &[u8],
    x_ptrs: &[*const f32],
    out_ptrs: &[*mut f32],
    n_rows: usize,
    _n_cols: usize,
    n_blocks: usize,
    batch_size: usize,
) {
    for blk in 0..n_blocks {
        let blk_offset = blk * Q4K_BLOCK_SIZE;

        for row in 0..n_rows {
            let base = (row * n_blocks + blk) * Q4K_BLOCK_BYTES;
            let d = f16::from_le_bytes([w_raw[base], w_raw[base + 1]]).to_f32();
            let dmin = f16::from_le_bytes([w_raw[base + 2], w_raw[base + 3]]).to_f32();
            let sc = &w_raw[base + 4..base + 16];
            let qs = &w_raw[base + 16..base + 144];
            let (scales, mins) = unpack_scales_q4k(sc);

            for t in 0..batch_size {
                let x_blk = std::slice::from_raw_parts(x_ptrs[t].add(blk_offset), Q4K_BLOCK_SIZE);
                let mut acc = 0.0f32;

                for pair in 0..4 {
                    let q_base = &qs[pair * 32..(pair + 1) * 32];

                    let sb_even = pair * 2;
                    let sv_even = d * (scales[sb_even] & 0x3F) as f32;
                    let mv_even = dmin * (mins[sb_even] & 0x3F) as f32;
                    for i in 0..32 {
                        let qi = q_base[i] & 0x0F;
                        acc += (sv_even * qi as f32 - mv_even) * x_blk[sb_even * 32 + i];
                    }

                    let sb_odd = pair * 2 + 1;
                    let sv_odd = d * (scales[sb_odd] & 0x3F) as f32;
                    let mv_odd = dmin * (mins[sb_odd] & 0x3F) as f32;
                    for i in 0..32 {
                        let qi = q_base[i] >> 4;
                        acc += (sv_odd * qi as f32 - mv_odd) * x_blk[sb_odd * 32 + i];
                    }
                }

                *out_ptrs[t].add(row) += acc;
            }
        }
    }
}

// =========================================================================
// TESTS
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_q4k_block(d: f32, dmin: f32, scales: &[u8; 8], mins: &[u8; 8], nibble_val: u8) -> Vec<u8> {
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
        for j in 0..4 {
            block[4 + j] = (scales[j] & 0x3F) | ((scales[4 + j] >> 4) << 6);
            block[8 + j] = (mins[j] & 0x3F) | ((mins[4 + j] >> 4) << 6);
        }
        block[12] = (scales[4] & 0x0F) | ((mins[4] & 0x0F) << 4);
        block[13] = (scales[5] & 0x0F) | ((mins[5] & 0x0F) << 4);
        block[14] = (scales[6] & 0x0F) | ((mins[6] & 0x0F) << 4);
        block[15] = (scales[7] & 0x0F) | ((mins[7] & 0x0F) << 4);
        for i in 0..128 {
            block[16 + i] = nibble_val | (nibble_val << 4);
        }
        block
    }

    fn make_weight(n_rows: usize, _n_cols: usize) -> Vec<u8> {
        let mut w = vec![0u8; 0];
        for r in 0..n_rows {
            let d = 0.5 + r as f32 * 0.1;
            let dm = 0.1 + r as f32 * 0.05;
            let scales = [(r as u8 + 1) & 0x3F; 8];
            let mins = [(r as u8 + 2) & 0x3F; 8];
            w.append(&mut make_q4k_block(d, dm, &scales, &mins, 8));
        }
        w
    }

    fn make_x(n_cols: usize) -> Vec<f32> {
        (0..n_cols).map(|i| (i % 100) as f32 / 50.0 - 1.0).collect()
    }

    fn test_batched_batch1_vs_single(n_rows: usize) {
        let n_cols = 256;
        let w = make_weight(n_rows, n_cols);
        let x = make_x(n_cols);

        let mut expected = vec![0.0f32; n_rows];
        fused_gemv_q4k(&w, &x, &mut expected, n_rows, n_cols);

        let mut actual = vec![0.0f32; n_rows];
        let x_ptrs = [x.as_ptr()];
        let out_ptrs = [actual.as_mut_ptr()];
        fused_gemv_q4k_batched(&w, &x_ptrs, &out_ptrs, n_rows, n_cols, 1);

        for r in 0..n_rows {
            let diff = (expected[r] - actual[r]).abs();
            assert!(diff < 0.01, "row={}: n_rows={} diff={:.6}", r, n_rows, diff);
        }
    }

    #[test]
    fn test_batched_batch1_vs_single_2rows() { test_batched_batch1_vs_single(2); }
    #[test]
    fn test_batched_batch1_vs_single_3rows() { test_batched_batch1_vs_single(3); }
    #[test]
    fn test_batched_batch1_vs_single_4rows() { test_batched_batch1_vs_single(4); }
    #[test]
    fn test_batched_batch1_vs_single_7rows() { test_batched_batch1_vs_single(7); }

    #[test]
    fn test_batched_scalar_vs_single_scalar() {
        let n_rows = 4;
        let n_cols = 256;
        let n_blocks = n_cols / Q4K_BLOCK_SIZE;
        let w = make_weight(n_rows, n_cols);
        let x = make_x(n_cols);

        let mut actual = vec![0.0f32; n_rows];
        let x_ptrs = [x.as_ptr()];
        let out_ptrs = [actual.as_mut_ptr()];
        unsafe { batched_scalar(&w, &x_ptrs, &out_ptrs, n_rows, n_cols, n_blocks, 1); }

        let mut expected = vec![0.0f32; n_rows];
        fused_gemv_q4k_scalar(&w, &x, &mut expected, n_rows, n_cols, n_blocks);

        for r in 0..n_rows {
            let diff = (expected[r] - actual[r]).abs();
            assert!(diff < 0.01, "row={}: diff={:.6}", r, diff);
        }
    }
}

// =========================================================================
// FFI C para integração com swamp-engine
// =========================================================================

use std::os::raw::{c_float, c_int};

#[no_mangle]
pub extern "C" fn swamp_fused_gemv_q4k(
    w_raw_ptr: *const u8,
    x_ptr:     *const c_float,
    out_ptr:   *mut c_float,
    n_rows:    c_int,
    n_cols:    c_int,
) {
    if w_raw_ptr.is_null() || x_ptr.is_null() || out_ptr.is_null() { return; }
    let n_rows = n_rows as usize;
    let n_cols = n_cols as usize;
    if n_cols % Q4K_BLOCK_SIZE != 0 { return; }

    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let raw_len  = n_rows * n_blocks * Q4K_BLOCK_BYTES;

    let w_raw = unsafe { std::slice::from_raw_parts(w_raw_ptr, raw_len) };
    let x     = unsafe { std::slice::from_raw_parts(x_ptr, n_cols) };
    let out   = unsafe { std::slice::from_raw_parts_mut(out_ptr, n_rows) };

    // Zera o buffer de saida antes de acumular
    out.iter_mut().for_each(|v| *v = 0.0);

    fused_gemv_q4k(w_raw, x, out, n_rows, n_cols);
}
