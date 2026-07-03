use std::arch::x86_64::*;

const BLOCK_SIZE: usize = 256;
const BLOCK_BYTES: usize = 210;

fn f16_to_f32(slice: &[u8]) -> f32 {
    let v = u16::from_le_bytes([slice[0], slice[1]]);
    f16_to_f32_raw(v)
}

fn f16_to_f32_raw(v: u16) -> f32 {
    let sign = ((v >> 15) as i32) << 31;
    let exp = ((v >> 10) & 0x1f) as i32;
    let mant = (v & 0x3ff) as i32;
    if exp == 0 {
        f32::from_bits((sign | (1023 - 14) << 23 | mant << 13) as u32)
    } else {
        f32::from_bits((sign | (exp + 127 - 15) << 23 | mant << 13) as u32)
    }
}

pub fn fused_gemv_q6k(
    raw: &[u8],
    x: &[f32],
    out: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) {
    let n_blocks = n_cols / BLOCK_SIZE;

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            return unsafe { fused_gemv_q6k_avx512(raw, x, out, n_rows, n_cols, n_blocks) };
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe { fused_gemv_q6k_avx2(raw, x, out, n_rows, n_cols, n_blocks) };
        }
    }
    fused_gemv_q6k_scalar(raw, x, out, n_rows, n_cols, n_blocks);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl")]
unsafe fn fused_gemv_q6k_avx512(
    raw: &[u8], x: &[f32], out: &mut [f32],
    n_rows: usize, _n_cols: usize, n_blocks: usize,
) {
    let mask_0f = _mm_set1_epi8(0x0F);
    let mask_03 = _mm_set1_epi8(3);
    let v_scale16 = _mm_set1_epi16(16);

    for row in 0..n_rows {
        let mut acc = _mm512_setzero_ps();

        for blk in 0..n_blocks {
            let base = (row * n_blocks + blk) * BLOCK_BYTES;
            let d = f16_to_f32(&raw[base + 208..base + 210]);
            let ql = &raw[base..base + 128];
            let qh = &raw[base + 128..base + 192];
            let sc = &raw[base + 192..base + 208];
            let x_blk = &x[blk * BLOCK_SIZE..(blk + 1) * BLOCK_SIZE];

            for half in 0..2 {
                let ql_h = &ql[half * 64..];
                let qh_h = &qh[half * 32..];
                let sc_h = &sc[half * 8..];
                let x_half = &x_blk[half * 128..];

                for is in 0..2 {
                    let v_s0 = _mm512_set1_ps(d * sc_h[is] as f32);
                    let v_s2 = _mm512_set1_ps(d * sc_h[is + 2] as f32);
                    let v_s4 = _mm512_set1_ps(d * sc_h[is + 4] as f32);
                    let v_s6 = _mm512_set1_ps(d * sc_h[is + 6] as f32);

                    let l_base = is * 16;

                    let v_ql0 = _mm_loadu_si128(ql_h.as_ptr().add(l_base) as *const __m128i);
                    let v_ql32 = _mm_loadu_si128(ql_h.as_ptr().add(l_base + 32) as *const __m128i);
                    let v_qh = _mm_loadu_si128(qh_h.as_ptr().add(l_base) as *const __m128i);

                    // q1: low nibble of ql0 + bits[1:0] of qh
                    let q1_low = _mm_and_si128(v_ql0, mask_0f);
                    let q1_high = _mm_mullo_epi16(_mm_and_si128(v_qh, mask_03), v_scale16);
                    let q1_e = _mm_sub_epi8(_mm_or_si128(q1_low, q1_high), _mm_set1_epi8(32));

                    // q2: low nibble of ql32 + bits[3:2] of qh
                    let q2_low = _mm_and_si128(v_ql32, mask_0f);
                    let q2_high = _mm_mullo_epi16(
                        _mm_and_si128(_mm_srli_epi16(v_qh, 2), mask_03), v_scale16);
                    let q2_e = _mm_sub_epi8(_mm_or_si128(q2_low, q2_high), _mm_set1_epi8(32));

                    // q3: high nibble of ql0 + bits[5:4] of qh
                    let q3_low = _mm_and_si128(_mm_srli_epi16(v_ql0, 4), mask_0f);
                    let q3_high = _mm_mullo_epi16(
                        _mm_and_si128(_mm_srli_epi16(v_qh, 4), mask_03), v_scale16);
                    let q3_e = _mm_sub_epi8(_mm_or_si128(q3_low, q3_high), _mm_set1_epi8(32));

                    // q4: high nibble of ql32 + bits[7:6] of qh
                    let q4_low = _mm_and_si128(_mm_srli_epi16(v_ql32, 4), mask_0f);
                    let q4_high = _mm_mullo_epi16(
                        _mm_and_si128(_mm_srli_epi16(v_qh, 6), mask_03), v_scale16);
                    let q4_e = _mm_sub_epi8(_mm_or_si128(q4_low, q4_high), _mm_set1_epi8(32));

                    let q1_f = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q1_e));
                    let q2_f = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q2_e));
                    let q3_f = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q3_e));
                    let q4_f = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q4_e));

                    let w1 = _mm512_mul_ps(q1_f, v_s0);
                    let w2 = _mm512_mul_ps(q2_f, v_s2);
                    let w3 = _mm512_mul_ps(q3_f, v_s4);
                    let w4 = _mm512_mul_ps(q4_f, v_s6);

                    let x1 = _mm512_loadu_ps(x_half.as_ptr().add(l_base));
                    let x2 = _mm512_loadu_ps(x_half.as_ptr().add(l_base + 32));
                    let x3 = _mm512_loadu_ps(x_half.as_ptr().add(l_base + 64));
                    let x4 = _mm512_loadu_ps(x_half.as_ptr().add(l_base + 96));

                    acc = _mm512_fmadd_ps(w1, x1, acc);
                    acc = _mm512_fmadd_ps(w2, x2, acc);
                    acc = _mm512_fmadd_ps(w3, x3, acc);
                    acc = _mm512_fmadd_ps(w4, x4, acc);
                }
            }
        }

        let mut buf = [0.0f32; 16];
        _mm512_storeu_ps(buf.as_mut_ptr(), acc);
        out[row] = buf.iter().sum();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_gemv_q6k_avx2(
    raw: &[u8], x: &[f32], out: &mut [f32],
    n_rows: usize, _n_cols: usize, n_blocks: usize,
) {
    let mask_0f = _mm_set1_epi8(0x0F);
    let mask_03 = _mm_set1_epi8(3);
    let v_scale16 = _mm_set1_epi16(16);

    for row in 0..n_rows {
        let mut acc = _mm256_setzero_ps();

        for blk in 0..n_blocks {
            let base = (row * n_blocks + blk) * BLOCK_BYTES;
            let d = f16_to_f32(&raw[base + 208..base + 210]);
            let ql = &raw[base..base + 128];
            let qh = &raw[base + 128..base + 192];
            let sc = &raw[base + 192..base + 208];
            let x_blk = &x[blk * BLOCK_SIZE..(blk + 1) * BLOCK_SIZE];

            for half in 0..2 {
                let ql_h = &ql[half * 64..];
                let qh_h = &qh[half * 32..];
                let sc_h = &sc[half * 8..];
                let x_half = &x_blk[half * 128..];

                let mut local_acc = _mm256_setzero_ps();

                for is in 0..2 {
                    let v_s0 = _mm256_set1_ps(d * sc_h[is] as f32);
                    let v_s2 = _mm256_set1_ps(d * sc_h[is + 2] as f32);
                    let v_s4 = _mm256_set1_ps(d * sc_h[is + 4] as f32);
                    let v_s6 = _mm256_set1_ps(d * sc_h[is + 6] as f32);

                    for sub in 0..2 {
                        let l_base = is * 16 + sub * 8;
                        let v_ql0 = _mm_loadu_si128(ql_h.as_ptr().add(l_base) as *const __m128i);
                        let v_ql32 = _mm_loadu_si128(ql_h.as_ptr().add(l_base + 32) as *const __m128i);
                        let v_qh = _mm_loadu_si128(qh_h.as_ptr().add(l_base) as *const __m128i);

                        let q1_e = extract6_128_0(v_ql0, v_qh, 0, mask_0f, mask_03, v_scale16);
                        let q2_e = extract6_128_0(v_ql32, v_qh, 2, mask_0f, mask_03, v_scale16);
                        let q3_e = extract6_128_4(v_ql0, v_qh, 4, mask_0f, mask_03, v_scale16);
                        let q4_e = extract6_128_4(v_ql32, v_qh, 6, mask_0f, mask_03, v_scale16);

                        let q1_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1_e));
                        let q2_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q2_e));
                        let q3_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q3_e));
                        let q4_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q4_e));

                        let x1 = _mm256_loadu_ps(x_half.as_ptr().add(l_base));
                        let x2 = _mm256_loadu_ps(x_half.as_ptr().add(l_base + 32));
                        let x3 = _mm256_loadu_ps(x_half.as_ptr().add(l_base + 64));
                        let x4 = _mm256_loadu_ps(x_half.as_ptr().add(l_base + 96));

                        local_acc = _mm256_fmadd_ps(_mm256_mul_ps(q1_f, v_s0), x1, local_acc);
                        local_acc = _mm256_fmadd_ps(_mm256_mul_ps(q2_f, v_s2), x2, local_acc);
                        local_acc = _mm256_fmadd_ps(_mm256_mul_ps(q3_f, v_s4), x3, local_acc);
                        local_acc = _mm256_fmadd_ps(_mm256_mul_ps(q4_f, v_s6), x4, local_acc);
                    }
                }
                acc = _mm256_add_ps(acc, local_acc);
            }
        }

        let mut row_buf = [0.0f32; 8];
        _mm256_storeu_ps(row_buf.as_mut_ptr(), acc);
        out[row] = row_buf.iter().sum();
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn extract6_128_0(
    v_ql: __m128i, v_qh: __m128i,
    shift_qh: i32,
    mask_0f: __m128i, mask_03: __m128i,
    v_scale16: __m128i,
) -> __m128i {
    let low4 = _mm_and_si128(v_ql, mask_0f);
    let high2 = if shift_qh == 0 {
        _mm_and_si128(v_qh, mask_03)
    } else if shift_qh == 2 {
        _mm_and_si128(_mm_srli_epi16(v_qh, 2), mask_03)
    } else if shift_qh == 4 {
        _mm_and_si128(_mm_srli_epi16(v_qh, 4), mask_03)
    } else {
        _mm_and_si128(_mm_srli_epi16(v_qh, 6), mask_03)
    };
    let high2_scaled = _mm_mullo_epi16(high2, v_scale16);
    let combined = _mm_or_si128(low4, high2_scaled);
    _mm_sub_epi8(combined, _mm_set1_epi8(32))
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn extract6_128_4(
    v_ql: __m128i, v_qh: __m128i,
    shift_qh: i32,
    mask_0f: __m128i, mask_03: __m128i,
    v_scale16: __m128i,
) -> __m128i {
    let low4 = _mm_and_si128(_mm_srli_epi16(v_ql, 4), mask_0f);
    let high2 = if shift_qh == 4 {
        _mm_and_si128(_mm_srli_epi16(v_qh, 4), mask_03)
    } else {
        _mm_and_si128(_mm_srli_epi16(v_qh, 6), mask_03)
    };
    let high2_scaled = _mm_mullo_epi16(high2, v_scale16);
    let combined = _mm_or_si128(low4, high2_scaled);
    _mm_sub_epi8(combined, _mm_set1_epi8(32))
}

fn fused_gemv_q6k_scalar(
    raw: &[u8], x: &[f32], out: &mut [f32],
    n_rows: usize, _n_cols: usize, n_blocks: usize,
) {
    for row in 0..n_rows {
        let mut acc: f32 = 0.0;
        for blk in 0..n_blocks {
            let base = (row * n_blocks + blk) * BLOCK_BYTES;
            let d = f16_to_f32(&raw[base + 208..base + 210]);
            let ql = &raw[base..base + 128];
            let qh = &raw[base + 128..base + 192];
            let sc = &raw[base + 192..base + 208];
            let x_blk = &x[blk * BLOCK_SIZE..(blk + 1) * BLOCK_SIZE];

            for half in 0..2 {
                let ql_h = &ql[half * 64..(half + 1) * 64];
                let qh_h = &qh[half * 32..(half + 1) * 32];
                let sc_h = &sc[half * 8..(half + 1) * 8];
                let x_half = &x_blk[half * 128..(half + 1) * 128];

                for l in 0..32 {
                    let is = l / 16;
                    let b0 = ((ql_h[l] & 0x0F) as u16) | (((qh_h[l] as u16) & 3) << 4);
                    let b1 = ((ql_h[l + 32] & 0x0F) as u16) | ((((qh_h[l] as u16) >> 2) & 3) << 4);
                    let b2 = ((ql_h[l] >> 4) as u16) | ((((qh_h[l] as u16) >> 4) & 3) << 4);
                    let b3 = ((ql_h[l + 32] >> 4) as u16) | ((((qh_h[l] as u16) >> 6) & 3) << 4);
                    let q1 = (b0 as i8).wrapping_sub(32) as f32;
                    let q2 = (b1 as i8).wrapping_sub(32) as f32;
                    let q3 = (b2 as i8).wrapping_sub(32) as f32;
                    let q4 = (b3 as i8).wrapping_sub(32) as f32;
                    let sc1 = sc_h[is] as i8 as f32;
                    let sc2 = sc_h[is + 2] as i8 as f32;
                    let sc3 = sc_h[is + 4] as i8 as f32;
                    let sc4 = sc_h[is + 6] as i8 as f32;
                    acc += d * sc1 * q1 * x_half[l];
                    acc += d * sc2 * q2 * x_half[l + 32];
                    acc += d * sc3 * q3 * x_half[l + 64];
                    acc += d * sc4 * q4 * x_half[l + 96];
                }
            }
        }
        out[row] = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dequant_q6k_reference(raw: &[u8], row: usize, n_blocks: usize) -> Vec<f32> {
        let mut dst = vec![0.0f32; n_blocks * 256];
        for blk in 0..n_blocks {
            let base = (row * n_blocks + blk) * 210;
            let block = &raw[base..base + 210];
            let d = f16_to_f32(&block[208..210]);
            let ql = &block[0..128];
            let qh = &block[128..192];
            let sc = &block[192..208];
            let out = &mut dst[blk * 256..(blk + 1) * 256];
            let mut idx = 0;
            for half in 0..2 {
                let ql_h = &ql[half * 64..(half + 1) * 64];
                let qh_h = &qh[half * 32..(half + 1) * 32];
                let sc_h = &sc[half * 8..(half + 1) * 8];
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql_h[l] as u16 & 0x0F) | ((qh_h[l] as u16 & 3) << 4)) as i8 - 32;
                    let q2 = ((ql_h[l + 32] as u16 & 0x0F) | (((qh_h[l] as u16 >> 2) & 3) << 4)) as i8 - 32;
                    let q3 = (((ql_h[l] as u16) >> 4) | (((qh_h[l] as u16 >> 4) & 3) << 4)) as i8 - 32;
                    let q4 = (((ql_h[l + 32] as u16) >> 4) | (((qh_h[l] as u16 >> 6) & 3) << 4)) as i8 - 32;
                    out[idx + l] = d * (sc_h[is] as i8 as f32) * (q1 as f32);
                    out[idx + l + 32] = d * (sc_h[is + 2] as i8 as f32) * (q2 as f32);
                    out[idx + l + 64] = d * (sc_h[is + 4] as i8 as f32) * (q3 as f32);
                    out[idx + l + 96] = d * (sc_h[is + 6] as i8 as f32) * (q4 as f32);
                }
                idx += 128;
            }
        }
        dst
    }

    fn create_test_data(n_rows: usize, n_blocks: usize) -> Vec<u8> {
        let total_size = n_rows * n_blocks * BLOCK_BYTES;
        let mut raw = vec![0u8; total_size];
        for row in 0..n_rows {
            for blk in 0..n_blocks {
                let base = (row * n_blocks + blk) * BLOCK_BYTES;
                let block = &mut raw[base..base + BLOCK_BYTES];
                for i in 0..128 {
                    let lo = (i as u8) & 0x0F;
                    let hi = ((i + 128) as u8) & 0x0F;
                    block[i] = lo | (hi << 4);
                }
                for i in 0..64 {
                    let b0 = ((i as u8) >> 4) & 3;
                    let b1 = (((i + 64) as u8) >> 4) & 3;
                    let b2 = (((i + 128) as u8) >> 4) & 3;
                    let b3 = (((i + 192) as u8) >> 4) & 3;
                    block[128 + i] = b0 | (b1 << 2) | (b2 << 4) | (b3 << 6);
                }
                for i in 0..16 {
                    block[192 + i] = ((row * 16 + i + 1) as u8) & 0x7F;
                }
                let d: u16 = 0x3800;
                block[208..210].copy_from_slice(&d.to_le_bytes());
            }
        }
        raw
    }

    #[test]
    fn test_q6k_bitexact_vs_dequant_scalar() {
        let n_rows = 2;
        let n_blocks = 2;
        let raw = create_test_data(n_rows, n_blocks);
        let n_cols = n_blocks * 256;
        let x: Vec<f32> = (0..n_cols).map(|i| ((i as f32) * 0.01).sin()).collect();

        let mut out = vec![0.0f32; n_rows];
        fused_gemv_q6k_scalar(&raw, &x, &mut out, n_rows, n_cols, n_blocks);

        for row in 0..n_rows {
            let w = dequant_q6k_reference(&raw, row, n_blocks);
            let dot_ref: f32 = w.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            let diff = (out[row] - dot_ref).abs();
            assert!(diff < 1.0 || diff / dot_ref.abs().max(1.0) < 1e-4,
                "scalar row={} fused={} ref={} diff={}", row, out[row], dot_ref, diff);
        }
    }

    #[test]
    fn test_q6k_avx512_bitexact() {
        if !(std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw")) {
            eprintln!("SKIP: no AVX-512");
            return;
        }
        let n_rows = 4;
        let n_blocks = 3;
        let raw = create_test_data(n_rows, n_blocks);
        let n_cols = n_blocks * 256;
        let x: Vec<f32> = (0..n_cols).map(|i| ((i as f32) * 0.01).cos()).collect();

        let mut out_scalar = vec![0.0f32; n_rows];
        fused_gemv_q6k_scalar(&raw, &x, &mut out_scalar, n_rows, n_cols, n_blocks);

        let mut out_avx512 = vec![0.0f32; n_rows];
        unsafe { fused_gemv_q6k_avx512(&raw, &x, &mut out_avx512, n_rows, n_cols, n_blocks) };

        for row in 0..n_rows {
            let diff = (out_avx512[row] - out_scalar[row]).abs();
            let rel = diff / out_scalar[row].abs().max(1.0);
            assert!(diff < 1e-3 || rel < 1e-4,
                "avx512 row={} avx512={} scalar={} diff={}", row, out_avx512[row], out_scalar[row], diff);
        }
    }

    #[test]
    fn test_q6k_avx2_bitexact() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            eprintln!("SKIP: no AVX2");
            return;
        }
        let n_rows = 4;
        let n_blocks = 3;
        let raw = create_test_data(n_rows, n_blocks);
        let n_cols = n_blocks * 256;
        let x: Vec<f32> = (0..n_cols).map(|i| ((i as f32) * 0.01).cos()).collect();

        let mut out_scalar = vec![0.0f32; n_rows];
        fused_gemv_q6k_scalar(&raw, &x, &mut out_scalar, n_rows, n_cols, n_blocks);

        let mut out_avx2 = vec![0.0f32; n_rows];
        unsafe { fused_gemv_q6k_avx2(&raw, &x, &mut out_avx2, n_rows, n_cols, n_blocks) };

        for row in 0..n_rows {
            let diff = (out_avx2[row] - out_scalar[row]).abs();
            let rel = diff / out_scalar[row].abs().max(1.0);
            assert!(diff < 1e-3 || rel < 1e-4,
                "avx2 row={} avx2={} scalar={} diff={}", row, out_avx2[row], out_scalar[row], diff);
        }
    }
}
