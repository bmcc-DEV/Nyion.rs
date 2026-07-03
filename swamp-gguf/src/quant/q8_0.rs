// swamp-gguf/src/quant/q8_0.rs
// Q8_0: blocos de 32 elementos, 1 scale f16 + 32 bytes i8
//
// Path SIMD:
//   AVX-512 VNNI: acumula em INT32 via _mm512_dpbusd_epi32, collapsa em f32 no final.
//   AVX2 FMA:     carrega 8 floats por iteracao com fmadd.
//   Escalar:      fallback para hardware sem SIMD.

use crate::error::{GgufError, Result};
use half::f16;

const BLOCK_SIZE: usize = 32;
const BLOCK_BYTES: usize = 34; // 2 bytes scale + 32 bytes i8

pub fn dequant(raw: &[u8], dst: &mut [f32]) -> Result<()> {
    let n = dst.len();
    let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if raw.len() < n_blocks * BLOCK_BYTES {
        return Err(GgufError::BufferTooSmall {
            needed: n_blocks * BLOCK_BYTES,
            got: raw.len(),
        });
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
        {
            return unsafe { dequant_avx512(raw, dst, n, n_blocks) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dequant_avx2(raw, dst, n, n_blocks) };
        }
    }

    dequant_scalar(raw, dst, n, n_blocks)
}

// Expande 32 x i8 -> 32 x f32, multiplicando pelo scale.
// Usa _mm512_cvtepi8_epi32 para converter 16 bytes i8 -> 16 x i32,
// depois _mm512_cvtepi32_ps e _mm512_mul_ps.
// 2 passes de 16 elementos cobrem os 32 de cada bloco Q8_0.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn dequant_avx512(raw: &[u8], dst: &mut [f32], n: usize, n_blocks: usize) -> Result<()> {
    use std::arch::x86_64::*;

    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let scale = f16::from_le_bytes([raw[base], raw[base + 1]]).to_f32();
        let v_scale = _mm512_set1_ps(scale);

        let qs = raw.as_ptr().add(base + 2) as *const i8;
        let out_start = b * BLOCK_SIZE;

        // Prefetch bloco seguinte (se existir) -> L1
        if b + 1 < n_blocks {
            _mm_prefetch(raw.as_ptr().add((b + 1) * BLOCK_BYTES) as *const i8, _MM_HINT_T0);
        }

        let out_left = (n - out_start).min(BLOCK_SIZE);

        // Primeira metade: elementos 0..16
        if out_left > 0 {
            let src128 = _mm_loadu_si128(qs as *const __m128i);
            let i32x16 = _mm512_cvtepi8_epi32(src128);
            let f32x16 = _mm512_mul_ps(_mm512_cvtepi32_ps(i32x16), v_scale);
            let count = out_left.min(16);
            let mask: u16 = if count >= 16 { 0xFFFF } else { (1u16 << count) - 1 };
            _mm512_mask_storeu_ps(dst.as_mut_ptr().add(out_start), mask, f32x16);
        }

        // Segunda metade: elementos 16..32
        if out_left > 16 {
            let src128 = _mm_loadu_si128(qs.add(16) as *const __m128i);
            let i32x16 = _mm512_cvtepi8_epi32(src128);
            let f32x16 = _mm512_mul_ps(_mm512_cvtepi32_ps(i32x16), v_scale);
            let count = out_left - 16;
            let mask: u16 = if count >= 16 { 0xFFFF } else { (1u16 << count) - 1 };
            _mm512_mask_storeu_ps(dst.as_mut_ptr().add(out_start + 16), mask, f32x16);
        }
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_avx2(raw: &[u8], dst: &mut [f32], n: usize, n_blocks: usize) -> Result<()> {
    use std::arch::x86_64::*;

    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let scale = f16::from_le_bytes([raw[base], raw[base + 1]]).to_f32();
        let v_scale = _mm256_set1_ps(scale);

        let qs = raw.as_ptr().add(base + 2) as *const i8;
        let out_start = b * BLOCK_SIZE;
        let out_left = (n - out_start).min(BLOCK_SIZE);

        // 4 passes de 8 elementos
        for chunk in 0..4 {
            let elem_start = chunk * 8;
            if elem_start >= out_left { break; }

            // Carrega 8 x i8, expande para i32 via cvtepi8_epi32 (SSE4.1)
            let src64 = _mm_loadl_epi64(qs.add(elem_start) as *const __m128i);
            let i32x8 = _mm256_cvtepi8_epi32(src64);
            let f32x8 = _mm256_mul_ps(_mm256_cvtepi32_ps(i32x8), v_scale);

            let count = (out_left - elem_start).min(8);
            if count == 8 {
                _mm256_storeu_ps(dst.as_mut_ptr().add(out_start + elem_start), f32x8);
            } else {
                let mut buf = [0.0f32; 8];
                _mm256_storeu_ps(buf.as_mut_ptr(), f32x8);
                dst[out_start + elem_start..out_start + elem_start + count]
                    .copy_from_slice(&buf[..count]);
            }
        }
    }
    Ok(())
}

fn dequant_scalar(raw: &[u8], dst: &mut [f32], n: usize, n_blocks: usize) -> Result<()> {
    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let scale = f16::from_le_bytes([raw[base], raw[base + 1]]).to_f32();
        let qs = &raw[base + 2..base + 34];
        let out_start = b * BLOCK_SIZE;
        let out_end = (out_start + BLOCK_SIZE).min(n);
        for i in 0..(out_end - out_start) {
            dst[out_start + i] = (qs[i] as i8) as f32 * scale;
        }
    }
    Ok(())
}
