// swamp-gguf/src/quant/iq2_xxs.rs
// IQ2_XXS: quantizacao imatrix ~2 bits/peso (66 bytes por bloco de 256 elementos)
// Layout:
//   d:   f16 (2 bytes)
//   qs:  [u8; 64] - 32 grupos de 8 elementos via grid lookup

use crate::error::{GgufError, Result};
use half::f16;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;


const BLOCK_SIZE: usize = 256;
const BLOCK_BYTES: usize = 66;

// Grid IQ2_XXS de referencia (256 entradas x 8 valores int8)
// Cada u64 empacota 8 valores de 8 bits: grid[i][j] = (entry >> 8*j) & 0xFF
// Derivado de ggml/src/ggml-quants.c: iq2xxs_grid
static GRID_IQ2_XXS: [u64; 256] = [
    0x0808080808080808, 0x080808080808082b, 0x0808080808082108, 0x080808080808212b,
    0x0808080808210808, 0x080808080821082b, 0x0808080808212108, 0x080808080821212b,
    0x0808080821080808, 0x080808082108082b, 0x0808080821082108, 0x080808082108212b,
    0x0808080821210808, 0x080808082121082b, 0x0808080821212108, 0x080808082121212b,
    0x0808082108080808, 0x080808210808082b, 0x0808082108082108, 0x080808210808212b,
    0x0808082108210808, 0x080808210821082b, 0x0808082108212108, 0x080808210821212b,
    0x0808082121080808, 0x080808212108082b, 0x0808082121082108, 0x080808212108212b,
    0x0808082121210808, 0x080808212121082b, 0x0808082121212108, 0x080808212121212b,
    0x0821080808080808, 0x082108080808082b, 0x0821080808082108, 0x082108080808212b,
    0x0821080808210808, 0x082108080821082b, 0x0821080808212108, 0x082108080821212b,
    0x0821080821080808, 0x082108082108082b, 0x0821080821082108, 0x082108082108212b,
    0x0821080821210808, 0x082108082121082b, 0x0821080821212108, 0x082108082121212b,
    0x0821082108080808, 0x082108210808082b, 0x0821082108082108, 0x082108210808212b,
    0x0821082108210808, 0x082108210821082b, 0x0821082108212108, 0x082108210821212b,
    0x0821082121080808, 0x082108212108082b, 0x0821082121082108, 0x082108212108212b,
    0x0821082121210808, 0x082108212121082b, 0x0821082121212108, 0x082108212121212b,
    0x2108080808080808, 0x210808080808082b, 0x2108080808082108, 0x210808080808212b,
    0x2108080808210808, 0x210808080821082b, 0x2108080808212108, 0x210808080821212b,
    0x2108080821080808, 0x210808082108082b, 0x2108080821082108, 0x210808082108212b,
    0x2108080821210808, 0x210808082121082b, 0x2108080821212108, 0x210808082121212b,
    0x2108082108080808, 0x210808210808082b, 0x2108082108082108, 0x210808210808212b,
    0x2108082108210808, 0x210808210821082b, 0x2108082108212108, 0x210808210821212b,
    0x2108082121080808, 0x210808212108082b, 0x2108082121082108, 0x210808212108212b,
    0x2108082121210808, 0x210808212121082b, 0x2108082121212108, 0x210808212121212b,
    0x2121080808080808, 0x212108080808082b, 0x2121080808082108, 0x212108080808212b,
    0x2121080808210808, 0x212108080821082b, 0x2121080808212108, 0x212108080821212b,
    0x2121080821080808, 0x212108082108082b, 0x2121080821082108, 0x212108082108212b,
    0x2121080821210808, 0x212108082121082b, 0x2121080821212108, 0x212108082121212b,
    0x2121082108080808, 0x212108210808082b, 0x2121082108082108, 0x212108210808212b,
    0x2121082108210808, 0x212108210821082b, 0x2121082108212108, 0x212108210821212b,
    0x2121082121080808, 0x212108212108082b, 0x2121082121082108, 0x212108212108212b,
    0x2121082121210808, 0x212108212121082b, 0x2121082121212108, 0x212108212121212b,
    0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808082108, 0x2b0808080808212b,
    0x2b08080808210808, 0x2b0808080821082b, 0x2b08080808212108, 0x2b0808080821212b,
    0x2b08080821080808, 0x2b0808082108082b, 0x2b08080821082108, 0x2b0808082108212b,
    0x2b08080821210808, 0x2b0808082121082b, 0x2b08080821212108, 0x2b0808082121212b,
    0x2b08082108080808, 0x2b08082108082b08, 0x2b0808210808082b, 0x2b08082108210808,
    0x2b08082108212108, 0x2b08082121080808, 0x2b0808212108082b, 0x2b08082121082108,
    0x2b08082121210808, 0x2b08210808080808, 0x2b08210808082108, 0x2b08210808212108,
    0x2b08210821080808, 0x2b0821082108082b, 0x2b08210821082108, 0x2b08210821212108,
    0x2b08212108080808, 0x2b08212108082b08, 0x2b0821210808082b, 0x2b08212108210808,
    0x2b08212108212108, 0x2b08212121080808, 0x2b0821212108082b, 0x2b08212121082108,
    0x2b08212121210808, 0x2b21080808080808, 0x2b2108080808082b, 0x2b21080808082108,
    0x2b21080808210808, 0x2b21080808212108, 0x2b21080821080808, 0x2b2108082108082b,
    0x2b21080821082108, 0x2b21080821210808, 0x2b21080821212108, 0x2b21082108080808,
    0x2b2108210808082b, 0x2b21082108082108, 0x2b21082108210808, 0x2b21082108212108,
    0x2b21082121080808, 0x2b2108212108082b, 0x2b21082121082108, 0x2b21082121210808,
    0x2b21210808080808, 0x2b2121080808082b, 0x2b21210808082108, 0x2b21210808210808,
    0x2b21210808212108, 0x2b21210821080808, 0x2b2121082108082b, 0x2b21210821082108,
    0x2b21210821210808, 0x2b21212108080808, 0x2b2121210808082b, 0x2b21212108082108,
    0x2b21212108210808, 0x2b21212121080808, 0x2b21212121210808, 0x2b21212121212108,
    0x2b21212121212121, 0x2b21212121212b21, 0x2b212121212121ab, 0x2b2121212121ab21,
    0xab08080808080808, 0xab0808080808082b, 0xab08080808082108, 0xab0808080808212b,
    0xab08080808210808, 0xab0808080821082b, 0xab08080808212108, 0xab0808080821212b,
    0xab08080821080808, 0xab0808082108082b, 0xab08080821082108, 0xab0808082108212b,
    0xab08080821210808, 0xab0808082121082b, 0xab08080821212108, 0xab0808082121212b,
    0xab08082108080808, 0xab0808210808082b, 0xab08082108082108, 0xab0808210808212b,
    0xab08082108210808, 0xab0808210821082b, 0xab08082108212108, 0xab0808210821212b,
    0xab08082121080808, 0xab0808212108082b, 0xab08082121082108, 0xab0808212108212b,
    0xab08082121210808, 0xab0808212121082b, 0xab08082121212108, 0xab0808212121212b,
    0xab21080808080808, 0xab2108080808082b, 0xab21080808082108, 0xab2108080808212b,
    0xab21080808210808, 0xab2108080821082b, 0xab21080808212108, 0xab2108080821212b,
    0xab21080821080808, 0xab2108082108082b, 0xab21080821082108, 0xab2108082108212b,
    0xab21080821210808, 0xab2108082121082b, 0xab21080821212108, 0xab2108082121212b,
];

use std::sync::OnceLock;

static GRID_IQ2_XXS_F32: OnceLock<[f32; 2048]> = OnceLock::new();

fn get_grid_iq2_xxs_f32() -> &'static [f32; 2048] {
    GRID_IQ2_XXS_F32.get_or_init(|| {
        let mut grid = [0.0f32; 2048];
        for i in 0..256 {
            let entry = GRID_IQ2_XXS[i];
            for j in 0..8 {
                grid[i * 8 + j] = ((entry >> (8 * j)) & 0x7F) as i8 as f32;
            }
        }
        grid
    })
}

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
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
            unsafe {
                let lut = get_grid_iq2_xxs_f32();
                for b in 0..n_blocks {
                    if b + 1 < n_blocks {
                        _mm_prefetch(raw.as_ptr().add((b + 1) * BLOCK_BYTES) as *const i8, _MM_HINT_T0);
                    }
                    let base = b * BLOCK_BYTES;
                    let d = f16::from_le_bytes([raw[base], raw[base + 1]]).to_f32();
                    let qs = &raw[base + 2..base + 66];
                    let out_start = b * BLOCK_SIZE;
                    let out_end = (out_start + BLOCK_SIZE).min(n);

                    // 16 loops (processa 16 elementos por iteracao = 2 grupos)
                    for g in 0..16 {
                        let g0 = g * 2;
                        let g1 = g0 + 1;
                        let global = out_start + g0 * 8;
                        if global >= out_end { break; }

                        let grid_idx0 = qs[g0 * 2] as i32;
                        let signs_sc0 = qs[g0 * 2 + 1];
                        let scale0 = d * (0.5 + (signs_sc0 & 0x0F) as f32) * 0.25;

                        let grid_idx1 = qs[g1 * 2] as i32;
                        let signs_sc1 = qs[g1 * 2 + 1];
                        let scale1 = d * (0.5 + (signs_sc1 & 0x0F) as f32) * 0.25;

                        let mut indices = [0i32; 16];
                        for j in 0..8 {
                            indices[j] = grid_idx0 * 8 + j as i32;
                            indices[j + 8] = grid_idx1 * 8 + j as i32;
                        }

                        let mut signs = [1.0f32; 16];
                        for j in 0..8 {
                            if ((signs_sc0 >> (4 + j / 2)) & 1) != 0 {
                                signs[j] = -1.0;
                            }
                            if ((signs_sc1 >> (4 + j / 2)) & 1) != 0 {
                                signs[j + 8] = -1.0;
                            }
                        }

                        let v_indices = _mm512_loadu_si512(indices.as_ptr() as *const __m512i);
                        let v_signs = _mm512_loadu_ps(signs.as_ptr());
                        let v_grid = _mm512_i32gather_ps(v_indices, lut.as_ptr(), 4);

                        let mut scales = [0.0f32; 16];
                        scales[..8].fill(scale0);
                        scales[8..].fill(scale1);
                        let v_scales = _mm512_loadu_ps(scales.as_ptr());

                        let v_out = _mm512_mul_ps(_mm512_mul_ps(v_grid, v_scales), v_signs);

                        let count = (out_end - global).min(16);
                        if count >= 16 {
                            _mm512_storeu_ps(dst.as_mut_ptr().add(global), v_out);
                        } else {
                            let mut buf = [0.0f32; 16];
                            _mm512_storeu_ps(buf.as_mut_ptr(), v_out);
                            std::ptr::copy_nonoverlapping(buf.as_ptr(), dst.as_mut_ptr().add(global), count);
                        }
                    }
                }
                return Ok(());
            }
        }

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let lut = get_grid_iq2_xxs_f32();
                for b in 0..n_blocks {
                    if b + 1 < n_blocks {
                        _mm_prefetch(raw.as_ptr().add((b + 1) * BLOCK_BYTES) as *const i8, _MM_HINT_T0);
                    }
                    let base = b * BLOCK_BYTES;
                    let d = f16::from_le_bytes([raw[base], raw[base + 1]]).to_f32();
                    let qs = &raw[base + 2..base + 66];
                    let out_start = b * BLOCK_SIZE;
                    let out_end = (out_start + BLOCK_SIZE).min(n);

                    for g in 0..32 {
                        let grid_idx = qs[g * 2] as usize;
                        let signs_sc = qs[g * 2 + 1];
                        let scale = d * (0.5 + (signs_sc & 0x0F) as f32) * 0.25;

                        let v_grid = _mm256_loadu_ps(lut.as_ptr().add(grid_idx * 8));

                        let mut signs = [1.0f32; 8];
                        for j in 0..8 {
                            if ((signs_sc >> (4 + j / 2)) & 1) != 0 {
                                signs[j] = -1.0;
                            }
                        }
                        let v_signs = _mm256_loadu_ps(signs.as_ptr());
                        let v_scale = _mm256_set1_ps(scale);
                        let v_out = _mm256_mul_ps(_mm256_mul_ps(v_grid, v_scale), v_signs);

                        let global = out_start + g * 8;
                        if global + 8 <= out_end {
                            _mm256_storeu_ps(dst.as_mut_ptr().add(global), v_out);
                        } else {
                            let mut buf = [0.0f32; 8];
                            _mm256_storeu_ps(buf.as_mut_ptr(), v_out);
                            for j in 0..8 {
                                if global + j < out_end { dst[global + j] = buf[j]; }
                            }
                        }
                    }
                }
                return Ok(());
            }
        }
    }

    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let d    = f16::from_le_bytes([raw[base], raw[base + 1]]).to_f32();
        let qs   = &raw[base + 2..base + 66];
        let out_start = b * BLOCK_SIZE;
        let out_end   = (out_start + BLOCK_SIZE).min(n);

        for g in 0..32 {
            let grid_idx   = qs[g * 2] as usize;
            let signs_sc   = qs[g * 2 + 1];
            let scale_nibble = (signs_sc & 0x0F) as f32;
            let scale = d * (0.5 + scale_nibble) * 0.25;

            let entry = GRID_IQ2_XXS[grid_idx];

            for j in 0..8 {
                let global = out_start + g * 8 + j;
                if global >= out_end { break; }

                let grid_val = ((entry >> (8 * j)) & 0x7F) as i8;
                let sign_bit = (signs_sc >> (4 + j / 2)) & 1;
                let sign = if sign_bit != 0 { -1.0f32 } else { 1.0 };
                dst[global] = scale * sign * grid_val as f32;
            }
        }
    }
    Ok(())
}

