// swamp-gguf/src/quant/q4_k.rs
// Q4_K: blocos de 256 elementos, super-bloco com sub-escalas 6-bit
//
// Layout (144 bytes / 256 elementos):
//   d:      f16 (2)   - scale geral
//   dmin:   f16 (2)   - min geral
//   scales: [u8; 12]  - 8 pares (scale, min) de 6 bits
//   qs:     [u8; 128] - 256 nibbles (4 bits cada)
//
// Path SIMD:
//   AVX-512 BW+VL: desempacota nibbles com shuffle de mascaras e fmsub_ps.
//   AVX2 FMA:      expande nibbles em 8 floats com fmsub.
//   Escalar:        fallback.

use crate::error::{GgufError, Result};
use half::f16;

const BLOCK_SIZE: usize = 256;
const BLOCK_BYTES: usize = 144;

pub fn dequant_q4k(raw: &[u8], dst: &mut [f32]) -> Result<()> {
    let n = dst.len();
    let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if raw.len() < n_blocks * BLOCK_BYTES {
        return Err(GgufError::BufferTooSmall {
            needed: n_blocks * BLOCK_BYTES,
            got: raw.len(),
        });
    }

    dequant_scalar(raw, dst, n, n_blocks)
}

// Extrai scales/mins (logica identica nos tres paths - sem SIMD, so 12 bytes)
#[inline(always)]
pub fn unpack_scales(sc: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];

    // For j = 0..3
    for j in 0..4 {
        scales[j] = sc[j] & 63;
        mins[j]   = sc[j + 4] & 63;
    }
    // For j = 4..7
    for j in 4..8 {
        scales[j] = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
        mins[j]   = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }

    (scales, mins)
}

// AVX-512 path: processa cada sub-bloco de 32 nibbles em dois passes AVX2 de 16 elementos.
// Dentro de cada passe:
//   1. Carrega 16 bytes compactados (= 32 nibbles, mas neste layout usamos 16 low + 16 high).
//   2. Separa low nibble (AND 0x0F) e high nibble (SHR 4 AND 0x0F).
//   3. Converte u8->i32->f32 em registradores de 512 bits.
//   4. Aplica fmsub: scale * q - min (sub em vez de subtrair depois).
// AVX-512 path: processa cada sub-bloco de 32 nibbles.
// Otimizado: carrega 64 bytes de uma vez com _mm512_loadu_si512, separando
// os nibbles em dois registradores de 512 bits (low/high). Extrai os sub-blocos
// via _mm512_extracti32x4_epi32 de forma totalmente desenrolada para maximo IPC.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn dequant_avx512(raw: &[u8], dst: &mut [f32], n: usize, n_blocks: usize) -> Result<()> {
    use std::arch::x86_64::*;

    macro_rules! dequant_subblock {
        ($lo_bytes:expr, $hi_bytes:expr, $sv:expr, $mv:expr, $dst:expr, $out_start:expr, $sb:expr, $out_left:expr, $nt:expr) => {
            let v_scale = _mm512_set1_ps($sv);
            let v_min   = _mm512_set1_ps($mv);

            let lo_i32x16 = _mm512_cvtepu8_epi32($lo_bytes);
            let lo_f32x16 = _mm512_cvtepi32_ps(lo_i32x16);
            let lo_out = _mm512_fmsub_ps(v_scale, lo_f32x16, v_min);

            let hi_i32x16 = _mm512_cvtepu8_epi32($hi_bytes);
            let hi_f32x16 = _mm512_cvtepi32_ps(hi_i32x16);
            let hi_out = _mm512_fmsub_ps(v_scale, hi_f32x16, v_min);

            let dst_lo = $dst.as_mut_ptr().add($out_start + $sb * 32);
            let dst_hi = $dst.as_mut_ptr().add($out_start + $sb * 32 + 16);

            let lo_count = ($out_left.saturating_sub($sb * 32)).min(16);
            if lo_count > 0 {
                let mask_lo: u16 = if lo_count >= 16 { 0xFFFF } else { (1u16 << lo_count) - 1 };
                _mm512_mask_storeu_ps(dst_lo, mask_lo, lo_out);
            }

            let hi_base = $sb * 32 + 16;
            if hi_base < $out_left {
                let hi_count = ($out_left - hi_base).min(16);
                let mask_hi: u16 = if hi_count >= 16 { 0xFFFF } else { (1u16 << hi_count) - 1 };
                _mm512_mask_storeu_ps(dst_hi, mask_hi, hi_out);
            }
        };
    }

    // dst_aligned: reservado para uso futuro com clflushopt ou prefetchw
    let dst_aligned = (dst.as_ptr() as usize) % 64 == 0;
    let _ = dst_aligned;


    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let d    = f16::from_le_bytes([raw[base],     raw[base + 1]]).to_f32();
        let dmin = f16::from_le_bytes([raw[base + 2], raw[base + 3]]).to_f32();
        let sc   = &raw[base + 4..base + 16];
        let qs   = raw.as_ptr().add(base + 16); // 128 bytes compactados

        if b + 1 < n_blocks {
            _mm_prefetch(raw.as_ptr().add((b + 1) * BLOCK_BYTES) as *const i8, _MM_HINT_T0);
        }

        let (scales, mins) = unpack_scales(sc);
        let out_start = b * BLOCK_SIZE;
        let out_left  = (n - out_start).min(BLOCK_SIZE);

        let mask_nibble512 = _mm512_set1_epi8(0x0F);

        // Bloco 0 (sub-blocos 0..4, bytes 0..64 de qs)
        if 0 < out_left {
            let packed512 = _mm512_loadu_si512(qs as *const __m512i);
            let lo_bytes512 = _mm512_and_si512(packed512, mask_nibble512);
            let hi_bytes512 = _mm512_and_si512(_mm512_srli_epi16(packed512, 4), mask_nibble512);

            // Sub-bloco 0
            if 0 * 32 < out_left {
                let sv = d    * (scales[0] & 0x3F) as f32;
                let mv = dmin * (mins[0]   & 0x3F) as f32;
                let lo_bytes = _mm512_extracti32x4_epi32::<0>(lo_bytes512);
                let hi_bytes = _mm512_extracti32x4_epi32::<0>(hi_bytes512);
                dequant_subblock!(lo_bytes, hi_bytes, sv, mv, dst, out_start, 0, out_left, dst_aligned);
            }
            // Sub-bloco 1
            if 1 * 32 < out_left {
                let sv = d    * (scales[1] & 0x3F) as f32;
                let mv = dmin * (mins[1]   & 0x3F) as f32;
                let lo_bytes = _mm512_extracti32x4_epi32::<1>(lo_bytes512);
                let hi_bytes = _mm512_extracti32x4_epi32::<1>(hi_bytes512);
                dequant_subblock!(lo_bytes, hi_bytes, sv, mv, dst, out_start, 1, out_left, dst_aligned);
            }
            // Sub-bloco 2
            if 2 * 32 < out_left {
                let sv = d    * (scales[2] & 0x3F) as f32;
                let mv = dmin * (mins[2]   & 0x3F) as f32;
                let lo_bytes = _mm512_extracti32x4_epi32::<2>(lo_bytes512);
                let hi_bytes = _mm512_extracti32x4_epi32::<2>(hi_bytes512);
                dequant_subblock!(lo_bytes, hi_bytes, sv, mv, dst, out_start, 2, out_left, dst_aligned);
            }
            // Sub-bloco 3
            if 3 * 32 < out_left {
                let sv = d    * (scales[3] & 0x3F) as f32;
                let mv = dmin * (mins[3]   & 0x3F) as f32;
                let lo_bytes = _mm512_extracti32x4_epi32::<3>(lo_bytes512);
                let hi_bytes = _mm512_extracti32x4_epi32::<3>(hi_bytes512);
                dequant_subblock!(lo_bytes, hi_bytes, sv, mv, dst, out_start, 3, out_left, dst_aligned);
            }
        }

        // Bloco 1 (sub-blocos 4..8, bytes 64..128 de qs)
        if 4 * 32 < out_left {
            let packed512 = _mm512_loadu_si512(qs.add(64) as *const __m512i);
            let lo_bytes512 = _mm512_and_si512(packed512, mask_nibble512);
            let hi_bytes512 = _mm512_and_si512(_mm512_srli_epi16(packed512, 4), mask_nibble512);

            // Sub-bloco 4
            if 4 * 32 < out_left {
                let sv = d    * (scales[4] & 0x3F) as f32;
                let mv = dmin * (mins[4]   & 0x3F) as f32;
                let lo_bytes = _mm512_extracti32x4_epi32::<0>(lo_bytes512);
                let hi_bytes = _mm512_extracti32x4_epi32::<0>(hi_bytes512);
                dequant_subblock!(lo_bytes, hi_bytes, sv, mv, dst, out_start, 4, out_left, dst_aligned);
            }
            // Sub-bloco 5
            if 5 * 32 < out_left {
                let sv = d    * (scales[5] & 0x3F) as f32;
                let mv = dmin * (mins[5]   & 0x3F) as f32;
                let lo_bytes = _mm512_extracti32x4_epi32::<1>(lo_bytes512);
                let hi_bytes = _mm512_extracti32x4_epi32::<1>(hi_bytes512);
                dequant_subblock!(lo_bytes, hi_bytes, sv, mv, dst, out_start, 5, out_left, dst_aligned);
            }
            // Sub-bloco 6
            if 6 * 32 < out_left {
                let sv = d    * (scales[6] & 0x3F) as f32;
                let mv = dmin * (mins[6]   & 0x3F) as f32;
                let lo_bytes = _mm512_extracti32x4_epi32::<2>(lo_bytes512);
                let hi_bytes = _mm512_extracti32x4_epi32::<2>(hi_bytes512);
                dequant_subblock!(lo_bytes, hi_bytes, sv, mv, dst, out_start, 6, out_left, dst_aligned);
            }
            // Sub-bloco 7
            if 7 * 32 < out_left {
                let sv = d    * (scales[7] & 0x3F) as f32;
                let mv = dmin * (mins[7]   & 0x3F) as f32;
                let lo_bytes = _mm512_extracti32x4_epi32::<3>(lo_bytes512);
                let hi_bytes = _mm512_extracti32x4_epi32::<3>(hi_bytes512);
                dequant_subblock!(lo_bytes, hi_bytes, sv, mv, dst, out_start, 7, out_left, dst_aligned);
            }
        }
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_avx2(raw: &[u8], dst: &mut [f32], n: usize, n_blocks: usize) -> Result<()> {
    use std::arch::x86_64::*;

    let mask_nibble = _mm_set1_epi8(0x0Fi8);

    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let d    = f16::from_le_bytes([raw[base],     raw[base + 1]]).to_f32();
        let dmin = f16::from_le_bytes([raw[base + 2], raw[base + 3]]).to_f32();
        let sc   = &raw[base + 4..base + 16];
        let qs   = raw.as_ptr().add(base + 16);

        let (scales, mins) = unpack_scales(sc);
        let out_start = b * BLOCK_SIZE;
        let out_left  = (n - out_start).min(BLOCK_SIZE);

        for sb in 0..8usize {
            if sb * 32 >= out_left { break; }
            let sv = d    * (scales[sb] & 0x3F) as f32;
            let mv = dmin * (mins[sb]   & 0x3F) as f32;
            let v_scale = _mm256_set1_ps(sv);
            let v_min   = _mm256_set1_ps(mv);

            let qs_sb = qs.add(sb * 16) as *const __m128i;
            let packed = _mm_loadu_si128(qs_sb);

            let lo_bytes = _mm_and_si128(packed, mask_nibble);
            let hi_bytes = _mm_and_si128(_mm_srli_epi16(packed, 4), mask_nibble);

            // Cada metade de 16 nibbles: processar em 2 chunks de 8
            for half in 0..2usize {
                let bytes = if half == 0 { lo_bytes } else { hi_bytes };
                let elem_base = sb * 32 + half * 16;

                // Primeiro chunk: elementos 0..8
                let src_lo = _mm_loadl_epi64(&bytes as *const __m128i);
                let i32x8 = _mm256_cvtepu8_epi32(src_lo);
                let f32x8 = _mm256_fmsub_ps(v_scale, _mm256_cvtepi32_ps(i32x8), v_min);
                let count0 = (out_left.saturating_sub(elem_base)).min(8);
                store_ps_partial(dst.as_mut_ptr().add(out_start + elem_base), f32x8, count0);

                // Segundo chunk: elementos 8..16
                if elem_base + 8 < out_left {
                    // shift right 8 bytes para obter os 8 bytes superiores
                    let src_hi = _mm_srli_si128(bytes, 8);
                    let i32x8h = _mm256_cvtepu8_epi32(src_hi);
                    let f32x8h = _mm256_fmsub_ps(v_scale, _mm256_cvtepi32_ps(i32x8h), v_min);
                    let count1 = (out_left.saturating_sub(elem_base + 8)).min(8);
                    store_ps_partial(dst.as_mut_ptr().add(out_start + elem_base + 8), f32x8h, count1);
                }
            }
        }
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn store_ps_partial(ptr: *mut f32, v: std::arch::x86_64::__m256, count: usize) {
    use std::arch::x86_64::*;
    if count == 0 { return; }
    if count >= 8 {
        _mm256_storeu_ps(ptr, v);
    } else {
        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), v);
        std::ptr::copy_nonoverlapping(buf.as_ptr(), ptr, count);
    }
}

fn dequant_scalar(raw: &[u8], dst: &mut [f32], n: usize, n_blocks: usize) -> Result<()> {
    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let d    = f16::from_le_bytes([raw[base],     raw[base + 1]]).to_f32();
        let dmin = f16::from_le_bytes([raw[base + 2], raw[base + 3]]).to_f32();
        let sc   = &raw[base + 4..base + 16];
        let qs   = &raw[base + 16..base + 144];

        let (scales, mins) = unpack_scales(sc);
        let out_start = b * BLOCK_SIZE;
        let out_end   = (out_start + BLOCK_SIZE).min(n);

        // 4 pares de subblocks (0,1), (2,3), (4,5), (6,7)
        for pair in 0..4 {
            let q_base = &qs[pair * 32..(pair + 1) * 32];
            
            // Subblock PAR (sb = pair * 2) -> lower nibbles
            let sb_even = pair * 2;
            let scale_even = d * (scales[sb_even] & 0x3F) as f32;
            let min_even   = dmin * (mins[sb_even] & 0x3F) as f32;
            
            for i in 0..32 {
                let global = out_start + sb_even * 32 + i;
                if global < out_end {
                    let qi = q_base[i] & 0x0F;
                    dst[global] = scale_even * qi as f32 - min_even;
                }
            }

            // Subblock IMPAR (sb = pair * 2 + 1) -> upper nibbles
            let sb_odd = pair * 2 + 1;
            let scale_odd = d * (scales[sb_odd] & 0x3F) as f32;
            let min_odd   = dmin * (mins[sb_odd] & 0x3F) as f32;
            
            for i in 0..32 {
                let global = out_start + sb_odd * 32 + i;
                if global < out_end {
                    let qi = q_base[i] >> 4;
                    dst[global] = scale_odd * qi as f32 - min_odd;
                }
            }
        }
    }
    Ok(())
}
