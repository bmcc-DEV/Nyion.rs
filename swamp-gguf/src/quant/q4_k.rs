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
