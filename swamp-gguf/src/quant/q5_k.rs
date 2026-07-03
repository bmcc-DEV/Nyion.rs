// swamp-gguf/src/quant/q5_k.rs
// Q5_K: 256 elementos, 5 bits por elemento (176 bytes)
// Layout:
//   d:      f16 (2)
//   dmin:   f16 (2)
//   scales: [u8; 12] - 8 pares (scale, min) de 6 bits
//   qh:     [u8; 32] - high bit de cada elemento
//   qs:     [u8; 128] - low 4 bits de cada elemento

use crate::error::{GgufError, Result};
use half::f16;

const BLOCK_SIZE: usize = 256;
const BLOCK_BYTES: usize = 176;

pub fn dequant_q5k(raw: &[u8], dst: &mut [f32]) -> Result<()> {
    let n = dst.len();
    let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if raw.len() < n_blocks * BLOCK_BYTES {
        return Err(GgufError::BufferTooSmall {
            needed: n_blocks * BLOCK_BYTES,
            got: raw.len(),
        });
    }

    for b in 0..n_blocks {
        let base  = b * BLOCK_BYTES;
        let d     = f16::from_le_bytes([raw[base], raw[base + 1]]).to_f32();
        let dmin  = f16::from_le_bytes([raw[base + 2], raw[base + 3]]).to_f32();
        let sc    = &raw[base + 4..base + 16];
        let qh    = &raw[base + 16..base + 48];
        let qs    = &raw[base + 48..base + 176];

        // Desempacota escalas e mins (mesmo algoritmo Q4_K)
        let mut scales = [0u8; 8];
        let mut mins   = [0u8; 8];
        for i in 0..4 {
            let s0 = sc[i];
            let s1 = sc[i + 4];
            let s2 = sc[i + 8];
            scales[2*i]     = (s0 & 0x3F) | ((s2 & 0x0F) << 6);
            scales[2*i + 1] = (s1 & 0x3F) | ((s2 >> 4) << 6);
            mins[2*i]       = (s0 >> 6) | ((s2 & 0x0F) << 2);
            mins[2*i + 1]   = (s1 >> 6) | ((s2 >> 4) << 2);
        }

        let out_start = b * BLOCK_SIZE;
        let out_end   = (out_start + BLOCK_SIZE).min(n);

        for sb in 0..8 {
            let scale_val = d * (scales[sb] & 0x3F) as f32;
            let min_val   = dmin * (mins[sb] & 0x3F) as f32;

            for i in 0..32 {
                let global = out_start + sb * 32 + i;
                if global >= out_end { break; }

                let elem = sb * 32 + i;
                // low 4 bits
                let ql = if i < 16 { qs[sb * 16 + i] & 0x0F } else { qs[sb * 16 + (i - 16)] >> 4 };
                // high bit de qh
                let hb_byte = qh[elem / 8];
                let hb = (hb_byte >> (elem % 8)) & 0x01;
                let qi = (ql as u32) | ((hb as u32) << 4);
                dst[global] = scale_val * qi as f32 - min_val;
            }
        }
    }
    Ok(())
}
