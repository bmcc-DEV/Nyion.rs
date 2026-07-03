// swamp-gguf/src/quant/q2_k.rs
// Q2_K: blocos de 256 elementos, 2 bits por elemento
// Layout (84 bytes):
//   scales: [u8; 16] - 16 sub-escalas de 4 bits (par: scale, min)
//   qs:     [u8; 64] - 256 x 2-bit values
//   d:      f16 (2 bytes)
//   dmin:   f16 (2 bytes)

use crate::error::{GgufError, Result};
use half::f16;

const BLOCK_SIZE: usize = 256;
const BLOCK_BYTES: usize = 84;

pub fn dequant_q2k(raw: &[u8], dst: &mut [f32]) -> Result<()> {
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
        let sc    = &raw[base..base + 16];
        let qs    = &raw[base + 16..base + 80];
        let d     = f16::from_le_bytes([raw[base + 80], raw[base + 81]]).to_f32();
        let dmin  = f16::from_le_bytes([raw[base + 82], raw[base + 83]]).to_f32();

        let out_start = b * BLOCK_SIZE;
        let out_end   = (out_start + BLOCK_SIZE).min(n);

        // 16 sub-blocos de 16 elementos cada (256 / 16 = 16)
        for sb in 0..16 {
            // scale e min empacotados: low nibble = scale, high nibble = min
            let sc_byte  = sc[sb / 2];
            let scale_nibble = if sb % 2 == 0 { sc_byte & 0x0F } else { sc_byte >> 4 };
            let scale = d * scale_nibble as f32;

            // min: bytes 8..15 no mesmo formato
            let min_byte = sc[8 + sb / 2];
            let min_nibble = if sb % 2 == 0 { min_byte & 0x0F } else { min_byte >> 4 };
            let min_val = dmin * min_nibble as f32;

            // cada sub-bloco tem 16 elementos de 2 bits
            // qs tem 64 bytes = 256 bits = 256 x 2-bit
            // sub-bloco sb ocupa qs[sb*4..(sb+1)*4]
            for i in 0..16 {
                let global = out_start + sb * 16 + i;
                if global >= out_end { break; }

                let qi_byte = qs[sb * 4 + i / 4];
                let shift = (i % 4) * 2;
                let qi = (qi_byte >> shift) & 0x03;
                dst[global] = scale * qi as f32 - min_val;
            }
        }
    }
    Ok(())
}
