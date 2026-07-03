// swamp-gguf/src/quant/q3_k.rs
// Q3_K: 256 elementos, 3 bits por elemento (110 bytes por bloco)

use crate::error::{GgufError, Result};
use half::f16;

const BLOCK_SIZE: usize = 256;
const BLOCK_BYTES: usize = 110;

pub fn dequant_q3k(raw: &[u8], dst: &mut [f32]) -> Result<()> {
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
        let hmask = &raw[base..base + 32];
        let qs    = &raw[base + 32..base + 96];
        let sc    = &raw[base + 96..base + 108];
        let d     = f16::from_le_bytes([raw[base + 108], raw[base + 109]]).to_f32();

        let out_start = b * BLOCK_SIZE;
        let out_end   = (out_start + BLOCK_SIZE).min(n);

        // Desempacota as 16 escalas usando a manipulacao de palavras de 32 bits equivalente ao C
        let mut aux = [0u32; 4];
        aux[0] = u32::from_le_bytes(sc[0..4].try_into().unwrap());
        aux[1] = u32::from_le_bytes(sc[4..8].try_into().unwrap());
        aux[2] = u32::from_le_bytes(sc[8..12].try_into().unwrap());

        let tmp = aux[2];
        let kmask1 = 0x03030303u32;
        let kmask2 = 0x0f0f0f0fu32;

        let a2 = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        let a3 = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        let a0 = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        let a1 = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        let mut scales = [0i8; 16];
        scales[0..4].copy_from_slice(&a0.to_le_bytes().map(|x| x as i8));
        scales[4..8].copy_from_slice(&a1.to_le_bytes().map(|x| x as i8));
        scales[8..12].copy_from_slice(&a2.to_le_bytes().map(|x| x as i8));
        scales[12..16].copy_from_slice(&a3.to_le_bytes().map(|x| x as i8));

        let mut is = 0;
        let mut m = 1u8;
        let mut q_offset = 0;

        for n_offset in (0..256).step_by(128) {
            let mut shift = 0;
            for j in 0..4 {
                let dl1 = d * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let global = out_start + n_offset + j * 32 + l;
                    if global >= out_end { break; }
                    let q_val = (qs[q_offset + l] >> shift) & 3;
                    let hm_val = hmask[l];
                    let bit = if (hm_val & m) != 0 { 0i8 } else { 4 };
                    dst[global] = dl1 * (q_val as i8 - bit) as f32;
                }

                let dl2 = d * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let global = out_start + n_offset + j * 32 + 16 + l;
                    if global >= out_end { break; }
                    let q_val = (qs[q_offset + 16 + l] >> shift) & 3;
                    let hm_val = hmask[16 + l];
                    let bit = if (hm_val & m) != 0 { 0i8 } else { 4 };
                    dst[global] = dl2 * (q_val as i8 - bit) as f32;
                }

                shift += 2;
                m <<= 1;
            }
            q_offset += 32;
        }
    }
    Ok(())
}
