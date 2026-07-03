// swamp-gguf/src/quant/q4_0.rs
// Q4_0: blocos de 32 elementos, 1 scale f16 + 16 bytes de nibbles

use crate::error::{GgufError, Result};
use half::f16;

// Bloco Q4_0: 2 bytes scale (f16) + 16 bytes de nibbles = 18 bytes por bloco de 32 elementos
const BLOCK_SIZE: usize = 32;
const BLOCK_BYTES: usize = 18;

pub fn dequant(raw: &[u8], dst: &mut [f32]) -> Result<()> {
    let n = dst.len();
    let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if raw.len() < n_blocks * BLOCK_BYTES {
        return Err(GgufError::BufferTooSmall {
            needed: n_blocks * BLOCK_BYTES,
            got: raw.len(),
        });
    }

    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let scale = f16::from_le_bytes([raw[base], raw[base + 1]]).to_f32();
        let nibbles = &raw[base + 2..base + 18];

        let out_start = b * BLOCK_SIZE;
        let out_end = (out_start + BLOCK_SIZE).min(n);

        for i in 0..(out_end - out_start) {
            let byte = nibbles[i / 2];
            let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            // Q4_0: nibbles em [0, 15], centrado em 8 -> [-8, 7]
            let val = (nibble as i32) - 8;
            dst[out_start + i] = val as f32 * scale;
        }
    }
    Ok(())
}
