// swamp-gguf/src/quant/q6_k.rs
use crate::error::{GgufError, Result};
use half::f16;

const BLOCK_SIZE: usize = 256;
const BLOCK_BYTES: usize = 210;

pub fn dequant_q6k(raw: &[u8], dst: &mut [f32]) -> Result<()> {
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

fn dequant_scalar(raw: &[u8], dst: &mut [f32], n: usize, n_blocks: usize) -> Result<()> {
    for b in 0..n_blocks {
        let base   = b * BLOCK_BYTES;
        let mut ql = &raw[base..base + 128];
        let mut qh = &raw[base + 128..base + 192];
        let mut sc = &raw[base + 192..base + 208];
        let d      = f16::from_le_bytes([raw[base + 208], raw[base + 209]]).to_f32();

        let mut y = [0.0f32; 256];
        let mut y_idx = 0;

        for _ in 0..2 {
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                
                y[y_idx + l + 0] = d * (sc[is + 0] as i8) as f32 * (q1 as f32);
                y[y_idx + l + 32] = d * (sc[is + 2] as i8) as f32 * (q2 as f32);
                y[y_idx + l + 64] = d * (sc[is + 4] as i8) as f32 * (q3 as f32);
                y[y_idx + l + 96] = d * (sc[is + 6] as i8) as f32 * (q4 as f32);
            }
            y_idx += 128;
            ql = &ql[64..];
            qh = &qh[32..];
            sc = &sc[8..];
        }

        let out_start = b * BLOCK_SIZE;
        let count = (n - out_start).min(256);
        dst[out_start..out_start + count].copy_from_slice(&y[..count]);
    }
    Ok(())
}
