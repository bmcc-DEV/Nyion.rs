// swamp-gguf/src/quant/f16.rs
// Dequantizacao F16 e BF16 para F32

use crate::error::{GgufError, Result};
use half::{f16, bf16};

pub fn dequant_f16(raw: &[u8], dst: &mut [f32]) -> Result<()> {
    let n = dst.len();
    if raw.len() < n * 2 {
        return Err(GgufError::BufferTooSmall { needed: n * 2, got: raw.len() });
    }
    for (i, d) in dst.iter_mut().enumerate() {
        let bytes: [u8; 2] = raw[i*2..(i+1)*2].try_into().unwrap();
        *d = f16::from_le_bytes(bytes).to_f32();
    }
    Ok(())
}

pub fn dequant_bf16(raw: &[u8], dst: &mut [f32]) -> Result<()> {
    let n = dst.len();
    if raw.len() < n * 2 {
        return Err(GgufError::BufferTooSmall { needed: n * 2, got: raw.len() });
    }
    for (i, d) in dst.iter_mut().enumerate() {
        let bytes: [u8; 2] = raw[i*2..(i+1)*2].try_into().unwrap();
        *d = bf16::from_le_bytes(bytes).to_f32();
    }
    Ok(())
}
