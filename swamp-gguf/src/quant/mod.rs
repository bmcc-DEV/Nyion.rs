// swamp-gguf/src/quant/mod.rs
// Dispatcher de dequantizacao

use crate::dtype::GgmlDType;
use crate::error::{GgufError, Result};

pub mod q4_0;
pub mod q4_k;
pub mod q8_0;
pub mod q5_k;
pub mod q6_k;
pub mod q2_k;
pub mod q3_k;
pub mod f16;
pub mod iq2_xxs;

/// Dispatch principal: dequantiza `raw` bytes do tipo `dtype` para `dst` (f32)
pub fn dequantize(dtype: GgmlDType, raw: &[u8], dst: &mut [f32]) -> Result<()> {
    match dtype {
        GgmlDType::F32  => dequant_f32(raw, dst),
        GgmlDType::F16  => f16::dequant_f16(raw, dst),
        GgmlDType::BF16 => f16::dequant_bf16(raw, dst),
        GgmlDType::Q4_0 => q4_0::dequant(raw, dst),
        GgmlDType::Q4_K => q4_k::dequant_q4k(raw, dst),
        GgmlDType::Q5_K => q5_k::dequant_q5k(raw, dst),
        GgmlDType::Q6_K => q6_k::dequant_q6k(raw, dst),
        GgmlDType::Q8_0 => q8_0::dequant(raw, dst),
        GgmlDType::Q2_K => q2_k::dequant_q2k(raw, dst),
        GgmlDType::Q3_K => q3_k::dequant_q3k(raw, dst),
        GgmlDType::IQ2_XXS => iq2_xxs::dequant(raw, dst),
        t => Err(GgufError::Dequant(format!("dequantizacao nao implementada para {}", t.name()))),
    }
}

fn dequant_f32(raw: &[u8], dst: &mut [f32]) -> Result<()> {
    if raw.len() < dst.len() * 4 {
        return Err(GgufError::BufferTooSmall { needed: dst.len() * 4, got: raw.len() });
    }
    for (i, d) in dst.iter_mut().enumerate() {
        let bytes = &raw[i*4..(i+1)*4];
        *d = f32::from_le_bytes(bytes.try_into().unwrap());
    }
    Ok(())
}
