// swamp-gguf/src/dtype.rs
// Tipos de quantizacao suportados pelo formato GGUF/GGML

use crate::error::{GgufError, Result};

/// Tipos de dados GGML - codigos conforme especificacao GGUF v3
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlDType {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
}

impl GgmlDType {
    pub fn from_u32(v: u32) -> Result<Self> {
        match v {
            0  => Ok(Self::F32),
            1  => Ok(Self::F16),
            2  => Ok(Self::Q4_0),
            3  => Ok(Self::Q4_1),
            6  => Ok(Self::Q5_0),
            7  => Ok(Self::Q5_1),
            8  => Ok(Self::Q8_0),
            9  => Ok(Self::Q8_1),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            15 => Ok(Self::Q8_K),
            16 => Ok(Self::IQ2_XXS),
            17 => Ok(Self::IQ2_XS),
            18 => Ok(Self::IQ3_XXS),
            19 => Ok(Self::IQ1_S),
            20 => Ok(Self::IQ4_NL),
            21 => Ok(Self::IQ3_S),
            22 => Ok(Self::IQ2_S),
            23 => Ok(Self::IQ4_XS),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::IQ1_M),
            30 => Ok(Self::BF16),
            _  => Err(GgufError::UnknownDType(v)),
        }
    }

    /// Numero de elementos por bloco de quantizacao
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::F64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
            Self::IQ2_XXS | Self::IQ2_XS | Self::IQ2_S => 256,
            Self::IQ3_XXS | Self::IQ3_S => 256,
            Self::IQ4_NL | Self::IQ4_XS => 32,
            Self::IQ1_S | Self::IQ1_M => 256,
        }
    }

    /// Tamanho em bytes por bloco de quantizacao
    pub fn type_size(&self) -> usize {
        match self {
            Self::F32  => 4,
            Self::F16  => 2,
            Self::BF16 => 2,
            Self::F64  => 8,
            Self::I8   => 1,
            Self::I16  => 2,
            Self::I32  => 4,
            Self::I64  => 8,
            // Q4_0: 32 elements, 2 bytes scale + 16 bytes nibbles = 18
            Self::Q4_0 => 18,
            // Q4_1: 32 elements, 2+2 bytes scales + 16 bytes nibbles = 20
            Self::Q4_1 => 20,
            // Q5_0: 32 elements, 2 bytes scale + 4 bytes high bits + 16 bytes nibbles = 22
            Self::Q5_0 => 22,
            // Q5_1: 32 elements, 2+2 bytes scales + 4 bytes high bits + 16 bytes = 24
            Self::Q5_1 => 24,
            // Q8_0: 32 elements, 2 bytes scale + 32 bytes = 34
            Self::Q8_0 => 34,
            // Q8_1: 32 elements, 4+4 bytes scales + 32 bytes = 40
            Self::Q8_1 => 40,
            // K-quants (256 elements per block):
            Self::Q2_K => 84,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_K => 292,
            // IQ quants:
            Self::IQ2_XXS => 66,
            Self::IQ2_XS  => 74,
            Self::IQ2_S   => 82,
            Self::IQ3_XXS => 98,
            Self::IQ3_S   => 110,
            Self::IQ4_NL  => 18,
            Self::IQ4_XS  => 136,
            Self::IQ1_S   => 50,
            Self::IQ1_M   => 56,
        }
    }

    /// Calcula numero de bytes para N elementos neste tipo
    pub fn nbytes_for(&self, n_elems: usize) -> usize {
        let bs = self.block_size();
        let blocks = (n_elems + bs - 1) / bs;
        blocks * self.type_size()
    }

    /// Retorna true se este tipo pode ser dequantizado para f32
    pub fn is_quantized(&self) -> bool {
        !matches!(self, Self::F32 | Self::F64 | Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::F32     => "F32",
            Self::F16     => "F16",
            Self::BF16    => "BF16",
            Self::Q4_0    => "Q4_0",
            Self::Q4_1    => "Q4_1",
            Self::Q5_0    => "Q5_0",
            Self::Q5_1    => "Q5_1",
            Self::Q8_0    => "Q8_0",
            Self::Q8_1    => "Q8_1",
            Self::Q2_K    => "Q2_K",
            Self::Q3_K    => "Q3_K",
            Self::Q4_K    => "Q4_K",
            Self::Q5_K    => "Q5_K",
            Self::Q6_K    => "Q6_K",
            Self::Q8_K    => "Q8_K",
            Self::IQ2_XXS => "IQ2_XXS",
            Self::IQ2_XS  => "IQ2_XS",
            Self::IQ2_S   => "IQ2_S",
            Self::IQ3_XXS => "IQ3_XXS",
            Self::IQ3_S   => "IQ3_S",
            Self::IQ4_NL  => "IQ4_NL",
            Self::IQ4_XS  => "IQ4_XS",
            Self::IQ1_S   => "IQ1_S",
            Self::IQ1_M   => "IQ1_M",
            Self::I8      => "I8",
            Self::I16     => "I16",
            Self::I32     => "I32",
            Self::I64     => "I64",
            Self::F64     => "F64",
        }
    }
}

impl std::fmt::Display for GgmlDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
