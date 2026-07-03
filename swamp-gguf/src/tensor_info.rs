// swamp-gguf/src/tensor_info.rs
// Informacoes de tensor e dequantizacao lazy

use crate::dtype::GgmlDType;
use crate::error::{GgufError, Result};
use crate::quant;

/// Informacoes sobre um tensor no arquivo GGUF
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgmlDType,
    /// Offset em bytes a partir do inicio dos dados de tensor (pos apos o alinhamento)
    pub offset: u64,
}

impl TensorInfo {
    /// Numero total de elementos no tensor
    pub fn n_elems(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }

    /// Tamanho em bytes dos dados quantizados deste tensor
    pub fn nbytes(&self) -> usize {
        self.dtype.nbytes_for(self.n_elems())
    }

    /// Retorna os bytes raw deste tensor a partir do slice de dados
    pub fn raw_bytes<'a>(&self, data: &'a [u8]) -> Result<&'a [u8]> {
        let start = self.offset as usize;
        let end = start + self.nbytes();
        data.get(start..end)
            .ok_or_else(|| GgufError::InvalidTensor(
                format!("{}: offset {} + {} bytes > data len {}", self.name, start, self.nbytes(), data.len())
            ))
    }

    /// Dequantiza para f32, escrevendo no buffer `dst`.
    /// dst deve ter pelo menos n_elems() entradas.
    pub fn dequantize_into(&self, data: &[u8], dst: &mut [f32]) -> Result<()> {
        let n = self.n_elems();
        if dst.len() < n {
            return Err(GgufError::BufferTooSmall { needed: n, got: dst.len() });
        }
        let raw = self.raw_bytes(data)?;
        quant::dequantize(self.dtype, raw, &mut dst[..n])
    }

    /// Dequantiza e retorna um Vec<f32> alocado
    pub fn dequantize(&self, data: &[u8]) -> Result<Vec<f32>> {
        let n = self.n_elems();
        let mut dst = vec![0.0f32; n];
        self.dequantize_into(data, &mut dst)?;
        Ok(dst)
    }

    pub fn shape_str(&self) -> String {
        let dims: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
        format!("[{}]", dims.join(", "))
    }
}
