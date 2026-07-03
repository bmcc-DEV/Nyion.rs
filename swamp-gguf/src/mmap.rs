// swamp-gguf/src/mmap.rs
// Leitor memory-mapped de arquivo GGUF

use crate::error::{GgufError, Result};
use crate::metadata::{Metadata, Reader, parse_metadata_value};
use crate::tensor_info::TensorInfo;
use crate::dtype::GgmlDType;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

const GGUF_MAGIC: [u8; 4] = [b'G', b'G', b'U', b'F'];
const GGUF_ALIGNMENT: u64 = 32;

/// Arquivo GGUF aberto via mmap com acesso zero-copy aos dados de tensor
pub struct GgufFile {
    pub(crate) mmap: Mmap,
    pub metadata: Metadata,
    pub tensors: Vec<TensorInfo>,
    /// Offset em bytes no mmap onde comecam os dados dos tensores
    pub data_offset: usize,
}

impl GgufFile {
    /// Abre um arquivo GGUF via memory-map
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::parse(mmap)
    }

    fn parse(mmap: Mmap) -> Result<Self> {
        let mut r = Reader::new(&mmap);

        // Header
        let magic: [u8; 4] = r.read_bytes(4)?.try_into().unwrap();
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }

        let version = r.read_u32()?;
        if version < 1 || version > 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = if version >= 2 { r.read_u64()? } else { r.read_u32()? as u64 };
        let kv_count     = if version >= 2 { r.read_u64()? } else { r.read_u32()? as u64 };

        // Metadados (KV)
        let mut kv = HashMap::new();
        for _ in 0..kv_count {
            let key   = r.read_string()?;
            let vtype = r.read_u32()?;
            let val   = parse_metadata_value(&mut r, vtype)?;
            kv.insert(key, val);
        }

        // Tensor infos
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name     = r.read_string()?;
            let n_dims   = r.read_u32()? as usize;
            let mut shape = vec![0u64; n_dims];
            for d in 0..n_dims {
                shape[d] = if version >= 2 { r.read_u64()? } else { r.read_u32()? as u64 };
            }
            let dtype_u32 = r.read_u32()?;
            let dtype     = GgmlDType::from_u32(dtype_u32)?;
            let offset    = r.read_u64()?;
            tensors.push(TensorInfo { name, shape, dtype, offset });
        }

        // Alinhamento: o inicio dos dados e alinhado a GGUF_ALIGNMENT bytes
        let header_end = r.pos() as u64;
        let align = GGUF_ALIGNMENT;
        let data_offset = ((header_end + align - 1) / align * align) as usize;

        // Ajusta offsets dos tensores para serem relativos ao mmap
        let mut tensors_abs = tensors;
        for t in &mut tensors_abs {
            t.offset += data_offset as u64;
        }

        Ok(Self {
            mmap,
            metadata: Metadata(kv),
            tensors: tensors_abs,
            data_offset,
        })
    }

    /// Encontra um tensor pelo nome
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Encontra um tensor pelo nome ou retorna erro
    pub fn tensor_or_err(&self, name: &str) -> Result<&TensorInfo> {
        self.tensor(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))
    }

    /// Slice do mmap correspondente aos dados de todos os tensores
    pub fn data(&self) -> &[u8] {
        &self.mmap[self.data_offset..]
    }

    /// Bytes raw de um tensor especifico (a partir do inicio do mmap)
    pub fn tensor_raw_bytes(&self, info: &TensorInfo) -> Result<&[u8]> {
        let start = info.offset as usize;
        let end   = start + info.nbytes();
        self.mmap.get(start..end)
            .ok_or_else(|| GgufError::InvalidTensor(
                format!("{}: range {}..{} fora do mmap ({})", info.name, start, end, self.mmap.len())
            ))
    }

    pub fn n_tensors(&self) -> usize {
        self.tensors.len()
    }

    pub fn architecture(&self) -> &str {
        self.metadata.architecture()
    }

    /// Retorna o tamanho total do arquivo memory-mapped em bytes
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    /// Dequantiza um tensor especifico preenchendo o buffer de destino
    pub fn dequantize_tensor(&self, info: &TensorInfo, dst: &mut [f32]) -> Result<()> {
        info.dequantize_into(&self.mmap, dst)
    }

    /// Dequantiza um tensor e aloca um novo Vec<f32> com o resultado
    pub fn dequantize_tensor_alloc(&self, info: &TensorInfo) -> Result<Vec<f32>> {
        info.dequantize(&self.mmap)
    }

    /// Ponteiro base do memory-map (mutavel para madvise) e tamanho total
    pub fn mmap_ptr_and_len(&self) -> (*mut u8, usize) {
        (self.mmap.as_ptr() as *mut u8, self.mmap.len())
    }

    /// Offset e tamanho do tensor raw dentro do mmap
    pub fn tensor_raw_offset_len(&self, name: &str) -> Option<(usize, usize)> {
        self.tensor(name).map(|t| (t.offset as usize, t.nbytes()))
    }
}
