// swamp-gguf/src/metadata.rs
// Parser de metadados GGUF (KV pairs)

use crate::error::{GgufError, Result};
use std::collections::HashMap;

/// Tipos de valores de metadados GGUF
#[derive(Debug, Clone)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    Array(MetadataArray),
}

#[derive(Debug, Clone)]
pub enum MetadataArray {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
    Bool(Vec<bool>),
    String(Vec<String>),
}

impl MetadataValue {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::U8(_)     => "u8",
            Self::I8(_)     => "i8",
            Self::U16(_)    => "u16",
            Self::I16(_)    => "i16",
            Self::U32(_)    => "u32",
            Self::I32(_)    => "i32",
            Self::F32(_)    => "f32",
            Self::U64(_)    => "u64",
            Self::I64(_)    => "i64",
            Self::F64(_)    => "f64",
            Self::Bool(_)   => "bool",
            Self::String(_) => "string",
            Self::Array(_)  => "array",
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U8(v)  => Some(*v as u32),
            Self::U16(v) => Some(*v as u32),
            Self::U32(v) => Some(*v),
            Self::I32(v) => Some(*v as u32),
            _            => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(v)  => Some(*v as u64),
            Self::U16(v) => Some(*v as u64),
            Self::U32(v) => Some(*v as u64),
            Self::U64(v) => Some(*v),
            Self::I32(v) => Some(*v as u64),
            Self::I64(v) => Some(*v as u64),
            _            => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            Self::U32(v) => Some(*v as f32),
            Self::I32(v) => Some(*v as f32),
            _            => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _               => None,
        }
    }

    pub fn as_u32_array(&self) -> Option<&[u32]> {
        match self {
            Self::Array(MetadataArray::U32(v)) => Some(v.as_slice()),
            _                                   => None,
        }
    }

    pub fn as_string_array(&self) -> Option<&[String]> {
        match self {
            Self::Array(MetadataArray::String(v)) => Some(v.as_slice()),
            _                                      => None,
        }
    }
}

/// Leitor de bytes little-endian sobre um slice
pub struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn read_u8(&mut self) -> Result<u8> {
        let b = self.data.get(self.pos).copied()
            .ok_or_else(|| GgufError::Io(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "read_u8")))?;
        self.pos += 1;
        Ok(b)
    }

    pub fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    pub fn read_u16(&mut self) -> Result<u16> {
        let b = self.data.get(self.pos..self.pos + 2)
            .ok_or_else(|| GgufError::Io(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "read_u16")))?;
        self.pos += 2;
        Ok(u16::from_le_bytes(b.try_into().unwrap()))
    }

    pub fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    pub fn read_u32(&mut self) -> Result<u32> {
        let b = self.data.get(self.pos..self.pos + 4)
            .ok_or_else(|| GgufError::Io(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "read_u32")))?;
        self.pos += 4;
        Ok(u32::from_le_bytes(b.try_into().unwrap()))
    }

    pub fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    pub fn read_u64(&mut self) -> Result<u64> {
        let b = self.data.get(self.pos..self.pos + 8)
            .ok_or_else(|| GgufError::Io(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "read_u64")))?;
        self.pos += 8;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }

    pub fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    pub fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    pub fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    pub fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    pub fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.data.get(self.pos..self.pos + len)
            .ok_or_else(|| GgufError::Io(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "read_string")))?;
        self.pos += len;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    pub fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let bytes = self.data.get(self.pos..self.pos + n)
            .ok_or_else(|| GgufError::Io(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "read_bytes")))?;
        self.pos += n;
        Ok(bytes)
    }
}

/// Parse de um valor de metadado dado o tipo (u32)
pub fn parse_metadata_value(r: &mut Reader<'_>, value_type: u32) -> Result<MetadataValue> {
    match value_type {
        0  => Ok(MetadataValue::U8(r.read_u8()?)),
        1  => Ok(MetadataValue::I8(r.read_i8()?)),
        2  => Ok(MetadataValue::U16(r.read_u16()?)),
        3  => Ok(MetadataValue::I16(r.read_i16()?)),
        4  => Ok(MetadataValue::U32(r.read_u32()?)),
        5  => Ok(MetadataValue::I32(r.read_i32()?)),
        6  => Ok(MetadataValue::F32(r.read_f32()?)),
        7  => Ok(MetadataValue::Bool(r.read_bool()?)),
        8  => Ok(MetadataValue::String(r.read_string()?)),
        9  => {
            // Array: tipo dos elementos + count + elementos
            let elem_type = r.read_u32()?;
            let count = r.read_u64()? as usize;
            let arr = parse_metadata_array(r, elem_type, count)?;
            Ok(MetadataValue::Array(arr))
        }
        10 => Ok(MetadataValue::U64(r.read_u64()?)),
        11 => Ok(MetadataValue::I64(r.read_i64()?)),
        12 => Ok(MetadataValue::F64(r.read_f64()?)),
        _  => Err(GgufError::UnknownDType(value_type)),
    }
}

fn parse_metadata_array(r: &mut Reader<'_>, elem_type: u32, count: usize) -> Result<MetadataArray> {
    match elem_type {
        0  => { let mut v = Vec::with_capacity(count); for _ in 0..count { v.push(r.read_u8()?); } Ok(MetadataArray::U8(v)) }
        1  => { let mut v = Vec::with_capacity(count); for _ in 0..count { v.push(r.read_i8()?); } Ok(MetadataArray::I8(v)) }
        2  => { let mut v = Vec::with_capacity(count); for _ in 0..count { v.push(r.read_u16()?); } Ok(MetadataArray::U16(v)) }
        3  => { let mut v = Vec::with_capacity(count); for _ in 0..count { v.push(r.read_i16()?); } Ok(MetadataArray::I16(v)) }
        4  => { let mut v = Vec::with_capacity(count); for _ in 0..count { v.push(r.read_u32()?); } Ok(MetadataArray::U32(v)) }
        5  => { let mut v = Vec::with_capacity(count); for _ in 0..count { v.push(r.read_i32()?); } Ok(MetadataArray::I32(v)) }
        6  => { let mut v = Vec::with_capacity(count); for _ in 0..count { v.push(r.read_f32()?); } Ok(MetadataArray::F32(v)) }
        7  => { let mut v = Vec::with_capacity(count); for _ in 0..count { v.push(r.read_bool()?); } Ok(MetadataArray::Bool(v)) }
        8  => { let mut v = Vec::with_capacity(count); for _ in 0..count { v.push(r.read_string()?); } Ok(MetadataArray::String(v)) }
        10 => { let mut v = Vec::with_capacity(count); for _ in 0..count { v.push(r.read_u64()?); } Ok(MetadataArray::U64(v)) }
        11 => { let mut v = Vec::with_capacity(count); for _ in 0..count { v.push(r.read_i64()?); } Ok(MetadataArray::I64(v)) }
        12 => { let mut v = Vec::with_capacity(count); for _ in 0..count { v.push(r.read_f64()?); } Ok(MetadataArray::F64(v)) }
        _  => Err(GgufError::UnknownDType(elem_type)),
    }
}

/// Mapa de metadados com acesso tipado e helpers de compat para arquiteturas
pub struct Metadata(pub HashMap<String, MetadataValue>);

impl Metadata {
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.0.get(key)
    }

    pub fn get_u32(&self, key: &str) -> Result<u32> {
        self.0.get(key)
            .and_then(|v| v.as_u32())
            .ok_or_else(|| GgufError::MetadataKeyNotFound(key.to_string()))
    }

    pub fn get_u32_or(&self, key: &str, default: u32) -> u32 {
        self.0.get(key).and_then(|v| v.as_u32()).unwrap_or(default)
    }

    pub fn get_f32(&self, key: &str) -> Result<f32> {
        self.0.get(key)
            .and_then(|v| v.as_f32())
            .ok_or_else(|| GgufError::MetadataKeyNotFound(key.to_string()))
    }

    pub fn get_f32_or(&self, key: &str, default: f32) -> f32 {
        self.0.get(key).and_then(|v| v.as_f32()).unwrap_or(default)
    }

    pub fn get_str(&self, key: &str) -> Result<&str> {
        self.0.get(key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| GgufError::MetadataKeyNotFound(key.to_string()))
    }

    pub fn get_u64(&self, key: &str) -> Result<u64> {
        self.0.get(key)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| GgufError::MetadataKeyNotFound(key.to_string()))
    }

    /// Detecta a arquitetura do modelo a partir dos metadados
    pub fn architecture(&self) -> &str {
        self.0.get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
    }

    /// Alias padrao para chave de metadado, com fallback entre prefixos de arquitetura.
    /// Suporta: llama, qwen2, mistral, phi3, gemma, starcoder2
    fn resolve(&self, base: &str) -> Option<&MetadataValue> {
        // chave direta
        if let Some(v) = self.0.get(base) {
            return Some(v);
        }
        // tenta substituir prefixo de arquitetura
        let arch = self.architecture();
        let prefixes = ["llama", "qwen2", "mistral", "phi3", "gemma", "starcoder2", "falcon", "mpt"];
        for pfx in prefixes {
            let candidate = base.replacen(pfx, arch, 1);
            if candidate != base {
                if let Some(v) = self.0.get(&candidate) {
                    return Some(v);
                }
            }
            // inverso: tenta substituir arch por pfx
            let candidate2 = base.replacen(arch, pfx, 1);
            if candidate2 != base {
                if let Some(v) = self.0.get(&candidate2) {
                    return Some(v);
                }
            }
        }
        None
    }

    // --- Helpers canonicos de arquitetura (prefixo "llama.") com compat automatico ---

    pub fn n_head(&self) -> Result<u32> {
        self.resolve("llama.attention.head_count")
            .and_then(|v| v.as_u32())
            .ok_or_else(|| GgufError::MetadataKeyNotFound("llama.attention.head_count".to_string()))
    }

    pub fn n_kv_head(&self) -> u32 {
        self.resolve("llama.attention.head_count_kv")
            .and_then(|v| v.as_u32())
            .unwrap_or_else(|| self.n_head().unwrap_or(1))
    }

    pub fn n_layer(&self) -> Result<u32> {
        self.resolve("llama.block_count")
            .and_then(|v| v.as_u32())
            .ok_or_else(|| GgufError::MetadataKeyNotFound("llama.block_count".to_string()))
    }

    pub fn n_embd(&self) -> Result<u32> {
        self.resolve("llama.embedding_length")
            .and_then(|v| v.as_u32())
            .ok_or_else(|| GgufError::MetadataKeyNotFound("llama.embedding_length".to_string()))
    }

    pub fn n_ff(&self) -> u32 {
        self.resolve("llama.feed_forward_length")
            .and_then(|v| v.as_u32())
            .unwrap_or(0)
    }

    pub fn n_ctx_train(&self) -> u32 {
        self.resolve("llama.context_length")
            .and_then(|v| v.as_u32())
            .unwrap_or(4096)
    }

    pub fn rope_freq_base(&self) -> f32 {
        self.resolve("llama.rope.freq_base")
            .and_then(|v| v.as_f32())
            .unwrap_or(10000.0)
    }

    pub fn rope_dimension(&self) -> u32 {
        if let Some(v) = self.resolve("llama.rope.dimension_count") {
            if let Some(u) = v.as_u32() { return u; }
        }
        // fallback: n_embd / n_head
        let embd = self.resolve("llama.embedding_length").and_then(|v| v.as_u32()).unwrap_or(1);
        let heads = self.resolve("llama.attention.head_count").and_then(|v| v.as_u32()).unwrap_or(1);
        embd / heads
    }
}
