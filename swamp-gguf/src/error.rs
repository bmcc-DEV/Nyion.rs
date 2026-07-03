// swamp-gguf/src/error.rs
// Tipos de erro do parser GGUF

use thiserror::Error;

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid GGUF magic: expected GGUF, got {0:?}")]
    InvalidMagic([u8; 4]),

    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    #[error("Unknown GGMLDType: {0}")]
    UnknownDType(u32),

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    #[error("Metadata key not found: {0}")]
    MetadataKeyNotFound(String),

    #[error("Type mismatch for metadata key '{key}': expected {expected}, got {got}")]
    MetadataTypeMismatch { key: String, expected: String, got: String },

    #[error("Dequantization error: {0}")]
    Dequant(String),

    #[error("Invalid tensor data: {0}")]
    InvalidTensor(String),

    #[error("Buffer too small: needed {needed}, got {got}")]
    BufferTooSmall { needed: usize, got: usize },
}

pub type Result<T> = std::result::Result<T, GgufError>;
