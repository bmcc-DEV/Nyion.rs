// swamp-gguf/src/lib.rs
// LLamañón.rs - swamp-gguf: parser GGUF + dequantizacao lazy, zero-copy

pub mod dtype;
pub mod error;
pub mod metadata;
pub mod mmap;
pub mod quant;
pub mod tensor_info;

// Re-exporta os tipos principais
pub use dtype::GgmlDType;
pub use error::{GgufError, Result};
pub use metadata::{Metadata, MetadataArray, MetadataValue};
pub use mmap::GgufFile;
pub use tensor_info::TensorInfo;
