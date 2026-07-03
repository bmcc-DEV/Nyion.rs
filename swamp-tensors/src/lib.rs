// swamp-tensors - lib.rs
// Tipos de tensor e operacoes CPU para LLamanyon.rs

pub mod layout;
pub mod tensor;
pub mod ops;
pub mod arena;
pub mod storage;

pub use layout::Layout;
pub use tensor::CpuTensor;
pub use ops::{rms_norm, softmax, rope, matmul};
pub use arena::TensorArena;
pub use storage::LscPrefetcher;
