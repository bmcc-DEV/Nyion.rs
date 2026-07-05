// swamp-gpu/src/lib.rs
// Rust bridge para GPU acceleration via CUDA C shared library
// Carrega libswamp_gpu.so em runtime com libloading

use libloading::{Library, Symbol};
use std::sync::OnceLock;
use thiserror::Error;
use tracing::info;

impl From<libloading::Error> for GpuError {
    fn from(e: libloading::Error) -> Self {
        GpuError::Symbol(format!("{:?}", e))
    }
}

#[derive(Error, Debug, Clone)]
pub enum GpuError {
    #[error("CUDA/GPU not available: {0}")]
    NotAvailable(String),
    #[error("Library load failed: {0}")]
    LibLoad(String),
    #[error("Symbol lookup failed: {0}")]
    Symbol(String),
    #[error("GPU kernel returned error code {0}")]
    KernelError(i32),
}

pub type Result<T> = std::result::Result<T, GpuError>;

static GPU_LIB: OnceLock<Result<Library>> = OnceLock::new();

fn try_lib() -> Result<&'static Library> {
    match GPU_LIB.get_or_init(|| {
        let paths = [
            "libswamp_gpu.so",
            "./swamp-gpu/libswamp_gpu.so",
            "../swamp-gpu/libswamp_gpu.so",
            "/media/bruno/Bruno/Swamp 5.0/llamanyon/swamp-gpu/libswamp_gpu.so",
        ];
        for path in &paths {
            match unsafe { Library::new(path) } {
                Ok(lib) => {
                    info!("GPU library loaded from: {}", path);
                    return Ok(lib);
                }
                Err(_) => continue,
            }
        }
        Err(GpuError::LibLoad("libswamp_gpu.so not found".into()))
    }) {
        Ok(lib) => Ok(lib),
        Err(e) => Err(e.clone()),
    }
}

/// Check if GPU acceleration is available
pub fn gpu_available() -> bool {
    try_lib().is_ok()
}

/// Initialize GPU context (call once at startup)
pub fn gpu_init() -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn() -> i32> = unsafe { lib.get(b"gpu_init")? };
    let ret = unsafe { func() };
    if ret != 0 {
        return Err(GpuError::KernelError(ret));
    }
    info!("GPU initialized (GTX 1650 Mobile)");
    Ok(())
}

/// Fused GPU attention: QK^T + softmax + weighted sum of V
/// All pointers are host (CPU) memory — copies happen internally.
pub fn gpu_attention_forward(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    output: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<()> {
    let lib = try_lib()?;

    assert_eq!(q.len(), n_heads * head_dim, "Q size mismatch");
    assert_eq!(k_cache.len(), n_kv_heads * seq_len * head_dim, "K cache size mismatch");
    assert_eq!(v_cache.len(), n_kv_heads * seq_len * head_dim, "V cache size mismatch");
    assert_eq!(output.len(), n_heads * head_dim, "output size mismatch");

    let func: Symbol<
        unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32, i32, i32, i32, i32) -> i32,
    > = unsafe { lib.get(b"gpu_attention_forward")? };

    let ret = unsafe {
        func(
            q.as_ptr(), k_cache.as_ptr(), v_cache.as_ptr(),
            output.as_mut_ptr(),
            n_heads as i32, n_kv_heads as i32, seq_len as i32, head_dim as i32,
        )
    };
    if ret != 0 {
        return Err(GpuError::KernelError(ret));
    }
    Ok(())
}

/// Run Q4_K GEMV on GPU (dequantize on-the-fly)
pub fn gpu_gemv_q4k(
    d_w: *const u8, d_x: *const f32, d_out: *mut f32,
    n_rows: i32, n_blocks: i32, stream: CudaStream,
) -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*const u8, *const f32, *mut f32, i32, i32, CudaStream)> =
        unsafe { lib.get(b"gpu_gemv_q4k")? };
    unsafe { func(d_w, d_x, d_out, n_rows, n_blocks, stream) };
    Ok(())
}

/// Upload weights (ring buffer) to GPU VRAM
pub fn gpu_upload_weights(h_w: *const u8, d_w: *mut *mut u8, bytes: usize, stream: CudaStream) -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*const u8, *mut *mut u8, usize, CudaStream)> =
        unsafe { lib.get(b"gpu_upload_weights")? };
    unsafe { func(h_w, d_w, bytes, stream) };
    Ok(())
}

/// Free GPU weights
pub fn gpu_free_weights(d_w: *mut u8) -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*mut u8)> =
        unsafe { lib.get(b"gpu_free_weights")? };
    unsafe { func(d_w) };
    Ok(())
}

/// Full GPU GEMV: copy x to GPU, run kernel, copy result back
pub fn gpu_gemv_q4k_full(
    d_w: *const u8, h_x: &[f32], h_out: &mut [f32],
    n_rows: i32, n_blocks: i32, stream: CudaStream,
) -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*const u8, *const f32, *mut f32, i32, i32, CudaStream)> =
        unsafe { lib.get(b"gpu_gemv_q4k_full")? };
    unsafe { func(d_w, h_x.as_ptr(), h_out.as_mut_ptr(), n_rows, n_blocks, stream) };
    Ok(())
}

/// GPU GEMV with pre-allocated persistent buffers (no malloc per call)
pub fn gpu_gemv_q4k_prealloc(
    d_w: *const u8, h_x: &[f32], h_out: &mut [f32],
    d_x: *mut f32, d_out: *mut f32,
    n_rows: i32, n_blocks: i32, max_rows: i32, max_cols: i32,
    stream: CudaStream,
) -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*const u8, *const f32, *mut f32, *mut f32, *mut f32, i32, i32, i32, i32, CudaStream)> =
        unsafe { lib.get(b"gpu_gemv_q4k_prealloc")? };
    unsafe { func(d_w, h_x.as_ptr(), h_out.as_mut_ptr(), d_x, d_out, n_rows, n_blocks, max_rows, max_cols, stream) };
    Ok(())
}

/// Pre-allocate persistent device buffers for GEMV
pub fn gpu_alloc_buffers(d_x: *mut *mut f32, d_out: *mut *mut f32, max_cols: i32, max_rows: i32, stream: CudaStream) -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*mut *mut f32, *mut *mut f32, i32, i32, CudaStream)> =
        unsafe { lib.get(b"gpu_alloc_buffers")? };
    unsafe { func(d_x, d_out, max_cols, max_rows, stream) };
    Ok(())
}

/// Free persistent device buffers
pub fn gpu_free_buffers(d_x: *mut f32, d_out: *mut f32) -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*mut f32, *mut f32)> =
        unsafe { lib.get(b"gpu_free_buffers")? };
    unsafe { func(d_x, d_out) };
    Ok(())
}

// ===========================================================================
// Swamp Continuum: meta-kernel CUDA persistente
// ===========================================================================

/// Initialize swamp continuum: ring buffer + state buffer + shutdown flag
pub fn gpu_swamp_init(
    d_ring: *mut *mut std::ffi::c_void,
    d_state: *mut *mut f32,
    d_shutdown: *mut *mut i32,
    stream: CudaStream,
) -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*mut *mut std::ffi::c_void, *mut *mut f32, *mut *mut i32, CudaStream)> =
        unsafe { lib.get(b"gpu_swamp_init")? };
    unsafe { func(d_ring, d_state, d_shutdown, stream) };
    Ok(())
}

/// Launch the persistent kernel (never returns — runs in loop)
pub fn gpu_swamp_launch(
    d_ring: *const std::ffi::c_void,
    d_w_base: *const u8,
    d_state: *mut f32,
    d_shutdown: *const i32,
    stream: CudaStream,
) -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*const std::ffi::c_void, *const u8, *mut f32, *const i32, CudaStream)> =
        unsafe { lib.get(b"gpu_swamp_launch")? };
    unsafe { func(d_ring, d_w_base, d_state, d_shutdown, stream) };
    Ok(())
}

/// Enqueue an opcode for the persistent kernel
pub fn gpu_swamp_enqueue(
    d_ring: *const std::ffi::c_void,
    op_type: i32, layer_id: i32,
    x_off: i32, w_off: i32, out_off: i32,
    rows: i32, n_blocks: i32,
    stream: CudaStream,
) -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*const std::ffi::c_void, i32, i32, i32, i32, i32, i32, i32, CudaStream)> =
        unsafe { lib.get(b"gpu_swamp_enqueue")? };
    unsafe { func(d_ring, op_type, layer_id, x_off, w_off, out_off, rows, n_blocks, stream) };
    Ok(())
}

/// Signal shutdown to the persistent kernel
pub fn gpu_swamp_shutdown(d_shutdown: *const i32, stream: CudaStream) -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*const i32, CudaStream)> =
        unsafe { lib.get(b"gpu_swamp_shutdown")? };
    unsafe { func(d_shutdown, stream) };
    Ok(())
}

/// Allocate device memory (raw bytes)
pub unsafe fn gpu_alloc(bytes: usize) -> Result<*mut std::ffi::c_void> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(usize) -> *mut std::ffi::c_void> =
        unsafe { lib.get(b"gpu_alloc")? };
    let ptr = unsafe { func(bytes) };
    if ptr.is_null() {
        return Err(GpuError::KernelError(-1));
    }
    Ok(ptr)
}

/// Allocate persistent K/V buffer on GPU: [n_kv_heads, max_seq_len, head_dim]
pub fn gpu_alloc_kv_buffer(n_kv_heads: usize, max_seq_len: usize, head_dim: usize) -> Result<*mut f32> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(i32, i32, i32, *mut usize) -> *mut f32> =
        unsafe { lib.get(b"gpu_alloc_kv_buffer")? };
    let mut out_bytes: usize = 0;
    let ptr = unsafe { func(n_kv_heads as i32, max_seq_len as i32, head_dim as i32, &mut out_bytes) };
    if ptr.is_null() {
        return Err(GpuError::KernelError(-1));
    }
    info!("GPU KV buffer allocated: {} MB", out_bytes as f64 / 1e6);
    Ok(ptr)
}

/// Free GPU K/V buffer
pub fn gpu_free(ptr: *mut std::ffi::c_void) -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*mut std::ffi::c_void)> =
        unsafe { lib.get(b"gpu_free")? };
    unsafe { func(ptr) };
    Ok(())
}

/// Copy one (kv_head, pos) entry from host to GPU K/V buffer
pub fn gpu_copy_kv_to_buffer(
    d_buf: *mut f32,
    h_src: &[f32],
    kv_head: usize,
    pos: usize,
    n_kv_heads: usize,
    max_seq_len: usize,
    head_dim: usize,
) -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*mut f32, *const f32, i32, i32, i32, i32, i32) -> i32> =
        unsafe { lib.get(b"gpu_copy_kv_to_buffer")? };
    let ret = unsafe {
        func(d_buf, h_src.as_ptr(), kv_head as i32, pos as i32,
             n_kv_heads as i32, max_seq_len as i32, head_dim as i32)
    };
    if ret != 0 {
        return Err(GpuError::KernelError(ret));
    }
    Ok(())
}

/// Copy entire layer position (all kv_heads) from host to GPU buffer in one call
pub fn gpu_copy_kv_layer(
    d_buf: *mut f32,
    h_src: &[f32],
    pos: usize,
    n_kv_heads: usize,
    max_seq_len: usize,
    head_dim: usize,
) -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*mut f32, *const f32, i32, i32, i32, i32) -> i32> =
        unsafe { lib.get(b"gpu_copy_kv_layer")? };
    let ret = unsafe {
        func(d_buf, h_src.as_ptr(), pos as i32, n_kv_heads as i32, max_seq_len as i32, head_dim as i32)
    };
    if ret != 0 {
        return Err(GpuError::KernelError(ret));
    }
    Ok(())
}

/// Device-pointer attention: K/V already on GPU, Q copied fresh per layer
/// kv_stride: striding between kv_head blocks (seq_len for contiguous, max_seq_len for persistent)
pub fn gpu_attention_device(
    d_q: *const f32,
    d_k: *const f32,
    d_v: *const f32,
    d_out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    kv_stride: usize,
) -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<
        unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32, i32, i32, i32, i32, i32) -> i32,
    > = unsafe { lib.get(b"gpu_attention_device")? };
    let ret = unsafe {
        func(d_q, d_k, d_v, d_out, n_heads as i32, n_kv_heads as i32, seq_len as i32, head_dim as i32, kv_stride as i32)
    };
    if ret != 0 {
        return Err(GpuError::KernelError(ret));
    }
    Ok(())
}

/// Allocate temp Q buffer on GPU and copy from host
pub fn gpu_alloc_and_copy_q(h_q: &[f32], n_heads: usize, head_dim: usize) -> Result<*mut f32> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*const f32, i32, i32) -> *mut f32> =
        unsafe { lib.get(b"gpu_alloc_and_copy_q")? };
    let ptr = unsafe { func(h_q.as_ptr(), n_heads as i32, head_dim as i32) };
    if ptr.is_null() {
        return Err(GpuError::KernelError(-1));
    }
    Ok(ptr)
}

/// Copy attention output from GPU to host
pub fn gpu_copy_output_to_host(h_out: &mut [f32], d_out: *const f32, n_heads: usize, head_dim: usize) -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*mut f32, *const f32, i32, i32) -> i32> =
        unsafe { lib.get(b"gpu_copy_output_to_host")? };
    let ret = unsafe { func(h_out.as_mut_ptr(), d_out, n_heads as i32, head_dim as i32) };
    if ret != 0 {
        return Err(GpuError::KernelError(ret));
    }
    Ok(())
}

/// Synchronize GPU device
pub fn gpu_sync() -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn() -> i32> = unsafe { lib.get(b"gpu_sync")? };
    let ret = unsafe { func() };
    if ret != 0 {
        return Err(GpuError::KernelError(ret));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA Stream API (for async/overlapped GPU execution)
// ---------------------------------------------------------------------------

/// Opaque handle to a CUDA stream
#[derive(Debug, Clone, Copy)]
pub struct CudaStream(*mut std::ffi::c_void);

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    pub fn null() -> Self {
        CudaStream(std::ptr::null_mut())
    }

    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    pub fn as_raw(&self) -> *mut std::ffi::c_void {
        self.0
    }
}

/// Create a CUDA stream
pub fn gpu_stream_create() -> Result<CudaStream> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn() -> *mut std::ffi::c_void> =
        unsafe { lib.get(b"gpu_stream_create")? };
    let ptr = unsafe { func() };
    if ptr.is_null() {
        return Err(GpuError::KernelError(-1));
    }
    Ok(CudaStream(ptr))
}

/// Destroy a CUDA stream
pub fn gpu_stream_destroy(stream: CudaStream) -> Result<()> {
    if stream.is_null() {
        return Ok(());
    }
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*mut std::ffi::c_void)> =
        unsafe { lib.get(b"gpu_stream_destroy")? };
    unsafe { func(stream.0) };
    Ok(())
}

/// Synchronize a CUDA stream (block until all work on stream completes)
pub fn gpu_stream_synchronize(stream: CudaStream) -> Result<()> {
    if stream.is_null() {
        return Err(GpuError::KernelError(-1));
    }
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*mut std::ffi::c_void) -> i32> =
        unsafe { lib.get(b"gpu_stream_synchronize")? };
    let ret = unsafe { func(stream.0) };
    if ret != 0 {
        return Err(GpuError::KernelError(ret));
    }
    Ok(())
}

/// Async copy host -> device on given stream
/// h_src should be page-locked for true async behavior
pub fn gpu_copy_to_device_async(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    bytes: usize,
    stream: CudaStream,
) -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*mut std::ffi::c_void, *const std::ffi::c_void, usize, *mut std::ffi::c_void) -> i32> =
        unsafe { lib.get(b"gpu_copy_to_device_async")? };
    let ret = unsafe { func(dst, src, bytes, stream.0) };
    if ret != 0 {
        return Err(GpuError::KernelError(ret));
    }
    Ok(())
}

/// Async copy device -> host on given stream
pub fn gpu_copy_to_host_async(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    bytes: usize,
    stream: CudaStream,
) -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*mut std::ffi::c_void, *const std::ffi::c_void, usize, *mut std::ffi::c_void) -> i32> =
        unsafe { lib.get(b"gpu_copy_to_host_async")? };
    let ret = unsafe { func(dst, src, bytes, stream.0) };
    if ret != 0 {
        return Err(GpuError::KernelError(ret));
    }
    Ok(())
}

/// Async batch KV copy for one position (all kv_heads)
pub fn gpu_copy_kv_layer_async(
    d_buf: *mut f32,
    h_src: &[f32],
    pos: usize,
    n_kv_heads: usize,
    max_seq_len: usize,
    head_dim: usize,
    stream: CudaStream,
) -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*mut f32, *const f32, i32, i32, i32, i32, *mut std::ffi::c_void) -> i32> =
        unsafe { lib.get(b"gpu_copy_kv_layer_async")? };
    let ret = unsafe {
        func(d_buf, h_src.as_ptr(), pos as i32, n_kv_heads as i32, max_seq_len as i32, head_dim as i32, stream.0)
    };
    if ret != 0 {
        return Err(GpuError::KernelError(ret));
    }
    Ok(())
}

/// Stream-based attention: Q already on device
pub fn gpu_attention_streamed(
    d_q: *const f32,
    d_k_cache: *const f32,
    d_v_cache: *const f32,
    d_out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    kv_stride: usize,
    stream: CudaStream,
) -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<
        unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32, i32, i32, i32, i32, i32, *mut std::ffi::c_void) -> i32,
    > = unsafe { lib.get(b"gpu_attention_streamed")? };
    let ret = unsafe {
        func(d_q, d_k_cache, d_v_cache, d_out,
             n_heads as i32, n_kv_heads as i32, seq_len as i32, head_dim as i32, kv_stride as i32,
             stream.0)
    };
    if ret != 0 {
        return Err(GpuError::KernelError(ret));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// FP16 KV cache support (half precision K/V for 2x memory bandwidth)
// ---------------------------------------------------------------------------

/// Stream-based attention with FP16 KV cache
pub fn gpu_attention_streamed_half(
    d_q: *const f32,
    d_k_cache: *const std::ffi::c_void,  // half*
    d_v_cache: *const std::ffi::c_void,  // half*
    d_out: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    kv_stride: usize,
    stream: CudaStream,
) -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<
        unsafe extern "C" fn(*const f32, *const std::ffi::c_void, *const std::ffi::c_void, *mut f32, i32, i32, i32, i32, i32, *mut std::ffi::c_void) -> i32,
    > = unsafe { lib.get(b"gpu_attention_streamed_half")? };
    let ret = unsafe {
        func(d_q, d_k_cache, d_v_cache, d_out,
             n_heads as i32, n_kv_heads as i32, seq_len as i32, head_dim as i32, kv_stride as i32,
             stream.0)
    };
    if ret != 0 {
        return Err(GpuError::KernelError(ret));
    }
    Ok(())
}

/// Async copy float host data → half GPU buffer for one layer position
pub fn gpu_copy_kv_layer_async_half(
    d_buf: *mut std::ffi::c_void,  // half*
    h_src: &[f32],
    pos: usize,
    n_kv_heads: usize,
    max_seq_len: usize,
    head_dim: usize,
    stream: CudaStream,
) -> Result<()> {
    let lib = try_lib()?;
    let func: Symbol<
        unsafe extern "C" fn(*mut std::ffi::c_void, *const f32, i32, i32, i32, i32, *mut std::ffi::c_void) -> i32,
    > = unsafe { lib.get(b"gpu_copy_kv_layer_async_half")? };
    let ret = unsafe {
        func(d_buf, h_src.as_ptr(), pos as i32, n_kv_heads as i32, max_seq_len as i32, head_dim as i32, stream.0)
    };
    if ret != 0 {
        return Err(GpuError::KernelError(ret));
    }
    Ok(())
}

/// Allocate persistent half-precision KV buffer
pub fn gpu_alloc_kv_buffer_half(
    n_kv_heads: usize,
    max_seq_len: usize,
    head_dim: usize,
) -> Result<*mut std::ffi::c_void> {
    let lib = try_lib()?;
    let func: Symbol<
        unsafe extern "C" fn(i32, i32, i32, *mut usize) -> *mut std::ffi::c_void,
    > = unsafe { lib.get(b"gpu_alloc_kv_buffer_half")? };
    let mut out_bytes: usize = 0;
    let ptr = unsafe { func(n_kv_heads as i32, max_seq_len as i32, head_dim as i32, &mut out_bytes) };
    if ptr.is_null() {
        return Err(GpuError::KernelError(-1));
    }
    Ok(ptr)
}

/// Create a CUDA Graph executable for attention with FP16 KV cache
pub fn gpu_graph_create_attention_half(
    d_q: *const f32,
    d_k_cache: *const std::ffi::c_void,  // half*
    d_v_cache: *const std::ffi::c_void,  // half*
    d_scores: *mut f32,
    d_output: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    kv_stride: usize,
) -> Result<AttentionGraph> {
    let lib = try_lib()?;
    let func: Symbol<
        unsafe extern "C" fn(*const f32, *const std::ffi::c_void, *const std::ffi::c_void, *mut f32, *mut f32, i32, i32, i32, i32, i32) -> *mut std::ffi::c_void,
    > = unsafe { lib.get(b"gpu_graph_create_attention_half")? };
    let ptr = unsafe {
        func(d_q, d_k_cache, d_v_cache, d_scores, d_output,
             n_heads as i32, n_kv_heads as i32, seq_len as i32, head_dim as i32, kv_stride as i32)
    };
    if ptr.is_null() {
        return Err(GpuError::KernelError(-1));
    }
    Ok(AttentionGraph(ptr))
}

// ---------------------------------------------------------------------------
// CUDA Graph API: reusable attention compute graph
// Eliminates kernel launch overhead for repeated attention calls
// ---------------------------------------------------------------------------

/// Opaque handle to a CUDA Graph executable for attention compute
pub struct AttentionGraph(*mut std::ffi::c_void);

unsafe impl Send for AttentionGraph {}
unsafe impl Sync for AttentionGraph {}

impl AttentionGraph {
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
}

/// Create a CUDA Graph executable for attention compute.
/// All parameters (including seq_len) are FIXED at graph creation time.
/// Caller should cache graphs by seq_len and create one per distinct value.
pub fn gpu_graph_create_attention(
    d_q: *const f32,
    d_k_cache: *const f32,
    d_v_cache: *const f32,
    d_scores: *mut f32,
    d_output: *mut f32,
    n_heads: usize,
    n_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    kv_stride: usize,
) -> Result<AttentionGraph> {
    let lib = try_lib()?;
    let func: Symbol<
        unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32, *mut f32, i32, i32, i32, i32, i32) -> *mut std::ffi::c_void,
    > = unsafe { lib.get(b"gpu_graph_create_attention")? };
    let ptr = unsafe {
        func(d_q, d_k_cache, d_v_cache, d_scores, d_output,
             n_heads as i32, n_kv_heads as i32, seq_len as i32, head_dim as i32, kv_stride as i32)
    };
    if ptr.is_null() {
        return Err(GpuError::KernelError(-1));
    }
    Ok(AttentionGraph(ptr))
}

/// Replay a fixed-parameter attention graph.
/// All parameters must match the graph creation parameters exactly.
pub fn gpu_graph_replay_attention(
    graph: &AttentionGraph,
    stream: CudaStream,
) -> Result<()> {
    if graph.is_null() {
        return Err(GpuError::KernelError(-1));
    }
    let lib = try_lib()?;
    let func: Symbol<
        unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void) -> i32,
    > = unsafe { lib.get(b"gpu_graph_replay_attention")? };
    let ret = unsafe { func(graph.0, stream.0) };
    if ret != 0 {
        return Err(GpuError::KernelError(ret));
    }
    Ok(())
}

/// Destroy an attention graph executable
pub fn gpu_graph_destroy(graph: AttentionGraph) -> Result<()> {
    if graph.is_null() {
        return Ok(());
    }
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*mut std::ffi::c_void)> =
        unsafe { lib.get(b"gpu_graph_destroy")? };
    unsafe { func(graph.0) };
    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA Event API (for HLC timeline correlation)
// ---------------------------------------------------------------------------

/// Opaque handle to a CUDA event
#[derive(Debug, Clone, Copy)]
pub struct CudaEvent(*mut std::ffi::c_void);

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl CudaEvent {
    /// Null/invalid event handle
    pub fn null() -> Self {
        CudaEvent(std::ptr::null_mut())
    }

    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
}

/// Create a CUDA event
pub fn gpu_event_create() -> Result<CudaEvent> {
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn() -> *mut std::ffi::c_void> =
        unsafe { lib.get(b"gpu_event_create")? };
    let ptr = unsafe { func() };
    if ptr.is_null() {
        return Err(GpuError::KernelError(-1));
    }
    Ok(CudaEvent(ptr))
}

/// Record a CUDA event on the given stream (null = default stream)
pub fn gpu_event_record(event: CudaEvent, stream: *mut std::ffi::c_void) -> Result<()> {
    if event.is_null() {
        return Err(GpuError::KernelError(-1));
    }
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void) -> i32> =
        unsafe { lib.get(b"gpu_event_record")? };
    let ret = unsafe { func(event.0, stream) };
    if ret != 0 {
        return Err(GpuError::KernelError(ret));
    }
    Ok(())
}

/// Synchronize a CUDA event (block until recorded work completes)
pub fn gpu_event_synchronize(event: CudaEvent) -> Result<()> {
    if event.is_null() {
        return Err(GpuError::KernelError(-1));
    }
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*mut std::ffi::c_void) -> i32> =
        unsafe { lib.get(b"gpu_event_synchronize")? };
    let ret = unsafe { func(event.0) };
    if ret != 0 {
        return Err(GpuError::KernelError(ret));
    }
    Ok(())
}

/// Elapsed time between two CUDA events in milliseconds
pub fn gpu_event_elapsed_ms(start: CudaEvent, end: CudaEvent) -> Result<f32> {
    if start.is_null() || end.is_null() {
        return Err(GpuError::KernelError(-1));
    }
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void) -> f32> =
        unsafe { lib.get(b"gpu_event_elapsed_ms")? };
    let ms = unsafe { func(start.0, end.0) };
    if ms < 0.0 {
        return Err(GpuError::KernelError(-1));
    }
    Ok(ms)
}

/// Destroy a CUDA event
pub fn gpu_event_destroy(event: CudaEvent) -> Result<()> {
    if event.is_null() {
        return Ok(());
    }
    let lib = try_lib()?;
    let func: Symbol<unsafe extern "C" fn(*mut std::ffi::c_void)> =
        unsafe { lib.get(b"gpu_event_destroy")? };
    unsafe { func(event.0) };
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_init() {
        if !gpu_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }
        assert!(gpu_init().is_ok());
    }

    #[test]
    fn test_gpu_attention_small() {
        if !gpu_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }
        gpu_init().unwrap();

        let n_heads = 2;
        let n_kv_heads = 1;
        let seq_len = 4;
        let head_dim = 8;

        let q = vec![1.0f32; n_heads * head_dim];
        let mut k_cache = vec![0.0f32; n_kv_heads * seq_len * head_dim];
        let mut v_cache = vec![0.0f32; n_kv_heads * seq_len * head_dim];

        for t in 0..seq_len {
            for d in 0..head_dim {
                k_cache[t * head_dim + d] = t as f32 + d as f32 * 0.1;
                v_cache[t * head_dim + d] = 1.0;
            }
        }

        let mut output = vec![0.0f32; n_heads * head_dim];
        let result = gpu_attention_forward(
            &q, &k_cache, &v_cache, &mut output,
            n_heads, n_kv_heads, seq_len, head_dim,
        );
        assert!(result.is_ok(), "attention failed: {:?}", result);

        let sum: f32 = output.iter().sum();
        assert!(sum > 0.0, "output sum should be positive, got {}", sum);
        println!("GPU attention test passed. output[0..4]: {:?}", &output[..4.min(output.len())]);
    }
}
