use std::sync::atomic::{AtomicU32, Ordering};

// ---------------------------------------------------------------------------
// GpuStreamManager: manages CUDA streams and pre-allocated buffers
// Provides async GPU operations without per-layer alloc/free overhead
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
pub struct GpuStreamManager {
    pub compute_stream: swamp_gpu::CudaStream,
    pub copy_stream: swamp_gpu::CudaStream,
    pub d_q: *mut f32,
    pub d_out: *mut f32,
    pub d_scores: Option<*mut f32>,
    /// Lazy CUDA Graph cache: graph_cache[seq_len] = Some(graph) or None
    graph_cache: Vec<Option<swamp_gpu::AttentionGraph>>,
    max_seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    initialised: bool,
}

#[cfg(feature = "gpu")]
impl GpuStreamManager {
    pub fn new(n_heads: usize, n_kv_heads: usize, max_seq_len: usize, head_dim: usize) -> Option<Self> {
        let cs = swamp_gpu::gpu_stream_create().ok()?;
        let cp = swamp_gpu::gpu_stream_create().ok()?;

        let q_bytes = n_heads * head_dim * 4;
        let out_bytes = n_heads * head_dim * 4;

        let d_q: *mut f32 = unsafe { swamp_gpu::gpu_alloc(q_bytes).ok()? } as *mut f32;
        let d_out: *mut f32 = unsafe { swamp_gpu::gpu_alloc(out_bytes).ok()? } as *mut f32;
        let d_scores = None;
        let graph_cache = (0..=max_seq_len).map(|_| None).collect();

        Some(GpuStreamManager {
            compute_stream: cs,
            copy_stream: cp,
            d_q,
            d_out,
            d_scores,
            graph_cache,
            max_seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
            initialised: true,
        })
    }

    /// Ensure scores buffer is allocated (lazy init, one allocation for max_seq_len)
    fn ensure_scores(&mut self) -> bool {
        if self.d_scores.is_some() {
            return true;
        }
        let scores_bytes = self.n_heads * self.max_seq_len * 4;
        let ptr: *mut f32 = match unsafe { swamp_gpu::gpu_alloc(scores_bytes) } {
            Ok(p) => p as *mut f32,
            Err(_) => return false,
        };
        if ptr.is_null() { return false; }
        self.d_scores = Some(ptr);
        true
    }

    /// Get or create a CUDA Graph for the given seq_len (FP16 KV cache).
    /// Falls back to None if graph creation fails.
    fn get_or_create_graph_half(
        &mut self,
        d_k_cache: *const std::ffi::c_void,
        d_v_cache: *const std::ffi::c_void,
        seq_len: usize,
        kv_stride: usize,
    ) -> Option<&swamp_gpu::AttentionGraph> {
        if seq_len > self.max_seq_len || seq_len == 0 {
            return None;
        }
        // Check cache first
        if self.graph_cache[seq_len].is_some() {
            return self.graph_cache[seq_len].as_ref();
        }
        // Create new graph
        let d_scores = self.d_scores.unwrap_or(std::ptr::null_mut());
        if d_scores.is_null() {
            return None;
        }
        match swamp_gpu::gpu_graph_create_attention_half(
            self.d_q as *const f32,
            d_k_cache,
            d_v_cache,
            d_scores,
            self.d_out,
            self.n_heads,
            self.n_kv_heads,
            seq_len,
            self.head_dim,
            kv_stride,
        ) {
            Ok(graph) => {
                self.graph_cache[seq_len] = Some(graph);
                self.graph_cache[seq_len].as_ref()
            }
            Err(_) => None,
        }
    }

    /// Async copy Q to device buffer using compute stream
    pub fn copy_q_async(&self, h_q: &[f32]) -> bool {
        let bytes = self.n_heads * self.head_dim * 4;
        swamp_gpu::gpu_copy_to_device_async(
            self.d_q as *mut _,
            h_q.as_ptr() as *const _,
            bytes,
            self.compute_stream,
        ).is_ok()
    }

    /// Async copy K/V to GPU persistent half buffer using copy stream
    pub fn copy_kv_async(&self, d_buf: *mut std::ffi::c_void, h_kv: &[f32], pos: usize, n_kv_heads: usize) -> bool {
        swamp_gpu::gpu_copy_kv_layer_async_half(
            d_buf, h_kv, pos, n_kv_heads, self.max_seq_len, self.head_dim, self.copy_stream,
        ).is_ok()
    }

    /// Launch attention kernel on compute stream (async) with FP16 KV cache.
    /// Uses CUDA Graph if available, falls back to streamed kernel launch.
    pub fn launch_attention_half(
        &mut self,
        d_k_cache: *const std::ffi::c_void,
        d_v_cache: *const std::ffi::c_void,
        seq_len: usize,
        kv_stride: usize,
        n_kv_heads: usize,
    ) -> bool {
        if !self.ensure_scores() {
            return false;
        }
        let d_scores = self.d_scores.unwrap_or(std::ptr::null_mut());
        if d_scores.is_null() {
            return false;
        }

        // Extract pointers before mutable borrow for graph cache
        let d_q = self.d_q as *const f32;
        let d_out = self.d_out;
        let n_heads = self.n_heads;
        let head_dim = self.head_dim;
        let stream = self.compute_stream;

        // Try CUDA Graph path first (graph was created with fixed params for this seq_len)
        if let Some(graph) = self.get_or_create_graph_half(d_k_cache, d_v_cache, seq_len, kv_stride) {
            if swamp_gpu::gpu_graph_replay_attention(graph, stream).is_ok() {
                return true;
            }
        }

        // Fallback: streamed kernel launch (half K/V)
        swamp_gpu::gpu_attention_streamed_half(
            d_q, d_k_cache, d_v_cache, d_out,
            n_heads, n_kv_heads, seq_len, head_dim, kv_stride, stream,
        ).is_ok()
    }

    /// Async copy output from GPU to host
    pub fn copy_output_async(&self, h_out: &mut [f32]) -> bool {
        let bytes = self.n_heads * self.head_dim * 4;
        swamp_gpu::gpu_copy_to_host_async(
            h_out.as_mut_ptr() as *mut _,
            self.d_out as *const _,
            bytes,
            self.compute_stream,
        ).is_ok()
    }

    /// Wait for compute stream to complete all pending operations
    pub fn sync_compute(&self) -> bool {
        swamp_gpu::gpu_stream_synchronize(self.compute_stream).is_ok()
    }

    /// Wait for copy stream to complete
    pub fn sync_copy(&self) -> bool {
        swamp_gpu::gpu_stream_synchronize(self.copy_stream).is_ok()
    }

    /// Check if all streams are operational
    pub fn is_operational(&self) -> bool {
        self.initialised
    }
}

#[cfg(feature = "gpu")]
impl Drop for GpuStreamManager {
    fn drop(&mut self) {
        let _ = swamp_gpu::gpu_stream_synchronize(self.compute_stream);
        let _ = swamp_gpu::gpu_stream_synchronize(self.copy_stream);
        // Destroy cached CUDA Graphs
        for graph in self.graph_cache.drain(..) {
            if let Some(g) = graph {
                swamp_gpu::gpu_graph_destroy(g).ok();
            }
        }
        if let Some(s) = self.d_scores {
            swamp_gpu::gpu_free(s as *mut _).ok();
        }
        swamp_gpu::gpu_free(self.d_q as *mut _).ok();
        swamp_gpu::gpu_free(self.d_out as *mut _).ok();
        let _ = swamp_gpu::gpu_stream_destroy(self.compute_stream);
        let _ = swamp_gpu::gpu_stream_destroy(self.copy_stream);
    }
}

// Non-GPU stub
#[cfg(not(feature = "gpu"))]
pub struct GpuStreamManager;
#[cfg(not(feature = "gpu"))]
impl GpuStreamManager {
    pub fn new(_: usize, _: usize, _: usize, _: usize) -> Option<Self> { None }
    pub fn is_operational(&self) -> bool { false }
}

// ---------------------------------------------------------------------------
// PerLayerGpuState: orchestrates one layer's GPU attention execution
// GPU device pointers are just addresses — safe to Send/Sync between threads
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
pub struct PerLayerGpuState {
    pub gpu: Box<GpuStreamManager>,
    pub d_k_buf: *mut std::ffi::c_void,
    pub d_v_buf: *mut std::ffi::c_void,
    pub max_seq_len: usize,
    pub n_kv_heads: usize,
}

#[cfg(feature = "gpu")]
unsafe impl Send for PerLayerGpuState {}
#[cfg(feature = "gpu")]
unsafe impl Sync for PerLayerGpuState {}

#[cfg(feature = "gpu")]
impl PerLayerGpuState {
    /// Pre-upload K/V to the GPU FP16 buffer for the given position.
    /// Must be called after `kv_cache.save()` and before `execute_attention_async()`.
    /// Returns true if the copy was submitted successfully.
    pub fn upload_kv_async(&mut self, h_k: &[f32], h_v: &[f32], pos: usize) -> bool {
        self.gpu.copy_kv_async(self.d_k_buf, h_k, pos, self.n_kv_heads)
            && self.gpu.copy_kv_async(self.d_v_buf, h_v, pos, self.n_kv_heads)
    }

    /// Run GPU attention for one layer asynchronously, using pre-uploaded KV
    /// (uploaded via `upload_kv_async`). Does NOT copy KV — assumes resident on GPU.
    /// Returns true if GPU execution was launched successfully.
    pub fn execute_attention_async(
        &mut self,
        h_q: &[f32],
        h_out: &mut [f32],
        pos: usize,
        seq_len: usize,
    ) -> bool {
        if !self.gpu.is_operational() {
            return false;
        }

        // Q copy + launch attention + output copy on compute stream
        let q_ok = self.gpu.copy_q_async(h_q);
        if !q_ok { return false; }

        let launch_ok = self.gpu.launch_attention_half(
            self.d_k_buf as *const std::ffi::c_void,
            self.d_v_buf as *const std::ffi::c_void,
            seq_len,
            self.max_seq_len,
            self.n_kv_heads,
        );
        if !launch_ok { return false; }

        let copy_out_ok = self.gpu.copy_output_async(h_out);
        if !copy_out_ok { return false; }

        true
    }

    /// Block until GPU attention completes (wait for compute stream)
    pub fn sync(&self) -> bool {
        self.gpu.sync_compute()
    }
}

#[cfg(feature = "gpu")]
impl Drop for PerLayerGpuState {
    fn drop(&mut self) {
        let _ = self.gpu.sync_compute();
        let _ = self.gpu.sync_copy();
        swamp_gpu::gpu_free(self.d_k_buf).ok();
        swamp_gpu::gpu_free(self.d_v_buf).ok();
    }
}

// ---------------------------------------------------------------------------
// WorkToken: lightweight RAII guard for tracking pipeline depth
// ---------------------------------------------------------------------------

/// Tracks how many in-flight GPU operations exist.
/// Used to regulate pipeline depth (max 2: one executing, one queued).
pub struct PipelineToken {
    counter: &'static AtomicU32,
}

impl PipelineToken {
    pub fn acquire(counter: &'static AtomicU32, max_depth: u32) -> Option<Self> {
        loop {
            let current = counter.load(Ordering::Relaxed);
            if current >= max_depth {
                return None;
            }
            if counter.compare_exchange_weak(current, current + 1, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                return Some(PipelineToken { counter });
            }
        }
    }
}

impl Drop for PipelineToken {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_token() {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let t1 = PipelineToken::acquire(&COUNTER, 2).unwrap();
        assert_eq!(COUNTER.load(Ordering::Relaxed), 1);
        let t2 = PipelineToken::acquire(&COUNTER, 2).unwrap();
        assert_eq!(COUNTER.load(Ordering::Relaxed), 2);
        let t3 = PipelineToken::acquire(&COUNTER, 2);
        assert!(t3.is_none()); // max depth reached
        drop(t2);
        assert_eq!(COUNTER.load(Ordering::Relaxed), 1);
        drop(t1);
        assert_eq!(COUNTER.load(Ordering::Relaxed), 0);
    }
}
