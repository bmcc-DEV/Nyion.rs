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
    pub window_size: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    // Per-layer weight pointers (uploaded once at init)
    pub d_q_weight: Vec<*mut u8>,
    pub d_k_weight: Vec<*mut u8>,
    pub d_v_weight: Vec<*mut u8>,
    pub d_o_weight: Vec<*mut u8>,
    pub d_gate_weight: Vec<*mut u8>,
    pub d_up_weight: Vec<*mut u8>,
    pub d_down_weight: Vec<*mut u8>,
    // Pre-allocated GEMV input/output buffers (shared across all layers)
    pub d_gemv_x: *mut f32,
    pub d_gemv_out: *mut f32,
    pub max_gemv_cols: i32,
    pub max_gemv_rows: i32,
    // Per-layer CUDA Graph handles (created lazily on first use of each layer)
    pub graph_qkv: Vec<Option<swamp_gpu::GemvGraph>>,
    pub graph_gate_up: Vec<Option<swamp_gpu::GemvGraph>>,
    pub graph_o: Vec<Option<swamp_gpu::GemvGraph>>,
    pub graph_down: Vec<Option<swamp_gpu::GemvGraph>>,
}

#[cfg(feature = "gpu")]
unsafe impl Send for PerLayerGpuState {}
#[cfg(feature = "gpu")]
unsafe impl Sync for PerLayerGpuState {}

#[cfg(feature = "gpu")]
impl PerLayerGpuState {
    pub fn new(
        window_size: usize, num_heads: usize, n_kv_heads: usize, head_dim: usize,
        num_layers: usize,
        d_q_weight: Vec<*mut u8>, d_k_weight: Vec<*mut u8>, d_v_weight: Vec<*mut u8>,
        d_o_weight: Vec<*mut u8>, d_gate_weight: Vec<*mut u8>, d_up_weight: Vec<*mut u8>,
        d_down_weight: Vec<*mut u8>,
        d_gemv_x: *mut f32, d_gemv_out: *mut f32,
        max_gemv_cols: i32, max_gemv_rows: i32,
    ) -> Option<Self> {
        swamp_gpu::gpu_init().ok()?;
        let d_k_buf = swamp_gpu::gpu_alloc_kv_buffer_half(n_kv_heads, window_size, head_dim).ok()?;
        let d_v_buf = swamp_gpu::gpu_alloc_kv_buffer_half(n_kv_heads, window_size, head_dim).ok()?;
        let gpu = GpuStreamManager::new(num_heads, n_kv_heads, window_size, head_dim)?;

        let n_none = vec![None; num_layers.max(1)];

        Some(Self {
            gpu: Box::new(gpu),
            d_k_buf, d_v_buf,
            window_size,
            n_kv_heads,
            head_dim,
            num_layers,
            d_q_weight, d_k_weight, d_v_weight,
            d_o_weight, d_gate_weight, d_up_weight, d_down_weight,
            d_gemv_x, d_gemv_out,
            max_gemv_cols, max_gemv_rows,
            graph_qkv: n_none.clone(),
            graph_gate_up: n_none.clone(),
            graph_o: n_none.clone(),
            graph_down: n_none.clone(),
        })
    }

    /// Pre-upload K/V to the GPU FP16 buffer for the given position.
    /// Only the last `window_size` positions are kept in VRAM.
    /// Returns true if the copy was submitted successfully.
    pub fn upload_kv_async(&mut self, h_k: &[f32], h_v: &[f32], pos: usize) -> bool {
        // Map position into window buffer (ring buffer of window_size)
        let wpos = pos % self.window_size;
        self.gpu.copy_kv_async(self.d_k_buf, h_k, wpos, self.n_kv_heads)
            && self.gpu.copy_kv_async(self.d_v_buf, h_v, wpos, self.n_kv_heads)
    }

    /// Run GPU attention with sparsa window.
    /// Only attends within the last `window_size` tokens in VRAM.
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
        let attn_len = seq_len.min(self.window_size);
        if attn_len == 0 { return false; }

        let q_ok = self.gpu.copy_q_async(h_q);
        if !q_ok { return false; }

        let launch_ok = self.gpu.launch_attention_half(
            self.d_k_buf as *const std::ffi::c_void,
            self.d_v_buf as *const std::ffi::c_void,
            attn_len,
            self.window_size,
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

    /// Upload ring weight tensor to GPU VRAM (one-time).
    pub fn upload_weight(h_w: *const u8, bytes: usize, stream: swamp_gpu::CudaStream) -> Option<*mut u8> {
        let mut d_w: *mut u8 = std::ptr::null_mut();
        swamp_gpu::gpu_upload_weights(h_w, &mut d_w as *mut *mut u8, bytes, stream).ok()?;
        Some(d_w)
    }

    // =====================================================================
    // CUDA Graph GEMV methods
    // =====================================================================

    /// Single GEMV (no graph caching, single shot) with layer index.
    pub fn execute_gemv_async(
        &mut self,
        layer_idx: usize,
        d_w: *const u8,
        h_x: &[f32],
        h_out: &mut [f32],
        n_rows: i32,
        n_blocks: i32,
    ) -> bool {
        if d_w.is_null() { return false; }
        let stream = self.gpu.compute_stream;
        let x_bytes = (n_blocks as usize) * 256 * 4;
        let out_bytes = (n_rows as usize) * 4;

        // Per-layer graph: replay if cached
        if let Some(ref graph) = self.graph_o[layer_idx] {
            return swamp_gpu::gpu_graph_replay_gemv(graph, stream).is_ok();
        }

        // Create graph for this layer, cache it
        let new_graph = swamp_gpu::gpu_graph_create_gemv_single(
            d_w, self.d_gemv_x, self.d_gemv_out,
            h_x.as_ptr(), h_out.as_mut_ptr(),
            n_rows, n_blocks, x_bytes as i32, out_bytes as i32,
        );
        match new_graph {
            Ok(g) => {
                self.graph_o[layer_idx] = Some(g);
                swamp_gpu::gpu_graph_replay_gemv(self.graph_o[layer_idx].as_ref().unwrap(), stream).is_ok()
            }
            Err(_) => {
                swamp_gpu::gpu_copy_to_device_async(
                    self.d_gemv_x as *mut _, h_x.as_ptr() as *const _, x_bytes, stream,
                ).is_ok()
                && swamp_gpu::gpu_gemv_q4k(
                    d_w, self.d_gemv_x as *const f32, self.d_gemv_out,
                    n_rows, n_blocks, stream,
                ).is_ok()
                && swamp_gpu::gpu_copy_to_host_async(
                    h_out.as_mut_ptr() as *mut _, self.d_gemv_out as *const _,
                    out_bytes, stream,
                ).is_ok()
            }
        }
    }

    /// Batch Q/K/V GEMVs using per-layer CUDA Graph.
    pub fn gemv_qkv_async(
        &mut self,
        layer_idx: usize,
        d_w_q: *const u8, d_w_k: *const u8, d_w_v: *const u8,
        h_x: &[f32],
        h_q: &mut [f32], h_k: &mut [f32], h_v: &mut [f32],
        n_rows_q: i32, n_rows_k: i32, n_rows_v: i32,
        n_blocks: i32,
    ) -> bool {
        if d_w_q.is_null() || d_w_k.is_null() || d_w_v.is_null() { return false; }
        let stream = self.gpu.compute_stream;
        let x_bytes = (n_blocks as usize) * 256 * 4;

        // Per-layer graph replay
        if let Some(ref graph) = self.graph_qkv[layer_idx] {
            return swamp_gpu::gpu_graph_replay_gemv(graph, stream).is_ok();
        }

        // First use: create graph for this layer
        let new_graph = swamp_gpu::gpu_graph_create_gemv_qkv(
            d_w_q, d_w_k, d_w_v,
            self.d_gemv_x, self.d_gemv_out,
            h_x.as_ptr(), h_q.as_mut_ptr(), h_k.as_mut_ptr(), h_v.as_mut_ptr(),
            n_rows_q, n_rows_k, n_rows_v, n_blocks,
            x_bytes as i32,
            (n_rows_q as usize * 4) as i32,
            (n_rows_k as usize * 4) as i32,
            (n_rows_v as usize * 4) as i32,
        );
        match new_graph {
            Ok(g) => {
                self.graph_qkv[layer_idx] = Some(g);
                swamp_gpu::gpu_graph_replay_gemv(self.graph_qkv[layer_idx].as_ref().unwrap(), stream).is_ok()
            }
            Err(_) => {
                if !swamp_gpu::gpu_copy_to_device_async(
                    self.d_gemv_x as *mut _, h_x.as_ptr() as *const _, x_bytes, stream,
                ).is_ok() { return false; }
                if !swamp_gpu::gpu_gemv_q4k(d_w_q, self.d_gemv_x as *const f32, self.d_gemv_out, n_rows_q, n_blocks, stream).is_ok() { return false; }
                if !swamp_gpu::gpu_copy_to_host_async(h_q.as_mut_ptr() as *mut _, self.d_gemv_out as *const _, (n_rows_q as usize) * 4, stream).is_ok() { return false; }
                if !swamp_gpu::gpu_gemv_q4k(d_w_k, self.d_gemv_x as *const f32, self.d_gemv_out, n_rows_k, n_blocks, stream).is_ok() { return false; }
                if !swamp_gpu::gpu_copy_to_host_async(h_k.as_mut_ptr() as *mut _, self.d_gemv_out as *const _, (n_rows_k as usize) * 4, stream).is_ok() { return false; }
                if !swamp_gpu::gpu_gemv_q4k(d_w_v, self.d_gemv_x as *const f32, self.d_gemv_out, n_rows_v, n_blocks, stream).is_ok() { return false; }
                if !swamp_gpu::gpu_copy_to_host_async(h_v.as_mut_ptr() as *mut _, self.d_gemv_out as *const _, (n_rows_v as usize) * 4, stream).is_ok() { return false; }
                true
            }
        }
    }

    /// Batch Gate/Up GEMVs using per-layer CUDA Graph.
    pub fn gemv_gate_up_async(
        &mut self,
        layer_idx: usize,
        d_w_gate: *const u8, d_w_up: *const u8,
        h_x: &[f32],
        h_gate: &mut [f32], h_up: &mut [f32],
        n_rows: i32, n_blocks: i32,
    ) -> bool {
        if d_w_gate.is_null() || d_w_up.is_null() { return false; }
        let stream = self.gpu.compute_stream;
        let x_bytes = (n_blocks as usize) * 256 * 4;

        if let Some(ref graph) = self.graph_gate_up[layer_idx] {
            return swamp_gpu::gpu_graph_replay_gemv(graph, stream).is_ok();
        }

        let new_graph = swamp_gpu::gpu_graph_create_gemv_gate_up(
            d_w_gate, d_w_up,
            self.d_gemv_x, self.d_gemv_out,
            h_x.as_ptr(), h_gate.as_mut_ptr(), h_up.as_mut_ptr(),
            n_rows, n_blocks,
            x_bytes as i32,
            (n_rows as usize * 4) as i32,
        );
        match new_graph {
            Ok(g) => {
                self.graph_gate_up[layer_idx] = Some(g);
                swamp_gpu::gpu_graph_replay_gemv(self.graph_gate_up[layer_idx].as_ref().unwrap(), stream).is_ok()
            }
            Err(_) => {
                if !swamp_gpu::gpu_copy_to_device_async(self.d_gemv_x as *mut _, h_x.as_ptr() as *const _, x_bytes, stream).is_ok() { return false; }
                let out_bytes = (n_rows as usize) * 4;
                if !swamp_gpu::gpu_gemv_q4k(d_w_gate, self.d_gemv_x as *const f32, self.d_gemv_out, n_rows, n_blocks, stream).is_ok() { return false; }
                if !swamp_gpu::gpu_copy_to_host_async(h_gate.as_mut_ptr() as *mut _, self.d_gemv_out as *const _, out_bytes, stream).is_ok() { return false; }
                if !swamp_gpu::gpu_gemv_q4k(d_w_up, self.d_gemv_x as *const f32, self.d_gemv_out, n_rows, n_blocks, stream).is_ok() { return false; }
                if !swamp_gpu::gpu_copy_to_host_async(h_up.as_mut_ptr() as *mut _, self.d_gemv_out as *const _, out_bytes, stream).is_ok() { return false; }
                true
            }
        }
    }
}

#[cfg(feature = "gpu")]
impl Drop for PerLayerGpuState {
    fn drop(&mut self) {
        let _ = self.gpu.sync_compute();
        let _ = self.gpu.sync_copy();
        swamp_gpu::gpu_free(self.d_k_buf).ok();
        swamp_gpu::gpu_free(self.d_v_buf).ok();
        swamp_gpu::gpu_free(self.d_gemv_x as *mut _).ok();
        swamp_gpu::gpu_free(self.d_gemv_out as *mut _).ok();
        for &p in &self.d_q_weight { if !p.is_null() { swamp_gpu::gpu_free_weights(p).ok(); } }
        for &p in &self.d_k_weight { if !p.is_null() { swamp_gpu::gpu_free_weights(p).ok(); } }
        for &p in &self.d_v_weight { if !p.is_null() { swamp_gpu::gpu_free_weights(p).ok(); } }
        for &p in &self.d_o_weight { if !p.is_null() { swamp_gpu::gpu_free_weights(p).ok(); } }
        for &p in &self.d_gate_weight { if !p.is_null() { swamp_gpu::gpu_free_weights(p).ok(); } }
        for &p in &self.d_up_weight { if !p.is_null() { swamp_gpu::gpu_free_weights(p).ok(); } }
        for &p in &self.d_down_weight { if !p.is_null() { swamp_gpu::gpu_free_weights(p).ok(); } }
        for g in self.graph_qkv.drain(..) { if let Some(g2) = g { let _ = swamp_gpu::gpu_graph_destroy_gemv(g2); } }
        for g in self.graph_gate_up.drain(..) { if let Some(g2) = g { let _ = swamp_gpu::gpu_graph_destroy_gemv(g2); } }
        for g in self.graph_o.drain(..) { if let Some(g2) = g { let _ = swamp_gpu::gpu_graph_destroy_gemv(g2); } }
        for g in self.graph_down.drain(..) { if let Some(g2) = g { let _ = swamp_gpu::gpu_graph_destroy_gemv(g2); } }
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
