// swamp-engine/src/cache.rs
// PagedKVCache: KV cache paginado com alocacao sob demanda para contexto > 2K tokens

use std::alloc::{alloc_zeroed, dealloc, Layout};

const DEFAULT_BLOCK_SIZE: usize = 32;

pub struct PagedKVCache {
    k_pages: Vec<*mut f32>,
    v_pages: Vec<*mut f32>,
    page_capacities: Vec<(Layout, usize)>, // (layout, byte_size)
    num_layers: usize,
    num_kv_heads: usize,
    pub max_seq_len: usize,
    pub block_size: usize,
    head_dim: usize,
    current_pos: usize,
}

unsafe impl Send for PagedKVCache {}
unsafe impl Sync for PagedKVCache {}

impl PagedKVCache {
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
    ) -> Self {
        Self::with_block_size(num_layers, num_kv_heads, max_seq_len, head_dim, DEFAULT_BLOCK_SIZE)
    }

    pub fn with_block_size(
        num_layers: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        block_size: usize,
    ) -> Self {
        let num_pages = (max_seq_len + block_size - 1) / block_size;
        Self {
            k_pages: Vec::with_capacity(num_pages),
            v_pages: Vec::with_capacity(num_pages),
            page_capacities: Vec::with_capacity(num_pages),
            num_layers,
            num_kv_heads,
            max_seq_len,
            block_size,
            head_dim,
            current_pos: 0,
        }
    }

    fn page_byte_size(&self) -> (Layout, usize) {
        let count = self.num_layers * self.num_kv_heads * self.block_size * self.head_dim;
        let byte_size = count * std::mem::size_of::<f32>();
        let layout = Layout::from_size_align(byte_size, 64).unwrap();
        (layout, byte_size)
    }

    fn ensure_page(&mut self, page_id: usize) {
        while page_id >= self.k_pages.len() {
            let (layout, byte_size) = self.page_byte_size();
            let k_ptr = unsafe { alloc_zeroed(layout) } as *mut f32;
            let v_ptr = unsafe { alloc_zeroed(layout) } as *mut f32;
            if k_ptr.is_null() || v_ptr.is_null() {
                panic!("PagedKVCache OOM at page {}", self.k_pages.len());
            }
            self.k_pages.push(k_ptr);
            self.v_pages.push(v_ptr);
            self.page_capacities.push((layout, byte_size));
        }
    }

    #[inline(always)]
    pub fn page_id(&self, pos: usize) -> usize {
        pos / self.block_size
    }

    #[inline(always)]
    pub fn slot_in_page(&self, pos: usize) -> usize {
        pos % self.block_size
    }

    #[inline(always)]
    pub fn offset_in_page(&self, layer: usize, kv_head: usize, slot: usize) -> usize {
        ((layer * self.num_kv_heads + kv_head) * self.block_size + slot) * self.head_dim
    }

    pub fn save(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        let pos = self.current_pos;
        let pid = self.page_id(pos);
        let slot = self.slot_in_page(pos);
        self.ensure_page(pid);

        for kv_head in 0..self.num_kv_heads {
            let off = self.offset_in_page(layer, kv_head, slot);
            let src_off = kv_head * self.head_dim;

            unsafe {
                std::ptr::copy_nonoverlapping(
                    k.as_ptr().add(src_off),
                    self.k_pages[pid].add(off),
                    self.head_dim,
                );
                std::ptr::copy_nonoverlapping(
                    v.as_ptr().add(src_off),
                    self.v_pages[pid].add(off),
                    self.head_dim,
                );
            }
        }
    }

    #[inline(always)]
    pub unsafe fn get_k_unchecked(&self, layer: usize, kv_head: usize, pos: usize) -> &[f32] {
        let pid = self.page_id(pos);
        let slot = self.slot_in_page(pos);
        let off = self.offset_in_page(layer, kv_head, slot);
        std::slice::from_raw_parts(self.k_pages[pid].add(off), self.head_dim)
    }

    #[inline(always)]
    pub unsafe fn get_v_unchecked(&self, layer: usize, kv_head: usize, pos: usize) -> &[f32] {
        let pid = self.page_id(pos);
        let slot = self.slot_in_page(pos);
        let off = self.offset_in_page(layer, kv_head, slot);
        std::slice::from_raw_parts(self.v_pages[pid].add(off), self.head_dim)
    }

    /// Retorna ponteiro raw para o inicio de K de (layer, kv_head) em uma page
    #[inline(always)]
    pub fn k_page_ptr(&self, layer: usize, kv_head: usize, page_id: usize) -> *const f32 {
        let off = self.offset_in_page(layer, kv_head, 0);
        unsafe { self.k_pages[page_id].add(off) as *const f32 }
    }

    /// Retorna ponteiro raw para o inicio de V de (layer, kv_head) em uma page
    #[inline(always)]
    pub fn v_page_ptr(&self, layer: usize, kv_head: usize, page_id: usize) -> *const f32 {
        let off = self.offset_in_page(layer, kv_head, 0);
        unsafe { self.v_pages[page_id].add(off) as *const f32 }
    }

    /// Save K/V at a specific position (for batched prefill)
    pub fn save_at(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        let pid = self.page_id(pos);
        let slot = self.slot_in_page(pos);
        self.ensure_page(pid);

        for kv_head in 0..self.num_kv_heads {
            let off = self.offset_in_page(layer, kv_head, slot);
            let src_off = kv_head * self.head_dim;

            unsafe {
                std::ptr::copy_nonoverlapping(
                    k.as_ptr().add(src_off),
                    self.k_pages[pid].add(off),
                    self.head_dim,
                );
                std::ptr::copy_nonoverlapping(
                    v.as_ptr().add(src_off),
                    self.v_pages[pid].add(off),
                    self.head_dim,
                );
            }
        }
    }

    pub fn advance(&mut self) {
        self.current_pos += 1;
        // Pages sao alocadas sob demanda, entao nao ha limite pratico
    }

    pub fn current_pos(&self) -> usize {
        self.current_pos
    }

    pub fn reset(&mut self) {
        self.current_pos = 0;
    }
}

impl Drop for PagedKVCache {
    fn drop(&mut self) {
        for i in 0..self.k_pages.len() {
            let (layout, _) = self.page_capacities[i];
            unsafe {
                dealloc(self.k_pages[i] as *mut u8, layout);
                dealloc(self.v_pages[i] as *mut u8, layout);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paged_cache_basic_rw() {
        let mut cache = PagedKVCache::with_block_size(2, 4, 128, 64, 32);
        let n_layers = 2;
        let n_kv = 4;
        let head_dim = 64;

        // Simula 3 tokens: salva K e V, verifica leitura
        for pos in 0..3 {
            let mut k = vec![0.0f32; n_kv * head_dim];
            let mut v = vec![0.0f32; n_kv * head_dim];
            for h in 0..n_kv {
                k[h * head_dim] = (pos * 100 + h) as f32;
                v[h * head_dim] = (pos * 1000 + h) as f32;
            }
            for l in 0..n_layers {
                cache.save(l, &k, &v);
            }
            cache.advance();
        }

        assert_eq!(cache.current_pos(), 3);

        // Verifica leitura
        for l in 0..n_layers {
            for h in 0..n_kv {
                for pos in 0..3 {
                    let k_slice = unsafe { cache.get_k_unchecked(l, h, pos) };
                    let v_slice = unsafe { cache.get_v_unchecked(l, h, pos) };
                    assert_eq!(k_slice[0], (pos * 100 + h) as f32,
                        "K mismatch layer={} head={} pos={}", l, h, pos);
                    assert_eq!(v_slice[0], (pos * 1000 + h) as f32,
                        "V mismatch layer={} head={} pos={}", l, h, pos);
                }
            }
        }
    }

    #[test]
    fn test_paged_cache_multiple_pages() {
        let block_size = 8;
        let mut cache = PagedKVCache::with_block_size(1, 1, 1024, 64, block_size);
        let n_tokens = 32; // 4 pages

        for pos in 0..n_tokens {
            let k = vec![pos as f32; 64];
            let v = vec![(pos * 2) as f32; 64];
            cache.save(0, &k, &v);
            cache.advance();
        }

        assert_eq!(cache.current_pos(), n_tokens);
        assert!(cache.k_pages.len() >= n_tokens / block_size);

        // Leitura spot-check entre pages
        for pos in 0..n_tokens {
            let k = unsafe { cache.get_k_unchecked(0, 0, pos) };
            let v = unsafe { cache.get_v_unchecked(0, 0, pos) };
            assert_eq!(k[0], pos as f32, "page={} slot={}", pos / block_size, pos % block_size);
            assert_eq!(v[0], (pos * 2) as f32);
        }

        // Verifica page_ptr
        for pid in 0..(n_tokens / block_size) {
            let k_ptr = cache.k_page_ptr(0, 0, pid);
            let v_ptr = cache.v_page_ptr(0, 0, pid);
            unsafe {
                assert_eq!(*k_ptr, (pid * block_size) as f32);
                assert_eq!(*v_ptr, (pid * block_size * 2) as f32);
            }
        }
    }

    #[test]
    fn test_paged_cache_reset() {
        let mut cache = PagedKVCache::with_block_size(1, 1, 1024, 64, 8);
        for pos in 0..10 {
            let k = vec![pos as f32; 64];
            let v = vec![pos as f32; 64];
            cache.save(0, &k, &v);
            cache.advance();
        }

        cache.reset();
        assert_eq!(cache.current_pos(), 0);

        // Re-escreve e verifica
        let k = vec![42.0f32; 64];
        let v = vec![42.0f32; 64];
        cache.save(0, &k, &v);
        cache.advance();

        let k_read = unsafe { cache.get_k_unchecked(0, 0, 0) };
        assert_eq!(k_read[0], 42.0);
    }
}
