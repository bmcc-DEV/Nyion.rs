// swamp-engine/src/cache.rs
// TieredPagedKVCache: KV cache com hierarquia de 3 niveis
//   Hot  (RAM)  — acesso rapido, LRU-bound
//   Cold (NVMe) — pagina serializada em arquivo temporario
//
// Uso: chame ensure_pages_hot() antes de acessar via k_page_ptr/v_page_ptr em paralelo.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fs::{File, OpenOptions};
use std::io::{Write, Read, Seek, SeekFrom};
use std::path::Path;

const DEFAULT_BLOCK_SIZE: usize = 32;

#[derive(Clone, Copy, PartialEq, Eq)]
enum PageState {
    Hot,
    Cold(u64),
}

pub struct PagedKVCache {
    // Page storage — k_pages[i] / v_pages[i] podem ser null se a pagina foi evictada
    k_pages: Vec<*mut f32>,
    v_pages: Vec<*mut f32>,
    page_capacities: Vec<(Layout, usize)>,
    page_state: Vec<PageState>,

    // LRU tracking (apenas paginas Hot estao na lista)
    lru_prev: Vec<Option<usize>>,
    lru_next: Vec<Option<usize>>,
    lru_head: Option<usize>,
    lru_tail: Option<usize>,
    hot_count: usize,

    // Cold storage (NVMe)
    cold_file: Option<File>,
    cold_dir: String,

    // Config
    ram_page_limit: usize,

    // Fixed params
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
        let page_size = num_layers * num_kv_heads * block_size * head_dim * std::mem::size_of::<f32>();

        let ram_page_limit = std::env::var("SWAMP_CACHE_RAM_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|mb| (mb * 1024 * 1024).max(page_size) / page_size)
            .unwrap_or(usize::MAX);

        let cold_dir = std::env::var("SWAMP_CACHE_COLD_DIR")
            .unwrap_or_else(|_| "/tmp/swamp_cache".to_string());

        Self {
            k_pages: Vec::with_capacity(num_pages),
            v_pages: Vec::with_capacity(num_pages),
            page_capacities: Vec::with_capacity(num_pages),
            page_state: Vec::with_capacity(num_pages),
            lru_prev: Vec::with_capacity(num_pages),
            lru_next: Vec::with_capacity(num_pages),
            lru_head: None,
            lru_tail: None,
            hot_count: 0,
            cold_file: None,
            cold_dir,
            ram_page_limit,
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

    fn init_cold_file(&mut self) {
        if self.cold_file.is_some() {
            return;
        }
        let path = Path::new(&self.cold_dir);
        let _ = std::fs::create_dir_all(path);
        let cold_path = path.join(format!("kv_cache_{}.bin", std::process::id()));
        self.cold_file = Some(
            OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .open(&cold_path)
                .expect("Failed to create KV cold storage file"),
        );
    }

    /// Alocar uma nova pagina Hot, adicionando-a ao LRU head.
    fn alloc_hot_page(&mut self) -> usize {
        let (layout, byte_size) = self.page_byte_size();
        let k_ptr = unsafe { alloc_zeroed(layout) } as *mut f32;
        let v_ptr = unsafe { alloc_zeroed(layout) } as *mut f32;
        if k_ptr.is_null() || v_ptr.is_null() {
            panic!("PagedKVCache OOM at page {}", self.k_pages.len());
        }

        let page_id = self.k_pages.len();
        self.k_pages.push(k_ptr);
        self.v_pages.push(v_ptr);
        self.page_capacities.push((layout, byte_size));
        self.page_state.push(PageState::Hot);
        self.lru_prev.push(None);
        self.lru_next.push(None);
        self.hot_count += 1;

        // Inserir no LRU head
        self.lru_next[page_id] = self.lru_head;
        if let Some(h) = self.lru_head {
            self.lru_prev[h] = Some(page_id);
        }
        self.lru_head = Some(page_id);
        if self.lru_tail.is_none() {
            self.lru_tail = Some(page_id);
        }

        page_id
    }

    /// Mover uma pagina Hot para o LRU head (mais recentemente usado).
    fn touch_lru(&mut self, page_id: usize) {
        if self.lru_head == Some(page_id) {
            return;
        }
        let prev = self.lru_prev[page_id];
        let next = self.lru_next[page_id];

        if let Some(p) = prev {
            self.lru_next[p] = next;
        }
        if let Some(n) = next {
            self.lru_prev[n] = prev;
        }
        if self.lru_tail == Some(page_id) {
            self.lru_tail = prev;
        }

        self.lru_prev[page_id] = None;
        self.lru_next[page_id] = self.lru_head;
        if let Some(h) = self.lru_head {
            self.lru_prev[h] = Some(page_id);
        }
        self.lru_head = Some(page_id);
        if self.lru_tail.is_none() {
            self.lru_tail = Some(page_id);
        }
    }

    /// Remover uma pagina do LRU (usado apos evictar).
    fn remove_from_lru(&mut self, page_id: usize) {
        let prev = self.lru_prev[page_id];
        let next = self.lru_next[page_id];
        if let Some(p) = prev {
            self.lru_next[p] = next;
        }
        if let Some(n) = next {
            self.lru_prev[n] = prev;
        }
        if self.lru_head == Some(page_id) {
            self.lru_head = next;
        }
        if self.lru_tail == Some(page_id) {
            self.lru_tail = prev;
        }
        self.lru_prev[page_id] = None;
        self.lru_next[page_id] = None;
    }

    /// Evictar a pagina menos recentemente usada para cold storage (NVMe).
    /// A pagina fica marcada como Cold, seu ponteiro RAM e liberado.
    fn evict_one_page(&mut self) {
        let evict = match self.lru_tail {
            Some(id) => id,
            None => return,
        };

        self.init_cold_file();

        if let Some(ref mut file) = self.cold_file {
            let (_, byte_size) = self.page_capacities[evict];
            let offset = file.seek(SeekFrom::End(0)).expect("cold store seek failed");

            unsafe {
                let k_slice = std::slice::from_raw_parts(self.k_pages[evict] as *const u8, byte_size);
                let v_slice = std::slice::from_raw_parts(self.v_pages[evict] as *const u8, byte_size);
                file.write_all(k_slice).expect("cold store write failed");
                file.write_all(v_slice).expect("cold store write failed");
            }

            self.page_state[evict] = PageState::Cold(offset);
            self.hot_count -= 1;

            self.remove_from_lru(evict);

            unsafe {
                dealloc(self.k_pages[evict] as *mut u8, self.page_capacities[evict].0);
                dealloc(self.v_pages[evict] as *mut u8, self.page_capacities[evict].0);
            }
            self.k_pages[evict] = std::ptr::null_mut();
            self.v_pages[evict] = std::ptr::null_mut();
        }
    }

    /// Recarregar uma pagina Cold da NVMe de volta para RAM.
    fn reload_cold_page(&mut self, page_id: usize) {
        let offset = match self.page_state[page_id] {
            PageState::Cold(off) => off,
            _ => return,
        };

        if self.hot_count >= self.ram_page_limit {
            self.evict_one_page();
        }

        // Re-alocar se foi desalocado
        if self.k_pages[page_id].is_null() {
            let (layout, byte_size) = self.page_byte_size();
            let k_ptr = unsafe { alloc_zeroed(layout) } as *mut f32;
            let v_ptr = unsafe { alloc_zeroed(layout) } as *mut f32;
            if k_ptr.is_null() || v_ptr.is_null() {
                panic!("PagedKVCache OOM on cold reload page {}", page_id);
            }
            self.k_pages[page_id] = k_ptr;
            self.v_pages[page_id] = v_ptr;
            self.page_capacities[page_id] = (layout, byte_size);
        }

        let (_, byte_size) = self.page_byte_size();
        if let Some(ref mut file) = self.cold_file {
            let mut buf = vec![0u8; byte_size * 2];
            file.seek(SeekFrom::Start(offset)).expect("cold store seek failed");
            file.read_exact(&mut buf).expect("cold store read failed");

            unsafe {
                std::ptr::copy_nonoverlapping(
                    buf.as_ptr() as *const f32,
                    self.k_pages[page_id],
                    byte_size / 4,
                );
                std::ptr::copy_nonoverlapping(
                    buf.as_ptr().add(byte_size) as *const f32,
                    self.v_pages[page_id],
                    byte_size / 4,
                );
            }
        }

        self.page_state[page_id] = PageState::Hot;
        self.hot_count += 1;

        // Inserir no LRU head
        self.lru_prev[page_id] = None;
        self.lru_next[page_id] = self.lru_head;
        if let Some(h) = self.lru_head {
            self.lru_prev[h] = Some(page_id);
        }
        self.lru_head = Some(page_id);
        if self.lru_tail.is_none() {
            self.lru_tail = Some(page_id);
        }
    }

    /// Garantir que uma pagina especifica esteja Hot (em RAM).
    /// Se estiver Cold, recarrega da NVMe.
    /// Se nunca foi alocada, aloca do zero.
    /// Evicta paginas frias se exceder o limite de RAM.
    fn ensure_page(&mut self, page_id: usize) {
        while page_id >= self.k_pages.len() {
            if self.hot_count >= self.ram_page_limit {
                self.evict_one_page();
            }
            self.alloc_hot_page();
        }

        match self.page_state[page_id] {
            PageState::Cold(_) => {
                self.reload_cold_page(page_id);
            }
            PageState::Hot => {}
        }

        self.touch_lru(page_id);
    }

    /// Garantir que todas as paginas em um intervalo estejam Hot.
    /// Deve ser chamado antes de entrar em codigo paralelo que usa `k_page_ptr`/`v_page_ptr`.
    pub fn ensure_pages_hot(&mut self, start_page: usize, end_page: usize) {
        for pid in start_page..=end_page {
            self.ensure_page(pid);
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

    /// Retorna slice para K — PANIC se a pagina estiver Cold.
    /// Use `ensure_pages_hot` antes de chamar esta funcao.
    #[inline(always)]
    pub unsafe fn get_k_unchecked(&self, layer: usize, kv_head: usize, pos: usize) -> &[f32] {
        let pid = self.page_id(pos);
        assert!(self.k_pages[pid] != std::ptr::null_mut(), "get_k_unchecked on cold page {}", pid);
        let slot = self.slot_in_page(pos);
        let off = self.offset_in_page(layer, kv_head, slot);
        std::slice::from_raw_parts(self.k_pages[pid].add(off), self.head_dim)
    }

    /// Retorna slice para V — PANIC se a pagina estiver Cold.
    #[inline(always)]
    pub unsafe fn get_v_unchecked(&self, layer: usize, kv_head: usize, pos: usize) -> &[f32] {
        let pid = self.page_id(pos);
        assert!(self.k_pages[pid] != std::ptr::null_mut(), "get_v_unchecked on cold page {}", pid);
        let slot = self.slot_in_page(pos);
        let off = self.offset_in_page(layer, kv_head, slot);
        std::slice::from_raw_parts(self.v_pages[pid].add(off), self.head_dim)
    }

    /// Retorna ponteiro raw para K — PANIC se a pagina estiver Cold.
    /// Use `ensure_pages_hot` antes de entrar em codigo paralelo.
    #[inline(always)]
    pub fn k_page_ptr(&self, layer: usize, kv_head: usize, page_id: usize) -> *const f32 {
        assert!(self.k_pages[page_id] != std::ptr::null_mut(), "k_page_ptr on cold page {}", page_id);
        let off = self.offset_in_page(layer, kv_head, 0);
        unsafe { self.k_pages[page_id].add(off) as *const f32 }
    }

    /// Retorna ponteiro raw para V — PANIC se a pagina estiver Cold.
    #[inline(always)]
    pub fn v_page_ptr(&self, layer: usize, kv_head: usize, page_id: usize) -> *const f32 {
        assert!(self.k_pages[page_id] != std::ptr::null_mut(), "v_page_ptr on cold page {}", page_id);
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
            if !self.k_pages[i].is_null() {
                let (layout, _) = self.page_capacities[i];
                unsafe {
                    dealloc(self.k_pages[i] as *mut u8, layout);
                    dealloc(self.v_pages[i] as *mut u8, layout);
                }
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
        let n_tokens = 32;

        for pos in 0..n_tokens {
            let k = vec![pos as f32; 64];
            let v = vec![(pos * 2) as f32; 64];
            cache.save(0, &k, &v);
            cache.advance();
        }

        assert_eq!(cache.current_pos(), n_tokens);
        assert!(cache.k_pages.len() >= n_tokens / block_size);

        for pos in 0..n_tokens {
            let k = unsafe { cache.get_k_unchecked(0, 0, pos) };
            let v = unsafe { cache.get_v_unchecked(0, 0, pos) };
            assert_eq!(k[0], pos as f32, "page={} slot={}", pos / block_size, pos % block_size);
            assert_eq!(v[0], (pos * 2) as f32);
        }

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

        let k = vec![42.0f32; 64];
        let v = vec![42.0f32; 64];
        cache.save(0, &k, &v);
        cache.advance();

        let k_read = unsafe { cache.get_k_unchecked(0, 0, 0) };
        assert_eq!(k_read[0], 42.0);
    }

    #[test]
    fn test_tiered_eviction_basic() {
        // Force small RAM limit: 2 pages max
        std::env::set_var("SWAMP_CACHE_RAM_MB", "0");
        let block_size = 4;
        let mut cache = PagedKVCache::with_block_size(1, 1, 128, 64, block_size);
        // Override ram_page_limit for test (0 MB = 0 pages, which would be problematic)
        // Instead, set it directly
        cache.ram_page_limit = 2;

        let n_tokens = 12; // 3 pages (block_size=4)

        for pos in 0..n_tokens {
            let k = vec![pos as f32; 64];
            let v = vec![(pos * 2) as f32; 64];
            cache.save(0, &k, &v);
            cache.advance();
        }

        assert_eq!(cache.current_pos(), n_tokens);

        // Pages 0 and 1 should have been evicted (cold) since we only keep 2 hot
        // Touch pages 2, 3 (most recent) to keep them hot — but save touched them all via LRU
        // The last 2 pages saved (pages 2 and then... wait, page_id = pos / 4)
        // page 0: tokens 0-3, page 1: tokens 4-7, page 2: tokens 8-11
        // Page 2 was touched last (positions 8-11), so it should be hot
        // Between pages 0 and 1, page 1 was touched more recently (positions 4-7 vs 0-3)
        // So page 0 should be evicted first

        // Ensure page 2 is hot before accessing
        cache.ensure_pages_hot(2, 2);
        let k2 = unsafe { cache.get_k_unchecked(0, 0, 8) };
        assert_eq!(k2[0], 8.0, "Page 2 (most recent) should be accessible");

        // Page 0 may be cold — ensure it first
        cache.ensure_pages_hot(0, 0);
        let k0 = unsafe { cache.get_k_unchecked(0, 0, 0) };
        assert_eq!(k0[0], 0.0, "Page 0 should be reloadable from cold storage");

        // Clean up env var
        std::env::remove_var("SWAMP_CACHE_RAM_MB");
    }

    #[test]
    fn test_tiered_no_eviction_within_limit() {
        let block_size = 4;
        let mut cache = PagedKVCache::with_block_size(1, 1, 128, 64, block_size);
        cache.ram_page_limit = 10; // Well above what we'll use

        let n_tokens = 8; // 2 pages
        for pos in 0..n_tokens {
            let k = vec![pos as f32; 64];
            let v = vec![(pos * 2) as f32; 64];
            cache.save(0, &k, &v);
            cache.advance();
        }

        // All pages should still be hot
        for pos in 0..n_tokens {
            let k = unsafe { cache.get_k_unchecked(0, 0, pos) };
            assert_eq!(k[0], pos as f32);
        }
    }
}
