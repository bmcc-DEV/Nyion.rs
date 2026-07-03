pub struct PrefetchEngine {
    mmap_ptr: *mut u8,
    mmap_len: usize,
}

unsafe impl Send for PrefetchEngine {}
unsafe impl Sync for PrefetchEngine {}

impl PrefetchEngine {
    pub fn new(mmap_ptr: *mut u8, mmap_len: usize) -> Self {
        Self { mmap_ptr, mmap_len }
    }

    pub fn prefetch_range(&self, offset: usize, len: usize) {
        if offset + len > self.mmap_len {
            return;
        }
        unsafe {
            let addr = self.mmap_ptr.add(offset) as *mut libc::c_void;
            libc::madvise(addr, len, libc::MADV_WILLNEED);
        }
    }

}
