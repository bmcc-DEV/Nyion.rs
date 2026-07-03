// swamp-tensors/src/storage.rs
// Prefetch de tensores usando chamadas de sistema assíncronas do Linux (readahead + fadvise)

use std::path::Path;
use std::os::unix::io::AsRawFd;
use std::fs::File;

pub struct LscPrefetcher {}

impl LscPrefetcher {
    pub fn new() -> Self {
        Self {}
    }

    /// Agenda o prefetch de uma faixa de bytes do arquivo do modelo de forma assíncrona.
    /// O kernel do Linux carregará as páginas do SSD NVMe para a RAM em background.
    pub fn prefetch_range(&self, path: &Path, offset: usize, len: usize) {
        if let Ok(file) = File::open(path) {
            let fd = file.as_raw_fd();
            unsafe {
                // 1. Avisa ao subsistema de memória virtual que usaremos esta faixa em breve
                libc::posix_fadvise(
                    fd,
                    offset as libc::off_t,
                    len as libc::off_t,
                    libc::POSIX_FADV_WILLNEED,
                );

                // 2. Aciona o readahead assíncrono do kernel do Linux para preencher o Page Cache
                libc::readahead(
                    fd,
                    offset as libc::off_t,
                    len,
                );
            }
        }
    }
}

impl Default for LscPrefetcher {
    fn default() -> Self {
        Self::new()
    }
}
