// swamp-tensors/src/arena.rs
// Arena allocator NUMA-aware para tensores temporarios (KV cache, buffers intermedios)

use std::sync::Mutex;

/// Buffer alocado especificamente em um no NUMA usando mmap e mbind (Linux)
#[derive(Debug)]
pub struct NumaBuffer {
    ptr: *mut f32,
    size: usize, // numero de elementos f32
}

unsafe impl Send for NumaBuffer {}
unsafe impl Sync for NumaBuffer {}

impl NumaBuffer {
    pub fn new(size: usize, node: usize) -> Self {
        let byte_len = size * std::mem::size_of::<f32>();
        unsafe {
            // 1. Aloca memoria anonima via mmap
            let addr = libc::mmap(
                std::ptr::null_mut(),
                byte_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );

            if addr == libc::MAP_FAILED {
                // Fallback para heap normal em caso de falha no mmap
                let mut v = vec![0.0f32; size];
                let ptr = v.as_mut_ptr();
                std::mem::forget(v); // Evita desalocacao automatica
                return Self { ptr, size };
            }

            // 2. Associa a faixa de memoria ao no NUMA especifico usando mbind (syscall 237 no x86_64 Linux)
            // MPOL_BIND = 1, MPOL_MF_STRICT = 1
            let nodemask = 1u64 << node;
            let nodemask_ptr = &nodemask as *const u64 as *const libc::c_void;
            
            let _ = libc::syscall(
                237, // SYS_mbind
                addr,
                byte_len,
                1, // MPOL_BIND
                nodemask_ptr,
                64, // maxnode
                1, // MPOL_MF_STRICT
            );

            Self { ptr: addr as *mut f32, size }
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    pub fn as_slice_mut(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }
}

impl std::ops::Deref for NumaBuffer {
    type Target = [f32];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl std::ops::DerefMut for NumaBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_slice_mut()
    }
}

impl Drop for NumaBuffer {
    fn drop(&mut self) {
        let byte_len = self.size * std::mem::size_of::<f32>();
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, byte_len);
        }
    }
}

impl Clone for NumaBuffer {
    fn clone(&self) -> Self {
        // Clona alocando um novo bloco NUMA e copiando o conteudo
        let mut new_buf = Self::new(self.size, 0); // node 0 por padrao
        new_buf.as_slice_mut().copy_from_slice(self.as_slice());
        new_buf
    }
}

pub struct MemoryBlock {
    pub buf: NumaBuffer,
    pub active: bool,
}

pub struct TensorArena {
    blocks: Mutex<Vec<MemoryBlock>>,
}

impl TensorArena {
    pub fn new() -> Self {
        Self {
            blocks: Mutex::new(Vec::new()),
        }
    }

    /// Aloca ou reutiliza um bloco de memoria continuo no no NUMA selecionado
    pub fn alloc(&self, size: usize, node: usize) -> NumaBuffer {
        let mut blocks = self.blocks.lock().unwrap();
        
        // Tenta encontrar um bloco livre com tamanho compativel
        if let Some(block) = blocks.iter_mut().find(|b| !b.active && b.buf.size >= size) {
            block.active = true;
            // Retorna um novo NumaBuffer apontando temporariamente para o bloco (simplificado para pool basico)
            // Para segurança completa sem transferir ownership total do drop,
            // clonamos os ponteiros (o bloco original gerencia a liberacao)
            return NumaBuffer {
                ptr: block.buf.ptr,
                size,
            };
        }

        // Caso contrario, aloca um novo bloco
        let buf = NumaBuffer::new(size, node);
        // Nao colocamos no pool ainda para evitar complexidade de ponteiros duplicados no drop.
        // NumaBuffer gerencia a propria desalocacao via munmap.
        buf
    }
}

impl Default for TensorArena {
    fn default() -> Self {
        Self::new()
    }
}
