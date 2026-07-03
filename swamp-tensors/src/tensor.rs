// swamp-tensors/src/tensor.rs
// CpuTensor: tensor carregado na CPU, contendo buffer f32 e layout

use crate::layout::Layout;
use crate::arena::NumaBuffer;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum TensorData {
    Heap(Vec<f32>),
    Numa(NumaBuffer),
}

impl std::ops::Deref for TensorData {
    type Target = [f32];
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Heap(v) => v,
            Self::Numa(n) => n,
        }
    }
}

impl std::ops::DerefMut for TensorData {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Heap(v) => v,
            Self::Numa(n) => n,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CpuTensor {
    layout: Layout,
    data: Arc<TensorData>,
}

impl CpuTensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        let layout = Layout::new(shape);
        assert_eq!(layout.size(), data.len(), "Tamanho dos dados nao condiz com o shape");
        Self {
            layout,
            data: Arc::new(TensorData::Heap(data)),
        }
    }

    pub fn new_numa(shape: Vec<usize>, node: usize) -> Self {
        let layout = Layout::new(shape);
        let size = layout.size();
        let buf = NumaBuffer::new(size, node);
        Self {
            layout,
            data: Arc::new(TensorData::Numa(buf)),
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self::new(shape, vec![0.0f32; size])
    }

    pub fn zeros_numa(shape: Vec<usize>, node: usize) -> Self {
        Self::new_numa(shape, node)
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    pub fn strides(&self) -> &[usize] {
        self.layout.strides()
    }

    pub fn size(&self) -> usize {
        self.layout.size()
    }

    /// Retorna view slice dos dados do tensor
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Retorna copia mutavel (caso seja o unico dono do Arc, evita alocacao)
    pub fn data_mut(&mut self) -> &mut [f32] {
        let data = Arc::make_mut(&mut self.data);
        match data {
            TensorData::Heap(v) => v.as_mut_slice(),
            TensorData::Numa(n) => n.as_slice_mut(),
        }
    }

    pub fn get(&self, indices: &[usize]) -> f32 {
        let idx = self.layout.offset_for(indices);
        self.data[idx]
    }

    pub fn set(&mut self, indices: &[usize], val: f32) {
        let idx = self.layout.offset_for(indices);
        let data = Arc::make_mut(&mut self.data);
        data[idx] = val;
    }
}
