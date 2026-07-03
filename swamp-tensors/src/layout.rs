// swamp-tensors/src/layout.rs
// Layout, shape, strides e offset para tensores n-dimensionais

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl Layout {
    pub fn new(shape: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&shape);
        Self { shape, strides, offset: 0 }
    }

    pub fn with_strides(shape: Vec<usize>, strides: Vec<usize>) -> Self {
        Self { shape, strides, offset: 0 }
    }

    pub fn with_offset(shape: Vec<usize>, strides: Vec<usize>, offset: usize) -> Self {
        Self { shape, strides, offset }
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        let mut stride = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }
        strides
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Calcula o offset linear para um conjunto de indices
    pub fn offset_for(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len(), "Dimensao de indices incorreta");
        let mut index = self.offset;
        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < self.shape[i], "Index {} fora dos limites para dimensao {}", idx, i);
            index += idx * self.strides[i];
        }
        index
    }

    /// Verifica se o layout e contiguo na memoria
    pub fn is_contiguous(&self) -> bool {
        let mut expected_stride = 1;
        for i in (0..self.shape.len()).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i];
        }
        true
    }
}
