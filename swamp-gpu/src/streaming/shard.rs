use std::path::Path;
use std::fs;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct TensorInfo {
    pub offset: u64,
    pub size: u64,
    pub n_blocks: u32,
}

#[derive(Debug, Deserialize)]
pub struct LayerInfo {
    pub id: u32,
    pub name: String,
    pub tensors: std::collections::HashMap<String, TensorInfo>,
}

#[derive(Debug, Deserialize)]
pub struct ShardIndex {
    pub format: String,
    pub model: String,
    pub quant: String,
    pub num_layers: u32,
    pub layers: Vec<LayerInfo>,
}

impl ShardIndex {
    pub fn from_path(path: &Path) -> std::io::Result<Self> {
        let content = fs::read_to_string(path)?;
        serde_json::from_str(&content).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn layer_offsets(&self, layer_id: u32, tensor_name: &str) -> Option<&TensorInfo> {
        self.layers
            .iter()
            .find(|l| l.id == layer_id)
            .and_then(|l| l.tensors.get(tensor_name))
    }
}

pub struct ShardReader {
    pub index: ShardIndex,
    data: memmap2::Mmap,
}

impl ShardReader {
    pub fn open<P: AsRef<Path>>(index_path: P, data_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let index = ShardIndex::from_path(index_path.as_ref())?;
        let file = fs::File::open(data_path.as_ref())?;
        let data = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Self { index, data })
    }

    pub fn get_tensor(&self, layer_id: u32, tensor_name: &str) -> Option<&[u8]> {
        let info = self.index.layer_offsets(layer_id, tensor_name)?;
        let start = info.offset as usize;
        let end = start + info.size as usize;
        if end > self.data.len() { return None; }
        Some(&self.data[start..end])
    }

    pub fn total_size(&self) -> usize {
        self.data.len()
    }
}
