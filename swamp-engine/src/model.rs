// swamp-engine/src/model.rs
// Model: carregador de tensores e metadados de GGUF do modelo

use swamp_gguf::GgufFile;
use std::path::Path;
use anyhow::{Result, Context};

pub struct ModelConfig {
    pub architecture: String,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub embed_dim: usize,
    pub context_len: usize,
    pub vocab_size: usize,
    pub rope_freq_base: f32,
    pub rope_dim: usize,
}

pub struct Model {
    pub gguf: GgufFile,
    pub config: ModelConfig,
}

impl Model {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let gguf = GgufFile::open(path.as_ref())
            .with_context(|| format!("Falha ao abrir o GGUF em: {:?}", path.as_ref()))?;

        let md = &gguf.metadata;

        // Extrai configuracoes usando o compat automatizado
        let architecture = gguf.architecture().to_string();
        let num_layers = md.n_layer()? as usize;
        let num_heads = md.n_head()? as usize;
        let num_kv_heads = md.n_kv_head() as usize;
        let embed_dim = md.n_embd()? as usize;
        let context_len = md.n_ctx_train() as usize;
        let rope_freq_base = md.rope_freq_base();
        let rope_dim = md.rope_dimension() as usize;

        // Obtem vocab_size do array de tokens
        let vocab_size = md.get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_string_array())
            .map(|arr| arr.len())
            .unwrap_or(32000); // fallback padrao

        let config = ModelConfig {
            architecture,
            num_layers,
            num_heads,
            num_kv_heads,
            embed_dim,
            context_len,
            vocab_size,
            rope_freq_base,
            rope_dim,
        };

        Ok(Self { gguf, config })
    }

    pub fn print_info(&self) {
        println!("=== Configurações do Model ===");
        println!("  Arquitetura:   {}", self.config.architecture);
        println!("  Camadas:       {}", self.config.num_layers);
        println!("  Embedding:     {}", self.config.embed_dim);
        println!("  Heads:         {} (KV: {})", self.config.num_heads, self.config.num_kv_heads);
        println!("  Contexto:      {}", self.config.context_len);
        println!("  RoPE freq:     {}", self.config.rope_freq_base);
        println!("  RoPE dim:      {}", self.config.rope_dim);
        println!("  Vocab size:    {}", self.config.vocab_size);
    }
}
