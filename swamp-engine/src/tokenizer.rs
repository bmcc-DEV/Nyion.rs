use std::sync::OnceLock;
use tokenizers::Tokenizer;

pub static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();

pub fn init_tokenizer(model_path: &str) {
    TOKENIZER.get_or_init(|| {
        Tokenizer::from_file(model_path).expect("Failed to load tokenizer from json")
    });
}

pub fn encode(text: &str) -> Vec<usize> {
    let sp = TOKENIZER.get().expect("Tokenizer not initialized");
    let encoding = sp.encode(text, false).unwrap();
    // Prepend BOS token (1) if not present, TinyLlama might not need it manually if encode adds it, but `tokenizers` usually doesn't add BOS by default unless configured. Let's force add `1` if it's missing, but actually let's just prepend 1 just in case as LLaMA requires it.
    let mut ids = vec![1usize];
    ids.extend(encoding.get_ids().iter().map(|&id| id as usize));
    ids
}

pub fn decode(ids: &[usize]) -> String {
    let sp = TOKENIZER.get().expect("Tokenizer not initialized");
    let ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
    sp.decode(&ids_u32, true).unwrap()
}
