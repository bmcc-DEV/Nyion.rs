use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub fn format_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    
    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str("<|system|>\n");
                prompt.push_str(&msg.content);
                prompt.push_str("</s>\n");
            }
            "user" => {
                prompt.push_str("<|user|>\n");
                prompt.push_str(&msg.content);
                prompt.push_str("</s>\n");
            }
            "assistant" => {
                prompt.push_str("<|assistant|>\n");
                prompt.push_str(&msg.content);
                prompt.push_str("</s>\n");
            }
            _ => {}
        }
    }
    
    // Sempre termina com <|assistant|> para o modelo completar
    prompt.push_str("<|assistant|>\n");
    prompt
}
