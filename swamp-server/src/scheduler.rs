// swamp-server/src/scheduler.rs
// Scheduler: Fila prioritária (BinaryHeap) integrada com LuaJIT + Continuous Batcher

use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use std::time::Instant;
use mlua::Lua;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueuedRequest {
    pub id: u64,
    pub prompt: Option<String>,
    pub messages: Option<Vec<swamp_engine::chat_template::ChatMessage>>,
    pub user_tier: String,
    pub workload_type: String, // CHAT, CODE, REASON, RAG
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

pub struct PriorityRequest {
    pub priority: u8, // 0 = maxima, 255 = minima
    pub created_at: Instant,
    pub req: QueuedRequest,
    pub tx: mpsc::Sender<String>,
}

// Implementacao do ordenamento para BinaryHeap (ordem decrescente de prioridade, ex: menor numero prioritario primeiro)
impl PartialEq for PriorityRequest {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.created_at == other.created_at
    }
}

impl Eq for PriorityRequest {}

impl PartialOrd for PriorityRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // Menor prioridade numerica (ex: 10 vs 50) deve vir PRIMEIRO na BinaryHeap
        // Portanto, invertemos o cmp da prioridade
        other.priority.cmp(&self.priority)
            .then_with(|| other.created_at.cmp(&self.created_at)) // FIFO se mesma prioridade
    }
}

pub struct PriorityQueueScheduler {
    heap: Arc<Mutex<BinaryHeap<PriorityRequest>>>,
    lua: Arc<Mutex<Lua>>,
    max_batch_size: usize,
}

impl PriorityQueueScheduler {
    pub fn new(policy_path: &str, max_batch_size: usize) -> anyhow::Result<Self> {
        let lua = Lua::new();
        // Carrega o script de politicas
        let script = std::fs::read_to_string(policy_path)?;
        lua.load(&script).exec()?;

        Ok(Self {
            heap: Arc::new(Mutex::new(BinaryHeap::new())),
            lua: Arc::new(Mutex::new(lua)),
            max_batch_size,
        })
    }

    /// Avalia a prioridade da requisicao via LuaJIT e adiciona a fila
    pub async fn submit(&self, req: QueuedRequest, tx: mpsc::Sender<String>) -> anyhow::Result<()> {
        let priority = {
            let lua = self.lua.lock().await;
            
            // Converte a struct QueuedRequest para uma tabela Lua para o script avaliar
            let req_json = serde_json::to_string(&req)?;
            let req_table: mlua::Value = lua.load(&format!("json = require('serde_json'); return {}", req_json)).eval().unwrap_or_else(|_| {
                // Fallback manual se o modulo json nao estiver disponivel
                let globals = lua.globals();
                let temp_table = lua.create_table().unwrap();
                temp_table.set("user_tier", req.user_tier.clone()).unwrap();
                temp_table.set("workload_type", req.workload_type.clone()).unwrap();
                temp_table.set("prompt_len", req.prompt.as_ref().map(|s| s.len()).unwrap_or(0)).unwrap();
                globals.set("temp_req", temp_table).unwrap();
                lua.load("temp_req").eval().unwrap()
            });

            // Chama a funcao do script Lua
            let assign_priority: mlua::Function = lua.globals().get("assign_priority")?;
            let priority_res: u8 = assign_priority.call(req_table)?;
            priority_res
        };

        tracing::info!("Requisicao id={} submetida. Prioridade avaliada via Lua: {}", req.id, priority);

        let priority_req = PriorityRequest {
            priority,
            created_at: Instant::now(),
            req,
            tx,
        };

        self.heap.lock().await.push(priority_req);
        Ok(())
    }

    /// Retorna o tamanho atual da fila
    pub async fn len(&self) -> usize {
        self.heap.lock().await.len()
    }

    /// Executa o loop continuo do batcher em background
    pub async fn start_batcher_loop(
        self: Arc<Self>,
        executor: swamp_engine::ModelExecutor,
        metrics: crate::metrics::ConcurrencyMetrics,
    ) {
        loop {
            // Dorme 50ms para acumular requisicoes no batch
            tokio::time::sleep(Duration::from_millis(50)).await;

            let mut batch = Vec::new();
            {
                let mut heap = self.heap.lock().await;
                metrics.queued_requests.set(heap.len() as f64);
                
                // Coleta ate max_batch_size da fila priorizada
                while batch.len() < self.max_batch_size {
                    if let Some(req) = heap.pop() {
                        batch.push(req);
                    } else {
                        break;
                    }
                }
                metrics.queued_requests.set(heap.len() as f64);
            }

            if batch.is_empty() {
                continue;
            }

            tracing::info!("Processando batch de {} requisicoes...", batch.len());

            // Processa o batch (como estamos rodando concorrencia assincrona,
            // executamos cada inferencia do batch em paralelo via tokio::spawn)
            for item in batch {
                let exec = executor.model.clone();
                let tx = item.tx;
                let queued_req = item.req;
                let metrics_clone = metrics.clone();

                metrics.active_requests.inc();
                tokio::spawn(async move {
                    let mut local_executor = swamp_engine::ModelExecutor::new(exec);
                    let start = Instant::now();
                    
                    let req = swamp_engine::InferenceRequest {
                        prompt: queued_req.prompt,
                        messages: queued_req.messages,
                        max_tokens: queued_req.max_tokens,
                        temperature: queued_req.temperature,
                        top_k: queued_req.top_k,
                        top_p: queued_req.top_p,
                    };

                    if let Err(e) = local_executor.generate(req, tx).await {
                        tracing::error!("Erro na geracao da requisicao {}: {:?}", queued_req.id, e);
                    }

                    metrics_clone.active_requests.dec();
                    metrics_clone.inference_latency.observe(start.elapsed().as_secs_f64());
                    metrics_clone.tokens_generated.inc_by(queued_req.max_tokens as f64);
                });
            }
        }
    }
}

use std::time::Duration;
