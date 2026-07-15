use anyhow::Result;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

/// Decide se uma camada executa no GPU ou CPU
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackDecision {
    Gpu(usize),
    Cpu,
}

pub struct LayerResult {
    pub decision: FallbackDecision,
    pub tensor_data: Option<Vec<u8>>,
}

/// Schedule de transfers para K tokens ahead
#[derive(Debug)]
pub struct LayerSchedule {
    pub token_id: usize,
    pub transfers: Vec<LayerTransfer>,
}

#[derive(Debug)]
pub struct LayerTransfer {
    pub layer_id: usize,
    pub tensor_name: String,
}

#[derive(Debug)]
struct PrefetchRecord {
    layer_id: usize,
    tensor_name: String,
    slot_index: usize,
    timeline_value: u64,
}

const ALL_WEIGHT_NAMES: &[&str] = &["q", "k", "v", "o", "gate", "up", "down"];
const CORE_WEIGHT_NAMES: &[&str] = &["q", "k", "v"];

pub struct HyperStreamEngine {
    reader: swamp_gpu::streaming::shard::ShardReader,
    transfer: swamp_gpu::streaming::transfer::TransferEngine,
    k: usize,
    num_layers: usize,
    prefetch_queue: VecDeque<PrefetchRecord>,
}

unsafe impl Send for HyperStreamEngine {}

impl HyperStreamEngine {
    pub fn open<P: AsRef<Path>>(
        backend: &Arc<swamp_gpu::VkBackend>,
        index_path: P,
        data_path: P,
        slot_size: u64,
    ) -> Result<Self> {
        let reader = swamp_gpu::streaming::shard::ShardReader::open(index_path, data_path)
            .map_err(|e| anyhow::anyhow!("abrindo shard reader: {}", e))?;
        let num_layers = reader.index.num_layers as usize;
        let transfer = swamp_gpu::streaming::transfer::TransferEngine::new(backend, slot_size)?;
        Ok(Self {
            reader,
            transfer,
            k: 3,
            num_layers,
            prefetch_queue: VecDeque::new(),
        })
    }

    pub fn telemetry(&self) -> &swamp_gpu::streaming::StreamTelemetry {
        &self.transfer.telemetry
    }

    pub fn k(&self) -> usize { self.k }

    pub fn set_k(&mut self, k: usize) {
        self.k = k.max(1);
        self.transfer.telemetry.k_current.store(self.k as u64, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn num_layers(&self) -> usize { self.num_layers }

    /// Resolve K tokens ahead em transfers individuais
    pub fn resolve_k(&self, tokens: &[usize], k: usize) -> Vec<LayerSchedule> {
        let mut schedules = Vec::with_capacity(k.min(tokens.len()));
        for (i, &token_id) in tokens.iter().enumerate().take(k) {
            let mut transfers = Vec::with_capacity(self.num_layers * ALL_WEIGHT_NAMES.len());
            for layer in 0..self.num_layers {
                for name in ALL_WEIGHT_NAMES {
                    if self.reader.index.layer_offsets(layer as u32, name).is_some() {
                        transfers.push(LayerTransfer {
                            layer_id: layer,
                            tensor_name: name.to_string(),
                        });
                    }
                }
            }
            schedules.push(LayerSchedule { token_id, transfers });
        }
        schedules
    }

    /// Enfileira upload de uma layer + tensor
    pub fn prefetch_layer(&mut self, layer_id: usize, tensor_name: &str) {
        let data = match self.reader.get_tensor(layer_id as u32, tensor_name) {
            Some(d) => d,
            None => { self.transfer.telemetry.record_miss(); return; }
        };
        if data.len() > self.transfer.triple_buffer.max_slot_size as usize {
            self.transfer.telemetry.record_miss();
            return;
        }
        match self.transfer.upload_slot(data) {
            Some((idx, tv)) => {
                self.prefetch_queue.push_back(PrefetchRecord {
                    layer_id,
                    tensor_name: tensor_name.to_string(),
                    slot_index: idx,
                    timeline_value: tv,
                });
                self.transfer.telemetry.record_prefetch_hit();
            }
            None => { self.transfer.telemetry.record_miss(); }
        }
    }

    /// Prefetch K tokens ahead: layer atual + K layers seguintes
    pub fn prefetch_k_ahead(&mut self, current_layer: usize) {
        let k = self.k;
        // prefetch proxima layer (todos os pesos)
        let next = current_layer + 1;
        if next < self.num_layers {
            for name in ALL_WEIGHT_NAMES {
                self.prefetch_layer(next, name);
            }
        }
        // layers mais distantes (so Q/K/V - atencao)
        for ahead in 2..=k {
            let far = current_layer + ahead;
            if far < self.num_layers {
                for name in CORE_WEIGHT_NAMES {
                    self.prefetch_layer(far, name);
                }
            }
        }
    }

    /// Prefetch usando predicoes DSPark
    pub fn prefetch_with_dspark(
        &mut self,
        hidden_state: &[f32],
        pos: usize,
        dspark: &crate::dspark::DSparkEngine,
        current_layer: usize,
    ) {
        let (predicted_tokens, confidences) = dspark.draft_model.draft(hidden_state);
        let k = self.k.min(predicted_tokens.len());

        for (i, (&_token, &conf)) in predicted_tokens[..k].iter().zip(confidences[..k].iter()).enumerate() {
            if conf < 0.3 {
                continue;
            }
            let layer_offset = current_layer + 1 + i * (self.num_layers / self.k.max(1));
            for layer in layer_offset..(layer_offset + 2).min(self.num_layers) {
                for name in if conf > 0.7 { ALL_WEIGHT_NAMES } else { CORE_WEIGHT_NAMES } {
                    self.prefetch_layer(layer, name);
                }
            }
        }
    }

    /// Espera layer ficar pronta, com timeout
    pub fn wait_for_layer(&mut self, layer_id: usize, tensor_name: &str, timeout_ns: u64) -> LayerResult {
        let pos = self.prefetch_queue.iter().position(|r| {
            r.layer_id == layer_id && r.tensor_name == tensor_name
        });

        if let Some(idx) = pos {
            let record = self.prefetch_queue.remove(idx).unwrap();
            if self.transfer.wait_for_timeline(record.timeline_value, timeout_ns) {
                self.transfer.triple_buffer.release_slot(record.slot_index);
                self.transfer.telemetry.record_prefetch_hit();
                return LayerResult { decision: FallbackDecision::Gpu(record.slot_index), tensor_data: None };
            } else {
                self.transfer.telemetry.record_prefetch_miss();
                self.transfer.triple_buffer.release_slot(record.slot_index);
            }
        } else {
            self.transfer.telemetry.record_prefetch_miss();
        }

        let data = self.reader.get_tensor(layer_id as u32, tensor_name).map(|s| s.to_vec());
        if data.is_some() {
            self.transfer.telemetry.record_miss();
        }
        LayerResult { decision: FallbackDecision::Cpu, tensor_data: data }
    }

    /// Le dados raw de um tensor do shard (zero-copy via mmap).
    /// Usado pelo scheduler para carregar pesos no GPU via streaming.
    pub fn read_tensor(&self, layer_id: usize, tensor_name: &str) -> Option<&[u8]> {
        self.reader.get_tensor(layer_id as u32, tensor_name)
    }

    /// Libera slots nao utilizados
    pub fn release_pending(&mut self) {
        while let Some(record) = self.prefetch_queue.pop_front() {
            let _ = self.transfer.wait_for_timeline(record.timeline_value, 0);
            self.transfer.triple_buffer.release_slot(record.slot_index);
        }
    }

    pub fn sync_all(&self) { self.transfer.sync_all(); }

    pub fn transfer_mut(&mut self) -> &mut swamp_gpu::streaming::transfer::TransferEngine {
        &mut self.transfer
    }
}
