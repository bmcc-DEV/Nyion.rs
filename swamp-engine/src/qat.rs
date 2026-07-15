use crate::model::Model;
use anyhow::{Result, Context};
use std::collections::HashMap;
use std::sync::Mutex;
use swamp_kernels::fused_gemv_q4k::unpack_scales_q4k;

#[derive(Clone)]
struct BlockStats {
    act_sum: [f64; 256],
    count: u64,
}

impl Default for BlockStats {
    fn default() -> Self { Self { act_sum: [0.0; 256], count: 0 } }
}

pub struct TensorStats {
    pub blocks: Vec<Mutex<BlockStats>>,
}

pub struct QatCalibrator {
    pub layer_stats: Vec<[TensorStats; 7]>,
    n_blocks_embed: usize,
    n_blocks_ffn: usize,
}

impl QatCalibrator {
    pub fn new(model: &Model) -> Self {
        Self::new_with_layers(model, model.config.num_layers)
    }

    pub fn new_with_layers(model: &Model, max_layers: usize) -> Self {
        let num_layers = model.config.num_layers.min(max_layers);
        let embed_dim = model.config.embed_dim;
        let n_blocks_embed = embed_dim / 256;
        // Compute ffn_dim from the first layer's ring
        let n_blocks_ffn = if num_layers > 0 {
            model.layer_rings[0].down_nc / 256
        } else { 1 };

        let layer_stats = (0..num_layers).map(|l| {
            let r = &model.layer_rings[l];
            [
                TensorStats { blocks: (0..r.q_nr * n_blocks_embed).map(|_| Mutex::new(BlockStats::default())).collect() },
                TensorStats { blocks: (0..r.k_nr * n_blocks_embed).map(|_| Mutex::new(BlockStats::default())).collect() },
                TensorStats { blocks: (0..r.v_nr * n_blocks_embed).map(|_| Mutex::new(BlockStats::default())).collect() },
                TensorStats { blocks: (0..r.o_nr * n_blocks_embed).map(|_| Mutex::new(BlockStats::default())).collect() },
                TensorStats { blocks: (0..r.gate_nr * n_blocks_embed).map(|_| Mutex::new(BlockStats::default())).collect() },
                TensorStats { blocks: (0..r.up_nr * n_blocks_embed).map(|_| Mutex::new(BlockStats::default())).collect() },
                TensorStats { blocks: (0..r.down_nr * n_blocks_ffn).map(|_| Mutex::new(BlockStats::default())).collect() },
            ]
        }).collect();

        Self { layer_stats, n_blocks_embed, n_blocks_ffn }
    }

    pub const T_Q: usize = 0;
    pub const T_K: usize = 1;
    pub const T_V: usize = 2;
    pub const T_O: usize = 3;
    pub const T_GATE: usize = 4;
    pub const T_UP: usize = 5;
    pub const T_DOWN: usize = 6;

    pub fn record_activation(
        &self, layer: usize, tensor_idx: usize, block_idx: usize, x_blk: &[f32]
    ) {
        if layer >= self.layer_stats.len() || tensor_idx >= 7 { return; }
        let ts = &self.layer_stats[layer][tensor_idx];
        if block_idx >= ts.blocks.len() { return; }
        let mut stats = ts.blocks[block_idx].lock().unwrap();
        let n = 256.min(x_blk.len());
        for i in 0..n { stats.act_sum[i] += x_blk[i].abs() as f64; }
        stats.count += 1;
    }

    #[inline]
    pub fn block_idx(row: usize, bc: usize, n_blocks_per_row: usize) -> usize {
        row * n_blocks_per_row + bc
    }

    fn tensor_gguf_name(layer: usize, ti: usize) -> String {
        match ti {
            0 => format!("blk.{}.attn_q.weight", layer),
            1 => format!("blk.{}.attn_k.weight", layer),
            2 => format!("blk.{}.attn_v.weight", layer),
            3 => format!("blk.{}.attn_output.weight", layer),
            4 => format!("blk.{}.ffn_gate.weight", layer),
            5 => format!("blk.{}.ffn_up.weight", layer),
            6 => format!("blk.{}.ffn_down.weight", layer),
            _ => panic!("invalid tensor index {ti}"),
        }
    }

    /// Apply QAT calibration, using GGUF dequantized FP32 weights as target.
    /// Pass `external_fp32` to supply original pre-quantization FP32 weights
    /// (keyed by GGUF tensor name, e.g. "blk.0.attn_q.weight").
    /// When None, falls back to `gguf.dequantize_tensor()` (best-effort on Q4_K).
    pub fn apply_ring_with_fp32(
        &self, model: &mut Model,
        external_fp32: Option<&HashMap<String, Vec<f32>>>,
    ) -> Result<u64> {
        let start = std::time::Instant::now();
        let mut total_updated = 0u64;

        for (l, ring) in model.layer_rings.iter_mut().enumerate() {
            if l >= self.layer_stats.len() { break; }
            let tl = &self.layer_stats[l];
            let tensor_meta: [(usize, usize, usize, usize, usize); 7] = [
                (0, ring.q_off,   ring.k_off,   ring.q_nr,   self.n_blocks_embed),
                (1, ring.k_off,   ring.v_off,   ring.k_nr,   self.n_blocks_embed),
                (2, ring.v_off,   ring.o_off,   ring.v_nr,   self.n_blocks_embed),
                (3, ring.o_off,   ring.gate_off, ring.o_nr,  self.n_blocks_embed),
                (4, ring.gate_off, ring.up_off,  ring.gate_nr, self.n_blocks_embed),
                (5, ring.up_off,  ring.down_off, ring.up_nr,  self.n_blocks_embed),
                (6, ring.down_off, ring.ring.len(), ring.down_nr, self.n_blocks_ffn),
            ];

            for &(ti, start_off, end_off, n_rows, n_blocks) in &tensor_meta {
                let ts = &tl[ti];
                let row_bytes = n_blocks * 144;
                let data = &mut ring.ring[start_off..end_off];

                // Obtain FP32 target for this tensor
                let gguf_name = Self::tensor_gguf_name(l, ti);
                let (n_cols, fp32_target) = if let Some(fp32_map) = external_fp32 {
                    if let Some(w) = fp32_map.get(&gguf_name) {
                        // n_cols = w.len() / n_rows
                        let nc = w.len() / n_rows;
                        (nc, w.clone())
                    } else {
                        continue;
                    }
                } else {
                    // Fallback: dequantize from GGUF
                    let tensor = match model.gguf.tensor_or_err(&gguf_name) {
                        Ok(t) => t,
                        Err(_) => continue,
                    };
                    let nc = tensor.shape[0] as usize;
                    let nr = tensor.shape[1] as usize;
                    let mut deq = vec![0.0f32; nr * nc];
                    model.gguf.dequantize_tensor(tensor, &mut deq)
                        .context(format!("dequantizando {gguf_name} para QAT"))?;
                    (nc, deq)
                };

                for row in 0..n_rows {
                    for bc in 0..n_blocks {
                        let off = row * row_bytes + bc * 144;
                        if off + 144 > data.len() { continue; }
                        let bi = row * n_blocks + bc;
                        if bi >= ts.blocks.len() { continue; }

                        let stats = ts.blocks[bi].lock().unwrap();
                        if stats.count < 1 { continue; }

                        // Build target block from FP32 weights
                        let block_start = row * n_cols + bc * 256;
                        let mut target = [0.0f64; 256];
                        for i in 0..256 {
                            target[i] = fp32_target[block_start + i] as f64;
                        }

                        let raw: &[u8] = &data[off..off + 144];
                        let d_cur = half::f16::from_le_bytes([raw[0], raw[1]]).to_f32() as f64;
                        let dm_cur = half::f16::from_le_bytes([raw[2], raw[3]]).to_f32() as f64;
                        let (scales, mins) = unpack_scales_q4k(&raw[4..16]);
                        let qs: &[u8] = &raw[16..144];

                        let total_act = stats.act_sum.iter().sum::<f64>();
                        if total_act < 1e-10 { continue; }
                        let mut imp = [0.0f64; 256];
                        for i in 0..256 { imp[i] = stats.act_sum[i] / total_act; }

                        // Wider grid: ±10%, ±5%, ±1%, ±0.5%, center
                        let dc = [d_cur * 0.90, d_cur * 0.95, d_cur * 0.99, d_cur * 0.995,
                                  d_cur,
                                  d_cur * 1.005, d_cur * 1.01, d_cur * 1.05, d_cur * 1.10];
                        let dmc = [dm_cur * 0.90, dm_cur * 0.95, dm_cur * 0.99, dm_cur * 0.995,
                                   dm_cur,
                                   dm_cur * 1.005, dm_cur * 1.01, dm_cur * 1.05, dm_cur * 1.10];
                        let (bd, bdm) = Self::gs(&target, &imp, qs, &scales, &mins, &dc, &dmc);

                        let d_new_f16 = half::f16::from_f32(bd as f32);
                        let dm_new_f16 = half::f16::from_f32(bdm as f32);
                        let d_old_f16 = half::f16::from_f32(d_cur as f32);
                        let dm_old_f16 = half::f16::from_f32(dm_cur as f32);
                        if d_new_f16.to_bits() != d_old_f16.to_bits()
                            || dm_new_f16.to_bits() != dm_old_f16.to_bits()
                        {
                            data[off..off + 2].copy_from_slice(&d_new_f16.to_le_bytes());
                            data[off + 2..off + 4].copy_from_slice(&dm_new_f16.to_le_bytes());
                            total_updated += 1;
                        }
                    }
                }
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        tracing::info!("QAT: {} blocks updated in {:.2}s", total_updated, elapsed);
        Ok(total_updated)
    }

    /// Apply QAT using GGUF dequantized FP32 as target (no external weights).
    pub fn apply_ring(&self, model: &mut Model) -> Result<u64> {
        self.apply_ring_with_fp32(model, None)
    }

    fn gs(
        target: &[f64; 256], imp: &[f64; 256], qs: &[u8],
        scales: &[u8; 8], mins: &[u8; 8],
        dc: &[f64; 9], dmc: &[f64; 9],
    ) -> (f64, f64) {
        let mut bd = dc[4]; let mut bdm = dmc[4]; let mut bm = f64::MAX;
        for &d in dc { for &dm in dmc {
            let mut m = 0.0f64;
            for sb in 0..8 {
                let sv = d * (scales[sb] as f64); let mv = dm * (mins[sb] as f64);
                for i in 0..16 {
                    let ql = (qs[sb * 16 + i] & 0x0F) as f64;
                    let qh = ((qs[sb * 16 + i] >> 4) & 0x0F) as f64;
                    let ie = sb * 32 + i * 2;
                    let io = ie + 1;
                    let ee = target[ie] - (sv * ql - mv);
                    let eo = target[io] - (sv * qh - mv);
                    m += (ee * ee) * imp[ie] + (eo * eo) * imp[io];
                }
            }
            if m < bm { bm = m; bd = d; bdm = dm; }
        }}
        (bd, bdm)
    }
}
