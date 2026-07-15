use crate::model::Model;
use anyhow::{Result, Context, bail};
use std::collections::HashMap;
use std::sync::Mutex;
use swamp_kernels::fused_gemv_q4k::unpack_scales_q4k;

// =========================================================================
// SmoothQuant — per-channel activation smoothing
//
//   Y = (X · diag(s)⁻¹) · (diag(s) · W)
//
//   s_j = max(|x_j|)^α / max(|w_j|)^(1-α)
//
//   Profiles per-channel max activation during calibration, then requantizes
//   weight blocks with columns scaled by s_j.
// =========================================================================

/// Per-input-channel activation statistics for SmoothQuant.
struct ChannelStats {
    max_abs: Vec<f64>,
    count: u64,
}

impl ChannelStats {
    fn new(n_channels: usize) -> Self {
        Self { max_abs: vec![0.0f64; n_channels], count: 0 }
    }

    fn record(&mut self, x: &[f32]) {
        let n = self.max_abs.len().min(x.len());
        for j in 0..n {
            let v = x[j].abs() as f64;
            if v > self.max_abs[j] {
                self.max_abs[j] = v;
            }
        }
        self.count += 1;
    }

    fn max_per_channel(&self) -> &[f64] { &self.max_abs }
}

/// Tensor-level SmoothQuant stats: per-channel max activation.
pub struct SmoothQuantTensorStats {
    pub channels: Mutex<ChannelStats>,
}

pub struct SmoothQuantCalibrator {
    /// [layer][tensor_idx] → per-channel activation stats
    pub layer_stats: Vec<[SmoothQuantTensorStats; 7]>,
    /// n_cols for each tensor (input dim, = s_j length)
    tensor_ncols: Vec<[usize; 7]>,
}

impl SmoothQuantCalibrator {
    /// Create calibrator. For each of the 7 GEMV tensors per layer, allocates
    /// a ChannelStats buffer of length n_cols (= weight matrix column count).
    pub fn new(model: &Model) -> Self {
        Self::new_with_layers(model, model.config.num_layers)
    }

    pub fn new_with_layers(model: &Model, max_layers: usize) -> Self {
        let num_layers = model.config.num_layers.min(max_layers);
        let mut layer_stats = Vec::with_capacity(num_layers);
        let mut tensor_ncols = Vec::with_capacity(num_layers);

        for l in 0..num_layers {
            let ring = &model.layer_rings[l];
            let ncols: [usize; 7] = [
                ring.q_nc, ring.k_nc, ring.v_nc,
                ring.o_nc, ring.gate_nc, ring.up_nc, ring.down_nc,
            ];
            tensor_ncols.push(ncols);
            layer_stats.push([
                SmoothQuantTensorStats { channels: Mutex::new(ChannelStats::new(ncols[0])) },
                SmoothQuantTensorStats { channels: Mutex::new(ChannelStats::new(ncols[1])) },
                SmoothQuantTensorStats { channels: Mutex::new(ChannelStats::new(ncols[2])) },
                SmoothQuantTensorStats { channels: Mutex::new(ChannelStats::new(ncols[3])) },
                SmoothQuantTensorStats { channels: Mutex::new(ChannelStats::new(ncols[4])) },
                SmoothQuantTensorStats { channels: Mutex::new(ChannelStats::new(ncols[5])) },
                SmoothQuantTensorStats { channels: Mutex::new(ChannelStats::new(ncols[6])) },
            ]);
        }

        Self { layer_stats, tensor_ncols }
    }

    /// Record one activation vector (input to a GEMV).
    /// `x` length must match the tensor's n_cols.
    pub fn record_activation(&self, layer: usize, tensor_idx: usize, x: &[f32]) {
        if layer >= self.layer_stats.len() || tensor_idx >= 7 { return; }
        let mut ch = self.layer_stats[layer][tensor_idx].channels.lock().unwrap();
        ch.record(x);
    }

    /// Compute smooth scales s_j for all tensors.
    /// `alpha` controls migration: 0.0 = only weight magnitude, 1.0 = only activation magnitude.
    /// Typical value: 0.5.
    /// Returns HashMap<gguf_name, Vec<f32>> mapping tensor name → [s_0, ..., s_{n_cols-1}].
    pub fn compute_scales(&self, model: &Model, alpha: f64) -> Result<HashMap<String, Vec<f32>>> {
        let mut scales_map = HashMap::new();

        for l in 0..self.layer_stats.len() {
            for ti in 0..7 {
                let gguf_name = Self::tensor_gguf_name(l, ti);
                let ncols = self.tensor_ncols[l][ti];
                let ch = self.layer_stats[l][ti].channels.lock().unwrap();
                if ch.count == 0 { continue; }

                // Dequantize weight tensor to get max |w_j| per column
                let tensor = match model.gguf.tensor_or_err(&gguf_name) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                let nr = tensor.shape[1] as usize;
                let nc = tensor.shape[0] as usize;
                let mut deq = vec![0.0f32; nr * nc];
                model.gguf.dequantize_tensor(tensor, &mut deq)?;

                // max |w_j| across all rows for column j
                let mut max_w = vec![0.0f64; nc];
                for row in 0..nr {
                    for j in 0..nc {
                        let v = deq[row * nc + j].abs() as f64;
                        if v > max_w[j] { max_w[j] = v; }
                    }
                }

                let act_max = ch.max_per_channel();
                let mut scales = Vec::with_capacity(ncols);
                for j in 0..ncols {
                    let a = act_max.get(j).copied().unwrap_or(1.0).max(1e-10);
                    let w = max_w.get(j).copied().unwrap_or(1.0).max(1e-10);
                    let s = a.powf(alpha) / w.powf(1.0 - alpha);
                    scales.push(s as f32);
                }
                scales_map.insert(gguf_name, scales);
            }
        }
        Ok(scales_map)
    }

    /// Apply SmoothQuant: multiply each weight column j by s_j, requantize
    /// Q4_K blocks, write back to ring.
    /// `scales_map` keyed by GGUF tensor name (e.g. "blk.0.attn_q.weight").
    pub fn apply(
        &self, model: &mut Model,
        scales_map: &HashMap<String, Vec<f32>>,
    ) -> Result<u64> {
        let start = std::time::Instant::now();
        let mut total_blocks = 0u64;

        for (l, ring) in model.layer_rings.iter_mut().enumerate() {
            if l >= self.layer_stats.len() { break; }
            let tensor_meta: [(usize, usize, usize, usize, usize); 7] = [
                (0, ring.q_off,   ring.k_off,   ring.q_nr,   ring.q_nc / 256),
                (1, ring.k_off,   ring.v_off,   ring.k_nr,   ring.k_nc / 256),
                (2, ring.v_off,   ring.o_off,   ring.v_nr,   ring.v_nc / 256),
                (3, ring.o_off,   ring.gate_off, ring.o_nr,  ring.o_nc / 256),
                (4, ring.gate_off, ring.up_off,  ring.gate_nr, ring.gate_nc / 256),
                (5, ring.up_off,  ring.down_off, ring.up_nr,  ring.up_nc / 256),
                (6, ring.down_off, ring.ring.len(), ring.down_nr, ring.down_nc / 256),
            ];

            for &(ti, start_off, end_off, n_rows, n_blocks) in &tensor_meta {
                let gguf_name = Self::tensor_gguf_name(l, ti);
                let s_j = match scales_map.get(&gguf_name) {
                    Some(s) => s,
                    None => continue,
                };
                let ncols = self.tensor_ncols[l][ti];
                if s_j.len() != ncols {
                    bail!("smooth scale length {} != n_cols {} for {}", s_j.len(), ncols, gguf_name);
                }

                let row_bytes = n_blocks * 144;
                let data = &mut ring.ring[start_off..end_off];

                // Dequantize full tensor to FP32, scale columns, requantize per block
                let raw_len = n_rows * row_bytes;
                let mut fp32 = vec![0.0f32; n_rows * ncols];
                // Dequantize row by row
                for row in 0..n_rows {
                    for bc in 0..n_blocks {
                        let off = row * row_bytes + bc * 144;
                        let raw = &data[off..off + 144];
                        let d = half::f16::from_le_bytes([raw[0], raw[1]]).to_f32() as f64;
                        let dm = half::f16::from_le_bytes([raw[2], raw[3]]).to_f32() as f64;
                        let (scales, mins) = unpack_scales_q4k(&raw[4..16]);
                        let qs = &raw[16..144];
                        for sb in 0..8 {
                            let sv = d * (scales[sb] as f64);
                            let mv = dm * (mins[sb] as f64);
                            for i in 0..16 {
                                let ql = (qs[sb * 16 + i] & 0x0F) as f64;
                                let qh = ((qs[sb * 16 + i] >> 4) & 0x0F) as f64;
                                let col = bc * 256 + sb * 32 + i * 2;
                                fp32[row * ncols + col]     = (sv * ql - mv) as f32;
                                fp32[row * ncols + col + 1] = (sv * qh - mv) as f32;
                            }
                        }
                    }
                }

                // Scale columns by s_j
                for row in 0..n_rows {
                    for j in 0..ncols {
                        fp32[row * ncols + j] *= s_j[j];
                    }
                }

                // Requantize blocks back to Q4_K
                for row in 0..n_rows {
                    for bc in 0..n_blocks {
                        let off = row * row_bytes + bc * 144;
                        let block_data = &mut data[off..off + 144];
                        Self::quant_q4k(&fp32[row * ncols + bc * 256..], block_data, 256.min(ncols - bc * 256));
                        total_blocks += 1;
                    }
                }
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        tracing::info!("SmoothQuant: {} blocks requantized in {:.2}s", total_blocks, elapsed);
        Ok(total_blocks)
    }

    /// Pack 8 scales (6-bit) and 8 mins (6-bit) into 12 bytes (Q4_K format).
    fn pack_scales(scales: &[u8; 8], mins: &[u8; 8]) -> [u8; 12] {
        let mut sc = [0u8; 12];
        for j in 0..4 {
            // sc[j] = scales[j] low 6 bits | scales[j+4] high 2 bits << 6
            sc[j] = (scales[j] & 0x3F) | ((scales[j + 4] >> 4) << 6);
            // sc[j+4] = mins[j] low 6 bits | mins[j+4] high 2 bits << 6
            sc[j + 4] = (mins[j] & 0x3F) | ((mins[j + 4] >> 4) << 6);
        }
        for j in 0..4 {
            // sc[8..12]: scales[4..7] low 4 bits, mins[4..7] low 4 bits
            sc[8 + j] = (scales[j + 4] & 0x0F) | ((mins[j + 4] & 0x0F) << 4);
        }
        sc
    }

    /// Quantize up to 256 FP32 values into a Q4_K block (144 bytes).
    /// dst must have length ≥ 144.
    fn quant_q4k(src: &[f32], dst: &mut [u8], n: usize) {
        const BLOCK_SIZE: usize = 256;
        let n = n.min(BLOCK_SIZE);
        dst.fill(0);

        // Pre-compute per-sub-block statistics
        let mut sub_scales = [0u8; 8];
        let mut sub_mins = [0u8; 8];

        for sb in 0..8 {
            let start = sb * 32;
            let end = (start + 32).min(n);
            if start >= n { break; }

            let (mut vmax, mut vmin) = (f32::NEG_INFINITY, f32::INFINITY);
            for i in start..end {
                vmax = vmax.max(src[i]);
                vmin = vmin.min(src[i]);
            }

            let range = (vmax - vmin).max(1e-10);
            let step = range / 63.0; // 6-bit → 63 intervals
            sub_scales[sb] = (vmax / step).round().max(0.0).min(63.0) as u8;
            sub_mins[sb] = (vmin / step).round().max(0.0).min(63.0) as u8;

            // Quantize 32 elements to 16 nibble-bytes
            for i in start..end {
                let q = ((src[i] - vmin) / step).round().max(0.0).min(15.0) as u8;
                let byte_off = 16 + sb * 16 + (i - start) / 2;
                let shift = if (i - start) % 2 == 0 { 0 } else { 4 };
                dst[byte_off] |= q << shift;
            }
        }

        // Compute block-level d/dm from sub-block maxima
        let mut global_max = f32::NEG_INFINITY;
        let mut global_min = f32::INFINITY;
        for sb in 0..8 {
            let start = sb * 32;
            let end = (start + 32).min(n);
            if start >= n { break; }
            for i in start..end {
                global_max = global_max.max(src[i]);
                global_min = global_min.min(src[i]);
            }
        }
        let d = (global_max - global_min).max(1e-10) / 7.0;
        let dm = global_min;

        dst[0..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
        dst[2..4].copy_from_slice(&half::f16::from_f32(dm).to_le_bytes());
        let packed = Self::pack_scales(&sub_scales, &sub_mins);
        dst[4..16].copy_from_slice(&packed);
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
}

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
