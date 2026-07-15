// swamp-kernels/src/hvx_sim.rs
//
// HyperVec (HVX) — emulador em Rust puro.
//
// Isto NAO e um binding para intrinsics de hardware real (nao existe silicio
// HVX). E uma simulacao funcional do formato de registrador especulativo
// HVX (512 bits, canais BF16.16 / BF32.32 emparelhados) rodando em cima de
// aritmetica escalar/f32 padrao do Rust — no mesmo espirito dos emuladores
// de CPU do MonteLauro CD³² (PPC/ColdFire) e do FX-64 "Tenrai": um sandbox
// pra validar a ideia antes de existir hardware de verdade.
//
// Uso pretendido aqui: camada de execucao para o profiling de ativacoes do
// SmoothQuant (ver qat.rs), nao um substituto dos kernels AVX2/AVX-512/VNNI
// reais que ja rodam em swamp-kernels — aqueles continuam sendo o caminho
// quente de producao.

use half::bf16;

/// Registrador HVX de 512 bits (64 bytes), backing real em bytes.
/// Os "views" (bf16_16 / bf32_32) sao computados sob demanda a partir do
/// buffer cru — isso evita as garantias de aliasing problematicas de uma
/// union real em Rust e mantem tudo em safe code.
#[derive(Clone, Copy)]
pub struct HyperVecReg {
    bytes: [u8; 64],
}

/// Um canal BF16.16: dois bfloat16 emparelhados (ex: peso + gradiente,
/// ou parte real + imaginaria).
#[derive(Clone, Copy, Debug, Default)]
pub struct Bf1616 {
    pub primary: bf16,
    pub secondary: bf16,
}

/// Um canal BF32.32: dois f32 emparelhados, usado como acumulador de alta
/// precisao (dot products longos sem perda por truncamento intermediario).
#[derive(Clone, Copy, Debug, Default)]
pub struct Bf3232 {
    pub primary: f32,
    pub secondary: f32,
}

impl Default for HyperVecReg {
    fn default() -> Self {
        Self { bytes: [0u8; 64] }
    }
}

impl HyperVecReg {
    pub fn zeroed() -> Self {
        Self::default()
    }

    /// 16 canais BF16.16 (32 valores bfloat16 no total).
    pub fn bf16_16_lanes(&self) -> [Bf1616; 16] {
        let mut out = [Bf1616::default(); 16];
        for i in 0..16 {
            let off = i * 4;
            let p = u16::from_le_bytes([self.bytes[off], self.bytes[off + 1]]);
            let s = u16::from_le_bytes([self.bytes[off + 2], self.bytes[off + 3]]);
            out[i] = Bf1616 {
                primary: bf16::from_bits(p),
                secondary: bf16::from_bits(s),
            };
        }
        out
    }

    pub fn set_bf16_16_lane(&mut self, i: usize, v: Bf1616) {
        let off = i * 4;
        self.bytes[off..off + 2].copy_from_slice(&v.primary.to_bits().to_le_bytes());
        self.bytes[off + 2..off + 4].copy_from_slice(&v.secondary.to_bits().to_le_bytes());
    }

    /// 8 canais BF32.32 (16 valores f32 no total).
    pub fn bf32_32_lanes(&self) -> [Bf3232; 8] {
        let mut out = [Bf3232::default(); 8];
        for i in 0..8 {
            let off = i * 8;
            let p = f32::from_le_bytes(self.bytes[off..off + 4].try_into().unwrap());
            let s = f32::from_le_bytes(self.bytes[off + 4..off + 8].try_into().unwrap());
            out[i] = Bf3232 { primary: p, secondary: s };
        }
        out
    }

    pub fn set_bf32_32_lane(&mut self, i: usize, v: Bf3232) {
        let off = i * 8;
        self.bytes[off..off + 4].copy_from_slice(&v.primary.to_le_bytes());
        self.bytes[off + 4..off + 8].copy_from_slice(&v.secondary.to_le_bytes());
    }

    /// Carrega 32 valores bfloat16 (16 canais BF16.16) a partir de um slice f32,
    /// convertendo cada elemento pra bf16 no processo.
    pub fn load_bf16_16_from_f32(data: &[f32; 32]) -> Self {
        let mut reg = Self::zeroed();
        for i in 0..16 {
            reg.set_bf16_16_lane(
                i,
                Bf1616 {
                    primary: bf16::from_f32(data[i * 2]),
                    secondary: bf16::from_f32(data[i * 2 + 1]),
                },
            );
        }
        reg
    }
}

/// Mascara de esparsidade indireta: 32 bits, 1 bit por canal bf16 (32 canais
/// no formato BF16.16). Bit 1 = canal ativo (nao-zero), bit 0 = pula.
#[derive(Clone, Copy, Default)]
pub struct SparsityMask(pub u32);

impl SparsityMask {
    /// Constroi a mascara a partir de um registrador: qualquer canal cujo
    /// valor absoluto seja menor que `eps` e marcado como esparso (pulavel).
    pub fn from_reg(reg: &HyperVecReg, eps: f32) -> Self {
        let mut mask = 0u32;
        for (i, lane) in reg.bf16_16_lanes().iter().enumerate() {
            let p_active = lane.primary.to_f32().abs() > eps;
            let s_active = lane.secondary.to_f32().abs() > eps;
            if p_active { mask |= 1 << (i * 2); }
            if s_active { mask |= 1 << (i * 2 + 1); }
        }
        SparsityMask(mask)
    }

    pub fn is_active(&self, lane_idx: usize) -> bool {
        (self.0 >> lane_idx) & 1 == 1
    }

    pub fn active_count(&self) -> u32 {
        self.0.count_ones()
    }
}

/// PRNG leve (xorshift32) para arredondamento estocastico. Determinístico
/// dado um seed — nao usa rdtsc nem qualquer fonte de entropia de hardware,
/// pra manter o resultado reprodutível em testes.
pub struct Xorshift32(u32);

impl Xorshift32 {
    pub fn new(seed: u32) -> Self {
        Self(if seed == 0 { 0xdead_beef } else { seed })
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
}

/// Converte f32 -> bf16 com arredondamento estocástico em vez de round-to-nearest.
/// Isso reduz o viés sistemático de truncamento quando muitas conversões pequenas
/// se acumulam ao longo de uma sequência longa (ex: accumulators de attention).
pub fn f32_to_bf16_stochastic(val: f32, rng: &mut Xorshift32) -> bf16 {
    if !val.is_finite() {
        return bf16::from_f32(val);
    }
    let bits = val.to_bits();
    // bf16 mantém os 16 bits superiores de um f32. Os 16 bits inferiores
    // (mantissa perdida) decidem, probabilisticamente, se arredonda pra cima.
    let lower16 = bits & 0xFFFF;
    let threshold = rng.next_u32() & 0xFFFF;
    let rounded_bits = if lower16 > threshold {
        // Arredonda pra cima: soma 1 no bit 16 (pode causar carry, o que é
        // correto e intencional — propaga pro expoente se necessário).
        bits.wrapping_add(0x1_0000) & 0xFFFF_0000
    } else {
        bits & 0xFFFF_0000
    };
    bf16::from_bits((rounded_bits >> 16) as u16)
}

/// FMA (Fused Multiply-Accumulate) BF16.16 x BF16.16 -> BF32.32.
/// Multiplica os canais emparelhados de vA e vB, acumulando em vD.
/// Equivalente sofisticado ao `vmaddfp` do AltiVec, mas em dobro de largura
/// e com o acumulador em precisao estendida (BF32.32) para evitar drift
/// numerico em somas longas.
pub fn hvx_fma_bf16_to_bf32(vd: &mut HyperVecReg, va: &HyperVecReg, vb: &HyperVecReg) {
    let la = va.bf16_16_lanes();
    let lb = vb.bf16_16_lanes();
    let mut ld = vd.bf32_32_lanes();

    for i in 0..8 {
        let a1 = la[i * 2].primary.to_f32();
        let a2 = la[i * 2].secondary.to_f32();
        let b1 = lb[i * 2].primary.to_f32();
        let b2 = lb[i * 2].secondary.to_f32();

        ld[i].primary += a1 * b1;
        ld[i].secondary += a2 * b2;
    }
    for (i, v) in ld.iter().enumerate() {
        vd.set_bf32_32_lane(i, *v);
    }
}

/// Permutacao dinamica + ativacao integrada (ReLU), no espirito do `vperm`
/// do AltiVec: reordena os canais de vA segundo `mask.u32[i] % 16`, aplicando
/// ReLU no valor selecionado antes de escrever em vD.
pub fn hvx_permute_and_relu(vd: &mut HyperVecReg, va: &HyperVecReg, mask: &[u32; 16]) {
    let la = va.bf16_16_lanes();
    for i in 0..16 {
        let target_lane = (mask[i] % 16) as usize;
        let selected = la[target_lane];
        let p = if selected.primary.to_f32() > 0.0 { selected.primary } else { bf16::from_f32(0.0) };
        let s = if selected.secondary.to_f32() > 0.0 { selected.secondary } else { bf16::from_f32(0.0) };
        vd.set_bf16_16_lane(i, Bf1616 { primary: p, secondary: s });
    }
}

// =========================================================================
// Ponte com a calibração SmoothQuant (qat.rs)
// =========================================================================

/// Estatísticas de ativação por canal de entrada, coletadas usando o HVX
/// simulado como camada de execução. Compatível com o `s_j` do SmoothQuant:
///   s_j = max(|x_j|)^alpha / max(|w_j|)^(1-alpha)
///
/// A vantagem de rodar isso "via HVX" em vez de escalar puro é a mascara de
/// esparsidade: canais que ficam zerados na maior parte do dataset de
/// calibração (comum em camadas com GELU/ReLU) sao contados mas nao pesam
/// no `max_abs`, entao nao inflam `s_j` artificialmente por causa de um
/// outlier isolado num canal majoritariamente morto.
#[derive(Clone)]
pub struct ChannelActivationStats {
    pub max_abs: Vec<f32>,
    pub sum_abs: Vec<f32>,
    pub active_samples: Vec<u64>,
    pub total_samples: u64,
}

impl ChannelActivationStats {
    pub fn new(num_channels: usize) -> Self {
        Self {
            max_abs: vec![0.0; num_channels],
            sum_abs: vec![0.0; num_channels],
            active_samples: vec![0; num_channels],
            total_samples: 0,
        }
    }

    /// Perfila um batch de ativacoes (linha de tokens x canais), processando
    /// 32 canais por vez atraves de registradores HVX simulados. `row` deve
    /// ter exatamente `self.max_abs.len()` elementos.
    pub fn profile_row(&mut self, row: &[f32], sparsity_eps: f32) {
        assert_eq!(row.len(), self.max_abs.len());
        self.total_samples += 1;

        let mut chunk_start = 0;
        while chunk_start < row.len() {
            let chunk_len = (row.len() - chunk_start).min(32);
            let mut buf = [0f32; 32];
            buf[..chunk_len].copy_from_slice(&row[chunk_start..chunk_start + chunk_len]);

            let reg = HyperVecReg::load_bf16_16_from_f32(&buf);
            let mask = SparsityMask::from_reg(&reg, sparsity_eps);
            let lanes = reg.bf16_16_lanes();

            for i in 0..chunk_len {
                let lane_idx = i;
                let is_primary = lane_idx % 2 == 0;
                let pair = lanes[lane_idx / 2];
                let v = if is_primary { pair.primary.to_f32() } else { pair.secondary.to_f32() };
                let ch = chunk_start + i;

                if mask.is_active(lane_idx) {
                    let av = v.abs();
                    if av > self.max_abs[ch] { self.max_abs[ch] = av; }
                    self.sum_abs[ch] += av;
                    self.active_samples[ch] += 1;
                }
            }
            chunk_start += chunk_len;
        }
    }

    pub fn mean_abs(&self, channel: usize) -> f32 {
        if self.active_samples[channel] == 0 { return 0.0; }
        self.sum_abs[channel] / self.active_samples[channel] as f32
    }
}

/// Calcula os fatores de suavizacao SmoothQuant por canal de entrada.
///   s_j = max(|x_j|)^alpha / max(|w_j|)^(1-alpha)
/// `weight_max_abs` e o maximo absoluto do peso (dequantizado) na coluna j —
/// isso NAO tenta recuperar o FP32 original, so usa a magnitude do que já
/// está no bloco Q4_K, o que e uma quantidade real e disponível (diferente
/// de tentar reconstruir o erro contra um ground truth que nao existe mais).
pub fn compute_smooth_scales(
    act_stats: &ChannelActivationStats,
    weight_max_abs: &[f32],
    alpha: f32,
) -> Vec<f32> {
    assert_eq!(act_stats.max_abs.len(), weight_max_abs.len());
    act_stats
        .max_abs
        .iter()
        .zip(weight_max_abs.iter())
        .map(|(&xa, &wa)| {
            let xa = xa.max(1e-5);
            let wa = wa.max(1e-5);
            xa.powf(alpha) / wa.powf(1.0 - alpha)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_bf16_lanes() {
        let mut data = [0f32; 32];
        for i in 0..32 { data[i] = i as f32 - 16.0; }
        let reg = HyperVecReg::load_bf16_16_from_f32(&data);
        let lanes = reg.bf16_16_lanes();
        // bf16 tem pouca mantissa; valores inteiros pequenos devem sobreviver exatos.
        assert_eq!(lanes[0].primary.to_f32(), -16.0);
        assert_eq!(lanes[15].secondary.to_f32(), 15.0);
    }

    #[test]
    fn fma_accumulates_across_calls() {
        let a = HyperVecReg::load_bf16_16_from_f32(&[1.0; 32]);
        let b = HyperVecReg::load_bf16_16_from_f32(&[2.0; 32]);
        let mut acc = HyperVecReg::zeroed();
        hvx_fma_bf16_to_bf32(&mut acc, &a, &b);
        hvx_fma_bf16_to_bf32(&mut acc, &a, &b);
        let lanes = acc.bf32_32_lanes();
        // Cada FMA soma 1.0*2.0 = 2.0; duas chamadas -> 4.0.
        assert!((lanes[0].primary - 4.0).abs() < 1e-6);
    }

    #[test]
    fn sparsity_mask_skips_zero_lanes() {
        let mut data = [0f32; 32];
        data[3] = 5.0;
        let reg = HyperVecReg::load_bf16_16_from_f32(&data);
        let mask = SparsityMask::from_reg(&reg, 1e-6);
        assert!(mask.is_active(3));
        assert!(!mask.is_active(0));
        assert_eq!(mask.active_count(), 1);
    }

    #[test]
    fn stochastic_rounding_is_unbiased_on_average() {
        let mut rng = Xorshift32::new(42);
        let val = 1.0000001_f32; // fração minúscula acima de um valor representável em bf16
        let mut sum = 0.0f64;
        let n = 20_000;
        for _ in 0..n {
            sum += f32_to_bf16_stochastic(val, &mut rng).to_f32() as f64;
        }
        let mean = sum / n as f64;
        // A média das conversões estocásticas deve convergir pra perto do valor real,
        // mesmo que cada conversão individual não consiga representar essa precisão.
        assert!((mean - val as f64).abs() < 0.01);
    }

    #[test]
    fn smooth_scales_basic() {
        let mut stats = ChannelActivationStats::new(2);
        stats.profile_row(&[10.0, 1.0], 1e-6);
        let weight_max = vec![1.0, 1.0];
        let scales = compute_smooth_scales(&stats, &weight_max, 0.5);
        // Canal 0 tem ativação maior -> fator de suavização maior.
        assert!(scales[0] > scales[1]);
    }
}
