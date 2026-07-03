// swamp-engine/src/thermal.rs
// ThermalLSC: Coordenador Termico real com base em Telemetria de Frequencia (LSC)

use std::fs;
use std::time::{Instant, Duration};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalState {
    Normal,
    Warning,
    Critical,
}

pub struct ThermalLSC {
    pub max_observed_freq: u64, // kHz
    pub throttle_count: u32,
    pub warning_temp: f32,
    pub critical_temp: f32,
    pub current_temp: f32,
    pub state: ThermalState,
    last_check: Instant,
}

impl ThermalLSC {
    pub fn new(warning_temp: f32, critical_temp: f32) -> Self {
        Self {
            max_observed_freq: 0,
            throttle_count: 0,
            warning_temp,
            critical_temp,
            current_temp: 45.0,
            state: ThermalState::Normal,
            last_check: Instant::now(),
        }
    }

    /// Le a frequencia atual do CPU0 via sysfs (kHz)
    pub fn current_freq_khz(&self) -> u64 {
        fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    }

    /// Verifica se estamos em throttling baseado na queda de frequencia (AVX-512 power limit hit)
    pub fn is_throttling(&mut self) -> bool {
        // Para evitar overhead excessivo, checa apenas a cada 50ms
        if self.last_check.elapsed() < Duration::from_millis(50) {
            return self.throttle_count > 0;
        }
        self.last_check = Instant::now();

        let freq = self.current_freq_khz();
        if freq > self.max_observed_freq {
            self.max_observed_freq = freq;
        }

        // Se a frequencia atual e < 70% da maxima, a back-reaction termica bateu forte
        if self.max_observed_freq > 0 && freq < (self.max_observed_freq * 70 / 100) {
            self.throttle_count = self.throttle_count.saturating_add(1);
            true
        } else {
            self.throttle_count = self.throttle_count.saturating_sub(1);
            false
        }
    }

    /// Calcula a coerência epsilon (Cε) da Teoria LSC (0.0 a 1.0)
    pub fn coherence(&self) -> f64 {
        let freq = self.current_freq_khz() as f64;
        let max = self.max_observed_freq as f64;
        if max == 0.0 {
            return 1.0; // Assume coerencia perfeita se nao sabemos o maximo ainda
        }
        (freq / max).clamp(0.0, 1.0)
    }

    // --- Retrocompatibilidade com mock anterior ---

    pub fn update_temperature(&mut self, temp: f32) -> ThermalState {
        self.current_temp = temp;
        // Na vida real leríamos de /sys/class/thermal/thermal_zone0/temp
        self.state
    }

    pub fn throttle_factor(&self) -> f32 {
        if self.max_observed_freq == 0 {
            return 1.0;
        }
        let c_epsilon = self.coherence();
        if c_epsilon < 0.6 {
            0.3 // Corte severo
        } else if c_epsilon < 0.86 {
            0.7 // Reduz levemente
        } else {
            1.0 // Normal
        }
    }
}

impl Default for ThermalLSC {
    fn default() -> Self {
        Self::new(75.0, 85.0)
    }
}

// Retrocompatibilidade para o executor não quebrar
pub type ThermalCoordinator = ThermalLSC;
