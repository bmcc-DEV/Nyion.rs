#!/bin/bash
# swamp-power-tune.sh
# Tuning de energia para o LLamañón.rs (Fase 3: mitigacao da back-reaction LSC)
# 
# ATENCAO: Requer privilegios de root (sudo).

echo "=== Swamp Power Tune (LSC Fase 3) ==="

# 1. Fixa o CPU governor em 'performance' para evitar histerese de P-state
echo "Configurando governor para 'performance'..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    if [ -f "$cpu" ]; then
        echo performance | sudo tee "$cpu" > /dev/null
    fi
done

# 2. Desativa C-states profundos (Wake-up latency fix)
echo "Desativando C-states profundos (dma_latency)..."
echo 1 | sudo tee /dev/cpu_dma_latency > /dev/null &
DMA_PID=$!

# 3. Aumenta PL1 via RAPL (se a BIOS e o modulo Intel RAPL permitirem)
RAPL_PATH="/sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw"
if [ -f "$RAPL_PATH" ]; then
    echo "Aumentando PL1 RAPL Limit (55W)..."
    echo 55000000 | sudo tee "$RAPL_PATH" > /dev/null
else
    echo "Intel RAPL nao disponivel ou sem permissao."
fi

# 4. Desabilita o AVX-512 heavy downclock (Experimental)
# O msr 0x1FC contem os flags de licensing do AVX.
# Dependendo do chip e da BIOS, isso impede o estrangulamento imediato de clock.
if command -v wrmsr >/dev/null 2>&1; then
    echo "Aplicando MSR override para AVX-512 (disable heavy license)..."
    sudo wrmsr -a 0x1FC 0x4004005f 2>/dev/null || echo "wrmsr falhou (MSR lockado ou sem msr module)."
else
    echo "msr-tools (wrmsr) nao instalado. Pulando tuning de MSR."
fi

echo "=== Tuning Concluido ==="
echo "Nota: O bloqueio de dma_latency (C-States) esta rodando em background (PID $DMA_PID)."
echo "Ele sera revertido quando voce fechar o terminal ou matar o processo."
