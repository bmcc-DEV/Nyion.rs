-- policies/swamp_policy.lua
-- Politica de roteamento e prioridade do LLamañón.rs

function select_profile(workload_type, prompt_len)
    -- CHAT: latência ultra-baixa, compressão KV agressiva
    -- CODE: latência média, sem compressão
    -- REASON: alta entropia, máxima precisão, sem decaimento
    -- RAG: contexto gigante, compressão de contexto ativada
    if workload_type == "CHAT" then
        return "low_latency"
    elseif workload_type == "CODE" then
        return "balanced"
    elseif workload_type == "REASON" then
        return "max_precision"
    elseif workload_type == "RAG" then
        return "compressed_context"
    else
        return "default"
    end
end

function assign_priority(req)
    -- Determina prioridade de 0 (máxima) a 255 (mínima)
    local priority = 100 -- padrão

    -- Tier do usuário
    if req.user_tier == "PREMIUM" then
        priority = priority - 50
    elseif req.user_tier == "SYSTEM" then
        priority = priority - 80
    end

    -- Tipo de workload
    if req.workload_type == "CHAT" then
        priority = priority - 10 -- Chat responde mais rápido
    elseif req.workload_type == "RAG" then
        priority = priority + 20 -- RAG pode esperar mais
    end

    return math.max(0, math.min(255, priority))
end

function should_throttle(thermal_state)
    -- Se estiver crítico, throttle imediato
    if thermal_state == "Critical" then
        return true
    end
    return false
end
