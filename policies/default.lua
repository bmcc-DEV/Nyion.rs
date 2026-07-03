-- default.lua – Politica termica padrao para LLamanon.rs
-- Edit this file at runtime; PolicyEngine detecta modificacoes e recarrega automaticamente.

local thermal = {}

function thermal.adapt_threads(c_epsilon, temp_celsius, freq_mhz)
    if temp_celsius > 85 then
        return 1  -- back off: 1 thread apenas
    elseif c_epsilon > 0.86 then
        return 6  -- coerencia alta: full throttle (6C/12T)
    elseif c_epsilon > 0.7 then
        return 4  -- medio: 4 threads
    elseif temp_celsius > 78 then
        return 2  -- aquecendo: conservador
    else
        return 6  -- frio: maximo
    end
end

-- should_skip_layer intentionally omitted:
-- skipping layers breaks the residual stream and causes severe quality degradation.
-- Use adapt_threads + suggest_batch_size for thermal backoff instead.

function thermal.suggest_batch_size(context_tokens, temp_celsius)
    if temp_celsius > 88 then
        return 128
    elseif temp_celsius > 82 then
        return 256
    elseif context_tokens > 1024 then
        return 1024
    else
        return 2048
    end
end

-- Export global
adapt_threads = thermal.adapt_threads
suggest_batch_size = thermal.suggest_batch_size
