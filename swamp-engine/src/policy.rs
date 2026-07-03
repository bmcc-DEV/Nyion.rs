// swamp-engine/src/policy.rs
// LuaJIT Policy Engine: decisoes termicas e de agendamento hot-reloadable

use mlua::{Lua, Result as LuaResult, Value};
use std::sync::{Arc, Mutex, atomic::{AtomicI64, Ordering}};
use std::path::Path;
use std::time::UNIX_EPOCH;

pub struct PolicyEngine {
    lua: Arc<Mutex<Lua>>,
    script_path: String,
    last_load_ns: AtomicI64,
    fallback_threads: usize,
}

impl PolicyEngine {
    pub fn new(script_path: &str, fallback_threads: usize) -> LuaResult<Self> {
        let lua = Lua::new();

        // Sandbox: limitar acesso a APIs perigosas
        lua.load("os = nil; io = nil; debug = nil").exec()?;

        let engine = Self {
            lua: Arc::new(Mutex::new(lua)),
            script_path: script_path.to_string(),
            last_load_ns: AtomicI64::new(0),
            fallback_threads,
        };

        engine.reload()?;
        Ok(engine)
    }

    pub fn reload(&self) -> LuaResult<()> {
        let script = std::fs::read_to_string(&self.script_path)
            .map_err(|e| mlua::Error::RuntimeError(format!("Failed to read policy: {}", e)))?;

        let lua = self.lua.lock().unwrap();
        lua.load(&script).exec()?;
        self.update_last_load();
        Ok(())
    }

    pub fn try_reload(&self) {
        let path = Path::new(&self.script_path);
        if let Ok(metadata) = path.metadata() {
            if let Ok(modified) = metadata.modified() {
                let modified_ns = modified.duration_since(UNIX_EPOCH)
                    .map(|d| d.as_nanos() as i64)
                    .unwrap_or(0);
                if modified_ns > self.last_load_ns.load(Ordering::Acquire) {
                    if let Err(e) = self.reload() {
                        eprintln!("[swamp] Policy reload failed: {}", e);
                    } else {
                        println!("[swamp] Policy hot-reloaded: {}", self.script_path);
                    }
                }
            }
        }
    }

    fn update_last_load(&self) {
        if let Ok(metadata) = Path::new(&self.script_path).metadata() {
            if let Ok(modified) = metadata.modified() {
                if let Ok(d) = modified.duration_since(UNIX_EPOCH) {
                    self.last_load_ns.store(d.as_nanos() as i64, Ordering::Release);
                }
            }
        }
    }

    pub fn adapt_threads(&self, c_epsilon: f64, temp_celsius: f64, freq_mhz: u64) -> usize {
        let lua = self.lua.lock().unwrap();

        match lua.globals().get::<Value>("adapt_threads") {
            Ok(Value::Function(f)) => {
                match f.call::<i64>((c_epsilon, temp_celsius, freq_mhz as i64)) {
                    Ok(n) => (n.max(1).min(6)) as usize,
                    Err(e) => {
                        eprintln!("[swamp] Lua adapt_threads error: {}", e);
                        self.fallback_threads
                    }
                }
            }
            _ => self.fallback_threads,
        }
    }

    pub fn should_skip_layer(&self, layer_idx: usize, temp_celsius: f64) -> bool {
        let lua = self.lua.lock().unwrap();

        match lua.globals().get::<Value>("should_skip_layer") {
            Ok(Value::Function(f)) => {
                match f.call::<bool>((layer_idx as i64, temp_celsius)) {
                    Ok(skip) => skip,
                    Err(e) => {
                        eprintln!("[swamp] Lua should_skip_layer error: {}", e);
                        false
                    }
                }
            }
            _ => false,
        }
    }

    pub fn suggested_batch_size(&self, context_tokens: usize, temp_celsius: f64) -> usize {
        let lua = self.lua.lock().unwrap();

        match lua.globals().get::<Value>("suggest_batch_size") {
            Ok(Value::Function(f)) => {
                match f.call::<i64>((context_tokens as i64, temp_celsius)) {
                    Ok(n) => (n.max(8).min(2048)) as usize,
                    Err(e) => {
                        eprintln!("[swamp] Lua suggest_batch_size error: {}", e);
                        512
                    }
                }
            }
            _ => 512,
        }
    }
}
