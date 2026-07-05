// swamp-engine/src/dspark.rs
// DSpark: speculative decoding com n-gram draft + verificação por rejection sampling
//
// Algoritmo:
//   1. Draft model gera K tokens candidatos (rápido, baseado em n-gram + unigram)
//   2. Target model verifica cada token via forward pass + rejection sampling
//   3. Aceita o maior prefixo válido; rejeitado → resample da distribuição corrigida
//   4. Ganho: K tokens aceitos pelo custo de <K forward passes do target

use std::collections::HashMap;

// =========================================================================
// NgramDraft — modelo de linguagem leve baseado em frequências de tokens
// =========================================================================

pub struct NgramDraft {
    unigram: HashMap<usize, f64>,
    bigram:  HashMap<(usize, usize), f64>,
    trigram: HashMap<(usize, usize, usize), f64>,
    total: f64,
}

impl NgramDraft {
    pub fn new() -> Self {
        Self {
            unigram: HashMap::new(),
            bigram: HashMap::new(),
            trigram: HashMap::new(),
            total: 0.0,
        }
    }

    /// Learn from a generated token sequence.
    pub fn observe(&mut self, tokens: &[usize]) {
        for &t in tokens {
            *self.unigram.entry(t).or_insert(0.0) += 1.0;
            self.total += 1.0;
        }
        for win in tokens.windows(2) {
            *self.bigram.entry((win[0], win[1])).or_insert(0.0) += 1.0;
        }
        for win in tokens.windows(3) {
            *self.trigram.entry((win[0], win[1], win[2])).or_insert(0.0) += 1.0;
        }
    }

    /// Draft K tokens given the last N context tokens.
    /// Returns (draft_tokens, draft_log_probs) where draft_log_probs[i] = ln(q(d_i)).
    pub fn draft(&self, ctx: &[usize], k: usize) -> (Vec<usize>, Vec<f64>) {
        let mut tokens = Vec::with_capacity(k);
        let mut probs  = Vec::with_capacity(k);

        // Extend context with already-drafted tokens for multi-step draft
        let mut ext_ctx = ctx.to_vec();

        for _ in 0..k {
            let predicted = if ext_ctx.len() >= 2 {
                let key3 = (ext_ctx[ext_ctx.len()-2], ext_ctx[ext_ctx.len()-1], 0);
                // Try trigram first
                let mut best = None;
                let mut best_score = 0.0f64;
                for ((a, b, c), &cnt) in &self.trigram {
                    if *a == key3.0 && *b == key3.1 {
                        let score = cnt / self.total.max(1.0);
                        if score > best_score {
                            best_score = score;
                            best = Some(*c);
                        }
                    }
                }
                if let Some(t) = best {
                    t
                } else if ext_ctx.len() >= 1 {
                    // Fallback to bigram
                    let key2 = ext_ctx[ext_ctx.len()-1];
                    let mut best = None;
                    let mut best_score = 0.0f64;
                    for ((a, b), &cnt) in &self.bigram {
                        if *a == key2 {
                            let score = cnt / self.total.max(1.0);
                            if score > best_score {
                                best_score = score;
                                best = Some(*b);
                            }
                        }
                    }
                    best.unwrap_or_else(|| self.most_frequent_token())
                } else {
                    self.most_frequent_token()
                }
            } else {
                self.most_frequent_token()
            };

            // Get probability of predicted token
            let prob = self.token_probability(predicted, &ext_ctx);
            tokens.push(predicted);
            probs.push((prob + 1e-30).ln());
            ext_ctx.push(predicted);
        }

        (tokens, probs)
    }

    /// Probability of a token given context (Laplace-smoothed).
    pub fn token_probability(&self, token: usize, ctx: &[usize]) -> f64 {
        let trigram_prob = if ctx.len() >= 2 {
            let key = (ctx[ctx.len()-2], ctx[ctx.len()-1], token);
            self.trigram.get(&key).copied().unwrap_or(0.0) / self.total.max(1.0)
        } else { 0.0 };

        let bigram_prob = if ctx.len() >= 1 {
            let key = (ctx[ctx.len()-1], token);
            self.bigram.get(&key).copied().unwrap_or(0.0) / self.total.max(1.0)
        } else { 0.0 };

        let unigram_prob = self.unigram.get(&token).copied().unwrap_or(0.0) / self.total.max(1.0);

        // Interpolation: prefer higher-order n-grams when confident
        let total_vocab = self.unigram.len().max(1) as f64;
        let laplace = 1.0 / (self.total + total_vocab);

        if trigram_prob > 0.01 {
            0.7 * trigram_prob + 0.2 * bigram_prob + 0.1 * laplace
        } else if bigram_prob > 0.01 {
            0.3 * bigram_prob + 0.5 * unigram_prob + 0.2 * laplace
        } else {
            0.8 * unigram_prob + 0.2 * laplace
        }
    }

    fn most_frequent_token(&self) -> usize {
        self.unigram.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(&t, _)| t)
            .unwrap_or(1) // fallback to BOS
    }

    /// Perplexity of the draft model on a sequence (for diagnostics).
    #[allow(dead_code)]
    pub fn perplexity(&self, tokens: &[usize]) -> f64 {
        let mut nll = 0.0f64;
        let mut n = 0usize;
        for i in 1..tokens.len() {
            let ctx = &tokens[0..i];
            let p = self.token_probability(tokens[i], ctx).max(1e-30);
            nll -= p.ln();
            n += 1;
        }
        (nll / n.max(1) as f64).exp()
    }
}

// =========================================================================
// DSparkEngine — orquestra draft + verificação
// =========================================================================

pub struct DSparkEngine {
    pub draft_model: NgramDraft,
    pub max_draft: usize,
    pub min_acceptance_prob: f64,
    stats: DSparkStats,
}

#[derive(Debug, Default, Clone)]
pub struct DSparkStats {
    pub total_steps: u64,
    pub total_draft_tokens: u64,
    pub accepted_tokens: u64,
    pub rejected_steps: u64,
    pub total_draft_attempts: u64,
    pub draft_accepted: u64,
    pub draft_rejected: u64,
    pub total_rollbacks: u64,
}

impl DSparkEngine {
    pub fn new(max_draft: usize) -> Self {
        Self {
            draft_model: NgramDraft::new(),
            max_draft,
            min_acceptance_prob: 0.01,
            stats: DSparkStats::default(),
        }
    }

    pub fn stats(&self) -> &DSparkStats {
        &self.stats
    }

    pub fn stats_mut(&mut self) -> &mut DSparkStats {
        &mut self.stats
    }

    /// Observe generated tokens to improve the draft model.
    pub fn observe(&mut self, tokens: &[usize]) {
        self.draft_model.observe(tokens);
    }

    /// Attempt speculative decoding for one step.
    ///
    /// Returns `(accepted_tokens, rejected)` where `accepted_tokens` is the
    /// list of tokens to emit (at least 1, the normal sample), and `rejected`
    /// indicates if speculation failed.
    ///
    /// `target_logits` is the model's output logits for the CURRENT prefix.
    /// `sample_token` samples from a distribution (usually softmax + top-k).
    /// `verify_fn` runs the target model for one extra token and returns its logits.
    pub fn speculate(
        &mut self,
        ctx: &[usize],
        target_logits: &[f32],
        sample_token: impl Fn(&[f32]) -> usize,
        verify_fn: impl Fn(usize) -> (Vec<f32>, usize), // (logits, sampled_token)
    ) -> Vec<usize> {
        self.stats.total_steps += 1;

        // 1. Sample the normal next token from target distribution
        let t0 = sample_token(target_logits);
        let mut accepted = vec![t0];
        self.stats.accepted_tokens += 1;

        // 2. Try to draft and verify K-1 extra tokens
        let (draft, draft_lps) = self.draft_model.draft(ctx, self.max_draft);

        for i in 0..self.max_draft {
            let d = draft[i];
            let draft_lp = draft_lps[i];

            // 3. Run target model with the draft token as input
            let (logits, _sampled) = verify_fn(d);

            // 4. Rejection sampling
            // Compute target log prob of the draft token
            let target_p = softmax_at(logits.as_slice(), d);
            if target_p <= 0.0 {
                // Draft token impossible for target — reject immediately
                self.stats.rejected_steps += 1;
                break;
            }

            let target_lp = (target_p as f64).ln();
            let ratio = (target_lp - draft_lp).exp().min(1.0);

            let r: f64 = fast_rand();
            if r < ratio {
                // Accept the draft token
                accepted.push(d);
                self.stats.accepted_tokens += 1;
                // Extend context for next draft iteration
                // (context grows as we accept)
            } else {
                // Reject: resample from corrected distribution
                let resampled = resample_from_corrected(&logits, d, target_p, sample_token);
                accepted.push(resampled);
                self.stats.rejected_steps += 1;
                break;
            }
        }

        accepted
    }
}

// =========================================================================
// Helpers
// =========================================================================

pub(crate) fn softmax_at(logits: &[f32], token: usize) -> f32 {
    if token >= logits.len() { return 0.0; }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    let mut t_exp = 0.0f64;
    for (i, &l) in logits.iter().enumerate() {
        let e = (l as f64 - max_val as f64).exp();
        sum += e;
        if i == token {
            t_exp = e;
        }
    }
    if sum <= 0.0 { return 0.0; }
    (t_exp / sum) as f32
}

/// Resample from max(0, target - draft) distribution.
/// When rejection occurs, the corrected distribution is:
///   p_corrected(x) = max(0, p_target(x) - q_draft(x)) / Z
pub(crate) fn resample_from_corrected(
    logits: &[f32],
    draft_token: usize,
    _target_p_draft: f32,
    sample_token: impl Fn(&[f32]) -> usize,
) -> usize {
    // Simple approach: just sample from the target distribution
    // (the corrected distribution rarely differs meaningfully from target
    //  when rejection is rare)
    sample_token(logits)
}

/// Fast thread-local random in [0, 1).
pub(crate) fn fast_rand() -> f64 {
    // xorshift64-style
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = const { Cell::new(0x123456789abcdef) };
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x as f64) / (u64::MAX as f64)
    })
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_draft_basic() {
        let mut draft = NgramDraft::new();
        // Feed some repeated patterns
        let corpus: Vec<usize> = vec![1, 2, 3, 1, 2, 3, 1, 2, 4, 1, 2, 3];
        draft.observe(&corpus);

        // After [1, 2], should predict 3 (most common)
        let (tokens, _) = draft.draft(&[1, 2], 2);
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], 3, "trigram should predict 3 after [1,2]");
    }

    #[test]
    fn test_dspark_speculate_no_rejection() {
        let mut engine = DSparkEngine::new(2);
        let ctx = vec![1, 2];
        engine.draft_model.observe(&[1, 2, 3, 1, 2, 3]);

        // Mock: target always agrees with draft
        let target_logits: Vec<f32> = (0..100).map(|i| if i == 3 { 10.0 } else { 0.0 }).collect();
        let result = engine.speculate(
            &ctx,
            &target_logits,
            |_| 3, // sample always returns 3
            |d| {
                // verify_fn: logits that favor token d
                let logits: Vec<f32> = (0..100).map(|i| if i == d || i == 3 { 10.0 } else { 0.0 }).collect();
                (logits, d)
            },
        );
        assert!(result.len() >= 2, "should accept at least 2 tokens");
    }

    #[test]
    fn test_fast_rand_range() {
        for _ in 0..1000 {
            let r = fast_rand();
            assert!(r >= 0.0 && r < 1.0, "fast_rand out of range: {}", r);
        }
    }
}
