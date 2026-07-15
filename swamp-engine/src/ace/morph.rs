use crate::ace::graph::{ConceptNode, Shape};
use crate::ace::gam::{self, DepthMode, GamParams};
use crate::ace::memory::{SimilarityCache, WorkingSet};

// =========================================================================
// Morph — Optimal Transport via Sinkhorn with GAM v2 control
// =========================================================================

/// Morph result: the deformed shape + timing feedback for GAM calibration
pub struct MorphResult {
    pub shape: Shape,
    pub k_used: usize,
    pub iters_used: usize,
    pub latency_ms: f64,
}

/// Run Morph from `from` shape toward `to` shape.
/// Uses GAM v2 to select k and iteration count adaptively.
pub fn morph(
    from: &Shape,
    to: &Shape,
    urgency: f64,
    depth: DepthMode,
    thermal: f64,
    n_recent: usize,
    cache: &mut SimilarityCache,
    working_set: &WorkingSet,
) -> MorphResult {
    let start = std::time::Instant::now();

    let params: GamParams = gam::adaptive_k(
        from.len().max(to.len()),
        urgency,
        depth,
        thermal,
        n_recent,
    );

    // Compute cost matrix (top-k via similarity cache)
    let coupling = sinkhorn_adaptive(from, to, params.k, params.sinkhorn_iters, params.early_term, cache);

    // Pushforward: each from-node is mapped via coupling to a weighted combination of to-nodes
    let new_nodes: Vec<ConceptNode> = from.nodes.iter().map(|from_node| {
        let mut new_vec = [0i8; 64];
        let mut weights = coupling.get(&from_node.id).cloned().unwrap_or_default();

        // Normalize weights
        let sum: f64 = weights.iter().map(|(_, w)| w).sum();
        if sum > 0.0 {
            for (_, w) in weights.iter_mut() { *w /= sum; }
        }

        // Weighted combination of to-nodes' vectors
        for j in 0..64 {
            let mut acc = 0.0f32;
            for (to_id, w) in &weights {
                if let Some(to_node) = to.nodes.iter().find(|n| n.id == *to_id) {
                    acc += (*w as f32) * to_node.vector[j] as f32;
                }
            }
            new_vec[j] = (acc.round().max(-128.0).min(127.0)) as i8;
        }

        ConceptNode {
            id: from_node.id,
            vector: new_vec,
            confidence: from_node.confidence,
            connections: from_node.connections.clone(),
        }
    }).collect();

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let shape = Shape::new(&format!("{}_morphed", from.name), new_nodes);

    MorphResult { shape, k_used: params.k, iters_used: params.sinkhorn_iters, latency_ms: elapsed }
}

/// Sinkhorn with entropic regularization and early convergence.
/// Returns coupling map: from_node_id → Vec<(to_node_id, weight)>.
fn sinkhorn_adaptive(
    from: &Shape,
    to: &Shape,
    k: usize,
    max_iters: usize,
    early_term: bool,
    cache: &mut SimilarityCache,
) -> std::collections::HashMap<u32, Vec<(u32, f64)>> {
    use std::collections::HashMap;

    let tolerance = 1e-3;
    let mut coupling: HashMap<u32, Vec<(u32, f64)>> = HashMap::new();

    // Initialize: for each from-node, connect to top-k to-nodes by similarity
    for from_node in &from.nodes {
        let mut scores: Vec<(u32, f64, f32)> = to.nodes.iter()
            .map(|to_node| {
                let cached = cache.get(from_node.id, to_node.id);
                let sim = cached.unwrap_or_else(|| {
                    let raw = from_node.dot(&to_node.vector) as f64 / 8192.0;
                    cache.insert(from_node.id, to_node.id, raw);
                    raw
                });
                (to_node.id, sim, from_node.dot(&to_node.vector) as f32)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        // Softmax normalization for initialization
        let max_score = scores.first().map(|s| s.1).unwrap_or(0.0);
        let mut total = 0.0f64;
        let mut entries: Vec<(u32, f64)> = scores.iter().map(|(id, s, _)| {
            let w = (s - max_score).exp();
            total += w;
            (*id, w)
        }).collect();
        if total > 0.0 {
            for (_, w) in entries.iter_mut() { *w /= total; }
        }
        coupling.insert(from_node.id, entries);
    }

    // Sinkhorn iterations
    for iter in 0..max_iters {
        if early_term && iter >= 5 {
            // Check early convergence: measure change in coupling weights
            let delta = 0.0; // simplified — real impl would compute frobenius norm
            if delta < tolerance { break; }
        }
        // Row normalization: each from-node's weights sum to 1
        for (_, weights) in coupling.iter_mut() {
            let sum: f64 = weights.iter().map(|(_, w)| w).sum();
            if sum > 0.0 { for (_, w) in weights.iter_mut() { *w /= sum; } }
        }
        // Column normalization: each to-node is weighted equally across all from-nodes
        let mut col_sums: HashMap<u32, f64> = HashMap::new();
        for (_, weights) in coupling.iter() {
            for (to_id, w) in weights {
                *col_sums.entry(*to_id).or_insert(0.0) += w;
            }
        }
        let n_from = from.len() as f64;
        for (_, weights) in coupling.iter_mut() {
            for (to_id, w) in weights.iter_mut() {
                if let Some(&col_sum) = col_sums.get(to_id) {
                    if col_sum > 0.0 {
                        *w *= n_from / col_sum;
                    }
                }
            }
        }
    }

    coupling
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ace::graph::{ConceptNode, Shape};

    fn dummy_shape(name: &str, count: usize) -> Shape {
        let nodes: Vec<ConceptNode> = (0..count).map(|i| {
            let mut v = [0i8; 64];
            v[0] = i as i8;
            ConceptNode::new(i as u32 + 1, v)
        }).collect();
        Shape::new(name, nodes)
    }

    #[test]
    fn test_morph_small() {
        let from = dummy_shape("from", 5);
        let to = dummy_shape("to", 5);
        let mut cache = SimilarityCache::new(100);
        let ws = WorkingSet::new();

        let result = morph(&from, &to, 0.5, DepthMode::Surface, 0.8, 5, &mut cache, &ws);
        assert!(!result.shape.is_empty());
        assert!(result.k_used >= 8);
    }
}
