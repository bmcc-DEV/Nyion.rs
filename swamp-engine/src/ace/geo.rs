use crate::ace::graph::{ConceptNode, Shape};

// =========================================================================
// GeoVex core operations — Phase 3a
// =========================================================================

/// Warp: apply linear transformation to a shape.
/// Each node vector v becomes M · v where M is a 64×64 matrix.
pub fn warp(shape: &Shape, matrix: &[f32; 4096], bias: &[f32; 64]) -> Shape {
    let nodes: Vec<ConceptNode> = shape.nodes.iter().map(|node| {
        let mut new_vec = [0i8; 64];
        for i in 0..64 {
            let mut sum = bias[i];
            for j in 0..64 {
                sum += matrix[i * 64 + j] * node.vector[j] as f32;
            }
            new_vec[i] = (sum.round().max(-128.0).min(127.0)) as i8;
        }
        ConceptNode {
            id: node.id,
            vector: new_vec,
            confidence: node.confidence,
            connections: node.connections.clone(),
        }
    }).collect();

    let mut result = Shape::new(&format!("{}_warped", shape.name), nodes);
    result.rebuild_edges(0.3);
    result
}

/// Merge: convex combination of two concepts.
/// result = alpha * a + (1 - alpha) * b.
pub fn merge_concepts(a: &[i8; 64], b: &[i8; 64], alpha: f32) -> [i8; 64] {
    let mut result = [0i8; 64];
    for j in 0..64 {
        let v = alpha * a[j] as f32 + (1.0 - alpha) * b[j] as f32;
        result[j] = (v.round().max(-128.0).min(127.0)) as i8;
    }
    result
}

/// Expand: generate K variants of each concept by adding Gaussian noise.
pub fn expand(shape: &Shape, sigma: f32, k: usize) -> Shape {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut new_nodes: Vec<ConceptNode> = shape.nodes.clone();

    for node in &shape.nodes {
        for _ in 0..k {
            let mut vec = [0i8; 64];
            for j in 0..64 {
                let noise = rng.gen::<f32>() * sigma * 2.0 - sigma;
                let v = (node.vector[j] as f32 + noise).round()
                    .max(-128.0).min(127.0);
                vec[j] = v as i8;
            }
            let dot = node.dot(&vec) as f32 / 8192.0;
            if dot > 0.7 {
                new_nodes.push(ConceptNode {
                    id: node.id.wrapping_add(new_nodes.len() as u32),
                    vector: vec,
                    confidence: (node.confidence as f32 * 0.8) as u16,
                    connections: vec![],
                });
            }
        }
    }

    let mut result = Shape::new(&format!("{}_expanded", shape.name), new_nodes);
    result.rebuild_edges(0.3);
    result
}

/// Compress: keep only top-P% of concepts by importance.
/// importance = confidence (higher = more important).
pub fn compress(shape: &Shape, keep_pct: f64) -> Shape {
    if keep_pct >= 1.0 { return shape.clone(); }
    let keep_count = ((shape.len() as f64) * keep_pct).ceil() as usize;
    let keep_count = keep_count.max(1).min(shape.len());

    let mut indexed: Vec<(usize, &ConceptNode)> = shape.nodes.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.confidence.cmp(&a.1.confidence));

    let keep_indices: Vec<usize> = indexed.iter().take(keep_count).map(|(i, _)| *i).collect();
    let kept: Vec<ConceptNode> = keep_indices.iter().map(|i| shape.nodes[*i].clone()).collect();

    Shape::new(&format!("{}_compressed", shape.name), kept)
}

/// Split: connected components at similarity threshold epsilon.
pub fn split(shape: &Shape, epsilon: f32) -> Vec<Shape> {
    let n = shape.nodes.len();
    if n == 0 { return vec![]; }

    // Build adjacency list
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for &(i, j, sim) in &shape.edges {
        if sim > epsilon {
            adj[i].push(j);
            adj[j].push(i);
        }
    }

    // BFS to find connected components
    let mut visited = vec![false; n];
    let mut components: Vec<Vec<usize>> = Vec::new();

    for start in 0..n {
        if visited[start] { continue; }
        let mut component = Vec::new();
        let mut stack = vec![start];
        while let Some(idx) = stack.pop() {
            if visited[idx] { continue; }
            visited[idx] = true;
            component.push(idx);
            for &neighbor in &adj[idx] {
                if !visited[neighbor] {
                    stack.push(neighbor);
                }
            }
        }
        if !component.is_empty() {
            components.push(component);
        }
    }

    components.iter().enumerate().map(|(ci, indices)| {
        let nodes: Vec<ConceptNode> = indices.iter().map(|i| shape.nodes[*i].clone()).collect();
        Shape::new(&format!("{}_split{}", shape.name, ci), nodes)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_node(id: u32, val: i8) -> ConceptNode {
        ConceptNode::new(id, [val; 64])
    }

    #[test]
    fn test_warp_identity() {
        let shape = Shape::new("test", vec![dummy_node(1, 42)]);
        let mut ident = [0.0f32; 4096];
        let mut bias = [0.0f32; 64];
        for i in 0..64 { ident[i * 64 + i] = 1.0; }
        bias[0] = 0.0;
        let warped = warp(&shape, &ident, &bias);
        assert_eq!(warped.nodes[0].vector[0], 42);
    }

    #[test]
    fn test_expand() {
        let shape = Shape::new("test", vec![dummy_node(1, 10)]);
        let expanded = expand(&shape, 0.5, 3);
        assert!(expanded.len() >= 1);
    }

    #[test]
    fn test_compress() {
        let shape = Shape::new("test", vec![
            dummy_node(1, 1),
            dummy_node(2, 2),
            dummy_node(3, 3),
        ]);
        let compressed = compress(&shape, 0.5);
        assert!(compressed.len() <= 2);
    }

    #[test]
    fn test_split() {
        let shape = Shape::new("test", vec![
            dummy_node(1, 0),
            dummy_node(2, 100),
        ]);
        let parts = split(&shape, 0.5);
        assert!(parts.len() >= 1);
    }
}
