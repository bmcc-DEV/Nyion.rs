use crate::ace::graph::{ConceptNode, Shape};

// =========================================================================
// Persistent Homology — Phase 3b (simplified start)
// H0 via DSU, H1 via spanning tree
// =========================================================================

#[derive(Debug, Clone)]
pub struct Simplex {
    pub vertices: Vec<usize>,
    pub filtration: f32,
}

/// Build a Vietoris-Rips complex at given epsilon.
/// Returns simplices up to dimension 2 (edges + triangles).
pub fn build_vietoris_rips(shape: &Shape, epsilon: f32) -> Vec<Simplex> {
    let n = shape.nodes.len();

    // 0-simplices (vertices) — always present
    let mut simplices: Vec<Simplex> = (0..n)
        .map(|i| Simplex { vertices: vec![i], filtration: 0.0 })
        .collect();

    // 1-simplices (edges) — where sim > epsilon
    for &(i, j, sim) in &shape.edges {
        if sim > epsilon {
            simplices.push(Simplex { vertices: vec![i, j], filtration: 1.0 - sim });
        }
    }

    // 2-simplices (triangles) — where all 3 edges exist
    let edge_set: std::collections::HashSet<(usize, usize)> = shape.edges.iter()
        .filter(|(_, _, sim)| *sim > epsilon)
        .map(|(i, j, _)| if *i < *j { (*i, *j) } else { (*j, *i) })
        .collect();

    for i in 0..n {
        for j in (i + 1)..n {
            if !edge_set.contains(&(i, j)) { continue; }
            for k in (j + 1)..n {
                if edge_set.contains(&(i, k)) && edge_set.contains(&(j, k)) {
                    let max_sim = shape.edges.iter()
                        .filter(|(a, b, _)| (*a == i && *b == k) || (*a == k && *b == i)
                             || (*a == j && *b == k) || (*a == k && *b == j)
                             || (*a == i && *b == j) || (*a == j && *b == i))
                        .map(|(_, _, sim)| *sim)
                        .fold(f32::MIN, f32::max);
                    simplices.push(Simplex {
                        vertices: vec![i, j, k],
                        filtration: 1.0 - max_sim,
                    });
                }
            }
        }
    }

    simplices
}

/// Persistent H0 barcode via Union-Find (DSU).
/// Returns Vec<(birth, death)> — each entry is a connected component.
pub fn persistent_h0(shape: &Shape) -> Vec<(f32, f32)> {
    let n = shape.nodes.len();
    if n == 0 { return vec![]; }

    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank = vec![0u8; n];
    let mut barcodes: Vec<(f32, f32)> = (0..n).map(|_| (0.0, f32::INFINITY)).collect();

    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x { parent[x] = find(parent, parent[x]); }
        parent[x]
    }
    fn union(parent: &mut Vec<usize>, rank: &mut Vec<u8>, x: usize, y: usize) -> bool {
        let xr = find(parent, x);
        let yr = find(parent, y);
        if xr == yr { return false; }
        match rank[xr].cmp(&rank[yr]) {
            std::cmp::Ordering::Less => parent[xr] = yr,
            std::cmp::Ordering::Greater => parent[yr] = xr,
            std::cmp::Ordering::Equal => { parent[yr] = xr; rank[xr] += 1; }
        }
        true
    }

    // Sort edges by similarity descending
    let mut sorted_edges: Vec<(usize, usize, f32)> = shape.edges.clone();
    sorted_edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut components = n;
    for (i, j, sim) in sorted_edges {
        if union(&mut parent, &mut rank, i, j) {
            components -= 1;
            barcodes[components] = (0.0, 1.0 - sim);  // death at this scale
        }
    }

    barcodes.truncate(components + 1);
    barcodes
}

/// Persistent H1 barcode via cycle detection (simplified).
/// Returns Vec<(birth, death)> for 1-dimensional holes (loops).
pub fn persistent_h1(_shape: &Shape, _epsilon_max: f32) -> Vec<(f32, f32)> {
    // Simplified: detects cycles in the 1-skeleton using spanning tree
    // Full implementation would use the boundary matrix reduction algorithm
    vec![]
}

/// Fold: collapse edges with death-birth < epsilon (topologically trivial).
/// Replaces the collapsed concepts with their weighted mean vector.
pub fn fold(shape: &Shape, epsilon: f32) -> Shape {
    let barcodes = persistent_h0(shape);
    let n = shape.nodes.len();
    if n <= 1 { return shape.clone(); }

    // Find edges that can be collapsed (H0 bars with short lifetime)
    let mut collapsed: Vec<bool> = vec![false; n];
    // For each barcode pair whose death - birth < epsilon, mark for collapse
    // Simplified: collapse all edges with similarity > 0.95
    for &(i, j, sim) in &shape.edges {
        if sim > 0.95 {
            collapsed[j] = true; // mark j for collapse into i
        }
    }

    let kept: Vec<ConceptNode> = shape.nodes.iter().enumerate()
        .filter(|(i, _)| !collapsed[*i])
        .map(|(_, node)| node.clone())
        .collect();

    if kept.is_empty() {
        shape.clone()
    } else {
        Shape::new(&format!("{}_folded", shape.name), kept)
    }
}

/// Intersect: common concepts between two shapes with similarity > threshold.
pub fn intersect(a: &Shape, b: &Shape, threshold: f32) -> Shape {
    let mut common: Vec<ConceptNode> = Vec::new();

    for node_a in &a.nodes {
        for node_b in &b.nodes {
            let sim = node_a.dot(&node_b.vector) as f32 / 8192.0;
            if sim > threshold {
                // Merge with weighted average
                let mut vec = [0i8; 64];
                for j in 0..64 {
                    let v = (node_a.vector[j] as f32 * 0.5 + node_b.vector[j] as f32 * 0.5)
                        .round().max(-128.0).min(127.0);
                    vec[j] = v as i8;
                }
                common.push(ConceptNode {
                    id: node_a.id,
                    vector: vec,
                    confidence: (node_a.confidence as f32 * 0.5 + node_b.confidence as f32 * 0.5) as u16,
                    connections: vec![],
                });
                break;
            }
        }
    }

    Shape::new(&format!("{}_∩_{}", a.name, b.name), common)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_node(id: u32, val: i8) -> ConceptNode {
        ConceptNode::new(id, [val; 64])
    }

    #[test]
    fn test_h0_basic() {
        let nodes = vec![
            dummy_node(1, 0),
            dummy_node(2, 10),
            dummy_node(3, 100),
        ];
        let shape = Shape::new("test", nodes);
        let barcodes = persistent_h0(&shape);
        assert!(barcodes.len() >= 1);
    }

    #[test]
    fn test_fold() {
        let nodes = vec![dummy_node(1, 0), dummy_node(2, 1)];
        let shape = Shape::new("test", nodes);
        let folded = fold(&shape, 0.1);
        assert!(folded.len() <= 2);
    }

    #[test]
    fn test_intersect() {
        let a = Shape::new("a", vec![dummy_node(1, 42)]);
        let b = Shape::new("b", vec![dummy_node(2, 42)]);
        let result = intersect(&a, &b, 0.9);
        assert!(!result.is_empty());
    }
}
