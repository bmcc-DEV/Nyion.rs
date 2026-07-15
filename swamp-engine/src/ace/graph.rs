use std::collections::HashMap;

// =========================================================================
// Core ACE data structures
// =========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationType {
    IsA,
    PartOf,
    Causes,
    Similar,
    Contradicts,
    Property,
    Instance,
    Evidence,
}

impl RelationType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::IsA => "is_a",
            Self::PartOf => "part_of",
            Self::Causes => "causes",
            Self::Similar => "similar",
            Self::Contradicts => "contradicts",
            Self::Property => "property",
            Self::Instance => "instance",
            Self::Evidence => "evidence",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConceptNode {
    pub id: u32,
    pub vector: [i8; 64],
    pub confidence: u16,
    pub connections: Vec<(u32, RelationType, f32)>,
}

impl ConceptNode {
    pub fn new(id: u32, vector: [i8; 64]) -> Self {
        Self { id, vector, confidence: u16::MAX / 2, connections: Vec::new() }
    }

    /// Similarity via dot product (calls fused_gemv_q4k under the hood)
    pub fn dot(&self, other: &[i8; 64]) -> i32 {
        let mut sum = 0i32;
        for j in 0..64 {
            sum += self.vector[j] as i32 * other[j] as i32;
        }
        sum
    }

    pub fn add_connection(&mut self, target: u32, rel: RelationType, weight: f32) {
        if let Some(existing) = self.connections.iter_mut().find(|(t, r, _)| *t == target && *r == rel) {
            existing.2 = existing.2 * 0.7 + weight * 0.3;
        } else {
            self.connections.push((target, rel, weight));
        }
    }

    pub fn connection_weight(&self, target: u32, rel: RelationType) -> f32 {
        self.connections.iter()
            .find(|(t, r, _)| *t == target && *r == rel)
            .map(|(_, _, w)| *w)
            .unwrap_or(0.0)
    }
}

#[derive(Debug, Clone)]
pub struct Shape {
    pub name: String,
    pub nodes: Vec<ConceptNode>,
    pub edges: Vec<(usize, usize, f32)>,
}

impl Shape {
    pub fn new(name: &str, nodes: Vec<ConceptNode>) -> Self {
        let edges = Self::compute_edges(&nodes, 0.5);
        Self { name: name.to_string(), nodes, edges }
    }

    pub fn empty(name: &str) -> Self {
        Self { name: name.to_string(), nodes: Vec::new(), edges: Vec::new() }
    }

    pub fn len(&self) -> usize { self.nodes.len() }
    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }

    /// Compute edges: connect nodes with dot product similarity > threshold
    pub fn compute_edges(nodes: &[ConceptNode], threshold: f32) -> Vec<(usize, usize, f32)> {
        let mut edges = Vec::new();
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let sim = nodes[i].dot(&nodes[j].vector) as f32 / 8192.0;
                if sim > threshold {
                    edges.push((i, j, sim));
                }
            }
        }
        edges
    }

    /// Recompute edges after mutations
    pub fn rebuild_edges(&mut self, threshold: f32) {
        self.edges = Self::compute_edges(&self.nodes, threshold);
    }

    pub fn node_ids(&self) -> Vec<u32> {
        self.nodes.iter().map(|n| n.id).collect()
    }
}

// =========================================================================
// GraphState — the dynamic belief graph
// =========================================================================

pub struct GraphState {
    pub nodes: HashMap<u32, ConceptNode>,
    next_id: u32,
    pub shape: Shape,
}

impl GraphState {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 1,
            shape: Shape::empty("belief_graph"),
        }
    }

    pub fn add_node(&mut self, vector: [i8; 64]) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        let node = ConceptNode::new(id, vector);
        self.nodes.insert(id, node);
        self.sync_shape();
        id
    }

    pub fn get(&self, id: u32) -> Option<&ConceptNode> {
        self.nodes.get(&id)
    }

    pub fn get_mut(&mut self, id: u32) -> Option<&mut ConceptNode> {
        self.nodes.get_mut(&id)
    }

    pub fn remove(&mut self, id: u32) {
        self.nodes.remove(&id);
        for node in self.nodes.values_mut() {
            node.connections.retain(|(t, _, _)| *t != id);
        }
        self.sync_shape();
    }

    pub fn len(&self) -> usize { self.nodes.len() }
    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }

    /// Find top-K most similar concepts to a query vector via dot product
    pub fn find_similar(&self, query: &[i8; 64], k: usize) -> Vec<(u32, f32)> {
        let mut scores: Vec<(u32, f32)> = self.nodes.iter()
            .map(|(id, node)| (*id, node.dot(query) as f32))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }

    /// Sync shape from node map
    fn sync_shape(&mut self) {
        let mut nodes: Vec<ConceptNode> = self.nodes.values().cloned().collect();
        nodes.sort_by_key(|n| n.id);
        let edges = Shape::compute_edges(&nodes, 0.3);
        self.shape = Shape { name: "belief_graph".to_string(), nodes, edges };
    }

    pub fn concept_count(&self) -> usize { self.nodes.len() }
}
