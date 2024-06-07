use core::fmt;
use std::{io, ops::Deref};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    backend::Backend,
    op::{BinaryOpDiscriminants, NoopDiscriminants, UnaryOpDiscriminants},
    ComputeGraph, Layout, Optimizer, TensorId, TensorSrc,
};

// Prints this many bytes of the mismatch.
const DATA_PRINT_LEN: usize = 2 * 16;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeInfo {
    /// This is only set for infernal tensors.
    pub id: Option<TensorId>,
    pub name: String,
    pub op: OpSimple,
    pub layout: Layout,
}

impl PartialEq for NodeInfo {
    fn eq(&self, other: &Self) -> bool {
        let id_eq = if let (Some(id), Some(other_id)) = (self.id, other.id) {
            id == other_id
        } else {
            true
        };
        // GGML's graph algo automatically sets these names, we don't, so we don't want to trip up
        // on these.
        let name_eq = if self.name.starts_with("leaf_") || other.name.starts_with("leaf_") {
            true
        } else {
            self.name.to_lowercase() == other.name.to_lowercase()
        };
        id_eq && name_eq && self.op == other.op && self.layout == other.layout
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Node {
    pub info: NodeInfo,
    pub parents: Vec<ParentNode>,
}

impl Deref for Node {
    type Target = NodeInfo;
    fn deref(&self) -> &Self::Target {
        &self.info
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParentNode {
    pub idx: ParentIdx,
    pub info: NodeInfo,
    // Only set for mmap'd weights.
    pub data: Option<WeightInfo>,
}

impl Deref for ParentNode {
    type Target = NodeInfo;
    fn deref(&self) -> &Self::Target {
        &self.info
    }
}

impl PartialEq for ParentNode {
    fn eq(&self, other: &Self) -> bool {
        self.info == other.info
        // TODO: see note in compare about data comparison
        // && self.data == other.data
        // TODO: compare idx too?
        // comparing idx means we're tying ourselves to the ggml representation which is not
        // necessarily ideal
        /*&& self.idx == other.idx*/
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ParentIdx {
    Node(usize),
    Leaf(usize),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WeightInfo {
    pub all_zero: bool,
    pub hash: blake3::Hash,
    pub mmap_offs: usize,
}

impl WeightInfo {
    pub fn new(data: &[u8], mmap_offs: Option<usize>) -> Option<Self> {
        mmap_offs.map(|mmap_offs| Self {
            all_zero: data.iter().all(|&x| x == 0),
            hash: blake3::hash(data),
            mmap_offs,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum OpSimple {
    Const,
    // NB: This is normally part of UnaryOp but easier to represent this way.
    Noop(NoopDiscriminants),
    Unary(UnaryOpDiscriminants),
    Binary(BinaryOpDiscriminants),
}

impl From<crate::Op> for OpSimple {
    fn from(op: crate::Op) -> Self {
        match op {
            crate::Op::Const => Self::Const,
            crate::Op::Unary { op, .. } => match op {
                crate::op::UnaryOp::Noop(n) => Self::Noop(n.into()),
                op => Self::Unary(op.into()),
            },
            crate::Op::Binary { op, .. } => Self::Binary(op.into()),
        }
    }
}

impl fmt::Display for OpSimple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Const => write!(f, "Const"),
            Self::Noop(n) => write!(f, "Noop({})", n),
            Self::Unary(n) => write!(f, "Unary({})", n),
            Self::Binary(n) => write!(f, "Binary({})", n),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct GraphHeader {
    pub n_nodes: usize,
    pub n_leaf_nodes: usize,
}

pub fn export_debug(graph: &impl Graph, writer: &mut impl io::Write) -> Result<(), ExportError> {
    if graph.n_nodes() == 0 {
        return Err(ExportError::EmptyGraph);
    }

    // This is a bit weird but we're serializing self not as a JSON object but as distinct values,
    // to enable streaming deserialization (would not be possible to deserialize individual struct
    // fields from a stream, at least AFAIK). This refers to the nodes list as well: it's not
    // a JSON list but instead JSON objects each representing a Node placed next to each other
    // in the file.
    let mut ser = serde_json::Serializer::new(writer);
    GraphHeader {
        n_nodes: graph.n_nodes(),
        n_leaf_nodes: graph.n_leaf_nodes(),
    }
    .serialize(&mut ser)?;
    for i in 0..graph.n_nodes() {
        let node = graph
            .get_node(i)
            .ok_or_else(|| ExportError::NodeNotFound(i))?;
        // TODO: look for data comments in `compare`
        // if node.data.is_empty() {
        //     return Err(ExportError::EmptyNodeData);
        // }
        node.serialize(&mut ser)?;
    }

    Ok(())
}

#[derive(Debug, Error)]
pub enum ExportError {
    #[error("graph is empty")]
    EmptyGraph,
    #[error("node data is empty")]
    EmptyNodeData,
    #[error("node {0:?} not found")]
    NodeNotFound(usize),
    #[error("error serializing node: {0}")]
    Encode(#[from] serde_json::Error),
}

pub trait Graph {
    fn n_nodes(&self) -> usize;
    fn n_leaf_nodes(&self) -> usize;
    fn get_node(&self, idx: usize) -> Option<Node>;
}

pub struct InfernalGraph<'a, O: Optimizer, S: TensorSrc> {
    graph: &'a ComputeGraph<O>,
    src: &'a S,
    backend: &'a crate::backend::Metal,
}

impl<'a, O: Optimizer, S: TensorSrc> InfernalGraph<'a, O, S> {
    pub fn new(graph: &'a ComputeGraph<O>, src: &'a S, backend: &'a crate::backend::Metal) -> Self {
        Self {
            graph,
            src,
            backend,
        }
    }
}

impl<'a, O: Optimizer, S: TensorSrc> Graph for InfernalGraph<'a, O, S> {
    fn n_nodes(&self) -> usize {
        self.graph.nodes().len()
    }

    fn n_leaf_nodes(&self) -> usize {
        self.graph.leaf_nodes.len()
    }

    fn get_node(&self, idx: usize) -> Option<Node> {
        let id = self.graph.nodes()[idx];
        let tensor = self.src.get_tensor(id).ok()?;

        // Collect parent nodes.
        let parents = tensor
            .op()
            .srcs()
            .iter()
            .map(|&src_id| {
                // Find parent in graph nodes or leaf_nodes lists. This is O(n) because we don't
                // have a mapping from tensor id to graph node index.
                // TODO(perf): construct a lookup map at the start
                let idx = if let Some(idx) = self.graph.nodes().iter().position(|&id| id == src_id)
                {
                    ParentIdx::Node(idx)
                } else {
                    // This is a leaf node, it must be present.
                    let idx = self
                        .graph
                        .leaf_nodes()
                        .iter()
                        .position(|&n| n == src_id)
                        .expect("node parent not in graph");
                    ParentIdx::Leaf(idx)
                };
                let src = self.src.get_tensor(src_id).unwrap();
                let buf = self
                    .backend
                    .get_tensor_buf(src)
                    .expect("parent buffer not present in backend")
                    .borrow();
                let data = buf.tensor_data(src);
                ParentNode {
                    idx,
                    info: NodeInfo {
                        id: Some(src.id()),
                        name: src.name().to_string(),
                        op: (*src.op()).into(),
                        layout: *src.layout(),
                    },
                    data: WeightInfo::new(
                        data,
                        if src.name().ends_with(".weight") {
                            Some(src.data_offs())
                        } else {
                            None
                        },
                    ),
                }
            })
            .collect();

        Some(Node {
            info: NodeInfo {
                id: Some(tensor.id()),
                name: tensor.name().to_string(),
                op: (*tensor.op()).into(),
                layout: *tensor.layout(),
            },
            parents,
        })
    }
}

impl crate::Op {
    #[inline(always)]
    fn srcs(&self) -> Vec<TensorId> {
        match self {
            Self::Const => Vec::new(),
            Self::Unary { a, .. } => {
                vec![*a]
            }
            Self::Binary { a, b, .. } => {
                vec![*a, *b]
            }
        }
    }
}

impl fmt::Display for GraphHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "nodes={n_nodes} leaf_nodes={n_leaf_nodes}",
            n_nodes = self.n_nodes,
            n_leaf_nodes = self.n_leaf_nodes
        )
    }
}

#[derive(Debug, Error)]
#[error("error decoding '{src}' graph: {err}")]
pub struct DecodeError {
    src: &'static str,
    err: serde_json::Error,
}

pub fn compare(
    actual_header: GraphHeader,
    mut actual_nodes: impl Iterator<Item = Result<Node, serde_json::Error>>,
    expected_header: GraphHeader,
    mut expected_nodes: impl Iterator<Item = Result<Node, serde_json::Error>>,
    ok_handler: impl Fn(&Node, usize),
) -> Result<Comparison, DecodeError> {
    let mut idx = 0;
    loop {
        let (actual, expected) = match (actual_nodes.next(), expected_nodes.next()) {
            (Some(a), Some(e)) => (a, e),
            (None, None) => break,
            _ => {
                return Ok(Comparison::GraphMismatch {
                    len: Mismatch {
                        actual: actual_header,
                        expected: expected_header,
                    },
                    at_node: idx,
                })
            }
        };

        let actual = actual.map_err(|err| DecodeError { src: "actual", err })?;
        let expected = expected.map_err(|err| DecodeError {
            src: "expected",
            err,
        })?;

        if actual.info != expected.info || actual.parents != expected.parents {
            // Return the first mismatch. Later we may print the first N mismatches.
            return Ok(Comparison::TensorMismatch {
                node_idx: idx,
                actual,
                expected,
            });
        }

        ok_handler(&expected, idx);

        idx += 1;
    }

    Ok(Comparison::Ok)
}

#[allow(clippy::large_enum_variant)]
pub enum Comparison {
    Ok,
    TensorMismatch {
        node_idx: usize,
        actual: Node,
        expected: Node,
    },
    GraphMismatch {
        len: Mismatch<GraphHeader>,
        at_node: usize,
    },
}

impl Comparison {
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Ok)
    }

    /// Gives a recommendation for why the tensors mismatch based on a heuristic approach, meaning
    /// it can be wrong but it may provide a good starting point.
    pub fn mismatch_reason(&self) -> Option<MismatchLikelyReason> {
        if let Self::TensorMismatch {
            actual, expected, ..
        } = self
        {
            if actual.op != expected.op {
                return Some(MismatchLikelyReason::DifferentOps);
            } else if actual.layout != expected.layout {
                if actual.layout.size() == expected.layout.size() {
                    // TODO: should this use stride?
                    if actual.layout.dims[0] == expected.layout.dims[1]
                        && actual.layout.dims[1] == expected.layout.dims[0]
                    {
                        if actual.layout.is_transposed() {
                            return Some(MismatchLikelyReason::Transposed);
                        } else {
                            return Some(MismatchLikelyReason::NotTransposed);
                        }
                    } else if matches!(actual.op, OpSimple::Noop(NoopDiscriminants::Reshape)) {
                        return Some(MismatchLikelyReason::InvalidReshape);
                    }
                } else if matches!(actual.op, OpSimple::Noop(NoopDiscriminants::View)) {
                    return Some(MismatchLikelyReason::InvalidView);
                }
            } else if actual.name != expected.name {
                return Some(MismatchLikelyReason::DifferentNames);
            }
        }

        None
    }
}

// For what each of these mean, look at the Display impl.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MismatchLikelyReason {
    Transposed,
    NotTransposed,
    InvalidReshape,
    InvalidView,
    InvalidResult,
    DifferentInputData,
    DifferentOps,
    DifferentNames,
}

impl fmt::Display for MismatchLikelyReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transposed => write!(f, "The tensor is transposed but should not be."),
            Self::NotTransposed => write!(f, "The tensor is not transposed but should be."),
            Self::InvalidReshape => write!(f, "This tensor is the result of a reshape operation but its layout differs. The reshape parameters are likely invalid."),
            Self::InvalidView => write!(f, "This tensor is the result of a view operation but its layout differs. The view parameters are likely invalid."),
            Self::DifferentOps => write!(f, "The tensors have different operations."),
            Self::InvalidResult => write!(f, "Everything checks out except the resulting values. This may be a bug in the op binding or kernel implementation."),
            Self::DifferentNames => write!(f, "The tensor names differ."),
            Self::DifferentInputData => write!(f, "One or more of the node's inputs differ."),
        }
    }
}

impl fmt::Display for Comparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let reason = self.mismatch_reason();
        match self {
            Self::Ok => writeln!(f, "OK – graphs match"),
            Self::TensorMismatch {
                node_idx,
                actual,
                expected,
            } => {
                let parents = expected
                    .parents
                    .iter()
                    .map(|p| p.name.as_str())
                    .collect::<Vec<_>>()
                    .join(",");
                let typ = expected.layout.typ();
                let op = expected.op.clone();
                let meta = if let Some(id) = actual.id {
                    format!("(t={typ:?} op={op} src=[{parents}] id={id})")
                } else {
                    format!("(t={typ:?} op={op} src=[{parents}])")
                };
                writeln!(
                    f,
                    "ERROR: Tensor '{name}' {meta} mismatch at node={node_idx}, reasons (in order of priority):",
                    name = expected.name
                )?;

                let mut reason_idx = 0;

                let parents: Vec<_> = actual
                    .parents
                    .iter()
                    .zip(expected.parents.iter())
                    .filter_map(|(a, e)| {
                        if a != e {
                            Some(Mismatch {
                                actual: a.clone(),
                                expected: e.clone(),
                            })
                        } else {
                            None
                        }
                    })
                    .collect();
                if !parents.is_empty() {
                    for parent in &parents {
                        reason_idx += 1;
                        // NB: Parent prints a new line after itself.
                        write!(f, "{reason_idx}. {parent}")?;
                    }
                }

                if actual.op != expected.op {
                    reason_idx += 1;
                    writeln!(
                        f,
                        "{reason_idx}. Operation: {got} != {exp}",
                        got = actual.op,
                        exp = expected.op
                    )?;
                }

                if actual.layout != expected.layout {
                    reason_idx += 1;
                    writeln!(
                        f,
                        "{reason_idx}. Layout: {got} != {exp}",
                        got = actual.layout,
                        exp = expected.layout
                    )?;
                }

                if actual.name != expected.name {
                    reason_idx += 1;
                    writeln!(
                        f,
                        "{reason_idx}. Name: '{got}' (n={got_len}) != '{exp}' (n={exp_len})",
                        got = actual.name,
                        got_len = actual.name.len(),
                        exp = expected.name,
                        exp_len = expected.name.len()
                    )?;
                }

                // Give heuristic advice for debugging.
                if let Some(reason) = reason {
                    writeln!(f)?;
                    writeln!(f, "GUESSED CAUSE: {reason}")?;
                }

                if reason_idx > 1 {
                    writeln!(f)?;
                    writeln!(f, "SUGGESTION: Multiple mismatches present, it's likely that the first error is causing a cascade. Fix the first error and check again.")?;
                }

                writeln!(f)?;
                writeln!(
                    f,
                    "(All mismatches printed in format `actual != expected`.)"
                )
            }
            Self::GraphMismatch { len, at_node } => {
                writeln!(
                    f,
                    "graph mismatch: different number of nodes: {got} != {exp} (detected at node={at_node})",
                    got = len.actual,
                    exp = len.expected
                )
            }
        }
    }
}

/// Utility for printing a mismatch.
pub struct Mismatch<T> {
    pub expected: T,
    pub actual: T,
}

impl<T: fmt::Display + PartialEq> fmt::Display for Mismatch<&'_ [T]> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // FIXME: handle case of actual.len() != expected.len() (e.g. when one is empty)

        let mut len = 0;
        for (a, e) in self.actual.iter().zip(self.expected.iter()) {
            if a != e {
                len += 1;
            }
        }

        debug_assert!(len > 0);
        writeln!(
            f,
            "{len}/{total} bytes differ, printing at most {DATA_PRINT_LEN} mismatches: ",
            total = self.expected.len()
        )?;

        let mut n_printed = 0;
        let mut prev_idx = None;
        for (i, (got, exp)) in self.actual.iter().zip(self.expected.iter()).enumerate() {
            if got != exp && n_printed < DATA_PRINT_LEN {
                if let Some(p) = prev_idx {
                    if p + 1 != i {
                        writeln!(f, "\t...")?;
                    }
                }
                writeln!(f, "   ┗━ i={i}: 0x{got} != 0x{exp}")?;
                n_printed += 1;
                prev_idx = Some(i);
            }
        }

        Ok(())
    }
}

impl fmt::Display for Mismatch<ParentNode> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Parent '{name}' mismatch{id}:",
            name = self.expected.name,
            id = if let Some(id) = self.actual.id {
                format!(" (id={id})")
            } else {
                String::new()
            },
        )?;

        if self.actual.idx != self.expected.idx {
            writeln!(
                f,
                "   ┗━ Node:   {got:?} != {exp:?}",
                got = self.actual.idx,
                exp = self.expected.idx
            )?;
        }

        if self.actual.name != self.expected.name {
            writeln!(
                f,
                "   ┗━ Name:   '{got}' != '{exp}'",
                got = self.actual.name,
                exp = self.expected.name
            )?;
        }

        if self.actual.op != self.expected.op {
            writeln!(
                f,
                "   ┗━ Op.:    {got} != {exp}",
                got = self.actual.op,
                exp = self.expected.op
            )?;
        }

        if self.actual.layout != self.expected.layout {
            writeln!(
                f,
                "   ┗━ Layout: {got} != {exp}",
                got = self.actual.layout,
                exp = self.expected.layout
            )?;
        }

        if self.actual.data != self.expected.data {
            if let (Some(actual), Some(expected)) = (&self.actual.data, &self.expected.data) {
                let data = Mismatch { actual, expected };
                write!(f, "   ┗━ {data}")?;
            } else if self.actual.data.is_some() {
                write!(
                    f,
                    "   ┗━ Got parent weight node but expected intermediate parent node"
                )?;
            } else {
                write!(
                    f,
                    "   ┗━ Expected parent weight node but got intermediate parent node"
                )?;
            }
        }

        Ok(())
    }
}

impl fmt::Display for Mismatch<&'_ WeightInfo> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Data: {got} != {exp} (blake3+hex)",
            got = self.actual.hash.to_hex(),
            exp = self.expected.hash.to_hex(),
        )?;

        if self.actual.all_zero != self.expected.all_zero {
            writeln!(
                f,
                "      ┗━ All zeros: {got} != {exp}",
                got = self.actual.all_zero,
                exp = self.expected.all_zero
            )?;
        }

        if self.actual.mmap_offs != self.expected.mmap_offs {
            writeln!(
                f,
                "      ┗━ Mmap offset: {got:?} != {exp:?}",
                got = self.actual.mmap_offs,
                exp = self.expected.mmap_offs
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::Type::*;

    use super::*;

    #[test]
    fn name_mismatch() {
        let mm = MismatchBuilder::default()
            .set_name(|e| format!("{}-fuckered", e))
            .finish();
        println!("{mm}");
        assert_eq!(
            mm.mismatch_reason(),
            Some(MismatchLikelyReason::DifferentNames)
        );
    }

    #[test]
    fn op_mismatch() {
        let mm = MismatchBuilder::default()
            .set_op(|_| OpSimple::Unary(UnaryOpDiscriminants::Rope))
            .finish();
        println!("{mm}");
        assert_eq!(
            mm.mismatch_reason(),
            Some(MismatchLikelyReason::DifferentOps)
        );
    }

    #[test]
    fn parents_mismatch() {
        let mm = MismatchBuilder::default()
            .set_parents(|e| {
                let mut a = e.to_vec();
                a[0].idx = ParentIdx::Node(999);
                a[0].info.name += "-fuckered";
                a[0].info.op = OpSimple::Binary(BinaryOpDiscriminants::Add);
                a[0].info.layout = Layout::new_1d(1, F32);
                a[0].data = WeightInfo::new(&[0, 0, 0, 0, 0], Some(999));
                a
            })
            .finish();
        println!("{mm}");
        assert_eq!(
            mm.mismatch_reason(),
            Some(MismatchLikelyReason::DifferentInputData)
        );
    }

    #[test]
    fn transposed_mismatch() {
        let mm = MismatchBuilder::default()
            .set_layout(|e| {
                // manual transpose, whatever
                let mut l = *e;

                let mut dims = l.dims;
                dims[0] = l.dims[1];
                dims[1] = l.dims[0];
                l.dims = dims;

                let mut stride = l.stride;
                stride[0] = l.stride[1];
                stride[1] = l.stride[0];
                l.stride = stride;

                l
            })
            .finish();
        println!("{mm}");
        assert_eq!(mm.mismatch_reason(), Some(MismatchLikelyReason::Transposed));
    }

    #[test]
    fn all_mismatches() {
        let mm = MismatchBuilder::default()
            .set_name(|e| format!("{}-fuckered", e))
            .set_op(|_| OpSimple::Unary(UnaryOpDiscriminants::Rope))
            .set_parents(|e| {
                let mut a = e.to_vec();

                a[0].idx = ParentIdx::Node(999);
                a[0].info.name += "-fuckered";
                a[0].info.op = OpSimple::Binary(BinaryOpDiscriminants::Add);
                a[0].info.layout = Layout::new_1d(1, F32);
                a[0].data = WeightInfo::new(&[0, 0, 0, 0, 0], None);

                a[1] = a[0].clone();

                a
            })
            .set_layout(|e| {
                // manual transpose, whatever
                let mut l = *e;

                let mut dims = l.dims;
                dims[0] = l.dims[1];
                dims[1] = l.dims[0];
                l.dims = dims;

                let mut stride = l.stride;
                stride[0] = l.stride[1];
                stride[1] = l.stride[0];
                l.stride = stride;

                l
            })
            .finish();
        println!("{mm}");
        assert_eq!(
            mm.mismatch_reason(),
            Some(MismatchLikelyReason::DifferentInputData)
        );
    }

    struct MismatchBuilder {
        cmp: Comparison,
        differs: bool,
    }

    impl Default for MismatchBuilder {
        /// By default everything matches, so something must be updated.
        fn default() -> Self {
            let mm = Comparison::TensorMismatch {
                node_idx: 2,
                actual: Node {
                    info: NodeInfo {
                        id: Some(TensorId::new(0, 2)),
                        name: "matmul".into(),
                        op: crate::op::Op::Binary {
                            op: crate::op::BinaryOp::Matmul,
                            a: TensorId::new(0, 0),
                            b: TensorId::new(0, 1),
                        }
                        .into(),
                        layout: Layout::new_2d(8, 10, F32),
                    },
                    parents: vec![
                        ParentNode {
                            idx: ParentIdx::Leaf(0),
                            info: NodeInfo {
                                name: "weight0".into(),
                                id: Some(TensorId::new(0, 0)),
                                op: OpSimple::Const,
                                layout: Layout::new_2d(8, 5, F32),
                            },
                            data: WeightInfo::new(&[0, 1, 2, 3, 4, 5, 6, 7], Some(0)),
                        },
                        ParentNode {
                            idx: ParentIdx::Leaf(1),
                            info: NodeInfo {
                                name: "weight1".into(),
                                id: Some(TensorId::new(0, 1)),
                                op: OpSimple::Const,
                                layout: Layout::new_2d(10, 5, F32),
                            },
                            data: WeightInfo::new(&[0, 1, 2, 3, 4, 5, 6, 7], Some(8)),
                        },
                    ],
                },
                expected: Node {
                    info: NodeInfo {
                        id: None,
                        name: "matmul".into(),
                        op: OpSimple::Binary(BinaryOpDiscriminants::Matmul),
                        layout: Layout::new_2d(8, 10, F32),
                    },
                    parents: vec![
                        ParentNode {
                            idx: ParentIdx::Leaf(0),
                            info: NodeInfo {
                                name: "weight0".into(),
                                id: Some(TensorId::new(0, 0)),
                                op: OpSimple::Const,
                                layout: Layout::new_2d(8, 5, F32),
                            },
                            data: WeightInfo::new(&[0, 1, 2, 3, 4, 5, 6, 7], Some(0)),
                        },
                        ParentNode {
                            idx: ParentIdx::Leaf(1),
                            info: NodeInfo {
                                name: "weight1".into(),
                                id: Some(TensorId::new(0, 1)),
                                op: OpSimple::Const,
                                layout: Layout::new_2d(10, 5, F32),
                            },
                            data: WeightInfo::new(&[0, 1, 2, 3, 4, 5, 6, 7], Some(8)),
                        },
                    ],
                },
            };
            Self {
                cmp: mm,
                differs: false,
            }
        }
    }

    impl MismatchBuilder {
        fn set_name(mut self, f: impl FnOnce(&str) -> String) -> Self {
            if let Comparison::TensorMismatch {
                actual, expected, ..
            } = &mut self.cmp
            {
                actual.info.name = f(&expected.name).into();
                if !self.differs {
                    self.differs = actual.name != expected.name;
                }
            }
            self
        }

        fn set_op(mut self, f: impl FnOnce(&OpSimple) -> OpSimple) -> Self {
            if let Comparison::TensorMismatch {
                actual, expected, ..
            } = &mut self.cmp
            {
                actual.info.op = f(&expected.op);
                if !self.differs {
                    self.differs = actual.op != expected.op;
                }
            }
            self
        }

        fn set_layout(mut self, f: impl FnOnce(&Layout) -> Layout) -> Self {
            if let Comparison::TensorMismatch {
                actual, expected, ..
            } = &mut self.cmp
            {
                actual.info.layout = f(&expected.layout);
                let differs = actual.layout != expected.layout;
                if !self.differs {
                    self.differs = differs;
                }
            }
            self
        }

        fn set_parents(mut self, f: impl FnOnce(&[ParentNode]) -> Vec<ParentNode>) -> Self {
            if let Comparison::TensorMismatch {
                actual, expected, ..
            } = &mut self.cmp
            {
                actual.parents = f(&expected.parents);
                if !self.differs {
                    self.differs = actual.parents != expected.parents;
                }
            }
            self
        }

        fn finish(self) -> Comparison {
            assert!(self.differs);
            self.cmp
        }
    }

    // #[test]
    // fn ser() {
    //     let node = Node {
    //         id: Some(TensorId::new(0, 0)),
    //         name: "foo".into(),
    //         op: OpSimple::Binary(BinaryOpDiscriminants::Matmul),
    //         layout: Layout::new(&[1, 2, 3, 4], F32),
    //         parents: vec![
    //             ParentNode {
    //                 idx: ParentIdx::Leaf(0),
    //                 id: Some(TensorId::new(0, 0)),
    //                 name: "foo".into(),
    //                 op: OpSimple::Const,
    //                 layout: Layout::new(&[1, 2, 3, 4], F32),
    //                 data: ParentData::new(&[0, 0, 0, 0, 0]),
    //             },
    //             ParentNode {
    //                 idx: ParentIdx::Leaf(1),
    //                 id: Some(TensorId::new(0, 1)),
    //                 name: "bar".into(),
    //                 op: OpSimple::Const,
    //                 layout: Layout::new(&[1, 2, 3, 4], F32),
    //                 data: ParentData::new(&[0, 0, 0, 0, 0]),
    //             },
    //         ],
    //         data: vec![0, 1, 2, 3],
    //     };
    //     // let res = rmp_serde::to_vec(&node).unwrap();
    //     // println!("{:?}", res);
    //     // let res: Node = rmp_serde::from_slice(&res).unwrap();
    //     // println!("{:?}", res);

    //     let mut buf = Vec::new();
    //     let mut ser = rmp_serde::Serializer::new(&mut buf);

    //     ser.serialize_seq(Some(2)).unwrap();
    //     node.serialize(&mut ser).unwrap();
    //     node.serialize(&mut ser).unwrap();

    //     println!("{:?}", buf);

    //     let res: Vec<Node> = rmp_serde::from_slice(&buf).unwrap();
    //     println!("{:#?}", res);
    // }
}
