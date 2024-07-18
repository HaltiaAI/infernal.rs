use std::collections::HashSet;

use crate::{Op, TensorId, TensorNotFound, TensorSrc};

pub use self::opt::{ConcurOptimizer, ExecPlan, Optimizer};

#[cfg(feature = "graph-inspect")]
pub mod inspect;
pub mod opt;

/// The compute graph contains the order of execution. Each tensor operation is an implicit graph
/// consisting of the operation itself and its parents (source tensors). E.g. multiplying 2 tensors
/// is a graph of 2 leaves (the 2 source tensors) and 1 root (the multiplication operation). This
/// type materializes this relationship by walking the graph backwards from the root to the leaves.
///
/// This structure should be reused to save allocations and reuse existing memory if the graph
/// layout does not change. If it does, you need to call `ComputeGraph::clear` after each new type
/// of graph.
pub struct ComputeGraph<O: Optimizer = ConcurOptimizer> {
    /// Leaf nodes are the input nodes into the compute graph.
    ///
    /// E.g. when multiplying 2 tensors to get a third, the 2 input tensors (that have operation
    /// "none") are the leaves and the resulting tensor is a normal node.
    leaf_nodes: Vec<TensorId>,
    /// All nodes that are not leaf nodes. These nodes are lazy, i.e. don't have any data and are
    /// materialized once the graph is computed.
    nodes: Vec<TensorId>,
    // TODO remove this
    visited: HashSet<TensorId>,
    exec_plan: ExecPlan,
    optimizer: O,
    output: Option<TensorId>,
}

impl Default for ComputeGraph<ConcurOptimizer> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            leaf_nodes: Vec::new(),
            nodes: Vec::new(),
            visited: HashSet::new(),
            exec_plan: ExecPlan::default(),
            optimizer: ConcurOptimizer::default(),
            output: None,
        }
    }
}

impl ComputeGraph<ConcurOptimizer> {
    pub fn forward_expand(
        mem_limit: usize,
        output: TensorId,
        ctx: &impl TensorSrc
    ) -> Result<Self, TensorNotFound> {
        let mut this = ComputeGraph::default();

        this.expand_forward(output, ctx)?;
        this.find_exec_plan(mem_limit, ctx)?;

        Ok(this)
    }
}

impl<O: Optimizer> ComputeGraph<O> {
    #[inline(always)]
    pub fn reuse(mut other: Self) -> Self {
        other.clear();
        other.reset_optimizations();
        ComputeGraph {
            leaf_nodes: other.leaf_nodes,
            nodes: other.nodes,
            visited: other.visited,
            exec_plan: other.exec_plan,
            optimizer: other.optimizer,
            output: None,
        }
    }

    /// The last result node on which the graph was expanded.
    ///
    /// This is just a helper to retrieve the graph's output for a complex operation, which you can
    /// also do by manualy saving the result tensor id.
    #[inline(always)]
    pub fn output(&self) -> Option<TensorId> {
        self.output
    }

    #[inline(always)]
    pub fn expand_forward(
        &mut self,
        output: TensorId,
        ctx: &impl TensorSrc,
    ) -> Result<(), TensorNotFound> {
        self.visit_node_parents(output, ctx)?;
        debug_assert_eq!(
            self.nodes.last(),
            Some(&output),
            "result node must be the last node in the graph"
        );
        self.output = Some(output);
        Ok(())
    }

    fn visit_node_parents(
        &mut self,
        t: TensorId,
        ctx: &impl TensorSrc,
    ) -> Result<(), TensorNotFound> {
        let visited = !self.visited.insert(t);
        if visited {
            return Ok(());
        }

        // TODO#perf: parallelize for large graphs (e.g. optional hint in params). probably need to
        // make the algorithm iterative too.
        match ctx.get_tensor(t)?.op {
            Op::Const => {
                // Reached a starting point in the graph.
                self.leaf_nodes.push(t);
                return Ok(());
            }
            Op::Unary { a, .. } => {
                self.visit_node_parents(a, ctx)?;
            }
            Op::Binary { a: x, b: y, .. } => {
                self.visit_node_parents(x, ctx)?;
                self.visit_node_parents(y, ctx)?;
            }
        }

        self.nodes.push(t);

        Ok(())
    }

    #[inline(always)]
    pub fn leaf_nodes(&self) -> &[TensorId] {
        &self.leaf_nodes
    }

    #[inline(always)]
    pub fn nodes(&self) -> &[TensorId] {
        &self.nodes
    }

    /// Clears the compute graph but keeps the underlying memory buffers for reuse.
    ///
    /// Optimization information is not cleared because the graph architecture (even if not the
    /// exact tensor dimensions) is assumed not to change between graph computations, such as between
    /// inference steps. To clear the optimization too, call [`Self::clear_optimization`].
    ///
    /// The tensor context is not cleared either, because it's usually a reference we have to it.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.leaf_nodes.clear();
        self.nodes.clear();
        self.visited.clear();
    }

    #[inline(always)]
    pub fn reset_optimizations(&mut self) {
        self.exec_plan.levels.clear();
        self.exec_plan.max_level_size = 0;
        self.exec_plan.mem_soft_limit = 0;
    }

    // Finds the optimal execution plan for this graph. Examples of an optimal plan might involve
    // parallel paths or fused tensor operations.
    //
    // Generally you only need to call this once, when the graph is first built. Note that the
    // optimizer should be called on a _worst case_ graph. See more in [`ExecPlan::max_block_size`].
    //
    // If you want to re-optimize the graph, call [`Self::reset`] first. However, when your graph
    // layout or the maximum memory size changes, you need to call this method again. This is yhour
    // responsibility as raph layout changes won't be auto-detected by the backend as an
    // optimization.
    #[inline(always)]
    pub fn find_exec_plan(
        &mut self,
        mem_limit: usize,
        ctx: &impl TensorSrc,
    ) -> Result<(), TensorNotFound> {
        self.exec_plan = self.optimizer.find_exec_plan(ctx, &self.nodes, mem_limit)?;
        Ok(())
    }

    /// Returns the execution plan if the graph has been optimized for the given max memory use (at
    /// least as much). If not, you must call [`Self::optimize`] with this value first.
    #[inline(always)]
    pub fn exec_plan(&self, mem_limit: usize) -> Option<&ExecPlan> {
        if mem_limit >= self.exec_plan.mem_soft_limit {
            Some(&self.exec_plan)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{backend, builder::NoSrc, Layout, TensorBuilder, Type};

    use super::*;

    const BUF_SIZE: usize = 1024 * 1024;

    #[test]
    fn finds_graph_concurrency() -> anyhow::Result<()> {
        let layout = Layout::new_2d(12, 12, Type::F32);
        let mut backend = backend::Metal::new().unwrap();

        let mut tb = TensorBuilder::<_, NoSrc>::prealloc(BUF_SIZE)?.with_name("cgraph_concur");

        // branch 0
        let a1 = tb.new_tensor(layout)?;
        let b1 = tb.new_tensor(layout)?;
        tb.fill_2d_f32(a1, 1.3);
        tb.fill_2d_f32(b1, 9.5);
        let a1b1 = tb.matmul(a1, b1)?;

        // branch 1
        let a2 = tb.new_tensor(layout)?;
        let b2 = tb.new_tensor(layout)?;
        tb.fill_2d_f32(a2, 2.3);
        tb.fill_2d_f32(b2, 9.5);
        let a2b2 = tb.matmul(a2, b2)?;

        // result
        let res = tb.add(a1b1, a2b2)?;

        let (ctx, _) = tb.finish(&mut backend)?;

        let mut graph = ComputeGraph::default();

        graph.expand_forward(res, &ctx).unwrap();
        graph.find_exec_plan(BUF_SIZE, &ctx).unwrap();

        // Branches 0 and 1 should be in the first concurrency block, before the first barrier.
        assert_eq!(
            dbg!(graph.exec_plan(BUF_SIZE)).unwrap().levels,
            vec![
                opt::ConcurLevel::from_slice(&[0, 1]),
                opt::ConcurLevel::from_slice(&[2])
            ],
        );

        Ok(())
    }
}
