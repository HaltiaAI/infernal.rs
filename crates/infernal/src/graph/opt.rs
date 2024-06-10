use std::{collections::HashSet, mem};

use smallvec::SmallVec;

use crate::{
    op::{Op, UnaryOp},
    Tensor, TensorId, TensorNotFound, TensorSrc,
};

pub trait Optimizer: Default {
    fn find_exec_plan(
        &mut self,
        ctx: &impl TensorSrc,
        nodes: &[TensorId],
        mem_limit: usize,
    ) -> Result<ExecPlan, TensorNotFound>;
}

pub type NodeIdx = usize;
/// A level of nodes that can be executed concurrently. It's rare to have more than 3 parallel
/// branches in the usual LLM operations.
pub type ConcurLevel = SmallVec<[NodeIdx; 3]>;

#[derive(Debug, Default)]
pub struct ExecPlan {
    /// The executable levels of nodes.
    ///
    /// If a level has multiple nodes, those nodes can be executed in parallel. Each level must be
    /// followed by a barrier to establish a proper dependency graph, so that the next level is not
    /// executed until the current level is finished.
    ///
    /// Note that total level sizes are not counted as
    pub levels: Vec<ConcurLevel>,
    /// The maximum concurrency achieved by this plan.
    ///
    /// Note that this only calculates nodes that are actually executed by the backend (i.e. does
    /// not track noops and consts).
    pub max_level_nodes: usize,
    /// This is the largest level size in the execution plan and thus the backend needs to be able
    /// to accommodate this value to execute the graph.
    ///
    /// This sums the _unique_ tensor sizes within the level. That is, if hypothetically the same
    /// source tensor is used for all the parallel operations in this level, that source tensor is
    /// only counted once. Moreover, if a tensor operation is inplace and doesn't require creating new
    /// memory regions, that is not counted.
    ///
    /// Further, this doesn't actually count buffer sizes underlying these tensors, just simply the
    /// tensors themselves. That is because backends can selectively load only the portion of
    /// a buffer representing the tensor, so for approximations of memory use this is sufficient.
    ///
    /// Note that to make this practical, you need to build and execute a worst case graph once at
    /// startup so all future use cases can be accommodated. To achieve this, e.g. for an LLM,
    /// create a graph with the largest possible embedding size (usually the context window length),
    /// create the graph, call the graph optimizer, and save the value.
    pub max_level_size: usize,
    /// The maximum memory size that can be used for levels. This is the value that was used for the
    /// optimization.
    ///
    /// Note that this is a soft limit and in case the graph requires more memory to execute one of
    /// its levels, then [`Self::max_level_size`] will be larger than this value. This can be used
    /// to fine tune the backend.
    pub mem_soft_limit: usize,
}

/// An optimizer that finds concurrency levels in the graph.
#[derive(Default)]
pub struct ConcurOptimizer {
    prev_levels: Vec<ConcurLevel>,
    cur_level: ConcurLevel,
    processed: Vec<bool>,
    // This is reset every level as we only want to sum sizes of nodes used in the current
    // execution level.
    // NB that there may be duplicate tensors used _across_ levels (e.g. same input tensor
    // on one side of a matmul), however, those are counted twice across levels because each
    // level is executed in isolation.
    //
    // FIXME TODO: implement no double counting for view operations. For this we could use (buf_id,
    // data_offset, layout_size|buf_size?) tuples in a hashset to more accurately mark visited buffers.
    seen_in_cur_level: HashSet<TensorId>,
    /// Starting node of the current level, when the levels are flattened to their nodes, not counting
    /// the barrier.
    level_idx: usize,
}

impl Optimizer for ConcurOptimizer {
    fn find_exec_plan(
        &mut self,
        ctx: &impl TensorSrc,
        nodes: &[TensorId],
        mem_limit: usize,
    ) -> Result<ExecPlan, TensorNotFound> {
        let search_depth = nodes.len();

        self.processed.resize(nodes.len(), false);
        // The start of the current concurrency level.
        self.level_idx = 0;

        let mut n_left = nodes.len();
        // The current node index to start searching for in current level.
        let mut cur_idx = 0;
        let mut max_level_size = 0;
        let mut max_level_nodes = 0;

        while n_left > 0 {
            let end = (cur_idx + search_depth).min(nodes.len());
            let mut level_size = 0;

            // This is reset every round as we only want to sum sizes of nodes used in the current
            // execution level.
            // NB that there may be duplicate tensors used _across_ levels (e.g. same input tensor
            // on one side of a matmul), however, those are counted twice across levels because each
            // level is executed in isolation.
            //
            // FIXME TODO: implement no double counting for view operations
            self.seen_in_cur_level.clear();

            // Count the number of nodes that will actually be executed by the backend concurrently
            // (so no const/noop nodes).
            let mut n_concur = 0;

            // if I want to do a range loop, I'll do a range loop
            #[allow(clippy::needless_range_loop)]
            for node_idx in cur_idx..end {
                if self.processed[node_idx] {
                    continue;
                }

                let node = ctx.get_tensor(nodes[node_idx])?;

                // NB: The reason we're still including consts and noops even though the backend
                // doesn't care about them is because the allocator does: it probably needs to
                // allocate memory for them.
                let (exe_flag, src_size) = match node.op {
                    Op::Const => (true, 0),
                    Op::Unary { a: x, .. } => self.can_exec_node(ctx, nodes, &[x])?,
                    Op::Binary { a: x, b: y, .. } => self.can_exec_node(ctx, nodes, &[x, y])?,
                };

                // TODO: memory check logic goes here (not required for llama for some reason)

                if exe_flag {
                    // Only count stats if we're going to execute the node.
                    let node_size = self.get_level_unique_tensor_size(node);
                    let op_size = node_size + src_size;
                    // If this operation would cause us to exceed the configured memory limit, this
                    // node is added to the next concurrency level.
                    if level_size + op_size > mem_limit {
                        break;
                    }

                    level_size += op_size;
                    self.processed[node_idx] = true;
                    self.cur_level.push(node_idx);
                    if !matches!(node.op, Op::Const)
                        && !matches!(
                            node.op,
                            Op::Unary {
                                op: UnaryOp::Noop(_),
                                ..
                            }
                        )
                    {
                        n_concur += 1;
                    }
                }
            }

            if level_size > max_level_size {
                max_level_size = level_size;
            }

            n_left -= self.cur_level.len();
            // Skip all processed nodes.
            while cur_idx < self.processed.len() && self.processed[cur_idx] {
                cur_idx += 1;
            }

            max_level_nodes = max_level_nodes.max(n_concur);

            self.level_idx += self.cur_level.len();
            self.prev_levels.push(mem::take(&mut self.cur_level));
        }

        self.cur_level.clear();
        self.processed.clear();
        self.level_idx = 0;

        Ok(ExecPlan {
            levels: mem::take(&mut self.prev_levels),
            max_level_nodes,
            max_level_size,
            mem_soft_limit: mem_limit,
        })
    }
}

impl ConcurOptimizer {
    /// Tests whether the node can be executed in the current concurrency level. If yes, true is
    /// returned and total size of its source nodes, not counting ones already seen in the current
    /// concurrently level.
    fn can_exec_node(
        &mut self,
        ctx: &impl TensorSrc,
        nodes: &[TensorId],
        srcs: &[TensorId],
    ) -> Result<(bool, usize), TensorNotFound> {
        let search_depth = nodes.len();
        let backtrack_idx = 0.max(self.level_idx as i64 - 2 * search_depth as i64) as usize;
        let mut size = 0;
        for src_id in srcs {
            let src_node = ctx.get_tensor(*src_id)?;
            // A leaf node automatically satisfies the concurrency condition because we
            // can directly execute `op(leaf, cur)`.
            if src_node.op == Op::Const {
                continue;
            }

            // Otherwise this source should be the output of previously computed nodes (meaning
            // their values are present by the time of reaching this node) to satisfy the
            // concurrency condition.
            if !self.is_computed(nodes, src_node.id, backtrack_idx) {
                return Ok((false, 0));
            }

            size += self.get_level_unique_tensor_size(src_node);
        }
        // NB: If node parents are leaves, we return true.
        Ok((true, size))
    }

    /// Returns true if the tensor is the output of a previously computed node. It starts the
    /// backtracking search at the given index as compromise not to degrade into quadratic
    /// runtime complexity.
    fn is_computed(&self, nodes: &[TensorId], node: TensorId, backtrack_idx: usize) -> bool {
        for node_idx in self.prev_levels.iter().flatten().skip(backtrack_idx) {
            if nodes[*node_idx] == node {
                return true;
            }
        }
        false
    }

    /// Only counts the given tensor if it hasn't been seen yet in this concurrency level.
    fn get_level_unique_tensor_size(&mut self, t: &Tensor) -> usize {
        // If this tensor is a view of another tensor, we only count the size of the original
        // tensor.
        if let Some(view) = t.view {
            if self.seen_in_cur_level.insert(view.origin) {
                return 0;
            }
        }

        if self.seen_in_cur_level.insert(t.id) {
            0
        } else {
            t.layout.size()
        }
    }
}
